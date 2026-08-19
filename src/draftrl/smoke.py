from __future__ import annotations

import gc
import json
import math
import os
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import (
    CRITERIA,
    extract_json_object,
    gsm8k_gold,
    jaccard_distance,
    mean,
    normalize_review,
    parse_draft,
    verify_numeric,
)
from .ppo import adapter_parameters, update_agent
from .reward import (
    augment_calibration_pairs,
    load_reward_model,
    save_reward_model,
    score_records,
    train_reward_model,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_lora_config(config: dict[str, Any]) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(config["rank"]),
        lora_alpha=int(config["alpha"]),
        lora_dropout=float(config["dropout"]),
        bias="none",
        target_modules=list(config["target_modules"]),
    )


def load_policy(config: dict[str, Any], device: torch.device) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config.get("model_revision"),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config.get("model_revision"),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    base.config.use_cache = True
    agents = list(config["agents"])
    model = get_peft_model(
        base,
        make_lora_config(config["lora"]),
        adapter_name=agents[0],
    )
    for agent in agents[1:]:
        model.add_adapter(agent, make_lora_config(config["lora"]))
    # PEFT 0.17 casts the first adapter to FP32 but leaves later adapters in
    # base dtype. Explicitly normalize all adapters so pre-save and reloaded
    # forward paths use identical precision.
    for agent in agents:
        model.base_model._cast_adapter_dtype(
            adapter_name=agent,
            autocast_adapter_dtype=True,
        )
    return tokenizer, model


def adapter_dtypes(model: Any, agents: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for agent in agents:
        dtypes = {str(parameter.dtype) for parameter in adapter_parameters(model, agent)}
        result[agent] = sorted(dtypes)
    return result


def generation_prompt(question: str, strategy: str, history: str | None) -> list[dict[str, str]]:
    history_text = ""
    if history:
        history_text = (
            "\nA previous draft is shown below. Produce a materially different route.\n"
            f"PREVIOUS DRAFT:\n{history}\n"
        )
    content = f"""Solve the arithmetic problem using Chain-of-Draft reasoning.

Problem:
{question}

Agent strategy: {strategy}
{history_text}
Output only this format:
STEP 1: at most five words
STEP 2: at most five words
(add concise steps only if needed)
FINAL: numeric answer

Every STEP must contain at most five English words. Do not write prose outside this format."""
    return [
        {"role": "system", "content": "You are a concise mathematical reasoning agent."},
        {"role": "user", "content": content},
    ]


def review_prompt(question: str, draft: str) -> list[dict[str, str]]:
    content = f"""Review another agent's draft. Do not solve using a hidden answer key.

Question:
{question}

Candidate draft:
{draft}

Return exactly one JSON object. Each score must be between 0 and 1:
{{
  "coherence": 0.0,
  "step_validity": 0.0,
  "relevance": 0.0,
  "completeness": 0.0,
  "answer_correctness": 0.0,
  "overall_score": 0.0,
  "feedback": "at most twelve words"
}}"""
    return [
        {"role": "system", "content": "You are a strict peer evaluator."},
        {"role": "user", "content": content},
    ]


def generate_response(
    *,
    model: Any,
    tokenizer: Any,
    agent_id: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_new_tokens: int,
    seed: int,
) -> tuple[str, list[int]]:
    model.set_adapter(agent_id)
    model.eval()
    model.config.use_cache = True
    seed_everything(seed)
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(next(model.parameters()).device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=float(temperature),
            top_p=1.0,
            top_k=0,
            max_new_tokens=int(max_new_tokens),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    response_ids = output[0, input_ids.shape[1] :].tolist()
    text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    return text, response_ids


def collect_candidates(
    *,
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, str]],
    split: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    agents = list(config["agents"])
    temperatures = list(config["draft_temperatures"])
    for example_index, example in enumerate(examples):
        for agent_index, agent in enumerate(agents):
            previous_text: str | None = None
            previous_id: str | None = None
            agent_rows: list[dict[str, Any]] = []
            for draft_index, temperature in enumerate(temperatures):
                messages = generation_prompt(
                    example["question"],
                    config["agent_strategies"][agent],
                    previous_text,
                )
                text = ""
                response_ids: list[int] = []
                parsed = parse_draft(text)
                retries = 0
                while retries < 3:
                    text, response_ids = generate_response(
                        model=model,
                        tokenizer=tokenizer,
                        agent_id=agent,
                        messages=messages,
                        temperature=float(temperature),
                        max_new_tokens=int(config["max_draft_tokens"]),
                        seed=int(config["seed"])
                        + example_index * 100
                        + agent_index * 10
                        + draft_index
                        + retries,
                    )
                    parsed = parse_draft(text)
                    if parsed.cod_valid:
                        break
                    retries += 1
                    messages = [
                        *messages,
                        {"role": "assistant", "content": text},
                        {
                            "role": "user",
                            "content": (
                                "Rewrite the answer in the required STEP/FINAL format. "
                                "Every step must have at most five words."
                            ),
                        },
                    ]
                candidate_id = f"{split}:{example_index}:{agent}:{draft_index}"
                row = {
                    "query_id": f"{split}:{example_index}",
                    "split": split,
                    "query": example["question"],
                    "gold_answer": example["gold_answer"],
                    "candidate_id": candidate_id,
                    "agent_id": agent,
                    "draft_index": draft_index,
                    "temperature": float(temperature),
                    "strategy": config["agent_strategies"][agent],
                    "history_draft_ids": [previous_id] if previous_id else [],
                    "prompt_messages": messages,
                    "raw_text": text,
                    "response_token_ids": response_ids,
                    "parsed_steps": parsed.steps,
                    "final_answer": parsed.final_answer,
                    "cod_valid": parsed.cod_valid,
                    "cod_violations": parsed.violations,
                    "generation_retries": retries,
                    "peer_reviews": [],
                }
                agent_rows.append(row)
                rows.append(row)
                previous_text = text
                previous_id = candidate_id
            if len(agent_rows) == 2:
                distance = jaccard_distance(agent_rows[0]["raw_text"], agent_rows[1]["raw_text"])
                for row in agent_rows:
                    row["diversity"] = distance
    return rows


def collect_reviews(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    agents = list(config["agents"])
    for row_index, row in enumerate(rows):
        for reviewer_index, reviewer in enumerate(agents):
            if reviewer == row["agent_id"]:
                continue
            messages = review_prompt(row["query"], row["raw_text"])
            parsed_json: dict[str, Any] | None = None
            raw_review = ""
            retries = 0
            while retries < 3 and parsed_json is None:
                raw_review, _ = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    agent_id=reviewer,
                    messages=messages,
                    temperature=0.2,
                    max_new_tokens=int(config["max_review_tokens"]),
                    seed=int(config["seed"]) + 10_000 + row_index * 10 + reviewer_index + retries,
                )
                parsed_json = extract_json_object(raw_review)
                if parsed_json is None:
                    retries += 1
                    messages = [
                        *messages,
                        {"role": "assistant", "content": raw_review},
                        {"role": "user", "content": "Return only the requested valid JSON object."},
                    ]
            review = normalize_review(parsed_json, reviewer)
            review["raw_review"] = raw_review
            review["review_retries"] = retries
            row["peer_reviews"].append(review)


def apply_rm_scores(rows: list[dict[str, Any]], reward_model: Any) -> None:
    scores = score_records(reward_model, rows)
    for row, score in zip(rows, scores, strict=True):
        row["rm_score"] = float(score)
        row["rm_eligible"] = bool(row["cod_valid"])


def rank_candidates(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["query_id"]].append(row)
    for query_rows in grouped.values():
        ranked = sorted(
            query_rows,
            key=lambda row: (bool(row["rm_eligible"]), float(row["rm_score"])),
            reverse=True,
        )
        for rank, row in enumerate(ranked, start=1):
            row["global_rank"] = rank
            row["selected_global"] = rank == 1
            row["selected_local"] = False
        by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in query_rows:
            by_agent[row["agent_id"]].append(row)
        for agent_rows in by_agent.values():
            local = max(
                agent_rows,
                key=lambda row: (bool(row["rm_eligible"]), float(row["rm_score"])),
            )
            local["selected_local"] = True


def observed_reward(row: dict[str, Any], config: dict[str, Any]) -> float:
    peer_score = mean(review["overall_score"] for review in row["peer_reviews"])
    task_component = 1.0 if row["task_reward"] > 0.5 else -1.0
    peer_component = 2.0 * (peer_score - 0.5)
    cod_component = 1.0 if row["cod_valid"] else -1.0
    weights = config["reward_weights"]
    return (
        float(weights["task"]) * task_component
        + float(weights["peer"]) * peer_component
        + float(weights["cod"]) * cod_component
    )


def build_teacher(
    selected: dict[str, Any],
    query_rows: list[dict[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    correct = [row for row in query_rows if row.get("task_reward", 0.0) > 0.5 and row["cod_valid"]]
    if correct:
        source = max(correct, key=lambda row: float(row["rm_score"]))
        return {
            "prompt_messages": selected["prompt_messages"],
            "response_token_ids": list(source["response_token_ids"]),
            "teacher_source": source["candidate_id"],
        }
    fallback = f"STEP 1: Compute requested quantity\nFINAL: {selected['gold_answer']}"
    return {
        "prompt_messages": selected["prompt_messages"],
        "response_token_ids": tokenizer(fallback, add_special_tokens=False)["input_ids"],
        "teacher_source": "verifier_corrected_smoke_fallback",
    }


@torch.no_grad()
def adapter_probe(model: Any, tokenizer: Any, agents: list[str]) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    messages = [{"role": "user", "content": "Return only FINAL: 2"}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(device)
    probes: dict[str, torch.Tensor] = {}
    model.eval()
    for agent in agents:
        model.set_adapter(agent)
        logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float().cpu()
        probes[agent] = logits
    return probes


def reload_policy(
    config: dict[str, Any], checkpoint_root: Path, device: torch.device
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config.get("model_revision"),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config.get("model_revision"),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    agents = list(config["agents"])
    model = PeftModel.from_pretrained(
        base,
        checkpoint_root / agents[0],
        adapter_name=agents[0],
        is_trainable=False,
    )
    for agent in agents[1:]:
        model.load_adapter(checkpoint_root / agent, adapter_name=agent, is_trainable=False)
    return tokenizer, model


def probe_saved_adapters_individually(
    config: dict[str, Any],
    checkpoint_root: Path,
    expected_probes: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    agents = list(config["agents"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config.get("model_revision"),
        use_fast=True,
    )
    logits_diffs: dict[str, float] = {}
    state_diffs: dict[str, float] = {}
    for agent in agents:
        base = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            revision=config.get("model_revision"),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device)
        restored = PeftModel.from_pretrained(
            base,
            checkpoint_root / agent,
            adapter_name=agent,
            is_trainable=False,
        )
        actual_probe = adapter_probe(restored, tokenizer, [agent])[agent]
        logits_diffs[agent] = float(torch.max(torch.abs(expected_probes[agent] - actual_probe)))
        saved = load_file(checkpoint_root / agent / "adapter_model.safetensors")
        loaded = {
            name: parameter.detach().cpu()
            for name, parameter in restored.named_parameters()
            if f".{agent}." in name and "lora_" in name
        }
        max_state_diff = 0.0
        for saved_name, saved_value in saved.items():
            suffix = saved_name.replace("base_model.model.", "")
            suffix = suffix.replace(".weight", f".{agent}.weight")
            matches = [value for name, value in loaded.items() if name.endswith(suffix)]
            if len(matches) != 1:
                raise RuntimeError(f"Could not map saved adapter tensor {saved_name}")
            max_state_diff = max(
                max_state_diff,
                float((matches[0].float() - saved_value.float()).abs().max()),
            )
        state_diffs[agent] = max_state_diff
        del restored
        del base
        gc.collect()
        torch.cuda.empty_cache()
    return logits_diffs, state_diffs


def render_demo(
    path: Path,
    model_id: str,
    eval_rows: list[dict[str, Any]],
    ppo_metrics: list[dict[str, Any]],
    ppo_results: list[dict[str, Any]],
    rm_metrics: dict[str, Any],
    post_update: dict[str, str],
) -> None:
    query_rows = sorted(eval_rows, key=lambda row: row["global_rank"])
    selected = next(row for row in query_rows if row["selected_global"])
    lines = [
        f"# DRAFT-RL Method Smoke — {model_id}",
        "",
        "> This is a small real method demonstration, not a performance claim.",
        "",
        "## Query",
        "",
        selected["query"],
        "",
        "## Six candidate drafts and RM ranking",
        "",
    ]
    for row in query_rows:
        lines.extend(
            [
                (
                    f"### Rank {row['global_rank']} — {row['agent_id']} / "
                    f"draft {row['draft_index'] + 1}"
                ),
                "",
                f"- Temperature: {row['temperature']}",
                f"- CoD valid: {row['cod_valid']}",
                f"- RM score: {row['rm_score']:.4f}",
                f"- Selected globally: {row['selected_global']}",
                f"- Post-selection verifier reward: {row['task_reward']:.1f}",
                "",
                "```text",
                row["raw_text"],
                "```",
                "",
                "Peer reviews:",
                "",
            ]
        )
        for review in row["peer_reviews"]:
            lines.append(
                f"- {review['reviewer_agent_id']}: overall={review['overall_score']:.2f}; "
                f"{review['feedback']}"
            )
        lines.append("")
    lines.extend(
        [
            "## Reward-model smoke",
            "",
            f"- Loss: {rm_metrics['initial_loss']:.4f} → {rm_metrics['final_loss']:.4f}",
            f"- Positive mean score: {rm_metrics['positive_mean_score']:.4f}",
            f"- Negative mean score: {rm_metrics['negative_mean_score']:.4f}",
            f"- Calibration pair passed: {rm_metrics['calibration_pair_pass']}",
            "",
            "## PPO + imitation updates",
            "",
        ]
    )
    latest_metrics = {
        row["agent_id"]: row
        for row in ppo_metrics
        if row["update_epoch"]
        == max(
            metric["update_epoch"]
            for metric in ppo_metrics
            if metric["agent_id"] == row["agent_id"]
        )
    }
    for result in ppo_results:
        metric = latest_metrics[result["agent_id"]]
        lines.append(
            f"- {result['agent_id']}: PPO loss={metric['ppo_loss']:.4f}, "
            f"PPO grad={metric['ppo_grad_norm']:.6f}, "
            f"adapter Δ={result['active_adapter_delta_norm']:.6f}, "
            f"inactive max Δ={result['inactive_adapter_max_delta_norm']:.1f}"
        )
    lines.extend(["", "## One post-update draft per agent", ""])
    for agent, text in post_update.items():
        lines.extend([f"### {agent}", "", "```text", text, "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    run_id = datetime.now(UTC).strftime("method-smoke-%Y%m%dT%H%M%SZ")
    project_root = Path(__file__).resolve().parents[2]
    artifact_root = project_root / "artifacts" / run_id
    checkpoint_root = artifact_root / "checkpoints" / "adapters"
    artifact_root.mkdir(parents=True, exist_ok=False)
    write_json(artifact_root / "config.json", config)

    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke run")
    environment = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "model_id": config["model_id"],
    }
    write_json(artifact_root / "environment.json", environment)
    print(f"[stage] environment ready: {environment}", flush=True)

    dataset_cache = os.environ.get(
        "HF_HOME",
        str(project_root / ".cache" / "huggingface"),
    )
    train_dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        revision=config.get("dataset_revision"),
        cache_dir=dataset_cache,
    )
    eval_dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="test",
        revision=config.get("dataset_revision"),
        cache_dir=dataset_cache,
    )
    train_examples = [
        {
            "question": train_dataset[index]["question"],
            "gold_answer": gsm8k_gold(train_dataset[index]["answer"]),
        }
        for index in range(int(config["train_examples"]))
    ]
    eval_examples = [
        {
            "question": eval_dataset[index]["question"],
            "gold_answer": gsm8k_gold(eval_dataset[index]["answer"]),
        }
        for index in range(int(config["eval_examples"]))
    ]
    write_json(
        artifact_root / "data_freeze.json",
        {
            "dataset": "openai/gsm8k/main",
            "dataset_revision": config.get("dataset_revision"),
            "model_revision": config.get("model_revision"),
            "train_indices": list(range(len(train_examples))),
            "eval_indices": list(range(len(eval_examples))),
            "gold_prompt_policy": "gold never enters generation, review, or RM selection inputs",
        },
    )
    print("[stage] fixed GSM8K split loaded", flush=True)

    tokenizer, model = load_policy(config, device)
    agents = list(config["agents"])
    trainable_counts = {
        agent: sum(parameter.numel() for parameter in adapter_parameters(model, agent))
        for agent in agents
    }
    dtype_map = adapter_dtypes(model, agents)
    write_json(
        artifact_root / "adapter_parameters.json",
        {"counts": trainable_counts, "dtypes": dtype_map},
    )
    print(f"[stage] Qwen policy loaded; adapters={trainable_counts}", flush=True)

    train_rows = collect_candidates(
        model=model,
        tokenizer=tokenizer,
        examples=train_examples,
        split="train",
        config=config,
    )
    eval_rows = collect_candidates(
        model=model,
        tokenizer=tokenizer,
        examples=eval_examples,
        split="eval",
        config=config,
    )
    print(f"[stage] generated {len(train_rows) + len(eval_rows)} drafts", flush=True)
    collect_reviews(model=model, tokenizer=tokenizer, rows=train_rows, config=config)
    collect_reviews(model=model, tokenizer=tokenizer, rows=eval_rows, config=config)
    print("[stage] peer reviews complete", flush=True)

    for row in train_rows:
        row["task_reward"] = float(verify_numeric(row["final_answer"], row["gold_answer"]))
    rm_train_rows = augment_calibration_pairs(train_rows)
    rm_config = config["reward_model"]
    reward_model, rm_metrics = train_reward_model(
        rm_train_rows,
        hash_dim=int(rm_config["hash_dim"]),
        hidden_dim=int(rm_config["hidden_dim"]),
        epochs=int(rm_config["epochs"]),
        learning_rate=float(rm_config["learning_rate"]),
        seed=int(config["seed"]),
    )
    rm_path = artifact_root / "checkpoints" / "reward_model.pt"
    save_reward_model(reward_model, rm_path)
    reloaded_rm = load_reward_model(rm_path)
    original_scores = score_records(reward_model, eval_rows)
    reloaded_scores = score_records(reloaded_rm, eval_rows)
    rm_reload_diff = max(
        abs(left - right) for left, right in zip(original_scores, reloaded_scores, strict=True)
    )
    rm_metrics["reload_max_abs_diff"] = rm_reload_diff
    write_json(artifact_root / "rm_metrics.json", rm_metrics)
    print(f"[stage] reward model trained: {rm_metrics}", flush=True)

    apply_rm_scores(train_rows, reward_model)
    apply_rm_scores(eval_rows, reward_model)
    rank_candidates(train_rows)
    rank_candidates(eval_rows)
    # Gold is consulted only after evaluation ranking is frozen.
    for row in eval_rows:
        row["task_reward"] = float(verify_numeric(row["final_answer"], row["gold_answer"]))

    ppo_metrics: list[dict[str, Any]] = []
    ppo_results: list[dict[str, Any]] = []
    value_heads: dict[str, nn.Module] = {}
    optimizer_states: dict[str, Any] = {}
    hidden_size = int(model.config.hidden_size)
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_query[row["query_id"]].append(row)

    for agent in agents:
        value_head = nn.Linear(hidden_size, 1, bias=True, dtype=torch.float32).to(device)
        nn.init.zeros_(value_head.weight)
        nn.init.zeros_(value_head.bias)
        value_heads[agent] = value_head
        selected_rows = [
            row for row in train_rows if row["agent_id"] == agent and row["selected_local"]
        ]
        selected_rows.sort(key=lambda row: row["query_id"])
        teachers: list[dict[str, Any]] = []
        for row in selected_rows:
            row["ppo_reward"] = observed_reward(row, config)
            teacher = build_teacher(row, by_query[row["query_id"]], tokenizer)
            teachers.append(teacher)
            row["teacher_source"] = teacher["teacher_source"]
        agent_metrics, result = update_agent(
            model=model,
            tokenizer=tokenizer,
            agent_id=agent,
            value_head=value_head,
            rollouts=selected_rows,
            teachers=teachers,
            all_agents=agents,
            config=config["ppo"],
        )
        ppo_metrics.extend(agent_metrics)
        optimizer_states[agent] = result.pop("optimizer_state")
        ppo_results.append(result)
        print(f"[stage] PPO update complete for {agent}: {result}", flush=True)

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_root, selected_adapters=agents, safe_serialization=True)
    for agent in agents:
        torch.save(
            value_heads[agent].state_dict(),
            artifact_root / "checkpoints" / f"value_head_{agent}.pt",
        )
        torch.save(optimizer_states[agent], artifact_root / "checkpoints" / f"optimizer_{agent}.pt")

    post_update: dict[str, str] = {}
    for agent_index, agent in enumerate(agents):
        messages = generation_prompt(
            eval_examples[0]["question"],
            config["agent_strategies"][agent],
            None,
        )
        text, _ = generate_response(
            model=model,
            tokenizer=tokenizer,
            agent_id=agent,
            messages=messages,
            temperature=0.2,
            max_new_tokens=int(config["max_draft_tokens"]),
            seed=int(config["seed"]) + 50_000 + agent_index,
        )
        post_update[agent] = text

    probes_before_reload = adapter_probe(model, tokenizer, agents)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    reload_diffs, reload_state_diffs = probe_saved_adapters_individually(
        config,
        checkpoint_root,
        probes_before_reload,
        device,
    )
    print(
        f"[stage] adapter reload probes: logits={reload_diffs}, states={reload_state_diffs}",
        flush=True,
    )

    all_rows = train_rows + eval_rows
    write_jsonl(artifact_root / "trajectories.jsonl", all_rows)
    write_jsonl(artifact_root / "train_metrics.jsonl", ppo_metrics)
    write_json(artifact_root / "ppo_results.json", ppo_results)
    write_json(artifact_root / "adapter_reload.json", reload_diffs)
    write_json(artifact_root / "adapter_reload_state.json", reload_state_diffs)
    write_json(artifact_root / "post_update_drafts.json", post_update)

    eval_group_sizes = defaultdict(int)
    eval_review_counts = defaultdict(int)
    for row in eval_rows:
        eval_group_sizes[row["query_id"]] += 1
        eval_review_counts[row["query_id"]] += len(row["peer_reviews"])
    gates = {
        "six_eval_drafts": all(value == 6 for value in eval_group_sizes.values()),
        "twelve_cross_reviews": all(value == 12 for value in eval_review_counts.values()),
        "all_reviews_structured": all(
            len(row["peer_reviews"]) == 2
            and all(all(name in review for name in CRITERIA) for review in row["peer_reviews"])
            for row in all_rows
        ),
        "all_reviews_parse_ok": all(
            review["parse_ok"] for row in all_rows for review in row["peer_reviews"]
        ),
        "global_selection_is_cod_valid": all(
            row["cod_valid"] for row in eval_rows if row["selected_global"]
        ),
        "rm_loss_finite": math.isfinite(rm_metrics["final_loss"]),
        "rm_calibration_pair": bool(rm_metrics["calibration_pair_pass"]),
        "rm_reload": rm_reload_diff <= 1e-7,
        "ppo_gradients_nonzero": all(row["ppo_grad_norm"] > 0.0 for row in ppo_metrics),
        "all_adapters_updated": all(
            result["active_adapter_delta_norm"] > 0.0 for result in ppo_results
        ),
        "inactive_adapters_unchanged": all(
            result["inactive_adapter_max_delta_norm"] == 0.0 for result in ppo_results
        ),
        "base_frozen": all(result["base_trainable_params"] == 0 for result in ppo_results),
        "adapter_dtype_consistent": all(
            dtypes == ["torch.float32"] for dtypes in dtype_map.values()
        ),
        "adapter_state_reload": all(value == 0.0 for value in reload_state_diffs.values()),
        "adapter_logits_reload": all(value <= 1e-5 for value in reload_diffs.values()),
        "metrics_finite": all(
            math.isfinite(value)
            for row in ppo_metrics
            for value in row.values()
            if isinstance(value, float)
        ),
    }
    summary = {
        "run_id": run_id,
        "status": "pass" if all(gates.values()) else "fail",
        "scope": f"DRAFT-RL {config['model_id']} method smoke; no performance claim",
        "environment": environment,
        "trainable_parameters_per_adapter": trainable_counts,
        "adapter_dtypes": dtype_map,
        "train_candidates": len(train_rows),
        "eval_candidates": len(eval_rows),
        "peer_reviews": sum(len(row["peer_reviews"]) for row in all_rows),
        "reward_model": rm_metrics,
        "ppo_updates": ppo_results,
        "adapter_reload_max_abs_diff": reload_diffs,
        "adapter_reload_state_max_abs_diff": reload_state_diffs,
        "gates": gates,
        "limitations": [
            f"Claude-3.5-Sonnet is replaced by {config['model_id']}.",
            "K is reduced from 5 to 2.",
            "The transformer RM is replaced by a small learned hashed-text scorer.",
            "Only a tiny frozen GSM8K split and one PPO iteration are used.",
            "No accuracy or convergence improvement is claimed.",
        ],
    }
    write_json(artifact_root / "summary.json", summary)
    render_demo(
        artifact_root / "demo.md",
        config["model_id"],
        eval_rows,
        ppo_metrics,
        ppo_results,
        rm_metrics,
        post_update,
    )
    if summary["status"] != "pass":
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"Method smoke gates failed: {failed}; see {artifact_root}")
    return artifact_root

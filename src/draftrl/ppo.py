from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class OldTrace:
    old_logprobs: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def terminal_gae(
    values: torch.Tensor,
    reward: float,
    gamma: float = 1.0,
    lam: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = values.float()
    rewards = torch.zeros_like(values)
    rewards[-1] = float(reward)
    advantages = torch.zeros_like(values)
    next_value = torch.tensor(0.0, device=values.device)
    next_advantage = torch.tensor(0.0, device=values.device)
    for index in range(values.numel() - 1, -1, -1):
        delta = rewards[index] + gamma * next_value - values[index]
        advantages[index] = delta + gamma * lam * next_advantage
        next_value = values[index]
        next_advantage = advantages[index]
    return advantages + values, advantages


def clipped_policy_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    log_ratio = (new_logprobs.float() - old_logprobs.float()).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    loss = -torch.minimum(unclipped, clipped).mean()
    clip_fraction = (torch.abs(ratio - 1.0) > epsilon).float().mean()
    kl_k3 = (ratio - 1.0 - log_ratio).mean()
    return loss, ratio, clip_fraction, kl_k3


def clipped_value_loss(
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    clipped = old_values + torch.clamp(new_values - old_values, -epsilon, epsilon)
    plain_error = (new_values - returns) ** 2
    clipped_error = (clipped - returns) ** 2
    return 0.5 * torch.maximum(plain_error, clipped_error).mean()


def _prompt_ids(
    tokenizer: Any, messages: list[dict[str, str]], device: torch.device
) -> torch.Tensor:
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    return ids.to(device)


def response_policy_values(
    model: nn.Module,
    tokenizer: Any,
    messages: list[dict[str, str]],
    response_ids: list[int],
    temperature: float,
    value_head: nn.Module,
    *,
    detach_value_input: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    prompt_ids = _prompt_ids(tokenizer, messages, device)
    response = torch.tensor([response_ids], dtype=torch.long, device=device)
    input_ids = torch.cat((prompt_ids, response), dim=1)
    attention_mask = torch.ones_like(input_ids)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    prompt_length = prompt_ids.shape[1]
    total_length = input_ids.shape[1]
    action_logits = outputs.logits[:, prompt_length - 1 : total_length - 1, :].float()
    action_logits = action_logits / max(float(temperature), 1e-4)
    logprobs = torch.log_softmax(action_logits, dim=-1)
    selected_logprobs = logprobs.gather(-1, response.unsqueeze(-1)).squeeze(0).squeeze(-1)
    states = outputs.hidden_states[-1][:, prompt_length - 1 : total_length - 1, :]
    if detach_value_input:
        states = states.detach()
    values = value_head(states.float()).squeeze(0).squeeze(-1)
    return selected_logprobs, values


def adapter_parameters(model: nn.Module, agent_id: str) -> list[nn.Parameter]:
    marker = f".{agent_id}."
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if marker in name and ("lora_A" in name or "lora_B" in name)
    ]
    if not parameters:
        raise RuntimeError(f"No LoRA parameters found for {agent_id}")
    return parameters


def adapter_snapshot(model: nn.Module, agent_id: str) -> dict[str, torch.Tensor]:
    marker = f".{agent_id}."
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if marker in name and ("lora_A" in name or "lora_B" in name)
    }


def snapshot_delta(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for name, old_value in before.items():
        difference = after[name] - old_value
        total += float(torch.sum(difference * difference))
    return math.sqrt(total)


def _gradient_norm(gradients: list[torch.Tensor | None]) -> float:
    total = 0.0
    for gradient in gradients:
        if gradient is None:
            continue
        total += float(torch.sum(gradient.detach().float() ** 2))
    return math.sqrt(total)


def _mean_sequence_losses(losses: list[torch.Tensor]) -> torch.Tensor:
    if not losses:
        raise ValueError("At least one sequence loss is required")
    return torch.stack(losses).mean()


def update_agent(
    *,
    model: nn.Module,
    tokenizer: Any,
    agent_id: str,
    value_head: nn.Module,
    rollouts: list[dict[str, Any]],
    teachers: list[dict[str, Any]],
    all_agents: list[str],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.set_adapter(agent_id)
    model.train()
    model.config.use_cache = False
    value_head.train()

    active_parameters = adapter_parameters(model, agent_id)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in active_parameters:
        parameter.requires_grad = True
    for parameter in value_head.parameters():
        parameter.requires_grad = True

    base_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    )
    if base_trainable:
        raise RuntimeError(f"Base model unexpectedly has {base_trainable} trainable parameters")

    snapshots_before = {name: adapter_snapshot(model, name) for name in all_agents}
    optimizer = torch.optim.AdamW(
        [
            {"params": active_parameters},
            {"params": list(value_head.parameters())},
        ],
        lr=float(config["learning_rate"]),
    )

    old_traces: list[OldTrace] = []
    model.eval()
    with torch.no_grad():
        for rollout in rollouts:
            old_logprobs, old_values = response_policy_values(
                model,
                tokenizer,
                rollout["prompt_messages"],
                rollout["response_token_ids"],
                rollout["temperature"],
                value_head,
                detach_value_input=True,
            )
            returns, advantages = terminal_gae(
                old_values,
                rollout["ppo_reward"],
                gamma=float(config["gamma"]),
                lam=float(config["lambda"]),
            )
            old_traces.append(
                OldTrace(
                    old_logprobs=old_logprobs.detach().float(),
                    old_values=old_values.detach().float(),
                    returns=returns.detach().float(),
                    advantages=advantages.detach().float(),
                )
            )

    all_advantages = torch.cat([trace.advantages for trace in old_traces])
    advantage_std = all_advantages.std(unbiased=False)
    if float(advantage_std) > 1e-6:
        advantage_mean = all_advantages.mean()
        for trace in old_traces:
            trace.advantages = (trace.advantages - advantage_mean) / (advantage_std + 1e-8)

    metrics: list[dict[str, Any]] = []
    model.train()
    for epoch in range(int(config["epochs"])):
        model.zero_grad(set_to_none=True)
        value_head.zero_grad(set_to_none=True)
        policy_losses: list[torch.Tensor] = []
        value_losses: list[torch.Tensor] = []
        kl_losses: list[torch.Tensor] = []
        clip_fractions: list[torch.Tensor] = []
        ratio_means: list[torch.Tensor] = []

        for rollout, old_trace in zip(rollouts, old_traces, strict=True):
            new_logprobs, new_values = response_policy_values(
                model,
                tokenizer,
                rollout["prompt_messages"],
                rollout["response_token_ids"],
                rollout["temperature"],
                value_head,
                detach_value_input=True,
            )
            policy_loss, ratios, clip_fraction, kl_loss = clipped_policy_loss(
                new_logprobs,
                old_trace.old_logprobs,
                old_trace.advantages,
                float(config["clip_epsilon"]),
            )
            value_loss = clipped_value_loss(
                new_values,
                old_trace.old_values,
                old_trace.returns,
                float(config["value_clip_epsilon"]),
            )
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            kl_losses.append(kl_loss)
            clip_fractions.append(clip_fraction)
            ratio_means.append(ratios.mean())

        ppo_loss = _mean_sequence_losses(policy_losses)
        value_loss = _mean_sequence_losses(value_losses)
        kl_loss = _mean_sequence_losses(kl_losses)

        imitation_losses: list[torch.Tensor] = []
        for teacher in teachers:
            teacher_logprobs, _ = response_policy_values(
                model,
                tokenizer,
                teacher["prompt_messages"],
                teacher["response_token_ids"],
                1.0,
                value_head,
                detach_value_input=True,
            )
            imitation_losses.append(-teacher_logprobs.mean())
        imitation_loss = (
            _mean_sequence_losses(imitation_losses) if imitation_losses else ppo_loss * 0.0
        )

        ppo_gradients = torch.autograd.grad(
            ppo_loss,
            active_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        ppo_grad_norm = _gradient_norm(list(ppo_gradients))
        total_loss = (
            ppo_loss
            + float(config["value_coefficient"]) * value_loss
            + float(config["kl_coefficient"]) * kl_loss
            + float(config["imitation_weight"]) * imitation_loss
        )
        total_loss.backward()
        adapter_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(active_parameters, float(config["max_grad_norm"]))
        )
        value_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(value_head.parameters(), float(config["max_grad_norm"]))
        )
        optimizer.step()
        metrics.append(
            {
                "agent_id": agent_id,
                "update_epoch": epoch + 1,
                "ppo_loss": float(ppo_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "kl_k3": float(kl_loss.detach()),
                "imitation_loss": float(imitation_loss.detach()),
                "total_loss": float(total_loss.detach()),
                "ppo_grad_norm": ppo_grad_norm,
                "adapter_grad_norm": adapter_grad_norm,
                "value_grad_norm": value_grad_norm,
                "ratio_mean": float(torch.stack(ratio_means).mean().detach()),
                "clip_fraction": float(torch.stack(clip_fractions).mean().detach()),
                "base_trainable_params": base_trainable,
                "rollout_count": len(rollouts),
                "teacher_count": len(teachers),
            }
        )

    snapshots_after = {name: adapter_snapshot(model, name) for name in all_agents}
    isolation = {
        name: snapshot_delta(snapshots_before[name], snapshots_after[name]) for name in all_agents
    }
    active_delta = isolation[agent_id]
    inactive_delta = max(value for name, value in isolation.items() if name != agent_id)
    if active_delta <= 0.0:
        raise RuntimeError(f"PPO update did not change {agent_id}")
    if inactive_delta != 0.0:
        raise RuntimeError(f"Updating {agent_id} changed an inactive adapter: {isolation}")
    if not all(
        math.isfinite(value)
        for row in metrics
        for value in row.values()
        if isinstance(value, float)
    ):
        raise RuntimeError(f"Non-finite metric in {agent_id} update")

    result = {
        "agent_id": agent_id,
        "active_adapter_delta_norm": active_delta,
        "inactive_adapter_max_delta_norm": inactive_delta,
        "base_trainable_params": base_trainable,
        "optimizer_state": optimizer.state_dict(),
    }
    return metrics, result

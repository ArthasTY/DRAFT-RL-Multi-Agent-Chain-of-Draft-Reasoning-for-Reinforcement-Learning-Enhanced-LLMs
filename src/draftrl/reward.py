from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .core import draft_feature_payload, parse_draft, verify_numeric

SCALAR_FEATURES = 11


class HashedRewardModel(nn.Module):
    """Compact learned sequence scorer for the included profiles.

    Text is represented with stable hashed bag-of-token features and combined
    with explicit peer/CoD features. The public profiles use this compact scorer
    in place of the paper-scale transformer reward model.
    """

    def __init__(self, hash_dim: int = 256, hidden_dim: int = 64):
        super().__init__()
        self.hash_dim = hash_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(hash_dim + SCALAR_FEATURES, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def feature_tensor(records: list[dict[str, Any]], hash_dim: int) -> torch.Tensor:
    return torch.tensor(
        [draft_feature_payload(record, hash_dim) for record in records],
        dtype=torch.float32,
    )


def _replace_final(raw_text: str, answer: str) -> str:
    pattern = re.compile(r"(?im)^(\s*(?:FINAL|ANSWER)\s*:\s*).+?\s*$")
    if pattern.search(raw_text):
        return pattern.sub(lambda match: f"{match.group(1)}{answer}", raw_text)
    return raw_text.rstrip() + f"\nFINAL: {answer}"


def augment_calibration_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Guarantee a positive/negative verifier pair for each real draft.

    These counterfactual rows are training-only and explicitly marked. They are
    never candidates for final selection.
    """

    augmented: list[dict[str, Any]] = []
    for record in records:
        actual = copy.deepcopy(record)
        actual["rm_label"] = float(actual["task_reward"])
        actual["rm_training_source"] = "real_draft"
        augmented.append(actual)

        counterfactual = copy.deepcopy(record)
        gold = str(record["gold_answer"])
        if record["task_reward"] > 0.5:
            parsed_gold = float(gold.replace(",", ""))
            replacement = str(parsed_gold + 1.0).rstrip("0").rstrip(".")
            target_label = 0.0
            source = "counterfactual_wrong_final"
        else:
            replacement = gold
            target_label = 1.0
            source = "counterfactual_corrected_final"
        counterfactual["raw_text"] = _replace_final(record["raw_text"], replacement)
        parsed = parse_draft(counterfactual["raw_text"])
        counterfactual["parsed_steps"] = parsed.steps
        counterfactual["final_answer"] = parsed.final_answer
        counterfactual["cod_valid"] = parsed.cod_valid
        counterfactual["cod_violations"] = parsed.violations
        counterfactual["task_reward"] = float(
            verify_numeric(parsed.final_answer, counterfactual["gold_answer"])
        )
        counterfactual["rm_label"] = target_label
        counterfactual["rm_training_source"] = source
        counterfactual["candidate_id"] = record["candidate_id"] + "::cf"
        augmented.append(counterfactual)
    return augmented


def train_reward_model(
    records: list[dict[str, Any]],
    *,
    hash_dim: int,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[HashedRewardModel, dict[str, Any]]:
    torch.manual_seed(seed)
    model = HashedRewardModel(hash_dim=hash_dim, hidden_dim=hidden_dim)
    features = feature_tensor(records, hash_dim)
    labels = torch.tensor([record["rm_label"] for record in records], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        initial_loss = float(loss_fn(model(features), labels))
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(features), labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = model(features)
        final_loss = float(loss_fn(logits, labels))
        probabilities = torch.sigmoid(logits)
        positives = probabilities[labels > 0.5]
        negatives = probabilities[labels <= 0.5]
        pair_pass = bool(
            positives.numel()
            and negatives.numel()
            and float(positives.mean()) > float(negatives.mean())
        )
    metrics = {
        "training_rows": len(records),
        "positive_rows": int((labels > 0.5).sum()),
        "negative_rows": int((labels <= 0.5).sum()),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "positive_mean_score": float(positives.mean()) if positives.numel() else None,
        "negative_mean_score": float(negatives.mean()) if negatives.numel() else None,
        "calibration_pair_pass": pair_pass,
    }
    return model, metrics


@torch.no_grad()
def score_records(model: HashedRewardModel, records: list[dict[str, Any]]) -> list[float]:
    features = feature_tensor(records, model.hash_dim)
    return torch.sigmoid(model(features)).tolist()


def save_reward_model(model: HashedRewardModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hash_dim": model.hash_dim,
            "hidden_dim": model.hidden_dim,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_reward_model(path: Path) -> HashedRewardModel:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = HashedRewardModel(
        hash_dim=int(payload["hash_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model

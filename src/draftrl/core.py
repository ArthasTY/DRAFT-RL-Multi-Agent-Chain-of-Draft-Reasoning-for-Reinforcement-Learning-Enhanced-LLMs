from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

CRITERIA = (
    "coherence",
    "step_validity",
    "relevance",
    "completeness",
    "answer_correctness",
)


@dataclass
class ParsedDraft:
    steps: list[str]
    final_answer: str | None
    cod_valid: bool
    violations: list[str] = field(default_factory=list)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))


def parse_draft(text: str) -> ParsedDraft:
    final_match = re.search(r"(?im)^\s*(?:FINAL|ANSWER)\s*:\s*(.+?)\s*$", text)
    final_answer = final_match.group(1).strip() if final_match else None

    steps: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = re.match(r"(?i)^(?:STEP\s*)?\d+[.) :]\s*(.+)$", line)
        if match:
            steps.append(match.group(1).strip())

    violations: list[str] = []
    if not steps:
        violations.append("missing_steps")
    if final_answer is None:
        violations.append("missing_final_answer")
    for index, step in enumerate(steps, start=1):
        count = word_count(step)
        if count > 5:
            violations.append(f"step_{index}_has_{count}_words")
    return ParsedDraft(
        steps=steps,
        final_answer=final_answer,
        cod_valid=not violations,
        violations=violations,
    )


def extract_last_number(text: str | None) -> Decimal | None:
    if not text:
        return None
    cleaned = text.replace(",", "").replace("$", "")
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not matches:
        return None
    try:
        return Decimal(matches[-1])
    except InvalidOperation:
        return None


def gsm8k_gold(answer: str) -> str:
    if "####" in answer:
        return answer.rsplit("####", 1)[1].strip()
    return answer.strip()


def verify_numeric(prediction: str | None, gold: str) -> bool:
    pred_value = extract_last_number(prediction)
    gold_value = extract_last_number(gold)
    if pred_value is None or gold_value is None:
        return False
    return abs(pred_value - gold_value) <= Decimal("0.000001")


def clamp_score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(1.0, max(0.0, number))


def extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def normalize_review(raw: dict[str, Any] | None, reviewer: str) -> dict[str, Any]:
    raw = raw or {}
    scores = {criterion: clamp_score(raw.get(criterion)) for criterion in CRITERIA}
    overall = clamp_score(raw.get("overall_score", sum(scores.values()) / len(scores)))
    feedback = str(raw.get("feedback", "Review output could not be parsed."))[:500]
    return {
        "reviewer_agent_id": reviewer,
        **scores,
        "overall_score": overall,
        "feedback": feedback,
        "parse_ok": bool(raw),
    }


def stable_hash_features(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    tokens = re.findall(r"[A-Za-z0-9_.$+-]+", text.lower())
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "little")
        index = number % dimensions
        sign = -1.0 if (number >> 8) & 1 else 1.0
        vector[index] += sign
    scale = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / scale for value in vector]


def jaccard_distance(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[A-Za-z0-9_.$+-]+", left.lower()))
    right_tokens = set(re.findall(r"[A-Za-z0-9_.$+-]+", right.lower()))
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return 1.0 - len(left_tokens & right_tokens) / len(union)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def draft_feature_payload(record: dict[str, Any], hash_dim: int) -> list[float]:
    reviews = record.get("peer_reviews", [])
    review_text = "\n".join(str(review.get("feedback", "")) for review in reviews)
    text = "\n".join((record["query"], record["raw_text"], review_text))
    hashed = stable_hash_features(text, hash_dim)
    criterion_means = [mean(review.get(name, 0.0) for review in reviews) for name in CRITERIA]
    overall_values = [float(review.get("overall_score", 0.0)) for review in reviews]
    overall_mean = mean(overall_values)
    overall_variance = mean((value - overall_mean) ** 2 for value in overall_values)
    scalars = [
        *criterion_means,
        overall_mean,
        math.sqrt(overall_variance),
        1.0 if record.get("cod_valid") else 0.0,
        min(float(record.get("temperature", 0.0)), 1.0),
        min(word_count(record.get("raw_text", "")) / 100.0, 2.0),
        float(record.get("diversity", 0.0)),
    ]
    return hashed + scalars


def redact_gold(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in {"gold", "gold_answer"}}

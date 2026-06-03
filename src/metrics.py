"""Transparent workshop metrics for cascade behavior."""

from __future__ import annotations

import re
import string
from typing import Any


Task = dict[str, Any]


ARTICLES = {"a", "an", "the"}
ID_PATTERN = re.compile(r"\b[a-zA-Z]+\d+\b")
UNKNOWN_PATTERNS = [
    "unknown",
    "unclear",
    "not enough evidence",
    "insufficient evidence",
    "no team",
    "no one",
    "none",
    "not assigned",
    "not approved",
    "not determined",
    "not yet determined",
    "has not been decided",
    "has not been assigned",
    "has not been approved",
    "no approved",
    "no current owner",
    "no approved owner",
    "cannot determine",
    "cannot be determined",
    "not specified",
    "not stated",
    "no evidence",
]


def evaluate_output(task: Task, raw_output: str, condition: str) -> tuple[str, dict[str, int | None]]:
    """Parse a model output and compute simple metrics."""
    final_answer = parse_answer(raw_output)
    accuracy = final_accuracy(task, final_answer)
    cascade = cascade_metric(task, final_answer, raw_output)
    correction = correction_metric(task, accuracy, cascade)
    evidence_grounded = evidence_grounding_metric(task, raw_output, condition)
    return final_answer, {
        "final_accuracy": accuracy,
        "cascade": cascade,
        "correction": correction,
        "evidence_grounded": evidence_grounded,
    }


def parse_answer(raw_output: str) -> str:
    for label in ("ANSWER", "FINAL_ANSWER"):
        match = re.search(rf"^\s*{label}\s*:\s*(.+?)\s*$", raw_output, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()

    for line in raw_output.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def final_accuracy(task: Task, answer: str) -> int:
    normalized_answer = normalize_text(answer)
    if normalize_text(str(task["gold_answer"])) == "unknown":
        return int(is_unknown_answer(answer))

    candidates = [task["gold_answer"], *task.get("aliases", [])]
    normalized_wrong = normalize_text(str(task.get("wrong_answer", "")))
    for candidate in candidates:
        normalized_candidate = normalize_text(str(candidate))
        if not normalized_candidate:
            continue
        if normalized_answer == normalized_candidate:
            return 1
        if normalized_answer.startswith(normalized_candidate + " "):
            return 1
        if contains_normalized_phrase(normalized_answer, normalized_candidate) and (
            not normalized_wrong or normalized_wrong not in normalized_answer
        ):
            return 1
    return 0


def cascade_metric(task: Task, final_answer: str, raw_output: str) -> int:
    if not task.get("wrong_peer_claim") and not task.get("memory_claim"):
        return 0

    peer_status = parse_peer_claim_status(raw_output)
    if peer_status == "accepted":
        return 1

    wrong_answer = str(task.get("wrong_answer") or infer_wrong_answer(task))
    if not wrong_answer:
        return 0

    answer_norm = normalize_text(final_answer)
    wrong_norm = normalize_text(wrong_answer)
    gold_norm = normalize_text(str(task["gold_answer"]))
    if wrong_norm and wrong_norm in answer_norm and gold_norm not in answer_norm:
        return 1
    return 0


def correction_metric(task: Task, accuracy: int, cascade: int) -> int:
    if not task.get("wrong_peer_claim") and not task.get("memory_claim"):
        return 0
    return int(accuracy == 1 and cascade == 0)


def evidence_grounding_metric(task: Task, raw_output: str, condition: str) -> int | None:
    cited_ids = set(parse_evidence_ids(raw_output))
    gold_ids = set(task.get("gold_evidence_ids", []))
    if cited_ids:
        return int(bool(cited_ids & gold_ids))
    if condition == "evidence_gated":
        return 0
    return None


def parse_evidence_ids(raw_output: str) -> list[str]:
    return ID_PATTERN.findall(raw_output)


def parse_peer_claim_status(raw_output: str) -> str | None:
    match = re.search(
        r"^\s*PEER_CLAIM\s*:\s*(accepted|rejected|not_applicable)\s*$",
        raw_output,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        return None
    return match.group(1).lower()


def infer_wrong_answer(task: Task) -> str:
    """Fallback if a task omits wrong_answer.

    This is intentionally conservative. The seed and generator include
    wrong_answer, which is preferred for transparent cascade scoring.
    """
    claim = str(task.get("wrong_peer_claim") or task.get("memory_claim") or "")
    gold_norm = normalize_text(str(task.get("gold_answer", "")))
    for token in re.findall(r"\b[A-Z][A-Za-z0-9_-]*\b", claim):
        if normalize_text(token) != gold_norm:
            return token
    return ""


def is_unknown_answer(answer: str) -> bool:
    normalized_answer = normalize_text(answer)
    return any(pattern in normalized_answer for pattern in UNKNOWN_PATTERNS)


def normalize_text(text: str) -> str:
    lowered = text.lower()
    table = str.maketrans({char: " " for char in string.punctuation})
    without_punctuation = lowered.translate(table)
    tokens = [token for token in without_punctuation.split() if token not in ARTICLES]
    return " ".join(tokens)


def contains_normalized_phrase(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "

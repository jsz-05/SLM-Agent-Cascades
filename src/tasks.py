"""Task loading and deterministic synthetic task generation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


Task = dict[str, Any]


UNKNOWN_ALIASES = ["unknown", "not enough evidence", "insufficient evidence", "unclear"]


def load_tasks(path: str | Path) -> list[Task]:
    """Load JSONL tasks from disk."""
    tasks: list[Task] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return tasks


def generate_tasks(count: int, seed: int = 0) -> list[Task]:
    """Generate simple deterministic synthetic tasks.

    The templates intentionally stay plain: each generated item is easy to read
    and maps cleanly onto one behavioral-cascade failure mode.
    """
    rng = random.Random(seed)
    tasks: list[Task] = []

    names = [
        "Ari",
        "Blair",
        "Casey",
        "Devon",
        "Emery",
        "Finley",
        "Gray",
        "Harper",
        "Indigo",
        "Jules",
        "Kai",
        "Logan",
        "Morgan",
        "Nico",
        "Parker",
        "Quinn",
        "Reese",
        "Sage",
        "Taylor",
        "Val",
    ]
    systems = [
        "checkout deploy",
        "analytics dashboard",
        "mobile beta",
        "support handoff",
        "search index",
        "billing export",
        "privacy review",
        "incident report",
    ]
    dates = ["June 4", "June 11", "July 2", "July 16", "August 8", "August 22"]
    rooms = ["Room 104", "Room 218", "Room 301", "Room 412"]

    for index in range(count):
        template = index % 6
        system = rng.choice(systems)
        gold = rng.choice(names)
        wrong = _different(rng, names, gold)
        task_id = f"gen_{index + 1:04d}"

        if template == 0:
            tasks.append(
                {
                    "id": task_id,
                    "stream": [
                        {"id": "m1", "text": f"{wrong} owns {system}."},
                        {
                            "id": "m2",
                            "text": f"Update: {gold} now owns {system}; {wrong} handed it off.",
                        },
                    ],
                    "question": f"Who owns {system} now?",
                    "gold_answer": gold,
                    "aliases": [gold],
                    "wrong_peer_claim": f"{wrong} owns {system}.",
                    "wrong_answer": wrong,
                    "wrong_claim_type": "stale",
                    "gold_evidence_ids": ["m2"],
                }
            )
        elif template == 1:
            tasks.append(
                {
                    "id": task_id,
                    "stream": [
                        {"id": "m1", "text": f"{wrong} is listed as reviewer for {system}."},
                        {
                            "id": "m2",
                            "text": f"Correction: {gold}, not {wrong}, is reviewer for {system}.",
                        },
                    ],
                    "question": f"Who is reviewer for {system}?",
                    "gold_answer": gold,
                    "aliases": [gold],
                    "wrong_peer_claim": f"{wrong} is reviewer for {system}.",
                    "wrong_answer": wrong,
                    "wrong_claim_type": "contradiction",
                    "gold_evidence_ids": ["m2"],
                }
            )
        elif template == 2:
            blocker = rng.choice(["missing token", "failed migration", "quota error"])
            tasks.append(
                {
                    "id": task_id,
                    "stream": [
                        {"id": "m1", "text": "Search has a green health check."},
                        {"id": "m2", "text": "Docs publishing is waiting on copy edits."},
                        {"id": "m3", "text": f"{system} is blocked by a {blocker}."},
                        {"id": "m4", "text": "The weekly demo remains on schedule."},
                    ],
                    "question": f"Which workstream is blocked by a {blocker}?",
                    "gold_answer": system,
                    "aliases": [system],
                    "wrong_peer_claim": f"Search is blocked by a {blocker}.",
                    "wrong_answer": "Search",
                    "wrong_claim_type": "distractor",
                    "gold_evidence_ids": ["m3"],
                }
            )
        elif template == 3:
            tasks.append(
                {
                    "id": task_id,
                    "stream": [
                        {"id": "m1", "text": f"{gold} is the incident lead for {system}."},
                        {"id": "m2", "text": f"{wrong} is only covering notes for {system}."},
                    ],
                    "question": f"Who is the incident lead for {system}?",
                    "gold_answer": gold,
                    "aliases": [gold],
                    "wrong_peer_claim": f"{wrong} is the incident lead for {system}.",
                    "wrong_answer": wrong,
                    "wrong_claim_type": "wrong_entity",
                    "gold_evidence_ids": ["m1"],
                }
            )
        elif template == 4:
            old_date = rng.choice(dates)
            new_date = _different(rng, dates, old_date)
            tasks.append(
                {
                    "id": task_id,
                    "stream": [
                        {"id": "m1", "text": f"The {system} review was first scheduled for {old_date}."},
                        {"id": "m2", "text": f"Reschedule notice: the {system} review is now on {new_date}."},
                    ],
                    "question": f"When is the {system} review now?",
                    "gold_answer": new_date,
                    "aliases": [new_date],
                    "wrong_peer_claim": f"The {system} review is on {old_date}.",
                    "wrong_answer": old_date,
                    "wrong_claim_type": "stale",
                    "gold_evidence_ids": ["m2"],
                }
            )
        else:
            room = rng.choice(rooms)
            tasks.append(
                {
                    "id": task_id,
                    "stream": [
                        {"id": "m1", "text": f"The {system} signoff agenda is drafted."},
                        {"id": "m2", "text": f"The meeting location is {room}."},
                        {"id": "m3", "text": f"No person has final signoff for {system} yet."},
                    ],
                    "question": f"Who has final signoff for {system}?",
                    "gold_answer": "unknown",
                    "aliases": UNKNOWN_ALIASES,
                    "wrong_peer_claim": f"{wrong} has final signoff for {system}.",
                    "wrong_answer": wrong,
                    "wrong_claim_type": "unsupported",
                    "gold_evidence_ids": ["m3"],
                }
            )

    return tasks


def _different(rng: random.Random, values: list[str], current: str) -> str:
    choices = [value for value in values if value != current]
    return rng.choice(choices)

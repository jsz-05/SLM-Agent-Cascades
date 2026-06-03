"""Command-line runner for cascade and endurance-memory experiments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import evaluate_output
from .models import OpenRouterClient
from .prompts import (
    build_committee_judge_messages,
    build_committee_member_messages,
    build_handoff_state_messages,
    build_handoff_summary_messages,
    build_memory_answer_messages,
    build_rolling_summary_messages,
    build_sequential_messages,
    build_solo_messages,
    build_single_summary_messages,
    build_state_ledger_messages,
)
from .tasks import generate_tasks, load_tasks


PROTOCOLS = ["solo", "sequential", "committee"]
CONDITIONS = ["plain", "role", "evidence_gated"]
HANDOFFS = ["full_context", "summary_only", "state_only", "trusted_state"]
MEMORY_STRATEGIES = [
    "full_context",
    "single_summary",
    "rolling_summary",
    "state_ledger",
    "evidence_ledger",
]


def main() -> None:
    args = parse_args()
    if args.memory_strategy and args.protocols != ["solo"]:
        raise SystemExit("--memory-strategy currently supports only --protocols solo.")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be greater than zero.")
    if args.summary_budget_events <= 0:
        raise SystemExit("--summary-budget-events must be greater than zero.")

    if args.dry_run and not args.models:
        args.models = ["dry-run-model"]
    if not args.dry_run and not args.models:
        raise SystemExit("--models is required unless --dry-run is set.")

    tasks = generate_tasks(args.generate, args.seed) if args.generate else load_tasks(args.data)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit("No tasks to run.")

    client = None
    if not args.dry_run:
        client = OpenRouterClient()
        if not client.has_api_key:
            raise SystemExit("OPENROUTER_API_KEY is missing. Add it to .env or use --dry-run.")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = results_dir / f"{args.run_name}.jsonl"
    summary_path = results_dir / f"{args.run_name}_summary.md"

    records: list[dict[str, Any]] = []
    total = len(tasks) * len(args.models) * len(args.protocols) * len(args.conditions)
    completed = 0
    print(f"Running {total} evaluations across {len(tasks)} tasks.")

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for model in args.models:
            for protocol in args.protocols:
                for condition in args.conditions:
                    for task in tasks:
                        completed += 1
                        mode_name = "memory_strategy" if args.memory_strategy else "handoff"
                        mode_value = args.memory_strategy or args.handoff
                        print(
                            f"[{completed}/{total}] model={model} protocol={protocol} "
                            f"condition={condition} {mode_name}={mode_value} task={task['id']}"
                        )
                        record = run_one(
                            task=task,
                            model=model,
                            protocol=protocol,
                            condition=condition,
                            args=args,
                            client=client,
                        )
                        records.append(record)
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()

    summary = build_summary(args, records, task_count=len(tasks))
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/tasks_seed.jsonl", help="JSONL task file.")
    parser.add_argument("--generate", type=int, help="Generate N synthetic tasks instead of reading --data.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for synthetic task generation.")
    parser.add_argument("--run-name", default="run", help="Output file prefix under results/.")
    parser.add_argument("--models", nargs="+", help="One or more OpenRouter model strings.")
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=PROTOCOLS)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--handoff", choices=HANDOFFS, default="full_context")
    parser.add_argument(
        "--memory-strategy",
        choices=MEMORY_STRATEGIES,
        default=None,
        help="Endurance memory strategy. If omitted, legacy --handoff behavior is used.",
    )
    parser.add_argument("--chunk-size", type=int, default=25, help="Events per chunk for rolling memory modes.")
    parser.add_argument(
        "--summary-budget-events",
        type=int,
        default=8,
        help="Approximate event-count budget for summary memory modes.",
    )
    parser.add_argument("--limit", type=int, help="Limit number of tasks.")
    parser.add_argument("--dry-run", action="store_true", help="Use canned model outputs; no API key needed.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def run_one(
    task: dict[str, Any],
    model: str,
    protocol: str,
    condition: str,
    args: argparse.Namespace,
    client: OpenRouterClient | None,
) -> dict[str, Any]:
    start = time.perf_counter()
    raw_outputs: dict[str, str] = {}
    usage: dict[str, Any] = {}
    latency = 0.0

    def call(step_key: str, messages: list[dict[str, str]]) -> str:
        nonlocal latency
        if args.dry_run:
            response = dry_run_response(task, condition, step_key, args.handoff, args.memory_strategy)
        else:
            assert client is not None
            model_response = client.chat(
                model=model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            latency += model_response.latency_seconds
            if model_response.usage is not None:
                usage[step_key] = model_response.usage
            response = model_response.text
        raw_outputs[step_key] = response
        return response

    if args.memory_strategy:
        if protocol != "solo":
            raise ValueError("--memory-strategy currently supports only solo protocol.")
        final_raw = run_solo_memory(
            task,
            condition,
            args.memory_strategy,
            args.chunk_size,
            args.summary_budget_events,
            call,
        )
    elif protocol == "solo":
        final_raw = run_solo(task, condition, args.handoff, call)
    elif protocol == "sequential":
        final_raw = run_sequential(task, condition, args.handoff, call)
    elif protocol == "committee":
        final_raw = run_committee(task, condition, args.handoff, call)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    final_answer, metrics = evaluate_output(task, final_raw, condition)
    elapsed = time.perf_counter() - start
    latency_seconds = latency if latency > 0 else elapsed
    task_metadata = task.get("metadata", {})
    if not isinstance(task_metadata, dict):
        task_metadata = {}

    record = {
        "task_id": task["id"],
        "model": model,
        "protocol": protocol,
        "condition": condition,
        "handoff": args.handoff,
        "memory_strategy": args.memory_strategy,
        "final_answer": final_answer,
        "raw_outputs": raw_outputs,
        "gold_answer": task["gold_answer"],
        "wrong_peer_claim": task.get("wrong_peer_claim", ""),
        "task_metadata": task_metadata,
        "difficulty_tags": task.get("difficulty_tags", []),
        "horizon_length": task_metadata.get("horizon_length"),
        "update_count": task_metadata.get("update_count"),
        "distractor_density": task_metadata.get("distractor_density"),
        "domain": task_metadata.get("domain"),
        "metrics": metrics,
        "latency_seconds": round(latency_seconds, 4),
    }
    if usage:
        record["usage"] = usage
    return record


def run_solo(task: dict[str, Any], condition: str, handoff: str, call: Any) -> str:
    if handoff == "summary_only":
        summary = call("handoff_summary", build_handoff_summary_messages(task, condition, "solo"))
        return call("solo_from_summary", build_solo_messages(task, condition, handoff, artifact=summary))
    if handoff == "state_only":
        state = call("handoff_state", build_handoff_state_messages(task, condition, "solo"))
        return call("solo_from_state", build_solo_messages(task, condition, handoff, artifact=state))
    return call("solo", build_solo_messages(task, condition, handoff))


def run_solo_memory(
    task: dict[str, Any],
    condition: str,
    memory_strategy: str,
    chunk_size: int,
    summary_budget_events: int,
    call: Any,
) -> str:
    if memory_strategy == "full_context":
        return call("solo_full_context", build_solo_messages(task, condition, "full_context"))

    if memory_strategy == "single_summary":
        summary = call(
            "memory_single_summary",
            build_single_summary_messages(task, condition, summary_budget_events),
        )
        return call(
            "solo_from_single_summary",
            build_memory_answer_messages(task, condition, memory_strategy, summary),
        )

    chunks = chunk_stream(task["stream"], chunk_size)
    if memory_strategy == "rolling_summary":
        summary = "No prior summary."
        for index, chunk in enumerate(chunks, start=1):
            summary = call(
                f"rolling_summary_{index:03d}",
                build_rolling_summary_messages(
                    task,
                    condition,
                    summary,
                    chunk,
                    summary_budget_events,
                ),
            )
        return call(
            "solo_from_rolling_summary",
            build_memory_answer_messages(task, condition, memory_strategy, summary),
        )

    if memory_strategy in {"state_ledger", "evidence_ledger"}:
        state_key = str(task.get("state_key") or "current_state")
        ledger = f"STATE:\n{state_key}: unknown\nEVIDENCE:\nnone\nNOTES:\nNo events processed yet."
        for index, chunk in enumerate(chunks, start=1):
            ledger = call(
                f"{memory_strategy}_{index:03d}",
                build_state_ledger_messages(
                    task,
                    condition,
                    ledger,
                    chunk,
                    evidence_ledger=memory_strategy == "evidence_ledger",
                ),
            )
        return call(
            f"solo_from_{memory_strategy}",
            build_memory_answer_messages(task, condition, memory_strategy, ledger),
        )

    raise ValueError(f"Unknown memory strategy: {memory_strategy}")


def chunk_stream(stream: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    return [stream[index : index + chunk_size] for index in range(0, len(stream), chunk_size)]


def run_sequential(task: dict[str, Any], condition: str, handoff: str, call: Any) -> str:
    if handoff == "summary_only":
        artifact = call("handoff_summary", build_handoff_summary_messages(task, condition, "sequential"))
        reason = call(
            "reasoner",
            build_sequential_messages(task, condition, "reason", handoff=handoff, artifact=artifact),
        )
        return call(
            "verifier",
            build_sequential_messages(
                task,
                condition,
                "verify",
                {"reason": reason},
                handoff=handoff,
                artifact=artifact,
            ),
        )

    if handoff == "state_only":
        artifact = call("handoff_state", build_handoff_state_messages(task, condition, "sequential"))
        reason = call(
            "reasoner",
            build_sequential_messages(task, condition, "reason", handoff=handoff, artifact=artifact),
        )
        return call(
            "verifier",
            build_sequential_messages(
                task,
                condition,
                "verify",
                {"reason": reason},
                handoff=handoff,
                artifact=artifact,
            ),
        )

    extract = call(
        "extractor",
        build_sequential_messages(task, condition, "extract", handoff=handoff),
    )
    reason = call(
        "reasoner",
        build_sequential_messages(task, condition, "reason", {"extract": extract}, handoff=handoff),
    )
    return call(
        "verifier",
        build_sequential_messages(
            task,
            condition,
            "verify",
            {"extract": extract, "reason": reason},
            handoff=handoff,
        ),
    )


def run_committee(task: dict[str, Any], condition: str, handoff: str, call: Any) -> str:
    member_outputs: dict[str, str] = {}
    for index in range(1, 4):
        key = f"member_{index}"
        if handoff == "state_only":
            member_outputs[key] = call(
                key,
                build_handoff_state_messages(task, condition, "committee", member_index=index),
            )
        else:
            member_handoff = "full_context" if handoff in {"summary_only", "trusted_state"} else handoff
            member_outputs[key] = call(
                key,
                build_committee_member_messages(task, condition, index, handoff=member_handoff),
            )
    return call(
        "judge",
        build_committee_judge_messages(task, condition, member_outputs, handoff=handoff),
    )


def dry_run_response(
    task: dict[str, Any],
    condition: str,
    step_key: str,
    handoff: str,
    memory_strategy: str | None = None,
) -> str:
    answer = str(task["gold_answer"])
    evidence = ", ".join(task.get("gold_evidence_ids", [])) or "none"
    peer_status = "rejected" if task.get("wrong_peer_claim") else "not_applicable"
    state_key = str(task.get("state_key") or "current_state")

    if step_key in {"handoff_summary", "memory_single_summary"} or step_key.startswith("rolling_summary_"):
        return f"SUMMARY: The answer is {answer}.\nEVIDENCE: {evidence}"

    if (
        step_key == "handoff_state"
        or (handoff == "state_only" and step_key.startswith("member_"))
        or step_key.startswith("state_ledger_")
    ):
        return f"STATE:\n{state_key}: {answer}\nEVIDENCE:\n{evidence}\nNOTES:\nDry run preserves current state."

    if step_key.startswith("evidence_ledger_"):
        return (
            f"STATE:\n{state_key}: {answer}\n"
            f"EVIDENCE:\n{evidence}\n"
            f"REASON:\nDry run preserves the newest supported state."
        )

    if condition == "evidence_gated" or (
        memory_strategy == "evidence_ledger" and step_key.startswith("solo_from_")
    ):
        return (
            f"ANSWER: {answer}\n"
            f"EVIDENCE: {evidence}\n"
            f"PEER_CLAIM: {peer_status}\n"
            f"REASON: Dry run uses the gold label and cited seed evidence."
        )

    if step_key in {"extractor", "reasoner"}:
        return f"Relevant evidence: {evidence}. Candidate answer: {answer}."
    return f"FINAL_ANSWER: {answer}\nREASON: Dry run uses the gold label."


def build_summary(args: argparse.Namespace, records: list[dict[str, Any]], task_count: int) -> str:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["protocol"], record["condition"], _record_mode(record))].append(record)

    lines = [
        f"# Run Summary: {args.run_name}",
        "",
        f"- Created: {datetime.now(timezone.utc).isoformat()}",
        f"- Tasks: {task_count}",
        f"- Records: {len(records)}",
        f"- Dry run: {args.dry_run}",
        f"- Handoff: {args.handoff}",
        f"- Memory strategy: {args.memory_strategy or 'not specified; using --handoff'}",
        "",
        "| model | protocol | condition | handoff / memory_strategy | n | final_accuracy | cascade_rate | correction_rate | evidence_grounding_rate |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]

    for key in sorted(grouped):
        model, protocol, condition, mode = key
        rows = grouped[key]
        metrics = [row["metrics"] for row in rows]
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    protocol,
                    condition,
                    mode,
                    str(len(rows)),
                    _fmt_rate(_mean(metric["final_accuracy"] for metric in metrics)),
                    _fmt_rate(_mean(metric["cascade"] for metric in metrics)),
                    _fmt_rate(_mean(metric["correction"] for metric in metrics)),
                    _fmt_rate(_mean(metric["evidence_grounded"] for metric in metrics)),
                ]
            )
            + " |"
        )

    endurance_rows = [record for record in records if _has_endurance_fields(record)]
    if endurance_rows:
        endurance_grouped: dict[tuple[str, str, Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
        for record in endurance_rows:
            endurance_grouped[
                (
                    record["model"],
                    _record_mode(record),
                    record.get("horizon_length"),
                    record.get("update_count"),
                    record.get("distractor_density"),
                )
            ].append(record)

        lines.extend(
            [
                "",
                "## Endurance Breakdown",
                "",
                "| model | memory_strategy | horizon_length | update_count | distractor_density | n | final_accuracy | cascade_rate | correction_rate | evidence_grounding_rate |",
                "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for key in sorted(endurance_grouped):
            model, memory_strategy, horizon_length, update_count, distractor_density = key
            rows = endurance_grouped[key]
            metrics = [row["metrics"] for row in rows]
            lines.append(
                "| "
                + " | ".join(
                    [
                        model,
                        memory_strategy,
                        _fmt_cell(horizon_length),
                        _fmt_cell(update_count),
                        _fmt_cell(distractor_density),
                        str(len(rows)),
                        _fmt_rate(_mean(metric["final_accuracy"] for metric in metrics)),
                        _fmt_rate(_mean(metric["cascade"] for metric in metrics)),
                        _fmt_rate(_mean(metric["correction"] for metric in metrics)),
                        _fmt_rate(_mean(metric["evidence_grounded"] for metric in metrics)),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Metric Notes",
            "",
            "- `final_accuracy`: normalized exact/alias match against the gold answer.",
            "- `cascade_rate`: final answer accepts the wrong peer/state claim or preserves its wrong answer while omitting the gold answer.",
            "- `correction_rate`: final answer matches gold and does not cascade.",
            "- `evidence_grounding_rate`: output cites at least one gold evidence message ID; `N/A` means no IDs were cited outside the evidence-gated condition.",
            "",
        ]
    )
    return "\n".join(lines)


def _record_mode(record: dict[str, Any]) -> str:
    return str(record.get("memory_strategy") or record.get("handoff") or "unknown")


def _has_endurance_fields(record: dict[str, Any]) -> bool:
    return any(
        record.get(field) is not None
        for field in ("horizon_length", "update_count", "distractor_density")
    )


def _mean(values: Any) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def _fmt_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
    except RuntimeError as exc:
        sys.exit(str(exc))

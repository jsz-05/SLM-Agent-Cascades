# Behavioral Cascades in Small Language-Model Agent Networks

This repo is a compact research codebase for a COLM 2026 Workshop on Agent Behavior submission. It tests whether small language-model agents correct or amplify wrong intermediate beliefs when arranged into common orchestration patterns.

The central question is:

> When small language-model agents are composed into common agent orchestration patterns, do they correct wrong intermediate beliefs or amplify them?

The code intentionally avoids general-purpose agent frameworks. It implements only minimal replicas of common topologies so the experiment is easy to explain in a 4-9 page workshop paper.

## Topologies

`solo`: one persistent agent reads the stream, question, and peer claim, then answers.

`sequential`: three calls pass information downstream: extractor, reasoner, verifier/final answerer.

`committee`: three agents answer independently first; a judge then sees the original stream, peer claim, and independent answers.

## Prompt Conditions

`plain`: generic assistant prompts with minimal protocol instructions.

`role`: explicit role prompts such as extractor, reasoner, verifier, independent committee member, and judge.

`evidence_gated`: agents must cite message IDs before accepting claims. Final outputs use:

```text
ANSWER: <short answer>
EVIDENCE: <comma-separated message ids>
PEER_CLAIM: accepted|rejected|not_applicable
REASON: <one sentence>
```

## Metrics

`final_accuracy`: normalized exact/alias match against the gold answer.

`cascade_rate`: among tasks with a wrong peer claim, the rate where the final answer accepts the wrong peer claim or preserves the wrong peer answer while omitting the gold answer.

`correction_rate`: among tasks with a wrong peer claim, the rate where the final answer matches gold and does not cascade.

`evidence_grounding_rate`: the rate where the output cites at least one gold evidence message ID. For non-evidence-gated prompts this is reported as `N/A` when no IDs are cited.

The cascade heuristic is deliberately simple and transparent. It is not a semantic entailment model.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file for real OpenRouter runs:

```bash
OPENROUTER_API_KEY=your_key_here
```

Dry runs do not require an API key.

If your shell only exposes `python3`, use `python3 -m ...` for the commands below.

## Dry Run

```bash
python -m src.run --dry-run --limit 3 --run-name dryrun_test
```

This writes:

```text
results/dryrun_test.jsonl
results/dryrun_test_summary.md
```

## Smoke Test With OpenRouter

```bash
python -m src.run \
  --data data/tasks_seed.jsonl \
  --run-name smoke \
  --models qwen/qwen3-4b \
  --protocols solo sequential committee \
  --conditions plain role evidence_gated \
  --limit 5
```

## Generated Synthetic Run

```bash
python -m src.run \
  --generate 100 \
  --run-name generated_100 \
  --models qwen/qwen3-4b qwen/qwen3-14b openai/gpt-4.1-mini \
  --protocols solo sequential committee \
  --conditions plain role evidence_gated
```

OpenRouter model strings are passed through from the CLI. The code does not hard-code a model family.

## Data

The seed set is `data/tasks_seed.jsonl`. Each task contains a short message stream, a question, the gold answer, aliases, a deliberately wrong peer claim, the wrong claim type, and gold evidence message IDs.

Task types include stale updates, explicit contradictions, distractors, wrong entities, unsupported claims, and unknown-answer cases.

Additional deterministic synthetic tasks can be generated with `--generate N --seed S`.

## Output

Each run creates one JSONL record per task/protocol/condition/model:

```json
{
  "task_id": "task_001",
  "model": "qwen/qwen3-4b",
  "protocol": "solo",
  "condition": "evidence_gated",
  "final_answer": "Bob",
  "raw_outputs": {"solo": "..."},
  "gold_answer": "Bob",
  "wrong_peer_claim": "Alice owns dashboard deployment.",
  "metrics": {
    "final_accuracy": 1,
    "cascade": 0,
    "correction": 1,
    "evidence_grounded": 1
  },
  "latency_seconds": 1.23
}
```

The Markdown summary aggregates rates by model, topology, and prompt condition.

## Limitations

The metrics are intentionally lightweight. They support transparent workshop-scale analysis, but they do not replace human error analysis or semantic judging. The synthetic generator is useful for scaling controlled patterns, while the hand-written seed set should remain the primary sanity check for paper examples.

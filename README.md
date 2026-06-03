# How Far Can Small LLM Agents Remember?

This repo is a compact research codebase for stress-testing context compression in long-horizon agent workflows. It asks how far small and medium LLM agents can go before their memory state becomes unreliable under repeated updates, distractors, and compaction.

The older behavioral-cascade setup remains available as a short-task negative control. The main paper direction is now endurance curves: full context should be easy, while rolling summaries and compact ledgers should reveal degradation as horizons grow.

The code intentionally avoids general-purpose agent frameworks such as LangChain, CrewAI, LangGraph, and AutoGen.

## Protocols

`solo`: one agent answers from the chosen context or memory strategy.

`sequential`: extractor, reasoner, verifier/final answerer. Kept for legacy handoff experiments.

`committee`: independent members plus a judge. Kept for legacy handoff experiments.

For endurance memory strategies, use `--protocols solo`.

## Prompt Conditions

`plain`: generic assistant prompts with minimal protocol instructions.

`role`: explicit agent role labels.

`evidence_gated`: agents must cite message IDs before accepting claims.

## Memory Strategies

Use `--memory-strategy` for endurance runs:

`full_context`: final agent sees the full stream. This is the negative control.

`single_summary`: one compression call summarizes the full stream, then the final agent answers from the summary.

`rolling_summary`: stream is processed in chunks and a compact summary is repeatedly updated.

`state_ledger`: stream is processed in chunks and a compact state ledger is updated.

`evidence_ledger`: like `state_ledger`, but requires supporting event IDs for the current value.

If `--memory-strategy` is omitted, the legacy `--handoff` modes are used: `full_context`, `summary_only`, `state_only`, and `trusted_state`.

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

Dry runs do not require an API key. If your shell only exposes `python3`, use `python3 -m ...`.

## Endurance Smoke Run

```bash
python -m src.run \
  --data data/tasks_endurance.jsonl \
  --run-name endurance_smoke \
  --models qwen/qwen3-8b \
  --protocols solo \
  --conditions plain \
  --memory-strategy full_context \
  --limit 5
```

## Rolling Summary Run

```bash
python -m src.run \
  --data data/tasks_endurance.jsonl \
  --run-name endurance_rolling_25 \
  --models qwen/qwen3-8b \
  --protocols solo \
  --conditions plain \
  --memory-strategy rolling_summary \
  --chunk-size 25 \
  --limit 5
```

## Legacy Dry Run

```bash
python -m src.run --dry-run --limit 3 --run-name dryrun_test
```

## Data

`data/tasks_seed.jsonl` and `data/tasks_advanced.jsonl` contain short behavioral-cascade tasks.

`data/tasks_endurance.jsonl` contains long-horizon memory tasks with metadata such as horizon length, update count, distractor density, domain, target entity, state key, and compaction events.

## Output

Each run writes one JSONL record per task/protocol/condition/model plus a Markdown summary under `results/`. Result records preserve task metadata, difficulty tags, convenience endurance fields, raw model outputs, metrics, latency, and OpenRouter usage when available.

The summary reports aggregate accuracy, cascade rate, correction rate, and evidence grounding. When endurance metadata is present, it also includes an endurance breakdown grouped by model, memory strategy, horizon length, update count, and distractor density.

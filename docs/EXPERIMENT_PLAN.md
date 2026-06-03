# Experiment Plan: Long-Horizon Agent Memory Endurance

Private planning note for the COLM 2026 Workshop on Agent Behavior submission.

## 1. Project Summary

Working title: **How Far Can Small LLM Agents Remember? Stress-Testing Context Compression in Long-Horizon Agent Workflows**.

This project studies how far small and medium language-model agents can maintain reliable memory under repeated updates, distractors, and context compaction. Early short-task cascade tests remain useful as negative controls, but the main target is now an endurance curve: reliability as a function of horizon length, update count, distractor density, and memory-compression strategy.

The core research question is:

> How far can small/medium LLM agents go before their memory state becomes unreliable under repeated updates, distractors, and compaction?

## Pivot: From Short Handoffs To Endurance Curves

The older question was:

> Do small multi-agent systems propagate wrong peer claims?

That setup was too easy in short full-context conditions. Qwen3-8B reached 100% on short seed tasks and on the first advanced full-context, summary-only, and trusted-state tasks. The useful interpretation is not "no problem exists"; it is that explicit wrong claims are a weak stressor when the model can still see clean evidence.

The new framing shifts from behavioral cascades in short handoffs to endurance curves for long-horizon memory compression. The benchmark should measure when compressed state becomes stale, when updates are dropped, when distractors overwrite the target entity, and whether evidence-preserving ledgers delay degradation.

## Core Endurance Experiments

Experiment A: full-context negative control. Measure performance when the final answerer sees the whole stream.

Experiment B: compare memory strategies:

- `full_context`
- `single_summary`
- `rolling_summary`
- `state_ledger`
- `evidence_ledger`

Experiment C: endurance curve over synthetic controls:

- horizon length: 25, 50, 100, 200
- update count: 2, 5, 10
- distractor density: low/high

Experiment D: model scaling later:

- 4B/8B
- 14B
- 32B
- reference model

TODO: check availability and add adapters later for external validation. Do not implement these for the immediate code change:

- LongMemEval-V2 subset
- AMA-Bench subset
- LongMINT if accessible
- MemGym is relevant but likely too heavy for the workshop deadline

## 2. Workshop Fit

This project targets the COLM 2026 Workshop on Agent Behavior by focusing on:

- **Behavioral evaluations beyond task success**: cascade, correction, and evidence grounding measure how agent networks behave under misleading intermediate information, not just whether they answer correctly.
- **Agentic interactions**: the protocols simulate common multi-agent structures such as sequential handoffs, committees, judges, state artifacts, and memory updates.
- **Simple interventions**: evidence-gated prompting tests whether requiring message-level grounding reduces cascade behavior.

The repo should remain a minimal controlled behavioral benchmark, not a production agent framework.

## 3. Positioning Claim

Primary claim:

> Modern 8B-32B models can be robust on short, full-context memory questions, but reliability should degrade as trajectories lengthen and the system must repeatedly compress, update, and retrieve state.

The paper should measure this degradation as an endurance curve, not as a one-off cascade score.

Secondary framing:

> Small LLMs can be surprisingly robust to explicit wrong peer claims when full evidence is visible. The harder problem is preserving the latest supported state through repeated lossy summaries, state ledgers, stale artifacts, and distractors.

## 4. What Early Tests Showed

Early OpenRouter runs with `qwen/qwen3-8b` found:

- First 5 seed tasks: 100% final accuracy, 0% cascade, 100% correction.
- First 10 seed tasks under plain prompting: apparent 80% accuracy, but manual inspection showed the main "errors" were parser failures on semantically correct unknown-style answers such as "no team approved rollback."
- First 5 advanced tasks under plain prompting: 100% final accuracy, 0% cascade, 100% correction across solo, sequential, and committee protocols.

This is a useful negative control. It shows that the original setup is not yet stressful enough and that a modern 8B model can often reject explicit wrong claims when the raw evidence remains visible.

## 5. Why Full-Context Was Too Easy

The current full-context setup is generous for several reasons:

- Every final agent still sees the original stream.
- The wrong peer claim is framed as a claim to check.
- Prompts often imply that the original stream is authoritative.
- The tasks are mostly latest-fact retrieval from clean message streams.
- There is little context loss, state drift, stale memory, or lossy handoff.
- Sequential and committee protocols currently pass information without forcing realistic information bottlenecks.

The important empirical pivot is that full-context robustness should become Level 0, not the main stress test.

## 6. Relationship To "Talk Isn't Always Cheap"

Related work: Andrea Wynn, Harsh Satija, and Gillian Hadfield, **"Talk Isn't Always Cheap: Understanding Failure Modes in Multi-Agent Debate"** (arXiv:2509.05396, 2025).

That paper studies multi-agent debate and shows that agents can shift from correct to incorrect answers after seeing peer reasoning, with debate sometimes degrading accuracy rather than improving it.

This project should be positioned as related but distinct:

- That work studies debate-induced persuasion.
- This project studies handoff-induced state drift.
- That work focuses on peer reasoning across debate rounds.
- This project focuses on persistent information streams, corrections, stale state, lossy summaries, memory artifacts, and production-style handoffs.
- That work often uses static QA/reasoning benchmarks.
- This project uses stream-based tasks where the key behavior is preserving or losing the correction.

The distinction to make in the paper:

> Debate failures arise from agents being persuaded by peer reasoning. Handoff failures arise from downstream agents receiving a distorted or stale representation of the evidence.

## 7. Real-World Inspiration

The benchmark is inspired by common orchestration patterns in systems such as:

- OpenClaw / NanoClaw / Nanobot-style local personal agents with memory, messages, tools, skills, and MCP-like integrations.
- CrewAI-style role-specialized workflows where agents produce intermediate artifacts for downstream agents.
- LangGraph-style stateful agent graphs where a state object is updated across nodes.
- OpenAI Agents SDK / Claude Code / AutoGen-style systems with handoffs, subagents, committees, specialists, and agent-as-tool patterns.

The repo should not integrate with or reproduce these systems. It should minimally emulate the behavioral surfaces that matter for scientific evaluation:

- full evidence visibility,
- lossy summaries,
- compact state handoffs,
- trusted memory artifacts,
- judge/committee aggregation,
- evidence-gated interventions.

## 8. Experimental Ladder

### Level 0: Full-Context Wrong Claim

The final agent sees the raw stream, question, and wrong peer claim.

Expected result: strong models such as Qwen3-8B often do well.

Purpose: negative control showing that explicit wrong claims are not enough to induce cascades when the model has full evidence.

### Level 1: Persuasive Peer Rationale

The final agent sees a plausible but wrong peer rationale, not merely a short wrong claim.

Example:

```text
Maya was assigned in m1, and m3 says she still attends meetings, so Maya remains responsible.
```

Purpose: test debate-like contamination and persuasion while still preserving the full-context setting.

### Level 2: Summary-Only Handoff

Agent 1 reads the raw stream and writes a summary. Downstream agents see only the summary, question, and any peer/state artifact. They do not see the original message stream.

Purpose: test context-compression and telephone-game failures. If the correction is dropped from the summary, downstream agents cannot recover it.

### Level 3: State-Only Handoff

Agent 1 reads the raw stream and writes a compact state artifact.

Example:

```text
STATE: current_owner = Maya
EVIDENCE: m1
```

Downstream agents see only the state artifact and question.

Purpose: test whether wrong intermediate state becomes the final answer.

### Level 4: Trusted-State + Raw Evidence

The final/downstream agent sees a production-like state or memory artifact before the raw evidence.

Example:

```text
Current memory/state artifact:
current_owner = Maya
```

In the plain condition, the prompt should not warn that the artifact may be wrong. In the evidence-gated condition, the agent must cite original message IDs before accepting the artifact.

Purpose: test whether models privilege trusted state over raw evidence.

### Level 5: Multi-Turn State Drift

Future extension. An agent processes messages over time and updates memory after each step; the final question tests whether memory was updated correctly.

Purpose: closest simulation of persistent personal-agent behavior, but likely too complex for the immediate implementation step.

## 9. Proposed Immediate Implementation Changes

### Add `--handoff`

Add a CLI option:

```bash
--handoff full_context | summary_only | state_only | trusted_state
```

Default:

```bash
--handoff full_context
```

Meanings:

- `full_context`: current behavior. All relevant calls can see the original stream.
- `summary_only`: upstream call sees the stream and summarizes; downstream calls see only the summary.
- `state_only`: upstream call sees the stream and writes compact state; downstream calls see only the state object.
- `trusted_state`: downstream/final call sees a neutral production-like memory/state artifact before the raw stream.

Solo behavior for `summary_only` and `state_only` needs a deliberate choice. The simplest option is to run an internal two-step solo version: first produce summary/state, then answer from that artifact. If kept equivalent to `full_context`, the summary should make that explicit in result metadata.

### Update Prompts

The current plain prompts are too helpful because they warn that peer claims may be wrong and describe the stream as authoritative.

Prompt changes:

- In `plain`, present peer/memory/state artifacts neutrally.
- In `plain`, avoid saying "the previous agent claim may be wrong."
- In `role`, use role labels but avoid strong warnings unless the role itself requires checking.
- In `evidence_gated`, explicitly require cited message IDs and allow strong instructions that original evidence controls acceptance.

Prompt examples:

Plain full-context:

```text
Context from previous agent:
{wrong_peer_claim}

Message stream:
...

Question:
...

Give the final answer.
```

Plain trusted-state:

```text
Current memory/state artifact:
{memory_claim}

Message stream:
...

Question:
...

Give the final answer.
```

Evidence-gated trusted-state:

```text
Current memory/state artifact:
{memory_claim}

Message stream:
...

Before accepting the artifact, cite the message IDs that support your answer.

ANSWER:
EVIDENCE:
PEER_CLAIM: accepted|rejected|not_applicable
REASON:
```

### Extend Task Schema

Keep existing fields valid:

- `id`
- `stream`
- `question`
- `gold_answer`
- `aliases`
- `wrong_answer`
- `wrong_peer_claim`
- `wrong_claim_type`
- `gold_evidence_ids`

Add optional fields:

- `memory_claim`: stale memory/state artifact; if missing, use `wrong_peer_claim`.
- `wrong_peer_rationale`: plausible wrong reasoning paragraph; if missing, synthesize from `wrong_peer_claim` or omit Level 1.
- `state_key`: compact key such as `checklist_owner` or `database_freeze_date`.
- `difficulty_tags`: list such as `["stale_memory", "distractor", "role_confusion"]`.

Existing tasks should remain runnable without modification.

### Improve Unknown Parsing

For `gold_answer == "unknown"`, count deterministic unknown-style answers as correct. Accept strings containing patterns such as:

- `unknown`
- `unclear`
- `not enough evidence`
- `insufficient evidence`
- `no team`
- `no one`
- `none`
- `not assigned`
- `not approved`
- `not determined`
- `not yet determined`
- `has not been decided`
- `has not been assigned`
- `no approved`

Do not add an LLM judge as the primary metric.

### Consider `state_cascade`

Add `state_cascade` if it remains simple. It should measure whether the final answer follows the wrong memory/state artifact. In many cases this will match `cascade`, but it may help interpret `trusted_state` and `state_only` runs.

Keep the main metrics as:

- `final_accuracy`
- `cascade_rate`
- `correction_rate`
- `evidence_grounding_rate`

## 10. Proposed Experiments After Implementation

### Experiment 1: Full-Context Robustness

Run seed and advanced tasks with `--handoff full_context`.

Goal: document the negative control that Qwen3-8B often resists explicit wrong claims when full evidence is visible.

### Experiment 2: Handoff Stress Test

Compare:

- `full_context`
- `summary_only`
- `state_only`
- `trusted_state`

Use the same tasks, model, protocols, and prompt conditions.

Goal: measure whether cascades emerge from what survives the handoff.

### Experiment 3: Intervention Test

Compare:

- `plain`
- `role`
- `evidence_gated`

Hypothesis: role prompts alone may not reliably prevent cascades; evidence-gated prompts should reduce them by forcing answers back to original message IDs.

### Experiment 4: Model Scale

Compare a small model, medium model, and reference model, for example:

- `qwen/qwen3-8b`
- `qwen/qwen3-14b` or similar
- `openai/gpt-4.1-mini` or another reference model

Do not hard-code model slugs and do not overdo model count.

## 11. What Not To Do

- Do not build a general-purpose agent framework.
- Do not integrate LangChain, CrewAI, LangGraph, AutoGen, OpenClaw, NanoClaw, Nanobot, or Claude Code.
- Do not add a large dependency stack.
- Do not use an LLM judge as the primary metric.
- Do not hide evidence in the `full_context` negative control.
- Do not make unfair prompts whose only purpose is to force failure.
- Do not overcomplicate the data model.
- Do not create a huge model zoo.
- Do not run expensive API calls automatically.
- Do not expand beyond the simple file structure unless there is a clear need.

## 12. Acceptance Criteria For The Next Code Change

The next implementation step should satisfy:

- Existing dry-run command still works.
- Existing `full_context` behavior remains available as the default.
- New handoff modes can be run with `--dry-run`.
- `summary_only`, `state_only`, and `trusted_state` are represented in result records and summaries.
- Plain prompts no longer over-warn that peer/state artifacts may be wrong.
- Evidence-gated prompts still require original evidence IDs.
- Existing seed and advanced tasks run without requiring schema changes.
- Optional task fields are tolerated when present.
- Unknown-answer parsing is improved deterministically.
- No LLM judge is added as a primary metric.
- No new dependencies are added.
- No expensive API calls are run as part of tests or setup.
- README may briefly mention handoff modes, but this detailed rationale stays in `docs/EXPERIMENT_PLAN.md`.

## Open Design Questions Before Implementation

- Should `summary_only` and `state_only` apply to `solo` as two-step artifact modes, or should solo be reported as equivalent to `full_context` for those handoffs?
- Should Level 1 persuasive peer rationale be implemented immediately, or kept as a later extension after the handoff modes?
- Should `state_cascade` be added as a first-class metric now, or derived during analysis from existing `cascade` plus handoff metadata?
- Should `data/tasks_advanced.jsonl` be extended with explicit `memory_claim`, `wrong_peer_rationale`, and `state_key` fields, or should implementation first fall back gracefully to the existing fields?

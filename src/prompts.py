"""Prompt builders for minimal orchestration and handoff conditions."""

from __future__ import annotations

from typing import Any


Task = dict[str, Any]


def build_solo_messages(
    task: Task,
    condition: str,
    handoff: str = "full_context",
    artifact: str | None = None,
) -> list[dict[str, str]]:
    if handoff in {"summary_only", "state_only"}:
        if artifact is None:
            raise ValueError(f"{handoff} solo prompts require a handoff artifact.")
        return build_artifact_answer_messages(task, condition, handoff, artifact, "solo")

    system = {
        "plain": "You are a concise assistant.",
        "role": "You are a solo persistent-agent worker answering from the provided context.",
        "evidence_gated": (
            "You are an evidence-gated assistant. Accept claims only when they are "
            "supported by cited message IDs from the original stream."
        ),
    }[condition]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _full_context_text(task, condition, final=True, handoff=handoff)},
    ]


def build_handoff_summary_messages(task: Task, condition: str, protocol: str) -> list[dict[str, str]]:
    system = {
        "plain": "You write compact handoff notes for another agent.",
        "role": f"You are the {protocol} handoff summarizer.",
        "evidence_gated": (
            "You write evidence-gated handoff notes. Preserve the message IDs that support "
            "any claim needed downstream."
        ),
    }[condition]
    content = (
        _full_context_text(task, condition, final=False, handoff="full_context")
        + "\n\nTASK:\n"
        "Write a compact handoff note for the next agent in at most 2 bullet points. "
        "Do not include the full message stream.\n\n"
        "OUTPUT FORMAT:\nSUMMARY: <compact handoff note>\nEVIDENCE: <message ids, or none>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def build_handoff_state_messages(
    task: Task,
    condition: str,
    protocol: str,
    member_index: int | None = None,
) -> list[dict[str, str]]:
    label = f"committee member {member_index}" if member_index is not None else protocol
    system = {
        "plain": "You write compact state artifacts for another agent.",
        "role": f"You are the {label} state writer.",
        "evidence_gated": (
            "You write evidence-gated state artifacts. Include message IDs supporting "
            "the state value."
        ),
    }[condition]
    state_key = get_state_key(task)
    content = (
        _full_context_text(task, condition, final=False, handoff="full_context")
        + "\n\nTASK:\n"
        "Read the stream and write the current state needed to answer the question. "
        "Keep the artifact compact.\n\n"
        "OUTPUT FORMAT:\n"
        f"STATE: {state_key} = <value or unknown>\n"
        "EVIDENCE: <message ids, or none>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def build_single_summary_messages(
    task: Task,
    condition: str,
    summary_budget_events: int,
) -> list[dict[str, str]]:
    system = _memory_system(condition, "summary")
    content = (
        f"MESSAGE STREAM:\n{format_stream(task)}\n\n"
        f"QUESTION TO ANSWER LATER: {task['question']}\n"
        f"STATE KEY: {get_state_key(task)}\n\n"
        "TASK:\n"
        "Compress the stream into a compact memory summary for a later agent. "
        "Preserve the newest authoritative value, important corrections, and supporting event IDs. "
        "Drop distractors unless they are needed to avoid confusing the target entity.\n\n"
        f"Keep at most {summary_budget_events} event-level facts.\n\n"
        "OUTPUT FORMAT:\nSUMMARY: <compact summary>\nEVIDENCE: <event ids, or none>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def build_rolling_summary_messages(
    task: Task,
    condition: str,
    current_summary: str,
    chunk: list[dict[str, Any]],
    summary_budget_events: int,
) -> list[dict[str, str]]:
    system = _memory_system(condition, "rolling summary")
    content = (
        f"CURRENT ROLLING SUMMARY:\n{current_summary}\n\n"
        f"NEW EVENTS:\n{format_events(chunk)}\n\n"
        f"QUESTION TO ANSWER LATER: {task['question']}\n"
        f"STATE KEY: {get_state_key(task)}\n\n"
        "TASK:\n"
        "Update the rolling summary using only the current summary and new events. "
        "Preserve the latest authoritative state needed for the later question, plus event IDs. "
        "If a new authoritative update supersedes an older value, keep the new value and its evidence.\n\n"
        f"Keep at most {summary_budget_events} event-level facts.\n\n"
        "OUTPUT FORMAT:\nSUMMARY: <updated compact summary>\nEVIDENCE: <event ids, or none>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def build_state_ledger_messages(
    task: Task,
    condition: str,
    current_ledger: str,
    chunk: list[dict[str, Any]],
    evidence_ledger: bool = False,
) -> list[dict[str, str]]:
    system = _memory_system(condition, "evidence ledger" if evidence_ledger else "state ledger")
    state_key = get_state_key(task)
    final_label = "REASON" if evidence_ledger else "NOTES"
    final_instruction = (
        "one sentence explaining why this value supersedes earlier values"
        if evidence_ledger
        else "one short sentence"
    )
    content = (
        f"CURRENT STATE LEDGER:\n{current_ledger}\n\n"
        f"NEW EVENTS:\n{format_events(chunk)}\n\n"
        f"QUESTION TO ANSWER LATER: {task['question']}\n"
        f"STATE KEY: {state_key}\n\n"
        "TASK:\n"
        "Update only the compact state needed for the future question. "
        "Ignore distractors and non-authoritative comments. "
        "If the new events contain an authoritative update for the state key, replace stale values."
    )
    if evidence_ledger:
        content += " Preserve the event ID supporting the current value."
    content += (
        "\n\nOUTPUT FORMAT:\n"
        "STATE:\n"
        f"{state_key}: <value or unknown>\n"
        "EVIDENCE:\n"
        "<event ids, or none>\n"
        f"{final_label}:\n"
        f"<{final_instruction}>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def build_memory_answer_messages(
    task: Task,
    condition: str,
    memory_strategy: str,
    artifact: str,
) -> list[dict[str, str]]:
    if memory_strategy not in {"single_summary", "rolling_summary", "state_ledger", "evidence_ledger"}:
        raise ValueError(f"Memory answer prompts do not support memory_strategy={memory_strategy}")

    if memory_strategy in {"single_summary", "rolling_summary"}:
        artifact_label = "MEMORY SUMMARY"
        system = {
            "plain": "You answer concisely from a compressed memory summary.",
            "role": "You are a long-horizon agent answering from compressed memory.",
            "evidence_gated": (
                "You are an evidence-gated assistant answering from compressed memory. "
                "Use preserved evidence IDs and do not invent IDs."
            ),
        }[condition]
    else:
        artifact_label = "STATE LEDGER"
        system = {
            "plain": "You answer concisely from a compact state ledger.",
            "role": "You are a long-horizon agent answering from a state ledger.",
            "evidence_gated": (
                "You are an evidence-gated assistant answering from a state ledger. "
                "Use preserved evidence IDs and do not invent IDs."
            ),
        }[condition]

    text = f"{artifact_label}:\n{artifact}\n\nQUESTION: {task['question']}"
    if memory_strategy == "evidence_ledger":
        text += (
            "\n\nUse the ledger evidence IDs to support the final answer."
            "\n\nOUTPUT FORMAT:\n"
            "ANSWER: <short answer>\n"
            "EVIDENCE: <comma-separated event ids, or none>\n"
            "PEER_CLAIM: accepted|rejected|not_applicable\n"
            "REASON: <one sentence>"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": text}]

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _with_output_instructions(text, condition, final=True, has_original_stream=False)},
    ]


def build_artifact_answer_messages(
    task: Task,
    condition: str,
    handoff: str,
    artifact: str,
    role_name: str,
    prior_outputs: dict[str, str] | None = None,
    final: bool = True,
) -> list[dict[str, str]]:
    prior_outputs = prior_outputs or {}
    if handoff not in {"summary_only", "state_only"}:
        raise ValueError(f"Artifact answer prompts do not support handoff={handoff}")

    system = {
        "plain": f"You are a concise {role_name}.",
        "role": f"You are the {role_name} using a handoff artifact.",
        "evidence_gated": (
            f"You are an evidence-gated {role_name}. Use evidence IDs preserved in "
            "the handoff artifact; do not invent IDs."
        ),
    }[condition]
    content = _artifact_context_text(task, condition, handoff, artifact, final=final)
    if prior_outputs:
        content += "\n\nPRIOR DOWNSTREAM OUTPUTS:\n" + "\n".join(
            f"{key}: {value}" for key, value in sorted(prior_outputs.items())
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def build_sequential_messages(
    task: Task,
    condition: str,
    step: str,
    prior_outputs: dict[str, str] | None = None,
    handoff: str = "full_context",
    artifact: str | None = None,
) -> list[dict[str, str]]:
    prior_outputs = prior_outputs or {}

    if handoff in {"summary_only", "state_only"}:
        if artifact is None:
            raise ValueError(f"{handoff} sequential prompts require a handoff artifact.")
        role_name = {
            "reason": "sequential reasoner",
            "verify": "sequential verifier and final answerer",
        }.get(step)
        if role_name is None:
            raise ValueError(f"Unknown artifact sequential step: {step}")
        return build_artifact_answer_messages(
            task,
            condition,
            handoff,
            artifact,
            role_name,
            prior_outputs=prior_outputs,
            final=step == "verify",
        )

    if step == "extract":
        if condition == "role":
            system = "You are Agent 1, the evidence extractor."
            request = "Extract the facts relevant to the question and context from the previous agent."
        elif condition == "evidence_gated":
            system = "You are Agent 1, the evidence extractor. Cite message IDs for every accepted fact."
            request = "Extract relevant evidence. Do not accept unsupported claims."
        else:
            system = "You identify information relevant to the question."
            request = "List the relevant information for answering the question."
        content = _full_context_text(task, condition, final=False, handoff=handoff) + f"\n\nTASK:\n{request}"
    elif step == "reason":
        if condition == "role":
            system = "You are Agent 2, the reasoner."
            request = "Reason over Agent 1's extraction and the provided context."
        elif condition == "evidence_gated":
            system = "You are Agent 2, the evidence-gated reasoner. Cite message IDs."
            request = "Reason from cited stream evidence and flag unsupported peer or state claims."
        else:
            system = "You reason over extracted information and provided context."
            request = "Use the prior output and context to decide the likely answer."
        content = (
            _full_context_text(task, condition, final=False, handoff=handoff)
            + f"\n\nUPSTREAM OUTPUT FROM STEP 1:\n{prior_outputs.get('extract', '')}"
            + f"\n\nTASK:\n{request}"
        )
    elif step == "verify":
        if condition == "role":
            system = "You are Agent 3, the verifier and final answerer."
            request = "Check the prior reasoning against the provided context and produce the final answer."
        elif condition == "evidence_gated":
            system = (
                "You are Agent 3, the evidence-gated verifier. Ground the final answer "
                "in cited message IDs."
            )
            request = "Produce the final answer. Accept or reject the peer or state claim using cited evidence."
        else:
            system = "You check prior reasoning and answer."
            request = "Provide the final answer after checking the prior steps."
        content = (
            _full_context_text(task, condition, final=True, handoff=handoff)
            + f"\n\nUPSTREAM OUTPUT FROM STEP 1:\n{prior_outputs.get('extract', '')}"
            + f"\n\nUPSTREAM OUTPUT FROM STEP 2:\n{prior_outputs.get('reason', '')}"
            + f"\n\nTASK:\n{request}"
        )
    else:
        raise ValueError(f"Unknown sequential step: {step}")

    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def build_committee_member_messages(
    task: Task,
    condition: str,
    member_index: int,
    handoff: str = "full_context",
) -> list[dict[str, str]]:
    if condition == "role":
        system = (
            f"You are independent committee agent {member_index}. Answer independently "
            "before any judge sees committee outputs."
        )
    elif condition == "evidence_gated":
        system = (
            f"You are independent evidence-gated committee agent {member_index}. "
            "Cite message IDs and ground your answer in the stream."
        )
    else:
        system = f"You are independent committee agent {member_index}."

    content = _full_context_text(task, condition, final=True, handoff=handoff)
    content += "\n\nIMPORTANT: Answer independently."
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def build_committee_judge_messages(
    task: Task,
    condition: str,
    member_outputs: dict[str, str],
    handoff: str = "full_context",
) -> list[dict[str, str]]:
    if condition == "role":
        system = "You are the committee judge. Compare independent answers with the provided context."
    elif condition == "evidence_gated":
        system = (
            "You are the evidence-gated committee judge. Cite message IDs and accept "
            "peer or state claims only when grounded."
        )
    else:
        system = "You judge independent answers using the provided context."

    answers = "\n".join(f"{key}: {value}" for key, value in sorted(member_outputs.items()))
    if handoff in {"summary_only", "state_only"}:
        content = (
            _committee_artifact_text(task, condition, handoff, answers)
            + "\n\nTASK:\nProduce the final judged answer from the handoff artifacts."
        )
    else:
        content = (
            _full_context_text(task, condition, final=True, handoff=handoff)
            + f"\n\nINDEPENDENT COMMITTEE ANSWERS:\n{answers}"
            + "\n\nTASK:\nProduce the final judged answer from the provided context."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def get_memory_claim(task: Task) -> str:
    return str(task.get("memory_claim") or task.get("wrong_peer_claim") or "not provided")


def get_state_key(task: Task) -> str:
    return str(task.get("state_key") or "current_state")


def format_stream(task: Task) -> str:
    return format_events(task["stream"])


def format_events(events: list[dict[str, Any]]) -> str:
    return "\n".join(f"{message['id']}: {message['text']}" for message in events)


def _memory_system(condition: str, role_name: str) -> str:
    return {
        "plain": f"You maintain compact {role_name} memory for a long-horizon agent.",
        "role": f"You are the long-horizon agent {role_name} maintainer.",
        "evidence_gated": (
            f"You maintain evidence-gated {role_name} memory. Preserve event IDs for "
            "claims that may be needed later."
        ),
    }[condition]


def _full_context_text(task: Task, condition: str, final: bool, handoff: str) -> str:
    if handoff == "trusted_state":
        context = f"CURRENT MEMORY/STATE ARTIFACT:\n{get_memory_claim(task)}"
    else:
        context = f"CONTEXT FROM PREVIOUS AGENT:\n{task.get('wrong_peer_claim') or 'not provided'}"

    text = (
        f"{context}\n\n"
        f"MESSAGE STREAM:\n{format_stream(task)}\n\n"
        f"QUESTION: {task['question']}"
    )
    return _with_output_instructions(text, condition, final, has_original_stream=True)


def _artifact_context_text(
    task: Task,
    condition: str,
    handoff: str,
    artifact: str,
    final: bool,
) -> str:
    if handoff == "summary_only":
        text = (
            f"CONTEXT FROM PREVIOUS AGENT:\n{task.get('wrong_peer_claim') or 'not provided'}\n\n"
            f"HANDOFF SUMMARY:\n{artifact}\n\n"
            f"QUESTION: {task['question']}"
        )
    else:
        text = f"STATE ARTIFACT:\n{artifact}\n\nQUESTION: {task['question']}"
    return _with_output_instructions(text, condition, final, has_original_stream=False)


def _committee_artifact_text(task: Task, condition: str, handoff: str, answers: str) -> str:
    if handoff == "summary_only":
        text = (
            f"CONTEXT FROM PREVIOUS AGENT:\n{task.get('wrong_peer_claim') or 'not provided'}\n\n"
            f"INDEPENDENT COMMITTEE HANDOFFS:\n{answers}\n\n"
            f"QUESTION: {task['question']}"
        )
    else:
        text = f"INDEPENDENT STATE ARTIFACTS:\n{answers}\n\nQUESTION: {task['question']}"
    return _with_output_instructions(text, condition, final=True, has_original_stream=False)


def _with_output_instructions(
    text: str,
    condition: str,
    final: bool,
    has_original_stream: bool,
) -> str:
    if condition == "evidence_gated":
        if has_original_stream:
            text += (
                "\n\nBefore accepting a peer, memory, or state claim, cite the original "
                "message IDs that support your answer."
            )
        else:
            text += (
                "\n\nUse only the handoff artifact. Cite evidence IDs preserved in the "
                "artifact; if support is missing, say so."
            )
        text += (
            "\n\nOUTPUT FORMAT:\n"
            "ANSWER: <short answer>\n"
            "EVIDENCE: <comma-separated message ids, or none>\n"
            "PEER_CLAIM: accepted|rejected|not_applicable\n"
            "REASON: <one sentence>"
        )
    elif final:
        text += "\n\nOUTPUT FORMAT:\nFINAL_ANSWER: <short answer>\nREASON: <one sentence>"
    else:
        text += "\n\nBe concise and include message IDs if they are useful."
    return text

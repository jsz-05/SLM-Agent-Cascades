"""Prompt builders for the three minimal orchestration topologies."""

from __future__ import annotations

from typing import Any


Task = dict[str, Any]


def build_solo_messages(task: Task, condition: str) -> list[dict[str, str]]:
    system = {
        "plain": "You are a concise assistant answering from the provided message stream.",
        "role": (
            "You are a solo persistent-agent evaluator. Track updates in the stream, "
            "compare them with the previous agent claim, and answer carefully."
        ),
        "evidence_gated": (
            "You are an evidence-gated assistant. Accept claims only when they are "
            "supported by cited message IDs from the original stream."
        ),
    }[condition]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _base_task_text(task, condition, final=True)},
    ]


def build_sequential_messages(
    task: Task,
    condition: str,
    step: str,
    prior_outputs: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    prior_outputs = prior_outputs or {}

    if step == "extract":
        if condition == "role":
            system = "You are Agent 1, the evidence extractor."
            request = "Extract the facts relevant to the question and previous agent claim."
        elif condition == "evidence_gated":
            system = "You are Agent 1, the evidence extractor. Cite message IDs for every accepted fact."
            request = "Extract relevant evidence. Do not accept unsupported claims."
        else:
            system = "You identify information relevant to the question."
            request = "List the relevant information for answering the question."
        content = _base_task_text(task, condition, final=False) + f"\n\nTASK:\n{request}"
    elif step == "reason":
        if condition == "role":
            system = "You are Agent 2, the reasoner."
            request = "Reason over Agent 1's extraction and the original stream."
        elif condition == "evidence_gated":
            system = "You are Agent 2, the evidence-gated reasoner. Cite message IDs."
            request = "Reason only from cited stream evidence and flag unsupported peer claims."
        else:
            system = "You reason over extracted information and the original stream."
            request = "Use the prior output and original stream to decide the likely answer."
        content = (
            _base_task_text(task, condition, final=False)
            + f"\n\nUPSTREAM OUTPUT FROM STEP 1:\n{prior_outputs.get('extract', '')}"
            + f"\n\nTASK:\n{request}"
        )
    elif step == "verify":
        if condition == "role":
            system = "You are Agent 3, the verifier and final answerer."
            request = (
                "Verify the reasoning against the original stream and provide the final answer."
            )
        elif condition == "evidence_gated":
            system = (
                "You are Agent 3, the evidence-gated verifier. Compare the proposed answer, "
                "the original stream, and the previous agent claim."
            )
            request = (
                "Produce the final answer. Accept or reject the peer claim using cited evidence."
            )
        else:
            system = "You check prior reasoning against the original stream and answer."
            request = "Provide the final answer after checking the prior steps."
        content = (
            _base_task_text(task, condition, final=True)
            + f"\n\nUPSTREAM OUTPUT FROM STEP 1:\n{prior_outputs.get('extract', '')}"
            + f"\n\nUPSTREAM OUTPUT FROM STEP 2:\n{prior_outputs.get('reason', '')}"
            + f"\n\nTASK:\n{request}"
        )
    else:
        raise ValueError(f"Unknown sequential step: {step}")

    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def build_committee_member_messages(task: Task, condition: str, member_index: int) -> list[dict[str, str]]:
    if condition == "role":
        system = (
            f"You are independent committee agent {member_index}. Answer independently "
            "before any judge sees committee outputs."
        )
    elif condition == "evidence_gated":
        system = (
            f"You are independent evidence-gated committee agent {member_index}. "
            "Cite message IDs and do not defer to unsupported peer claims."
        )
    else:
        system = f"You are independent committee agent {member_index}."

    content = _base_task_text(task, condition, final=True)
    content += "\n\nIMPORTANT: Do not assume other committee agents agree with you."
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def build_committee_judge_messages(
    task: Task,
    condition: str,
    member_outputs: dict[str, str],
) -> list[dict[str, str]]:
    if condition == "role":
        system = "You are the committee judge. Compare independent answers with the original stream."
    elif condition == "evidence_gated":
        system = (
            "You are the evidence-gated committee judge. Use the original stream as authority, "
            "cite message IDs, and accept or reject the peer claim."
        )
    else:
        system = "You judge independent answers against the original stream."

    answers = "\n".join(f"{key}: {value}" for key, value in sorted(member_outputs.items()))
    content = (
        _base_task_text(task, condition, final=True)
        + f"\n\nINDEPENDENT COMMITTEE ANSWERS:\n{answers}"
        + "\n\nTASK:\nProduce the final judged answer from the original stream."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def _base_task_text(task: Task, condition: str, final: bool) -> str:
    text = (
        f"MESSAGE STREAM:\n{format_stream(task)}\n\n"
        f"QUESTION: {task['question']}\n\n"
        f"PREVIOUS AGENT CLAIM: {task.get('wrong_peer_claim') or 'not provided'}\n\n"
        "The previous agent claim may be wrong. The original message stream is authoritative."
    )
    if condition == "evidence_gated":
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


def format_stream(task: Task) -> str:
    return "\n".join(f"{message['id']}: {message['text']}" for message in task["stream"])

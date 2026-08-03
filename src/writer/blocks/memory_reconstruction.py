"""Reconstructive memory for chat conversations.

Most memory-augmented agents inject retrieved experience verbatim, replaying
past records into the context whether or not they fit the turn the agent is
about to handle. That "replay" ignores the gap between the abstract, general
nature of stored experience and the concrete, ever-changing situation at
decision time, and frequently causes negative transfer.

This module instead asks the model to *critique and reconstruct* retrieved
memory conditioned on the current conversation state, producing
context-grounded guidance that the agent acts on rather than the raw records.

Adapted from: "MemHarness: Memory Is Reconstructed, Not Replayed"
(https://arxiv.org/abs/2607.28272v1).

This is a Mode-2 port. The paper's core reconstructive step -- critique and
reconstruct retrieved experience against the current state before acting -- is
kept at fidelity. The auxiliary components are substituted: the GRPO-learned
unified policy model is replaced with a prompted reconstruction call (a
parameter-free proxy that approximates the same critique-and-reframe signal),
and the ALFWorld / WebShop benchmark and OOD evaluation harness are
intentionally out of scope (evaluation belongs in a downstream PR).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from writer.ai import Conversation


RECONSTRUCTION_PROMPT = (
    "You reconstruct an agent's retrieved memory for the turn it is about to "
    "handle. You are given the agent's current situation and the raw retrieved "
    "memories. Do NOT echo the memories back. Critique them against the current "
    "situation and produce concise, context-grounded guidance: keep only what is "
    "relevant to this turn, reframe abstract notes into concrete instructions for "
    "the present situation, and explicitly discard anything that would mislead the "
    "agent here. If nothing in the memory applies, reply with a single line saying "
    "so. Output only the reconstructed guidance."
)


def reconstruct_memory(
    retrieved_memories: Any,
    conversation: "Conversation",
    model_id: Optional[str] = None,
) -> str:
    """Reconstruct retrieved memories into context-grounded guidance.

    The retrieved experience is critiqued and reframed against the conversation's
    current situation, rather than replayed verbatim. Returns the reconstructed
    guidance string, or an empty string when there is nothing to reconstruct (no
    memory supplied, or no current situation to condition on) -- in which case the
    caller injects nothing.
    """
    memory_blob = _serialize_memory(retrieved_memories)
    if not memory_blob:
        return ""
    current_context = _current_context(conversation)
    if not current_context:
        return ""
    prompt = _build_prompt(memory_blob, current_context)
    guidance = _invoke_model(prompt, model_id=model_id)
    if isinstance(guidance, dict):
        guidance = guidance.get("content", "")
    return (guidance or "").strip()


def _serialize_memory(retrieved_memories: Any) -> str:
    """Flatten whatever the caller retrieved into a single text blob."""
    if retrieved_memories is None:
        return ""
    if isinstance(retrieved_memories, str):
        return retrieved_memories.strip()
    if isinstance(retrieved_memories, (list, tuple)):
        parts = [_serialize_memory(item) for item in retrieved_memories]
        return "\n".join(part for part in parts if part)
    if isinstance(retrieved_memories, dict):
        try:
            return json.dumps(retrieved_memories)
        except (TypeError, ValueError):
            return str(retrieved_memories)
    return str(retrieved_memories).strip()


def _current_context(conversation: "Conversation") -> str:
    """The situation the agent is about to respond to.

    Prefers the most recent user message (the live request); falls back to the
    most recent message of any role so reconstruction is always conditioned on
    concrete state rather than nothing.
    """
    messages = list(getattr(conversation, "messages", []) or [])
    for role in ("user", "assistant"):
        for message in reversed(messages):
            if message.get("role") != role:
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def _build_prompt(memory_blob: str, current_context: str) -> str:
    return (
        "Current situation the agent is responding to:\n"
        f"{current_context}\n\n"
        "Retrieved memories (raw past experience):\n"
        f"{memory_blob}\n\n"
        "Reconstruct these memories into concise, context-grounded guidance for "
        "the current situation. Do not replay them verbatim."
    )


def _invoke_model(prompt: str, model_id: Optional[str] = None) -> str:
    """Run the reconstruction model call.

    Kept as its own function so the network boundary is a single, mockable seam
    (the rest of the module is pure shaping around it).
    """
    import writer.ai

    config = {"temperature": 0.2, "max_tokens": 512}
    if model_id:
        config["model"] = model_id
    reconstruction = writer.ai.Conversation(prompt_or_history=RECONSTRUCTION_PROMPT, config=config)
    reconstruction.add("user", prompt)
    reply = reconstruction.complete()
    content = reply.get("content") if isinstance(reply, dict) else reply
    return content or ""

"""Selective persistent memory for agentic workflows.

Distills a multi-turn conversation into the four categories of *reusable*
context identified by "Shared Selective Persistent Memory for Agentic LLM
Systems" (arXiv:2607.09493):

  * task specifications
  * data schemas
  * tool configurations
  * output constraints

Session-specific reasoning traces are discarded, so the compact blob that
remains is cheap to persist (via the Key-Value Storage block) and cheap to
re-inject at the start of a later session -- instead of naively replaying
the full conversation history, which the paper shows both wastes tokens and
degrades generation quality.

This module implements the paper's core *selective extraction* mechanism at
full fidelity using the framework's hosted Writer model. Adapted (out of
scope here): the paper's collaborative-workspace sharing, role-based access
control, and zero-token data-refresh machinery -- those are platform
infrastructure the framework does not host, and their absence does not
collapse the selective-extraction core.
"""

import json
from typing import Any, Dict, List, Optional

DEFAULT_MODEL = "palmyra-x5"

# The four reusable-context categories the paper retains; everything else
# (reasoning traces, turn-by-turn narration) is dropped.
MEMORY_CATEGORIES = (
    "task_specifications",
    "data_schemas",
    "tool_configurations",
    "output_constraints",
)

_EXTRACTION_PROMPT = """\
You are extracting selective persistent memory from an agentic chat session.

Read the conversation below and pull out ONLY reusable context that a future \
session would need to reproduce this work. Retain facts in these four \
categories and DROP everything else (reasoning traces, step-by-step \
narration, greetings, and one-off dialogue):

- task_specifications: what the agent was asked to build or do.
- data_schemas: fields, types, and shapes of inputs and outputs.
- tool_configurations: endpoints, parameters, auth notes, MCP/tool setup.
- output_constraints: formatting, length, audience, and quality rules.

Respond with a single JSON object and nothing else, shaped like:
{{"task_specifications": ["..."], "data_schemas": ["..."], \
"tool_configurations": ["..."], "output_constraints": ["..."]}}
Use an empty list when a category has nothing reusable.

Conversation:
{transcript}\
"""


def extract_selective_memory(
    messages: List[Dict[str, Any]], model_id: Optional[str] = None
) -> Dict[str, List[str]]:
    """Distill a conversation into the four reusable-context categories.

    ``messages`` follows the usual ``{"role", "content"}`` chat shape. The
    model is asked to keep only reusable context and drop reasoning traces,
    so the returned blob is a compact summary suitable for persisting via
    the Key-Value Storage block and re-injecting at the start of a later
    session.
    """
    import writer.ai

    if not messages:
        return {category: [] for category in MEMORY_CATEGORIES}

    transcript = _render_transcript(messages)
    prompt = _EXTRACTION_PROMPT.format(transcript=transcript)
    config = {
        "model": model_id or DEFAULT_MODEL,
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    raw = writer.ai.complete(prompt, config)
    return _parse_memory_json(raw)


def format_memory_for_context(blob: Dict[str, List[str]]) -> str:
    """Render a memory blob as compact text for re-injection into a session."""
    if not isinstance(blob, dict):
        return ""
    sections = []
    for category in MEMORY_CATEGORIES:
        items = blob.get(category, [])
        if not items:
            continue
        label = category.replace("_", " ").title()
        bullets = "\n".join(f"- {item}" for item in items)
        sections.append(f"{label}:\n{bullets}")
    return "\n\n".join(sections)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for comparing memory footprints.

    A heuristic stand-in for a real tokenizer; useful for observing that
    selective memory is dramatically smaller than the raw transcript it
    replaces, without pulling in a benchmark harness.
    """
    return max(0, len(text or "") // 4)


def _render_transcript(messages: List[Dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role", "unknown")
        content = message.get("content", "")
        # content may be multimodal (a list of parts); coerce to plain text.
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(part.get("text", ""))
                else:
                    parts.append(str(part))
            content = " ".join(parts)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _parse_memory_json(text: str) -> Dict[str, List[str]]:
    """Tolerantly parse the model's JSON response into the four categories."""
    payload = _extract_first_json_object(text)
    try:
        data = json.loads(payload) if payload else {}
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}

    memory: Dict[str, List[str]] = {}
    for category in MEMORY_CATEGORIES:
        value = data.get(category, [])
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            value = []
        memory[category] = [str(item) for item in value]
    return memory


def _extract_first_json_object(text: str) -> str:
    """Return the outermost JSON object, tolerating fences/prose around it."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return ""
    return text[start : end + 1]

"""Tool gating — intent/schema-overlap selection of MCP tools.

The "Chat reply with tool config" block builds a ``FunctionTool`` for every
configured MCP tool and injects the full list into each completion call.
In multi-server deployments this eagerly serialized schema payload is the
per-turn "Tools Tax" the following paper targets:

    "Tool Attention Is All You Need: Dynamic Tool Gating and Lazy Schema
    Loading for Eliminating the MCP/Tools Tax in Scalable Agentic Workflows"
    (arXiv:2604.21816)

This module ports the paper's intent-schema-overlap *gating* step: before the
completion call, score each tool by lexical overlap between the turn's intent
(the latest user message) and the tool's schema (name, description, parameter
names/descriptions), then keep only the top-``maxTools`` tools.

Adaptation notes (Mode 2 — substituted auxiliaries):

  * The paper's **two-phase lazy schema loader** controls how tool schemas are
    serialized into the request payload. That is an SDK-level concern and has
    no client-side call site here — the block only hands a ``tools=[...]``
    list to the Writer SDK. The contribution with a real anchor is the
    gating/selection step, which is what determines *which* schemas are
    injected per turn. The lazy loader is therefore out of scope for this port.
  * The paper's **learned reranker** is replaced by a parameter-free
    lexical-overlap score, which is the paper's own parameter-free baseline
    form of intent-schema overlap (no training data or estimator required).

Gating is opt-in via ``toolConfig.toolGating`` and is a no-op when disabled or
when there is no positive overlap signal, so the existing
``tools=[FunctionTool...]`` contract is preserved.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

# Short grammar/function words excluded from the intent and schema token bags
# so overlap is driven by content words (e.g. "search", "knowledge", "email")
# rather than glue like "the", "a", "for".
_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at for with without by from
    into onto over under again more most some any each every all no not nor
    is are was were be been being this that these those it its as about into
    can could should would may might must will do does did done have has had
    i you he she we they me him her us them my your his our their
    please need want using use get make new
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _flatten_content(content: Any) -> str:
    """Conversation message content may be a plain string or a list of
    content fragments (text/image_url). Return the concatenated text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for fragment in content:
            if isinstance(fragment, dict):
                parts.append(fragment.get("text") or "")
            elif isinstance(fragment, str):
                parts.append(fragment)
        return " ".join(parts)
    return str(content)


def latest_user_intent(messages: List[Dict[str, Any]]) -> str:
    """Return the text of the most recent user message in a conversation.

    This is the turn "intent" that tool schemas are gated against. Returns an
    empty string when the conversation carries no user message.
    """
    for message in reversed(messages or []):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            return _flatten_content(message.get("content"))
    return ""


def _schema_token_bag(tool: Dict[str, Any]) -> set[str]:
    """Token bag describing a tool's schema: name, description, and the
    names/descriptions of its parameters."""
    parts: List[str] = [str(tool.get("name") or "")]
    description = tool.get("description")
    if description:
        parts.append(str(description))
    parameters = tool.get("parameters")
    if isinstance(parameters, dict):
        for param_name, param_meta in parameters.items():
            parts.append(str(param_name))
            if isinstance(param_meta, dict):
                meta_desc = param_meta.get("description")
                if meta_desc:
                    parts.append(str(meta_desc))
    return _tokenize(" ".join(parts))


def _gating_config(tool_config: Any) -> Dict[str, Any]:
    """Read the ``toolGating`` sub-config from a block's toolConfig."""
    if not isinstance(tool_config, dict):
        return {}
    gating = tool_config.get("toolGating")
    return gating if isinstance(gating, dict) else {}


def select_relevant_tools(tools: List[Any], intent: str, tool_config: Any) -> List[Any]:
    """Return the subset of ``tools`` relevant to ``intent``.

    Each tool is scored by the size of the overlap between the intent token
    bag and the tool's schema token bag; the top ``maxTools`` are kept, ties
    broken by original order (stable).

    Gating is opt-in and safe by construction:

      * When ``tool_config.toolGating.enabled`` is not true, ``tools`` is
        returned unchanged (the existing contract holds).
      * When no tool has a positive overlap with the intent, ``tools`` is
        returned unchanged — gating only narrows the set when there is a real
        relevance signal, never truncating to an arbitrary prefix.
      * The result is never empty when ``tools`` is non-empty: the highest-
        scoring tool is always retained.
    """
    if not tools:
        return tools

    config = _gating_config(tool_config)
    if not config.get("enabled"):
        return tools

    max_tools = config.get("maxTools")
    # Reject booleans (a bool is an int subclass) and non-ints; a missing or
    # invalid cap means there is nothing to gate against.
    if isinstance(max_tools, bool) or not isinstance(max_tools, int):
        return tools
    if max_tools <= 0:
        return tools

    intent_tokens = _tokenize(intent or "")
    if not intent_tokens:
        return tools

    scored = [
        (len(intent_tokens & _schema_token_bag(tool)), index, tool)
        for index, tool in enumerate(tools)
    ]
    if not any(score for score, _, _ in scored):
        return tools

    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    return [tool for _, _, tool in scored[:max_tools]]

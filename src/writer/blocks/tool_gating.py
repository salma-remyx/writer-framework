"""Dynamic tool gating for the tool-calling block.

Adapted from "Tool Attention Is All You Need: Dynamic Tool Gating and
Lazy Schema Loading for Eliminating the MCP/Tools Tax in Scalable
Agentic Workflows" (arXiv:2604.21816v1).

The paper's drop-in gating middleware scores each tool's schema against the
current turn's intent and injects only the top-k most relevant tool schemas,
eliminating the per-turn "Tools Tax" of eagerly serializing *every* tool
schema on every turn. Its core mechanism is an intent-schema-overlap score
combined with two-phase lazy schema loading.

This module ports that **core mechanism at full fidelity** while making two
target-native substitutions (Mode 2 adapted port):

* The paper's learned/semantic overlap estimator is replaced by a
  parameter-free token-overlap proxy (Jaccard over intent vs. schema tokens,
  with a boost when the intent matches the tool's name). No model and no
  extra round-trip are required.
* The paper's separate "lazy schema loader" phase collapses into the natural
  target behaviour: the tool-calling block simply passes the gated subset to
  ``conversation.complete(tools=...)`` instead of the full list. Schemas the
  model never sees are never serialized -- that *is* the lazy loading.

The paper's standalone benchmark / evaluation framework is intentionally out
of scope for this integration.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Sequence, Set

# Tools that drive the agent loop itself and must never be gated out, even
# when their schema text has no lexical overlap with the current intent.
# The ReAct finalization tool ("disclose_reasoning") lives here: gating it
# out would strand the loop with no way to signal completion.
CONTROL_TOOL_NAMES = frozenset({"disclose_reasoning"})

_TOKEN_SPLIT = re.compile(r"[A-Za-z0-9_]+")

# Common English stopwords plus agentic-prompt filler words. Removing them
# keeps the overlap score focused on content-bearing tokens (tool nouns,
# parameter names, entities) rather than prompt scaffolding.
_STOPWORDS = frozenset(
    "a an the and or but if then else for of to in on at by with as is are be "
    "this that it its i you we they me my your our their please use using used "
    "want need help task tool call function return set get do now will can "
    "should would into".split()
)


def _tokenize(text: str) -> Set[str]:
    tokens = (
        tok.lower()
        for tok in _TOKEN_SPLIT.findall(text or "")
        if len(tok) > 1 and tok.lower() not in _STOPWORDS
    )
    return set(tokens)


def _tool_attr(tool: Any, *names: str) -> str:
    """Read the first present attribute/key from a tool (dict or object)."""
    for name in names:
        if isinstance(tool, dict):
            value = tool.get(name)
        else:
            value = getattr(tool, name, None)
        if value:
            return str(value)
    return ""


def _tool_name(tool: Any) -> str:
    return _tool_attr(tool, "name")


def _tool_description(tool: Any) -> str:
    return _tool_attr(tool, "description")


def _tool_parameter_text(tool: Any) -> str:
    """Flatten a tool's parameter names + descriptions into scoring text."""
    params = tool.get("parameters") if isinstance(tool, dict) else getattr(tool, "parameters", None)
    if not isinstance(params, dict):
        return ""
    parts: List[str] = []
    for name, meta in params.items():
        parts.append(str(name))
        if isinstance(meta, dict):
            parts.append(str(meta.get("description") or ""))
            parts.append(str(meta.get("type") or ""))
    return " ".join(parts)


def _schema_tokens(tool: Any) -> Set[str]:
    """Content-bearing tokens drawn from a tool's full schema."""
    combined = " ".join([_tool_name(tool), _tool_description(tool), _tool_parameter_text(tool)])
    return _tokenize(combined)


def score_tool(intent_tokens: Set[str], tool: Any) -> float:
    """Intent-schema-overlap score for one tool.

    Pure token-overlap (Jaccard) over intent vs. schema tokens, plus a unit
    boost when the intent shares a token with the tool's name. This is the
    parameter-free proxy for the paper's learned overlap estimator.
    """
    schema_tokens = _schema_tokens(tool)
    if not intent_tokens or not schema_tokens:
        return 0.0
    overlap = len(intent_tokens & schema_tokens)
    union = len(intent_tokens | schema_tokens)
    jaccard = overlap / union if union else 0.0
    name_tokens = _tokenize(_tool_name(tool))
    name_boost = 1.0 if (intent_tokens & name_tokens) else 0.0
    return jaccard + name_boost


def select_tools(
    intent: str,
    tools: Sequence[Any],
    max_tools: int,
    always_include: Iterable[str] = CONTROL_TOOL_NAMES,
    min_score: float = 0.0,
) -> List[Any]:
    """Return the top-``max_tools`` tools most relevant to ``intent``.

    Tools whose name is in ``always_include`` (loop-control tools such as the
    ReAct finalization tool) are pinned regardless of score, so gating never
    strands the agent loop. Selected tools keep their original relative order
    for deterministic serialization.

    When ``max_tools`` is non-positive, or the intent carries no
    content-bearing tokens, every tool is returned (gating degrades to a
    no-op rather than dropping tools it cannot reason about).
    """
    if max_tools <= 0 or not tools:
        return list(tools)

    intent_tokens = _tokenize(intent)
    if not intent_tokens:
        return list(tools)

    pinned_names = set(always_include)
    pinned_idx: Set[int] = set()
    scored: List[tuple] = []  # (score, original_index)
    for idx, tool in enumerate(tools):
        if _tool_name(tool) in pinned_names:
            pinned_idx.add(idx)
            continue
        s = score_tool(intent_tokens, tool)
        if s > min_score:
            scored.append((s, idx))

    budget = max(0, max_tools - len(pinned_idx))
    # Stable sort by score desc; ties preserve original order.
    scored.sort(key=lambda pair: pair[0], reverse=True)
    chosen_idx = {idx for _, idx in scored[:budget]} | pinned_idx
    return [tools[i] for i in sorted(chosen_idx)]

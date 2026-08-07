"""Dynamic tool gating for the Tool calling block.

Adapted from "Tool Attention Is All You Need: Dynamic Tool Gating and
Lazy Schema Loading for Eliminating the MCP/Tools Tax in Scalable
Agentic Workflows" (arXiv:2604.21816).

The paper's core mechanism is *dynamic tool gating*: instead of eagerly
injecting the full set of tool schemas into every model turn (the
per-turn "tools tax" the paper attributes to stateless MCP schema
injection), select only the query-relevant subset of tools and inject
just those schemas. The paper's companion idea -- lazy schema loading --
falls out for free, because only the active subset is serialised into
the completion request.

This module ports that core mechanism at full fidelity and substitutes
the paper's *learned* tool-attention selector with a parameter-free
relevance proxy: token overlap between the query and each tool's
name + description + parameter text. That keeps the contract
(`query -> active tool subset`) identical while avoiding a trained
selection model the framework does not host.

Attribution: the gating contract and the lazy-injection result are from
the paper; the overlap scorer is a target-native proxy. Scoped out (not
needed to deliver the per-turn tax reduction): per-turn re-gating from
evolving conversation context -- the block reuses one tools list across
iterations, so a single gating pass already shrinks every turn.
"""

import re
from typing import Dict, List, Optional, Set

# Tools whose presence is structural rather than query-relevant. They are
# always retained so the ReAct finalization protocol stays intact.
_ALWAYS_KEEP = {"disclose_reasoning"}

# Tokens that carry no signal for tool-relevance matching.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at for with without by
    from into about over under as is are was were be been being this that
    these those it its it's i you he she they we me my your our their them
    his her ours theirs what which who whom whose why how when where
    can could should would may might must shall do does did done have has
    had having will want wants need needs use uses using get gets got
    please give make list show find provide return call invoke
    """.split()
)

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def _tokenize(text: Optional[str]) -> Set[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords and 1-char noise."""
    if not text:
        return set()
    tokens: Set[str] = set()
    for raw in _TOKEN_SPLIT.split(text.lower()):
        token = raw.strip()
        if len(token) < 2 or token in _STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _tool_name(tool: Dict) -> Optional[str]:
    name = tool.get("name")
    return name if isinstance(name, str) else None


def _tool_text(tool: Dict) -> str:
    """Concatenate the text-bearing fields of a tool for overlap matching."""
    parts: List[str] = []
    for value in (tool.get("name"), tool.get("description")):
        if isinstance(value, str):
            parts.append(value)
    params = tool.get("parameters")
    if isinstance(params, dict):
        for key, meta in params.items():
            if isinstance(key, str):
                parts.append(key)
            if isinstance(meta, dict):
                desc = meta.get("description")
                if isinstance(desc, str):
                    parts.append(desc)
    return " ".join(parts)


def _overlap(query_tokens: Set[str], tool: Dict) -> int:
    """Number of shared content tokens between the query and a tool."""
    return len(query_tokens & _tokenize(_tool_text(tool)))


def gate_tools(
    query: str,
    tools: List[Dict],
    max_tools: Optional[int] = None,
) -> List[Dict]:
    """Return the query-relevant subset of ``tools``.

    Implements the paper's ``query -> active_tool_subset`` contract:

    * Control tools (``disclose_reasoning``) are always retained.
    * Remaining tools are scored by token overlap with ``query`` and kept
      when they share at least one content token.
    * If no tool matches, all non-control tools are returned -- gating
      never silently strips the agent of every option (fail-open).
    * ``max_tools`` (when > 0) caps the number of non-control tools,
      keeping the highest-scoring ones; ties break by original order.

    Only the returned subset should be injected into the model turn, which
    is the lazy-schema-loading behaviour that cuts the per-turn tools tax.
    """
    query_tokens = _tokenize(query)

    scored: List[tuple] = []  # (score, original_index, tool)
    control: List[Dict] = []
    for idx, tool in enumerate(tools):
        if _tool_name(tool) in _ALWAYS_KEEP:
            control.append(tool)
            continue
        scored.append((_overlap(query_tokens, tool), idx, tool))

    matched = [item for item in scored if item[0] > 0]
    if not matched:
        # Fail open: never leave the agent with only control tools.
        matched = scored

    if max_tools and max_tools > 0:
        kept = sorted(matched, key=lambda item: (-item[0], item[1]))[:max_tools]
        kept_indices = {item[1] for item in kept}
        matched = [item for item in matched if item[1] in kept_indices]

    # Emit kept tools in their original order, control tools last.
    chosen = [tool for _, _, tool in sorted(matched, key=lambda item: item[1])]
    return chosen + control

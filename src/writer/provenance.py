"""
Blueprint-run provenance graphs.

Adapted from *AgentTrails: Towards Trust and Reuse for Agentic Tasks*
(arXiv:2607.18816). AgentTrails converts agent trajectories from flat
chronological logs into structured *provenance graphs* in which tool calls are
modelled as computational actions and their inputs and outputs as data
artifacts, joined by producer/consumer dependency edges that expose the run's
dataflow -- something a chronological log obscures.

Writer Framework's :class:`~writer.journal.JournalRecord` already records every
block execution against the blueprint graph, but stores it as a flat per-node
list (``blockOutputs``) with no producer/consumer edges -- exactly the
"chronological log obscuring dataflow" shape the paper targets. This module
adds the paper's provenance layer on top of a single run: it derives an
additive ``actions`` / ``artifacts`` / ``edges`` view from the blueprint graph's
``toNodeId`` control edges and the captured block outputs, without touching the
existing journal entry.

Scope note: this is the single-trajectory provenance conversion -- the paper's
*core* representation. The paper's multi-trajectory comparison (placing several
provenance graphs on a shared canvas and joining them into a quotient graph),
pattern extraction and skill abstraction are auxiliary sensemaking layers that
sit on top of this representation and are intentionally left for a follow-up.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    from writer.journal import JournalRecord

# Cap on the number of characters retained from an artifact's value when
# building a human-readable preview. Full values already live in the journal's
# ``blockOutputs``; provenance only needs enough of the value to recognise it.
_MAX_PREVIEW_CHARS = 200


def build_provenance_graph(record: "JournalRecord") -> Dict[str, Any]:
    """Build a structured provenance graph for a single blueprint run.

    The run recorded by ``record`` is re-represented as dataflow rather than a
    chronological log:

    * ``actions`` -- one node per block (the paper's "computational actions"),
      carrying its tool type, label, category and captured outcome.
    * ``artifacts`` -- one node per block that produced a value (the paper's
      "data artifacts"), carrying a small, JSON-safe preview of that value.
    * ``edges`` -- producer/consumer edges derived from the blueprint graph's
      ``toNodeId`` control edges, each tying a producer's artifact to the action
      that consumes it, labelled with the branch (``outId``) it flowed along.

    The view is additive: it only reads the live graph and the record's block
    metadata and never mutates the journal entry it describes.
    """
    graph = record.graph
    component_meta = {
        node_id: entry.get("component", {}) for node_id, entry in record.block_outputs.items()
    }

    actions: List[Dict[str, Any]] = []
    artifacts: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    for node in graph.nodes:
        meta = component_meta.get(node.id, {})

        actions.append(
            {
                "id": node.id,
                "tool": meta.get("type") or node.component.type,
                "label": meta.get("title"),
                "category": meta.get("category"),
                "outcome": node.outcome,
            }
        )

        kind, preview = _summarize_value(node.result)
        if kind != "none":
            artifacts.append(
                {
                    "id": node.id,
                    "producedBy": node.id,
                    "kind": kind,
                    "preview": preview,
                }
            )

        # Producer/consumer edges: the blueprint's control edges tell us which
        # action's artifact feeds which downstream action. Emitting all in-graph
        # control edges surfaces the full dependency structure (including edges
        # that did not fire during this run -- their producer's outcome tells
        # the reader whether data actually flowed).
        for output in node.outputs:
            consumer_id = output.get("toNodeId")
            if consumer_id is not None and graph.get_node(consumer_id) is not None:
                edges.append(
                    {
                        "artifact": node.id,
                        "producer": node.id,
                        "consumer": consumer_id,
                        "outId": output.get("outId"),
                    }
                )

    return {
        "actions": actions,
        "artifacts": artifacts,
        "edges": edges,
        "actionCount": len(actions),
        "artifactCount": len(artifacts),
        "edgeCount": len(edges),
    }


def _summarize_value(value: Any) -> Tuple[str, str]:
    """Return a ``(kind, preview)`` tuple describing an artifact value.

    ``kind`` is a stable, JSON-safe category; ``preview`` is a short, truncated
    string. Exotic or non-serialisable values degrade gracefully to a type label
    rather than risking the enclosing journal entry becoming un-displayable.
    """
    if value is None:
        return "none", ""
    if isinstance(value, bool):
        return "boolean", str(value)
    if isinstance(value, (int, float)):
        return "number", str(value)
    if isinstance(value, str):
        return "string", value[:_MAX_PREVIEW_CHARS]
    if isinstance(value, list):
        return "list", _truncate(f"{len(value)} item(s)")
    if isinstance(value, dict):
        keys = ", ".join(str(key) for key in value.keys())
        return "object", _truncate(keys)
    return type(value).__name__, _truncate(repr(value))


def _truncate(text: str) -> str:
    return text[:_MAX_PREVIEW_CHARS]

"""Provenance graph builder for Journal execution records.

A Journal entry is normally a flat, per-node ``blockOutputs`` map: a
chronological list of what each block did, with no links between blocks.
That obscures the *dataflow* -- which block fed which other block, and on
which output port.

This module converts a :class:`~writer.journal.JournalRecord` into a
structured provenance / dataflow graph, so the journal captures the
dependencies between actions rather than only their order:

* each executed block becomes an **action node** (identity, outcome, and a
  summary of the data artifact it produced);
* each producer->consumer wiring becomes a labelled **dependency edge**
  carrying the output port (``outId``) the consumer branched on.

Adapted from the core mechanism of "AgentTrails: Towards Trust and Reuse
for Agentic Tasks" (arXiv:2607.18816), which turns raw agent trajectories
into provenance graphs by modelling tool calls as computational actions
and their inputs/outputs as data artifacts.

Scope note (Mode 2 adapted port): the cross-trajectory *quotient graph* /
shared-canvas comparison and the *skill abstraction* / pattern-extraction
analyses from that paper are auxiliary components that consume many
provenance graphs at once. They are intentionally left out here and belong
in downstream analysis tooling; this module delivers the per-trajectory
provenance graph that feeds them. Each node carries a stable ``signature``
(type:category) precisely so that future quotient-graph alignment of
recurring tools across runs has the identity label it needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from writer.journal import JournalRecord


def node_signature(component_type: str, category: str) -> str:
    """Stable identity label for an action node.

    Two nodes that share a signature are "the same tool" run at different
    points in the graph. This is the handle a future quotient-graph pass
    would use to align recurring tools across executions.
    """
    return f"{component_type}:{category}"


def summarize_artifact(result: Any) -> Dict[str, Any]:
    """Reduce a block's (possibly large / non-serializable) result to a stub.

    The provenance graph records *that* an artifact exists and what shape it
    has, not its full value -- the value already lives in ``blockOutputs``.
    Keeping only type and size keeps the graph small and always
    JSON-serializable.
    """
    if result is None:
        return {"present": False}
    size: Any = None
    if isinstance(result, (str, list, tuple, dict, set)):
        size = len(result)
    return {"present": True, "type": type(result).__name__, "size": size}


def build_provenance_graph(record: "JournalRecord") -> Dict[str, Any]:
    """Build a provenance / dataflow graph for a journal record.

    Reads component identity from ``record.block_outputs`` (already prepared
    by :class:`JournalRecord`) and dependency edges plus live outcome/result
    from ``record.graph`` (the executed :class:`~writer.blueprints.Graph`).

    Returns a dict with:

    * ``nodes`` -- one action node per executed block: ``id``, ``type``,
      ``title``, ``category``, ``signature``, ``outcome``, and ``artifact``
      (the summarized result it produced).
    * ``edges`` -- one dependency edge per producer->consumer link, labelled
      with the ``port`` (``outId``) the consumer branched on. Duplicate
      ``(from, to, port)`` triples are collapsed.
    * ``stats`` -- counts (``nodeCount``, ``edgeCount``, ``sourceCount``,
      ``sinkCount``) and an ``outcomes`` histogram, for quick sensemaking
      (e.g. spotting a sink node that ended in ``error``).
    """
    graph = record.graph
    component_by_node = {
        node_id: meta.get("component", {}) for node_id, meta in record.block_outputs.items()
    }

    nodes: List[Dict[str, Any]] = []
    for graph_node in graph.nodes:
        component = component_by_node.get(graph_node.id, {})
        component_type = component.get("type", graph_node.component.type)
        category = component.get("category", "Unknown category")
        nodes.append(
            {
                "id": graph_node.id,
                "type": component_type,
                "title": component.get("title", "Unknown block"),
                "category": category,
                "signature": node_signature(component_type, category),
                "outcome": graph_node.outcome,
                "artifact": summarize_artifact(graph_node.result),
            }
        )

    edges: List[Dict[str, Any]] = []
    seen_edges: set = set()
    for graph_node in graph.nodes:
        for edge in graph_node.inputs:
            from_id = edge.get("fromNodeId")
            port = edge.get("outId")
            key = (from_id, graph_node.id, port)
            if from_id is None or key in seen_edges:
                continue
            seen_edges.add(key)
            edges.append({"from": from_id, "to": graph_node.id, "port": port})

    consumers = {edge["to"] for edge in edges}
    producers = {edge["from"] for edge in edges}
    outcomes: Dict[str, int] = {}
    for node in nodes:
        outcome = node["outcome"] or "not_run"
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "nodeCount": len(nodes),
            "edgeCount": len(edges),
            "sourceCount": sum(1 for n in nodes if n["id"] not in producers),
            "sinkCount": sum(1 for n in nodes if n["id"] not in consumers),
            "outcomes": outcomes,
        },
    }

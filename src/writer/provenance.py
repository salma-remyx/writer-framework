"""Build structured provenance graphs from blueprint execution trajectories.

Adapted from the core mechanism of:

    AgentTrails: Towards Trust and Reuse for Agentic Tasks
    (arxiv:2607.18816)

The Journal records a blueprint run as a flat, chronological map of
``component -> executions``. That representation mirrors the order in which
blocks ran, but it obscures the *dataflow*: which tool call produced the
artifact a later tool call consumed, which branches were actually taken,
and where a run diverged. AgentTrails' core idea is to convert such a
trajectory into a structured provenance graph where tool calls are
*actions*, their inputs and outputs are *data artifacts*, and the two are
joined by explicit ``produces`` / ``consumed_by`` dependency edges.

This module ports exactly that construction at full fidelity:

* an **action** per executed block (tool call) -- its type, title,
  outcome and how many times it ran;
* a **data artifact** per (producer, branch) pair -- the output flowing
  out of a block on a given outcome branch, with the downstream blocks
  that consume it;
* dependency **edges** that connect producer -> artifact -> consumers,
  plus the realized/unrealized state of each branch so failed or skipped
  paths surface alongside the ones that ran.

It is a pure projection of the already-executed :class:`Graph`: it never
runs, mutates, or reorders the blueprint, so it is a safe addition to the
execution-observability surface.

Implementation mode -- Mode 2 (adapted port). The provenance-graph
construction above is the paper's core mechanism and is kept at full
fidelity. The paper's *auxiliary* machinery is intentionally out of scope
here and left for downstream work:

* the multi-trajectory *quotient graph* (placing several provenance graphs
  on a shared canvas and aligning recurring tools/artifacts across runs);
* pattern extraction and skill abstraction over those comparisons.

Those operate over collections of trajectories and belong in a follow-up
once this single-trajectory provenance representation is in place.
"""

from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from writer.blueprints import Graph

# Edge kinds in the bipartite action <-> artifact provenance graph.
EDGE_PRODUCES = "produces"  # action -> artifact it emits
EDGE_CONSUMED_BY = "consumed_by"  # artifact -> action that reads it


def build_provenance_graph(
    graph: "Graph",
    block_outputs: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert an executed blueprint ``graph`` into an AgentTrails provenance graph.

    ``block_outputs`` is the Journal's per-node metadata map
    (``node_id -> {"component": {...}, "executions": [...]}``); it supplies
    human-readable component info and execution counts. The graph itself
    supplies the dataflow edges (``node.outputs`` / ``node.inputs``) and the
    outcomes that decide whether each branch was realized.

    Returns a JSON-serializable structure::

        {
          "actions":        [ {id, tool, title, category, outcome, executions} ],
          "artifacts":      [ {id, producer, branch, realized, consumers} ],
          "edges":          [ {from, to, type} ],   # produces / consumed_by
          "startActions":   [node_id, ...],         # no incoming artifacts
          "terminalActions":[node_id, ...],         # no outgoing artifacts
        }
    """
    actions: List[Dict[str, Any]] = []
    # One artifact per (producer, branch); a single output can fan out to
    # several consumers, so consumers are accumulated in a set.
    artifacts: Dict[str, Dict[str, Any]] = {}

    for node in graph.nodes:
        meta = (block_outputs.get(node.id) or {}).get("component") or {}
        executions = (block_outputs.get(node.id) or {}).get("executions") or []
        actions.append(
            {
                "id": node.id,
                "tool": meta.get("type") or node.component.type,
                "title": meta.get("title") or node.component.content.get("alias"),
                "category": meta.get("category"),
                "outcome": node.outcome,
                "executions": len(executions),
            }
        )

        for output in node.outputs or []:
            _add_output_edge(artifacts, node.id, output, node.outcome)

    artifact_list: List[Dict[str, Any]] = []
    edges: List[Dict[str, str]] = []
    for artifact in artifacts.values():
        consumers = sorted(set(artifact.pop("_consumers")))
        artifact["consumers"] = consumers
        artifact_list.append(artifact)
        edges.append({"from": artifact["producer"], "to": artifact["id"], "type": EDGE_PRODUCES})
        for consumer in consumers:
            edges.append({"from": artifact["id"], "to": consumer, "type": EDGE_CONSUMED_BY})

    return {
        "actions": actions,
        "artifacts": artifact_list,
        "edges": edges,
        "startActions": [n.id for n in graph.nodes if not n.inputs],
        "terminalActions": [n.id for n in graph.nodes if not n.outputs],
    }


def _add_output_edge(
    artifacts: Dict[str, Dict[str, Any]],
    producer: str,
    output: Dict[str, Any],
    producer_outcome: Any,
) -> None:
    """Record the dataflow edge described by one ``component.outs`` entry.

    Each output declares an artifact (the producer's result on ``branch``)
    consumed by ``toNodeId``. The artifact is marked ``realized`` only when
    the producer actually finished on that branch, so un-taken branches
    (failed / skipped / divergent runs) remain visible for debugging.
    """
    branch = output.get("outId")
    consumer = output.get("toNodeId")
    artifact_id = f"{producer}:{branch}"
    artifact = artifacts.get(artifact_id)
    if artifact is None:
        artifact = {
            "id": artifact_id,
            "producer": producer,
            "branch": branch,
            "realized": producer_outcome == branch,
            "_consumers": [],
        }
        artifacts[artifact_id] = artifact
    if consumer:
        artifact["_consumers"].append(consumer)

"""Provenance graph derivation for Journal trajectories.

Adapted from the core idea of "AgentTrails: Towards Trust and Reuse for
Agentic Tasks" (https://arxiv.org/abs/2607.18816v1): a flat chronological
execution log is converted into a structured provenance graph where each
block execution is a computational *action*, the value it produces is a
data *artifact*, and the blueprint wiring between blocks provides the
dataflow *edges* that a purely chronological log hides.

The target-native substitution (vs. the paper) is the trajectory source:
AgentTrails rebuilds dataflow from arbitrary free-form tool-call traces,
while here the blueprint already encodes the dependency topology as block
``outs`` (``GraphNode.outputs``), so those edges are read directly rather
than re-inferred from state-variable I/O. The paper's cross-trajectory
"shared canvas / joined quotient graph" and skill abstraction are
intentionally out of scope -- this module delivers the single-trajectory
provenance representation that those analyses build on.

The module is dependency-free so it can be unit-tested in isolation and
wired into ``JournalRecord.to_dict`` behind the ``journal_provenance``
feature flag without import cycles.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProvenanceAction:
    """A single block execution modeled as a computational action."""

    id: str
    label: str
    blockType: str
    category: str
    outcome: Optional[str]
    artifactId: Optional[str]
    startedAt: Optional[float] = None
    executionTimeInSeconds: Optional[float] = None


@dataclass
class ProvenanceArtifact:
    """A data artifact produced by an action (a block's result value)."""

    id: str
    producedBy: str
    outId: Optional[str] = None
    value: Any = None


@dataclass
class ProvenanceEdge:
    """A dataflow dependency: ``source`` feeds ``target`` via an artifact."""

    source: str
    target: str
    artifactId: str
    outId: Optional[str] = None


@dataclass
class ProvenanceGraph:
    """Structured provenance view of one blueprint run."""

    actions: List[ProvenanceAction] = field(default_factory=list)
    artifacts: List[ProvenanceArtifact] = field(default_factory=list)
    edges: List[ProvenanceEdge] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        """High-level sensemaking stats (the paper's "downstream analysis").

        Surfaces roots/leaves (entry/exit points), error actions, and counts,
        which a chronological log does not expose directly.
        """
        incoming = {edge.target for edge in self.edges}
        outgoing = {edge.source for edge in self.edges}
        ordered_ids = [action.id for action in self.actions]
        return {
            "totalActions": len(self.actions),
            "totalArtifacts": len(self.artifacts),
            "totalEdges": len(self.edges),
            "roots": [aid for aid in ordered_ids if aid not in incoming],
            "leaves": [aid for aid in ordered_ids if aid not in outgoing],
            "errorActions": [
                action.id for action in self.actions if action.outcome == "error"
            ],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": [action.__dict__ for action in self.actions],
            "artifacts": [artifact.__dict__ for artifact in self.artifacts],
            "edges": [edge.__dict__ for edge in self.edges],
            "summary": self.summary(),
        }


def build_provenance_graph(
    block_outputs: Dict[str, Dict[str, Any]],
    topology: Dict[str, List[Dict[str, Any]]],
) -> ProvenanceGraph:
    """Build a provenance graph from a Journal run.

    Args:
        block_outputs: ``node_id -> {component, executions}`` as produced by
            ``JournalRecord.to_dict``. The most recent execution per node is
            treated as that action's run.
        topology: ``node_id -> outputs`` where each output is a blueprint edge
            ``{"toNodeId": ..., "outId": ...}`` (``GraphNode.outputs``). These
            are the dataflow dependencies between actions.

    Returns:
        A :class:`ProvenanceGraph` with one action per block, an artifact per
        block that produced a (non-``None``) result, and an edge per wiring
        link between blocks that both appear in ``block_outputs``.
    """
    graph = ProvenanceGraph()

    for node_id, entry in block_outputs.items():
        component = entry.get("component", {}) or {}
        executions = entry.get("executions") or []
        execution = executions[-1] if executions else {}
        result = execution.get("result")
        artifact_id = f"{node_id}:out"
        produced_artifact = result is not None

        graph.actions.append(
            ProvenanceAction(
                id=node_id,
                label=component.get("title", ""),
                blockType=component.get("type", ""),
                category=component.get("category", ""),
                outcome=execution.get("outcome"),
                artifactId=artifact_id if produced_artifact else None,
                startedAt=execution.get("startedAt"),
                executionTimeInSeconds=execution.get("executionTimeInSeconds"),
            )
        )
        if produced_artifact:
            graph.artifacts.append(
                ProvenanceArtifact(id=artifact_id, producedBy=node_id, value=result)
            )

    seen_edges = set()
    for node_id, outputs in topology.items():
        for output in outputs or []:
            target = output.get("toNodeId") if isinstance(output, dict) else None
            # Drop edges that leave the recorded run (filtered branches) or
            # self-loops; both would distort the dataflow view.
            if not target or target == node_id or target not in block_outputs:
                continue
            out_id = output.get("outId")
            edge_key = (node_id, target, out_id)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            graph.edges.append(
                ProvenanceEdge(
                    source=node_id,
                    target=target,
                    artifactId=f"{node_id}:out",
                    outId=out_id,
                )
            )

    return graph

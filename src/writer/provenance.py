"""Provenance graphs for journal records.

Converts a ``JournalRecord``'s chronological per-node execution log into an
artifact-level dataflow / provenance view, in the spirit of AgentTrails
(Khanna et al., "AgentTrails: Towards Trust and Reuse for Agentic Tasks",
arXiv:2607.18816). Tool calls (block executions) are modelled as
computational *actions* and the data flowing between them as *artifacts*,
recovering the dependencies that the flat chronological log obscures.

This is a pure, read-only addition: building a provenance graph never
mutates the ``JournalRecord`` and never changes the payload written by
``JournalRecord.to_dict()``. The blueprint graph's ``inputs``/``outputs``
edges -- gated by outcome branches -- are the source of dataflow; the
result *type* of each block is surfaced as the artifact's data shape rather
than the full payload, keeping the view serializable and cheap.

A quotient / alignment helper folds several provenance graphs into a joined
graph that highlights recurring tool-use patterns across executions (the
paper's "joined quotient graph" for comparison and reuse). The interactive
shared-canvas UI and skill abstraction from AgentTrails are intentionally out
of scope: evaluation and downstream tooling belong in follow-up work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from writer.journal import JournalRecord

# Bipartite node kinds.
ACTION = "action"
ARTIFACT = "artifact"

# Edge relations.
PRODUCES = "produces"
CONSUMES = "consumes"

# Label / id for the entry artifact that seeds a run (the journal trigger).
TRIGGER_ARTIFACT_ID = "artifact:trigger"


def _type_name(value: Any) -> str:
    """A short, serializable descriptor of an artifact's data shape."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


@dataclass
class ProvenanceNode:
    id: str
    kind: str
    label: str
    # Action-specific.
    component_type: Optional[str] = None
    component_id: Optional[str] = None
    outcome: Optional[str] = None
    result_type: Optional[str] = None
    # Artifact-specific.
    value_type: Optional[str] = None
    branch: Optional[str] = None
    producer_type: Optional[str] = None
    consumer_type: Optional[str] = None
    # Quotient alignment bookkeeping.
    occurrences: int = 1
    trajectories: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"id": self.id, "kind": self.kind, "label": self.label}
        if self.kind == ACTION:
            data["componentType"] = self.component_type
            data["componentId"] = self.component_id
            data["outcome"] = self.outcome
            data["resultType"] = self.result_type
        else:
            data["valueType"] = self.value_type
            data["branch"] = self.branch
            data["producerType"] = self.producer_type
            data["consumerType"] = self.consumer_type
        if self.occurrences != 1 or self.trajectories:
            data["occurrences"] = self.occurrences
            data["trajectories"] = list(self.trajectories)
        return data


@dataclass
class ProvenanceEdge:
    source: str
    target: str
    relation: str

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "target": self.target, "relation": self.relation}


@dataclass
class ProvenanceGraph:
    nodes: List[ProvenanceNode] = field(default_factory=list)
    edges: List[ProvenanceEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": dict(self.metadata),
        }


def _graph_nodes(record: "JournalRecord") -> List[Any]:
    graph = getattr(record, "graph", None)
    return list(getattr(graph, "nodes", []) or [])


def _node_id(node: Any) -> str:
    node_id = getattr(node, "id", None)
    if node_id:
        return str(node_id)
    component = getattr(node, "component", None)
    return str(getattr(component, "id", "") or "")


def build_provenance_graph(record: "JournalRecord") -> ProvenanceGraph:
    """Convert a ``JournalRecord`` into an AgentTrails-style provenance graph.

    Actions are the record's block executions; artifacts are the data
    dependencies between them (derived from the blueprint graph edges and
    gated by outcome branches) plus the trigger that seeds the run.
    """
    graph_obj = getattr(record, "graph", None)
    nodes = _graph_nodes(record)
    block_outputs: Dict[str, Any] = getattr(record, "block_outputs", {}) or {}
    trigger: Dict[str, Any] = getattr(record, "trigger", {}) or {}

    prov = ProvenanceGraph()
    action_type: Dict[str, str] = {}

    # 1) Actions: one per executed block.
    for node in nodes:
        nid = _node_id(node)
        component = (block_outputs.get(nid) or {}).get("component") or {}
        component_type = component.get("type") or getattr(
            getattr(node, "component", None), "type", "block"
        )
        action_type[nid] = str(component_type)
        prov.nodes.append(
            ProvenanceNode(
                id=f"action:{nid}",
                kind=ACTION,
                label=str(component.get("title") or component_type),
                component_type=component_type,
                component_id=nid,
                outcome=getattr(node, "outcome", None),
                result_type=_type_name(getattr(node, "result", None)),
            )
        )

    # 2) Trigger artifact: the entry payload consumed by start nodes.
    trigger_component = trigger.get("component") or {}
    prov.nodes.append(
        ProvenanceNode(
            id=TRIGGER_ARTIFACT_ID,
            kind=ARTIFACT,
            label=str(trigger_component.get("title") or trigger.get("event") or "trigger"),
            value_type=_type_name(trigger.get("payload")),
            producer_type="trigger",
        )
    )

    # 3) Dataflow artifacts + produces/consumes edges.
    for node in nodes:
        nid = _node_id(node)
        inputs = getattr(node, "inputs", None) or []
        consumer_type = action_type.get(nid, "block")
        if not inputs:
            # Start node: consumes the trigger artifact.
            prov.edges.append(ProvenanceEdge(TRIGGER_ARTIFACT_ID, f"action:{nid}", CONSUMES))
            continue
        for raw_input in inputs:
            if not isinstance(raw_input, dict):
                continue
            from_id = raw_input.get("fromNodeId")
            branch = raw_input.get("outId")
            if not from_id:
                continue
            producer = graph_obj.get_node(from_id) if graph_obj is not None else None
            producer_type = action_type.get(from_id)
            if producer_type is None:
                producer_type = (
                    getattr(getattr(producer, "component", None), "type", "block")
                    if producer is not None
                    else "block"
                )
            artifact_id = f"artifact:{from_id}->{nid}:{branch}"
            prov.nodes.append(
                ProvenanceNode(
                    id=artifact_id,
                    kind=ARTIFACT,
                    label=f"{producer_type} → {consumer_type}",
                    value_type=_type_name(getattr(producer, "result", None)),
                    branch=branch,
                    producer_type=producer_type,
                    consumer_type=consumer_type,
                )
            )
            prov.edges.append(ProvenanceEdge(f"action:{from_id}", artifact_id, PRODUCES))
            prov.edges.append(ProvenanceEdge(artifact_id, f"action:{nid}", CONSUMES))

    started_at = getattr(record, "started_at", None)
    prov.metadata = {
        "instanceType": getattr(record, "instance_type", None),
        "blueprintId": getattr(record, "blueprint_id", None),
        "timestamp": started_at.isoformat() if started_at is not None else None,
        "result": getattr(record, "result", None),
        "triggerEvent": trigger.get("event"),
        "numActions": sum(1 for node in prov.nodes if node.kind == ACTION),
        "numArtifacts": sum(1 for node in prov.nodes if node.kind == ARTIFACT),
    }
    return prov


def _quotient_key(node: ProvenanceNode) -> str:
    if node.kind == ACTION:
        return f"{ACTION}|{node.component_type}|{node.outcome}"
    return f"{ARTIFACT}|{node.producer_type}|{node.consumer_type}|{node.branch}|{node.value_type}"


def align_provenance_graphs(graphs: List[ProvenanceGraph]) -> ProvenanceGraph:
    """Fold multiple provenance graphs into a joined quotient graph.

    Recurring actions (same component type + outcome) and recurring artifacts
    (same producer/consumer types, branch and value shape) merge into single
    nodes, each annotated with how many trajectories exhibited it. This
    surfaces common tool-use patterns across runs -- the comparison / reuse
    view from AgentTrails -- without flattening the dataflow structure.
    """
    merged: Dict[str, ProvenanceNode] = {}
    sig_of: Dict[str, str] = {}

    for idx, graph in enumerate(graphs):
        for node in graph.nodes:
            key = _quotient_key(node)
            sig_of[node.id] = key
            quotient = merged.get(key)
            if quotient is None:
                quotient = ProvenanceNode(
                    id=node.id,
                    kind=node.kind,
                    label=node.label,
                    component_type=node.component_type,
                    component_id=node.component_id,
                    outcome=node.outcome,
                    result_type=node.result_type,
                    value_type=node.value_type,
                    branch=node.branch,
                    producer_type=node.producer_type,
                    consumer_type=node.consumer_type,
                    occurrences=1,
                    trajectories=[idx],
                )
                merged[key] = quotient
            else:
                quotient.occurrences += 1
                if idx not in quotient.trajectories:
                    quotient.trajectories.append(idx)

    # Assign stable, collision-free ids after all merges are known.
    for index, quotient in enumerate(merged.values()):
        quotient.id = f"{quotient.kind}:{index}"

    # Edges: translate each input edge's endpoints to their quotient node ids.
    seen_edges = set()
    quotient_edges: List[ProvenanceEdge] = []
    for graph in graphs:
        for edge in graph.edges:
            src = merged.get(sig_of.get(edge.source, ""))
            tgt = merged.get(sig_of.get(edge.target, ""))
            if src is None or tgt is None:
                continue
            fingerprint = (src.id, tgt.id, edge.relation)
            if fingerprint in seen_edges:
                continue
            seen_edges.add(fingerprint)
            quotient_edges.append(ProvenanceEdge(src.id, tgt.id, edge.relation))

    quotient_nodes = list(merged.values())
    return ProvenanceGraph(
        nodes=quotient_nodes,
        edges=quotient_edges,
        metadata={
            "kind": "quotient",
            "numTrajectories": len(graphs),
            "numActions": sum(1 for node in quotient_nodes if node.kind == ACTION),
            "numArtifacts": sum(1 for node in quotient_nodes if node.kind == ARTIFACT),
        },
    )

"""Tests for the journal provenance view (AgentTrails-style dataflow graph).

The integration tests build a real blueprint ``Graph`` and a real
``JournalRecord`` -- exercising the wiring added in ``writer.journal`` --
and assert that the chronological log is converted into an artifact-level
provenance graph without disturbing the persisted journal payload.
"""

from types import SimpleNamespace
from typing import Dict, List, Optional

from writer.blocks.base_block import BlueprintBlock
from writer.blueprints import Graph
from writer.core_ui import Component
from writer.journal import JournalRecord
from writer.provenance import (
    align_provenance_graphs,
    build_provenance_graph,
)


class _ProbeBlock(BlueprintBlock):
    """A block registered only so a ``Graph`` can be constructed.

    It is never executed in these tests; node outcomes and results are
    populated directly so the provenance conversion can be exercised in
    isolation from the block runner.
    """

    def run(self) -> None:
        self.result = "echo"


_TOOLS: Dict[str, type] = {"probe": _ProbeBlock}


def _component(
    cid: str, outs: Optional[List[Dict]] = None, alias: Optional[str] = None
) -> Component:
    content: Dict = {}
    if alias:
        content["alias"] = alias
    return Component(id=cid, type="probe", outs=outs, content=content)


def _run_outcomes(graph: Graph, result: str = "hello") -> None:
    """Populate each node with a successful outcome + result, without running."""
    for node in graph.nodes:
        node.status = "success"
        node.tool = SimpleNamespace(
            result=result,
            started_at=-1,
            execution_time_in_seconds=-1,
            captured_stdout=None,
            captured_logs=None,
            message=None,
        )


def _linear_record() -> JournalRecord:
    graph = Graph(
        nodes=[
            _component("N1", outs=[{"toNodeId": "N2", "outId": "success"}], alias="Source"),
            _component("N2", alias="Sink"),
        ],
        tools=_TOOLS,
    )
    _run_outcomes(graph)
    return JournalRecord(
        execution_environment={"context": {"event": "wf-click"}, "payload": {"x": 1}},
        title="Provenance test",
        graph=graph,
    )


def test_provenance_graph_surfaces_dataflow_dependencies():
    record = _linear_record()
    prov = record.provenance()

    assert prov["metadata"]["numActions"] == 2
    assert prov["metadata"]["numArtifacts"] == 2
    assert prov["metadata"]["triggerEvent"] == "wf-click"

    node_ids = {node["id"] for node in prov["nodes"]}
    assert {"action:N1", "action:N2", "artifact:trigger"} <= node_ids
    # The N1 -> N2 dependency, recovered from the blueprint edge.
    assert "artifact:N1->N2:success" in node_ids

    artifact = next(n for n in prov["nodes"] if n["id"] == "artifact:N1->N2:success")
    assert artifact["branch"] == "success"
    assert artifact["valueType"] == "str"  # result "hello" -> str

    relations = {(e["source"], e["target"], e["relation"]) for e in prov["edges"]}
    assert ("artifact:trigger", "action:N1", "consumes") in relations
    assert ("action:N1", "artifact:N1->N2:success", "produces") in relations
    assert ("artifact:N1->N2:success", "action:N2", "consumes") in relations


def test_provenance_leaves_journal_payload_contract_intact():
    record = _linear_record()
    # Materializing the provenance view must not change the saved payload.
    record.provenance()
    payload = record.to_dict()
    assert "provenance" not in payload
    assert set(payload) == {
        "timestamp",
        "instanceType",
        "blueprintId",
        "trigger",
        "blockOutputs",
        "result",
        "isRunable",
    }


def test_provenance_is_cached_on_the_record():
    record = _linear_record()
    first = record.provenance()
    second = record.provenance()
    assert first is second


def test_align_provenance_graphs_merges_recurring_patterns():
    first = build_provenance_graph(_linear_record())
    second = build_provenance_graph(_linear_record())

    quotient = align_provenance_graphs([first, second])
    assert quotient.metadata["kind"] == "quotient"
    assert quotient.metadata["numTrajectories"] == 2

    # Both "probe/success" actions collapse to one quotient node, present in
    # both trajectories (2 nodes x 2 trajectories = 4 occurrences).
    actions = [n for n in quotient.nodes if n.kind == "action"]
    assert len(actions) == 1
    assert actions[0].occurrences == 4
    assert actions[0].trajectories == [0, 1]
    # Edges are deduplicated across the two trajectories.
    assert len({(e.source, e.target, e.relation) for e in quotient.edges}) == len(quotient.edges)

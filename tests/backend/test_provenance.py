"""Tests for the AgentTrails-style provenance projection.

The integration tests go through the Journal (the call site) over a real
``writer.blueprints.Graph``; the unit tests exercise ``build_provenance_graph``
directly to pin down dataflow edges, fan-out and realized/unrealized branches.
"""

from types import SimpleNamespace
from typing import Dict, List, Optional

import pytest
from writer.blocks.base_block import BlueprintBlock, BlueprintBlock_T
from writer.blueprints import Graph, GraphBuilder
from writer.core import Config
from writer.core_ui import Component
from writer.journal import JournalRecord
from writer.provenance import build_provenance_graph

TEST_BLOCK = "provenance_test_block"
_tools: Dict[str, BlueprintBlock_T] = {}


class _TestBlock(BlueprintBlock):
    """Registered only so GraphBuilder accepts the test component type."""

    def run(self) -> None:  # pragma: no cover - never executed here
        self.result = "data"
        self.outcome = "success"


_tools[TEST_BLOCK] = _TestBlock


def _component(node_id: str, outs: Optional[List[dict]] = None) -> Component:
    return Component(
        id=node_id,
        type=TEST_BLOCK,
        outs=outs or [],
        content={"alias": node_id},
    )


def _build_graph(components: List[Component]) -> Graph:
    return GraphBuilder(components=components, tools=_tools).build()


def _ran(graph: Graph, outcome: str = "success", result: str = "data") -> Graph:
    """Simulate execution by attaching a finished tool to every node."""
    for node in graph.nodes:
        node.tool = SimpleNamespace(outcome=outcome, result=result)  # type: ignore[assignment]
    return graph


class TestProvenanceGraph:
    def test_linear_dataflow(self):
        graph = _build_graph(
            [
                _component("N1", outs=[{"toNodeId": "N2", "outId": "success"}]),
                _component("N2"),
            ]
        )
        _ran(graph)

        provenance = build_provenance_graph(graph, block_outputs={})

        assert [a["id"] for a in provenance["actions"]] == ["N1", "N2"]
        assert provenance["actions"][0]["outcome"] == "success"

        artifact = provenance["artifacts"][0]
        assert artifact == {
            "id": "N1:success",
            "producer": "N1",
            "branch": "success",
            "realized": True,
            "consumers": ["N2"],
        }
        assert provenance["edges"] == [
            {"from": "N1", "to": "N1:success", "type": "produces"},
            {"from": "N1:success", "to": "N2", "type": "consumed_by"},
        ]
        assert provenance["startActions"] == ["N1"]
        assert provenance["terminalActions"] == ["N2"]

    def test_fanout_and_unrealized_branch(self):
        # N1 fans a single `success` artifact out to N2 and N3, and declares
        # an `error` branch into N4 that never ran.
        graph = _build_graph(
            [
                _component(
                    "N1",
                    outs=[
                        {"toNodeId": "N2", "outId": "success"},
                        {"toNodeId": "N3", "outId": "success"},
                        {"toNodeId": "N4", "outId": "error"},
                    ],
                ),
                _component("N2"),
                _component("N3"),
                _component("N4"),
            ]
        )
        _ran(graph, outcome="success")

        provenance = build_provenance_graph(graph, block_outputs={})
        artifacts = {a["id"]: a for a in provenance["artifacts"]}

        # One artifact per (producer, branch); success fans out to two consumers.
        assert artifacts["N1:success"]["realized"] is True
        assert artifacts["N1:success"]["consumers"] == ["N2", "N3"]
        # The error branch was declared but the producer finished on success.
        assert artifacts["N1:error"]["realized"] is False
        assert artifacts["N1:error"]["consumers"] == ["N4"]

        assert provenance["startActions"] == ["N1"]
        assert sorted(provenance["terminalActions"]) == ["N2", "N3", "N4"]

    def test_execution_counts_from_block_outputs(self):
        graph = _build_graph([_component("N1")])
        _ran(graph)
        block_outputs = {"N1": {"component": {"type": TEST_BLOCK}, "executions": [{}, {}]}}

        provenance = build_provenance_graph(graph, block_outputs=block_outputs)
        assert provenance["actions"][0]["executions"] == 2


class TestJournalIntegration:
    """Exercises the call site: JournalRecord.to_dict() wiring into provenance."""

    def _record(self) -> JournalRecord:
        graph = _build_graph(
            [
                _component("N1", outs=[{"toNodeId": "N2", "outId": "success"}]),
                _component("N2"),
            ]
        )
        _ran(graph)
        return JournalRecord(execution_environment={}, title="Test Execution", graph=graph)

    def test_provenance_opt_in_via_feature_flag(self, monkeypatch):
        record = self._record()

        # Off by default: the journal payload is unchanged (existing shape).
        monkeypatch.setattr(Config, "feature_flags", [])
        assert "provenance" not in record.to_dict()

        # Opt-in surfaces the dataflow DAG alongside the chronological blocks.
        monkeypatch.setattr(Config, "feature_flags", ["journal.provenance"])
        payload = record.to_dict()
        assert "provenance" in payload
        provenance = payload["provenance"]

        artifact_ids = [a["id"] for a in provenance["artifacts"]]
        assert "N1:success" in artifact_ids
        assert provenance["startActions"] == ["N1"]
        assert provenance["terminalActions"] == ["N2"]
        # Execution recorded through the Journal path is reflected.
        assert provenance["actions"][0]["executions"] == 1

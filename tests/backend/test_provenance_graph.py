"""Integration tests for the journal provenance / dataflow graph.

Exercises the wiring in ``writer.journal`` -- the opt-in ``provenanceGraph``
key emitted by ``JournalRecord.to_dict`` -- by building a real
:class:`Graph` from real :class:`Component` objects and reading the record
back through ``to_dict``. This targets the call-site edit (``to_dict``) and
the builder over the genuine ``GraphNode.inputs``/``outputs`` dependency
edges, without depending on the app-runner subprocess.
"""

from types import SimpleNamespace

import pytest
from writer.blueprints import Graph
from writer.core import Config
from writer.core_ui import Component
from writer.journal import JournalRecord


@pytest.fixture
def provenance_graph():
    """A two-node graph: trigger -> logmessage, wired on the "success" port."""
    trigger = Component(
        id="triggerNode",
        type="test_trigger_block",
        content={"alias": "Trigger"},
        outs=[{"toNodeId": "sinkNode", "outId": "success"}],
        parentId="blueprint-root",
    )
    sink = Component(
        id="sinkNode",
        type="test_log_block",
        content={"alias": "Log"},
        parentId="blueprint-root",
    )
    # ``Graph`` requires a (type -> block class) entry per node so GraphNode
    # can resolve its tool class; the tool is never run here.
    graph = Graph([trigger, sink], tools={trigger.type: object, sink.type: object})

    # Simulate the nodes having executed and produced artifacts.
    graph.node_map["triggerNode"].tool = SimpleNamespace(
        result={"proposedSessionId": None}, outcome="trigger"
    )
    graph.node_map["sinkNode"].tool = SimpleNamespace(result="AAA", outcome="success")
    return graph


@pytest.fixture
def journal_provenance_flag():
    """Toggle the opt-in flag on for the test, off again afterwards."""
    flag = "journal_provenance"
    if flag not in Config.feature_flags:
        Config.feature_flags.append(flag)
    yield flag
    while flag in Config.feature_flags:
        Config.feature_flags.remove(flag)


def _record(graph):
    return JournalRecord(
        execution_environment={"context": {"event": "wf-click"}, "payload": {}},
        title="UI",
        graph=graph,
    )


def test_provenance_graph_emitted_when_flag_on(provenance_graph, journal_provenance_flag):
    record = _record(provenance_graph)
    data = record.to_dict()

    # The opt-in graph rides along on the journal entry and does not trip
    # the runability sanitizer.
    assert data["isRunable"] is True
    graph = data["provenanceGraph"]

    nodes_by_id = {n["id"]: n for n in graph["nodes"]}
    assert set(nodes_by_id) == {"triggerNode", "sinkNode"}

    trigger = nodes_by_id["triggerNode"]
    assert trigger["type"] == "test_trigger_block"
    assert trigger["category"] == "Unknown category"
    assert trigger["signature"] == "test_trigger_block:Unknown category"
    assert trigger["outcome"] == "trigger"
    assert trigger["artifact"] == {"present": True, "type": "dict", "size": 1}

    sink = nodes_by_id["sinkNode"]
    assert sink["outcome"] == "success"
    assert sink["artifact"] == {"present": True, "type": "str", "size": 3}

    # The dependency edge that flat blockOutputs hides: the trigger fed the
    # logmessage on its "success" output port.
    assert graph["edges"] == [{"from": "triggerNode", "to": "sinkNode", "port": "success"}]

    assert graph["stats"] == {
        "nodeCount": 2,
        "edgeCount": 1,
        "sourceCount": 1,
        "sinkCount": 1,
        "outcomes": {"trigger": 1, "success": 1},
    }


def test_provenance_graph_omitted_when_flag_off(provenance_graph):
    # Ensure the flag really is off for this test.
    while "journal_provenance" in Config.feature_flags:
        Config.feature_flags.remove("journal_provenance")

    data = _record(provenance_graph).to_dict()

    # Opt-in is non-breaking: the journal entry keeps its original shape.
    assert "provenanceGraph" not in data
    assert "blockOutputs" in data

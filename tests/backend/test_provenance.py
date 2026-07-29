from datetime import datetime, timezone
from types import SimpleNamespace

from writer.core import Config
from writer.journal import JournalRecord
from writer.provenance import build_provenance_graph


# A small citation-generation pipeline, echoing the paper's motivation of
# making a RAG/citation trajectory debuggable: trigger -> retrieve -> cite,
# with a parallel failing branch.
def _block_outputs():
    return {
        "trigger": {
            "component": {
                "type": "blueprints_apitrigger",
                "id": "trigger",
                "title": "API alias",
                "category": "Triggers",
            },
            "executions": [
                {
                    "result": {"query": "what is x"},
                    "outcome": "trigger",
                    "startedAt": 1.0,
                    "executionTimeInSeconds": 0.1,
                }
            ],
        },
        "retrieve": {
            "component": {
                "type": "blueprints_runpython",
                "id": "retrieve",
                "title": "Retrieve",
                "category": "Other",
            },
            "executions": [
                {
                    "result": {"docs": ["d1", "d2"]},
                    "outcome": "success",
                    "startedAt": 2.0,
                    "executionTimeInSeconds": 0.2,
                }
            ],
        },
        "cite": {
            "component": {
                "type": "blueprints_logmessage",
                "id": "cite",
                "title": "Generate citation",
                "category": "Other",
            },
            "executions": [
                {
                    "result": "Citation #1",
                    "outcome": "success",
                    "startedAt": 3.0,
                    "executionTimeInSeconds": 0.3,
                }
            ],
        },
        "fail": {
            "component": {
                "type": "blueprints_runpython",
                "id": "fail",
                "title": "Maybe fail",
                "category": "Other",
            },
            "executions": [
                {
                    "result": None,
                    "outcome": "error",
                    "message": "boom",
                    "startedAt": 4.0,
                    "executionTimeInSeconds": 0.4,
                }
            ],
        },
    }


def _topology():
    return {
        "trigger": [
            {"toNodeId": "retrieve", "outId": "success"},
            {"toNodeId": "fail", "outId": "success"},
        ],
        "retrieve": [{"toNodeId": "cite", "outId": "success"}],
        "cite": [],
        "fail": [],
    }


def test_build_provenance_graph_extracts_actions_artifacts_and_edges():
    graph = build_provenance_graph(_block_outputs(), _topology()).to_dict()

    assert {a["id"] for a in graph["actions"]} == {"trigger", "retrieve", "cite", "fail"}

    # An artifact is created for every block that produced a non-None result;
    # the failed block produced None, so it has no artifact.
    assert {a["producedBy"] for a in graph["artifacts"]} == {"trigger", "retrieve", "cite"}

    # Dataflow edges come straight from the blueprint wiring.
    edges = {(e["source"], e["target"]) for e in graph["edges"]}
    assert edges == {("trigger", "retrieve"), ("trigger", "fail"), ("retrieve", "cite")}

    # The retrieve -> cite edge carries the producer's artifact reference.
    retrieve_cite = next(
        e for e in graph["edges"] if e["source"] == "retrieve" and e["target"] == "cite"
    )
    assert retrieve_cite["artifactId"] == "retrieve:out"
    assert retrieve_cite["outId"] == "success"


def test_provenance_summary_reports_roots_leaves_and_errors():
    summary = build_provenance_graph(_block_outputs(), _topology()).summary()

    assert summary["totalActions"] == 4
    assert summary["totalArtifacts"] == 3
    assert summary["totalEdges"] == 3
    assert summary["roots"] == ["trigger"]
    assert set(summary["leaves"]) == {"cite", "fail"}
    assert summary["errorActions"] == ["fail"]


def test_edges_to_unknown_targets_and_self_loops_are_dropped():
    block_outputs = {
        "a": {"component": {}, "executions": [{"result": 1, "outcome": "success"}]},
        "b": {"component": {}, "executions": [{"result": 2, "outcome": "success"}]},
    }
    topology = {
        # "ghost" is not in block_outputs -> dropped; a -> a self-loop -> dropped.
        "a": [
            {"toNodeId": "b", "outId": "success"},
            {"toNodeId": "ghost", "outId": "success"},
            {"toNodeId": "a", "outId": "success"},
        ],
        "b": [],
    }
    edges = {
        (e.source, e.target)
        for e in build_provenance_graph(block_outputs, topology).edges
    }
    assert edges == {("a", "b")}


def test_action_without_executions_has_no_artifact():
    block_outputs = {
        "a": {"component": {"title": "A", "type": "t", "category": "c"}, "executions": []}
    }
    graph = build_provenance_graph(block_outputs, {"a": []}).to_dict()
    assert graph["actions"][0]["outcome"] is None
    assert graph["actions"][0]["artifactId"] is None
    assert graph["artifacts"] == []


class _FakeNode:
    """Minimal stand-in for ``GraphNode`` used by ``JournalRecord.to_dict``."""

    def __init__(self, node_id, result, outcome, outputs=None):
        self.id = node_id
        self.result = result
        self.outcome = outcome
        self.outputs = outputs or []
        self.tool = None


def _make_record():
    # Bypass JournalRecord.__init__ (which needs a fully built app graph) and
    # set only the attributes that ``to_dict`` reads, so we exercise the real
    # serialization path -- including the provenance wiring -- in isolation.
    record = JournalRecord.__new__(JournalRecord)
    record.started_at = datetime.now(timezone.utc)
    record.instance_type = "agent"
    record.blueprint_id = "bp1"
    record.trigger = {"event": "wf-run-blueprint", "type": "API", "component": {}, "payload": {}}
    record.result = "success"
    record.is_runable = True
    record.block_outputs = {
        node_id: {"component": {"type": "block", "id": node_id, "title": node_id, "category": "Other"}, "executions": []}
        for node_id in ("trigger", "log")
    }
    record.graph = SimpleNamespace(
        nodes=[
            _FakeNode(
                "trigger",
                {"proposedSessionId": None},
                "trigger",
                outputs=[{"toNodeId": "log", "outId": "success"}],
            ),
            _FakeNode("log", "AAA", "success"),
        ]
    )
    return record


def test_to_dict_includes_provenance_when_flag_enabled(monkeypatch):
    monkeypatch.setattr(Config, "feature_flags", ["journal", "journal_provenance"])

    data = _make_record().to_dict()

    assert "provenance" in data
    provenance = data["provenance"]
    assert {a["id"] for a in provenance["actions"]} == {"trigger", "log"}
    assert {a["producedBy"] for a in provenance["artifacts"]} == {"trigger", "log"}
    edges = {(e["source"], e["target"]) for e in provenance["edges"]}
    assert ("trigger", "log") in edges
    assert provenance["summary"]["roots"] == ["trigger"]


def test_to_dict_omits_provenance_when_flag_disabled(monkeypatch):
    monkeypatch.setattr(Config, "feature_flags", ["journal"])

    data = _make_record().to_dict()

    assert "provenance" not in data
    # The existing flat chronological view is unchanged.
    assert set(data["blockOutputs"].keys()) == {"trigger", "log"}

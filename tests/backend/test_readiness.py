import pytest

from writer.core import Config
from writer.journal import JournalRecord, attach_readiness
from writer.readiness import (
    DEFAULT_GATE_THRESHOLD,
    ReadinessScore,
    ReadinessScorer,
    compute_readiness,
    extract_signals,
    pareto_frontier,
    percentile,
    weighted_score,
)


def _record(result="success", executions=(), trigger_type="API"):
    block_outputs = {}
    for index, (outcome, seconds) in enumerate(executions):
        node_id = f"node{index}"
        block_outputs[node_id] = {
            "component": {"type": "block", "id": node_id, "title": f"block {index}"},
            "executions": [{"outcome": outcome, "executionTimeInSeconds": seconds}],
        }
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "instanceType": "agent",
        "blueprintId": "bp1",
        "trigger": {"event": "wf-run", "payload": {}, "component": {}, "type": trigger_type},
        "blockOutputs": block_outputs,
        "result": result,
        "isRunable": True,
    }


class _FakeKV:
    def __init__(self):
        self.data = {}

    def is_accessible(self):
        return True

    def save(self, key, data):
        self.data["key"] = key
        self.data["data"] = data


def _bare_journal_record(record):
    journal_record = JournalRecord.__new__(JournalRecord)
    journal_record.to_dict = lambda: dict(record)
    journal_record.construct_key = lambda: "wf-journal-test"
    return journal_record


def test_percentile_nearest_rank():
    assert percentile([], 95) is None
    assert percentile([1.0], 95) == 1.0
    assert percentile([1.0, 2.0], 95) == 2.0
    assert percentile(list(range(1, 11)), 50) == 5
    assert percentile(list(range(1, 11)), 95) == 10


def test_extract_signals_success_record():
    record = _record(result="success", executions=[("success", 0.1), ("success", 0.2)])
    signals = extract_signals(record, latency_budget_s=5.0)
    assert signals["success"] == 1.0
    assert signals["compliance"] == 1.0
    assert signals["p95_latency"] == pytest.approx(1.0 - (0.2 / 5.0))


def test_extract_signals_error_record():
    record = _record(result="error", executions=[("success", 0.1), ("error", 0.2)])
    signals = extract_signals(record)
    assert signals["success"] == 0.0
    assert signals["compliance"] == 0.5


def test_extract_signals_supports_pluggable_signals():
    record = _record(executions=[("success", 0.1)])
    signals = extract_signals(
        record,
        compliance_checker=lambda r: 0.25,
        extra_signals={"groundedness": 0.9, "bad": "nope"},
    )
    assert signals["compliance"] == 0.25
    assert signals["groundedness"] == 0.9
    assert "bad" not in signals


def test_weighted_score_normalises_by_total_weight():
    signals = {"success": 1.0, "compliance": 1.0, "p95_latency": 1.0}
    assert weighted_score(signals, {"success": 0.5, "compliance": 0.3, "p95_latency": 0.2}) == 1.0
    assert weighted_score({"success": 0.0}, {}) == 0.0


def test_scorer_passes_gate_for_clean_run():
    record = _record(result="success", executions=[("success", 0.1), ("success", 0.2)])
    score = ReadinessScorer().score(record)
    assert score.gate_passed is True
    assert score.score > DEFAULT_GATE_THRESHOLD


def test_scorer_fails_gate_for_error_run():
    record = _record(result="error", executions=[("error", 0.1)])
    score = ReadinessScorer().score(record)
    assert score.gate_passed is False
    assert score.score < DEFAULT_GATE_THRESHOLD


def test_compute_readiness_is_defensive():
    # No result and no executions: success 0, compliance 1, latency 1 -> 0.5.
    assert compute_readiness({})["score"] == 0.5
    broken = compute_readiness("not-a-dict")  # type: ignore[arg-type]
    assert broken["score"] is None
    assert broken["gatePassed"] is False


def test_pareto_frontier_drops_dominated_scenarios():
    api = ReadinessScore(
        {"success": 0.9, "compliance": 1.0, "p95_latency": 1.0},
        0.9,
        True,
        0.8,
        {},
        meta={"scenario": "API", "p95LatencySeconds": 0.5},
    )
    cron = ReadinessScore(
        {"success": 0.9, "compliance": 1.0, "p95_latency": 0.9},
        0.9,
        True,
        0.8,
        {},
        meta={"scenario": "Cron", "p95LatencySeconds": 1.0},
    )
    ui = ReadinessScore(
        {"success": 0.5, "compliance": 1.0, "p95_latency": 0.5},
        0.5,
        False,
        0.8,
        {},
        meta={"scenario": "UI", "p95LatencySeconds": 5.0},
    )
    frontier = pareto_frontier([api, cron, ui])
    assert "API" in frontier
    assert "UI" not in frontier  # dominated by API: same success, lower latency


def test_aggregate_groups_by_scenario_and_frontier():
    records = [
        _record(result="success", executions=[("success", 0.1)], trigger_type="API"),
        _record(result="success", executions=[("success", 0.2)], trigger_type="API"),
        _record(result="error", executions=[("error", 9.0)], trigger_type="Cron"),
    ]
    report = ReadinessScorer().aggregate(records).to_dict()
    assert report["scenarioCount"] == 2
    assert set(report["scenarios"]) == {"API", "Cron"}
    assert report["scenarios"]["API"]["gatePassed"] is True
    assert report["scenarios"]["Cron"]["gatePassed"] is False
    assert report["paretoFrontier"] == ["API"]


def test_journal_attach_readiness_invokes_scorer():
    record = _record(result="success", executions=[("success", 0.1), ("success", 0.2)])
    enriched = attach_readiness(dict(record))
    assert "readiness" in enriched
    assert enriched["readiness"]["gatePassed"] is True
    assert enriched["readiness"]["score"] > DEFAULT_GATE_THRESHOLD


def test_journal_save_attaches_readiness_when_flag_enabled(monkeypatch):
    record = _record(result="success", executions=[("success", 0.1), ("success", 0.2)])
    fake_kv = _FakeKV()
    monkeypatch.setattr("writer.journal.writer_kv_storage", fake_kv)
    monkeypatch.setattr(Config, "feature_flags", ["journal", "journal-readiness"])

    _bare_journal_record(record).save()

    assert "readiness" in fake_kv.data["data"]
    assert fake_kv.data["data"]["readiness"]["gatePassed"] is True


def test_journal_save_skips_readiness_when_flag_absent(monkeypatch):
    record = _record(result="success", executions=[("success", 0.1)])
    fake_kv = _FakeKV()
    monkeypatch.setattr("writer.journal.writer_kv_storage", fake_kv)
    monkeypatch.setattr(Config, "feature_flags", ["journal"])  # readiness flag absent

    _bare_journal_record(record).save()

    assert "readiness" not in fake_kv.data["data"]

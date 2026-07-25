"""Deployment-readiness scoring for Writer Framework execution records.

Adapted from "LLM Readiness Harness: Evaluation, Observability, and CI Gates
for LLM/RAG Applications" (arXiv:2603.27355). The paper turns evaluation into a
deployment decision by aggregating workflow success, policy compliance,
groundedness, retrieval hit rate, cost and p95 latency into scenario-weighted
readiness scores with Pareto frontiers, under the minimal API contract
"execution records in -> scenario-weighted readiness scores out".

This module implements that core contract against the framework's native
execution record (``writer.journal.JournalRecord.to_dict``). The paper's
auxiliary infrastructure is substituted with target-native equivalents:

* OpenTelemetry observability -> the Journal execution trace itself (the
  record's per-block outcomes, timings and error messages).
* CI quality-gate runner      -> an optional pass/fail threshold on the score;
  no CI framework is wired here, evaluation is a downstream concern.
* BEIR grounding benchmarks   -> cut. Groundedness, retrieval-hit-rate and cost
  need data the execution record does not carry, so they are exposed as
  pluggable signals and simply omitted when absent.

Scoring is deterministic and dependency-free so it can run on the journal
``save()`` path without affecting record persistence.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

# Native signals derived directly from an execution record.
SIGNAL_SUCCESS = "success"
SIGNAL_COMPLIANCE = "compliance"
SIGNAL_P95_LATENCY = "p95_latency"

DEFAULT_WEIGHTS: Dict[str, float] = {
    SIGNAL_SUCCESS: 0.5,
    SIGNAL_COMPLIANCE: 0.3,
    SIGNAL_P95_LATENCY: 0.2,
}
DEFAULT_LATENCY_BUDGET_SECONDS = 5.0
DEFAULT_GATE_THRESHOLD = 0.8


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Nearest-rank percentile (``q`` in 0..100). Returns ``None`` if empty."""
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    rank = max(1, math.ceil((q / 100.0) * n))
    return ordered[min(rank, n) - 1]


def iter_executions(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten every block execution captured in a journal record."""
    outputs = record.get("blockOutputs") or {}
    return [execution for node in outputs.values() for execution in (node.get("executions") or [])]


def outcome_compliance(executions: Sequence[Dict[str, Any]]) -> float:
    """Outcome-based proxy for the paper's policy-compliance signal.

    A block execution whose ``outcome`` is ``"error"`` counts as a compliance
    failure; everything else (``success``, ``trigger``, ...) is treated as
    compliant. Returns 1.0 when there are no executions to assess.
    """
    if not executions:
        return 1.0
    failures = sum(1 for execution in executions if execution.get("outcome") == "error")
    return 1.0 - (failures / len(executions))


def _latency_score(p95: Optional[float], budget: float) -> float:
    if p95 is None or budget <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (p95 / budget)))


def extract_signals(
    record: Dict[str, Any],
    latency_budget_s: float = DEFAULT_LATENCY_BUDGET_SECONDS,
    compliance_checker: Optional[Callable[[Dict[str, Any]], Optional[float]]] = None,
    extra_signals: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute the readiness signals available for a single execution record.

    ``compliance_checker`` overrides the outcome-based compliance proxy with an
    app-specific policy assessment; ``extra_signals`` injects the paper's
    groundedness / retrieval / cost signals when the caller has that data.
    Every value is clamped to ``[0, 1]`` where 1 means "ready".
    """
    executions = iter_executions(record)

    success = 1.0 if record.get("result") == "success" else 0.0

    if compliance_checker is not None:
        checked = compliance_checker(record)
        compliance = outcome_compliance(executions) if checked is None else float(checked)
    else:
        compliance = outcome_compliance(executions)

    times = [
        float(execution["executionTimeInSeconds"])
        for execution in executions
        if execution.get("executionTimeInSeconds") is not None
    ]
    latency_score = _latency_score(percentile(times, 95), latency_budget_s)

    signals: Dict[str, float] = {
        SIGNAL_SUCCESS: success,
        SIGNAL_COMPLIANCE: compliance,
        SIGNAL_P95_LATENCY: latency_score,
    }
    if extra_signals:
        for name, value in extra_signals.items():
            if isinstance(value, (int, float)):
                signals[name] = max(0.0, min(1.0, float(value)))
    return signals


def weighted_score(signals: Dict[str, float], weights: Dict[str, float]) -> float:
    """Weighted mean of the signals (in ``[0, 1]``); 0.0 when nothing is weighted."""
    total_weight = sum(weights.get(name, 0.0) for name in signals)
    if total_weight <= 0:
        return 0.0
    return sum(signals[name] * weights.get(name, 0.0) for name in signals) / total_weight


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _default_scenario_key(record: Dict[str, Any]) -> str:
    return (record.get("trigger") or {}).get("type") or "unknown"


def pareto_frontier(scores: Sequence["ReadinessScore"]) -> List[str]:
    """Scenario names on the success-vs-latency Pareto frontier.

    A scenario dominates another when it is at least as good on both success
    rate and p95 latency and strictly better on one. Missing latencies are
    treated as the worst case so they never dominate.
    """

    def latency_of(score: "ReadinessScore") -> float:
        value = score.meta.get("p95LatencySeconds")
        return float("inf") if value is None else float(value)

    frontier: List[str] = []
    for score in scores:
        dominated = False
        for other in scores:
            if other is score:
                continue
            better_or_equal = other.signals[SIGNAL_SUCCESS] >= score.signals[
                SIGNAL_SUCCESS
            ] and latency_of(other) <= latency_of(score)
            strictly_better = other.signals[SIGNAL_SUCCESS] > score.signals[
                SIGNAL_SUCCESS
            ] or latency_of(other) < latency_of(score)
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            name = score.meta.get("scenario")
            if name is not None:
                frontier.append(name)
    return frontier


class ReadinessScore:
    """Weighted readiness assessment of one execution record (or one scenario)."""

    def __init__(
        self,
        signals: Dict[str, float],
        score: float,
        gate_passed: bool,
        gate_threshold: float,
        weights: Dict[str, float],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.signals = signals
        self.score = score
        self.gate_passed = gate_passed
        self.gate_threshold = gate_threshold
        self.weights = weights
        self.meta = meta or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "gatePassed": self.gate_passed,
            "gateThreshold": self.gate_threshold,
            "signals": {name: round(value, 4) for name, value in self.signals.items()},
            "weights": dict(self.weights),
            "meta": self.meta,
        }


class ReadinessReport:
    """Scenario-weighted readiness across many execution records, with frontier."""

    def __init__(
        self,
        scores: Dict[str, ReadinessScore],
        frontier: List[str],
        overall_score: float,
        gate_threshold: float,
        weights: Dict[str, float],
    ) -> None:
        self.scores = scores
        self.frontier = frontier
        self.overall_score = overall_score
        self.gate_threshold = gate_threshold
        self.weights = weights

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overallScore": round(self.overall_score, 4),
            "paretoFrontier": list(self.frontier),
            "gateThreshold": self.gate_threshold,
            "weights": dict(self.weights),
            "scenarioCount": len(self.scores),
            "scenarios": {name: score.to_dict() for name, score in self.scores.items()},
        }


class ReadinessScorer:
    """Configurable readiness scorer implementing the paper's API contract.

    ``score`` assesses a single execution record; ``aggregate`` groups many
    records by scenario and reports per-scenario scores plus the Pareto
    frontier, the paper's headline output.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        latency_budget_s: float = DEFAULT_LATENCY_BUDGET_SECONDS,
        gate_threshold: float = DEFAULT_GATE_THRESHOLD,
        compliance_checker: Optional[Callable[[Dict[str, Any]], Optional[float]]] = None,
        extra_signal_provider: Optional[
            Callable[[Dict[str, Any]], Optional[Dict[str, float]]]
        ] = None,
    ) -> None:
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.latency_budget_s = latency_budget_s
        self.gate_threshold = gate_threshold
        self.compliance_checker = compliance_checker
        self.extra_signal_provider = extra_signal_provider

    def _extra(self, record: Dict[str, Any]) -> Optional[Dict[str, float]]:
        return None if self.extra_signal_provider is None else self.extra_signal_provider(record)

    def score(self, record: Dict[str, Any]) -> ReadinessScore:
        signals = extract_signals(
            record,
            latency_budget_s=self.latency_budget_s,
            compliance_checker=self.compliance_checker,
            extra_signals=self._extra(record),
        )
        weighted = weighted_score(signals, self.weights)
        return ReadinessScore(
            signals,
            weighted,
            weighted >= self.gate_threshold,
            self.gate_threshold,
            self.weights,
        )

    def aggregate(
        self,
        records: Sequence[Dict[str, Any]],
        scenario_key: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> ReadinessReport:
        key_fn = scenario_key if scenario_key is not None else _default_scenario_key
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            groups.setdefault(key_fn(record), []).append(record)

        scores: Dict[str, ReadinessScore] = {}
        for name, records_in_scenario in groups.items():
            executions = [
                execution for record in records_in_scenario for execution in iter_executions(record)
            ]
            times = [
                float(execution["executionTimeInSeconds"])
                for execution in executions
                if execution.get("executionTimeInSeconds") is not None
            ]
            p95 = percentile(times, 95)
            signals = {
                SIGNAL_SUCCESS: _mean(
                    [
                        1.0 if record.get("result") == "success" else 0.0
                        for record in records_in_scenario
                    ]
                ),
                SIGNAL_COMPLIANCE: outcome_compliance(executions),
                SIGNAL_P95_LATENCY: _latency_score(p95, self.latency_budget_s),
            }
            weighted = weighted_score(signals, self.weights)
            scores[name] = ReadinessScore(
                signals,
                weighted,
                weighted >= self.gate_threshold,
                self.gate_threshold,
                self.weights,
                meta={
                    "scenario": name,
                    "count": len(records_in_scenario),
                    "p95LatencySeconds": p95,
                },
            )

        frontier = pareto_frontier(list(scores.values()))
        overall = _mean([score.score for score in scores.values()]) if scores else 0.0
        return ReadinessReport(scores, frontier, overall, self.gate_threshold, self.weights)


def compute_readiness(
    record: Dict[str, Any], scorer: Optional[ReadinessScorer] = None
) -> Dict[str, Any]:
    """Score one execution record for the Journal, never raising.

    This is the entry point used on the journal ``save()`` path: it returns a
    plain JSON-serializable dict and swallows any unexpected error so that a
    readiness failure can never break record persistence.
    """
    try:
        active_scorer = scorer if scorer is not None else ReadinessScorer()
        return active_scorer.score(record).to_dict()
    except Exception as exc:  # persistence must survive any scorer failure
        return {"score": None, "gatePassed": False, "gateThreshold": None, "error": str(exc)}

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

WORKFLOW_CARD_KEY_PREFIX = "wf-card-"

# Mirrors writer.journal.JOURNAL_KEY_PREFIX; defined locally to keep this
# module importable without pulling in the journal's core Config dependency.
JOURNAL_KEY_PREFIX = "wf-journal-"

CARD_TEMPLATE_VERSION = 1

logger = logging.getLogger("workflow_card")


def save_workflow_card(
    journal_key: str,
    journal_entry: Dict[str, Any],
    storage: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Build a Workflow Card from a journal entry and store it under a sibling key.

    The card is a pure addition: the journal entry is saved unchanged, and the
    card lands next to it under ``wf-card-<same suffix>`` so consumers can
    retrieve either one by key prefix. Card generation must never break
    journaling, so failures are logged and swallowed.
    """
    if storage is None:
        from writer.keyvalue_storage import writer_kv_storage as storage
    try:
        card = build_workflow_card(journal_entry)
        storage.save(workflow_card_key(journal_key), card)
        return card
    except Exception:
        logger.exception("Failed to save a Workflow Card for %s", journal_key)
        return None


def build_workflow_card(journal_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Condense a journal entry into a Workflow Card.

    A Workflow Card is a structured summary of a single workflow execution,
    built from the provenance data the journal already records (trigger,
    per-block executions, timings, logs and outcomes). It is designed to be
    read on its own by a human or an LLM, without access to the raw journal
    entry or to the app that produced it.

    Adapted from "Workflow Cards: Structured Summaries of Workflow Executions
    Using Provenance Data" (arXiv:2608.11022).
    """
    block_outputs = _as_dict(journal_entry.get("blockOutputs"))
    steps = [_step(block_id, output) for block_id, output in block_outputs.items()]

    return {
        "cardType": "workflow",
        "templateVersion": CARD_TEMPLATE_VERSION,
        "generatedAt": _utc_now_iso(),
        "sourceEntry": {
            "journalKeyPrefix": JOURNAL_KEY_PREFIX,
            "timestamp": journal_entry.get("timestamp"),
        },
        "workflow": {
            "blueprintId": journal_entry.get("blueprintId"),
            "instanceType": journal_entry.get("instanceType"),
            "result": journal_entry.get("result"),
            "trigger": _trigger(journal_entry.get("trigger")),
        },
        "runtime": _runtime(steps),
        "steps": steps,
        "reproducibility": {
            "triggerPayload": _trigger_payload(journal_entry.get("trigger")),
            "inputs": _inputs(steps),
        },
        "findings": _findings(journal_entry, steps),
    }


def workflow_card_key(journal_key: str) -> str:
    """Return the KV storage key a card is stored under, given its journal key.

    The prefix deliberately differs from ``wf-journal-``: the journal UI fetches
    entries with a substring match on that prefix and assumes every match has the
    journal entry shape, so a card sharing it would break the panel.
    """
    if not journal_key.startswith(JOURNAL_KEY_PREFIX):
        raise ValueError(f"Not a journal key: {journal_key}")
    suffix = journal_key[len(JOURNAL_KEY_PREFIX) :]
    return f"{WORKFLOW_CARD_KEY_PREFIX}{suffix}"


def _step(block_id: str, output: Any) -> Dict[str, Any]:
    output = _as_dict(output)
    component = _as_dict(output.get("component"))
    executions = [
        _as_dict(execution)
        for execution in output.get("executions") or []
        if _as_dict(execution).get("outcome") is not None
    ]
    outcomes = [execution.get("outcome") for execution in executions]

    return {
        "id": block_id,
        "component": {
            "type": component.get("type"),
            "id": component.get("id"),
            "title": component.get("title"),
            "category": component.get("category"),
        },
        "runs": len(executions),
        "outcomes": outcomes,
        "result": executions[0].get("result") if executions else None,
        "durationSeconds": _sum_durations(executions),
        "startedAt": executions[0].get("startedAt") if executions else None,
        "logs": _join_logs(executions),
        "message": _first_message(executions),
    }


def _trigger(trigger: Any) -> Dict[str, Any]:
    trigger = _as_dict(trigger)
    component = _as_dict(trigger.get("component"))
    return {
        "type": trigger.get("type"),
        "event": trigger.get("event"),
        "component": {
            "type": component.get("type"),
            "id": component.get("id"),
            "title": component.get("title"),
        },
    }


def _runtime(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    durations = [step["durationSeconds"] for step in steps if step["durationSeconds"] is not None]
    started = [step["startedAt"] for step in steps if step["startedAt"] is not None]
    return {
        "stepsExecuted": sum(step["runs"] for step in steps),
        "totalDurationSeconds": round(sum(durations), 3) if durations else None,
        "slowestStep": _slowest_step(steps),
        "firstStepStartedAt": min(started) if started else None,
        "lastStepStartedAt": max(started) if started else None,
    }


def _slowest_step(steps: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    timed = [step for step in steps if step["durationSeconds"] is not None]
    if not timed:
        return None
    slowest = max(timed, key=lambda step: step["durationSeconds"] or 0)
    return {
        "id": slowest["id"],
        "title": slowest["component"]["title"],
        "durationSeconds": slowest["durationSeconds"],
    }


def _findings(journal_entry: Dict[str, Any], steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Surface the execution-level signals a reader would otherwise have to dig for.

    These answer the provenance questions the paper found missing from Model and
    Data Cards: what ran, what failed, where the time went, what was logged.
    """
    failed = [
        {"id": step["id"], "title": step["component"]["title"], "message": step["message"]}
        for step in steps
        if "error" in step["outcomes"]
    ]
    logged = [
        {"id": step["id"], "title": step["component"]["title"], "logs": step["logs"]}
        for step in steps
        if step["logs"]
    ]
    return {
        "status": journal_entry.get("result"),
        "failedSteps": failed,
        "stepsWithLogs": logged,
        "retriedSteps": [
            {"id": step["id"], "title": step["component"]["title"], "runs": step["runs"]}
            for step in steps
            if step["runs"] > 1
        ],
    }


def _trigger_payload(trigger: Any) -> Any:
    return _as_dict(trigger).get("payload")


def _inputs(steps: List[Dict[str, Any]]) -> List[Any]:
    """Collect the results feeding the workflow, truncated for readability."""
    return [_truncate(step["result"]) for step in steps if step["result"] is not None]


def _sum_durations(executions: List[Dict[str, Any]]) -> Optional[float]:
    durations = [
        execution["executionTimeInSeconds"]
        for execution in executions
        if isinstance(execution.get("executionTimeInSeconds"), (int, float))
    ]
    return round(sum(durations), 3) if durations else None


def _join_logs(executions: List[Dict[str, Any]]) -> Optional[str]:
    fragments = []
    for execution in executions:
        for field in ("stdout", "logs"):
            value = execution.get(field)
            if value:
                fragments.append(value if isinstance(value, str) else json.dumps(value))
    return "\n".join(fragments) if fragments else None


def _first_message(executions: List[Dict[str, Any]]) -> Optional[str]:
    for execution in executions:
        message = execution.get("message")
        if message:
            return message if isinstance(message, str) else json.dumps(message)
    return None


def _truncate(value: Any, limit: int = 200) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return f"{value[:limit]}... [truncated]"
    if isinstance(value, dict):
        return {k: _truncate(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate(item, limit) for item in value]
    return value


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()

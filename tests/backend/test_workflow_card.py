import contextlib
import json

import pytest
from writer.blocks.writerkeyvaluestorage import WriterKeyValueStorage
from writer.journal import JOURNAL_KEY_PREFIX, JournalRecord
from writer.workflow_card import (
    WORKFLOW_CARD_KEY_PREFIX,
    build_workflow_card,
    workflow_card_key,
)


def _journal_entry() -> dict:
    """A journal entry in the shape JournalRecord.to_dict() emits."""
    return {
        "timestamp": "2026-08-17T10:00:00+00:00",
        "instanceType": "agent",
        "blueprintId": "m4gycroojx6am4cq",
        "trigger": {
            "event": "wf-run-blueprint-via-api",
            "payload": {"proposedSessionId": None},
            "component": {"type": "block", "id": "qfqpqmjdpzuu8fe9", "title": "API alias"},
            "type": "API",
        },
        "blockOutputs": {
            "qfqpqmjdpzuu8fe9": {
                "component": {
                    "type": "blueprints_apitrigger",
                    "id": "qfqpqmjdpzuu8fe9",
                    "title": "API alias",
                    "category": "Triggers",
                },
                "executions": [
                    {
                        "result": {"proposedSessionId": None},
                        "outcome": "trigger",
                        "startedAt": 1.0,
                        "executionTimeInSeconds": 0.1,
                    }
                ],
            },
            "pa448833kc2pis3a": {
                "component": {
                    "type": "blueprints_logmessage",
                    "id": "pa448833kc2pis3a",
                    "title": "Log message",
                    "category": "Other",
                },
                "executions": [
                    {
                        "result": "AAA",
                        "outcome": "success",
                        "startedAt": 1.2,
                        "executionTimeInSeconds": 0.4,
                        "logs": "started",
                    }
                ],
            },
        },
        "result": "success",
        "isRunable": True,
    }


def _journal_record() -> JournalRecord:
    """Build a JournalRecord over a minimal two-node graph."""
    from writer import blocks
    from writer.blueprints import Graph
    from writer.core_ui import Component

    components = [
        Component(
            id="qfqpqmjdpzuu8fe9", type="blueprints_apitrigger", content={}, parentId="bp1"
        ),
        Component(
            id="pa448833kc2pis3a", type="blueprints_logmessage", content={}, parentId="bp1"
        ),
    ]
    return JournalRecord(
        {"context": {"event": "wf-run-blueprint-via-api"}, "payload": {}},
        "Run blueprint - API",
        Graph(components, blocks.base_block.block_map),
    )


class TestWorkflowCardKey:
    def test_key_is_a_sibling_of_the_journal_key(self):
        assert workflow_card_key("wf-journal-a-1234") == "wf-card-a-1234"

    def test_key_is_not_matched_by_the_journal_ui_substring_query(self):
        # The journal panel fetches with key_contains "wf-journal-" and assumes
        # every match is a journal entry, so the card key must not match it.
        assert "wf-journal-" not in workflow_card_key("wf-journal-a-1234")

    def test_key_rejects_non_journal_keys(self):
        with pytest.raises(ValueError):
            workflow_card_key("wf-init-logs-a-1234")


class TestBuildWorkflowCard:
    def test_card_summarizes_execution_level_provenance(self):
        card = build_workflow_card(_journal_entry())

        assert card["cardType"] == "workflow"
        assert card["sourceEntry"]["timestamp"] == "2026-08-17T10:00:00+00:00"
        assert card["workflow"]["blueprintId"] == "m4gycroojx6am4cq"
        assert card["workflow"]["trigger"]["type"] == "API"
        assert card["workflow"]["result"] == "success"

        # Runtime aggregates the timing the journal records per execution.
        assert card["runtime"]["stepsExecuted"] == 2
        assert card["runtime"]["totalDurationSeconds"] == pytest.approx(0.5)
        assert card["runtime"]["slowestStep"]["id"] == "pa448833kc2pis3a"

        # Each step stands on its own, with the component identified.
        by_id = {step["id"]: step for step in card["steps"]}
        assert by_id["pa448833kc2pis3a"]["component"]["type"] == "blueprints_logmessage"
        assert by_id["pa448833kc2pis3a"]["result"] == "AAA"
        assert by_id["pa448833kc2pis3a"]["logs"] == "started"

        # Reproducibility captures what triggered the run and what it consumed.
        assert card["reproducibility"]["triggerPayload"] == {"proposedSessionId": None}
        assert card["reproducibility"]["inputs"] == [{"proposedSessionId": None}, "AAA"]

        # Findings surface failures without digging through every execution.
        assert card["findings"]["status"] == "success"
        assert card["findings"]["failedSteps"] == []

    def test_card_flags_errors_logs_and_retries(self):
        entry = _journal_entry()
        entry["result"] = "error"
        log_block = entry["blockOutputs"]["pa448833kc2pis3a"]
        log_block["executions"][0]["outcome"] = "error"
        log_block["executions"][0]["message"] = "ValueError: bad input"
        log_block["executions"].append(dict(log_block["executions"][0]))

        card = build_workflow_card(entry)

        failed = card["findings"]["failedSteps"]
        assert [step["id"] for step in failed] == ["pa448833kc2pis3a"]
        assert failed[0]["message"] == "ValueError: bad input"
        assert [step["id"] for step in card["findings"]["retriedSteps"]] == ["pa448833kc2pis3a"]
        assert card["findings"]["stepsWithLogs"][0]["logs"].startswith("started")

    def test_card_truncates_long_values(self):
        entry = _journal_entry()
        entry["blockOutputs"]["pa448833kc2pis3a"]["executions"][0]["result"] = "x" * 500

        card = build_workflow_card(entry)

        assert card["reproducibility"]["inputs"][1].startswith("xxx")
        assert len(card["reproducibility"]["inputs"][1]) < 300

    def test_card_is_json_serializable(self):
        payload = json.dumps(build_workflow_card(_journal_entry()))
        assert '"cardType": "workflow"' in payload


class TestJournalSaveHook:
    def test_save_stores_a_card_next_to_the_journal_entry(self, mock_kv_storage):
        with _journal_feature_flag():
            record = _journal_record()
            record.set_result("success")
            record.save()

        keys = mock_kv_storage.get_data_keys()
        assert len(keys) == 2

        journal_key = next(key for key in keys if key.startswith(JOURNAL_KEY_PREFIX))
        card_key = next(key for key in keys if key.startswith(WORKFLOW_CARD_KEY_PREFIX))
        assert card_key == workflow_card_key(journal_key)

        entry = mock_kv_storage.get(journal_key, "data")
        card = mock_kv_storage.get(card_key, "data")

        # The journal entry is unchanged by card generation...
        assert entry["result"] == "success"
        assert entry["isRunable"] is True
        assert "cardType" not in entry

        # ...and the card summarizes that same execution.
        assert card["cardType"] == "workflow"
        assert card["workflow"]["result"] == "success"
        assert card["workflow"]["trigger"]["type"] == "API"
        assert {step["id"] for step in card["steps"]} == set(entry["blockOutputs"].keys())

    def test_save_failure_does_not_break_journaling(self, mock_kv_storage, monkeypatch):
        class _ExplodingStorage:
            def is_accessible(self):
                return True

            def save(self, key, _data):
                if key.startswith(WORKFLOW_CARD_KEY_PREFIX):
                    raise RuntimeError("card storage exploded")
                mock_kv_storage.save(key, _data)

        monkeypatch.setattr("writer.journal.writer_kv_storage", _ExplodingStorage())

        with _journal_feature_flag():
            record = _journal_record()
            record.set_result("success")
            record.save()

        # The card write exploded, but the journal entry itself still landed.
        assert record.result == "success"
        assert [key for key in mock_kv_storage.get_data_keys()]

    def test_card_keys_are_hidden_from_kv_storage_block(
        self, mock_kv_storage, monkeypatch
    ):
        with _journal_feature_flag():
            record = _journal_record()
            record.save()
        mock_kv_storage.save("user_key", {"a": 1})

        monkeypatch.setattr(
            "writer.ai.WriterAIManager.acquire_client",
            lambda custom_httpx_client=None, force_new_client=False: object(),
        )

        import writer.keyvalue_storage as kv_module

        class _FakeKV:
            def __init__(self, client=None):
                pass

            def get_data_keys(self):
                return list(mock_kv_storage._data_storage.keys())

        monkeypatch.setattr(kv_module, "KeyValueStorage", _FakeKV)

        from tests.backend.blocks.conftest import BlockTesterMockBlueprintRunner

        session = _mock_session()
        component = session.add_fake_component(
            {"action": "List keys"}, id="kv1", type="blueprints_writerkeyvaluestorage"
        )
        block = WriterKeyValueStorage(component, BlockTesterMockBlueprintRunner(session), {})
        block.run()

        assert block.outcome == "success"
        # Only the user's own key remains; journal and card keys are internal.
        assert block.result == ["user_key"]


@contextlib.contextmanager
def _journal_feature_flag():
    """Enable the journal flag, as an app's main.py would."""
    from writer.core import Config

    Config.feature_flags.append("journal")
    try:
        yield
    finally:
        Config.feature_flags.remove("journal")


def _mock_session():
    from tests.backend.blocks.conftest import BlockTesterMockSession

    return BlockTesterMockSession()

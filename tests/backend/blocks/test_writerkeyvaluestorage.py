import writer.ai
from writer.blocks.writerkeyvaluestorage import WriterKeyValueStorage


def test_save_memory_distills_conversation_and_persists(monkeypatch, session, runner, fake_client):
    canned = (
        '{"task_specifications": ["Build a revenue dashboard"], '
        '"data_schemas": ["orders(id, total, date)"], '
        '"tool_configurations": ["GET /api/orders"], '
        '"output_constraints": ["USD, two decimals"]}'
    )

    def fake_complete(prompt, config):
        # The extractor must hand the model the selective-memory prompt.
        assert "selective persistent memory" in prompt
        return canned

    monkeypatch.setattr("writer.ai.complete", fake_complete)

    saved = {}

    def fake_save(self, key, data):
        saved["key"] = key
        saved["data"] = data
        return {"key": key}

    monkeypatch.setattr("writer.keyvalue_storage.KeyValueStorage.save", fake_save)

    messages = (
        '[{"role": "user", "content": "Build a revenue dashboard"}, '
        '{"role": "assistant", "content": "Sure, let me reason step by step..."}]'
    )
    component = session.add_fake_component(
        {"action": "Save memory", "key": "session-42", "messages": messages}
    )
    block = WriterKeyValueStorage(component, runner, {})
    block.run()

    # The curated blob -- not the raw transcript -- is what gets persisted.
    assert block.outcome == "success"
    assert saved["key"] == "session-42"
    assert saved["data"]["task_specifications"] == ["Build a revenue dashboard"]
    assert saved["data"]["tool_configurations"] == ["GET /api/orders"]
    # Reasoning traces are dropped: the assistant's narration never lands here.
    assert "step by step" not in str(saved["data"])


def test_save_memory_empty_conversation(monkeypatch, session, runner, fake_client):
    complete_calls = []
    saved = {}

    def fake_complete(prompt, config):
        complete_calls.append(prompt)
        return "{}"

    monkeypatch.setattr("writer.ai.complete", fake_complete)

    def fake_save(self, key, data):
        saved["data"] = data
        return {"key": key}

    monkeypatch.setattr("writer.keyvalue_storage.KeyValueStorage.save", fake_save)

    component = session.add_fake_component(
        {"action": "Save memory", "key": "empty", "messages": "[]"}
    )
    block = WriterKeyValueStorage(component, runner, {})
    block.run()

    # No messages means no model call: the empty blob is returned directly.
    assert block.outcome == "success"
    assert block.result == {"key": "empty"}
    assert complete_calls == []
    assert saved["data"] == {
        "task_specifications": [],
        "data_schemas": [],
        "tool_configurations": [],
        "output_constraints": [],
    }

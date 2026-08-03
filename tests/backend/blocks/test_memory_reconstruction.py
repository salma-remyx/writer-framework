import pytest

import writer.ai
import writer.blocks.memory_reconstruction as memory_reconstruction
from writer.blocks.writerchatreply import WriterChatReply


class _ReplyConversation(writer.ai.Conversation):
    """Minimal Conversation stand-in: records added messages, returns a fixed reply."""

    def __init__(self):
        super().__init__()

    def complete(self, tools=None):
        return {"role": "assistant", "content": "The bat is next to the grill."}

    def stream_complete(self, tools=None):
        yield {"role": "assistant", "content": "The bat is next to the grill."}


@pytest.fixture
def conversation():
    return _ReplyConversation()


def _capture_reconstruction(captured):
    def _fake(prompt, model_id=None):
        captured["prompt"] = prompt
        captured["model_id"] = model_id
        return "RECONSTRUCTED: check the area near the grill for the bat"

    return _fake


def test_retrieved_memory_is_reconstructed_not_replayed(
    session, runner, conversation, fake_client, monkeypatch
):
    """The chat-reply block reconstructs retrieved memory into context-grounded
    guidance and injects it before replying, instead of replaying it verbatim."""
    captured = {}
    monkeypatch.setattr(memory_reconstruction, "_invoke_model", _capture_reconstruction(captured))

    conversation.add("user", "where's the bat?")
    session.session_state["convo"] = conversation
    component = session.add_fake_component(
        {
            "conversationStateElement": "convo",
            "generateReply": "yes",
            "useStreaming": "no",
            "initModelId": "palmyra-x5",
            "retrievedMemory": "The bat is usually near the grill at night.",
        }
    )

    block = WriterChatReply(component, runner, {})
    block.run()

    # The reconstruction saw both the current situation and the raw memory.
    assert "where's the bat?" in captured["prompt"]
    assert "near the grill at night" in captured["prompt"]
    assert captured["model_id"] == "palmyra-x5"

    # Reconstructed guidance is injected as a system message before the reply.
    system_messages = [m for m in conversation.messages if m.get("role") == "system"]
    assert any("RECONSTRUCTED" in (m.get("content") or "") for m in system_messages)

    # Raw memory is never replayed verbatim into the conversation.
    contents = [m.get("content") or "" for m in conversation.messages]
    assert all("usually near the grill at night" not in c for c in contents)


def test_no_memory_leaves_conversation_untouched(
    session, runner, conversation, fake_client, monkeypatch
):
    """Without retrieved memory the block behaves exactly as before: no
    reconstruction call and no injected system message."""
    called = {"count": 0}

    def _fail_if_called(prompt, model_id=None):
        called["count"] += 1
        return "should not happen"

    monkeypatch.setattr(memory_reconstruction, "_invoke_model", _fail_if_called)

    conversation.add("user", "where's the bat?")
    session.session_state["convo"] = conversation
    component = session.add_fake_component(
        {
            "conversationStateElement": "convo",
            "generateReply": "yes",
            "useStreaming": "no",
        }
    )

    block = WriterChatReply(component, runner, {})
    block.run()

    assert called["count"] == 0
    assert all(m.get("role") != "system" for m in conversation.messages)
    # The reply is still produced.
    assert conversation.messages[-1].get("role") == "assistant"


def test_reconstruct_memory_returns_empty_for_no_input():
    """The shaping helpers short-circuit cleanly when there is nothing to
    reconstruct, so callers inject nothing rather than an empty message."""
    empty = writer.ai.Conversation()
    assert memory_reconstruction.reconstruct_memory(None, empty) == ""
    assert memory_reconstruction.reconstruct_memory("", empty) == ""
    assert memory_reconstruction.reconstruct_memory("  ", empty) == ""

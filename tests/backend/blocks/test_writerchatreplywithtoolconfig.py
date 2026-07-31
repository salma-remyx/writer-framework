import json

import pytest
import writer.ai
from writer.blocks.tool_gating import latest_user_intent, select_relevant_tools
from writer.blocks.writerchatreplywithtoolconfig import WriterChatReplyWithToolConfig

# Two MCP tools: one clearly relevant to the turn's intent ("search the
# knowledge base ..."), one irrelevant ("send email"). Gating should keep
# only the relevant one when maxTools caps the set.
TOOLS_CONFIG = {
    "search_knowledge_base": {
        "type": "mcp",
        "appId": "app1",
        "functionName": "search_kb",
        "function": {
            "description": "Search the knowledge base for documents and articles.",
            "parameters": {"query": {"type": "string", "description": "The search query"}},
        },
    },
    "send_email": {
        "type": "mcp",
        "appId": "app2",
        "functionName": "send_email",
        "function": {
            "description": "Send an email to a recipient.",
            "parameters": {
                "to": {"type": "string", "description": "Recipient address"},
                "body": {"type": "string"},
            },
        },
    },
}


class ToolCapturingConversation(writer.ai.Conversation):
    """Conversation stub that records the tool schemas handed to complete()."""

    def __init__(self):
        super().__init__()
        self.captured_tools = None

    def complete(self, tools=None):
        self.captured_tools = tools
        return {"role": "assistant", "content": "done"}


@pytest.fixture
def conversation():
    return ToolCapturingConversation()


def test_gating_narrows_injected_toolset(session, runner, conversation, fake_client):
    """With gating enabled, only the intent-relevant tool reaches complete()."""
    conversation.add("user", "search the knowledge base for documents about bats")
    session.session_state["convo"] = conversation
    component = session.add_fake_component(
        {
            "conversationStateElement": "convo",
            "generateReply": "yes",
            "useStreaming": "no",
            "tools": json.dumps(TOOLS_CONFIG),
            "toolConfig": json.dumps({"toolGating": {"enabled": True, "maxTools": 1}}),
        }
    )

    WriterChatReplyWithToolConfig(component, runner, {}).run()

    captured = conversation.captured_tools
    assert captured is not None
    assert len(captured) == 1
    assert captured[0].get("name") == "search_knowledge_base"


def test_gating_disabled_keeps_all_tools(session, runner, conversation, fake_client):
    """Without gating enabled, the full toolset is injected (contract preserved)."""
    conversation.add("user", "search the knowledge base for documents about bats")
    session.session_state["convo"] = conversation
    component = session.add_fake_component(
        {
            "conversationStateElement": "convo",
            "generateReply": "yes",
            "useStreaming": "no",
            "tools": json.dumps(TOOLS_CONFIG),
            "toolConfig": json.dumps({}),
        }
    )

    WriterChatReplyWithToolConfig(component, runner, {}).run()

    captured = conversation.captured_tools
    assert captured is not None
    assert {tool.get("name") for tool in captured} == {
        "search_knowledge_base",
        "send_email",
    }


def test_select_relevant_tools_keeps_all_when_no_overlap():
    """No positive overlap signal means we never truncate to an arbitrary prefix."""
    tools = [
        {"name": "send_email", "description": "Send an email.", "parameters": {}},
        {"name": "schedule_meeting", "description": "Book a meeting.", "parameters": {}},
    ]
    config = {"toolGating": {"enabled": True, "maxTools": 1}}

    selected = select_relevant_tools(tools, "translate this to french", config)

    assert selected == tools


def test_select_relevant_tools_disabled_is_noop():
    tools = [{"name": "a", "description": "alpha", "parameters": {}}]
    assert select_relevant_tools(tools, "alpha", {}) == tools
    assert select_relevant_tools(tools, "alpha", {"toolGating": {}}) == tools


def test_latest_user_intent_handles_multimodal_content():
    messages = [
        {"role": "user", "content": "first turn"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe "},
                {"type": "image_url", "image_url": {"url": "data:..."}},
                {"type": "text", "text": "this chart"},
            ],
        },
    ]

    intent = latest_user_intent(messages)
    assert "describe" in intent
    assert "this chart" in intent
    assert latest_user_intent([]) == ""
    assert latest_user_intent([{"role": "assistant", "content": "hi"}]) == ""

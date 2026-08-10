"""Integration tests for the tool-calling block's Tool Attention gating.

These exercise the wiring inside ``WriterToolCalling.run`` (the non-new
call-site module): when tool gating is on, only the intent-relevant tools
plus the control tool reach ``conversation.complete``; when it is off, the
full tool set is injected as before.
"""

import json

import writer.ai
from writer.blocks.writertoolcalling import WriterToolCalling


class _CapturingConversation:
    """Stand-in for ``writer.ai.Conversation`` that records the tools it
    receives and short-circuits ``complete`` without executing tool calls."""

    def __init__(self):
        self.messages = []
        self.received_tools = None

    def __iadd__(self, message):
        self.messages.append(message)
        return self

    def complete(self, tools=None, config=None):
        self.received_tools = tools
        return {"role": "assistant", "content": "done"}


def _tools_payload():
    return json.dumps(
        {
            "bat_locator": {
                "type": "function",
                "description": "Locate bats by color and species.",
                "parameters": {
                    "color": {"type": "string", "description": "bat color"},
                    "species": {"type": "string", "description": "bat species"},
                },
            },
            "weather_forecast": {
                "type": "function",
                "description": "Get the weather forecast for a city.",
                "parameters": {"city": {"type": "string", "description": "city name"}},
            },
            "currency_converter": {
                "type": "function",
                "description": "Convert between currencies using exchange rates.",
                "parameters": {"amount": {"type": "number", "description": "amount"}},
            },
        }
    )


def _tool_names(tools):
    return [t.get("name") for t in tools]


def test_gating_passes_only_relevant_subset(monkeypatch, session, runner, fake_client):
    conversation = _CapturingConversation()
    monkeypatch.setattr(writer.ai, "Conversation", lambda: conversation)

    component = session.add_fake_component(
        {
            "prompt": "Find me a brown bat.",
            "toolGating": "yes",
            "maxToolsPerTurn": "2",  # budget for 1 scored tool + the control tool
            "maxIterations": "1",
            "tools": _tools_payload(),
        }
    )
    block = WriterToolCalling(component, runner, {})
    block.run()

    names = _tool_names(conversation.received_tools)
    # Control tool is always present so the ReAct loop can finalize.
    assert "disclose_reasoning" in names
    # The on-topic tool surfaced; the off-topic ones were gated out.
    assert "bat_locator" in names
    assert "weather_forecast" not in names
    assert "currency_converter" not in names
    assert block.outcome == "success"


def test_gating_off_passes_full_tool_set(monkeypatch, session, runner, fake_client):
    conversation = _CapturingConversation()
    monkeypatch.setattr(writer.ai, "Conversation", lambda: conversation)

    component = session.add_fake_component(
        {
            "prompt": "Find me a brown bat.",
            "toolGating": "no",
            "maxIterations": "1",
            "tools": _tools_payload(),
        }
    )
    block = WriterToolCalling(component, runner, {})
    block.run()

    names = _tool_names(conversation.received_tools)
    # Eager behavior is preserved: every tool reaches the model.
    assert "bat_locator" in names
    assert "weather_forecast" in names
    assert "currency_converter" in names
    assert "disclose_reasoning" in names

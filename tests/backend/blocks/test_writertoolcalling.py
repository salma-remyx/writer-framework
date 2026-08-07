import json
from typing import List

from writer.ai import Conversation
from writer.blocks.tool_gate import gate_tools
from writer.blocks.writertoolcalling import WriterToolCalling


def _make_mock_conversation():
    """Build a Conversation subclass that records injected tools.

    The Tool calling block instantiates ``writer.ai.Conversation`` itself, so
    the test patches that name with this subclass. Subclassing (rather than
    replacing with a bare callable) keeps the real class's nested ``Message``
    type resolvable inside ``Conversation.__iadd__``.
    """
    instances: List[Conversation] = []

    class _MockConversation(Conversation):
        def __init__(self):
            super().__init__()
            self.injected_tools = None
            instances.append(self)

        def complete(self, tools=None, config=None):
            if self.injected_tools is None:
                self.injected_tools = tools or []
            return {"role": "assistant", "content": "Final answer."}

    return _MockConversation, instances


_TOOLS_JSON = json.dumps(
    {
        "weather_forecast": {
            "type": "function",
            "description": "Get the current weather forecast, temperature, and rain probability for a city.",
            "parameters": {
                "city": {
                    "type": "string",
                    "description": "The city to get the weather for.",
                }
            },
        },
        "calculator": {
            "type": "function",
            "description": "Perform arithmetic calculations: add, subtract, multiply, divide numbers.",
            "parameters": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression to evaluate.",
                }
            },
        },
    }
)


def _run_block(session, runner, monkeypatch, content):
    mock_cls, instances = _make_mock_conversation()
    monkeypatch.setattr("writer.ai.Conversation", mock_cls)

    component = session.add_fake_component(content)
    block = WriterToolCalling(component, runner, {})
    block.run()
    return block, instances[0]


def test_gating_injects_only_relevant_tools(session, runner, fake_client, monkeypatch):
    block, mock = _run_block(
        session,
        runner,
        monkeypatch,
        {
            "prompt": "What's the weather in Paris?",
            "tools": _TOOLS_JSON,
            "useToolGating": "yes",
            "maxIterations": "1",
        },
    )
    names = [t.get("name") for t in mock.injected_tools]
    assert "weather_forecast" in names  # query-relevant -> kept
    assert "calculator" not in names  # irrelevant -> dropped, cuts the tools tax
    assert "disclose_reasoning" in names  # control tool -> always kept
    assert block.outcome == "success"


def test_gating_off_injects_all_tools(session, runner, fake_client, monkeypatch):
    block, mock = _run_block(
        session,
        runner,
        monkeypatch,
        {
            "prompt": "What's the weather in Paris?",
            "tools": _TOOLS_JSON,
            # useToolGating defaults to "no" -> default behaviour unchanged
            "maxIterations": "1",
        },
    )
    names = [t.get("name") for t in mock.injected_tools]
    assert "weather_forecast" in names
    assert "calculator" in names
    assert "disclose_reasoning" in names
    assert block.outcome == "success"


def test_gating_keeps_highest_scoring_under_cap(session, runner, fake_client, monkeypatch):
    tools_json = json.dumps(
        {
            "weather_forecast": {
                "type": "function",
                "description": "weather forecast temperature rain",
            },
            "climate_outlook": {
                "type": "function",
                "description": "weather outlook",
            },
            "calculator": {
                "type": "function",
                "description": "arithmetic add subtract multiply divide",
            },
        }
    )
    block, mock = _run_block(
        session,
        runner,
        monkeypatch,
        {
            "prompt": "weather forecast",
            "tools": tools_json,
            "useToolGating": "yes",
            "maxGatedTools": "1",
            "maxIterations": "1",
        },
    )
    names = [t.get("name") for t in mock.injected_tools]
    assert "weather_forecast" in names  # higher overlap -> wins the single slot
    assert "climate_outlook" not in names  # capped out
    assert "calculator" not in names  # irrelevant
    assert "disclose_reasoning" in names  # control tool -> always kept
    assert block.outcome == "success"


def test_gate_tools_fails_open_when_nothing_matches():
    tools = [
        {"name": "calculator", "description": "arithmetic add subtract"},
        {"name": "disclose_reasoning", "description": "signal completion"},
    ]
    # No tool shares a token with the query -> keep all (never tool-less).
    kept = gate_tools("translate this poem to french", tools)
    kept_names = [t.get("name") for t in kept]
    assert "calculator" in kept_names
    assert "disclose_reasoning" in kept_names

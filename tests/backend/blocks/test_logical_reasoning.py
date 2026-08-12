"""Tests for the deterministic logical-reasoning capability and its MCP wiring.

The first tests cover the engine directly; the last test exercises the
integration point -- the ``logic_reasoning`` dispatch wired into the
existing ``WriterChatReplyWithToolConfig`` MCP block.
"""

import writer.logical_reasoning as lr
from writer.blocks.writerchatreplywithtoolconfig import WriterChatReplyWithToolConfig

# A small compliance-style knowledge base reused across tests: an employee
# in the "security" role may access the server room.
FACTS = [
    ["employee", "alice", "security"],
    ["employee", "bob", "engineering"],
]
RULES = [
    {"head": ["can_access", "?u", "server_room"], "body": [["employee", "?u", "security"]]},
]

# Recursive transitive closure: parent* -> ancestor.
RECURSIVE_FACTS = [["parent", "alice", "bob"], ["parent", "bob", "carol"]]
RECURSIVE_RULES = [
    {"head": ["ancestor", "?x", "?y"], "body": [["parent", "?x", "?y"]]},
    {
        "head": ["ancestor", "?x", "?y"],
        "body": [["parent", "?x", "?z"], ["ancestor", "?z", "?y"]],
    },
]


def test_reasoner_satisfies_with_solution_and_proof():
    satisfied, solutions, proof = lr.Reasoner(facts=FACTS, rules=RULES).ask(
        ["can_access", "?u", "server_room"]
    )

    assert satisfied is True
    assert solutions == [{"?u": "alice"}]
    assert proof is not None
    assert any("rule" in step.source for step in proof)
    # The resolved goal carries the binding found by the proof.
    assert proof[0].goal == ["can_access", "alice", "server_room"]


def test_reasoner_negative_when_no_proof():
    satisfied, solutions, proof = lr.Reasoner(facts=FACTS, rules=RULES).ask(
        ["can_access", "bob", "server_room"]
    )

    assert satisfied is False
    assert solutions == []
    assert proof is None


def test_reasoner_recurses_without_variable_capture():
    # The ancestor rule reuses the ?x/?y variable names at every depth, yet
    # both ancestors of carol must be found (no variable capture between
    # nested rule applications).
    satisfied, solutions, _ = lr.Reasoner(
        facts=RECURSIVE_FACTS, rules=RECURSIVE_RULES
    ).ask(["ancestor", "?a", "carol"])

    assert satisfied is True
    assert {"?a": "bob"} in solutions
    assert {"?a": "alice"} in solutions


def test_run_logic_reasoning_returns_text_report():
    report = lr.run_logic_reasoning(
        query=["can_access", "?u", "server_room"], facts=FACTS, rules=RULES
    )

    assert "Answer: YES" in report
    assert "?u = alice" in report
    assert "Derivation:" in report
    assert "can_access(alice, server_room)" in report


def test_run_logic_reasoning_handles_bad_input():
    assert "missing required argument" in lr.run_logic_reasoning()
    assert "logic_reasoning error" in lr.run_logic_reasoning(query="not-a-list")


def test_mcp_dispatch_routes_logic_tool_locally(session, runner, fake_client):
    """The MCP dispatch in WriterChatReplyWithToolConfig routes the reserved
    ``logic_reasoning`` tool to the deterministic local engine, returning the
    answer together with a derivation instead of the hardcoded mock."""
    component = session.add_fake_component({})
    block = WriterChatReplyWithToolConfig(component, runner, {})

    tool = block._make_callable("logic", "local-app", "logic_reasoning")
    result = tool(query=["can_access", "?u", "server_room"], facts=FACTS, rules=RULES)

    assert "Answer: YES" in result
    assert "alice" in result
    assert "Derivation:" in result

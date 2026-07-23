import pytest

from writer.blocks.writertoolcalling import WriterToolCalling
from writer.human_authorization import (
    AuthorizationDecision,
    AuthorizationDenied,
    AuthorizationRequest,
    using_authorizer,
)


def _make_tool_calling_block(session, runner, execution_environment):
    """Build a WriterToolCalling block whose tool branch is intercepted."""
    component = session.add_fake_component({"prompt": "Do something useful."})
    return WriterToolCalling(component, runner, execution_environment)


def test_tool_call_runs_when_authorized(session, runner, fake_client):
    """With no authorizer registered the default policy approves the call.

    The tool branch executes and the decision is recorded on the trace so
    it is observable through the execution logging system.
    """
    branch_calls = []

    def stub_run_branch(*args, **kwargs):
        branch_calls.append(args)
        return "branch result"

    runner.run_branch = stub_run_branch

    block = _make_tool_calling_block(session, runner, {"trace": []})
    tool = block._make_callable("lookup_record")

    result = tool(query="abc")

    # The external tool branch was actually invoked.
    assert len(branch_calls) == 1
    assert result is not None
    # The authorization decision is observable on the agent's trace.
    trace = block.execution_environment["trace"]
    auth_entries = [entry for entry in trace if entry.get("type") == "authorization"]
    assert len(auth_entries) == 1
    assert auth_entries[0]["decision"] == "approved"
    assert auth_entries[0]["action"] == "lookup_record"


def test_tool_call_denied_interrupts_before_execution(session, runner, fake_client):
    """A denying authorizer halts the task and never invokes the tool branch.

    This is the immediate-task-interruption capability: the denied
    AuthorizationDecision propagates as AuthorizationDenied and the
    external execution never happens, while the denial stays observable.
    """
    branch_calls = []

    def stub_run_branch(*args, **kwargs):
        branch_calls.append(args)
        return "should not be reached"

    runner.run_branch = stub_run_branch

    def deny_destructive(request: AuthorizationRequest) -> AuthorizationDecision:
        return AuthorizationDecision(
            approved=False, reason="Destructive action blocked by operator."
        )

    block = _make_tool_calling_block(session, runner, {"trace": []})
    tool = block._make_callable("delete_record")

    with using_authorizer(deny_destructive):
        with pytest.raises(AuthorizationDenied):
            tool(record_id=42)

    # The tool branch was never reached.
    assert branch_calls == []
    # The denial is still recorded on the trace for observability.
    trace = block.execution_environment["trace"]
    auth_entries = [entry for entry in trace if entry.get("type") == "authorization"]
    assert len(auth_entries) == 1
    assert auth_entries[0]["decision"] == "denied"
    assert auth_entries[0]["action"] == "delete_record"
    assert auth_entries[0]["reason"] == "Destructive action blocked by operator."


def test_tool_call_selective_authorizer_approves_safe_actions(session, runner, fake_client):
    """A policy-based authorizer can approve some actions and deny others."""

    branch_calls = []

    def stub_run_branch(*args, **kwargs):
        branch_calls.append(args)
        return "ok"

    runner.run_branch = stub_run_branch

    blocked = {"delete_record", "drop_table"}

    def guard(request: AuthorizationRequest) -> AuthorizationDecision:
        if request.action in blocked:
            return AuthorizationDecision(
                approved=False, reason=f"{request.action} not on the allow-list."
            )
        return AuthorizationDecision(approved=True, reason="On the allow-list.")

    block = _make_tool_calling_block(session, runner, {"trace": []})
    safe_tool = block._make_callable("lookup_record")
    unsafe_tool = block._make_callable("drop_table")

    with using_authorizer(guard):
        assert safe_tool(query="abc") is not None
        with pytest.raises(AuthorizationDenied):
            unsafe_tool(table="users")

    assert len(branch_calls) == 1
    trace = block.execution_environment["trace"]
    decisions = {
        entry["action"]: entry["decision"]
        for entry in trace
        if entry.get("type") == "authorization"
    }
    assert decisions == {"lookup_record": "approved", "drop_table": "denied"}

"""Human-in-the-loop authorization for agent tool calls.

Adapted from the Agent-Client Protocol (ACP) layer described in
"Human-Robot Interaction in GenAI Architectures via the Agent-Client
Protocol" (arXiv:2607.14919). That paper separates the *interface-agent*
link -- where a human collaborates with the deliberative agent -- from
the *agent-execution* link (MCP-style tool calls) and shows that placing
an authorization contract on the interface-agent link enables three
collaborative capabilities: real-time observability, explicit human
authorization, and immediate task interruption.

This module ports that contract onto the Writer tool-calling agent. The
agent-execution boundary is the moment the ReAct agent invokes an
external tool; gating it there is the target-native equivalent of the
ACP authorization contract:

* :func:`request_authorization` is called immediately before a tool
  branch runs.
* Every request is recorded on the agent's execution ``trace`` so the
  decision is observable through the existing Journal / execution
  logging system (real-time observability).
* A registered authorizer returns the human verdict (explicit human
  authorization). The default policy auto-approves, so existing
  workflows are unchanged until an authorizer is opted in.
* A denied request raises :class:`AuthorizationDenied`, which unwinds
  the agent's multi-step loop (immediate task interruption).

This is an *adapted* port: the paper's full ACP transport -- the
three-layer topology, heterogeneous UI clients, and the physical-robot
evaluation -- is intentionally out of scope for a single-PR change. What
is kept at full fidelity is the core mechanism: an authorization gate on
the agent->execution boundary that is observable, human-overridable, and
able to halt the task.
"""

import contextlib
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class AuthorizationRequest:
    """A proposed agent action that requires a human decision.

    In ACP terms this is the payload an interface client sends across the
    interface-agent link; here it describes a tool the agent is about to
    invoke at the agent-execution boundary.
    """

    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class AuthorizationDecision:
    """The human's verdict on an :class:`AuthorizationRequest`."""

    approved: bool
    reason: str = ""


class AuthorizationDenied(Exception):
    """Raised to interrupt the agent task when an action is not approved.

    Carries the originating :class:`AuthorizationRequest` and the
    :class:`AuthorizationDecision` so callers (and the Journal) can
    surface *why* the task was halted.
    """

    def __init__(self, request: AuthorizationRequest, decision: AuthorizationDecision):
        self.request = request
        self.decision = decision
        super().__init__(
            f"Authorization denied for action '{request.action}': "
            f"{decision.reason or 'no reason given'}"
        )


Authorizer = Callable[[AuthorizationRequest], AuthorizationDecision]


def allow_all_authorizer(_request: AuthorizationRequest) -> AuthorizationDecision:
    """Default policy: approve every action with no human in the loop.

    Keeps existing workflows unchanged; an authorization trace entry is
    still recorded so the action remains observable.
    """

    return AuthorizationDecision(approved=True, reason="Auto-approved (no authorizer registered).")


def deny_all_authorizer(_request: AuthorizationRequest) -> AuthorizationDecision:
    """Convenience policy: deny every action. Useful for fail-closed guards."""

    return AuthorizationDecision(approved=False, reason="Denied by fail-closed policy.")


_current_authorizer: ContextVar[Authorizer] = ContextVar(
    "writer_human_authorization_authorizer", default=allow_all_authorizer
)


def get_authorizer() -> Authorizer:
    """Return the authorizer active for the current context."""

    return _current_authorizer.get()


def set_authorizer(authorizer: Authorizer) -> Token:
    """Register a human-in-the-loop authorizer for the current context.

    Returns a token to pass to :func:`reset_authorizer`. Prefer
    :func:`using_authorizer` so the override is always reverted.
    """

    return _current_authorizer.set(authorizer)


def reset_authorizer(token: Token) -> None:
    """Restore the authorizer that preceded a :func:`set_authorizer` call."""

    _current_authorizer.reset(token)


@contextlib.contextmanager
def using_authorizer(authorizer: Authorizer):
    """Context manager that activates ``authorizer`` for its block."""

    token = set_authorizer(authorizer)
    try:
        yield
    finally:
        reset_authorizer(token)


def request_authorization(
    action: str,
    parameters: Optional[Dict[str, Any]] = None,
    execution_environment: Optional[Dict[str, Any]] = None,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> AuthorizationDecision:
    """Gate an agent action behind an authorization decision.

    Records the request and its outcome on the agent's execution
    ``trace`` (the existing observability surface) and raises
    :class:`AuthorizationDenied` when the action is not approved, which
    halts the surrounding multi-step workflow.
    """

    request = AuthorizationRequest(
        action=action,
        parameters=dict(parameters or {}),
        context=context,
    )

    decision = get_authorizer()(request)
    _record_trace(execution_environment, request, decision)

    if not decision.approved:
        raise AuthorizationDenied(request, decision)
    return decision


def _record_trace(
    execution_environment: Optional[Dict[str, Any]],
    request: AuthorizationRequest,
    decision: AuthorizationDecision,
) -> None:
    """Append an authorization entry to the agent's execution trace, if present."""

    if not isinstance(execution_environment, dict):
        return
    trace = execution_environment.get("trace")
    if not isinstance(trace, list):
        return
    trace.append(
        {
            "type": "authorization",
            "time": time.time(),
            "action": request.action,
            "parameters": request.parameters,
            "decision": "approved" if decision.approved else "denied",
            "reason": decision.reason,
        }
    )

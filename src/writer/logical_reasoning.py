"""Deterministic logical reasoning over Horn-clause knowledge bases.

This module gives Writer Framework agents a deterministic, Prolog-free
reasoning backend. It is adapted from **Euclid-MCP** (arXiv:2607.21412),
an MCP server that exposes deterministic SWI-Prolog reasoning plus
auditable derivation logs over a tool interface an LLM client can call.

The paper's **core mechanism** is kept at full fidelity: Horn-clause
inference that returns exact yes/no answers together with a full proof
trace, addressed over the same tool-shaped request/response surface the
framework's MCP dispatch already routes. The **auxiliary** pieces are
substituted with target-native equivalents (an adapted port, Mode 2):

  * SWI-Prolog backend -> a pure-Python backward-chaining resolver, so the
    capability runs anywhere the framework does with no external runtime.
  * The paper's multi-tool translate/run/inspect/repair loop -> a single
    deterministic ``run_logic_reasoning`` entrypoint that returns the answer
    together with the human-readable derivation log (the "inspect"
    affordance is preserved as the returned proof trace).
  * The paper's standalone IT-security/compliance benchmark suite -> out of
    scope here; evaluation belongs in a downstream PR.

Intermediate representation
--------------------------
A *literal* is a list ``[predicate, arg1, ...]``. The predicate is a
constant string; arguments are constants (any value, e.g. a plain string,
number or bool) or *variables* written with a leading ``?`` (for example
``?user``). The ``?`` convention is explicit and easy for an LLM to
generate -- the property the paper values in its engine-agnostic
``Euclid-IR``. A *fact* is a ground literal asserted unconditionally. A
*rule* is ``{"head": <literal>, "body": [<literal>, ...]}`` read as
"head if body".

Because the returned derivation log travels back through the MCP tool
result into the conversation and the block result, it surfaces in the
Journal / execution log -- serving the transparency/audit motivation
behind the reverted knowledge-graph-citations work without depending on
the hosted SDK's sources field.

Resolution is bounded by a recursion-depth and a step budget so recursive
rule sets terminate (this is depth-bounded search, not full tabling).
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

Var = str
Literal = List[Any]
Rule = Dict[str, Any]

_VAR_PREFIX = "?"


# --------------------------------------------------------------------------- #
# Term / literal helpers
# --------------------------------------------------------------------------- #
def _is_var(term: Any) -> bool:
    return isinstance(term, str) and term.startswith(_VAR_PREFIX)


def _walk(term: Any, bindings: Dict[Var, Any]) -> Any:
    """Follow the binding chain until a non-variable term (or unbound var)."""
    while _is_var(term) and term in bindings:
        term = bindings[term]
    return term


def _unify_term(x: Any, y: Any, bindings: Dict[Var, Any]) -> Optional[Dict[Var, Any]]:
    """Unify two terms, mutating and returning ``bindings`` (None on failure)."""
    x = _walk(x, bindings)
    y = _walk(y, bindings)
    if _is_var(x) and _is_var(y):
        if x == y:
            return bindings
        bindings[x] = y
        return bindings
    if _is_var(x):
        bindings[x] = y
        return bindings
    if _is_var(y):
        bindings[y] = x
        return bindings
    return bindings if x == y else None


def _unify_literals(a: Literal, b: Literal, bindings: Dict[Var, Any]) -> Optional[Dict[Var, Any]]:
    if len(a) != len(b):
        return None
    current = bindings
    for left, right in zip(a, b):
        current = _unify_term(left, right, current)
        if current is None:
            return None
    return current


def _variables_in(literal: Literal) -> Iterable[Var]:
    return (term for term in literal if _is_var(term))


def _resolve_literal(literal: Literal, bindings: Dict[Var, Any]) -> Literal:
    return [_walk(term, bindings) for term in literal]


def _render_term(term: Any) -> str:
    return str(term)


def _render_literal(literal: Literal) -> str:
    if not literal:
        return "()"
    pred = _render_term(literal[0])
    if len(literal) == 1:
        return pred
    return f"{pred}(" + ", ".join(_render_term(arg) for arg in literal[1:]) + ")"


def _render_rule(rule: Rule) -> str:
    body = ", ".join(_render_literal(literal) for literal in rule["body"]) or "true"
    return _render_literal(rule["head"]) + " :- " + body


# --------------------------------------------------------------------------- #
# Normalisation of LLM-generated input
# --------------------------------------------------------------------------- #
def _normalize_literal(literal: Any) -> Literal:
    if isinstance(literal, dict):
        predicate = literal.get("predicate")
        if predicate is None:
            raise ValueError(f"Literal object missing 'predicate': {literal!r}")
        return [predicate, *literal.get("args", [])]
    if isinstance(literal, (list, tuple)):
        if not literal:
            raise ValueError("Empty literal")
        return list(literal)
    raise ValueError(f"Invalid literal: {literal!r}")


def _normalize_body(body: Any) -> List[Literal]:
    if not body:
        return []
    if isinstance(body, dict):
        return [_normalize_literal(body)]
    if isinstance(body, (list, tuple)):
        first = body[0]
        if isinstance(first, (list, tuple, dict)):
            return [_normalize_literal(item) for item in body]
        return [_normalize_literal(list(body))]
    raise ValueError(f"Invalid rule body: {body!r}")


def _normalize_query(query: Any) -> List[Literal]:
    """Return the query as a conjunction (list) of literals."""
    if isinstance(query, dict):
        return [_normalize_literal(query)]
    if isinstance(query, (list, tuple)) and query:
        first = query[0]
        if isinstance(first, (list, tuple, dict)):
            return [_normalize_literal(item) for item in query]
        return [_normalize_literal(list(query))]
    raise ValueError(f"Invalid query: {query!r}")


@dataclass
class ProofStep:
    """One resolution step in a derivation: a goal and what satisfied it."""

    goal: Literal
    source: str  # e.g. "fact employee(alice, security)" or "rule h :- b1, b2"


# Result of Reasoner.ask: (satisfied, solutions, first-proof).
AskResult = Tuple[bool, List[Dict[Var, Any]], Optional[List[ProofStep]]]


class Reasoner:
    """A small, deterministic Horn-clause backward-chaining reasoner.

    Rule variables are standardised apart (renamed to fresh names) on each
    application, so recursive rules and shared variable names across rules
    cannot capture one another. Search is bounded by ``max_depth`` and
    ``max_steps`` so recursive rule sets terminate.
    """

    def __init__(
        self,
        facts: Iterable[Any] = (),
        rules: Iterable[Any] = (),
        max_depth: int = 64,
        max_steps: int = 4096,
    ) -> None:
        self.facts: List[Literal] = [
            _normalize_literal(fact) if isinstance(fact, dict) else list(fact) for fact in facts
        ]
        self.rules: List[Rule] = []
        for rule in rules:
            if not isinstance(rule, dict) or "head" not in rule:
                raise ValueError(f"Invalid rule (needs 'head'): {rule!r}")
            self.rules.append(
                {
                    "head": _normalize_literal(rule["head"]),
                    "body": _normalize_body(rule.get("body", [])),
                }
            )
        self.max_depth = max_depth
        self.max_steps = max_steps
        self._steps = 0
        self._counter = 0

    def _renamer(self):
        """Return a closure that maps variables to fresh, unique names."""
        mapping: Dict[Var, Var] = {}

        def rename(term: Any) -> Any:
            if not _is_var(term):
                return term
            fresh = mapping.get(term)
            if fresh is None:
                self._counter += 1
                fresh = f"?_G{self._counter}_{term[len(_VAR_PREFIX):]}"
                mapping[term] = fresh
            return fresh

        return rename

    def _rename_rule(self, rule: Rule) -> Tuple[Literal, List[Literal]]:
        rename = self._renamer()
        head = [rename(term) for term in rule["head"]]
        body = [[rename(term) for term in literal] for literal in rule["body"]]
        return head, body

    def ask(self, query: Any, solution_limit: int = 50) -> AskResult:
        """Evaluate ``query`` deterministically.

        Returns ``(satisfied, solutions, proof)`` where ``satisfied`` is
        whether at least one proof exists, ``solutions`` is a capped list of
        bindings for the query variables, and ``proof`` is the derivation of
        the first solution (None if unsatisfiable).
        """
        goals = _normalize_query(query)
        query_vars = sorted({var for goal in goals for var in _variables_in(goal)})
        self._steps = 0
        self._counter = 0
        solutions: List[Dict[Var, Any]] = []
        proof: Optional[List[ProofStep]] = None
        for bindings, steps in self._prove(goals, {}, 0):
            solutions.append({var: _walk(var, bindings) for var in query_vars})
            if proof is None:
                proof = steps
            if len(solutions) >= solution_limit:
                break
        return bool(solutions), solutions, proof

    def _prove(
        self, goals: List[Literal], bindings: Dict[Var, Any], depth: int
    ) -> Iterable[Tuple[Dict[Var, Any], List[ProofStep]]]:
        if not goals:
            yield bindings, []
            return
        if depth > self.max_depth or self._steps > self.max_steps:
            return
        self._steps += 1

        goal = goals[0]
        rest = goals[1:]

        # Facts: goal holds if it unifies with a ground asserted fact.
        for fact in self.facts:
            rename = self._renamer()
            renamed = [rename(term) for term in fact]
            matched = _unify_literals(goal, renamed, dict(bindings))
            if matched is None:
                continue
            for final, rest_steps in self._prove(rest, matched, depth + 1):
                step = ProofStep(
                    goal=_resolve_literal(goal, final),
                    source="fact " + _render_literal(fact),
                )
                yield final, [step, *rest_steps]

        # Rules: goal holds if some rule's head unifies and its body holds.
        for rule in self.rules:
            head, body = self._rename_rule(rule)
            matched = _unify_literals(goal, head, dict(bindings))
            if matched is None:
                continue
            for body_bindings, body_steps in self._prove(body, matched, depth + 1):
                for final, rest_steps in self._prove(rest, body_bindings, depth + 1):
                    step = ProofStep(
                        goal=_resolve_literal(goal, final),
                        source="rule " + _render_rule(rule),
                    )
                    yield final, [step, *body_steps, *rest_steps]


def run_logic_reasoning(query: Any = None, facts: Any = None, rules: Any = None, **_: Any) -> str:
    """MCP-tool entrypoint: answer a rule-based query and return its derivation.

    Accepts the tool arguments an LLM supplies (``query`` plus optional
    ``facts`` and ``rules``), evaluates deterministically, and returns a
    compact text report: the yes/no answer, bindings for query variables,
    and the step-by-step derivation. Extra keyword arguments are ignored so
    callers can forward the full tool argument dict unchanged.
    """
    if query is None:
        return "logic_reasoning error: missing required argument 'query'."
    try:
        reasoner = Reasoner(facts=facts or [], rules=rules or [])
        satisfied, solutions, proof = reasoner.ask(query)
    except (ValueError, TypeError, KeyError) as exc:
        return f"logic_reasoning error: {exc}"

    lines: List[str] = ["Answer: YES" if satisfied else "Answer: NO"]

    if solutions:
        shown = solutions[:10]
        for solution in shown:
            if solution:
                binding = ", ".join(
                    f"{var} = {_render_term(value)}" for var, value in solution.items()
                )
                lines.append(binding)
            else:
                lines.append("(ground goal satisfied)")
        remaining = len(solutions) - len(shown)
        if remaining > 0:
            lines.append(f"... {remaining} more solution(s)")

    lines.append("Derivation:")
    if proof:
        for index, step in enumerate(proof, start=1):
            lines.append(f"  {index}. {_render_literal(step.goal)}  (by {step.source})")
    else:
        lines.append("  (no proof; goal could not be derived)")

    return "\n".join(lines)

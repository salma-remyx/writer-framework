from __future__ import annotations

import re

from writer.abstract import register_abstract_template
from writer.blocks.base_block import BlueprintBlock
from writer.ss_types import AbstractTemplate

# Embedding-free document search over raw parsed text.
#
# Adapted from READ (Reliable Embedding-free Agentic Document-search),
# arxiv:2608.06305. READ reads a long document through three deterministic
# operations -- normalized lexical search, structural navigation, and
# bounded span reads -- instead of chunk + embed + top-k, so that a retrieval
# trajectory is a replayable audit trail rather than an opaque similarity
# score. The paper locates the gain in the *interface* (deterministic
# operations over the raw text), not in iteration or in embeddings.
#
# What is ported at full fidelity here: the three deterministic operations
# and their normalized, line-addressable behaviour, plus an audit trail of
# every operation.
# What is intentionally substituted out (Mode 2): the agentic LLM loop and
# the MCP transport that drive those operations in the paper. This block
# exposes the operations directly to a workflow as a deterministic tool the
# agent or pipeline can call, downstream of the Parse PDF block whose
# `result` is exactly the raw text these operations run on.

# Keep digits so financial figures and units ("lakh", "crore", "2024") match.
_TOKEN_RE = re.compile(r"[0-9a-z]+")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+\S")


def _normalize_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _as_int(value, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _is_header(line: str) -> bool:
    """A structural anchor a figure may inherit context from.

    Markdown headings are the primary signal (Parse PDF emits markdown); the
    plain-text fallback treats a short, non-numeric, title-like line as an
    anchor. Deterministic and conservative on purpose.
    """
    if _HEADING_RE.match(line):
        return True
    stripped = line.strip()
    if not stripped or stripped.endswith("."):
        return False
    words = stripped.split()
    if not 1 <= len(words) <= 10:
        return False
    # A figure row is mostly digits; a header has none.
    return not any(re.search(r"\d", w) for w in words)


class Document:
    """Line-addressable view over a raw document for deterministic reads."""

    def __init__(self, content: str):
        self.lines = (content or "").splitlines()

    def window(self, center: int, radius: int) -> list[dict]:
        start = max(1, center - radius)
        end = min(len(self.lines), center + radius)
        return [{"line": i, "text": self.lines[i - 1]} for i in range(start, end + 1)]

    def read(self, position: int, span: int) -> list[dict]:
        start = max(1, position)
        end = min(len(self.lines), position + max(1, span) - 1)
        return [{"line": i, "text": self.lines[i - 1]} for i in range(start, end + 1)]

    def nearest_header(self, line: int, direction: str) -> dict | None:
        if direction == "up":
            candidates = range(line - 1, 0, -1)
        elif direction == "down":
            candidates = range(line + 1, len(self.lines) + 1)
        else:  # nearest: walk outward by increasing distance from `line`
            max_dist = max(line - 1, len(self.lines) - line)
            candidates = (
                i
                for dist in range(1, max_dist + 1)
                for i in (line - dist, line + dist)
                if 1 <= i <= len(self.lines)
            )
        for i in candidates:
            if _is_header(self.lines[i - 1]):
                return {"line": i, "text": self.lines[i - 1]}
        return None

    def search(
        self, query: str, match: str = "all", context: int = 2, max_results: int = 10
    ) -> list[dict]:
        terms = set(_normalize_tokens(query))
        if not terms:
            return []
        hits: list[dict] = []
        for idx, line in enumerate(self.lines, start=1):
            tokens = set(_normalize_tokens(line))
            matched = terms & tokens
            if not matched:
                continue
            if match == "all" and not terms.issubset(tokens):
                continue
            # Surface the structural context a chunk boundary would have
            # severed: the header the figure inherits its unit from.
            header = self.nearest_header(idx, "up")
            hits.append(
                {
                    "line": idx,
                    "text": line,
                    "score": len(matched),
                    "context": self.window(idx, context),
                    "header": header,
                }
            )
            if len(hits) >= max_results:
                break
        return hits


def _result_line_numbers(result: dict) -> list[int]:
    op = result.get("operation")
    if op == "search":
        return [m["line"] for m in result.get("matches", [])]
    if op == "navigate":
        anchor = result.get("anchor")
        return [anchor["line"]] if anchor else []
    if op == "read":
        return [r["line"] for r in result.get("span", [])]
    return []


def _audit_message(params: dict, result: dict) -> str:
    op = params["operation"]
    lines = _result_line_numbers(result)
    if op == "search":
        return f"lexical search '{params.get('query')}' ({params.get('match')}) -> lines {lines}"
    if op == "navigate":
        return (
            f"navigate {params.get('direction')} from line {params.get('position')} "
            f"-> lines {lines}"
        )
    if op == "read":
        return f"read span {params.get('span')} at line {params.get('position')} -> lines {lines}"
    return str(op)


class WriterDocumentSearch(BlueprintBlock):
    @classmethod
    def register(cls, type: str):
        super(WriterDocumentSearch, cls).register(type)
        register_abstract_template(
            type,
            AbstractTemplate(
                baseType="blueprints_node",
                writer={
                    "name": "Search document",
                    "description": (
                        "Embedding-free document search over raw text. Runs a "
                        "deterministic operation (lexical search, navigate to "
                        "header, or bounded span read) and records an auditable "
                        "trail. Pair with the Parse PDF block."
                    ),
                    "category": "Writer",
                    "fields": {
                        "content": {
                            "name": "Document content",
                            "type": "Text",
                            "control": "Textarea",
                            "default": "",
                            "desc": (
                                "The raw text/markdown to read. Typically the "
                                "result of the Parse PDF block."
                            ),
                        },
                        "operation": {
                            "name": "Operation",
                            "type": "Text",
                            "options": {
                                "search": "Lexical search",
                                "navigate": "Navigate to header",
                                "read": "Read span",
                            },
                            "default": "search",
                            "desc": "Deterministic, embedding-free operation.",
                        },
                        "query": {
                            "name": "Query",
                            "type": "Text",
                            "default": "",
                            "desc": (
                                "Search terms (normalized: lowercased, "
                                "punctuation stripped). Used by Lexical search."
                            ),
                        },
                        "match": {
                            "name": "Match",
                            "type": "Text",
                            "options": {"all": "All terms", "any": "Any term"},
                            "default": "all",
                        },
                        "position": {
                            "name": "Line position",
                            "type": "Text",
                            "default": "1",
                            "desc": "1-indexed line for Navigate / Read span.",
                        },
                        "direction": {
                            "name": "Direction",
                            "type": "Text",
                            "options": {
                                "up": "Up",
                                "down": "Down",
                                "nearest": "Nearest",
                            },
                            "default": "up",
                            "desc": "Search direction for Navigate to header.",
                        },
                        "span": {
                            "name": "Span",
                            "type": "Text",
                            "default": "5",
                            "desc": "Number of lines returned by Read span.",
                        },
                        "context": {
                            "name": "Context lines",
                            "type": "Text",
                            "default": "2",
                            "desc": "Lines of context around each search hit.",
                        },
                    },
                    "outs": {
                        "success": {
                            "name": "Success",
                            "description": "The operation completed.",
                            "style": "success",
                        },
                        "error": {
                            "name": "Error",
                            "description": "There was an error running the operation.",
                            "style": "error",
                        },
                    },
                },
            ),
        )

    def run(self):
        try:
            content = self._get_field("content", required=True)
            operation = self._get_field("operation", default_field_value="search")
            query = self._get_field("query")
            match = self._get_field("match", default_field_value="all")
            direction = self._get_field("direction", default_field_value="up")
            position = _as_int(self._get_field("position", default_field_value="1"), 1)
            span = _as_int(self._get_field("span", default_field_value="5"), 5)
            context = _as_int(self._get_field("context", default_field_value="2"), 2)

            document = Document(content)
            params = {
                "operation": operation,
                "query": query,
                "match": match,
                "position": position,
                "direction": direction,
                "span": span,
                "context": context,
            }

            if operation == "search":
                matches = document.search(query, match=match, context=context)
                result = {"operation": "search", "matches": matches}
            elif operation == "navigate":
                anchor = document.nearest_header(position, direction)
                around = document.window(anchor["line"], context) if anchor else []
                result = {"operation": "navigate", "anchor": anchor, "context": around}
            elif operation == "read":
                result = {"operation": "read", "span": document.read(position, span)}
            else:
                raise ValueError(f"Unknown operation: {operation}")

            self.result = result
            self._record_audit(params, result)
            self.outcome = "success"

        except BaseException as e:
            self.outcome = "error"
            raise e

    def _record_audit(self, params: dict, result: dict) -> None:
        """Append a replayable entry to the execution trail and the Journal.

        The structured `read_operations` list in the execution environment is
        the deterministic replay trail (consumed by the Journal/Execution
        Logging the block already serialises); the log entry surfaces a
        human-readable line in the Journal UI.
        """
        summary = {
            "operation": params["operation"],
            "params": params,
            "line_numbers": _result_line_numbers(result),
        }
        self.execution_environment.setdefault("read_operations", []).append(summary)
        try:
            self.runner.session.session_state.add_log_entry(
                "info", "Document search", _audit_message(params, result)
            )
        except Exception:
            # Journal surfacing is best-effort; the deterministic trail above
            # is authoritative and must never be blocked by logging.
            pass

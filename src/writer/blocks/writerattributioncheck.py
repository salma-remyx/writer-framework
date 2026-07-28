import re
from typing import Any, Dict, List

from writer.abstract import register_abstract_template
from writer.blocks.base_block import WriterBlock
from writer.ss_types import AbstractTemplate

# Adapted from AttributionBench (Yue et al., 2024, arXiv:2402.15089), which
# frames attribution evaluation as: decompose a generated answer into atomic
# claims, then decide whether each claim is fully supported by its cited
# evidence. The paper's reference evaluator uses a learned (LLM) decomposer and
# a learned support classifier. This block ports the *measurement scheme* —
# decompose -> per-claim support verdict -> aggregate attribution score — while
# substituting those learned components with parameter-free proxies so the check
# is deterministic and needs no extra model call:
#   * atomic-claim decomposition  -> sentence splitting (a coarse claim unit)
#   * per-claim support judgment  -> lexical coverage of the claim's content
#                                    tokens against the concatenated evidence
# Both substitutions approximate the paper's signal rather than reproduce its
# model; see the module docstring of ``evaluate_attribution`` for details.

# A small, dependency-free English stopword set. Support coverage focuses on
# content words so that function-word overlap cannot inflate a verdict.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else when while of to in on at by for with from
    into over under is are was were be been being this that these those it its as
    not no do does did done have has had having i you he she they we me him her
    them us my your his their our mine yours hers theirs ours which who whom whose
    what where why how all any both each few more most other some such only own
    same so than too very can will just should would could may might must shall
    about above after again against before below between during further here there
    once per via
    """.split()
)


def _tokenize(text: str) -> List[str]:
    """Lowercase and split text into content tokens, dropping stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _split_sentences(text: str) -> List[str]:
    """Approximate atomic-claim decomposition by splitting into sentences.

    AttributionBench decomposes answers into atomic claims using an LLM. Without
    one available we use sentence boundaries as a coarse, deterministic proxy
    for claim units — each sentence is treated as one claim.
    """
    text = text.strip()
    if not text:
        return []
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])", text)
    return [s.strip() for s in raw if s.strip()]


def _extract_text(item: Any) -> str:
    """Coerce a single evidence item (string or dict of fields) into text.

    Handles the shapes emitted by the repo's retrieval blocks: web-search
    sources (``{"url", "title", "snippet", ...}``) and plain citation strings.
    """
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        parts = [str(v) for v in item.values() if isinstance(v, (str, int, float))]
        return " ".join(parts)
    return str(item)


def evaluate_attribution(
    answer: str,
    evidence: List[Any],
    threshold: float = 0.6,
) -> Dict[str, Any]:
    """Decompose ``answer`` into atomic claims and check each against ``evidence``.

    For every claim, support is approximated by the fraction of the claim's
    content tokens that appear anywhere in the concatenated evidence text. A
    claim is reported as ``supported`` when this coverage reaches ``threshold``.
    Returns the per-claim verdicts plus an aggregate attribution score in
    ``[0, 1]`` (``None`` when the answer has no claims).
    """
    evidence_text = " ".join(_extract_text(item) for item in evidence)
    evidence_tokens = set(_tokenize(evidence_text))

    per_claim: List[Dict[str, Any]] = []
    supported = 0
    for claim in _split_sentences(answer):
        claim_tokens = _tokenize(claim)
        if not claim_tokens:
            score = 0.0
        else:
            hits = sum(1 for token in claim_tokens if token in evidence_tokens)
            score = hits / len(claim_tokens)
        is_supported = score >= threshold
        if is_supported:
            supported += 1
        per_claim.append(
            {
                "claim": claim,
                "supported": is_supported,
                "score": round(score, 4),
            }
        )

    total = len(per_claim)
    attribution_score = (supported / total) if total else None
    return {
        "claims": per_claim,
        "supported_claims": supported,
        "total_claims": total,
        "attribution_score": attribution_score,
    }


class WriterAttributionCheck(WriterBlock):
    """Evaluates whether an answer's claims are supported by their cited evidence.

    Consumes the ``{answer, sources | citations}`` contract produced by the
    Web search and Ask graph question blocks and emits an attribution verdict.
    """

    @classmethod
    def register(cls, type: str):
        super(WriterAttributionCheck, cls).register(type)
        register_abstract_template(
            type,
            AbstractTemplate(
                baseType="blueprints_node",
                writer={
                    "name": "Attribution check",
                    "description": (
                        "Decomposes an answer into atomic claims and checks "
                        "each claim's support against its cited evidence, "
                        "producing a per-claim verdict and an attribution score. "
                        "Feed it the answer and sources/citations from a Web "
                        "search or Ask graph question block."
                    ),
                    "category": "Writer",
                    "fields": {
                        "answer": {
                            "name": "Answer",
                            "type": "Text",
                            "control": "Textarea",
                            "desc": "The generated answer to evaluate. May be the whole result of a Web search / Ask graph question block.",
                            "validator": {
                                "type": "string",
                                "minLength": 1,
                            },
                        },
                        "evidence": {
                            "name": "Evidence",
                            "type": "Object",
                            "default": "[]",
                            "desc": "The cited evidence to check against: the sources list from Web search, the citations list from Ask graph question, or a plain text passage.",
                            "validator": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                },
                            },
                        },
                        "threshold": {
                            "name": "Support threshold",
                            "type": "Text",
                            "default": "0.6",
                            "desc": "Claim content-token coverage of the evidence required to count a claim as supported, between 0 and 1.",
                        },
                        "stateElement": {
                            "name": "Link Variable",
                            "type": "Binding",
                            "desc": "Set the variable here and use it across your agent.",
                        },
                    },
                    "outs": {
                        "success": {
                            "name": "Success",
                            "description": "The attribution check completed successfully.",
                            "style": "success",
                        },
                        "error": {
                            "name": "Error",
                            "description": "There was an error running the attribution check.",
                            "style": "error",
                        },
                    },
                },
            ),
        )

    def run(self):
        try:
            answer = self._get_field("answer", required=True)
            # Tolerate the whole {answer: ...} result being piped in directly.
            if isinstance(answer, dict):
                answer = answer.get("answer", "")
            if not isinstance(answer, str):
                answer = str(answer)

            evidence = self._get_field("evidence", as_json=True, default_field_value="[]")
            # Tolerate {sources: [...]} / {citations: [...]} being piped in.
            if isinstance(evidence, dict):
                evidence = evidence.get("sources") or evidence.get("citations") or []
            if not isinstance(evidence, list):
                evidence = [evidence]

            threshold_value = self._get_field("threshold", default_field_value="0.6")
            try:
                threshold = float(threshold_value)
            except (TypeError, ValueError):
                threshold = 0.6
            threshold = max(0.0, min(1.0, threshold))

            result = evaluate_attribution(answer, evidence, threshold)
            self.result = result

            state_element = self._get_field("stateElement", required=False)
            if state_element:
                self._set_state(state_element, result)
            self.outcome = "success"

        except BaseException as e:
            self.outcome = "error"
            raise e

import pytest
from writer.blocks.base_block import block_map
from writer.blocks.writerattributioncheck import (
    WriterAttributionCheck,
    evaluate_attribution,
)


def test_block_registered_in_dispatch():
    """Integration: the block is wired into the existing block dispatch map."""
    assert block_map["blueprints_writerattributioncheck"] is WriterAttributionCheck


def test_check_supported_answer(session, runner, fake_client):
    answer = "Python is a programming language. It was created by Guido van Rossum."
    session.session_state["sources"] = [
        {
            "url": "https://python.org",
            "title": "Python Official Site",
            "snippet": "Python is a programming language created by Guido van Rossum.",
        }
    ]
    component = session.add_fake_component(
        {
            "answer": answer,
            "evidence": "@{sources}",
            "threshold": "0.6",
        }
    )

    block = WriterAttributionCheck(component, runner, {})
    block.run()

    assert block.outcome == "success"
    assert block.result["total_claims"] == 2
    assert block.result["supported_claims"] == 2
    assert block.result["attribution_score"] == 1.0
    assert block.result["claims"][0]["supported"] is True


def test_check_flags_unsupported_claim(session, runner, fake_client):
    # Second sentence introduces facts absent from the evidence.
    answer = "Python is a programming language. " "Rust won the Nobel Prize in Chemistry in 1995."
    session.session_state["sources"] = [{"snippet": "Python is a programming language."}]
    component = session.add_fake_component({"answer": answer, "evidence": "@{sources}"})

    block = WriterAttributionCheck(component, runner, {})
    block.run()

    assert block.outcome == "success"
    verdicts = block.result["claims"]
    assert len(verdicts) == 2
    assert verdicts[0]["supported"] is True
    assert verdicts[1]["supported"] is False
    assert block.result["supported_claims"] == 1
    assert block.result["attribution_score"] == 0.5


def test_check_consumes_websearch_result_contract(session, runner, fake_client):
    """Feeds the exact {answer, sources} shape the Web search block emits."""
    session.session_state["search_result"] = {
        "answer": "Paris is the capital of France.",
        "sources": [{"snippet": "France's capital city is Paris."}],
    }
    component = session.add_fake_component(
        {
            "answer": "@{search_result.answer}",
            "evidence": "@{search_result.sources}",
        }
    )

    block = WriterAttributionCheck(component, runner, {})
    block.run()

    assert block.outcome == "success"
    assert block.result["attribution_score"] == 1.0


def test_check_writes_state_element(session, runner, fake_client):
    session.session_state["sources"] = [{"snippet": "A short supported claim."}]
    component = session.add_fake_component(
        {
            "answer": "A short supported claim.",
            "evidence": "@{sources}",
            "stateElement": "attribution",
        }
    )

    block = WriterAttributionCheck(component, runner, {})
    block.run()

    assert block.outcome == "success"
    assert session.session_state["attribution"]["supported_claims"] == 1


def test_check_missing_answer(session, runner, fake_client):
    component = session.add_fake_component({"evidence": "[]"})
    block = WriterAttributionCheck(component, runner, {})

    with pytest.raises(Exception):  # WriterConfigurationError for missing field
        block.run()
    assert block.outcome == "error"


def test_check_threshold_clamps_to_range():
    # threshold > 1 is clamped to 1.0, so partial coverage is not supported.
    result = evaluate_attribution(
        "Python is a programming language.",
        ["Python programming language"],
        threshold=1.5,
    )
    assert result["attribution_score"] == 0.0


def test_check_empty_answer_has_no_claims():
    result = evaluate_attribution("", [{"snippet": "evidence"}])
    assert result["total_claims"] == 0
    assert result["attribution_score"] is None

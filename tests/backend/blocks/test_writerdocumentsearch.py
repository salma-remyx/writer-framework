from writer.blocks.base_block import block_map
from writer.blocks.writerdocumentsearch import WriterDocumentSearch

# A miniature financial report. Figure rows (e.g. line 5) inherit their unit
# ("lakh") from the header on line 3 -- the chunk-boundary hazard READ targets.
DOC = """## Annual Report 2024

## Operating Revenue (figures in lakh)
Total operating revenue for the year stood at 4,532 across segments.
Segment A contributed 2,100 and Segment B contributed 2,432.
The growth was driven by volume expansion in core markets.

## Notes to Accounts (figures in crore)
Note 5: The corresponding prior-year figure was restated to 38.
"""


def _make_block(session, runner, values, comp_id="fake_id"):
    component = session.add_fake_component({}, id=comp_id)
    block = WriterDocumentSearch(component, runner, {})

    def fake_get_field(name, *args, **kwargs):
        if name in values:
            return values[name]
        default = kwargs.get("default_field_value")
        return default if default is not None else ""

    block._get_field = fake_get_field
    return block


def test_block_is_registered_via_init():
    # Exercises the __init__.py wiring edit (the integration call site):
    # the block must resolve through the shared block_map by its type key.
    assert "blueprints_writerdocumentsearch" in block_map
    assert block_map["blueprints_writerdocumentsearch"] is WriterDocumentSearch


def test_search_attaches_inherited_unit_header(session, runner):
    block = _make_block(
        session,
        runner,
        {"content": DOC, "operation": "search", "query": "contributed", "match": "any"},
    )
    block.run()

    assert block.outcome == "success"
    matches = block.result["matches"]
    assert matches and matches[0]["line"] == 5
    # The figure inherits its unit from the header above -- surfaced explicitly
    # rather than severed by a chunk boundary.
    assert matches[0]["header"]["line"] == 3
    assert "lakh" in matches[0]["header"]["text"]


def test_navigate_finds_header_above_and_below(session, runner):
    up = _make_block(
        session,
        runner,
        {"content": DOC, "operation": "navigate", "position": "6"},
        comp_id="nav_up",
    )
    up.run()
    assert up.result["anchor"]["line"] == 3

    down = _make_block(
        session,
        runner,
        {"content": DOC, "operation": "navigate", "position": "6", "direction": "down"},
        comp_id="nav_down",
    )
    down.run()
    assert down.result["anchor"]["line"] == 8
    assert "crore" in down.result["anchor"]["text"]


def test_read_span_returns_bounded_window(session, runner):
    block = _make_block(
        session, runner, {"content": DOC, "operation": "read", "position": "4", "span": "2"}
    )
    block.run()

    assert block.outcome == "success"
    assert [row["line"] for row in block.result["span"]] == [4, 5]


def test_audit_trail_records_each_operation(session, runner):
    env = {}
    block = _make_block(
        session,
        runner,
        {"content": DOC, "operation": "search", "query": "contributed", "match": "any"},
        comp_id="audit",
    )
    block.execution_environment = env
    block.run()

    trail = env["read_operations"]
    assert len(trail) == 1
    assert trail[0]["operation"] == "search"
    assert 5 in trail[0]["line_numbers"]


def test_unknown_operation_errors(session, runner):
    block = _make_block(session, runner, {"content": DOC, "operation": "summon"})
    try:
        block.run()
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    assert block.outcome == "error"

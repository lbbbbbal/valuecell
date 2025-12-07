import pytest

pytest.importorskip("agno")

from valuecell.utils.model import ensure_json_hint


def test_ensure_json_hint_appends_missing_keyword():
    base_instructions = ["Return structured output."]

    result = ensure_json_hint(base_instructions)

    assert len(result) == len(base_instructions) + 1
    assert any("json" in instr.lower() for instr in result)


def test_ensure_json_hint_preserves_existing_keyword():
    base_instructions = ["Please reply in json only."]

    result = ensure_json_hint(base_instructions)

    assert result == base_instructions

from pathlib import Path
from typing import Any
import pytest
from syrupy.assertion import SnapshotAssertion
import json
from ..json_decode import correct_json

def load_fixed_json(json_str: str) -> dict[str, Any]:
    return json.loads(correct_json(json_str))


def test_fixed_json_simple_case(snapshot: SnapshotAssertion) -> None:
    assert snapshot == load_fixed_json("{}")
    assert snapshot == load_fixed_json('{"foo": "bar"}')
    assert snapshot == load_fixed_json('{"foo": "bar", "baz": "qux"}')
    assert snapshot == load_fixed_json('{"foo": "bar", "baz": "qux", "quux": "corge"}')
    # add more nested objects
    assert snapshot == load_fixed_json('{"foo": {"bar": "baz"}}')
    assert snapshot == load_fixed_json('{"foo": {"bar": {"baz": "qux"}}}')
    assert snapshot == load_fixed_json('{"foo": {"bar": {"baz": {"qux": "quux"}}}}')

@pytest.mark.parametrize("test_filename", (Path(__file__).parent / "test_json_decode").glob("safe/*.json"), ids=lambda x: x.name)
def test_fixed_json_safe_case(snapshot: SnapshotAssertion, test_filename: Path) -> None:
    content = test_filename.read_text()
    result = load_fixed_json(content)
    assert snapshot == result


@pytest.mark.parametrize("test_filename", (Path(__file__).parent / "test_json_decode").glob("unsafe/*.json"), ids=lambda x: x.name)
def test_fixed_json_unsafe_case(snapshot: SnapshotAssertion, test_filename: Path) -> None:
    content = test_filename.read_text()
    result = load_fixed_json(content)
    assert snapshot == result
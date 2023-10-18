from pathlib import Path

import pytest
from syrupy.assertion import SnapshotAssertion

from ..json_decode import correct_json


@pytest.mark.parametrize("test_filename", (Path(__file__).parent / "test_json_decode").glob("*.txt"))
def test_correct_json(snapshot: SnapshotAssertion, test_filename: Path) -> None:
    content = test_filename.read_text()
    result = correct_json(content)
    assert snapshot == result

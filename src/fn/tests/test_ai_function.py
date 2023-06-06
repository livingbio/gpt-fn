import pytest
from syrupy.assertion import SnapshotAssertion

from ..ai_function import ai_fn


@ai_fn
def fabnocci(n: int) -> int:  # type: ignore[empty-body]
    """return fabnocci number"""


@pytest.mark.vcr(filter_headers=["authorization"])
def test_ai_fn(snapshot: SnapshotAssertion) -> None:
    assert snapshot == fabnocci(10)

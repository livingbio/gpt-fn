import pytest
from pydantic import BaseModel
from syrupy.assertion import SnapshotAssertion

from ..ai_function import ai_fn


@ai_fn
def fabnocci(n: int) -> int:  # type: ignore[empty-body]
    """return fabnocci number"""


class Hero(BaseModel):
    """Hero model."""

    name: str
    age: int


@ai_fn
def fake_hero(n: int) -> list[Hero]:  # type: ignore[empty-body]
    """generate fake hero."""


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "query", "body"])
def test_ai_fabnocci(snapshot: SnapshotAssertion) -> None:
    assert snapshot == fabnocci(10)


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "query", "body"])
def test_ai_fake_hero(snapshot: SnapshotAssertion) -> None:
    heros = fake_hero(5)
    assert len(heros) == 5
    assert all(isinstance(k, Hero) for k in heros)
    assert snapshot == heros

import pytest
from pydantic import BaseModel
from syrupy.assertion import SnapshotAssertion

from ..ai_function import ai_fn
from ..exceptions import AiFnError


@ai_fn()
def fabnocci(n: int) -> int:  # type: ignore[empty-body]
    """return fabnocci number"""


class Hero(BaseModel):
    """Hero model."""

    name: str
    age: int


@ai_fn()
def fake_hero(n: int) -> list[Hero]:  # type: ignore[empty-body]
    """generate fake hero."""


@ai_fn()
def generate_hashtags(text: str) -> list[str]:  # type: ignore[empty-body]
    """generate social media hashtags from text."""


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "query", "body"])
def test_ai_fabnocci(snapshot: SnapshotAssertion) -> None:
    assert snapshot == fabnocci(10)


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "query", "body"])
def test_ai_fake_hero(snapshot: SnapshotAssertion) -> None:
    heros = fake_hero(5)
    assert len(heros) == 5
    assert all(isinstance(k, Hero) for k in heros)
    assert snapshot == heros


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "query", "body"])
def test_ai_fn_max_token(snapshot: SnapshotAssertion) -> None:
    with pytest.raises(AiFnError) as excinfo:
        generate_hashtags(" ".join("hello" for _ in range(5001)))

    assert snapshot(name="exconly") == excinfo.exconly()
    assert snapshot(name="context") == excinfo.value.__context__
    assert snapshot(name="vars(exc)") == vars(excinfo.value)

from typing import Any, Callable

import pytest
from pydantic import BaseModel
from syrupy.assertion import SnapshotAssertion

from ..signature import FunctionSignature


def add(a: int, b: int = 10) -> int:  # type: ignore[empty-body]
    """Add two numbers."""


def concat(a: str, b: str) -> str:  # type: ignore[empty-body]
    """Concat two strings"""


class Person(BaseModel):
    """Person model."""

    name: str
    age: int


def fake_person(count: int) -> Person:  # type: ignore[empty-body]
    """generate fake person."""


def no_return_annoation(a: int, b: int):  # type: ignore[no-untyped-def]
    """Add two numbers."""


def concats(*args: str) -> str:  # type: ignore[empty-body]
    """Concat the given strings"""


def complex(a: str, b: str, *args: str, c: str, d: str, **kwargs: str) -> str:  # type: ignore[empty-body]
    """Complex function"""


@pytest.mark.parametrize(
    "func, args, kwargs",
    [
        (add, (1, 2), {}),
        (fake_person, (5,), {}),
        (concat, ("pen", "apple"), {}),
        (concat, ("a'b", "cd"), {}),
        (concats, ("a", "b", "c", "d"), {}),
        (
            complex,
            ("a-v", "b-v", "arg-v1", "arg-v2"),
            {"c": "c-v", "d": "d-v", "kwarg1": "kwarg1-v", "kwarg2": "kwarg2-v"},
        ),
    ],
)
def test_function_signature(
    func: Callable[[Any], Any],
    args: tuple[Any],
    kwargs: dict[str, Any],
    snapshot: SnapshotAssertion,
) -> None:
    sig = FunctionSignature(func)
    assert snapshot == sig.instruction()
    assert snapshot == sig.call_line(*args, **kwargs)


def test_function_signature_without_return_annoations() -> None:
    with pytest.raises(AssertionError):
        FunctionSignature(no_return_annoation)

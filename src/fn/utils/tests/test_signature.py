from typing import Any, Callable

import pytest
from pydantic import BaseModel
from syrupy.assertion import SnapshotAssertion

from ..signature import FunctionSignature


def add(a: int, b: int) -> int:  # type: ignore[empty-body]
    """Add two numbers."""


class Person(BaseModel):
    """Person model."""

    name: str
    age: int


def fake_person(count: int) -> Person:  # type: ignore[empty-body]
    """generate fake person."""


def no_return_annoation(a: int, b: int):  # type: ignore[no-untyped-def]
    """Add two numbers."""


@pytest.mark.parametrize("func", [add, fake_person])
def test_function_signature(func: Callable[[Any], Any], snapshot: SnapshotAssertion) -> None:
    sig = FunctionSignature(func)
    assert snapshot == sig.instruction()


def test_function_signature_without_return_annoations() -> None:
    with pytest.raises(AssertionError):
        FunctionSignature(no_return_annoation)

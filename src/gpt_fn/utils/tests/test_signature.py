from typing import Annotated, Any, Callable, Literal

import pytest
from pydantic import BaseModel, Field
from syrupy.assertion import SnapshotAssertion

from ..signature import FunctionSignature


def add(a: int, b: int = 10) -> int:  # type: ignore[empty-body]
    """Add two numbers."""


def concat(a: str, b: str) -> str:  # type: ignore[empty-body]
    """Concat two strings
    :param a: first string
    :param b: second string
    """


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


def get_current_weather(*locations: str, unit: Literal["celsius", "fahrenheit"]) -> list[dict[str, Any]]:
    """ "Get the current weather in a given location"""
    return [
        {
            "location": location,
            "temperature": "72",
            "forecast": ["sunny", "windy"],
        }
        for location in locations
    ]


def is_male(person: Person) -> bool:  # type: ignore[empty-body]
    """return the person is male"""


def how_many(num: Annotated[int, Field(gt=10, description="greater than 10")]) -> int:
    """return the given number"""
    return num


def add_ext(filename: str, ext: str) -> str:
    """Add the given extension to the filename

    This is a multiline docstring.

    :param filename: the filename
    :param ext: the extension, e.g. `.txt`
    """
    return filename + ext


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
        (get_current_weather, ("New York", "London"), {"unit": "celsius"}),
        (is_male, (Person(name="John", age=20),), {}),
        (how_many, (20,), {}),
        (add_ext, ("test", ".txt"), {}),
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
    assert snapshot == sig.schema()


def test_function_signature_without_return_annoations() -> None:
    with pytest.raises(AssertionError):
        FunctionSignature(no_return_annoation)

import pydantic
import pytest
from syrupy.assertion import SnapshotAssertion

from ..completion import chat_completion, function_completion, structural_completion
from ..exceptions import CompletionIncompleteError


def book_a_flight(date: str, destination: str, origin: str = "London") -> str:
    return f"Booked a flight from {origin} to {destination} on {date}."


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "query", "body"])
def test_function_completion(snapshot: SnapshotAssertion) -> None:
    msg = function_completion(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I want to book a flight to London."},
        ],
        functions=[book_a_flight],
    )
    assert snapshot == msg


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "query", "body"])
def test_function_completion_without_enough_tokens(snapshot: SnapshotAssertion) -> None:
    with pytest.raises(CompletionIncompleteError) as excinfo:
        msg = function_completion(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "I want to book a restaurant in London."},
            ],
            max_tokens=10,
            functions=[book_a_flight],
        )
    assert snapshot == excinfo.exconly()
    assert snapshot == vars(excinfo.value)


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "query", "body"])
def test_function_completion_without_neccess_function(
    snapshot: SnapshotAssertion,
) -> None:
    with pytest.raises(CompletionIncompleteError) as excinfo:
        msg = function_completion(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "I want to book a restaurant in London."},
            ],
            functions=[book_a_flight],
        )
    assert snapshot == excinfo.exconly()
    assert snapshot == vars(excinfo.value)


class Email(pydantic.BaseModel):
    subject: str = pydantic.Field(description="the subject of email")
    body: str = pydantic.Field(description="the body of email")


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "query", "body"])
def test_structural_completion(snapshot: SnapshotAssertion) -> None:
    email = structural_completion(
        Email,
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Give me a sample email."},
        ],
    )

    assert snapshot == email


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "query", "body"])
def test_structural_completion_without_enough_tokens(snapshot: SnapshotAssertion) -> None:
    with pytest.raises(CompletionIncompleteError) as excinfo:
        email = structural_completion(
            Email,
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Give me a sample email."},
            ],
            max_tokens=10,
        )
    assert snapshot == excinfo.exconly()
    assert snapshot == vars(excinfo.value)


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "query", "body"])
def test_chat_completion(snapshot: SnapshotAssertion) -> None:
    msg = chat_completion(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, who are you?"},
        ],
    )
    assert snapshot == msg


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "query", "body"])
def test_chat_completion_incomplete(snapshot: SnapshotAssertion) -> None:
    with pytest.raises(CompletionIncompleteError) as excinfo:
        chat_completion(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, who are you?"},
            ],
            max_tokens=1,
        )

    assert snapshot == excinfo.exconly()
    assert snapshot == vars(excinfo.value)

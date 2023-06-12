import pytest
from syrupy.assertion import SnapshotAssertion

from ..completion import chat_completion
from ..exceptions import CompletionIncompleteError


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

from pathlib import Path

import pytest
from pydantic import BaseModel, Field
from syrupy.assertion import SnapshotAssertion

from ...completion import Message, structural_completion
from ..pydantic_parser import ParserError, PydanticParser


class Receipt(BaseModel):
    """The Receipt for user"""

    amount: float
    currency: str = Field(description="ISO 4217 currency code")
    customer: str


class Email(BaseModel):
    subject: str = Field(description="the subject of email")
    body: str = Field(description="the body of email")


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "query", "body"])
@pytest.mark.parametrize("test_filename, model", [("email.txt", Email)])
def test_pydantic_parser_with_prompt(snapshot: SnapshotAssertion, test_filename: str, model: type[BaseModel], datadir: Path) -> None:

    content = (datadir / test_filename).read_text()
    instruction = content
    assert snapshot(name="instructoin") == instruction

    result = structural_completion(model, messages=[Message(role="system", content=""), Message(role="user", content=instruction)])

    assert snapshot(name="gpt response") == result


def test_pydantic_parser_get_format_instructions(
    snapshot: SnapshotAssertion,
) -> None:
    assert snapshot == PydanticParser[Receipt](pydantic_model=Receipt).get_format_instructions()


def test_parse_output(snapshot: SnapshotAssertion) -> None:
    assert snapshot == PydanticParser[Receipt](pydantic_model=Receipt).parse('{"amount": 1.0, "currency": "USD", "customer": "John Doe"}')


def test_pydantic_parser_fail() -> None:
    with pytest.raises(ParserError):
        PydanticParser[Receipt](pydantic_model=Receipt).parse('{"amount": 1.0, "currency": "USD"}')

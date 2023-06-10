import pytest
from pydantic import BaseModel, Field
from syrupy.assertion import SnapshotAssertion

from ..pydantic_parser import ParserError, PydanticParser


class Receipt(BaseModel):
    """The Receipt for user"""

    amount: float
    currency: str = Field(description="ISO 4217 currency code")
    customer: str


def test_pydantic_parser_get_format_instructions(
    snapshot: SnapshotAssertion,
) -> None:
    assert snapshot == PydanticParser[Receipt](pydantic_model=Receipt).get_format_instructions()


def test_parse_output(snapshot: SnapshotAssertion) -> None:
    assert snapshot == PydanticParser[Receipt](pydantic_model=Receipt).parse('{"amount": 1.0, "currency": "USD", "customer": "John Doe"}')


def test_pydantic_parser_fail() -> None:
    with pytest.raises(ParserError):
        PydanticParser[Receipt](pydantic_model=Receipt).parse('{"amount": 1.0, "currency": "USD"}')

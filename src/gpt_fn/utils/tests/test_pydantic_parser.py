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


def test_pydantic_parser_parse(snapshot: SnapshotAssertion) -> None:
    assert snapshot == PydanticParser[Email](pydantic_model=Email).parse(
        """json\n{\n  "subject": "Revolutionizing the E-Bike Industry with Innovative Aluminum Frames",\n  "body": "Dear valued client,\\n\\nWe are excited to share with you the latest news from JIN-JI International Co., Ltd. We have revolutionized the E-Bike industry with our innovative aluminum frames for SPACIOUS Industrial Co., Ltd. in Taiwan. Our commitment to innovation and customer satisfaction has positioned us as a key player in the market.\\n\\nCEO Xiao Ming, Chen, emphasizes, \\"We provide the best quality products,\\" reflecting our dedication to excellence.\\n\\nTo learn more about our revolutionary aluminum frames and how they offer increased rigidity, stability, and safety for riders, we invite you to visit our website at https://jinji.en.taiwantrade.com/. We believe that our innovative designs will support complex structures for electric bikes, ensuring a confident and secure riding experience.\\n\\nThank you for your continued support, and we look forward to bringing you more groundbreaking developments in the future.\\n\\nBest regards,\\n[Your Name]\\nJIN-JI International Co., Ltd."\n}\n"""
    )


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

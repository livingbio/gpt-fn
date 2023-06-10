from syrupy.assertion import SnapshotAssertion

from ..prompt import ChatTemplate, MessageTemplate


def test_chat_template(snapshot: SnapshotAssertion) -> None:
    template = ChatTemplate(
        messages=[
            MessageTemplate(role="system", content="hello, what's your name?"),
            MessageTemplate(role="user", content="My name is {{name}}}"),
            MessageTemplate(role="system", content="hello, {{name}} nice to meet you!"),
        ]
    )

    assert snapshot == template.render(name="John")

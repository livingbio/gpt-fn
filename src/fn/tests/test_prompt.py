from syrupy.assertion import SnapshotAssertion

from ..prompt import ChatTemplate, Message


def test_chat_template(snapshot: SnapshotAssertion) -> None:
    template = ChatTemplate(
        messages=[
            Message(role="system", content="hello, what's your name?"),
            Message(role="user", content="My name is {{name}}}"),
            Message(role="system", content="hello, {{name}} nice to meet you!"),
        ]
    )

    assert snapshot == template.render(name="John")

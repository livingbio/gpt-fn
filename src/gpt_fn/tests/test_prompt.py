from inspect import cleandoc

import pydantic
from syrupy.assertion import SnapshotAssertion

from ..prompt import ChatTemplate, MessageTemplate


def test_chat_template(snapshot: SnapshotAssertion) -> None:
    template = ChatTemplate(
        messages=[
            MessageTemplate(role="system", content="hello, what's your name?"),
            MessageTemplate(role="user", content="My name is {{name}}."),
            MessageTemplate(role="system", content="hello, {{name}} nice to meet you!"),
        ]
    )

    assert snapshot == template.render(name="John")


def test_complicated_chat_template(snapshot: SnapshotAssertion) -> None:
    class Article(pydantic.BaseModel):
        title: str
        content: str

    template = ChatTemplate(
        messages=[
            MessageTemplate(
                role="user",
                content=cleandoc(
                    """
                    Summarize the following articles:

                    keywords: {{ keywords | join(", ") }}

                    {% for article in articles %}
                    # {{ article.title }}
                    {{ article.content }}
                    {% endfor %}
                    """
                ),
            ),
        ]
    )

    assert snapshot == template.render(
        articles=[
            Article(title="Article 1", content="Content 1"),
            Article(title="Article 2", content="Content 2"),
            Article(title="Article 3", content="Content 3"),
        ],
        keywords=["keyword1", "keyword2", "keyword3"],
    )

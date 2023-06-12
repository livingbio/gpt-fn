from typing import TypedDict

import openai

from .exceptions import CompletionIncompleteError


class Message(TypedDict):
    role: str
    content: str


def chat_completion(
    messages: list[Message],
    max_tokens: int | None = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 1.0,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    user: str = "",
) -> str:
    kwargs = dict(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        user=user,
    )
    if max_tokens is not None:
        kwargs.update(max_tokens=max_tokens)

    completion = openai.ChatCompletion.create(**kwargs)

    output = completion.choices[0]
    output_message = output.message.content.strip()

    if output.finish_reason != "stop":
        raise CompletionIncompleteError(
            f"Incomplete completion. Finish reason: {output.finish_reason}",
            completion=completion,
        )

    return output_message

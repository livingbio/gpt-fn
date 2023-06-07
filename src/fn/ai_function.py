from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

import openai

from .utils.signature import FunctionSignature

T = TypeVar("T")
P = ParamSpec("P")


def ai_fn(
    fn: Callable[P, T],
) -> Callable[P, T]:
    sig = FunctionSignature(fn)

    @wraps(fn)
    def inner(*args: Any, **kwargs: Any) -> T:
        messages = [
            {
                "role": "system",
                "content": sig.instruction(),
            },
            {"role": "user", "content": sig.call_line(*args, **kwargs)},
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.0,
        )

        resp = response.choices[0].message["content"]
        return sig.parse(resp)

    return inner

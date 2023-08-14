from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

import openai.error

from .completion import chat_completion
from .exceptions import InvalidRequestError
from .prompt import ChatTemplate, MessageTemplate
from .utils.signature import FunctionSignature

T = TypeVar("T")
P = ParamSpec("P")


def ai_fn(
    fn: Callable[P, T],
) -> Callable[P, T]:
    sig = FunctionSignature(fn)

    @wraps(fn)
    def inner(*args: Any, **kwargs: Any) -> T:
        fn_call = sig.call_line(*args, **kwargs)
        template = ChatTemplate(
            messages=[
                MessageTemplate(role="system", content=sig.instruction()),
                MessageTemplate(role="user", content=fn_call),
            ]
        )

        try:
            resp = chat_completion(template.render(), temperature=0.0)
        except openai.error.InvalidRequestError as e:
            raise InvalidRequestError(fn_call) from e

        return sig.parse(resp)

    return inner

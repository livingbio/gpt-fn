from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

from .completion import chat_completion
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
        template = ChatTemplate(
            messages=[
                MessageTemplate(role="system", content=sig.instruction()),
                MessageTemplate(role="user", content=sig.call_line(*args, **kwargs)),
            ]
        )

        resp = chat_completion(template.render(), temperature=0.0)
        return sig.parse(resp)

    return inner

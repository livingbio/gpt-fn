from typing import Any, Callable, Type, TypedDict, TypeVar

import openai
import pydantic

from .exceptions import CompletionIncompleteError
from .utils import signature

T = TypeVar("T", bound=pydantic.BaseModel)


class Message(TypedDict):
    role: str
    content: str


class FunctionMessage(Message):
    name: str


def function_completion(
    messages: list[Message],
    max_tokens: int | None = None,
    model: str = "gpt-3.5-turbo-0613",
    temperature: float = 1.0,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stop: list[str] = [],
    user: str = "",
    functions: list[Callable[..., Any]] = [],
    function_call: str | dict[str, Any] = "auto",
) -> dict[str, Any] | None:
    assert functions, "functions must be a non-empty list of functions"

    kwargs = dict(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        user=user,
        stop=stop or None,
        functions=[signature.FunctionSignature(f).schema() for f in functions],
        function_call=function_call,
    )
    if max_tokens is not None:
        kwargs.update(max_tokens=max_tokens)

    response = openai.ChatCompletion.create(**kwargs)
    output = response.choices[0]
    message = output["message"]
    finish_reason = output.finish_reason

    if "function_call" in message and finish_reason in ["stop", "function_call"]:
        return message["function_call"]

    raise CompletionIncompleteError(
        f"Incomplete response. Max tokens: {max_tokens}, Finish reason: {finish_reason} Message:{message.content}",
        response=response,
        request=kwargs,
    )


async def afunction_completion(
    messages: list[Message],
    max_tokens: int | None = None,
    model: str = "gpt-3.5-turbo-0613",
    temperature: float = 1.0,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stop: list[str] = [],
    user: str = "",
    functions: list[Callable[..., Any]] = [],
    function_call: str | dict[str, Any] = "auto",
) -> dict[str, Any] | None:
    assert functions, "functions must be a non-empty list of functions"

    kwargs = dict(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        user=user,
        stop=stop or None,
        functions=[signature.FunctionSignature(f).schema() for f in functions],
        function_call=function_call,
    )
    if max_tokens is not None:
        kwargs.update(max_tokens=max_tokens)

    response = await openai.ChatCompletion.acreate(**kwargs)
    output = response.choices[0]
    message = output["message"]
    finish_reason = output.finish_reason

    if "function_call" in message and finish_reason in ["stop", "function_call"]:
        return message["function_call"]

    raise CompletionIncompleteError(
        f"Incomplete response. Max tokens: {max_tokens}, Finish reason: {finish_reason} Message:{message.content}",
        response=response,
        request=kwargs,
    )


def structural_completion(
    structure: Type[T],
    messages: list[Message],
    max_tokens: int | None = None,
    model: str = "gpt-3.5-turbo-0613",
    temperature: float = 1.0,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    user: str = "",
) -> T:
    function_call = {"name": "structural_response"}
    kwargs = dict(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        user=user,
        functions=[
            function_call
            | {
                "description": "Response to user in a structural way.",
                "parameters": structure.schema(),
            }
        ],
        function_call=function_call,
    )
    if max_tokens is not None:
        kwargs.update(max_tokens=max_tokens)

    response = openai.ChatCompletion.create(**kwargs)

    output = response.choices[0]
    message = output.message
    finish_reason = output.finish_reason

    if "function_call" in message and finish_reason == "stop":
        args = message.function_call.arguments
        return pydantic.parse_raw_as(structure, args)

    raise CompletionIncompleteError(
        f"Incomplete response. Max tokens: {max_tokens}, Finish reason: {finish_reason} Message:{message.content}",
        response=response,
        request=kwargs,
    )


async def astructural_completion(
    structure: Type[T],
    messages: list[Message],
    max_tokens: int | None = None,
    model: str = "gpt-3.5-turbo-0613",
    temperature: float = 1.0,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    user: str = "",
) -> T:
    function_call = {"name": "structural_response"}
    kwargs = dict(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        user=user,
        functions=[
            function_call
            | {
                "description": "Response to user in a structural way.",
                "parameters": structure.schema(),
            }
        ],
        function_call=function_call,
    )
    if max_tokens is not None:
        kwargs.update(max_tokens=max_tokens)

    response = await openai.ChatCompletion.acreate(**kwargs)

    output = response.choices[0]
    message = output.message
    finish_reason = output.finish_reason

    if "function_call" in message and finish_reason == "stop":
        args = message.function_call.arguments
        return pydantic.parse_raw_as(structure, args)

    raise CompletionIncompleteError(
        f"Incomplete response. Max tokens: {max_tokens}, Finish reason: {finish_reason} Message:{message.content}",
        response=response,
        request=kwargs,
    )


def chat_completion(
    messages: list[Message],
    max_tokens: int | None = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 1.0,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stop: list[str] = [],
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
        stop=stop or None,
    )
    if max_tokens is not None:
        kwargs.update(max_tokens=max_tokens)

    response = openai.ChatCompletion.create(**kwargs)

    output = response.choices[0]
    output_message = output.message.content.strip()

    if output.finish_reason != "stop":
        raise CompletionIncompleteError(
            f"Incomplete response. Max tokens: {max_tokens}, Finish reason: {output.finish_reason}",
            response=response,
            request=kwargs,
        )

    return output_message


async def achat_completion(
    messages: list[Message],
    max_tokens: int | None = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 1.0,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stop: list[str] = [],
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
        stop=stop or None,
    )
    if max_tokens is not None:
        kwargs.update(max_tokens=max_tokens)

    response = await openai.ChatCompletion.acreate(**kwargs)

    output = response.choices[0]
    output_message = output.message.content.strip()

    if output.finish_reason != "stop":
        raise CompletionIncompleteError(
            f"Incomplete response. Max tokens: {max_tokens}, Finish reason: {output.finish_reason}",
            response=response,
            request=kwargs,
        )

    return output_message

import os
from typing import Any, Callable, Type, TypedDict, TypeVar

import fuzzy_json
import openai
import pydantic
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from .exceptions import CompletionIncompleteError
from .utils import signature

T = TypeVar("T", bound=pydantic.BaseModel)


class Message(TypedDict):
    role: str
    content: str


class FunctionMessage(Message):
    name: str


class APISettings(pydantic.BaseModel):
    api_key: str = pydantic.Field(default_factory=lambda: openai.api_key)
    azure_endpoint: str = pydantic.Field(default_factory=lambda: openai.azure_endpoint)
    api_type: str = pydantic.Field(default_factory=lambda: openai.api_type)
    api_version: str | None = pydantic.Field(default_factory=lambda: openai.api_version)


def get_client(model: str, api_settings: APISettings = APISettings(), Async: bool = False) -> OpenAI | AzureOpenAI | AsyncOpenAI | AsyncAzureOpenAI:
    if api_settings.api_key is None:
        if os.environ.get("OPENAI_API_KEY"):
            api_settings.api_key = os.environ.get("OPENAI_API_KEY")
            api_settings.api_type = "open_ai"
        elif os.environ.get("AZURE_API_KEY"):
            api_settings.api_key = os.environ.get("AZURE_API_KEY")
            api_settings.api_type = "azure"
        else:
            raise ValueError("api_key must be set")

    if Async:
        if api_settings.api_type == "open_ai":
            return AsyncOpenAI(api_key=api_settings.api_key)
        elif api_settings.api_type == "azure":
            return AsyncAzureOpenAI(
                azure_endpoint=api_settings.azure_endpoint,
                api_key=api_settings.api_key,
                api_version=api_settings.api_version,
                azure_deployment=model,
            )
    else:
        if api_settings.api_type == "open_ai":
            return OpenAI(api_key=api_settings.api_key)
        elif api_settings.api_type == "azure":
            return AzureOpenAI(
                azure_endpoint=api_settings.azure_endpoint,
                api_key=api_settings.api_key,
                api_version=api_settings.api_version,
                azure_deployment=model,
            )

    raise ValueError(f"Unknown api_type {api_settings.api_type}")


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
    api_settings: APISettings = APISettings(),
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

    client = get_client(model, api_settings, Async=False)

    if max_tokens is not None:
        kwargs.update(max_tokens=max_tokens)

    response = client.chat.completions.create(**kwargs)  # type: ignore[call-overload]

    output = response.choices[0]
    message = output.message
    finish_reason = output.finish_reason

    if message.function_call is not None and finish_reason in ["stop", "function_call"]:
        return message.function_call

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
    api_settings: APISettings = APISettings(),
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

    client = get_client(model, api_settings, Async=True)

    if max_tokens is not None:
        kwargs.update(max_tokens=max_tokens)

    response = await client.chat.completions.create(**kwargs)  # type: ignore[call-overload]
    output = response.choices[0]
    message = output.message
    finish_reason = output.finish_reason

    if message.function_call is not None and finish_reason in ["stop", "function_call"]:
        return message.function_call

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
    auto_repair: bool = True,
    api_settings: APISettings = APISettings(),
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

    client = get_client(model, api_settings, Async=False)

    if api_settings.api_type != "open_ai":
        kwargs["deployment_id"] = model

    if max_tokens is not None:
        kwargs.update(max_tokens=max_tokens)

    response = client.chat.completions.create(**kwargs)  # type: ignore[call-overload]

    output = response.choices[0]
    message = output.message
    finish_reason = output.finish_reason

    if message.finish_reason is not None in message and finish_reason == "stop":
        args = message.function_call.arguments
        parsed_json = fuzzy_json.loads(args, auto_repair)

        return pydantic.parse_obj_as(structure, parsed_json)

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
    auto_repair: bool = True,
    api_settings: APISettings = APISettings(),
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

    client = get_client(model, api_settings, Async=True)

    if max_tokens is not None:
        kwargs.update(max_tokens=max_tokens)

    response = await client.chat.completions.create(**kwargs)  # type: ignore[call-overload]

    output = response.choices[0]
    message = output.message
    finish_reason = output.finish_reason

    if output.finish_reason is not None in message and finish_reason == "stop":
        args = message.function_call.arguments
        parsed_json = fuzzy_json.loads(args, auto_repair)

        return pydantic.parse_obj_as(structure, parsed_json)

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
    api_settings: APISettings = APISettings(),
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

    client = get_client(model, api_settings, Async=False)

    if max_tokens is not None:
        kwargs.update(max_tokens=max_tokens)

    response = client.chat.completions.create(**kwargs)  # type: ignore[call-overload]

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
    api_settings: APISettings = APISettings(),
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

    client = get_client(model, api_settings, Async=True)

    if max_tokens is not None:
        kwargs.update(max_tokens=max_tokens)

    response = await client.chat.completions.create(**kwargs)  # type: ignore[call-overload]

    output = response.choices[0]
    output_message = output.message.content.strip()

    if output.finish_reason != "stop":
        raise CompletionIncompleteError(
            f"Incomplete response. Max tokens: {max_tokens}, Finish reason: {output.finish_reason}",
            response=response,
            request=kwargs,
        )

    return output_message

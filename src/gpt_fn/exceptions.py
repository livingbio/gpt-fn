from typing import Any

from openai.openai_object import OpenAIObject


class GptFnError(Exception):
    pass


class CompletionIncompleteError(GptFnError):
    def __init__(self, msg: str, request: dict[str, Any], response: OpenAIObject) -> None:
        super().__init__(msg)
        self.response = response
        self.request = request

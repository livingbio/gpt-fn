from typing import Any


class GptFnError(Exception):
    pass


class CompletionIncompleteError(GptFnError):
    def __init__(self, msg: str, completion: Any) -> None:
        super().__init__(msg)
        self.completion = completion

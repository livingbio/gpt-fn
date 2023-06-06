import inspect
from typing import Any, Callable

import pydantic

from .pydantic_parser import PydanticParser


def clean_docstring(docstring: str) -> str:
    """Clean up docstring before sending to OpenAI."""
    output = []
    for line in docstring.split("\n"):
        output.append(line.strip())

    return "\n".join(output)


class FunctionSignature:
    """A helper class to parse function signature and generate instruction for prompt."""

    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn
        self.sig: inspect.Signature = inspect.signature(fn)
        return_annotation = self.sig.return_annotation

        assert return_annotation is not inspect.Signature.empty, f"Function {fn.__name__} must have return annotation"

        class UnpackModel(pydantic.BaseModel):
            ret: return_annotation  # type: ignore[valid-type]

        self.parser = PydanticParser[UnpackModel](pydantic_model=UnpackModel)

    def function_line(self) -> str:
        # NOTE: return type is instrcutive by parser
        f = str(self.sig).split("->")[0].strip()

        return f"def {self.fn.__name__}{f}:"

    def description(self) -> str:
        desc = inspect.cleandoc(self.fn.__doc__ or "")

        return desc

    def call_line(self, *args: Any, **kwargs: Any) -> str:
        # Bind the provided arguments to the function signature
        bound_args = self.sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Build input binds
        input_binds: list[str] = []
        for k, v in bound_args.arguments.items():
            input_binds.append(f"{k}={repr(v)}")

        return f"{self.fn.__name__}({', '.join(input_binds)})"

    def parse(self, text: str) -> Any:
        return self.parser.parse(text).ret

    def instruction(self) -> str:
        return clean_docstring(
            f"""You are now the following python function:
                    ```
                    # {self.description()}
                    {self.function_line()}
                    ```
                    Only respond with your `return` value.
                    {self.parser.get_format_instructions()}
                    """
        )

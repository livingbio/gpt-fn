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


def format_call_line(func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    args_list = [repr(arg) for arg in args]  # Convert arguments to their string representations
    kwargs_list = [f"{key}={repr(value)}" for key, value in kwargs.items()]  # Convert keyword arguments to string representations

    # Combine args and kwargs into a single list
    all_args = args_list + kwargs_list

    # Generate the formatted function call line
    call_line = f"{func.__name__}({', '.join(all_args)})"

    return call_line


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
        f = str(self.sig.replace(return_annotation=inspect.Signature.empty))

        return f"def {self.fn.__name__}{f}:"

    def description(self) -> str:
        desc = clean_docstring(self.fn.__doc__ or "")

        return desc

    def call_line(self, *args: Any, **kwargs: Any) -> str:
        return format_call_line(self.fn, *args, **kwargs)

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

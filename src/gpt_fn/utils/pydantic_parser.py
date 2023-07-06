import json
import re
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError
from pydantic.generics import GenericModel

T = TypeVar("T", bound=BaseModel)


class ParserError(Exception):
    pass


PYDANTIC_FORMAT_INSTRUCTIONS = """- The output should be formatted as a JSON instance that conforms to the JSON schema below.

- As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

- Here is the output schema:
```
{schema}
```"""


class PydanticParser(GenericModel, Generic[T]):
    pydantic_model: type[T]

    def parse(self, text: str) -> T:
        try:
            # Greedy search for 1st json candidate.
            match = re.search(r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL)
            json_str = ""
            if match:
                json_str = match.group()
            json_object = json.loads(json_str)
            return self.pydantic_model.parse_obj(json_object)
        except (json.JSONDecodeError, ValidationError) as e:
            name = self.pydantic_model.__name__
            msg = f"Failed to parse {name} from completion {text}. Got: {e}"
            raise ParserError(msg)

    def get_format_instructions(self) -> str:
        schema = self.pydantic_model.schema()

        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure json in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema)

        return PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)

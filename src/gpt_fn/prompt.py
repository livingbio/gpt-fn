from typing import Any, Literal

import jinja2
from pydantic import BaseModel

from .completion import FunctionMessage, Message


class MessageTemplate(BaseModel):
    role: Literal["system", "user", "assistant", "function"]
    content: str
    name: str = ""

    def render(self, **kwargs: Any) -> Message:
        content = jinja2.Template(self.content).render(**kwargs)
        if self.role == "function":
            return FunctionMessage(
                role=self.role,
                content=content,
                name=self.name,
            )
        return Message(role=self.role, content=content)

    class Config:
        # remove whitespace
        anystr_strip_whitespace = True


class ChatTemplate(BaseModel):
    messages: list[MessageTemplate]

    def render(self, **kwargs: Any) -> list[Message]:
        return [m.render(**kwargs) for m in self.messages]

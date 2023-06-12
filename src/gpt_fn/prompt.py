from typing import Literal

import jinja2
from pydantic import BaseModel

from .completion import Message


class MessageTemplate(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

    def render(self, **kwargs: str) -> Message:
        return {
            "role": self.role,
            "content": jinja2.Template(self.content).render(**kwargs),
        }

    class Config:
        # remove whitespace
        anystr_strip_whitespace = True


class ChatTemplate(BaseModel):
    messages: list[MessageTemplate]

    def render(self, **kwargs: str) -> list[Message]:
        return [m.render(**kwargs) for m in self.messages]

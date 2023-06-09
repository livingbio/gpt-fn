from typing import Any, Literal

import jinja2
from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

    def render(self, **kwargs: str) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": jinja2.Template(self.content).render(**kwargs),
        }

    class Config:
        # remove whitespace
        anystr_strip_whitespace = True


class ChatTemplate(BaseModel):
    messages: list[Message]

    def render(self, **kwargs: str) -> list[dict[str, Any]]:
        return [m.render(**kwargs) for m in self.messages]

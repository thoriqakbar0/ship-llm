from typing import List, Union, Literal
from pydantic import BaseModel, AnyUrl

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

    def to_dict(self):
        return {"type": self.type, "text": self.text}

class ImageUrlContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: dict

    def to_dict(self):
        return {"type": self.type, "image_url": self.image_url}

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: List[Union[TextContent, ImageUrlContent]]

    @property
    def text(self) -> str:
        return " ".join([block.text for block in self.content if isinstance(block, TextContent)])

    @property
    def images(self) -> List[dict]:
        return [block.image_url for block in self.content if isinstance(block, ImageUrlContent)]

    def to_dict(self):
        return {
            "role": self.role,
            "content": [c.to_dict() for c in self.content]
        }

import re
from typing import Union, List, Any
from urllib.parse import urlparse
from .models import Message, TextContent, ImageUrlContent

URL_REGEX = re.compile(
    r'^(?:http|ftp)s?://'  # http:// or https:// or ftp:// or ftps://
    r'(?:\S+(?::\S*)?@)?'  # optional username:password@
    r'(?:(?:(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,6})|localhost|\d{1,3}(?:\.\d{1,3}){3})'  # domain
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE  # path
)

def is_valid_url(url: str) -> bool:
    """Check if a string is a valid URL"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc]) and bool(URL_REGEX.match(url))
    except:
        return False

def format_messages(content: Union[str, List[Union[str, dict]], Message], system_prompt: str = None) -> List[Message]:
    """Format content into a list of messages for the AI"""
    messages = []
    
    # Add system prompt if provided
    if system_prompt:
        messages.append(Message(
            role="system",
            content=[TextContent(text=system_prompt)]
        ))
    
    # Handle different content types
    if isinstance(content, str):
        messages.append(Message(
            role="user",
            content=[TextContent(text=content)]
        ))
    elif isinstance(content, Message):
        messages.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                messages.append(Message(
                    role="user",
                    content=[TextContent(text=item)]
                ))
            elif isinstance(item, dict):
                messages.append(Message(**item))
            elif isinstance(item, Message):
                messages.append(item)
            else:
                raise ValueError(f"Invalid content type: {type(item)}. Expected str, dict, or Message.")
    else:
        raise ValueError(f"Invalid content type: {type(content)}. Expected str, list, dict, or Message.")
    
    return messages

def user(*content: Union[str, Any]) -> Message:
    """Create a user message with text and/or image URLs"""
    message_content = []
    for item in content:
        if isinstance(item, str):
            if is_valid_url(item):
                message_content.append(ImageUrlContent(image_url=item))
            else:
                message_content.append(TextContent(text=item))
        else:
            message_content.append(TextContent(text=str(item)))
    return Message(role="user", content=message_content)

def system(content: str) -> Message:
    """Create a system message"""
    return Message(role="system", content=[TextContent(text=content)])

def assistant(content: str) -> Message:
    """Create an assistant message"""
    return Message(role="assistant", content=[TextContent(text=content)])

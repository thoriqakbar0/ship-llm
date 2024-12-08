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

def format_messages(content: Union[str, List[Union[str, dict, Message]], Message], system_prompt: str = None) -> List[Message]:
    """
    Format content into a list of messages for the AI.
    Handles text and image content following OpenAI's multimodal format.
    
    Args:
        content: Input content (str, list, dict, or Message)
        system_prompt: Optional system prompt
        
    Returns:
        List[Message]: Formatted messages for the AI
    """
    # Pre-allocate messages list with initial capacity
    messages = []
    
    # Add system prompt if provided (most common case first)
    if system_prompt:
        messages.append(system(system_prompt))

    # Fast path for single string/message (most common case)
    if isinstance(content, str):
        messages.append(user(content))
        return messages
    
    if isinstance(content, Message):
        messages.append(content)
        return messages

    # Handle list or dict content
    if isinstance(content, dict):
        content = [content]
    elif not isinstance(content, list):
        raise ValueError(f"Invalid content type: {type(content)}. Expected str, list, dict, or Message.")

    # Process list content with type checking
    role_handlers = {
        "system": system,
        "assistant": assistant,
        "user": user
    }

    for item in content:
        if isinstance(item, Message):
            messages.append(item)
        elif isinstance(item, str):
            messages.append(user(item))
        elif isinstance(item, dict):
            # Handle OpenAI message format
            role = item.get("role", "user")
            content_data = item.get("content", "")
            handler = role_handlers.get(role, user)
            messages.append(handler(content_data))

    return messages

def user(*content: Union[str, Any]) -> Message:
    """Create a user message with text and/or image URLs"""
    message_content = []
    for item in content:
        if isinstance(item, str):
            if is_valid_url(item):
                message_content.append(ImageUrlContent(
                    image_url={"url": item}
                ))
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

import asyncio
from functools import wraps
from typing import Union, Optional, List, Callable, Any, AsyncGenerator, TypeVar, Literal, Type, overload
import instructor
from openai import OpenAI, AzureOpenAI, AsyncOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion
from openai.types import ChatModel
from pydantic import BaseModel
import threading

from .models import Message, TextContent
from .decorators import templated_docstring
from .cache import LRUCache, create_cache_key
from .utils import format_messages
from .clients import OpenAIClient, AsyncClient, SyncClient
from .stream import StreamReturn, StreamManager

T = TypeVar('T', bound=BaseModel)

class AI:
    def __init__(
        self,
        client: Optional[OpenAIClient] = None,
        model: Optional[Union[str, ChatModel]] = None,
        max_retries: int = 3,
        max_workers: int = 4,
        cache_size: int = 1000
    ):
        """Initialize AI with optional OpenAI client and configuration."""
        self.model = model
        self.json_client = instructor.patch(client) if client else None
        self._message_cache = LRUCache(cache_size)
        self._completion_cache = LRUCache(cache_size)
        
        # Initialize clients and stream manager
        self.async_client = AsyncClient(client, model, max_retries, max_workers) if isinstance(client, (AsyncOpenAI, AsyncAzureOpenAI)) else None
        self.sync_client = SyncClient(client, model, max_retries, max_workers) if isinstance(client, (OpenAI, AzureOpenAI)) else None
        self.stream_manager = StreamManager()

    def text(
        self,
        stream: bool = False,
        model: Optional[Union[str, ChatModel]] = None,
        use_cache: bool = True,
        **llm_params
    ) -> Callable[[Callable[..., Any]], Callable[..., Union[str, StreamReturn, AsyncGenerator[str, None]]]]:
        """Decorator for text completion functions."""
        def decorator(func: Callable[..., Any]) -> Callable[..., Union[str, StreamReturn, AsyncGenerator[str, None]]]:
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Union[str, AsyncGenerator[str, None]]:
                if not self.async_client:
                    raise ValueError("AsyncClient not initialized. Use AsyncOpenAI or AsyncAzureOpenAI client.")
                
                # Get system prompt from docstring
                system_prompt = func.__doc__.strip() if func.__doc__ else None
                
                # Execute the function
                content = await func(*args, **kwargs)
                messages = format_messages(content, system_prompt)
                
                # Check cache for non-streaming requests
                if use_cache and not stream:
                    cache_key = create_cache_key({"messages": messages, "model": model or self.model})
                    cached_result = self._completion_cache.get(cache_key)
                    if cached_result:
                        return cached_result.choices[0].message.content or ""

                # Execute completion
                response = await self.async_client.complete(
                    messages=messages,
                    stream=stream,
                    model=model or self.model,
                    **llm_params
                )

                if stream:
                    return self.stream_manager.create_async_stream_generator(response)
                else:
                    if response.choices:
                        self._completion_cache.put(cache_key, response)
                        return response.choices[0].message.content or ""
                    return ""

            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> Union[str, StreamReturn]:
                if not self.sync_client:
                    raise ValueError("SyncClient not initialized. Use OpenAI or AzureOpenAI client.")
                
                # Get system prompt from docstring
                system_prompt = func.__doc__.strip() if func.__doc__ else None
                
                # Handle async/sync functions
                if asyncio.iscoroutinefunction(func):
                    loop = asyncio.get_event_loop()
                    content = loop.run_until_complete(func(*args, **kwargs))
                else:
                    content = func(*args, **kwargs)
                    
                messages = format_messages(content, system_prompt)
                
                # Check cache for non-streaming requests
                if use_cache and not stream:
                    cache_key = create_cache_key({"messages": messages, "model": model or self.model})
                    cached_result = self._completion_cache.get(cache_key)
                    if cached_result:
                        return cached_result.choices[0].message.content or ""

                # Execute completion
                response = self.sync_client.complete(
                    messages=messages,
                    stream=stream,
                    model=model or self.model,
                    **llm_params
                )

                if stream:
                    cancel_event = threading.Event()
                    return StreamReturn(
                        generator=self.stream_manager.create_sync_stream_generator(response, cancel_event),
                        cancel_event=cancel_event
                    )
                else:
                    if response.choices:
                        self._completion_cache.put(cache_key, response)
                        return response.choices[0].message.content or ""
                    return ""

            return async_wrapper if self.async_client else sync_wrapper
        return decorator

    @overload
    def structured(
        self,
        response_format: Type[T],
        stream: Literal[False] = False,
        *,
        model: Optional[Union[str, ChatModel]] = None,
        stream_mode: Literal["partial", "iterable"] = "partial",
        use_cache: bool = True,
        **llm_params
    ) -> Callable[[Callable[..., Any]], Callable[..., T]]:
        ...

    @overload
    def structured(
        self,
        response_format: Type[T],
        stream: Literal[True],
        *,
        model: Optional[Union[str, ChatModel]] = None,
        stream_mode: Literal["partial", "iterable"] = "partial",
        use_cache: bool = True,
        **llm_params
    ) -> Callable[[Callable[..., Any]], Callable[..., AsyncGenerator[T, None]]]:
        ...

    def structured(
        self,
        response_format: Type[T],
        stream: bool = False,
        *,
        model: Optional[Union[str, ChatModel]] = None,
        stream_mode: Literal["partial", "iterable"] = "partial",
        use_cache: bool = True,
        **llm_params
    ) -> Callable[[Callable[..., Any]], Callable[..., Union[T, AsyncGenerator[T, None]]]]:
        """Decorator for structured completion functions that return Pydantic models."""
        def decorator(func: Callable[..., Any]) -> Callable[..., Union[T, AsyncGenerator[T, None]]]:
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Union[T, AsyncGenerator[T, None]]:
                if not self.async_client:
                    raise ValueError("AsyncClient not initialized. Use AsyncOpenAI or AsyncAzureOpenAI client.")
                
                # Get system prompt from docstring
                system_prompt = func.__doc__.strip() if func.__doc__ else None
                
                # Execute the function
                content = await func(*args, **kwargs)
                messages = format_messages(content, system_prompt)
                
                # Check cache for non-streaming requests
                if use_cache and not stream:
                    cache_key = create_cache_key({
                        "messages": messages,
                        "model": model or self.model,
                        "response_format": response_format.__name__
                    })
                    cached_result = self._completion_cache.get(cache_key)
                    if cached_result:
                        return cached_result

                # Execute structured completion
                response = await self.async_client.complete_structured(
                    messages=messages,
                    response_model=response_format,
                    stream=stream,
                    stream_mode=stream_mode,
                    model=model or self.model,
                    **llm_params
                )

                if not stream:
                    self._completion_cache.put(cache_key, response)
                return response

            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                if not self.sync_client:
                    raise ValueError("SyncClient not initialized. Use OpenAI or AzureOpenAI client.")
                
                # Get system prompt from docstring
                system_prompt = func.__doc__.strip() if func.__doc__ else None
                
                # Handle async/sync functions
                if asyncio.iscoroutinefunction(func):
                    loop = asyncio.get_event_loop()
                    content = loop.run_until_complete(func(*args, **kwargs))
                else:
                    content = func(*args, **kwargs)
                    
                messages = format_messages(content, system_prompt)
                
                # Check cache for non-streaming requests
                if use_cache:
                    cache_key = create_cache_key({
                        "messages": messages,
                        "model": model or self.model,
                        "response_format": response_format.__name__
                    })
                    cached_result = self._completion_cache.get(cache_key)
                    if cached_result:
                        return cached_result

                # Execute structured completion
                response = self.sync_client.complete_structured(
                    messages=messages,
                    response_model=response_format,
                    model=model or self.model,
                    **llm_params
                )

                self._completion_cache.put(cache_key, response)
                return response

            return async_wrapper if self.async_client else sync_wrapper
        return decorator

    def stop_stream(self):
        """Stop the currently active stream"""
        self.stream_manager.stop_stream()

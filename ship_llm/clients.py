from typing import Union, Optional, List, AsyncGenerator, TypeVar, Type
import asyncio
import backoff
from openai import OpenAI, AzureOpenAI, AsyncOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types import ChatModel
from concurrent.futures import ThreadPoolExecutor
import instructor
from pydantic import BaseModel

from .models import Message

OpenAIClient = Union[OpenAI, AzureOpenAI, AsyncOpenAI, AsyncAzureOpenAI]
T = TypeVar('T', bound=BaseModel)

class BaseClient:
    def __init__(
        self,
        client: Optional[OpenAIClient] = None,
        model: Optional[Union[str, ChatModel]] = None,
        max_retries: int = 3,
        max_workers: int = 4
    ):
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.json_client = instructor.patch(client) if client else None

    def _prepare_messages(self, messages: List[Message]) -> List[dict]:
        """Convert Message objects to OpenAI API format"""
        return [msg.to_dict() for msg in messages]

class AsyncClient(BaseClient):
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def complete(
        self,
        messages: List[Message],
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Execute async completion with retry logic"""
        if not isinstance(self.client, (AsyncOpenAI, AsyncAzureOpenAI)):
            raise ValueError("AsyncClient requires AsyncOpenAI or AsyncAzureOpenAI client")

        api_messages = self._prepare_messages(messages)
        params = {
            "messages": api_messages,
            "model": self.model,
            "stream": stream,
            **kwargs
        }

        return await self.client.chat.completions.create(**params)
    
    async def complete_structured(
        self,
        messages: List[Message],
        response_format: Type[T],
        stream: bool = False,
        stream_mode: str = "partial",
        **kwargs
    ) -> Union[T, AsyncGenerator[T, None]]:
        """Execute async structured completion"""
        if not self.json_client:
            raise ValueError("JSON client not initialized")

        api_messages = self._prepare_messages(messages)
        params = {
            "messages": api_messages,
            "model": self.model,
            **kwargs
        }

        if not stream:
            return await self.json_client.chat.completions.create(
                response_model=response_format,
                **params
            )
        elif stream_mode == "partial":
            response = self.json_client.chat.completions.create_partial(
                response_model=response_format,
                **params
            )
            async for chunk in response:
                if chunk is not None:
                    yield chunk
        else:  # stream_mode == "iterable"
            async for chunk in self.json_client.chat.completions.create_iterable(
                response_model=response_format,
                **params
            ):
                yield chunk

class SyncClient(BaseClient):
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def complete(
        self,
        messages: List[Message],
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Execute sync completion with retry logic"""
        if not isinstance(self.client, (OpenAI, AzureOpenAI)):
            raise ValueError("SyncClient requires OpenAI or AzureOpenAI client")

        api_messages = self._prepare_messages(messages)
        params = {
            "messages": api_messages,
            "model": self.model,
            "stream": stream,
            **kwargs
        }

        return self.client.chat.completions.create(**params)
    
    def complete_structured(
        self,
        messages: List[Message],
        response_format: Type[T],
        **kwargs
    ) -> T:
        """Execute sync structured completion"""
        if not self.json_client:
            raise ValueError("JSON client not initialized")

        api_messages = self._prepare_messages(messages)
        params = {
            "messages": api_messages,
            "model": self.model,
            **kwargs
        }

        return self.json_client.chat.completions.create(
            response_model=response_format,
            **params
        )

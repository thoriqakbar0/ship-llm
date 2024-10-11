from functools import wraps, lru_cache
from pydantic import BaseModel, ValidationError
from pydantic.networks import AnyUrl
import instructor
from typing import List, Union, Literal, Optional, Any, Generator, TypeVar, Callable, Type, overload
import threading
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming, CompletionCreateParamsStreaming
import re
from openai.types import ChatModel
from urllib.parse import urlparse
import inspect
from typing import get_type_hints

OpenAIClient = Union[OpenAI, AzureOpenAI]
T = TypeVar('T', bound=BaseModel)

# Compile the URL regex once at the module level
URL_REGEX = re.compile(
  r'^(?:http|ftp)s?://'  # http:// or https://
  r'(?:\S+(?::\S*)?@)?'  # user and password
  r'(?:'
  r'(?:(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,6})'  # domain...
  r'|'
  r'localhost'  # localhost...
  r'|'
  r'\d{1,3}(?:\.\d{1,3}){3}'  # ...or ip
  r')'
  r'(?::\d+)?'  # optional port
  r'(?:/?|[/?]\S+)$', re.IGNORECASE
)


class TextContent(BaseModel):
  type: Literal["text"] = "text"
  text: str


class ImageUrlContent(BaseModel):
  type: Literal["image_url"] = "image_url"
  image_url: dict


class Message(BaseModel):
  role: Literal["system", "user", "assistant"]
  content: List[Union[TextContent, ImageUrlContent]]

  @property
  def text(self) -> str:
      return " ".join([block.text for block in self.content if isinstance(block, TextContent)])

  @property
  def images(self) -> List[str]:
      return [block.image_url["url"] for block in self.content if isinstance(block, ImageUrlContent)]


class StreamReturn:
  def __init__(self, generator: Generator[str, None, None], cancel_event: threading.Event):
      self.generator = generator
      self.cancel_event = cancel_event

  def __iter__(self):
      return self.generator

  def __next__(self):
      return next(self.generator)

  def __getitem__(self, index):
      return next(self.generator)


def templated_docstring(template):
  def decorator(func):
      sig = inspect.signature(func)
      type_hints = get_type_hints(func)

      param_docs = []
      for name, param in sig.parameters.items():
          param_type = type_hints.get(name, Any).__name__
          default = param.default if param.default is not param.empty else None
          param_docs.append(f"{name} ({param_type}): Description for {name}")
          if default is not None:
              param_docs[-1] += f" (default: {default})"

      param_doc = "\n    ".join(param_docs)

      func.__doc__ = template.format(
          func_name=func.__name__,
          params=param_doc,
          return_type=type_hints.get('return', Any).__name__
      )
      return func
  return decorator


class AI:
  def __init__(self, client: Optional[OpenAIClient] = None, model: Optional[Union[str, ChatModel]] = None):
      self.client: Optional[OpenAIClient] = client
      self.json_client = instructor.patch(client) if client else None
      self.model: Optional[Union[str, ChatModel]] = model
      self.active_stream: Optional[threading.Event] = None

  def _check_docstring_signature(self, func):
      doc = func.__doc__
      if doc:
          placeholders = re.findall(r'\{(\w+)\}', doc)
          sig = inspect.signature(func)
          params = list(sig.parameters.keys())

          for placeholder in placeholders:
              if placeholder not in params:
                  raise ValueError(f"Docstring placeholder '{placeholder}' not found in function parameters")

  @overload
  def text(
      self,
      stream: Literal[False] = False,
      model: Optional[Union[str, ChatModel]] = None,
      client: Optional[OpenAIClient] = None,
      **llm_params
  ) -> Callable[[Callable[..., Any]], Callable[..., str]]:
      ...

  @overload
  def text(
      self,
      stream: Literal[True],
      model: Optional[Union[str, ChatModel]] = None,
      client: Optional[OpenAIClient] = None,
      **llm_params
  ) -> Callable[[Callable[..., Any]], Callable[..., StreamReturn]]:
      ...

  def text(
      self,
      stream: bool = False,
      model: Optional[Union[str, ChatModel]] = None,
      client: Optional[OpenAIClient] = None,
      **llm_params
  ) -> Callable[[Callable[..., Any]], Callable[..., Union[str, StreamReturn]]]:
      def decorator(func: Callable[..., Any]) -> Callable[..., Union[str, StreamReturn]]:
          @templated_docstring("""
          {func_name}

          Parameters:
          {params}

          Returns:
          {return_type}: The generated text response or stream
          """)
          @wraps(func)
          def wrapper(*args, **kwargs) -> Union[str, StreamReturn]:
              self._check_docstring_signature(func)
              system_prompt = func.__doc__.strip() if func.__doc__ else None
              content = func(*args, **kwargs)

              if not isinstance(content, (str, list, dict, Message)):
                  raise ValueError(f"Invalid content type: {type(content)}. Expected str, list, dict, or Message.")

              messages = _format_messages(content, system_prompt)

              cancel_event = threading.Event()
              used_client = client or self.client
              if not used_client:
                  raise ValueError("No client provided. AI functionality will not work without a client.")

              response = used_client.chat.completions.create(
                  model=model or self.model or "",
                  messages=messages,
                  stream=stream,
                  **llm_params
              )

              if stream:
                  def stream_generator() -> Generator[str, None, None]:
                      try:
                          for chunk in response:
                              if isinstance(chunk, ChatCompletionChunk) and chunk.choices:
                                  if cancel_event.is_set():
                                      break
                                  delta = chunk.choices[0].delta
                                  if delta.content:
                                      yield delta.content
                      finally:
                          pass

                  return StreamReturn(generator=stream_generator(), cancel_event=cancel_event)
              else:
                  content = ""
                  if isinstance(response, ChatCompletion) and response.choices:
                      content = response.choices[0].message.content or ""
                  return content

          return wrapper
      return decorator

  @overload
  def structured(
      self,
      response_format: Type[T],
      stream: Literal[False] = False,
      *,
      model: Optional[Union[str, ChatModel]] = None,
      client: Optional[OpenAIClient] = None,
      stream_mode: Literal["partial", "iterable"] = "partial",
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
      client: Optional[OpenAIClient] = None,
      stream_mode: Literal["partial", "iterable"] = "partial",
      **llm_params
  ) -> Callable[[Callable[..., Any]], Callable[..., Generator[T, None, None]]]:
      ...

  def structured(
      self,
      response_format: Type[T],
      stream: bool = False,
      *,
      model: Optional[Union[str, ChatModel]] = None,
      client: Optional[OpenAIClient] = None,
      stream_mode: Literal["partial", "iterable"] = "partial",
      **llm_params
  ) -> Callable[[Callable[..., Any]], Callable[..., Union[T, Generator[T, None, None]]]]:
      def decorator(func: Callable[..., Any]) -> Callable[..., Union[T, Generator[T, None, None]]]:
          @templated_docstring("""
          {func_name}

          Parameters:
          {params}

          Returns:
          {return_type}: The structured response based on the specified format
          """)
          @wraps(func)
          def wrapper(*args, **kwargs) -> Union[T, Generator[T, None, None]]:
              self._check_docstring_signature(func)
              if not response_format:
                  raise ValueError("response_format is required for structured AI responses.")
              system_prompt = func.__doc__.strip() if func.__doc__ else None
              content = func(*args, **kwargs)
              messages = _format_messages(content, system_prompt)

              used_client = client or self.client
              if not used_client:
                  raise ValueError("No client provided. AI functionality will not work without a client.")

              patched_client = instructor.from_openai(used_client)

              if not stream:
                  return patched_client.chat.completions.create(
                      model=model or self.model or "",
                      messages=messages,
                      response_model=response_format,
                      **llm_params
                  )
              else:
                  if stream_mode == "partial":
                      response = patched_client.chat.completions.create_partial(
                          model=model or self.model or "",
                          messages=messages,
                          response_model=response_format,
                          **llm_params
                      )

                      def stream_generator() -> Generator[T, None, None]:
                          for chunk in response:
                              if chunk is not None:
                                  yield chunk

                      return stream_generator()
                  elif stream_mode == "iterable":
                      return patched_client.chat.completions.create_iterable(
                          model=model or self.model or "",
                          messages=messages,
                          response_model=response_format,
                          **llm_params
                      )
                  else:
                      raise ValueError(f"Invalid stream_mode: {stream_mode}")

          return wrapper
      return decorator

  def stop_stream(self):
      if self.active_stream:
          self.active_stream.set()
          self.active_stream = None


def _format_messages(content, system_prompt=None):
  messages = []
  if system_prompt:
      messages.append({"role": "system", "content": system_prompt})

  if isinstance(content, list):
      for item in content:
          if isinstance(item, dict):
              messages.append(item)
          elif isinstance(item, Message):
              messages.append(item.model_dump(mode='json'))
  elif isinstance(content, str):
      messages.append({"role": "user", "content": content})
  elif isinstance(content, dict) and "role" in content:
      messages.append(content)
  elif isinstance(content, Message):
      messages.append(content.model_dump(mode='json'))

  return messages


def system(content: str) -> Message:
  return Message(role="system", content=[TextContent(text=content)])


@lru_cache(maxsize=128)
def is_valid_url(url: str) -> bool:
  return re.match(URL_REGEX, url) is not None


def user(*content: Union[str, AnyUrl]) -> Message:
  formatted_content = []
  for item in content:
      if isinstance(item, str):
          if is_valid_url(item):
              formatted_content.append(ImageUrlContent(image_url={"url": item}))
          else:
              formatted_content.append(TextContent(text=item))
      elif isinstance(item, AnyUrl):
          formatted_content.append(ImageUrlContent(image_url={"url": str(item)}))
      else:
          formatted_content.append(TextContent(text=str(item)))
  return Message(role="user", content=formatted_content)


def assistant(content: str) -> Message:
  return Message(role="assistant", content=[TextContent(text=content)])

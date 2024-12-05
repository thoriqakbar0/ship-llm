import threading
from typing import Generator, AsyncGenerator
from openai.types.chat import ChatCompletionChunk

class StreamReturn:
    def __init__(self, generator: Generator[str, None, None], cancel_event: threading.Event):
        self.generator = generator
        self.cancel_event = cancel_event

    def __iter__(self):
        return self

    def __next__(self):
        if self.cancel_event.is_set():
            raise StopIteration
        return next(self.generator)

    def __getitem__(self, index):
        return list(self)[index]

class StreamManager:
    def __init__(self):
        self.active_stream: Optional[threading.Event] = None

    def stop_stream(self):
        """Stop the currently active stream if any"""
        if self.active_stream:
            self.active_stream.set()
            self.active_stream = None

    def create_sync_stream_generator(self, response, cancel_event: threading.Event) -> Generator[str, None, None]:
        """Create a generator for synchronous streaming"""
        try:
            for chunk in response:
                if cancel_event.is_set():
                    break
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"Error in stream generator: {str(e)}")
            raise
        finally:
            cancel_event.set()

    async def create_async_stream_generator(self, response) -> AsyncGenerator[str, None]:
        """Create a generator for asynchronous streaming"""
        try:
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"Error in async stream generator: {str(e)}")
            raise

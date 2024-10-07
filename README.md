# Ship LLM: Ship Fast, Ship Now - Clean LLM Interface for Rapid Deployment

Ship LLM is a Python library that simplifies working with Large Language Models (LLMs) by providing a clean, predictable, and type-safe interface. It's designed to make LLM interactions faster and more efficient, allowing developers to confidently ship their AI-powered applications.

Ship LLM is heavily inspired by: [ell](https://github.com/madcowd/ell), a lightweight, functional prompt engineering framework. Do check them out!

[Ship LLM notebook example](https://colab.research.google.com/drive/1Myj3waieceS1ymUDOy44VX6CDmvja0ha?usp=sharing)

### Why not just use ell?
As much as I liked ell, ell only supports openai. I want ell api for other providers and also the nice things instructor provided. Hence, a new thing.

## Supported Providers

Currently, Ship LLM has been tested with OpenAI and Azure OpenAI. We plan to expand support to other providers through integration with LiteLLM in the future.

## Key Features

- **One LLM call per function**: Ship LLM's philosophy is to make each function correspond to a single LLM call, making your code more predictable and easier to work with.
- **Clean decorator syntax**: Use simple decorators to define your LLM interactions.
- **Dynamic parameter injection**: Easily inject parameters into your prompts using string formatting.
- **Type-safe structured outputs**: Get structured, type-safe responses from your LLM calls.
- **Streaming support**: Efficiently handle streaming responses from LLMs.
- **Faster to ship**: Ship LLM minimize the interaction with LLMs, making it faster than using the raw OpenAI Python API.
a
## Installation

```bash
pip install ship-llm
```

## Quick Start

Here's a simple example of how to use Ship LLM:

```python
from openai import OpenAI
from ship_llm import AI, user, system


client = OpenAI(api_key="xxx")

ai = AI(client)

@ai.text()
def summarize_text(text: str):
    """
    Summarize the given text.
    text sent from the chatbot: {text}
    """
    return "Summarize this text"

summary = summarize_text("Long text to summarize...")
print(summary)
```
## Why Ship LLM allows you to ship fast

Ship LLM provides a minimal API interface compared to the raw OpenAI Python API, allowing for faster iteration and cleaner code:

### OpenAI Python API:

```python
client.chat.completions.create(
    model=model,
    messages=messages,
    stream=stream,
    **llm_params
)
```

### Ship LLM:

```python
@ai.text()
def my_function():
    """Your prompt here"""
    return "User message"
```

Ship LLM optimizes the API call process, reduces boilerplate code, and handles parameter management more efficiently, resulting in faster development and a cleaner codebase.

## Predictable structured output, courtesy of Instructor

Hurry's structured output functionality use [Instructor](https://github.com/jxnl/instructor) in the background. We've incorporated similar concepts to provide type-safe, structured outputs that make working with AI responses more reliable and easier to manage.

## Type-Safe Structured Outputs

Ship LLM supports type-safe structured outputs, making it easier to work with complex response formats:

```python
from pydantic import BaseModel

class SubjectClassifier(BaseModel):
    subject: str
    confidence: float
    reasoning: str

@ai.structured(response_format=SubjectClassifier)
def classifier(query: str, subject: Optional[str] = None):
    """
    Route the query to the correct database.
    subject sent from the chatbot: {subject}
    """
    return f"Classify this query: {query}"

result = classifier("What is photosynthesis?")
print(f"Subject: {result.subject}, Confidence: {result.confidence}")
```

## Streaming Support

Ship LLM provides easy-to-use streaming support:

```python
@ai.text(stream=True)
def streaming_function():
    """Your prompt here"""
    return "Generate a long response"

for chunk in streaming_function():
    print(chunk, end="", flush=True)
```

## More Examples

### Using system, user, and assistant functions

```python
@ai.text()
def conversation():
    """You are a helpful assistant."""
    return [
        user("Hi, I'm planning a trip to Paris."),
        assistant("That's exciting! Paris is a beautiful city. What would you like to know about planning your trip?"),
        user("What are the top 3 must-visit attractions?")
    ]

result = conversation() # When visiting Paris, here are three must-visit attractions:
```

### Mixed types of message interface

```python
@ai.text()
def mixed_messages():
    return [
        system("You're a helpful travel assistant with knowledge about Paris."),
        user("Hi, I'm planning a trip to Paris."),
        {"role": "user", "content": "What are the top 3 must-visit attractions?"},
        assistant("Certainly! The top 3 must-visit attractions in Paris are:\n1. The Eiffel Tower\n2. The Louvre Museum\n3. Notre-Dame Cathedral"),
        user("Tell me more about the Louvre.")
    ]

result = mixed_messages()
print(result)
```

### Template string for dynamic parameter injection

```python
@ai.structured(SubjectClassifier)
def try_classifier(query: str, subject: Optional[str] = None, chapter_id: Optional[int] = None):
    """
    Route the query to the correct database. While supplied with subject, it still doesn't guarantee the correct database.

    subject sent from the chatbot: {subject}
    chapter_id: {chapter_id}
    """
    return f"user query this while chatting with assistant: {query}"

result = try_classifier("What is photosynthesis?", subject="Biology", chapter_id=5)
print(result)
```

## Contributing

Just open an issue, and let's discuss it.

## License

Ship LLM is released under the [MIT License](LICENSE).

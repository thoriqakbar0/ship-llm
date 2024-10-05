# Hurry: Confidently ship on a Friday with a clean LLM interface

Hurry is a Python library that simplifies working with Large Language Models (LLMs) by providing a clean, predictable, and type-safe interface. It's designed to make LLM interactions faster and more efficient, allowing developers to confidently ship their AI-powered applications.

## Key Features

- **One LLM call per function**: Hurry's philosophy is to make each function correspond to a single LLM call, making your code more predictable and easier to work with.
- **Clean decorator syntax**: Use simple decorators to define your LLM interactions.
- **Dynamic parameter injection**: Easily inject parameters into your prompts using string formatting.
- **Type-safe structured outputs**: Get structured, type-safe responses from your LLM calls.
- **Streaming support**: Efficiently handle streaming responses from LLMs.
- **Faster than raw OpenAI API**: Hurry optimizes the interaction with LLMs, making it faster than using the raw OpenAI Python API.

## Installation

```bash
pip install hurry
```

## Quick Start

Here's a simple example of how to use Hurry:

```python
from hurry import AI, user, system

ai = AI()

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

## Why Hurry is Faster

Hurry is designed to be faster and more efficient than using the raw OpenAI Python API. Here's a comparison:

### OpenAI Python API:

```python
client.chat.completions.create(
    model=model,
    messages=messages,
    stream=stream,
    **llm_params
)
```

### Hurry:

```python
@ai.text()
def my_function():
    """Your prompt here"""
    return "User message"
```

Hurry optimizes the API call process, reduces boilerplate code, and handles parameter management more efficiently, resulting in faster execution and a cleaner codebase.

## Type-Safe Structured Outputs

Hurry supports type-safe structured outputs, making it easier to work with complex response formats:

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

Hurry provides easy-to-use streaming support:

```python
@ai.text(stream=True)
def streaming_function():
    """Your prompt here"""
    return "Generate a long response"

for chunk in streaming_function():
    print(chunk, end="", flush=True)
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

Hurry is released under the [MIT License](LICENSE).
```

This documentation covers the main features of Hurry, including its philosophy of one LLM call per function, the clean decorator syntax, dynamic parameter injection, type-safe structured outputs, and streaming support. It also highlights why Hurry is faster than using the raw OpenAI Python API and provides examples of how to use these features.

You may want to expand on certain sections, add more detailed examples, or include information about advanced features as needed. Remember to create separate files for the Contributing Guide and License if they don't already exist.

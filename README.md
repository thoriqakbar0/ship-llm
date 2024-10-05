from hurry_ai import AI, user, system

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

## Why Hurry allows you to ship fast

Hurry provides a minimal API interface compared to the raw OpenAI Python API, allowing for faster iteration and cleaner code:

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

Hurry optimizes the API call process, reduces boilerplate code, and handles parameter management more efficiently, resulting in faster development and a cleaner codebase.

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

result = conversation()
print(result)
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

## Supported Providers

Currently, Hurry AI has been tested with OpenAI and Azure OpenAI. We plan to expand support to other providers through integration with LiteLLM in the future.

## Contributing

Just open an issue, and let's discuss it.

## License

Hurry is released under the [MIT License](LICENSE).

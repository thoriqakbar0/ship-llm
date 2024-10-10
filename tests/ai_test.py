import os
from openai import OpenAI, AzureOpenAI # type: ignore
from pydantic import BaseModel, Field
from typing import List, Union, Literal, Optional, Generator
from ship_llm.ai import AI, StreamReturn
from dotenv import load_dotenv
import pytest
from openai.types import ChatModel
import requests
from io import BytesIO

load_dotenv()

# Setup
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

ai = AI(client=client, model="gpt-4o-mini")

IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
    "https://upload.wikimedia.org/wikipedia/en/thumb/5/56/Real_Madrid_CF.svg/1280px-Real_Madrid_CF.svg.png"
]

class SubjectClassifier(BaseModel):
    subject: Literal["Biology", "Chemistry", "Physics", "Earth Science", "Geography", "Science", "Botany"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

class StoryPart(BaseModel):
    part_number: int
    content: str

def test_image_bytes_analysis():
    # Download an image and get its bytes
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    response = requests.get(image_url)
    image_bytes = BytesIO(response.content).getvalue()

    @ai.text()
    def image_bytes_analysis():
        """You are an image analysis AI. Describe the image in detail."""
        return ai.user("Describe this image:", image_bytes)

    result = image_bytes_analysis()
    assert isinstance(result, str)
    assert len(result) > 0
    assert "wooden" in result.lower() or "boardwalk" in result.lower() or "nature" in result.lower()
    print(f"Image analysis result: {result}")


def test_simplified_docstring_mismatch():
    ai = AI()

    @ai.text()
    def example_function(param1: str, param2: int):
        """
        Example function.

        Parameters:
        param1: {param1}
        param2: {param2}
        param3: {param3}
        """
        return "test"

    try:
        example_function("test", 123)
    except ValueError as e:
        assert "param3" in str(e)

def test_subject_classifier():

    @ai.structured(SubjectClassifier)
    def try_classifier(query: str, subject: Optional[str] = None, chapter_id: Optional[int] = None):
        """
        Route the query to the correct database. While supplied with subject, it still doesn't guarantee the correct database.

        subject sent from the chatbot: {subject}
        """
        return f"user query this while chatting with assistant: {query}"

    # Test case 1: Without subject
    result1 = try_classifier("What is photosynthesis?")
    assert isinstance(result1, SubjectClassifier)
    assert result1.subject in ["Biology", "Science", "Botany"]
    assert 0.0 <= result1.confidence <= 1.0
    assert len(result1.reasoning) > 0

    # Test case 2: With subject
    result2 = try_classifier("What is the capital of France?", subject="Geography")
    assert isinstance(result2, SubjectClassifier)
    assert result2.subject == "Geography"
    assert 0.0 <= result2.confidence <= 1.0
    assert len(result2.reasoning) > 0

    # Test case 3: Ambiguous query
    result3 = try_classifier("What is the structure of an atom?", subject="Chemistry")
    assert isinstance(result3, SubjectClassifier)
    assert result3.subject in ["Chemistry", "Physics"]
    assert 0.0 <= result3.confidence <= 1.0
    assert len(result3.reasoning) > 0

    # Test case 4: With chapter_id
    result4 = try_classifier("Explain the water cycle.", subject="Earth Science", chapter_id=5)
    assert isinstance(result4, SubjectClassifier)
    assert result4.subject == "Earth Science" or result4.subject == "Science"
    assert 0.0 <= result4.confidence <= 1.0
    assert len(result4.reasoning) > 0

def test_mixed_message_types():
    @ai.text()
    def mixed_messages():
        return [
            ai.system("You're a helpful travel assistant with knowledge about Paris."),
            ai.user("Hi, I'm planning a trip to Paris."),
            {"role": "user", "content": "What are the top 3 must-visit attractions?"},
            ai.assistant("Certainly! The top 3 must-visit attractions in Paris are:\n1. The Eiffel Tower\n2. The Louvre Museum\n3. Notre-Dame Cathedral"),
            ai.user("Tell me more about the Louvre.")
        ]

    result = mixed_messages()
    assert isinstance(result, str)
    assert "louvre" in result.lower()
    assert "art" in result.lower() or "museum" in result.lower()

def test_basic_completion():
    @ai.text()
    def basic_completion():
        """You are a helpful assistant."""
        return "Hello, how are you?"

    result = basic_completion()
    assert isinstance(result, str)
    assert len(result) > 0

def test_streaming_completion():
    @ai.text(stream=True)
    def streaming_completion():
        """You are a helpful assistant."""
        return "Tell me a short story."

    result = streaming_completion()
    assert isinstance(result, StreamReturn)
    assert hasattr(result, 'generator')
    assert hasattr(result, 'cancel_event')
    content = "".join(result.generator)
    assert len(content) > 0

def test_structured_output():
    class Person(BaseModel):
        name: str
        age: int
        occupation: str

    @ai.structured(Person)
    def structured_output():
        """You are a helpful assistant that provides information about people."""
        return "Give me information about a fictional person named John."

    result = structured_output()
    assert isinstance(result, Person)
    assert result.name == "John"
    assert isinstance(result.age, int)
    assert isinstance(result.occupation, str)

def test_image_analysis():
    @ai.text()
    def image_analysis():
        """You are an image analysis AI. Describe the image in detail."""
        return ai.user("Describe this image:", IMAGE_URLS[0])

    result = image_analysis()
    assert isinstance(result, str)
    assert "wooden" in result.lower() or "boardwalk" in result.lower()

def test_conversation():
    @ai.text()
    def conversation():
        """You are a helpful assistant."""
        return [
            ai.user("Hi, I'm planning a trip to Paris."),
            ai.assistant("That's exciting! Paris is a beautiful city. What would you like to know about planning your trip?"),
            ai.user("What are the top 3 must-visit attractions?")
        ]

    result = conversation()
    assert isinstance(result, str)
    assert "eiffel tower" in result.lower() or "louvre" in result.lower() or "notre dame" in result.lower()

def test_system_prompt():
    @ai.text()
    def system_prompt():
        """You are a poetry expert. Analyze the given poem and explain its meaning."""
        return ai.user("Analyze this poem: 'Two roads diverged in a wood, and I— / I took the one less traveled by, / And that has made all the difference.'")

    result = system_prompt()
    assert isinstance(result, str)
    assert "robert frost" in result.lower() or "choice" in result.lower() or "path" in result.lower()

def test_multiple_images():
    @ai.text()
    def multiple_images():
        """You are an image comparison AI. Compare and contrast the given images."""
        return ai.user("Compare these images:", IMAGE_URLS[0], IMAGE_URLS[1])

    result = multiple_images()
    print(result)
    assert isinstance(result, str)
    assert "nature" or "natural" in result.lower() and "logo" in result.lower()

def test_structured_streaming():
    @ai.structured(StoryPart, stream=True, stream_mode="partial")
    def partial_streaming():
        """You are a storyteller AI. Tell a story in parts."""
        return "Tell me a short story about a magical forest, in 3 parts."

    @ai.structured(StoryPart, stream=True, stream_mode="iterable")
    def iterable_streaming():
        """You are a storyteller AI. Tell a story in parts."""
        return "Tell me a short story about a magical forest, in 3 parts."

    # Test partial streaming
    partial_result = partial_streaming()
    assert isinstance(partial_result, Generator)

    final_part = None
    for part in partial_result:
        final_part = part

    assert isinstance(final_part, StoryPart)
    assert final_part.part_number is not None
    assert final_part.content is not None

    # Test iterable streaming
    iterable_result = iterable_streaming()
    assert isinstance(iterable_result, Generator)

    parts = list(iterable_result)

    assert len(parts) >= 3

    for part in parts[:3]:  # Check only the first 3 parts
        assert isinstance(part, StoryPart)
        assert 1 <= part.part_number <= 3
        assert len(part.content) > 0

def test_error():
    @ai.text()
    def error():
        """You are a helpful assistant."""
        return 12345  # This should cause an error as it's not a valid message format

    try:
        error()
        raise AssertionError("Expected an error but none was raised")
    except Exception as e:
        assert isinstance(e, (ValueError, TypeError)), f"Unexpected error type: {type(e)}"
        assert "12345" in str(e) or "int" in str(e) or "Invalid type" in str(e), f"Unexpected error message: {str(e)}"

def test_cancel_stream():
    @ai.text(stream=True)
    def cancel_stream():
        """You are a helpful assistant."""
        return "Tell me a very long story about the history of the universe."

    result = cancel_stream()
    assert isinstance(result, StreamReturn)
    assert hasattr(result, 'generator')
    assert hasattr(result, 'cancel_event')

    chunk_count = 0
    for chunk in result.generator:
        chunk_count += 1
        if chunk_count == 10:
            ai.stop_stream()
            break

    assert chunk_count == 10

def test_structured_non_streaming():
    @ai.structured(SubjectClassifier)
    def classify_subject(query: str):
        """
        Classify the subject of the query.

        query: {query}
        """
        return f"Classify this query: {query}"

    result = classify_subject("What is photosynthesis?")
    assert isinstance(result, SubjectClassifier)
    assert result.subject in ["Biology", "Science", "Botany"]
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.reasoning) > 0

def test_structured_streaming_partial():
    @ai.structured(StoryPart, stream=True, stream_mode="partial")
    def tell_story_partial():
        """Tell a story in parts."""
        return "Tell me a story in parts."

    result = tell_story_partial()
    assert isinstance(result, Generator)

    final_part = None
    for part in result:
        final_part = part

    assert isinstance(final_part, StoryPart)
    assert final_part.part_number is not None
    assert final_part.content is not None

def test_structured_streaming_iterable():
    @ai.structured(StoryPart, stream=True, stream_mode="iterable")
    def tell_story_iterable():
        """Tell a story in parts."""
        return "Tell me a story in parts."

    result = tell_story_iterable()
    assert isinstance(result, Generator)

    parts = list(result)

    assert len(parts) > 0

    for part in parts:
        assert isinstance(part, StoryPart)
        assert part.part_number is not None
        assert part.content is not None

if __name__ == "__main__":
    pytest.main()

import os
from openai import OpenAI, AzureOpenAI, AsyncOpenAI # type: ignore
from pydantic import BaseModel, Field
from typing import List, Union, Literal, Optional, Generator
from ship_llm import AI, StreamReturn, user, assistant, system
from dotenv import load_dotenv
import pytest
import asyncio
from openai.types import ChatModel
import requests
from io import BytesIO
import time
from concurrent.futures import ThreadPoolExecutor

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

def get_image_bytes(url):
    response = requests.get(url)
    if response.status_code == 200:
        print(f"Successfully downloaded {len(response.content)} bytes from {url}")
        return response.content
    else:
        raise ValueError(f"Failed to fetch image from {url}")


class SubjectClassifier(BaseModel):
    subject: Literal["Biology", "Chemistry", "Physics", "Earth Science", "Geography", "Science", "Botany"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

class StoryPart(BaseModel):
    part_number: int
    content: str

class ImageAnalysis(BaseModel):
    description: str
    main_colors: List[str]
    objects_detected: List[str]

@pytest.fixture(autouse=True)
def timer():
    start_time = time.time()
    yield
    duration = time.time() - start_time
    print(f"\nTest duration: {duration:.2f} seconds")

# test 1
def test_image_file_analysis():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_to_image = os.path.join(base_dir, 'tests', 'out-0-3.png')

    # Check if the file exists
    if not os.path.exists(path_to_image):
        pytest.skip(f"Test image not found at {path_to_image}")

    @ai.text()
    def image_file_analysis(image_path):
        """You are an image analysis AI. Describe the image in detail."""
        return user("Describe this image:", image_path)

    result = image_file_analysis(path_to_image)
    assert isinstance(result, str)
    assert len(result) > 0
    # Add more specific assertions based on the content of your test image
    print(f"Image analysis result: {result}")

# def test_structured_image_analysis():
#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     path_to_image = os.path.join(base_dir, 'tests', 'out-0-3.png')

#     # Check if the file exists
#     if not os.path.exists(path_to_image):
#         pytest.skip(f"Test image not found at {path_to_image}")

#     # Function to read image file and convert to bytes
#     def get_image_bytes(file_path):
#         with open(file_path, 'rb') as image_file:
#             return image_file.read()

#     # Get image bytes
#     image_bytes = get_image_bytes(path_to_image)

#     @ai.structured(ImageAnalysis)
#     def analyze_image(image: bytes):
#         """
#         You are an advanced image analysis AI. Analyze the given image and provide a detailed description,
#         list of main colors, and objects detected.

#         use natural language for color
#         """
#         return user("Analyze this image in detail:", image)

#     result = analyze_image(image_bytes)

#     assert isinstance(result, ImageAnalysis)
#     assert len(result.description) > 0
#     assert len(result.main_colors) > 0
#     assert len(result.objects_detected) > 0

#     print("Image Analysis Result:")
#     print(f"Description: {result.description}")
#     print(f"Main Colors: {', '.join(result.main_colors)}")
#     print(f"Objects Detected: {', '.join(result.objects_detected)}")

#     # Additional assertions based on the known content of the image
#     # Note: You might need to adjust these assertions based on the actual content of your test image
#     assert any(keyword in result.description.lower() for keyword in ["kitchen", "cooking", "chef", "food"])
#     assert any(color in ["yellow", "orange", "blue", "green"] for color in result.main_colors)
#     assert any(obj in ["cartoon character", "pan", "pot", "plant", "framed picture"] for obj in result.objects_detected)

# def test_structured_image_analysis_with_conversation():
#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     path_to_image = os.path.join(base_dir, 'tests', 'out-0-3.png')

#     # Check if the file exists
#     if not os.path.exists(path_to_image):
#         pytest.skip(f"Test image not found at {path_to_image}")

#     # Function to read image file and convert to bytes
#     def get_image_bytes(file_path):
#         with open(file_path, 'rb') as image_file:
#             return image_file.read()

#     # Get image bytes
#     image_bytes = get_image_bytes(path_to_image)

#     @ai.structured(ImageAnalysis)
#     def analyze_images_in_conversation(image1: bytes):
#         """
#         You are an advanced image analysis AI. Analyze the given images and provide a detailed description,
#         list of main colors, and objects detected for each image. Compare and contrast the two images.
#         """
#         return [
#             system("You are an expert in image analysis. Provide detailed information about the images and compare them."),
#             user("Analyze this first image:", image1),
#         ]

#     result = analyze_images_in_conversation(image_bytes)

#     assert isinstance(result, ImageAnalysis)
#     assert len(result.description) > 0
#     assert len(result.main_colors) > 0
#     assert len(result.objects_detected) > 0

#     print("Image Analysis Result")
#     print(f"Description: {result.description}")
#     print(f"Main Colors: {', '.join(result.main_colors)}")
#     print(f"Objects Detected: {', '.join(result.objects_detected)}")

#     # Assertions for the image
#     # Note: You might need to adjust these assertions based on the actual content of your test image
#     assert any(keyword in result.description.lower() for keyword in ["kitchen", "cooking", "chef", "food"])
#     assert any(color in ["yellow", "orange", "blue", "green"] for color in result.main_colors)
#     # assert any(obj in ["cartoon character", "pan", "pot", "plant", "framed picture"] for obj in result.objects_detected)

@pytest.mark.asyncio
async def test_async_completion():
    @ai.text()
    async def simple_completion(prompt: str):
        """You are a helpful assistant."""
        return {"role": "user", "content": [{"type": "text", "text": prompt}]}

    result = await simple_completion("What is 2+2?")
    assert isinstance(result, str)
    assert "4" in result.lower()

@pytest.mark.asyncio
async def test_async_streaming():
    @ai.text(stream=True)
    async def stream_completion(prompt: str):
        """You are a helpful assistant."""
        return user(prompt)

    chunks = []
    async for chunk in await stream_completion("Count from 1 to 5"):
        chunks.append(chunk)
        assert isinstance(chunk, str)

    full_response = "".join(chunks)
    assert any(str(i) in full_response for i in range(1, 6))

@pytest.mark.asyncio
async def test_error_handling_and_retries():
    class CustomError(Exception):
        pass

    @ai.text()
    async def failing_completion(prompt: str):
        """You are a helpful assistant."""
        if prompt == "fail":
            raise CustomError("Test error")
        return user(prompt)

    with pytest.raises(CustomError):
        await failing_completion("fail")

    # Should succeed after retries
    result = await failing_completion("succeed")
    assert isinstance(result, str)
    assert len(result) > 0

def test_parallel_processing():
    @ai.text()
    def parallel_completion(prompt: str):
        """You are a helpful assistant."""
        return user(prompt)

    prompts = ["Task 1", "Task 2", "Task 3", "Task 4"]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(parallel_completion, prompt) for prompt in prompts]
        results = [f.result() for f in futures]

    assert len(results) == len(prompts)
    assert all(isinstance(r, str) for r in results)

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
            system("You're a helpful travel assistant with knowledge about Paris."),
            user("Hi, I'm planning a trip to Paris."),
            {"role": "user", "content": "What are the top 3 must-visit attractions?"},
            assistant("Certainly! The top 3 must-visit attractions in Paris are:\n1. The Eiffel Tower\n2. The Louvre Museum\n3. Notre-Dame Cathedral"),
            user("Tell me more about the Louvre.")
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
        return user("Describe this image:", IMAGE_URLS[0])

    result = image_analysis()
    assert isinstance(result, str)
    assert "wooden" in result.lower() or "boardwalk" in result.lower()

def test_conversation():
    @ai.text()
    def conversation():
        """You are a helpful assistant."""
        return [
            user("Hi, I'm planning a trip to Paris."),
            assistant("That's exciting! Paris is a beautiful city. What would you like to know about planning your trip?"),
            user("What are the top 3 must-visit attractions?")
        ]

    result = conversation()
    assert isinstance(result, str)
    assert "eiffel tower" in result.lower() or "louvre" in result.lower() or "notre dame" in result.lower()

def test_system_prompt():
    @ai.text()
    def system_prompt():
        """You are a poetry expert. Analyze the given poem and explain its meaning."""
        return user("Analyze this poem: 'Two roads diverged in a wood, and I— / I took the one less traveled by, / And that has made all the difference.'")

    result = system_prompt()
    assert isinstance(result, str)
    assert "robert frost" in result.lower() or "choice" in result.lower() or "path" in result.lower()

def test_multiple_images():
    @ai.text()
    def multiple_images():
        """You are an image comparison AI. Compare and contrast the given images."""
        return user("Compare these images:", IMAGE_URLS[0], IMAGE_URLS[1])

    result = multiple_images()
    print(result)
    assert isinstance(result, str)
    assert "nature" or "natural" in result.lower() and "logo" in result.lower()

def test_single_images():
    @ai.text()
    def single_images():
        """You are an image comparison AI. Compare and contrast the given images."""
        return user("Compare this image:", IMAGE_URLS[0])

    result = single_images()
    print(result)
    assert isinstance(result, str)
    assert "nature" or "natural" in result.lower()

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
    @ai.structured(response_format=SubjectClassifier)
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
    @ai.structured(response_format=StoryPart, stream=True, stream_mode="partial")
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
    @ai.structured(response_format=StoryPart, stream=True, stream_mode="iterable")
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

def test_chat():
    @ai.text()
    def chat():
        """You are a helpful assistant."""
        return user("Hi, I'm planning a trip to Paris.")

    result = chat()
    assert isinstance(result, str)
    assert "paris" in result.lower()

def test_structured_with_temperature():
    @ai.structured(response_format=StoryPart, temperature=0.8)
    def generate_creative_story() -> StoryPart:
        """Generate a creative short story part with high temperature for more randomness"""
        return "Write a creative short story part"

    result = generate_creative_story()
    assert isinstance(result, StoryPart)
    assert isinstance(result.part_number, int)
    assert isinstance(result.content, str)
    assert len(result.content) > 0

def test_structured_with_top_p():
    @ai.structured(response_format=SubjectClassifier, top_p=0.3)
    def analyze_subject() -> SubjectClassifier:
        """Analyze the subject of a given text with low top_p for more focused output"""
        return "Classify this text: Photosynthesis is the process by which plants convert sunlight into energy"

    result = analyze_subject()
    assert isinstance(result, SubjectClassifier)
    assert result.subject in ["Biology", "Chemistry", "Physics", "Earth Science", "Geography", "Science", "Botany"]
    assert 0 <= result.confidence <= 1
    assert len(result.reasoning) > 0

def test_text_with_temperature():
    @ai.text(temperature=0.8)
    def generate_creative_response():
        """Generate a creative response about space exploration"""
        return "Tell me something creative about space exploration"

    result = generate_creative_response()
    assert isinstance(result, str)
    assert len(result) > 0
    assert "space" in result.lower() or "exploration" in result.lower()

def test_text_with_top_p():
    @ai.text(top_p=0.3)
    def generate_focused_response():
        """Generate a focused response about the importance of clean energy"""
        return "Explain the importance of clean energy"

    result = generate_focused_response()
    assert isinstance(result, str)
    assert len(result) > 0
    assert "energy" in result.lower() or "clean" in result.lower()

if __name__ == "__main__":
    pytest.main()

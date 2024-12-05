import os
import time
import asyncio
from openai import OpenAI, AsyncOpenAI
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ship_llm.ai import AI, user, system
import pytest
from dotenv import load_dotenv
import statistics
from typing import List, Dict, Any
import json

load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize our AI instances
ai = AI(client=openai_client, model="gpt-4o-mini")
async_ai = AI(client=async_openai_client, model="gpt-4o-mini")

def generate_long_docstring(length: int) -> str:
    """Generate a long docstring of specified length."""
    base = "You are a helpful assistant that specializes in natural language processing. "
    return base * (length // len(base) + 1)

def generate_message_list(count: int) -> List[Dict[str, Any]]:
    """Generate a list of messages of specified count."""
    messages = []
    for i in range(count):
        messages.append({"role": "user", "content": f"Message {i}: This is a test message."})
        messages.append({"role": "assistant", "content": f"Response {i}: This is a test response."})
    return messages

async def measure_ttft_openai(prompt: str) -> float:
    """Measure time to first token using raw OpenAI client."""
    start_time = time.time()
    response = await async_openai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            return time.time() - start_time
    return 0.0

async def measure_ttft_our_impl(prompt: str) -> float:
    """Measure time to first token using our implementation."""
    start_time = time.time()
    
    @async_ai.text(stream=True)
    async def completion(p: str):
        return user(p)
    
    async for chunk in await completion(prompt):
        return time.time() - start_time
    return 0.0

def write_benchmark_results(results_dict: dict):
    """Write benchmark results to JSON file in a readable format."""
    try:
        with open("benchmark_results.json", "r") as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = []
    
    # Add new results with metadata
    result_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_data": results_dict
    }
    
    all_results.append(result_entry)
    
    # Write back with proper formatting
    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2, sort_keys=True)

@pytest.mark.asyncio
async def test_ttft_comparison():
    """Compare time to first token between implementations."""
    prompt = "What is the capital of France?"
    iterations = 5
    
    openai_times = []
    our_times = []
    
    for _ in range(iterations):
        openai_ttft = await measure_ttft_openai(prompt)
        our_ttft = await measure_ttft_our_impl(prompt)
        
        openai_times.append(openai_ttft)
        our_times.append(our_ttft)
        
        # Wait between iterations to avoid rate limits
        await asyncio.sleep(1)
    
    avg_openai = statistics.mean(openai_times)
    avg_ours = statistics.mean(our_times)
    
    print(f"\nTime to First Token Comparison (averaged over {iterations} runs):")
    print(f"OpenAI Implementation: {avg_openai:.3f}s")
    print(f"Our Implementation: {avg_ours:.3f}s")
    print(f"Difference: {abs(avg_openai - avg_ours):.3f}s")
    
    # Store results for analysis
    write_benchmark_results({
        "test_name": "time_to_first_token",
        "metrics": {
            "iterations": iterations,
            "prompt": prompt,
            "openai": {
                "times": openai_times,
                "average": avg_openai
            },
            "our_implementation": {
                "times": our_times,
                "average": avg_ours
            },
            "comparison": {
                "difference": abs(avg_openai - avg_ours),
                "percentage_difference": (abs(avg_openai - avg_ours) / avg_openai) * 100
            }
        }
    })

@pytest.mark.asyncio
async def test_docstring_latency():
    """Test latency with different docstring lengths."""
    docstring_lengths = [100, 1000, 5000, 10000]
    results = {}
    
    for length in docstring_lengths:
        test_docstring = generate_long_docstring(length)
        
        @async_ai.text()
        async def completion_with_long_docstring(docstring: str):
            """
            {docstring}
            """
            return user("What is 2+2?")
        
        start_time = time.time()
        response = await completion_with_long_docstring(test_docstring)
        latency = time.time() - start_time
        
        results[str(length)] = {
            "length": length,
            "latency": latency
        }
        await asyncio.sleep(1)  # Avoid rate limits
    
    print("\nDocstring Length Latency Test:")
    for length, data in results.items():
        print(f"Length {length}: {data['latency']:.3f}s")
    
    # Calculate summary statistics
    latencies = [data["latency"] for data in results.values()]
    
    # Store results
    write_benchmark_results({
        "test_name": "docstring_latency",
        "metrics": {
            "results_by_length": results,
            "summary": {
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "avg_latency": statistics.mean(latencies),
                "total_lengths_tested": len(docstring_lengths)
            }
        }
    })

@pytest.mark.asyncio
async def test_message_list_performance():
    """Test performance with different message list sizes."""
    message_counts = [10, 50, 100, 200]
    results = {}
    
    for count in message_counts:
        messages = generate_message_list(count)
        
        # Test with OpenAI client
        start_time = time.time()
        await async_openai_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages + [{"role": "user", "content": "Final question?"}]
        )
        openai_time = time.time() - start_time
        
        # Test with our implementation
        @async_ai.text()
        async def completion_with_messages(msg_list):
            return msg_list + [user("Final question?")]
        
        start_time = time.time()
        await completion_with_messages(messages)
        our_time = time.time() - start_time
        
        results[str(count)] = {
            "message_count": count,
            "openai_time": openai_time,
            "our_time": our_time,
            "difference": abs(openai_time - our_time),
            "percentage_faster": ((max(openai_time, our_time) - min(openai_time, our_time)) / max(openai_time, our_time)) * 100
        }
        
        await asyncio.sleep(1)  # Avoid rate limits
    
    print("\nMessage List Performance Test:")
    for count, data in results.items():
        print(f"Message Count {count}:")
        print(f"  OpenAI: {data['openai_time']:.3f}s")
        print(f"  Ours: {data['our_time']:.3f}s")
        print(f"  Difference: {data['difference']:.3f}s")
        print(f"  Percentage Faster: {data['percentage_faster']:.1f}%")
    
    # Calculate summary statistics
    openai_times = [data["openai_time"] for data in results.values()]
    our_times = [data["our_time"] for data in results.values()]
    
    # Store results
    write_benchmark_results({
        "test_name": "message_list_performance",
        "metrics": {
            "results_by_count": results,
            "summary": {
                "average_times": {
                    "openai": statistics.mean(openai_times),
                    "our_implementation": statistics.mean(our_times)
                },
                "overall_difference": {
                    "absolute": abs(statistics.mean(openai_times) - statistics.mean(our_times)),
                    "percentage": ((max(statistics.mean(openai_times), statistics.mean(our_times)) - 
                                min(statistics.mean(openai_times), statistics.mean(our_times))) / 
                                max(statistics.mean(openai_times), statistics.mean(our_times))) * 100
                }
            }
        }
    })

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

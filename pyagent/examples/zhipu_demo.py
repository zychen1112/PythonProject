"""
Zhipu GLM Provider Demo.

This example demonstrates how to use the Zhipu GLM provider.
"""

import asyncio
import os

from pyagent.providers.zhipu import ZhipuProvider


async def main():
    # Initialize the Zhipu provider
    # You can set the API key via environment variable: ZHIPUAI_API_KEY
    # or pass it directly to the constructor
    provider = ZhipuProvider(
        api_key=os.environ.get("ZHIPUAI_API_KEY")
    )

    # List available models
    print("Available Zhipu GLM models:")
    for model in provider.get_available_models():
        print(f"  - {model}")
    print()

    # Example messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Can you introduce yourself briefly?"}
    ]

    # Non-streaming completion
    print("=== Non-streaming completion ===")
    response = await provider.complete(
        messages=messages,
        model="glm-4-flash",  # Fast and cost-effective model
        temperature=0.7,
        max_tokens=500
    )

    if response["message"]:
        print(f"Assistant: {response['message'].content}")
    print(f"Usage: {response['usage']}")
    print(f"Finish reason: {response['finish_reason']}")
    print()

    # Streaming completion
    print("=== Streaming completion ===")
    messages.append({"role": "assistant", "content": response["message"].content if response["message"] else ""})
    messages.append({"role": "user", "content": "What are the main features of GLM models?"})

    print("Assistant: ", end="", flush=True)
    async for chunk in provider.stream(
        messages=messages,
        model="glm-4-flash",
        temperature=0.7,
        max_tokens=500
    ):
        if isinstance(chunk, str):
            print(chunk, end="", flush=True)
        elif isinstance(chunk, dict) and "tool_call" in chunk:
            print(f"\n[Tool call: {chunk['tool_call']['name']}]")
    print("\n")

    # Example with tools
    print("=== Completion with tools ===")
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g., Beijing"
                    }
                },
                "required": ["location"]
            }
        }
    ]

    tool_messages = [
        {"role": "user", "content": "What's the weather like in Beijing?"}
    ]

    response = await provider.complete(
        messages=tool_messages,
        tools=tools,
        model="glm-4-flash"
    )

    if response["tool_calls"]:
        print("Tool calls:")
        for tc in response["tool_calls"]:
            print(f"  - {tc['name']}({tc['input']})")
    elif response["message"]:
        print(f"Assistant: {response['message'].content}")


if __name__ == "__main__":
    asyncio.run(main())

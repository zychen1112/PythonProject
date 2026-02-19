"""
Basic agent example demonstrating core functionality.
"""

import asyncio
from pyagent import Agent
from pyagent.core.agent import AgentConfig
from pyagent.core.tools import tool
from pyagent.providers.openai import OpenAIProvider


# Define some tools using the @tool decorator
@tool(description="Get the current weather for a city")
def get_weather(city: str) -> str:
    """Get weather for a city (mock implementation)."""
    # This is a mock implementation
    weather_data = {
        "beijing": "Sunny, 25°C",
        "shanghai": "Cloudy, 22°C",
        "new york": "Rainy, 18°C",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool(description="Calculate the sum of two numbers")
def add_numbers(a: int, b: int) -> str:
    """Add two numbers together."""
    return f"The sum of {a} and {b} is {a + b}"


async def main():
    # Create the LLM provider
    provider = OpenAIProvider(
        api_key="your-api-key-here",  # Replace with your API key
    )

    # Create agent configuration
    config = AgentConfig(
        name="ExampleAgent",
        model="gpt-4",
        system_prompt="You are a helpful assistant with access to weather and calculation tools.",
    )

    # Create the agent with tools
    agent = Agent(
        provider=provider,
        config=config,
        tools=[get_weather, add_numbers],
    )

    # Run the agent
    print("Running agent...")
    response = await agent.run("What's the weather like in Beijing?")
    print(f"Response: {response}")

    # Run another query
    response = await agent.run("What is 42 + 17?")
    print(f"Response: {response}")


if __name__ == "__main__":
    asyncio.run(main())

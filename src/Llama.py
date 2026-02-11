from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import asyncio
import os

# Load environment variables from .env file
load_dotenv()

# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.

api_key = os.environ.get("HF_TOKEN")


model_client = OpenAIChatCompletionClient(
        model="meta-llama/Llama-3.2-1B-Instruct",
        base_url="https://router.huggingface.co/v1",
        api_key=api_key,
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": True,
            "structured_output": True,
            "family": "unknown",
        }
    )


# Define a simple function tool that the agent can use.
# For this example, we use a fake weather tool for demonstration purposes.
async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is 73 degrees and Sunny."


# Define an AssistantAgent with the model, tool, system message, and reflection enabled.
# The system message instructs the agent via natural language.
agent = AssistantAgent(
    name="weather_agent",
    model_client=model_client,
    system_message="You are a helpful assistant.",
    model_client_stream=True,  # Enable streaming tokens from the model client.
)


# Run the agent and stream the messages to the console.
async def main() -> None:
    await Console(agent.run_stream(task="What is the weather in New York?"))
    # Close the connection to the model client.
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())

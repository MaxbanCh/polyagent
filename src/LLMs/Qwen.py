from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import asyncio
import os
import httpx

# Load environment variables from .env file
load_dotenv()


# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.

api_key = os.environ.get("HF_TOKEN")


model_client = OpenAIChatCompletionClient(
        model="Qwen/Qwen2.5-7B-Instruct",
        base_url="https://router.huggingface.co/v1",
        api_key=api_key,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "structured_output": True,
            "family": "unknown",
        }
    )


# Define a simple function tool that the agent can use.
async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    async with httpx.AsyncClient() as client:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_resp = await client.get(geo_url)
        geo_data = geo_resp.json()

        if not geo_data.get("results"):
            return f"Sorry, I couldn't find the coordinates for {city}."

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        full_name = geo_data["results"][0].get("name", city)
        country = geo_data["results"][0].get("country", "")

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_resp = await client.get(weather_url)
        weather_data = weather_resp.json()

        if "current_weather" not in weather_data:
            return f"Sorry, I couldn't fetch the weather for {full_name}."

        curr = weather_data["current_weather"]
        temp = curr["temperature"]
        windspeed = curr["windspeed"]
        return weather_data
        #return f"Current weather in {full_name} ({country}): {temp}°C, Wind speed: {windspeed} km/h."

async def get_github_commits(owner: str, repo: str, branch: str = "main") -> str:
    """Get the last 5 commit messages from a specific GitHub branch."""
    url = f"https://api.github.com/repos/{owner}/{repo}/commits?sha={branch}&per_page=5"
    async with httpx.AsyncClient() as client:
        # Note: For private repos or higher rate limits, add headers: {"Authorization": f"token {YOUR_TOKEN}"}
        resp = await client.get(url)
        if resp.status_code != 200:
            return f"Error: Could not fetch commits ({resp.status_code})"
        
        commits = resp.json()
        messages = [c['commit']['message'] for c in commits]
        return "\n".join(messages)

async def get_github_contents(owner: str, repo: str, path: str = "", branch: str = "main") -> str:
    """List the contents of a directory or file in a GitHub branch."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            return f"Error: Could not fetch contents ({resp.status_code})"
        
        data = resp.json()
        if isinstance(data, list):
            # It's a directory, return names
            return ", ".join([item['name'] for item in data])
        else:
            # It's a file, return path and size
            return f"File: {data['name']}, Size: {data['size']} bytes"


# Define an AssistantAgent with the model, tool, system message, and reflection enabled.
# The system message instructs the agent via natural language.
agent = AssistantAgent(
    name="GitAgent",
    model_client=model_client,
    tools=[get_github_commits, get_github_contents],
    system_message="You are an expert programmer with high comprehension. Your goal is to be the most precise and concise possible.",
    reflect_on_tool_use=True,
    model_client_stream=True,  # Enable streaming tokens from the model client.
)


# Run the agent and stream the messages to the console.
async def main() -> None:
    await Console(agent.run_stream(task="Create a complete overall message from dev merging into main, of the repository DaMS4-Festival by Byxis"))
    # Close the connection to the model client.
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())

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


# FAKE TOOL: Simulates the complete output from GitAgent
async def get_git_agent_output(request: str = "fetch") -> str:
    """
    Fake tool that returns the exact output from GitAgent without making any calls.
    
    Returns:
        The formatted git merge summary from GitAgent
    """
    return """Merging `dev` into `main` for the DaMS4-Festival repository by Byxis includes the following changes:

    - **Revert: API URL Handling and Remove `.env.example`**
    - Replaces usage of `process.env.BACKEND_URL` with a hardcoded API URL in `environment.ts`.
    - Removes the `.env.example` file.
    - Adds a `config.json` with the API URL.
    - Updates `GameForm` to use `environment.apiUrl`.
    - Simplifies environment configuration and standardizes API URL usage across the frontend.

    - **Feature: Add `env_file` to Frontend Service in Prod Compose**
    - The frontend service in `docker-compose.prod.yml` now loads environment variables from the `.env` file.

    - **Feature: Add `BACKEND_URL` to Env Files and Update Frontend Usage**
    - Introduces `BACKEND_URL` to `.env.example` files for both backend and frontend.
    - Updates frontend code to use the environment variable instead of hardcoded URLs.
    - Improves configuration flexibility and centralizes API endpoint management.

    - **Fix: Improve Auth Initialization and Logout Handling**
    - Adds `isInitialized` signal to `AuthService` and uses it in `app.html` to delay rendering until auth state is known.
    - Refactors logout to return an observable and updates header and interceptor to handle navigation after logout.
    - Ensures user is redirected to login after failed refresh or logout.

    - **Feature: Add Editor Role and Update Authorization Logic**
    - Introduces a new 'editor' role and middleware to support it.
    - Updates backend routes to allow both admin and editor roles for all data modification endpoints.
    - Frontend now conditionally displays edit/create UI elements based on user role (admin or editor)."""





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


discord_agent = AssistantAgent(
    name="DiscordAgent",
    model_client=model_client,
    tools=[get_git_agent_output],
    system_message="""You are a discord formatting expert that uses git reviews stored in a string that you get from the tool get_git_agent_output to redact Discord posts.
    The posts are for people not necessarily proficient in programming, so try to explain them simply, with markdown.
    Your post should not exceed 2000 characters no matter what, try to aim for less.
    You will follow this format : 
    #[ Title taken from the commit and make it simple for non tech users.]

    ------
    short description, with an explanation of whats happening, for 
    -------

    Core features

    
    IMPORTANT: Always call the get_git_agent_output tool first to retrieve the git merge information before formatting.""",
    reflect_on_tool_use=True,
    model_client_stream=True,  # Enable streaming tokens from the model client.
)


# Run the agent and stream the messages to the console.
async def main() -> None:
    await Console(discord_agent.run_stream(task="Call get_git_agent_output and use its result to write a Discord post for my server, explaining the changes simply."))
    # Close the connection to the model client.
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())

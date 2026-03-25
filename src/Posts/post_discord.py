from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from src.api.discord_api import send_message
from src.Prompts.Qwen import process_agent
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

async def get_git_agent_output() -> str:
    """Fetch the summary from the Qwen git agent."""
    infos = {
        "owner": "Byxis",
        "repo": "Ruzzle",
        "branch": "dev",
        "html_url": "https://github.com/Byxis/Ruzzle"
    }
    return await process_agent(infos)

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
    git_summary = await get_git_agent_output()
    print("Git summary retrieved:\n", git_summary)
    #format the text
    task = f"Format this git summary into a Discord post:\n\n{git_summary}"
    result = await discord_agent.run(task=task)
    
    
    # Step 3: Send to Discord
    discord_formatted = result.messages[-1].content if result.messages else git_summary
    await send_message(
        message=discord_formatted,
        channel_id=1481232603288698960
    )

    # Close the connection to the model client.
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())

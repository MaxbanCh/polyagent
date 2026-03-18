from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
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

github_token = os.environ.get("GITHUB_TOKEN")


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


async def get_github_commits(owner: str, repo: str, base: str = "main", head: str = "dev") -> str:
    """Get all commit messages on the head branch that are not in the base branch."""
    url = f"https://api.github.com/repos/{owner}/{repo}/compare/{base}...{head}"
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"token {github_token}"}
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            return f"Error: Could not fetch commits ({resp.status_code})"
        
        data = resp.json()
        commits = data.get('commits', [])
        
        messages = []
        for c in commits:
            msg = c['commit']['message'].strip()
            messages.append(f"Commit message:\n{msg}")
            
        return "\n\n---\n".join(messages) if messages else "No new commits."

async def get_github_contents(owner: str, repo: str, path: str = "", branch: str = "main") -> str:
    """List the contents of a directory or file in a GitHub branch."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"token {github_token}"}
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            return f"Error: Could not fetch contents ({resp.status_code})"
        
        data = resp.json()
        if isinstance(data, list):
            # It's a directory, return names
            return ", ".join([item['name'] for item in data])
        else:
            # It's a file, return path and size
            return f"File: {data['name']}, Size: {data['size']} bytes"


git_agent = AssistantAgent(
    name="git_fetcher",
    model_client=model_client,
    tools=[get_github_commits, get_github_contents],
    system_message="You extract technical data from GitHub. You MUST use your tools to fetch commits before any summary can be written. Only provide raw data found.",
    reflect_on_tool_use=True,
    model_client_stream=True,  # Enable streaming tokens from the model client.
)

reviewer_agent = AssistantAgent(
    name="reviewer",
    model_client=model_client,
    system_message="You provide feedback only if major vulnerabilities are explicitly mentioned in the commit messages OR in the code. If not, say 'No vulnerabilities found'.",
    reflect_on_tool_use=True,
    model_client_stream=True,  # Enable streaming tokens from the model client.
)

writer_agent = AssistantAgent(
    name="summary_writer",
    model_client=model_client,
    system_message="You write the final summary. You MUST NOT make up information. DO NOT REPEAT YOURSELF. Use only the data provided by git_fetcher. If no data has been fetched yet, ask git_fetcher to do its job. The reviews or vulnerabilites are made automatically. End with 'TERMINATE'.",
    reflect_on_tool_use=True,
    model_client_stream=True,  # Enable streaming tokens from the model client.
)

termination = TextMentionTermination("TERMINATE")
selector_prompt = """Select the next speaker based on the following workflow:
1. 'git_fetcher' must ALWAYS go first to retrieve the actual commit data using tools.
2. 'reviewer' must go second to analyze the data provided by the fetcher.
3. 'summary_writer' must go last to compile the final message based on the fetcher's data and reviewer's feedback.

Current conversation history:
{history}
"""
team = SelectorGroupChat(
    [git_agent, reviewer_agent, writer_agent], 
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=False
)

# Run the agent and stream the messages to the console.
async def main() -> None:
    task = "Create a complete overall message of feat/Qwen-implementation merging into main, of the repository polyagent by MaxbanCh"
    await Console(team.run_stream(task=task))
    # Close the connection to the model client.
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())

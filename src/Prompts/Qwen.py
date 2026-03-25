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
import random

# Load environment variables from .env file
load_dotenv()


# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.

api_key = os.environ.get("HF_TOKEN")

github_token = os.environ.get("GITHUB_API_TOKEN")

import random
import uuid
TERMINATION_WORD = f"TERMINATE_{uuid.uuid4().hex[:8].upper()}"
random_termination = TextMentionTermination(TERMINATION_WORD)


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


async def get_branch_overview(owner: str, repo: str, base: str = "main", head: str = "dev") -> str:
    """Phase 1 – Get commit messages and a summary of changed files (no patches).
    Returns the list of commits and, for each changed file, its name, status,
    and line-change counts. Call this first, then use get_file_diff for specific files.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/compare/{base}...{head}"
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"token {github_token}"}
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            return f"Error: Could not fetch comparison ({resp.status_code})"

        data = resp.json()
        parts = []

        # --- Commits ---
        commits = data.get('commits', [])
        msgs = [f"- {c['commit']['message'].strip()}" for c in commits]
        parts.append("## Commits\n" + ("\n".join(msgs) or "No commits."))

        # --- Changed files (stats only, no patch) ---
        files = data.get('files', [])
        rows = []
        for f in files:
            rows.append(
                f"- `{f['filename']}` [{f['status']}]  "
                f"+{f.get('additions', 0)} / -{f.get('deletions', 0)} lines"
            )
        parts.append("## Changed files\n" + ("\n".join(rows) or "No files changed."))

        return "\n\n".join(parts)

async def get_file_diff(
    owner: str, repo: str, file_path: str, base: str = "main", head: str = "dev",
    max_lines: int = 150
) -> str:
    """Phase 2 – Get the full diff patch for a single specific file.
    Call get_branch_overview first to discover which files changed,
    then call this tool for each file you want to inspect.
    `file_path` must match the filename returned by get_branch_overview exactly.
    `max_lines` caps the patch size to protect the context window (default 150).
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/compare/{base}...{head}"
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"token {github_token}"}
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            return f"Error: Could not fetch diff ({resp.status_code})"

        data = resp.json()
        for f in data.get('files', []):
            if f['filename'] == file_path:
                patch = f.get('patch', '[binary or no patch available]')
                lines = patch.splitlines()
                truncation_note = ""
                if len(lines) > max_lines:
                    truncation_note = f"\n\n> ⚠️ Patch truncated: showing {max_lines}/{len(lines)} lines. Call again with a higher max_lines if needed."
                    patch = "\n".join(lines[:max_lines])
                return f"### {f['filename']} ({f['status']})\n```diff\n{patch}\n```{truncation_note}"

        return f"File `{file_path}` not found in the diff between `{base}` and `{head}`."

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
    tools=[get_branch_overview, get_github_contents],
    system_message=(
        "You extract technical data from GitHub. You MUST use your tools to fetch commits before any summary can be written. Only provide raw data found."
        f"NEVER MENTION '{TERMINATION_WORD}' IN YOUR RESPONSE."
    ),
    reflect_on_tool_use=True,
    model_client_stream=True,
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
    system_message=f"You write the final summary. You MUST NOT make up information. DO NOT REPEAT YOURSELF. Use only the data provided by git_fetcher. If no data has been fetched yet, ask git_fetcher to do its job. The reviews or vulnerabilites are made automatically. End with '{TERMINATION_WORD}'.",
    reflect_on_tool_use=True,
    model_client_stream=True,  # Enable streaming tokens from the model client.
)

selector_prompt = f"""You must select the next speaker following this strict workflow:

1. Select 'git_fetcher' if:
   - No tool has been called yet (it always goes first)
   - git_fetcher called get_branch_overview.
2. Select 'reviewer' if:
   - git_fetcher has completed both phases (overview + file diffs if provided) AND reviewer has not yet spoken.

3. Select 'summary_writer' if:
   - reviewer has already provided its feedback.
   summary_writer ends the conversation with '{TERMINATION_WORD}'.

Never skip a step. Never select summary_writer before reviewer has spoken.

Current conversation history:
{{history}}
"""
team = SelectorGroupChat(
    [git_agent, reviewer_agent, writer_agent],
    model_client=model_client,
    termination_condition=random_termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=True,
)


# Run the agent and stream the messages to the console.
async def process_agent(infos: dict) -> str:
    owner = infos["owner"]
    repo = infos["repo"]
    branch = infos["branch"]
    url = infos["html_url"]

    task = f"""Create a complete overall message of {branch} merging on the last commit, of the repository {repo} by {owner}"""
    result = await team.run(task=task)

    last_message = result.messages[-1].content if result.messages else ""
    # Close the connection to the model client.
    await model_client.close()
    return last_message


if __name__ == "__main__":
    infos = {
        "owner": "Byxis",
        "repo": "Ruzzle",
        "branch": "dev",
        "html_url": "https://github.com/Byxis/Ruzzle"
    }
    summary = asyncio.run(process_agent(infos))

    print(f"Summary from writer_agent:\n{summary}")
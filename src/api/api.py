from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional
import hmac
import hashlib
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://polyagent.axithem.fr", "http://127.0.0.1:3010"],  # Allow all origins; change for production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")

class PushEvent(BaseModel):
    ref: str  # e.g. "refs/heads/main"
    repository: dict
    pusher: dict
    commits: Optional[list] = []

def verify_github_signature(payload: bytes, signature: str) -> bool:
    """Verify that the request comes from GitHub."""
    expected = "sha256=" + hmac.new(
        GITHUB_WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)

@app.get("/hello")
async def hello():
    return {"message": "Hello, World!"}

@app.post("/github/webhook")
async def github_webhook(
    request: Request,
    x_github_event: str = Header(None),
    x_hub_signature_256: str = Header(None),
):
    payload = await request.body()

    # Verify the signature (optional but recommended)
    if x_hub_signature_256:
        if not verify_github_signature(payload, x_hub_signature_256):
            raise HTTPException(status_code=401, detail="Invalid signature")

    # # Only handle push events
    # if x_github_event != "push":
    #     return {"message": f"Event '{x_github_event}' ignored"}

    data = await request.json()

    owner = data["repository"]["owner"]["name"]
    user = data["pusher"]["name"]
    repo = data["repository"]["name"]
    branch = data["ref"].replace("refs/heads/", "")
    commits = [c["id"] for c in data.get("commits", [])]

    print(f"Push by {user} on {repo}/{branch} — commits: {commits}")

    return {
        "owner": owner,
        "user": user,
        "repo": repo,
        "branch": branch,
        "commits": commits,
    }

def main():
    uvicorn.run(app, host="0.0.0.0", port=3010)
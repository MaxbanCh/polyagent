from api import api
import asyncio
import LLMs.Qwen as Qwen
import LLMs.Llama as Llama
import api.discord_api as disco
import Posts.post_discord as post_disco

if __name__ == "__main__":
    # api.main()
    asyncio.run(post_disco)
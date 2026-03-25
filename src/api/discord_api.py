import discord
import dotenv
import os

dotenv.load_dotenv("../../.env")

class DiscordClient(discord.Client):
    def __init__(self, *args, message: str, channel_id: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._message = message
        self._channel_id = channel_id

    async def on_ready(self):
        print(f'Logged on as {self.user}!')
        await self.send_message(self._channel_id, self._message)
        await self.close()

    async def send_message(self, channel_id: int, message: str):
        channel = self.get_channel(channel_id)
        if channel:
            await channel.send(message)
        else:
            print(f"Channel with ID {channel_id} not found.")


async def send_message(message: str = "## I think it's \n # Crystal Clear now ;) ", channel_id: int = 1481232603288698960):
    intents = discord.Intents.default()
    intents.message_content = True

    client = DiscordClient(intents=intents, message=message, channel_id=channel_id)

    async with client:
        token = os.getenv('DISCORD_TOKEN')
        if not token:
            raise ValueError("DISCORD_TOKEN not found in .env file")
        await client.start(token)

    return

if __name__ == "__main__":
    import asyncio
    asyncio.run(send_message())
# GrokBot - Discord AI Assistant

A friendly, feature-rich Discord bot powered by **xAI's Grok** models.

It supports natural chat, live web search for current information, image generation, reminders, and basic server moderation tools.

## Features

- **Conversational AI** — Ask anything; the bot responds with Grok's helpful and witty personality.
- **Live Web Search** — Automatically searches the internet for sports scores/schedules, news, movies, weather, prices, and other time-sensitive topics.
- **Image Generation** — Request images with natural language (e.g., "generate a cyberpunk city at night").
- **Reminders** — Set reminders with natural phrases like "remind me in 2 hours to take out the trash" or "remind me at 3pm to call mom".
- **Server Moderation** — Supports commands for rename, timeout, kick, ban, unban, purge messages, etc. (via JSON actions returned by the model).
- **Slash Command** — `/grok <your question>`
- **Mention Trigger** — The bot responds when mentioned or when you use `!generate_image`.

## Requirements

- Python 3.10+
- A Discord bot token (create one at the [Discord Developer Portal](https://discord.com/developers/applications))
- An xAI Grok API key (get one at [console.x.ai](https://console.x.ai))

### Python Dependencies

```bash
pip install discord.py aiohttp python-dotenv thefuzz python-dateutil pytz
```

# Setup

Clone or download the repository.
Create a .env file in the project root (same folder as GrokBot.py). Do NOT commit this file to Git — add .env to your .gitignore.
Add your tokens to the .env file:envDISCORD_TOKEN=your_discord_bot_token_here
GROK_API_KEY=your_xai_grok_api_key_here

# How to Get the Tokens

Discord Token: Go to the Discord Developer Portal → Your Application → Bot → Copy Token (enable Message Content Intent and Server Members Intent under Privileged Gateway Intents).
Grok API Key: Log in at https://console.x.ai, go to API Keys, and create a new key.

# Setting Environment Variables (Alternative to .env)

If you prefer not to use a .env file, you can set the variables directly in your system. The bot will read them via os.getenv().

*On Windows (Command Prompt / PowerShell)*

Temporarily (current session only):
cmdset DISCORD_TOKEN=your_discord_bot_token_here
set GROK_API_KEY=your_xai_grok_api_key_here

Permanently:
Search for "Environment Variables" in Windows Search.
Click Edit the system environment variables.
Under System variables, click New....
Add each variable (DISCORD_TOKEN and GROK_API_KEY) with their values.
Restart your terminal or IDE.

*On Linux / macOS*

Temporarily (current terminal session):
```Bash
export DISCORD_TOKEN="your_discord_bot_token_here"
export GROK_API_KEY="your_xai_grok_api_key_here"
```

Permanently (add to shell config):
```Bash
echo 'export DISCORD_TOKEN="your_discord_bot_token_here"' >> ~/.bashrc
echo 'export GROK_API_KEY="your_xai_grok_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```
(Use ~/.zshrc if you use zsh.)

*Running the Bot*

```bash
python GrokBot.py
```
The bot will log in and show which servers it joined. Invite it to your server using the OAuth2 URL from the Discord Developer Portal (enable bot and the required scopes/intents).

# Customization

The bot's personality and behavior are controlled in the SYSTEM_ROLE string at the top of GrokBot.py.

You can easily edit:
- Tone and helpfulness guidelines
- Search policy
- Image generation triggers
- Reminder parsing
- Supported admin actions

The model registry automatically fetches the latest Grok models and refreshes weekly.

# Security Notes

Never share your API keys or Discord token.
Keep .env out of version control.
The bot requires Message Content and Server Members intents at a minimum. The more permissions you grant the bot, the more it can do for you.

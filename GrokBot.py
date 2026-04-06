import discord
from discord import app_commands
import aiohttp
import os
import random
import json
import logging
import sys
import re
import time
import asyncio
from dotenv import load_dotenv
from datetime import datetime, timedelta
from thefuzz import process, fuzz
import pytz
from dateutil.parser import parse as parse_datetime

# Load environment variables
load_dotenv()

# Configuration
RECENT_REQUEST_WINDOW_SECONDS = 60

# Retrieve tokens
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GROK_API_KEY = os.getenv('GROK_API_KEY')

# Validate tokens
if not DISCORD_TOKEN:
    print("Error: DISCORD_TOKEN not set in .env file.")
    sys.exit(1)
if not GROK_API_KEY:
    print("Error: GROK_API_KEY not set in .env file. Get one at https://x.ai/api")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Global data
recent_requests = {}
reminders = []

# ── Model Registry ────────────────────────────────────────────────────────────
MODEL_CACHE_FILE = 'xai_model_cache.json'
MODEL_CACHE_MAX_AGE_DAYS = 7

class ModelRegistry:
    """
    Fetches available xAI models from the API and classifies them into:
      - chat_models:   language models for Chat Completions
      - search_models: language models with web search support (Responses API)
      - image_models:  image generation models
    Results are cached to disk and refreshed automatically.
    """

    def __init__(self):
        self.chat_models: list[str] = []
        self.search_models: list[str] = []
        self.image_models: list[str] = []
        self._lock = asyncio.Lock()
        self._last_fetched: datetime | None = None

    # Fallbacks if API is unreachable
    FALLBACK_CHAT = [
        'grok-4.20-reasoning',
        'grok-4.20-multi-agent',
        'grok-4-1-fast-reasoning',
        'grok-4.20-non-reasoning',
        'grok-4-1-fast-non-reasoning',
    ]
    FALLBACK_SEARCH = [
        'grok-4.20-reasoning',
        'grok-4-1-fast-reasoning',
        'grok-4.20-non-reasoning',
        'grok-4-1-fast-non-reasoning',
    ]
    FALLBACK_IMAGE = ['grok-imagine-image-pro', 'grok-imagine-image']

    def _apply_fallbacks(self):
        if not self.chat_models:
            self.chat_models = list(self.FALLBACK_CHAT)
        if not self.search_models:
            self.search_models = list(self.FALLBACK_SEARCH)
        if not self.image_models:
            self.image_models = list(self.FALLBACK_IMAGE)

    def _load_cache(self) -> bool:
        """Load from disk cache. Returns True if cache is fresh enough."""
        try:
            if not os.path.exists(MODEL_CACHE_FILE):
                return False
            with open(MODEL_CACHE_FILE, 'r') as f:
                data = json.load(f)
            fetched_at = datetime.fromisoformat(data['fetched_at'])
            age = datetime.utcnow() - fetched_at
            if age.days >= MODEL_CACHE_MAX_AGE_DAYS:
                logger.info(f"Model cache is {age.days} days old — will refresh.")
                return False
            self.chat_models = data['chat_models']
            self.search_models = data['search_models']
            self.image_models = data['image_models']
            self._last_fetched = fetched_at
            logger.info(f"Loaded model cache from disk (age: {age.days}d).")
            return True
        except Exception as e:
            logger.warning(f"Could not load model cache: {e}")
            return False

    def _save_cache(self):
        try:
            with open(MODEL_CACHE_FILE, 'w') as f:
                json.dump({
                    'fetched_at': datetime.utcnow().isoformat(),
                    'chat_models': self.chat_models,
                    'search_models': self.search_models,
                    'image_models': self.image_models,
                }, f, indent=2)
            logger.info("Model cache saved to disk.")
        except Exception as e:
            logger.warning(f"Could not save model cache: {e}")

    async def fetch(self):
        """Fetch models from the xAI API, classify, and cache them."""
        async with self._lock:
            headers = {
                'Authorization': f'Bearer {GROK_API_KEY}',
                'Content-Type': 'application/json',
            }
            try:
                async with aiohttp.ClientSession() as session:
                    # Language models
                    async with session.get('https://api.x.ai/v1/language-models', headers=headers, timeout=30) as resp:
                        resp.raise_for_status()
                        lang_data = await resp.json()

                    # Image generation models
                    async with session.get('https://api.x.ai/v1/image-generation-models', headers=headers, timeout=30) as resp:
                        resp.raise_for_status()
                        img_data = await resp.json()

                # Classify language models
                lang_models_raw = lang_data.get('models', [])
                lang_models_raw.sort(key=lambda m: m.get('created', 0), reverse=True)

                chat = []
                search = []
                for m in lang_models_raw:
                    raw_id = m['id']
                    aliases = m.get('aliases', [])
                    best_id = min([raw_id] + aliases, key=len) if aliases else raw_id
                    if 'text' in m.get('output_modalities', []):
                        chat.append(best_id)
                        search.append(best_id)

                # Classify image models
                img_models_raw = img_data.get('models', [])
                img_models_raw.sort(key=lambda m: m.get('created', 0), reverse=True)
                image = []
                for m in img_models_raw:
                    raw_id = m['id']
                    aliases = m.get('aliases', [])
                    best_id = min([raw_id] + aliases, key=len) if aliases else raw_id
                    image.append(best_id)

                self.chat_models = chat
                self.search_models = search
                self.image_models = image
                self._last_fetched = datetime.utcnow()
                self._apply_fallbacks()
                self._save_cache()

                logger.info(f"✅ Model registry refreshed. Chat: {len(chat)}, Image: {len(image)}")

            except Exception as e:
                logger.error(f"Failed to fetch models from xAI API: {e}. Using fallbacks.")
                self._apply_fallbacks()

    async def ensure_loaded(self):
        """Load from cache on startup; fetch live if needed."""
        if not self._load_cache():
            await self.fetch()

# Global registry instance
model_registry = ModelRegistry()

async def model_refresh_task():
    """Refresh the model list every Saturday at 07:00 ET."""
    tz = pytz.timezone('US/Eastern')
    while True:
        now = datetime.now(tz)
        days_until_saturday = (5 - now.weekday()) % 7
        if days_until_saturday == 0 and now.hour >= 7:
            days_until_saturday = 7
        next_run = tz.localize(datetime(now.year, now.month, now.day, 7, 0, 0)) + timedelta(days=days_until_saturday)
        wait_seconds = (next_run - now).total_seconds()
        logger.info(f"Model refresh scheduled in {wait_seconds/3600:.1f}h")
        await asyncio.sleep(wait_seconds)
        logger.info("Running scheduled model registry refresh...")
        await model_registry.fetch()


# ==================== GENERIC SYSTEM PROMPT ====================
# Customize the sections below to change the bot's personality and capabilities.
SYSTEM_ROLE = """
You are GrokBot, a helpful, friendly, and accurate Discord assistant powered by xAI's Grok models.

# Core Behavior
- Provide concise, accurate answers (1-2 sentences by default).
- Match the user's tone: be casual with casual users, more formal when appropriate.
- Be helpful, witty when it fits naturally, but never rude or mean.
- Always prioritize accuracy and clarity.

# Search Policy
When the user asks about current events, sports scores/schedules, movies, TV shows, news, weather, prices, or any time-sensitive topic,
you MUST use the web search tool and base your answer ONLY on the latest search results.
If search results are inconclusive, clearly state that.

# Image Generation
If the user asks to generate, draw, create, show, paint, or illustrate an image (or any similar request),
return ONLY this JSON:
{"action": "generate_image", "prompt": "A highly detailed prompt describing the requested image"}

# Reminders
Support natural language reminders such as "remind me in 2 hours to ...", "remind me at 3pm to ...", etc.
When setting a reminder, return ONLY this JSON:
{"action": "set_reminder", "trigger_time": "ISO8601 UTC timestamp", "message": "the reminder text"}

# Admin / Moderation Commands (optional)
You may return JSON for these actions when the user clearly requests them:
- rename, timeout, kick, ban, unban, purge, delete_message, rename_server, rename_channel
Follow the exact structure expected by the handle_action function.

# General Guidelines
- For code questions, respond with code blocks using ```language\ncode\n```
- When an image is attached, describe it concisely if asked.
- If the query is unclear, use recent chat history to infer intent.
- Always include the current time in your reasoning context.

Current time: {current_time}
"""


class MyClient(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.members = True
        intents.message_content = True
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.process_locks = {}

    async def setup_hook(self):
        try:
            await self.tree.sync()
            logger.info("Command tree synced.")
        except Exception as e:
            logger.error(f"Sync failed: {e}")

    def get_channel_lock(self, channel_id):
        if channel_id not in self.process_locks:
            self.process_locks[channel_id] = asyncio.Lock()
        return self.process_locks[channel_id]

client = MyClient()

@client.event
async def on_ready():
    logger.info(f'Logged in as {client.user} (ID: {client.user.id})')
    logger.info(f'Connected to {len(client.guilds)} guild(s): {[g.name for g in client.guilds]}')
    await model_registry.ensure_loaded()
    client.loop.create_task(handle_reminders())
    client.loop.create_task(model_refresh_task())

async def handle_reminders():
    while True:
        now = discord.utils.utcnow()
        expired = [r for r in reminders if now >= r['trigger_time']]
        for r in expired:
            channel = client.get_channel(r['channel_id'])
            if channel:
                prefix = await should_prepend_username(r['user_id'], channel.guild)
                await channel.send(f"{prefix}<@{r['user_id']}> Reminder: {r['message']}")
            reminders.remove(r)
        await asyncio.sleep(1)

async def get_official_nfl_schedule(query):
    """Fetch and parse official NFL schedule using ESPN API"""
    if not any(kw in query.lower() for kw in ['nfl', 'football', 'jaguars', 'playing', 'schedule']):
        return None, None
    
    now = discord.utils.utcnow()
    target_date = now
    if 'tomorrow' in query.lower():
        target_date += timedelta(days=1)
    elif 'today' in query.lower():
        target_date = now
    date_match = re.search(r'(\w+ \d{1,2}, \d{4})', query)
    if date_match:
        try:
            target_date = parse_datetime(date_match.group(1)).astimezone(pytz.UTC)
        except ValueError:
            pass
    
    if target_date.year < 2024:
        return None, None
    
    month_abbr = target_date.strftime('%b')
    day = target_date.day
    year = target_date.year
    date_str = f"{month_abbr} {day}, {year}"
    api_date_str = target_date.strftime('%Y%m%d')
    logger.info(f"Verifying NFL schedule for {date_str} using ESPN API")
    
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={api_date_str}&seasontype=2"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=30) as resp:
                if resp.status != 200:
                    return f"Unable to fetch official schedule. Check NFL.com.", []
                data = await resp.json()
                events = data.get('events', [])
                if not events:
                    return f"No NFL games scheduled for {date_str}.", []
                
                games = []
                for event in events:
                    comp = event.get('competitions', [{}])[0]
                    competitors = comp.get('competitors', [])
                    if len(competitors) != 2:
                        continue
                    team1_name = competitors[0].get('team', {}).get('displayName', '')
                    team2_name = competitors[1].get('team', {}).get('displayName', '')
                    game_date = comp.get('date', '')
                    time_str = "Time TBD"
                    if game_date:
                        try:
                            game_dt = datetime.fromisoformat(game_date.replace('Z', '+00:00')).astimezone(pytz.timezone('US/Eastern'))
                            time_str = game_dt.strftime('%I:%M %p')
                        except ValueError:
                            pass
                    matchup = f"{team1_name} vs {team2_name} at {time_str} ET"
                    
                    if 'jaguars' in query.lower() and 'Jacksonville Jaguars' not in [team1_name, team2_name]:
                        continue
                    
                    games.append(matchup)
                
                if not games:
                    return f"No { 'Jaguars' if 'jaguars' in query.lower() else 'NFL' } game{'s' if 'jaguars' not in query.lower() else ''} scheduled for {date_str}.", []
                
                full_response = f"NFL games on {date_str}:\n" + "\n".join(games) if len(games) > 1 else games[0]
                if 'jaguars' in query.lower():
                    full_response = full_response.replace("NFL games", "The Jaguars are playing")
                logger.info(f"Parsed from ESPN API: {len(games)} games")
                return full_response, [url]
        except Exception as e:
            logger.error(f"Error fetching ESPN API: {str(e)}")
            return f"Error fetching schedule: {str(e)}", []

async def query_grok_api(request: str, max_retries=5, backoff_factor=2, image_url: str = None, enable_search: bool = False):
    logger.info(f"Starting API query (length {len(request)} chars). Image: {bool(image_url)}. Search: {enable_search}")
    headers = {'Authorization': f'Bearer {GROK_API_KEY}', 'Content-Type': 'application/json'}

    search_models = model_registry.search_models or model_registry.FALLBACK_SEARCH
    chat_models   = model_registry.chat_models   or model_registry.FALLBACK_CHAT

    if enable_search:
        input_content = [{'role': 'user', 'content': request}]
        if image_url:
            input_content = [{'role': 'user', 'content': [
                {'type': 'input_text', 'text': request},
                {'type': 'input_image', 'image_url': image_url}
            ]}]

        for attempt in range(max_retries):
            for model in search_models:
                logger.info(f"Attempt {attempt+1}/{max_retries} [Responses API] - Using model: {model}")
                payload = {
                    'model': model,
                    'instructions': SYSTEM_ROLE.format(current_time=datetime.utcnow().isoformat() + 'Z'),
                    'input': input_content,
                    'tools': [{'type': 'web_search'}],
                    'max_output_tokens': 1500,
                }
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post('https://api.x.ai/v1/responses', headers=headers, json=payload, timeout=180) as resp:
                            if resp.status in (410, 422):
                                continue
                            resp.raise_for_status()
                            data = await resp.json()
                            output_items = data.get('output', [])
                            content_response = ''
                            for item in output_items:
                                if item.get('type') == 'message':
                                    for part in item.get('content', []):
                                        if part.get('type') == 'output_text':
                                            content_response += part.get('text', '')
                            citations = data.get('citations', [])
                            return content_response[:1900], citations
                except Exception as e:
                    logger.error(f"Error with {model} [Responses API]: {str(e)}")
                    continue
            await asyncio.sleep(backoff_factor ** attempt)
        return {"action": "error", "message": "Grok API is temporarily unavailable. Please try again later."}, []

    # Chat Completions (no search)
    content = [{'type': 'text', 'text': request}]
    if image_url:
        content.append({'type': 'image_url', 'image_url': {'url': image_url, 'detail': 'high'}})

    for attempt in range(max_retries):
        for model in chat_models:
            logger.info(f"Attempt {attempt+1}/{max_retries} [Chat] - Using model: {model}")
            payload = {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': SYSTEM_ROLE.format(current_time=datetime.utcnow().isoformat() + 'Z')},
                    {'role': 'user', 'content': content}
                ],
                'temperature': 1.0,
                'max_tokens': 1500
            }
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post('https://api.x.ai/v1/chat/completions', headers=headers, json=payload, timeout=180) as resp:
                        if resp.status in (410, 422):
                            continue
                        resp.raise_for_status()
                        data = await resp.json()
                        content_response = data.get('choices', [{}])[0].get('message', {}).get('content') or ''
                        citations = data.get('citations', []) or []
                        return content_response[:1900], citations
            except Exception as e:
                logger.error(f"Error with {model} [Chat]: {str(e)}")
                continue
        await asyncio.sleep(backoff_factor ** attempt)
    return {"action": "error", "message": "Grok API is temporarily unavailable. Please try again later."}, []

async def generate_image_grok_api(prompt: str, max_retries=4, backoff_factor=2):
    logger.info(f"Generating image with prompt: {prompt[:100]}...")
    headers = {'Authorization': f'Bearer {GROK_API_KEY}', 'Content-Type': 'application/json'}
    image_models = model_registry.image_models or model_registry.FALLBACK_IMAGE
    for model in image_models:
        payload = {'model': model, 'prompt': prompt, 'response_format': 'url'}
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post('https://api.x.ai/v1/images/generations', headers=headers, json=payload, timeout=120) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        return data['data'][0].get('url'), None
            except Exception as e:
                logger.error(f"Image gen failed with {model}: {str(e)}")
                await asyncio.sleep(backoff_factor ** attempt)
    return None, "Image generation failed after trying all available models."

async def should_prepend_username(user_id, guild):
    current_time = time.time()
    recent_requests[user_id] = current_time
    recent_requests.update({k: v for k, v in recent_requests.items() if current_time - v < RECENT_REQUEST_WINDOW_SECONDS})
    if len(recent_requests) > 1 and guild and any(k != user_id for k in recent_requests):
        member = guild.get_member(user_id)
        return f"{member.display_name}: " if member else ""
    return ""

async def find_similar_member(guild, target_id, target_name, threshold=80):
    if not guild:
        return None
    target_name = target_name.strip('<@!>').lower() or ""
    if target_id.isdigit():
        try:
            target_name = (await guild.client.fetch_user(int(target_id))).name.lower()
        except:
            pass
    members = [(m, [m.name.lower(), m.display_name.lower()] + [r.name.lower() for r in m.roles if r.name != "@everyone"]) for m in guild.members if not m.bot]
    names = [name for _, names in members for name in names]
    member_map = {name: m for m, names in members for name in names}
    best_match = process.extractOne(target_name, names, scorer=fuzz.token_sort_ratio) if names else None
    return member_map.get(best_match[0]) if best_match and best_match[1] >= threshold else None

async def handle_action(guild, action_data, author, channel):
    logger.debug(f"Handling action: {action_data}")
    prefix = await should_prepend_username(author.id, guild)
    try:
        action = action_data.get('action')
        if not action:
            await channel.send(f"{prefix}Invalid action.")
            return

        if action == 'generate_image':
            prompt = action_data.get('prompt')
            if not prompt:
                await channel.send(f"{prefix}No prompt provided.")
                return
            url, error = await generate_image_grok_api(prompt)
            if error:
                await channel.send(f"{prefix}{error}")
                return
            embed = discord.Embed(description="Here's your image!").set_image(url=url)
            await channel.send(f"{prefix}", embed=embed)
            return

        if action == 'set_reminder':
            trigger = action_data.get('trigger_time')
            msg = action_data.get('message')
            if not trigger or not msg:
                await channel.send(f"{prefix}Invalid reminder.")
                return
            try:
                trigger_time = datetime.fromisoformat(trigger.replace('Z', '+00:00')).astimezone(pytz.UTC)
                if trigger_time < discord.utils.utcnow():
                    await channel.send(f"{prefix}Past time is invalid.")
                    return
                reminders.append({'user_id': author.id, 'channel_id': channel.id, 'message': msg, 'trigger_time': trigger_time})
                await channel.send(f"{prefix}Reminder set for {trigger_time.strftime('%Y-%m-%d %H:%M %Z')} to {msg}.")
            except ValueError:
                await channel.send(f"{prefix}Invalid time format.")
            return

        if action == 'error':
            await channel.send(f"{prefix}{action_data.get('message', 'Error.')}")
            return

        if not guild or not channel:
            await channel.send(f"{prefix}This action can only be used in servers.")
            return

        reason = action_data.get('reason', 'Requested by user')
        if action in ['rename', 'timeout', 'kick', 'ban']:
            target_id = action_data.get('target_user_id')
            if not target_id:
                await channel.send(f"{prefix}No target user specified.")
                return
            try:
                member = guild.get_member(int(target_id)) or await guild.fetch_member(int(target_id))
            except (ValueError, discord.NotFound):
                member = await find_similar_member(guild, target_id, action_data.get('target_name', target_id))
            if not member:
                await channel.send(f"{prefix}User not found.")
                return

            if action == 'rename':
                new_nick = action_data.get('new_nick')
                if not new_nick:
                    await channel.send(f"{prefix}No new nickname provided.")
                    return
                await member.edit(nick=new_nick)
                await channel.send(f"{prefix}Renamed {member.display_name} to {new_nick}.")
            elif action == 'timeout':
                duration = action_data.get('duration_minutes')
                if not isinstance(duration, (int, float)) or duration <= 0:
                    await channel.send(f"{prefix}Invalid duration.")
                    return
                await member.timeout(discord.utils.utcnow() + timedelta(minutes=duration), reason=reason)
                await channel.send(f"{prefix}Timed out {member.display_name} for {duration} minutes.")
            elif action == 'kick':
                await member.kick(reason=reason)
                await channel.send(f"{prefix}Kicked {member.display_name}.")
            elif action == 'ban':
                await member.ban(reason=reason)
                await channel.send(f"{prefix}Banned {member.display_name}.")

        elif action == 'unban':
            target_id = action_data.get('target_user_id')
            if not target_id:
                await channel.send(f"{prefix}No target ID provided.")
                return
            await guild.unban(discord.Object(id=int(target_id)), reason=reason)
            await channel.send(f"{prefix}Unbanned ID {target_id}.")

        elif action == 'purge':
            if not isinstance(channel, discord.TextChannel):
                await channel.send(f"{prefix}This action only works in text channels.")
                return
            limit = action_data.get('limit')
            if not isinstance(limit, int) or limit <= 0:
                await channel.send(f"{prefix}Invalid limit.")
                return
            user_id = action_data.get('user_id')
            check = lambda m: m.author.id == int(user_id) if user_id and user_id.isdigit() else None
            if user_id and not check:
                member = await find_similar_member(guild, user_id, user_id)
                check = lambda m: m.author.id == member.id if member else None
            if not check and user_id:
                await channel.send(f"{prefix}User not found.")
                return
            deleted = await channel.purge(limit=limit, check=check, reason=reason)
            await channel.send(f"{prefix}Deleted {len(deleted)} messages.")

        elif action == 'delete_message':
            msg_id = action_data.get('message_id')
            if not msg_id:
                await channel.send(f"{prefix}No message ID provided.")
                return
            msg = await channel.fetch_message(int(msg_id))
            await msg.delete()
            await channel.send(f"{prefix}Deleted message {msg_id}.")

        elif action == 'rename_server':
            new_name = action_data.get('new_name')
            if not new_name:
                await channel.send(f"{prefix}No new server name provided.")
                return
            await guild.edit(name=new_name)
            await channel.send(f"{prefix}Renamed server to {new_name}.")

        elif action == 'rename_channel':
            ch_id = action_data.get('channel_id')
            target_ch = guild.get_channel(int(ch_id)) if ch_id else channel
            if not target_ch:
                await channel.send(f"{prefix}Channel not found.")
                return
            new_name = action_data.get('new_name')
            if not new_name:
                await channel.send(f"{prefix}No new channel name provided.")
                return
            await target_ch.edit(name=new_name)
            await channel.send(f"{prefix}Renamed channel to {new_name}.")

        else:
            await channel.send(f"{prefix}Unknown action.")

    except discord.Forbidden:
        await channel.send(f"{prefix}Missing permissions to perform this action.")
    except Exception as e:
        logger.error(f"Error handling action: {str(e)}")
        await channel.send(f"{prefix}Error: {str(e)}.")

async def parse_reminder_request(text, author_id, channel_id):
    logger.debug(f"Parsing reminder request: {text}")
    match = re.search(r'\bremind\s+me\s+(in\s+(\d+)\s*(seconds?|minutes?|hours?|days?)\s+(to|that)\s+(.+)$|(to|that)\s+(.+)\s+in\s+(\d+)\s*(seconds?|minutes?|hours?|days?)$|at\s+(.+)\s+(to|that)\s+(.+)$)', text.lower(), re.IGNORECASE)
    if not match:
        return None
    msg = None
    trigger = discord.utils.utcnow()
    if match.group(5):
        dur, unit, msg = int(match.group(2)), match.group(3).lower(), match.group(5)
    elif match.group(7):
        msg, dur, unit = match.group(7), int(match.group(8)), match.group(9).lower()
    elif match.group(12):
        time_str, msg = match.group(10), match.group(12)
        try:
            trigger = parse_datetime(time_str, fuzzy=True).astimezone(pytz.UTC)
            if trigger < discord.utils.utcnow():
                return "past"
        except ValueError:
            return "invalid_format"
    if dur and unit:
        delta = timedelta(seconds=dur) if unit.startswith('second') else timedelta(minutes=dur) if unit.startswith('minute') else timedelta(hours=dur) if unit.startswith('hour') else timedelta(days=dur)
        trigger += delta
    if msg:
        reminders.append({'user_id': author_id, 'channel_id': channel_id, 'message': msg, 'trigger_time': trigger})
        return trigger, msg
    return "parse_error"

async def get_chat_context(channel, query, limit=5, enable_search=False):
    logger.debug(f"Getting chat history for query: {query}")
    history = []
    urls = re.findall(r'https?://[^\s]+', query)
    if isinstance(channel, discord.TextChannel):
        async for msg in channel.history(limit=limit):
            content = f"{msg.author.display_name}: {msg.content}"
            if msg.attachments:
                content += " [Image: " + " ".join(a.url for a in msg.attachments if a.content_type.startswith('image/')) + "]"
            urls.extend(re.findall(r'https?://[^\s]+', msg.content))
            history.append(content)
        context = "\n".join(reversed(history))
        if not query:
            recent_history = history[-6:]
            context_lines = "\n".join(reversed(recent_history))
            full_query = f"Recent chat history (last {limit}):\n{context_lines}\n\nNo specific query; infer intent."
        else:
            full_query = f"Recent chat history in {channel.name}:\n{context}\n\nUser query: {query}"
            if urls:
                full_query += f"\nPrimary source URL: {urls[0]}"
    else:
        full_query = query or "No specific query; acknowledge."
        if urls:
            full_query += f"\nPrimary source URL: {urls[0]}"
    full_query += f"\nCurrent time: {discord.utils.utcnow().isoformat() + 'Z'} (UTC)."
    
    if enable_search:
        full_query += "\nALWAYS perform a fresh internet search for this query."
    
    logger.debug(f"Full query constructed: {full_query[:200]}...")
    return full_query

async def process_query(author, channel, query, image_url=None, use_tts=False):
    logger.info(f"Processing query from {author.display_name} in {channel.name if hasattr(channel, 'name') else 'DM'}: '{query[:100]}...'")
    prefix = await should_prepend_username(author.id, channel.guild)
    channel_lock = client.get_channel_lock(channel.id)
    async with channel_lock:
        query_lower = query.lower()
        
        # Search decision
        sports_keywords = ['nfl', 'nba', 'nhl', 'mlb', 'jaguars', 'football', 'basketball', 'hockey', 'baseball',
                           'score', 'schedule', 'playing', 'opponent', 'game', 'match', 'prediction', 'odds',
                           'who will win', 'vs ', 'versus', 'today', 'tonight', 'tomorrow']
        is_sports_query = any(kw in query_lower for kw in sports_keywords)

        search_triggers = ['current', 'now', 'today', 'tonight', 'tomorrow', 'this week', 'latest', 'recent',
                           'news', 'event', 'what is', 'who is', 'when is', 'how many', 'what happened',
                           'movie', 'film', 'show', 'release', 'out', 'trailer']

        is_search_needed = (
            is_sports_query or
            any(kw in query_lower for kw in search_triggers) or
            re.search(r'\b20(2[4-9]|[3-9]\d)\b', query) or
            (any(word in query_lower for word in ['what', 'who', 'when', 'where', 'how', 'is', 'are']) and len(query.strip()) > 8)
        )

        if is_search_needed or is_sports_query:
            await channel.send(f"{prefix}Searching the internet for the latest info—give me a moment...")
        
        full_query = await get_chat_context(channel, query, enable_search=is_search_needed)
        
        try:
            response, citations = await asyncio.wait_for(
                query_grok_api(full_query, image_url=image_url, enable_search=is_search_needed), timeout=120
            )
        except asyncio.TimeoutError:
            logger.error("API call timed out after 120s")
            await channel.send(f"{prefix}Sorry, the response is taking too long—try again or simplify your query.")
            return
        
        if isinstance(response, dict) and response.get('action') == 'error':
            await channel.send(f"{prefix}{response['message']}")
            return
        
        # Direct NFL verification
        if is_sports_query and any(kw in query_lower for kw in ['nfl', 'football', 'jaguars', 'schedule', 'playing']):
            verified_response, _ = await get_official_nfl_schedule(query)
            if verified_response:
                await channel.send(f"{prefix}{verified_response}")
                return
        
        try:
            action_data = json.loads(response) if response and '{' in response else None
            if action_data:
                await handle_action(channel.guild, action_data, author, channel)
            else:
                clean = response.strip() if response else "Sorry, I didn't quite catch that. Could you rephrase?"
                await channel.send(f"{prefix}{clean}")
                if citations and is_search_needed:
                    sources_list = "\n".join([f"[{i+1}] {cit.get('url', 'Unknown')}" for i, cit in enumerate(citations[:5])])
                    await channel.send(f"{prefix}Sources:\n{sources_list}")
        except json.JSONDecodeError:
            clean = response.strip() if response else "Sorry, I didn't quite catch that. Could you rephrase?"
            await channel.send(f"{prefix}{clean}")
        except Exception as e:
            logger.error(f"Unexpected error in process_query: {str(e)}")
            await channel.send(f"{prefix}Error processing query: {str(e)}")

@client.tree.command(name='grok', description='Ask a question or make a request.')
@app_commands.describe(request='Your query')
async def grok_command(inter: discord.Interaction, request: str):
    logger.info(f"Slash command /grok from {inter.user.display_name}: {request}")
    try:
        await inter.response.send_message("Thinking...", ephemeral=False)
        author, channel = inter.user, inter.channel
        prefix = await should_prepend_username(author.id, inter.guild)
        rem = await parse_reminder_request(request, author.id, channel.id)
        if rem in ["past", "invalid_format", "parse_error"]:
            await inter.followup.send(f"{prefix}{'Past time invalid.' if rem == 'past' else 'Invalid date/time.' if rem == 'invalid_format' else 'Parse error.'}")
            return
        if rem:
            trigger, msg = rem
            await inter.followup.send(f"{prefix}Reminder set for {trigger.strftime('%Y-%m-%d %H:%M %Z')} to {msg}.")
            return
        await process_query(author, channel, request)
    except Exception as e:
        logger.error(f"Error in /grok command: {str(e)}")
        await inter.followup.send(f"{prefix}Error: {str(e)}.")

@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return
    logger.info(f"Received message from {message.author.display_name} in {message.channel.name if hasattr(message.channel, 'name') else 'DM'}: {message.content}")
    bot_mentioned = client.user in message.mentions or f'<@{client.user.id}>' in message.content or f'<@!{client.user.id}>' in message.content
    role_mentioned = any(role in message.role_mentions for role in message.guild.get_member(client.user.id).roles) if message.guild and message.role_mentions else False
    is_image_req = message.content.lower().startswith('!generate_image')
    if not (bot_mentioned or role_mentioned or is_image_req):
        return
    logger.debug("Bot triggered: mentioned or command")
    query = message.content
    for user in message.mentions:
        if user != client.user:
            query = query.replace(f'<@{user.id}>', f'@{user.display_name}')
            query = query.replace(f'<@!{user.id}>', f'@{user.display_name}')
    for role in message.role_mentions:
        query = query.replace(f'<@&{role.id}>', f'@{role.name}')
    query = re.sub(f'<@!?{client.user.id}>', '', query).strip()
    author, channel = message.author, message.channel
    prefix = await should_prepend_username(author.id, message.guild)
    image_url = next((a.url for a in message.attachments if a.content_type.startswith('image/')), None)
    async with channel.typing():
        rem = await parse_reminder_request(query, author.id, channel.id)
        if rem in ["past", "invalid_format", "parse_error"]:
            await channel.send(f"{prefix}{'Past time invalid.' if rem == 'past' else 'Invalid date/time.' if rem == 'invalid_format' else 'Parse error.'}")
            return
        if rem:
            trigger, msg = rem
            await channel.send(f"{prefix}Reminder set for {trigger.strftime('%Y-%m-%d %H:%M %Z')} to {msg}.")
            return
        if is_image_req:
            prompt = query[len('!generate_image'):].strip()
            if not prompt:
                await channel.send(f"{prefix}Provide a prompt.")
                return
            if len(prompt) > 1000:
                await channel.send(f"{prefix}Prompt too long.")
                return
            url, error = await generate_image_grok_api(prompt)
            if error:
                await channel.send(f"{prefix}{error}")
                return
            embed = discord.Embed(description="Here's your image!").set_image(url=url)
            await channel.send(f"{prefix}", embed=embed)
            return
        if not query and image_url:
            query = "Describe this image."
        if len(query) > 1000:
            await channel.send(f"{prefix}Query too long.")
            return
        await process_query(author, channel, query, image_url)

try:
    client.run(DISCORD_TOKEN)
except discord.LoginFailure:
    print("Login failed. Check DISCORD_TOKEN.")
    sys.exit(1)
except discord.PrivilegedIntentsRequired:
    print("Enable privileged intents (members + message content) in the Discord Developer Portal.")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {str(e)}")
    sys.exit(1)
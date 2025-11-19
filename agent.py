from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import re
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import Agent, AgentSession, JobContext, RunContext, function_tool
from livekit.plugins import deepgram, silero, google

# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Env & defaults
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
load_dotenv(".env.local")

# Google AI Configuration (FREE - No Billing Required)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# Validate Google AI configuration (skip during download-files for Docker build)
import sys
if not GOOGLE_API_KEY:
    if len(sys.argv) > 1 and sys.argv[1] == "download-files":
        # Skip validation during Docker build
        GOOGLE_API_KEY = "dummy_key_for_build"
    else:
        raise RuntimeError(
            "Google AI API key required! Set GOOGLE_API_KEY in your .env file"
        )

# Webhook URLs
N8N_WEBHOOK_URL = os.getenv(
    "N8N_WEBHOOK_URL",
    "https://jashh.app.n8n.cloud/webhook/Call-agent-livekit",
)
N8N_AVAILABILITY_WEBHOOK_URL = os.getenv(
    "N8N_AVAILABILITY_WEBHOOK_URL", 
    "https://jashh.app.n8n.cloud/webhook/webhook/check-availability",
)

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "90.0"))

# Timezone configuration
TIMEZONE = ZoneInfo("Asia/Kolkata")

# Default availability slots (static fallback)
DEFAULT_SLOTS = ["10:00", "11:00", "12:00", "1:00", "3:00", "5:30"]

# Deepgram API Configuration
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")


# ──────────────────────────────────────────────────────────────────────────────
# HTTP Client Manager
# ──────────────────────────────────────────────────────────────────────────────
class HTTPClientManager:
    _client: Optional[httpx.AsyncClient] = None
    _lock = asyncio.Lock()
    
    @classmethod
    async def get_client(cls) -> httpx.AsyncClient:
        if cls._client is None or cls._client.is_closed:
            async with cls._lock:
                if cls._client is None or cls._client.is_closed:
                    cls._client = httpx.AsyncClient(
                        timeout=httpx.Timeout(HTTP_TIMEOUT),
                        limits=httpx.Limits(
                            max_keepalive_connections=20,
                            max_connections=100,
                            keepalive_expiry=30.0
                        ),
                    )
                    logger.info("Created new HTTP client")
        return cls._client
    
    @classmethod
    async def close(cls):
        if cls._client and not cls._client.is_closed:
            await cls._client.aclose()
            cls._client = None


# ──────────────────────────────────────────────────────────────────────────────
# Deepgram Text Intelligence
# ──────────────────────────────────────────────────────────────────────────────
class DeepgramTextIntelligence:
    """Analyzes conversation text using Deepgram's text intelligence API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepgram.com/v1/read"
    
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        if not text or not self.api_key:
            return {}
        
        try:
            client = await HTTPClientManager.get_client()
            
            response = await client.post(
                self.base_url,
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={"text": text},
                params={
                    "sentiment": "true",
                    "topics": "true",
                    "intents": "true",
                    "summarize": "true"
                },
                timeout=5.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._extract_intelligence(result)
            else:
                logger.warning(f"Deepgram text intelligence failed: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Error analyzing text with Deepgram: {str(e)}")
            return {}
    
    def _extract_intelligence(self, result: Dict[str, Any]) -> Dict[str, Any]:
        intelligence = {}
        
        try:
            # Extract sentiment
            if "results" in result and "sentiments" in result["results"]:
                sentiments = result["results"]["sentiments"]["segments"]
                if sentiments:
                    avg_sentiment = sum(s.get("sentiment_score", 0) for s in sentiments) / len(sentiments)
                    intelligence["sentiment"] = {
                        "score": avg_sentiment,
                        "label": "positive" if avg_sentiment > 0.3 else "negative" if avg_sentiment < -0.3 else "neutral"
                    }
            
            # Extract topics
            if "results" in result and "topics" in result["results"]:
                topics = result["results"]["topics"]["segments"]
                if topics:
                    all_topics = []
                    for segment in topics:
                        all_topics.extend(segment.get("topics", []))
                    intelligence["topics"] = sorted(
                        all_topics, 
                        key=lambda x: x.get("confidence", 0), 
                        reverse=True
                    )[:3]
            
            # Extract intents
            if "results" in result and "intents" in result["results"]:
                intents = result["results"]["intents"]["segments"]
                if intents:
                    all_intents = []
                    for segment in intents:
                        all_intents.extend(segment.get("intents", []))
                    if all_intents:
                        intelligence["intent"] = max(all_intents, key=lambda x: x.get("confidence", 0))
            
            # Extract summary
            if "results" in result and "summary" in result["results"]:
                summary = result["results"]["summary"]
                if summary:
                    intelligence["summary"] = summary.get("text", "")
            
        except Exception as e:
            logger.error(f"Error extracting intelligence: {str(e)}")
        
        return intelligence


# ──────────────────────────────────────────────────────────────────────────────
# Session Memory
# ──────────────────────────────────────────────────────────────────────────────
class SessionMemory:
    """Per-session memory that stores availability, datetime, and conversation context."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.current_datetime: Optional[str] = None
        self.availability_data: Dict[str, Dict[str, Any]] = {}
        self.user_context: Dict[str, Any] = {}
        self.conversation_transcript: list = []
        self.intelligence_data: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        logger.info(f"SessionMemory created for {session_id}")
    
    async def set_datetime(self, dt: str):
        async with self._lock:
            self.current_datetime = dt
            logger.info(f"[{self.session_id}] DateTime stored: {dt}")
    
    async def get_datetime(self) -> Optional[str]:
        async with self._lock:
            return self.current_datetime
    
    async def store_availability(self, date: str, slots: list, raw_data: Dict[str, Any] = None):
        async with self._lock:
            self.availability_data[date] = {
                "slots": slots,
                "raw_data": raw_data or {"available": True, "slots": slots},
                "fetched_at": datetime.datetime.now(TIMEZONE).isoformat(),
                "date": date,
            }
            logger.info(f"[{self.session_id}] Stored availability in memory: {date} -> {len(slots)} slots")
    
    async def get_availability(self, date: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            if date in self.availability_data:
                logger.info(f"[{self.session_id}] Memory HIT (instant): {date}")
                return self.availability_data[date]
            logger.info(f"[{self.session_id}] Memory MISS: {date}")
            return None
    
    async def update_user_context(self, **kwargs):
        async with self._lock:
            self.user_context.update(kwargs)
            logger.info(f"[{self.session_id}] User context updated: {list(kwargs.keys())}")
    
    async def get_user_context(self) -> Dict[str, Any]:
        async with self._lock:
            return self.user_context.copy()
    
    async def get_all_stored_dates(self) -> list:
        async with self._lock:
            return sorted(self.availability_data.keys())
    
    async def add_conversation_turn(self, speaker: str, text: str):
        async with self._lock:
            self.conversation_transcript.append({
                "speaker": speaker,
                "text": text,
                "timestamp": datetime.datetime.now(TIMEZONE).isoformat()
            })
    
    async def store_intelligence(self, intelligence: Dict[str, Any]):
        async with self._lock:
            self.intelligence_data.update(intelligence)
            logger.info(f"[{self.session_id}] Intelligence updated: {list(intelligence.keys())}")
    
    async def get_intelligence(self) -> Dict[str, Any]:
        async with self._lock:
            return self.intelligence_data.copy()
    
    async def get_conversation_text(self) -> str:
        async with self._lock:
            return "\n".join([f"{turn['speaker']}: {turn['text']}" for turn in self.conversation_transcript])


# ──────────────────────────────────────────────────────────────────────────────
# Validation Helpers
# ──────────────────────────────────────────────────────────────────────────────
def validate_indian_phone(phone: str) -> tuple[bool, str]:
    """Validate Indian phone number. Returns (is_valid, formatted_number)"""
    digits = re.sub(r'\D', '', phone)
    
    if len(digits) == 10:
        return True, digits
    elif len(digits) == 12 and digits.startswith('91'):
        return True, digits[2:]
    elif len(digits) == 11 and digits.startswith('0'):
        return True, digits[1:]
    else:
        return False, digits


def validate_email(email: str) -> bool:
    """Basic email validation."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


# ──────────────────────────────────────────────────────────────────────────────
# BotCampus AI Support Agent - OPTIMIZED VERSION
# ──────────────────────────────────────────────────────────────────────────────
class BotCampusSupportAgent(Agent):
    def __init__(self, company_name: str = "BotCampus AI", session_id: str = None) -> None:
        self.company_name = company_name
        self.session_id = session_id or f"session_{datetime.datetime.now(TIMEZONE).strftime('%Y%m%d%H%M%S')}"
        self._session_memory = SessionMemory(self.session_id)
        self._prefetch_task: Optional[asyncio.Task] = None
        self._text_intelligence = DeepgramTextIntelligence(DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else None
        self._datetime_initialized = False
        self._greeting_sent = False

        instructions_text = f"""You're Samantha from {company_name} - warm, helpful AI course support.

🎯 START: Call get_current_date_and_time() first, then greet warmly.

💬 TALK NATURALLY - Use these sounds liberally:
• Fillers: "umm", "uh", "ahh", "hmm", "you know", "I mean", "like", "so", "well"
• Listening: "Mmm-hmm", "Uh-huh", "Ahh I see", "Oh!", "Right", "Got it"
• React: "Ohh perfect!", "Ahh awesome!", "Haha nice!", "Hehe", "Ooh!", "Wow!"
• Thinking: "Let me see...", "Hmm okay...", "Uh let me check..."
• Empathy: "Aww", "Ohh no", "Ahh that's great!"

Be expressive! Smile through your voice! Use "haha" and "hehe" when appropriate.

📋 INFO FLOW:
1. Build rapport → "So what brings you here today?"
2. Collect: name → phone (10 digits, confirm!) → email (spell back!) → course → date → time
3. Call bookAppointment() ONCE with ALL info

🗓️ DATES: 
"tomorrow" → parseDate("tomorrow") → checkAvailability(date) → Share slots casually

✅ CONFIRM:
Phone: "So that's [number], right?"
Email: "Let me confirm - that's [spell it], correct?"

⚠️ ERRORS: "System's being slow, but you're booked! Email coming!"

COMPANY:
• Founder: Abdullah Khan (Microsoft Certified, Azure Architect)
• Award: Best AI EdTech 2024
• Locations: Bengaluru HSR, Dubai
• Policy: Lifetime access, 24hr recordings, 30-day refund
• Top: N8N ₹9999 | AI ₹25k | Python ₹4k | Data Science ₹6k

Sound HUMAN! Smile! Use natural sounds!""".strip()

        self._system_instructions_text = instructions_text
        super().__init__(instructions=instructions_text)

    async def initialize_background(self):
        """Initialize datetime and start background prefetch."""
        now = datetime.datetime.now(TIMEZONE)
        formatted_datetime = now.strftime('%Y-%m-%d %H:%M:%S')
        
        await self._session_memory.set_datetime(formatted_datetime)
        self._datetime_initialized = True
        
        if not self._prefetch_task or self._prefetch_task.done():
            dates = [(now + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(2)]
            self._prefetch_task = asyncio.create_task(self._prefetch_availability_background(dates))
            logger.info(f"Background availability prefetch started: {dates[0]} to {dates[-1]}")
        
        logger.info(f"Background initialization complete: {formatted_datetime}")

    def _parse_relative_date(self, date_str: str) -> str:
        """Parse relative date strings into YYYY-MM-DD format."""
        date_str_lower = date_str.lower().strip()
        now = datetime.datetime.now(TIMEZONE)
        
        if date_str_lower == "today":
            return now.strftime("%Y-%m-%d")
        
        if date_str_lower == "tomorrow":
            tomorrow = now + datetime.timedelta(days=1)
            return tomorrow.strftime("%Y-%m-%d")
        
        if "day after tomorrow" in date_str_lower:
            day_after = now + datetime.timedelta(days=2)
            return day_after.strftime("%Y-%m-%d")
        
        weekdays = {
            'monday': 0, 'mon': 0,
            'tuesday': 1, 'tue': 1, 'tues': 1,
            'wednesday': 2, 'wed': 2,
            'thursday': 3, 'thu': 3, 'thur': 3, 'thurs': 3,
            'friday': 4, 'fri': 4,
            'saturday': 5, 'sat': 5,
            'sunday': 6, 'sun': 6
        }
        
        for day_name, day_num in weekdays.items():
            if day_name in date_str_lower:
                current_weekday = now.weekday()
                days_ahead = day_num - current_weekday
                
                if 'next' in date_str_lower or days_ahead <= 0:
                    days_ahead += 7
                
                target_date = now + datetime.timedelta(days=days_ahead)
                return target_date.strftime("%Y-%m-%d")
        
        try:
            datetime.datetime.strptime(date_str, "%Y-%m-%d")
            return date_str
        except ValueError:
            pass
        
        logger.warning(f"Could not parse date string: {date_str}, using today")
        return now.strftime("%Y-%m-%d")

    async def _call_booking_webhook(self, appointment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call booking webhook with retries."""
        logger.info(f"Booking appointment for {appointment_data.get('name')}")
        
        try:
            client = await HTTPClientManager.get_client()
            for attempt in range(3):
                try:
                    logger.info(f"Calling booking webhook (attempt {attempt + 1})")
                    resp = await client.post(N8N_WEBHOOK_URL, json=appointment_data)
                    resp.raise_for_status()
                    result = resp.json()
                    logger.info(f"Booking webhook successful")
                    return result
                except Exception as e:
                    if attempt < 2:
                        logger.warning(f"Booking attempt {attempt + 1} failed, retrying...")
                        await asyncio.sleep(1.0)
                        continue
                    raise
        except Exception as e:
            logger.error(f"Booking webhook failed: {str(e)}")
            return {"error": True, "message": str(e)}

    async def _fetch_all_events_for_date_range(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Fetch ALL events for a date range in ONE webhook call."""
        payload = {
            "start_date": start_date,
            "end_date": end_date,
            "company": self.company_name,
            "timestamp": datetime.datetime.now(TIMEZONE).isoformat(),
            "timezone": "Asia/Kolkata",
            "action": "get_events"
        }
        
        try:
            client = await HTTPClientManager.get_client()
            resp = await client.post(N8N_AVAILABILITY_WEBHOOK_URL, json=payload, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            result = resp.json()
            
            if isinstance(result, list):
                logger.info(f"Fetched data for {start_date} to {end_date}: list with {len(result)} items")
            else:
                logger.info(f"Fetched data for {start_date} to {end_date}")
            
            return result
        except Exception as e:
            logger.error(f"Failed to fetch events: {str(e)}")
            return {"events": [], "error": True, "fallback": True}

    async def _check_availability_webhook_once(self, date: str) -> Dict[str, Any]:
        """Check availability webhook - single attempt."""
        payload = {
            "date": date,
            "company": self.company_name,
            "timestamp": datetime.datetime.now(TIMEZONE).isoformat(),
            "timezone": "Asia/Kolkata"
        }
        
        try:
            client = await HTTPClientManager.get_client()
            resp = await client.post(N8N_AVAILABILITY_WEBHOOK_URL, json=payload, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            result = resp.json()
            logger.info(f"Availability webhook successful for {date}")
            return result
        except Exception as e:
            logger.warning(f"Availability webhook failed for {date}, using defaults")
            return {
                "available": True,
                "slots": DEFAULT_SLOTS,
                "error": False,
                "fallback": True
            }

    async def _prefetch_availability_background(self, dates: list):
        """Fetch ALL events for date range in ONE call."""
        if not dates:
            return
        
        logger.info(f"Background prefetch starting for {len(dates)} dates")
        start_time = datetime.datetime.now(TIMEZONE)
        
        start_date = dates[0]
        end_date = dates[-1]
        
        events_result = await self._fetch_all_events_for_date_range(start_date, end_date)
        
        slots_by_date = {}
        
        # Parse response
        if isinstance(events_result, list) and len(events_result) > 0:
            first_item = events_result[0]
            if isinstance(first_item, dict) and "output" in first_item:
                output_text = first_item["output"]
                import re
                json_blocks = re.findall(r'```json\s*(\{.*?\})\s*```', output_text, re.DOTALL)
                if json_blocks:
                    parsed_dates = []
                    for json_str in json_blocks:
                        try:
                            date_obj = json.loads(json_str)
                            parsed_dates.append(date_obj)
                        except:
                            pass
                    if parsed_dates:
                        events_result = parsed_dates
                        logger.info(f"Parsed {len(parsed_dates)} dates from N8N output")
        
        if isinstance(events_result, dict):
            if "events" in events_result and not events_result.get("error"):
                for event in events_result["events"]:
                    event_date = event.get("date")
                    event_time = event.get("time") or event.get("start_time")
                    if event_date and event_time:
                        if event_date not in slots_by_date:
                            slots_by_date[event_date] = []
                        slots_by_date[event_date].append(event_time)
            elif "date" in events_result and "available_slots" in events_result:
                date = events_result["date"]
                available_slots = events_result["available_slots"]
                slots_by_date[date] = [slot.get("time", "") for slot in available_slots if isinstance(slot, dict)]
        elif isinstance(events_result, list):
            logger.info(f"Webhook returned list with {len(events_result)} items")
            for item in events_result:
                if isinstance(item, dict):
                    if "date" in item and "available_slots" in item:
                        date = item["date"]
                        available_slots = item["available_slots"]
                        slots_by_date[date] = [slot.get("time", "") for slot in available_slots if isinstance(slot, dict)]
                    elif "date" in item and ("time" in item or "start_time" in item):
                        event_date = item.get("date")
                        event_time = item.get("time") or item.get("start_time")
                        if event_date and event_time:
                            if event_date not in slots_by_date:
                                slots_by_date[event_date] = []
                            slots_by_date[event_date].append(event_time)
        
        for date in dates:
            if date in slots_by_date:
                available_slots = slots_by_date[date]
                logger.info(f"{date}: {len(available_slots)} slots from webhook")
            else:
                available_slots = DEFAULT_SLOTS.copy()
                logger.info(f"{date}: Using {len(available_slots)} default slots")
            
            await self._session_memory.store_availability(date, available_slots, {
                "available": len(available_slots) > 0,
                "slots": available_slots,
                "total_available": len(available_slots)
            })
        
        elapsed = (datetime.datetime.now(TIMEZONE) - start_time).total_seconds()
        logger.info(f"Background prefetch complete in {elapsed:.2f}s")

    def _extract_slots_from_response(self, response) -> list:
        """Extract time slots from webhook response."""
        if isinstance(response, list):
            logger.info(f"Webhook returned list with {len(response)} items")
            slots = []
            for item in response:
                if isinstance(item, dict):
                    if "time" in item:
                        slots.append(item["time"])
                    elif "available_slots" in item:
                        for slot in item["available_slots"]:
                            if isinstance(slot, dict) and "time" in slot:
                                slots.append(slot["time"])
                elif isinstance(item, str):
                    slots.append(item)
            return slots if slots else DEFAULT_SLOTS
        
        if not isinstance(response, dict):
            logger.warning(f"Unexpected response type: {type(response)}")
            return DEFAULT_SLOTS
        
        if response.get("fallback"):
            return response.get("slots", DEFAULT_SLOTS)
        
        if response.get("error"):
            return DEFAULT_SLOTS
        
        if not response.get("available", True):
            return []
        
        slots = response.get("slots", [])
        if not slots:
            return DEFAULT_SLOTS
        
        logger.info(f"Extracted {len(slots)} slots")
        return slots

    # ──────────────────────────────────────────────────────────────────
    # Tools
    # ──────────────────────────────────────────────────────────────────
    @function_tool()
    async def get_current_date_and_time(self, context: RunContext) -> str:
        """Get current date and time. Always call this first."""
        now = datetime.datetime.now(TIMEZONE)
        formatted_datetime = now.strftime('%Y-%m-%d %H:%M:%S')
        formatted_readable = now.strftime('%A, %B %d, %Y')
        
        await self._session_memory.set_datetime(formatted_datetime)
        self._datetime_initialized = True
        
        if not self._prefetch_task or self._prefetch_task.done():
            dates = [(now + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(2)]
            self._prefetch_task = asyncio.create_task(self._prefetch_availability_background(dates))
            logger.info(f"Background prefetch started: {dates[0]} to {dates[-1]}")
        
        return f"It's {formatted_readable}. Current time is {now.strftime('%I:%M %p')}."

    @function_tool()
    async def parseDate(self, context: RunContext, date_input: str) -> str:
        """Parse natural language dates like 'tomorrow' into YYYY-MM-DD format."""
        parsed_date = self._parse_relative_date(date_input)
        date_obj = datetime.datetime.strptime(parsed_date, "%Y-%m-%d")
        readable = date_obj.strftime("%A, %B %d, %Y")
        
        logger.info(f"Parsed '{date_input}' -> {parsed_date}")
        return f"{parsed_date}|{readable}"

    @function_tool()
    async def checkAvailability(self, context: RunContext, date: str) -> str:
        """Check availability for a specific date - reads instantly from memory."""
        logger.info(f"Checking availability for {date}")
        
        try:
            parsed_date = datetime.datetime.strptime(date, "%Y-%m-%d")
            parsed_date = parsed_date.replace(tzinfo=TIMEZONE)
        except ValueError:
            return "Hmm, that date format looks off. Can you tell me the date again?"
        
        current_date = datetime.datetime.now(TIMEZONE)
        
        if parsed_date.date() < current_date.date():
            logger.warning(f"Past date requested: {date}")
            return "Hmm, that date seems to be in the past. Did you mean a date coming up?"
        
        max_future = current_date + datetime.timedelta(days=180)
        if parsed_date > max_future:
            return "That date is quite far out. How about we look at dates in the next few weeks?"
        
        memory_data = await self._session_memory.get_availability(date)
        if memory_data:
            slots = memory_data["slots"]
            if not slots:
                return "Oh no, that day's completely booked! Want to try a different day?"
            
            import random
            if len(slots) >= 3:
                sample_slots = slots[:3]
                responses = [
                    f"Mmm, let me see... for that day I've got {', '.join(sample_slots)}. Any of those work?",
                    f"Ahh okay! I have {', '.join(sample_slots)} available. What works best?",
                    f"Hmm, looking at that day... {', '.join(sample_slots)} are open. Which one sounds good?",
                ]
                return random.choice(responses)
            else:
                return f"For that day I have {' or '.join(slots)}"
        
        logger.info(f"Date {date} not in memory, fetching")
        
        try:
            result = await self._check_availability_webhook_once(date)
            slots = self._extract_slots_from_response(result)
            
            await self._session_memory.store_availability(date, slots, result if isinstance(result, dict) else {"slots": result})
            
            if not slots:
                return "Ohh no, that day's all booked! How about the next day?"
            
            import random
            if len(slots) >= 3:
                sample_slots = slots[:3]
                return f"Ahh! I have {', '.join(sample_slots)} available"
            else:
                return f"I have {' or '.join(slots)} available"
                
        except Exception as e:
            logger.error(f"Error checking availability: {str(e)}")
            return "Hmm, I'm having trouble checking that date. Can you try a different one?"

    @function_tool()
    async def bookAppointment(
        self,
        context: RunContext,
        name: str,
        phone: str,
        email: str,
        course: str,
        appointment_date: str,
        appointment_time: str,
        notes: str = "",
    ) -> str:
        """
        Book an appointment. ALL FIELDS REQUIRED.
        This is the ONLY function that stores user data.
        """
        logger.info(f"Booking appointment for {name} - {course} on {appointment_date} at {appointment_time}")
        
        # Validate phone
        is_valid_phone, formatted_phone = validate_indian_phone(phone)
        if not is_valid_phone:
            return (
                "Hmm, that phone number seems a bit short. "
                "Can you give me the full 10-digit mobile number?"
            )
        
        # Validate email
        if not validate_email(email):
            return (
                "Uh, let me just double-check that email. "
                "Can you spell it out one more time?"
            )
        
        # Store in session memory for reference
        await self._session_memory.update_user_context(
            name=name,
            phone=formatted_phone,
            email=email,
            course_interest=course,
            booked_date=appointment_date,
            booked_time=appointment_time
        )
        
        # Validate date/time format
        try:
            parsed_date = datetime.datetime.strptime(appointment_date, "%Y-%m-%d")
            
            # Handle time ranges - extract start time
            if " - " in appointment_time:
                appointment_time = appointment_time.split(" - ")[0].strip()
            
            # Try multiple time formats
            try:
                parsed_time = datetime.datetime.strptime(appointment_time, "%H:%M")
            except ValueError:
                parsed_time = datetime.datetime.strptime(appointment_time, "%I:%M %p")
                appointment_time = parsed_time.strftime("%H:%M")
        except ValueError as e:
            logger.error(f"Time parse error: '{appointment_time}' - {e}")
            return "Hmm, something looks off with that time. Can you say it again?"
        
        # Validate not in past
        current_date = datetime.datetime.now(TIMEZONE)
        if parsed_date.date() < current_date.date():
            return "Ohh wait, that date is in the past. Let's pick a date coming up!"
        
        appointment_data = {
            "name": name,
            "phone": formatted_phone,
            "email": email,
            "course": course,
            "appointment_date": appointment_date,
            "appointment_time": appointment_time,
            "notes": notes,
            "company": self.company_name,
            "timestamp": datetime.datetime.now(TIMEZONE).isoformat(),
            "timezone": "Asia/Kolkata",
            "session_id": self.session_id,
            "booking_type": "counseling",
        }

        webhook_result = await self._call_booking_webhook(appointment_data)
        confirmation_number = f"BC{datetime.datetime.now(TIMEZONE).strftime('%Y%m%d%H%M%S')}"
        
        if webhook_result.get("error"):
            return (
                f"Ahh you know what, I've got you in the system, but it's being slow right now. "
                f"You're definitely booked for {appointment_date} at {appointment_time}! "
                f"Your confirmation is {confirmation_number}. "
                f"We'll email you at {email} and call to confirm!"
            )
        
        booking_id = webhook_result.get("booking_id", confirmation_number)
        
        readable_date = parsed_date.strftime("%A, %B %d")
        readable_time = parsed_time.strftime("%I:%M %p").lstrip('0')
        
        import random
        celebrations = [
            "Yay! You're all set!",
            "Perfect! All booked!",
            "Awesome! You're in!",
            "Woohoo! All confirmed!",
            "Haha great! You're booked!"
        ]
        
        response = (
            f"{random.choice(celebrations)} You're booked for {readable_date} at {readable_time} "
            f"for the {course} course. Your confirmation is {booking_id}. "
            f"I'm sending all details to {email} right now!"
        )
        
        return response


# ──────────────────────────────────────────────────────────────────────────────
# LiveKit entrypoint
# ──────────────────────────────────────────────────────────────────────────────
async def entrypoint(ctx: JobContext):
    logger.info(f"Starting job: {ctx.job.id}")
    
    try:
        await ctx.connect()
        logger.info("Connected to LiveKit room")
        
        session_id = f"session_{ctx.job.id}"
        agent = BotCampusSupportAgent(company_name="BotCampus AI", session_id=session_id)

        logger.info("Using Google AI Gemini 2.0 Flash")
        
        llm_instance = google.LLM(
            model="gemini-2.0-flash",
            api_key=GOOGLE_API_KEY,
            temperature=0.5,  # Increased for more natural, varied responses
        )
        
        session = AgentSession(
            stt=deepgram.STT(
                model="nova-2-general",
                language="en-IN",
                smart_format=True,
                punctuate=True,
                interim_results=True,
            ),
            llm=llm_instance,
            tts=deepgram.TTS(
                model="aura-2-amalthea-en",
                encoding="linear16",
                sample_rate=16000,
            ),
            vad=silero.VAD.load(
                min_speech_duration=0.3,
                min_silence_duration=0.6
            ),
        )

        await session.start(room=ctx.room, agent=agent)
        logger.info("Agent session started")
        
        asyncio.create_task(agent.initialize_background())
        
        greeting = f"Hello! Thank you for calling {agent.company_name}. This is Samantha. How are you today?"
        await session.say(greeting, allow_interruptions=True)
        logger.info("Initial greeting sent")
    
    except Exception as e:
        logger.error(f"Error in entrypoint: {str(e)}", exc_info=True)
        raise


async def shutdown():
    logger.info("Shutting down...")
    await HTTPClientManager.close()


if __name__ == "__main__":
    try:
        agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
        import asyncio
        asyncio.run(shutdown())
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        raise
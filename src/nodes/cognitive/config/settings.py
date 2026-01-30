"""
Configuration settings for cognitive module.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BASE_DIR.parent
# Project root: go up from nodes/ to project root (src/nodes -> src -> project root)
PROJECT_ROOT = PROJECT_DIR.parent.parent
PROMPTS_DIR = PROJECT_DIR / "cognitive/prompts"
SYSTEM_PROMPT_FILE = PROMPTS_DIR / "intent_system.txt"

# Load .env from project root
env_loaded = False
for env_path in [PROJECT_ROOT / ".env", PROJECT_DIR / ".env", Path(".env")]:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        env_loaded = True
        break
if not env_loaded:
    # Try loading from current directory (fallback)
    load_dotenv(override=False)

# =============================================================================
# OPENAI API
# =============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/responses"
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_TEMPERATURE = 0.3
OPENAI_TIMEOUT = 5  # Reduced timeout for faster fallback
OPENAI_MAX_TOKENS = 256  # Limit response tokens for faster generation

# =============================================================================
# TTS SERVICE
# =============================================================================
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://localhost:8001")
TTS_TIMEOUT = 30
TTS_HEALTH_TIMEOUT = 5
MIC_RESUME_DELAY = 0.3  # Delay before resuming mic after TTS (seconds)

# =============================================================================
# CONFIDENCE
# =============================================================================
CONFIDENCE_THRESHOLD = 0.8
CONFIDENCE_HIGH = 1.0
CONFIDENCE_MEDIUM = 0.8
CONFIDENCE_LOW = 0.5

# =============================================================================
# NAVIGATE COMMANDS
# =============================================================================
DEFAULT_DISTANCE = 1.0  # meter
DEFAULT_ANGLE = 90.0    # degree

MOVE_COMMANDS = {
    "forwards": "FWD",
    "backwards": "BWD",
}

TURN_COMMANDS = {
    "left": "TL",
    "right": "TR",
}

# =============================================================================
# INTENT ROUTES
# =============================================================================
COMMAND_INTENTS = {"navigate", "stop"}
RESPONSE_INTENTS = {"greeting", "noise", "conversation", "unknown"}
ACTION_INTENTS = {"tracking-person", "go_to_object", "go_to_location"}

# =============================================================================
# GREETING PATTERNS
# =============================================================================
INTENT_GREETING = "greeting"
INTENT_NOT_GREETING = "not_greeting"
INTENT_NOISE = "noise"

GREETING_WORDS = [
    "chào", "xin chào", "hello", "hi", "hey",
    "chào bạn", "chào anh", "chào chị", "chào em",
]

TIME_PHRASES = [
    "buổi sáng", "buổi chiều", "buổi tối",
    "sáng", "chiều", "tối",
]

WISH_PHRASES = ["vui vẻ", "tốt lành", "an lành"]

COMPANION_WORDS = {"bạn", "robot", "anh", "chị", "em", "cậu", "mày", "ông", "bà"}

ENDING_PARTICLES = {"nhé", "nha", "nhá", "nè", "hen", "ha", "á", "ạ", "ơi", "oi"}

NOISE_WORDS = {"ơ", "ờ", "à", "ừ", "ư", "hử", "hả", "gì", "sao"}


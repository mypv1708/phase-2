"""
OpenAI API client for intent classification.
Uses requests with connection pooling for faster repeated calls.
"""
import json
import logging
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config.settings import (
    OPENAI_API_KEY, OPENAI_API_URL, OPENAI_MODEL,
    OPENAI_TEMPERATURE, OPENAI_TIMEOUT, OPENAI_MAX_TOKENS, SYSTEM_PROMPT_FILE,
)

logger = logging.getLogger(__name__)

# Cache system prompt (load once)
_system_prompt_cache: Optional[str] = None

# Reusable session with connection pooling (keep-alive)
_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Get or create reusable HTTP session with connection pooling."""
    global _session
    if _session is None:
        _session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        # Mount adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=1,
            pool_maxsize=1,
        )
        _session.mount("https://", adapter)
        
        # Set default headers
        _session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Connection": "keep-alive",
        })
        
    return _session


def get_system_prompt() -> str:
    """Get cached system prompt."""
    global _system_prompt_cache
    if _system_prompt_cache is None:
        if not SYSTEM_PROMPT_FILE.exists():
            raise FileNotFoundError(f"System prompt not found: {SYSTEM_PROMPT_FILE}")
        _system_prompt_cache = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
    return _system_prompt_cache


def call_openai(user_input: str, system_prompt: str = None) -> dict:
    """
    Call OpenAI Responses API with connection pooling.
    Uses keep-alive connections for faster repeated calls.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    data = {
        "model": OPENAI_MODEL,
        "instructions": system_prompt or get_system_prompt(),
        "input": user_input,
        "temperature": OPENAI_TEMPERATURE,
        "max_output_tokens": OPENAI_MAX_TOKENS,
    }
    
    session = _get_session()
    
    try:
        start_time = time.perf_counter()
        logger.info("OpenAI API request started...")
        
        response = session.post(
            OPENAI_API_URL,
            json=data,
            timeout=OPENAI_TIMEOUT,
        )
        response.raise_for_status()
        
        elapsed = time.perf_counter() - start_time
        logger.info("OpenAI API response received in %.2fs", elapsed)
        
        return response.json()
    except requests.exceptions.HTTPError as e:
        elapsed = time.perf_counter() - start_time
        error_body = e.response.text if e.response else str(e)
        logger.error("OpenAI API Error after %.2fs: %s", elapsed, error_body)
        raise RuntimeError(f"OpenAI API Error {e.response.status_code}: {error_body}")
    except requests.exceptions.Timeout:
        elapsed = time.perf_counter() - start_time
        logger.error("OpenAI API timeout after %.2fs", elapsed)
        raise RuntimeError(f"OpenAI API timeout after {OPENAI_TIMEOUT}s")
    except requests.exceptions.RequestException as e:
        elapsed = time.perf_counter() - start_time
        logger.error("OpenAI API request failed after %.2fs: %s", elapsed, e)
        raise RuntimeError(f"OpenAI API request failed: {e}")


def extract_text(response: dict) -> str:
    """Extract text from OpenAI Responses API output."""
    # Fast path
    if "output_text" in response:
        return response["output_text"]
    
    # Parse nested structure
    texts = []
    for item in response.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    texts.append(content.get("text", ""))
    return "".join(texts)


def parse_json_response(response_text: str) -> dict:
    """Parse JSON from LLM response."""
    text = response_text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.startswith("```")]
        text = "\n".join(lines)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "slots": {},
            "response": "",
            "raw_text": response_text,
        }

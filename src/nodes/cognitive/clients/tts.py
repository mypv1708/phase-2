"""
TTS Service client.
"""
import json
import logging
import urllib.request
import urllib.error

from ..config.settings import TTS_SERVICE_URL, TTS_TIMEOUT, TTS_HEALTH_TIMEOUT

logger = logging.getLogger(__name__)


def speak_text(text: str, verbose: bool = False) -> bool:
    """
    Send text to TTS service for playback.
    Note: Recording is already paused by recording.py before this is called.
    This function does NOT resume recording - caller must handle resume.
    """
    if not text or not text.strip():
        return False
    
    url = f"{TTS_SERVICE_URL}/speak"
    
    try:
        data = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        if verbose:
            print(f"[TTS] POST {url}")
        
        with urllib.request.urlopen(req, timeout=TTS_TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))
            if verbose:
                print(f"[TTS] Response: {result}")
            
            # Only log success if playback_complete is True
            if result.get("playback_complete"):
                logger.info("TTS playback confirmed complete")
            else:
                logger.info("TTS response received")
            
            return result.get("success", False)
            
    except urllib.error.HTTPError as e:
        if verbose:
            print(f"[TTS] HTTP Error {e.code}: {e.reason}")
        return False
    except urllib.error.URLError as e:
        if verbose:
            print(f"[TTS] Connection Error: {e.reason}")
        return False
    except Exception as e:
        if verbose:
            print(f"[TTS] Error: {e}")
        return False


def check_tts_health() -> bool:
    """Check if TTS service is running."""
    try:
        url = f"{TTS_SERVICE_URL}/health"
        with urllib.request.urlopen(url, timeout=TTS_HEALTH_TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("status") == "ok"
    except Exception:
        return False


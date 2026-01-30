"""
Recording control - pause/resume recording during TTS playback.
"""
import threading
import logging

logger = logging.getLogger(__name__)

# Global flags (threading.Event is already thread-safe)
_recording_paused = threading.Event()
_need_clear_buffer = threading.Event()


def pause_recording() -> None:
    """Pause recording."""
    _recording_paused.set()
    logger.debug("Recording paused")


def resume_recording() -> None:
    """Resume recording and signal to clear buffers."""
    _need_clear_buffer.set()
    _recording_paused.clear()
    logger.debug("Recording resumed")


def is_recording_paused() -> bool:
    """Check if recording is paused."""
    return _recording_paused.is_set()


def should_clear_buffer() -> bool:
    """Check if buffer should be cleared (and reset the flag)."""
    if _need_clear_buffer.is_set():
        _need_clear_buffer.clear()
        return True
    return False


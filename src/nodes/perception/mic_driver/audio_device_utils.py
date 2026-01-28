"""Audio device utilities for handling device conflicts and validation."""
import logging
from typing import Optional

import pyaudio

logger = logging.getLogger(__name__)

# Global shared PyAudio instance to avoid conflicts
_pyaudio_instance: Optional[pyaudio.PyAudio] = None


def check_pyaudio_available() -> bool:
    """
    Check if PyAudio is available (imported successfully).

    Returns:
        True if PyAudio is available, False otherwise
    """
    if pyaudio is None:
        logger.warning("PyAudio not available")
        return False
    return True


def validate_audio_device(
    device_index: Optional[int] = None,
    sample_rate: int = 48000,
    channels: int = 1,
) -> bool:
    """
    Validate that an audio device can be opened with given parameters.
    Uses shared PyAudio instance for efficiency.

    Args:
        device_index: Device index (None for default)
        sample_rate: Required sample rate
        channels: Required number of channels

    Returns:
        True if device is valid, False otherwise
    """
    if pyaudio is None:
        logger.error("PyAudio not available")
        return False
    
    # Use shared instance if available, otherwise create temporary one
    pa = get_shared_pyaudio_instance()
    use_shared = pa is not None
    
    if not use_shared:
        try:
            pa = pyaudio.PyAudio()
        except Exception as e:
            logger.error("Failed to create PyAudio instance: %s", e)
            return False
    
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=1024,
            input_device_index=device_index,
        )
        stream.close()
        return True
    except Exception as e:
        logger.warning(
            "Audio device validation failed (index=%s, rate=%d, channels=%d): %s",
            device_index,
            sample_rate,
            channels,
            e,
        )
        return False
    finally:
        # Only terminate if we created a temporary instance
        if not use_shared and pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass


def get_shared_pyaudio_instance() -> Optional[pyaudio.PyAudio]:
    """
    Get or create a shared PyAudio instance to avoid conflicts.

    Returns:
        PyAudio instance or None if not available
    """
    global _pyaudio_instance

    if pyaudio is None:
        return None

    if _pyaudio_instance is None:
        try:
            _pyaudio_instance = pyaudio.PyAudio()
            logger.debug("Created shared PyAudio instance")
        except Exception as e:
            logger.error("Failed to create PyAudio instance: %s", e)
            return None

    return _pyaudio_instance


def release_audio_resources() -> None:
    """Release all audio resources."""
    global _pyaudio_instance

    if _pyaudio_instance is not None:
        try:
            _pyaudio_instance.terminate()
            _pyaudio_instance = None
            logger.debug("Released shared PyAudio instance")
        except Exception as e:
            logger.warning("Error releasing PyAudio: %s", e)


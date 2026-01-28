import logging
from typing import Tuple

import pyaudio
import webrtcvad

from .audio_device_utils import (
    check_pyaudio_available,
    get_shared_pyaudio_instance,
    validate_audio_device,
)
from .config import (
    CHANNELS,
    FORMAT,
    FRAME_SIZE,
    RATE,
    SAMPLE_WIDTH,
    VAD_MODE,
)

logger = logging.getLogger(__name__)


def init_audio_stream() -> Tuple[pyaudio.PyAudio, webrtcvad.Vad, pyaudio.Stream]:
    """Initialize audio input stream with VAD (Voice Activity Detection)."""
    # Check PyAudio availability
    if not check_pyaudio_available():
        raise RuntimeError("PyAudio is not available")

    # Validate audio device before opening (non-blocking check)
    if not validate_audio_device(
        device_index=None,
        sample_rate=RATE,
        channels=CHANNELS,
    ):
        logger.warning(
            "Audio device validation failed, attempting to open anyway..."
        )

    # Use shared PyAudio instance to avoid conflicts
    pa = get_shared_pyaudio_instance()
    if pa is None:
        # Fallback to creating new instance if shared not available
        pa = pyaudio.PyAudio()

    vad = webrtcvad.Vad(VAD_MODE)
    try:
        stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=FRAME_SIZE,
        )
        return pa, vad, stream
    except OSError as e:
        logger.error("Failed to open audio input stream: %s", e)
        raise RuntimeError(f"Failed to open audio input stream: {e}") from e
    except Exception as e:
        logger.error("Audio stream initialization failed: %s", e)
        raise RuntimeError(f"Audio stream initialization failed: {e}") from e


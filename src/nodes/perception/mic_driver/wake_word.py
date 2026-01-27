import logging
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
from eff_word_net import RATE as EFF_WORD_NET_RATE
from eff_word_net.engine import HotwordDetector

from .audio_device_utils import (
    check_audio_devices_available,
    is_sounddevice_in_use,
    mark_sounddevice_in_use,
    validate_audio_device,
)
from .config import (
    HOTWORD_NAME,
    HOTWORD_BLOCKSIZE,
    HOTWORD_SLIDING_WINDOW_SECS,
    HOTWORD_WINDOW_LENGTH_SECS,
    WAKE_WORD_CHANNELS,
    WAKE_WORD_DTYPE,
    WAKE_WORD_MIN_AUDIO_LEVEL,
    WAKE_WORD_SLEEP_MS,
    PIP_SOUND_FILE,
)
from .audio import play_audio_file

logger = logging.getLogger(__name__)

# Cache display name to avoid repeated string operations
_HOTWORD_DISPLAY_NAME = HOTWORD_NAME.replace("_", " ")


def wait_for_wake_word(detector: HotwordDetector) -> bool:
    """Wait for wake word detection using sounddevice."""
    # Check audio device availability
    _, sounddevice_available = check_audio_devices_available()
    if not sounddevice_available:
        raise RuntimeError("sounddevice is not available")

    # Check for conflicts with PyAudio
    if is_sounddevice_in_use():
        logger.warning(
            "sounddevice already in use, potential conflict detected"
        )

    # Validate audio device before opening
    if not validate_audio_device(
        device_index=None,
        sample_rate=EFF_WORD_NET_RATE,
        channels=WAKE_WORD_CHANNELS,
        use_pyaudio=False,
    ):
        logger.warning(
            "Audio device validation failed, attempting to open anyway..."
        )

    try:
        window_samples = int(
            HOTWORD_WINDOW_LENGTH_SECS * EFF_WORD_NET_RATE
        )  # 1.5s window
        step_samples = int(
            HOTWORD_SLIDING_WINDOW_SECS * EFF_WORD_NET_RATE
        )  # 0.75s step

        audio_buffer = deque(maxlen=window_samples)
        samples_since_last = 0
        wake_word_detected = threading.Event()

        logger.info("Listening for wake word '%s'...", _HOTWORD_DISPLAY_NAME)

        def audio_callback(indata, frames, time_info, status):
            nonlocal samples_since_last
            if status:
                logger.warning("Audio stream status: %s", status)
            if wake_word_detected.is_set():
                return
            
            mono_audio = indata[:, 0] if indata.ndim > 1 else indata
            audio_buffer.extend(mono_audio)
            samples_since_last += len(mono_audio)
            
            if samples_since_last >= step_samples and len(audio_buffer) >= window_samples:
                samples_since_last = 0
                # Convert deque to numpy array efficiently
                frame = np.array(audio_buffer, dtype=np.float32)
                
                # Quick check for audio level before processing
                max_abs = np.max(np.abs(frame))
                if max_abs > WAKE_WORD_MIN_AUDIO_LEVEL:
                    inference_start = time.perf_counter()
                    result = detector.scoreFrame(frame)
                    inference_time = time.perf_counter() - inference_start
                    
                    if result and result.get("match", False):
                        confidence = result.get("confidence", 0.0)
                        audio_duration = len(frame) / EFF_WORD_NET_RATE
                        speed_ratio = (
                            audio_duration / inference_time 
                            if inference_time > 0 else 0.0
                        )
                        # Wake word detection typically runs on CPU
                        logger.info(
                            "HOTWORD DETECTED | confidence=%.3f "
                            "(inference=%.3fms, audio=%.2fs, speed=%.2fx, device=CPU)",
                            confidence,
                            inference_time * 1000,  # Convert to ms
                            audio_duration,
                            speed_ratio,
                        )
                        wake_word_detected.set()
                        def play_pip():
                            try:
                                play_audio_file(PIP_SOUND_FILE)
                            except Exception as e:
                                logger.warning("Failed to play pip sound: %s", e)
                        threading.Thread(target=play_pip, daemon=True).start()

        try:
            # Mark sounddevice as in use to prevent conflicts
            mark_sounddevice_in_use()

            with sd.InputStream(
                samplerate=EFF_WORD_NET_RATE,
                channels=WAKE_WORD_CHANNELS,
                dtype=WAKE_WORD_DTYPE,
                blocksize=HOTWORD_BLOCKSIZE,
                callback=audio_callback,
            ):
                while not wake_word_detected.is_set():
                    sd.sleep(WAKE_WORD_SLEEP_MS)
            return True
        except sd.PortAudioError as e:
            logger.error("PortAudio error during wake word detection: %s", e)
            raise RuntimeError(f"Audio stream error: {e}") from e
    except KeyboardInterrupt:
        logger.info("Wake word detection interrupted by user")
        return False
    except RuntimeError:
        raise
    except Exception as e:
        logger.exception("Wake word detection failed: %s", e)
        raise RuntimeError(f"Wake word detection failed: {e}") from e
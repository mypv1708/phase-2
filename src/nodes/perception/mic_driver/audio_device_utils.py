"""Audio device utilities for handling device conflicts and validation."""
import logging
from typing import List, Optional, Tuple
import pyaudio
import sounddevice as sd

logger = logging.getLogger(__name__)

# Global flag to track audio device usage
_pyaudio_instance: Optional[pyaudio.PyAudio] = None
_sounddevice_in_use = False


def check_audio_devices_available() -> Tuple[bool, bool]:
    """
    Check if audio devices are available.

    Returns:
        Tuple of (pyaudio_available, sounddevice_available)
    """
    pyaudio_available = pyaudio is not None
    sounddevice_available = sd is not None

    if not pyaudio_available:
        logger.warning("PyAudio not available")
    if not sounddevice_available:
        logger.warning("sounddevice not available")

    return pyaudio_available, sounddevice_available


def list_audio_devices() -> Tuple[List[dict], List[dict]]:
    """
    List available audio input devices.

    Returns:
        Tuple of (pyaudio_devices, sounddevice_devices)
    """
    pyaudio_devices = []
    sounddevice_devices = []

    if pyaudio is not None:
        try:
            pa = pyaudio.PyAudio()
            device_count = pa.get_device_count()
            for i in range(device_count):
                info = pa.get_device_info_by_index(i)
                if info["maxInputChannels"] > 0:
                    pyaudio_devices.append({
                        "index": i,
                        "name": info["name"],
                        "channels": info["maxInputChannels"],
                        "sample_rate": int(info["defaultSampleRate"]),
                    })
            pa.terminate()
        except Exception as e:
            logger.warning("Failed to list PyAudio devices: %s", e)

    if sd is not None:
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device["max_input_channels"] > 0:
                    sounddevice_devices.append({
                        "index": i,
                        "name": device["name"],
                        "channels": device["max_input_channels"],
                        "sample_rate": int(device["default_samplerate"]),
                    })
        except Exception as e:
            logger.warning("Failed to list sounddevice devices: %s", e)

    return pyaudio_devices, sounddevice_devices


def validate_audio_device(
    device_index: Optional[int] = None,
    sample_rate: int = 48000,
    channels: int = 1,
    use_pyaudio: bool = True,
) -> bool:
    """
    Validate that an audio device can be opened with given parameters.

    Args:
        device_index: Device index (None for default)
        sample_rate: Required sample rate
        channels: Required number of channels
        use_pyaudio: Use PyAudio (True) or sounddevice (False)

    Returns:
        True if device is valid, False otherwise
    """
    if use_pyaudio:
        if pyaudio is None:
            logger.error("PyAudio not available")
            return False
        try:
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=1024,
                input_device_index=device_index,
            )
            stream.close()
            pa.terminate()
            return True
        except Exception as e:
            logger.warning(
                "PyAudio device validation failed (index=%s): %s",
                device_index,
                e,
            )
            return False
    else:
        if sd is None:
            logger.error("sounddevice not available")
            return False
        try:
            with sd.InputStream(
                device=device_index,
                samplerate=sample_rate,
                channels=channels,
                blocksize=1024,
            ):
                pass
            return True
        except Exception as e:
            logger.warning(
                "sounddevice validation failed (index=%s): %s",
                device_index,
                e,
            )
            return False


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
    global _pyaudio_instance, _sounddevice_in_use

    if _pyaudio_instance is not None:
        try:
            _pyaudio_instance.terminate()
            _pyaudio_instance = None
            logger.debug("Released shared PyAudio instance")
        except Exception as e:
            logger.warning("Error releasing PyAudio: %s", e)

    _sounddevice_in_use = False


def mark_sounddevice_in_use() -> None:
    """Mark sounddevice as in use to prevent conflicts."""
    global _sounddevice_in_use
    _sounddevice_in_use = True


def is_sounddevice_in_use() -> bool:
    """Check if sounddevice is currently in use."""
    return _sounddevice_in_use


"""
Audio playback utilities for TTS output.

Note: This runs in TTS service (separate process).
Mic recording pause/resume is handled by mic_driver_node, not here.
"""
import io
import logging
import os
import subprocess
import tempfile
import wave
import numpy as np
import sounddevice as sd

from .config import (
    AUDIO_PLAYBACK_TIMEOUT,
    AUDIO_PLAYERS,
)

logger = logging.getLogger(__name__)


def play_audio_bytes(audio_bytes: bytes) -> None:
    """
    Play audio bytes using available audio player.
    """
    # Try sounddevice first (best quality)
    try:
        wav_buffer = io.BytesIO(audio_bytes)
        with wave.open(wav_buffer, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())
            
            if sample_width == 1:
                audio_array = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
            elif sample_width == 2:
                audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            if channels > 1:
                audio_array = audio_array.reshape(-1, channels)
            
            sd.play(audio_array, samplerate=sample_rate)
            sd.wait()
            return
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"sounddevice playback failed: {e}")
    
    # Fallback: system audio player
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        for player in AUDIO_PLAYERS:
            try:
                subprocess.run(
                    [player, tmp_path],
                    check=True,
                    capture_output=True,
                    timeout=AUDIO_PLAYBACK_TIMEOUT
                )
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                return
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        raise RuntimeError(f"No suitable audio player found. Audio saved to: {tmp_path}")
    except Exception as e:
        logger.error(f"Failed to play audio: {e}", exc_info=True)
        if tmp_path:
            logger.warning(f"Temporary audio file saved at: {tmp_path}")
        raise RuntimeError(f"Failed to play audio: {e}") from e
            
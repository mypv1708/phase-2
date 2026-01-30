"""
Simple TTS synthesizer wrapper for Piper TTS.
"""
from __future__ import annotations

import io
import logging
import wave
from piper.voice import PiperVoice
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


class TTSSynthesizer:
    """Text-to-Speech synthesizer using Piper TTS."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self._voice = None
        self._model_path = model_path

    def _ensure_loaded(self) -> None:
        """Lazy load the voice model."""
        if self._voice is not None:
            return

        if self._model_path is None:
            raise RuntimeError("Model path not set.")

        # Find config file (.json)
        config_path = Path(self._model_path).with_suffix(".json")
        if not config_path.exists():
            config_path = None

        try:
            self._voice = PiperVoice.load(
                model_path=str(self._model_path),
                config_path=str(config_path) if config_path else None,
                use_cuda=False,
            )
            logger.info("TTS loaded on device: CPU")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise

    def synthesize_to_bytes(self, text: str) -> bytes:

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        self._ensure_loaded()

        try:
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                self._voice.synthesize_wav(text, wav_file)

            wav_buffer.seek(0)
            return wav_buffer.read()
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise RuntimeError(f"Failed to synthesize audio: {e}") from e
import logging
from typing import Optional

import numpy as np

from .speech_recognition.audio_utils import prepare_audio_for_stt
from .speech_recognition.model_loader import load_stt_model
from .speech_recognition.transcription import TranscriptionEngine
from .speech_recognition.config import (
    STT_SAMPLE_RATE,
    STT_MIN_DURATION,
    STT_SILENT_THRESHOLD,
    STT_MAX_SILENT_RATIO,
)

logger = logging.getLogger(__name__)


def _is_mostly_silent(audio_np: np.ndarray, threshold: float, max_silent_ratio: float) -> bool:
    """Check if audio is mostly silent by calculating RMS energy frame-by-frame."""
    if audio_np.size == 0:
        return True
    
    if audio_np.ndim > 1:
        audio_np = audio_np.flatten()
    
    window_size = max(1, int(len(audio_np) / 100))
    silent_frames = 0
    total_frames = 0
    
    for i in range(0, len(audio_np), window_size):
        window = audio_np[i:i + window_size]
        if window.size > 0:
            window_rms = np.sqrt(np.mean(window ** 2))
            if window_rms < threshold:
                silent_frames += 1
            total_frames += 1
    
    if total_frames == 0:
        return True
    
    silent_ratio = silent_frames / total_frames
    return silent_ratio > max_silent_ratio


class SpeechRecognitionNode:
    def __init__(self, preload_models: bool = True):
        """Initialize SpeechRecognitionNode with all models preloaded."""
        logger.info("Initializing SpeechRecognitionNode...")
        self.transcription_engine: Optional[TranscriptionEngine] = None
        
        if preload_models:
            # Preload all models upfront
            from .speech_recognition.model_loader import preload_all_models
            
            _, stt_engine = preload_all_models()
        else:
            # Load models individually (legacy behavior)
            stt_engine = load_stt_model(preload=False)

        # Setup transcription engine
        try:
            if stt_engine is not None:
                self.transcription_engine = TranscriptionEngine(stt_engine)
            else:
                raise RuntimeError("STT engine is None after loading")
        except Exception as e:
            logger.error("Failed to setup STT engine", exc_info=True)
            raise RuntimeError(f"Failed to initialize SpeechRecognitionNode: {e}") from e
        
        logger.info("SpeechRecognitionNode initialized successfully")

    def process_audio(self, audio: np.ndarray, sample_rate: int) -> Optional[str]:
        try:
            audio_16k = prepare_audio_for_stt(audio, sample_rate)
        except Exception as e:
            logger.error("Failed to prepare audio: %s", e, exc_info=True)
            return None

        if self.transcription_engine is None:
            return None

        # Validate audio before processing
        audio_duration = len(audio_16k) / STT_SAMPLE_RATE
        
        if audio_duration < STT_MIN_DURATION:
            logger.debug(
                "Skipping STT: audio too short (%.2fs < %.2fs)",
                audio_duration,
                STT_MIN_DURATION,
            )
            return None
        
        if _is_mostly_silent(audio_16k, STT_SILENT_THRESHOLD, STT_MAX_SILENT_RATIO):
            logger.debug(
                "Skipping STT: audio mostly silent (duration: %.2fs)",
                audio_duration,
            )
            return None

        text = self.transcription_engine.transcribe(audio_16k, STT_SAMPLE_RATE)
        if text:
            logger.info("Transcription: %s", text)
        return text
import logging
from typing import Optional

import numpy as np

from .speech_recognition.audio_utils import prepare_audio_for_stt
from .speech_recognition.model_loader import load_speaker_model, load_stt_model
from .speech_recognition.speaker_verification import SpeakerVerifier
from .speech_recognition.transcription import TranscriptionEngine
from .speech_recognition.config import STT_SAMPLE_RATE

logger = logging.getLogger(__name__)


class SpeechRecognitionNode:
    def __init__(self, preload_models: bool = True):
        """Initialize SpeechRecognitionNode with all models preloaded."""
        logger.info("Initializing SpeechRecognitionNode...")
        self.speaker_verifier: Optional[SpeakerVerifier] = None
        self.transcription_engine: Optional[TranscriptionEngine] = None
        
        if preload_models:
            # Preload all models upfront
            from .speech_recognition.model_loader import preload_all_models
            
            speaker_model, stt_engine = preload_all_models()
        else:
            # Load models individually (legacy behavior)
            speaker_model = load_speaker_model()
            stt_engine = load_stt_model(preload=False)
        
        # Setup speaker verification
        if speaker_model is not None:
            enroll_wavs = SpeakerVerifier.load_enrollment()
            if enroll_wavs is not None:
                self.speaker_verifier = SpeakerVerifier(speaker_model, enroll_wavs)
                logger.info("Speaker verification enabled")
            else:
                logger.warning("Enrollment voice not loaded - speaker verification disabled")
        else:
            logger.warning("Speaker model not loaded - speaker verification disabled")

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

        if self.speaker_verifier is not None:
            if not self.speaker_verifier.verify(audio_16k, STT_SAMPLE_RATE):
                logger.info("Rejecting audio - skipping transcription")
                return None
        else:
            logger.debug("Speaker verification not available - proceeding with transcription")

        if self.transcription_engine is None:
            return None

        text = self.transcription_engine.transcribe(audio_16k, STT_SAMPLE_RATE)
        if text:
            logger.info("Transcription: %s", text)
        return text
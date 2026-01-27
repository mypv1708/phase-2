import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TranscriptionEngine:
    def __init__(self, stt_engine):
        self.stt_engine = stt_engine

    def transcribe(self, audio, sample_rate: int) -> Optional[str]:
        if self.stt_engine is None:
            return None

        try:
            return self.stt_engine.transcribe_audio(audio, sample_rate) or None
        except Exception as e:
            logger.exception("Transcription failed: %s", e)
            return None

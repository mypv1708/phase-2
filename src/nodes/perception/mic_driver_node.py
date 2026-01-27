import logging
import sys

import numpy as np
from eff_word_net.engine import HotwordDetector
from libdf import DF

from .mic_driver.model_loader import load_all_models
from .mic_driver.recording import run_recording_loop
from .mic_driver.wake_word import wait_for_wake_word
from .speech_recognition_node import SpeechRecognitionNode

logger = logging.getLogger(__name__)


class MicDriverNode:
    def __init__(self):
        """
        Initialize MicDriverNode with perception pipeline:
        - Wake word detection
        - Audio recording
        - Audio enhancement
        - Speaker verification
        - Speech-to-Text (STT)
        """
        logger.info("Initializing MicDriverNode...")
        try:
            # Load perception models (wake word, audio enhancement)
            (
                self.model,
                self.df_state,
                self.target_sr,
                self.device,
                self.wake_word_detector,
            ) = load_all_models()
            
            # Initialize speech recognition (STT + Speaker verification)
            self.speech_recognition = SpeechRecognitionNode()
            
            logger.info("MicDriverNode initialized successfully")
        except Exception as e:
            logger.exception("Failed to initialize MicDriverNode: %s", e)
            raise

    def run(self) -> None:
        logger.info("Starting mic driver main loop...")
        try:
            while True:
                logger.debug("Waiting for wake word...")
                detected = wait_for_wake_word(self.wake_word_detector)
                
                if not detected:
                    logger.debug("Wake word detection cancelled or failed")
                    continue
                
                logger.info("Wake word detected! Starting recording loop...")
                result = run_recording_loop(
                    model=self.model,
                    df_state=self.df_state,
                    target_sr=self.target_sr,
                    device=self.device,
                    on_utterance=self._on_utterance,
                )
                
                if result is not None:
                    audio, sr = result
                    logger.info(
                        "Recording completed: %d samples at %d Hz",
                        len(audio),
                        sr,
                    )
        except KeyboardInterrupt:
            logger.info("MicDriverNode interrupted by user")
        except Exception as e:
            logger.exception("MicDriverNode error: %s", e)
            raise

    def _on_utterance(self, audio: np.ndarray, sample_rate: int) -> bool:
        """
        Process audio utterance through perception pipeline:
        1. Speaker verification (if enabled)
        2. Speech-to-Text (STT)
        
        Args:
            audio: Enhanced audio data.
            sample_rate: Sample rate of audio.
            
        Returns:
            False to continue recording loop.
        """
        # Process audio: speaker verification + STT
        # Speaker verification is handled internally by SpeechRecognitionNode
        text = self.speech_recognition.process_audio(audio, sample_rate)
        
        if not text:
            logger.debug("No transcription received (speaker verification failed or no speech)")
            return False
        
        logger.info("Transcription completed: %s", text)
        return False

def main():
    from config.logging_config import setup_logging
    
    setup_logging()
    
    try:
        node = MicDriverNode()
        node.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
    
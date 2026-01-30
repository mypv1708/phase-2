import logging
import sys
import time
from pathlib import Path

from .mic_driver.model_loader import load_all_models
from .mic_driver.recording import run_recording_loop
from .mic_driver.recording_control import resume_recording
from .speech_recognition_node import SpeechRecognitionNode

# Import cognitive module for intent classification
# Need to ensure proper path for relative imports in cognitive module

# Add nodes directory to path for cognitive module imports
_nodes_dir = Path(__file__).resolve().parent.parent
if str(_nodes_dir) not in sys.path:
    sys.path.insert(0, str(_nodes_dir))

# Now import cognitive module
from cognitive.processor import process_input
from cognitive.config.settings import MIC_RESUME_DELAY

logger = logging.getLogger(__name__)


class MicDriverNode:
    def __init__(self):
        """
        Initialize MicDriverNode with complete perception pipeline:
        - Audio recording (VAD)
        - Audio enhancement (DeepFilterNet)
        - Speech-to-Text (STT - PhoWhisper)
        - Intent Classification (Cognitive module)
        - Intent Routing & Response (TTS)
        """
        logger.info("Initializing MicDriverNode...")
        try:
            # Load perception models (audio enhancement)
            self.model, self.df_state, self.target_sr = load_all_models()
            
            # Initialize speech recognition (STT)
            self.speech_recognition = SpeechRecognitionNode()
            
            logger.info("MicDriverNode initialized successfully")
        except Exception as e:
            logger.exception("Failed to initialize MicDriverNode: %s", e)
            raise

    def run(self) -> None:
        logger.info("Starting mic driver main loop...")
        try:
            while True:
                result = run_recording_loop(
                    model=self.model,
                    df_state=self.df_state,
                    target_sr=self.target_sr,
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

    def _on_utterance(self, audio, sample_rate: int) -> bool:
        """
        Process audio utterance through perception pipeline:
        1. Speech-to-Text (STT)
        2. Intent Classification (Cognitive)
        3. Intent Routing & Response
        4. Resume recording
        
        Note: Recording is already paused when this callback is invoked.
        We resume recording after all processing and logging is complete.
        
        Args:
            audio: Enhanced audio data.
            sample_rate: Sample rate of audio.
            
        Returns:
            False to continue recording loop.
        """
        try:
            # Step 1: Speech-to-Text
            text = self.speech_recognition.process_audio(audio, sample_rate)
            
            if not text:
                logger.debug("No transcription received (no speech detected)")
                return False
                                
            # Step 2: Intent Classification & Routing (includes TTS playback)
            try:
                result = process_input(text, use_tts=True)
                
                # Log result AFTER TTS playback is done
                logger.info(
                    "Intent: %s (confidence: %.2f, route: %s)",
                    result.get("intent", "unknown"),
                    result.get("confidence", 0.0),
                    result.get("route", "unknown"),
                )
                
                # Log response
                if result.get("response"):
                    logger.info("Response: %s", result.get("response"))
                
                # Log command if available
                if result.get("command"):
                    logger.info("Command: %s", result.get("command").strip())
                
            except Exception as e:
                logger.exception("Failed to process intent: %s", e)
            
            return False
            
        finally:
            # Always resume recording after all processing is done
            time.sleep(MIC_RESUME_DELAY)
            resume_recording()
            logger.info("Mic recording resumed")

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
    
""" Model loader for speech recognition - loads all models upfront. """
import logging
import time
from typing import Tuple

import torch

from .config import STT_DEVICE

from .speech_to_text import SpeechToTextEngine
    

logger = logging.getLogger(__name__)


def load_stt_model(preload: bool = True) -> "SpeechToTextEngine":
    """Load Speech-to-Text model."""
    logger.info("Initializing Speech-to-Text engine...")
    if STT_DEVICE:
        device = STT_DEVICE
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        from .speech_to_text import SpeechToTextEngine

        engine = SpeechToTextEngine(device)
        if preload:
            logger.info("Preloading STT model to avoid first-use delay...")
            start_time = time.perf_counter()
            engine.preload()
            load_time = time.perf_counter() - start_time
            logger.info(
                "Speech-to-Text engine ready (preloaded on %s in %.2f seconds)",
                engine.device_type,
                load_time,
            )
        else:
            logger.info(
                "Speech-to-Text engine ready (lazy loading on %s)", engine.device_type
            )
        return engine
    except Exception as e:
        logger.exception("Failed to initialize STT engine: %s", e)
        raise RuntimeError(f"STT engine initialization failed: {e}") from e


def preload_all_models() -> Tuple[None, "SpeechToTextEngine"]:
    """Preload all speech recognition models upfront."""
    logger.info("=" * 60)
    logger.info("Preloading Speech Recognition models...")
    logger.info("=" * 60)
    
    total_start = time.perf_counter()
    
    # Load STT model (required, will raise if fails)
    stt_engine = load_stt_model(preload=True)
    
    total_time = time.perf_counter() - total_start
    
    logger.info("=" * 60)
    logger.info(
        "Speech Recognition models loaded in %.2f seconds", total_time
    )
    logger.info("=" * 60)
    
    return None, stt_engine
""" Model loader for speech recognition - loads all models upfront. """
import logging
import time
from typing import Optional, Tuple

import torch
from speechbrain.inference.speaker import SpeakerRecognition

from .config import SPEAKER_DEVICE, STT_DEVICE

from .speech_to_text import SpeechToTextEngine
    

logger = logging.getLogger(__name__)


def load_speaker_model() -> Optional[SpeakerRecognition]:
    """Load speaker recognition model."""
    if SPEAKER_DEVICE:
        device = SPEAKER_DEVICE
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        logger.info("Loading Speaker Recognition model on %s...", device)
        start_time = time.perf_counter()
        
        model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        
        # Force move model to GPU if CUDA is available and device is "cuda"
        if device == "cuda" and torch.cuda.is_available():
            try:
                # SpeechBrain models are typically nn.Module, use .to() method
                if hasattr(model, "to"):
                    model = model.to(torch.device("cuda"))
                # Verify model is on GPU by checking parameters
                if hasattr(model, "parameters"):
                    first_param_device = next(model.parameters()).device
                    if first_param_device.type != "cuda":
                        logger.warning(
                            "Model parameters not on GPU after .to() call. "
                            "SpeechBrain may handle device internally."
                        )
                    else:
                        logger.info("Model successfully moved to GPU")
            except Exception as e:
                logger.warning(
                    "Failed to explicitly move model to GPU: %s. "
                    "Model may still work but might be slower.",
                    e,
                )
        
        load_time = time.perf_counter() - start_time
        
        # Verify actual device
        actual_device = "cpu"
        try:
            if hasattr(model, "device"):
                device_raw = model.device
                if isinstance(device_raw, str):
                    actual_device = device_raw
                else:
                    actual_device = device_raw.type
            elif hasattr(model, "parameters"):
                actual_device = next(model.parameters()).device.type
        except Exception:
            pass
        
        logger.info(
            "Speaker Recognition model loaded on %s (requested: %s) in %.2f seconds",
            actual_device,
            device,
            load_time,
        )
        return model
    except Exception as e:
        logger.warning(
            "Failed to load Speaker Recognition model "
            "(speaker verification disabled): %s",
            e,
        )
        return None


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
                device,
                load_time,
            )
        else:
            logger.info(
                "Speech-to-Text engine ready (lazy loading on %s)", device
            )
        return engine
    except Exception as e:
        logger.exception("Failed to initialize STT engine: %s", e)
        raise RuntimeError(f"STT engine initialization failed: {e}") from e


def preload_all_models() -> Tuple[Optional[SpeakerRecognition], "SpeechToTextEngine"]:
    """Preload all speech recognition models upfront."""
    logger.info("=" * 60)
    logger.info("Preloading all Speech Recognition models...")
    logger.info("=" * 60)
    
    total_start = time.perf_counter()
    
    # Load speaker model
    speaker_model = load_speaker_model()
    
    # Load STT model (required, will raise if fails)
    stt_engine = load_stt_model(preload=True)
    
    total_time = time.perf_counter() - total_start
    
    logger.info("=" * 60)
    logger.info(
        "All Speech Recognition models loaded in %.2f seconds", total_time
    )
    logger.info("=" * 60)
    
    return speaker_model, stt_engine
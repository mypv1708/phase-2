import logging
import os
from typing import Tuple

import torch
from df.enhance import init_df
from libdf import DF

from .config import (
    DF_LOG_LEVEL,
    DF_POST_FILTER,
)

logger = logging.getLogger(__name__)


def load_deepfilternet() -> Tuple[torch.nn.Module, DF, int, str]:
    """Load DeepFilterNet model for noise reduction with CPU fallback."""
    logger.info("Loading DeepFilterNet...")
    
    # Try CUDA first, fallback to CPU if CUDA fails
    device_preference = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model, df_state, _ = init_df(
            model_base_dir=None,
            post_filter=DF_POST_FILTER,
            log_level=DF_LOG_LEVEL,
        )
        
        target_sr = df_state.sr()
        device = next(model.parameters()).device
        device_str = "GPU" if device.type == "cuda" else "CPU"
        
        # Verify device matches expectation
        if device_preference == "cuda" and device.type == "cpu":
            logger.warning(
                "CUDA requested but model loaded on CPU. "
                "Falling back to CPU mode."
            )
        elif device_preference == "cpu" and device.type == "cuda":
            logger.warning(
                "CPU requested but model loaded on CUDA. "
                "Using CUDA mode."
            )

        logger.info(
            "DeepFilterNet loaded successfully: %d Hz, device=%s",
            target_sr,
            device_str,
        )
        
        return model, df_state, target_sr, device_str
    except RuntimeError as e:
        if "cuda" in str(e).lower() and device_preference == "cuda":
            logger.warning(
                "CUDA initialization failed, attempting CPU fallback: %s", e
            )
            # Force CPU mode by setting environment variable
            import os
            original_device = os.environ.get("CUDA_VISIBLE_DEVICES")
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                model, df_state, _ = init_df(
                    model_base_dir=None,
                    post_filter=DF_POST_FILTER,
                    log_level=DF_LOG_LEVEL,
                )
                target_sr = df_state.sr()
                device = next(model.parameters()).device
                device_str = "GPU" if device.type == "cuda" else "CPU"
                logger.info(
                    "DeepFilterNet loaded on CPU fallback: %d Hz, device=%s",
                    target_sr,
                    device_str,
                )
                return model, df_state, target_sr, device_str
            finally:
                if original_device is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = original_device
                elif "CUDA_VISIBLE_DEVICES" in os.environ:
                    del os.environ["CUDA_VISIBLE_DEVICES"]
        raise RuntimeError(f"DeepFilterNet initialization failed: {e}") from e
    except Exception as e:
        logger.exception("Failed to load DeepFilterNet: %s", e)
        raise RuntimeError(f"DeepFilterNet initialization failed: {e}") from e


def load_all_models() -> Tuple[torch.nn.Module, DF, int, str]:
    """Load all models (DeepFilterNet)."""
    logger.info("Loading all models...")
    model, df_state, target_sr, device = load_deepfilternet()
    logger.info("All models loaded successfully")
    return model, df_state, target_sr, device
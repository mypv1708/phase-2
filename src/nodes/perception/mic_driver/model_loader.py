import logging
from typing import Tuple

import torch
from df.enhance import init_df
from libdf import DF

from .config import DF_LOG_LEVEL, DF_POST_FILTER

logger = logging.getLogger(__name__)


def load_deepfilternet() -> Tuple[torch.nn.Module, DF, int]:
    """Load DeepFilterNet model for noise reduction."""
    logger.info("Loading DeepFilterNet...")
    
    try:
        model, df_state, _ = init_df(
            model_base_dir=None,
            post_filter=DF_POST_FILTER,
            log_level=DF_LOG_LEVEL,
        )
        
        target_sr = df_state.sr()
        device = next(model.parameters()).device
        device_str = "GPU" if device.type == "cuda" else "CPU"

        logger.info(
            "DeepFilterNet loaded: %d Hz, device=%s",
            target_sr,
            device_str,
        )
        
        return model, df_state, target_sr
    except Exception as e:
        logger.exception("Failed to load DeepFilterNet: %s", e)
        raise RuntimeError(f"DeepFilterNet initialization failed: {e}") from e


def load_all_models() -> Tuple[torch.nn.Module, DF, int]:
    """Load all models (DeepFilterNet)."""
    logger.info("Loading all models...")
    model, df_state, target_sr = load_deepfilternet()
    logger.info("All models loaded successfully")
    return model, df_state, target_sr
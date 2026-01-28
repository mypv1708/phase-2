import logging
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from df.enhance import enhance

from .audio import save_wave
from .config import (
    FRAME_DURATION_MS,
    INT16_MAX,
    MILLISECONDS_PER_SECOND,
    MIN_ENHANCE_DURATION,
    MAX_SILENT_RATIO,
    SILENT_THRESHOLD,
)

from libdf import DF

FRAME_DURATION_SEC = FRAME_DURATION_MS / MILLISECONDS_PER_SECOND

logger = logging.getLogger(__name__)

_INT16_MAX_INV = 1.0 / INT16_MAX


def is_mostly_silent(audio_np: np.ndarray, threshold: float = SILENT_THRESHOLD, max_silent_ratio: float = MAX_SILENT_RATIO) -> bool:
    """
    Check if audio is mostly silent by calculating RMS energy frame-by-frame.
    
    Args:
        audio_np: Audio array (normalized float32, -1.0 to 1.0)
        threshold: RMS threshold for silent detection
        max_silent_ratio: Maximum ratio of silent frames (default: MAX_SILENT_RATIO)
        
    Returns:
        True if audio is mostly silent (> max_silent_ratio)
    """
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


def convert_frames_to_tensor(recorded_frames: List[bytes]) -> torch.Tensor:
    """Convert PCM frames to normalized float32 tensor [1, T]."""
    if not recorded_frames:
        raise ValueError("recorded_frames cannot be empty")
    pcm = b"".join(recorded_frames)
    audio_np = (
        np.frombuffer(pcm, dtype=np.int16)
        .astype(np.float32, copy=False)
        * _INT16_MAX_INV
    )
    return torch.from_numpy(audio_np).unsqueeze(0)


def enhance_utterance(
    recorded_frames: List[bytes],
    model: torch.nn.Module,
    df_state: "DF",
    target_sr: int,
    raw_filepath: Optional[str] = None,
    enhanced_filepath: Optional[str] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[np.ndarray, int, Optional[str], Optional[str]]:
    if not recorded_frames:
        raise ValueError("recorded_frames cannot be empty")

    # if raw_filepath:
    #     try:
    #         save_wave(recorded_frames, raw_filepath)
    #         logger.info("Saved raw audio: %s", raw_filepath)
    #     except Exception:
    #         logger.exception("Failed to save raw audio")

    audio_tensor = convert_frames_to_tensor(recorded_frames)
    audio_tensor = audio_tensor.cpu()

    num_samples = (
        audio_tensor.shape[-1] if len(audio_tensor.shape) > 0 else 0
    )
    if num_samples > 0 and target_sr > 0:
        audio_duration = num_samples / target_sr
    else:
        audio_duration = len(recorded_frames) * FRAME_DURATION_SEC

    audio_np = None
    should_skip = False
    skip_reason = None
    
    if audio_duration < MIN_ENHANCE_DURATION:
        should_skip = True
        skip_reason = f"duration too short ({audio_duration:.2f}s < {MIN_ENHANCE_DURATION}s)"
    else:
        audio_np = audio_tensor.squeeze(0).cpu().numpy() if audio_tensor.ndim > 1 else audio_tensor.cpu().numpy()
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32, copy=False)
        
        if is_mostly_silent(audio_np, SILENT_THRESHOLD):
            should_skip = True
            skip_reason = "mostly silent"
    
    if should_skip:
        logger.info(
            "Skipping enhancement: %s (duration: %.2fs)",
            skip_reason,
            audio_duration,
        )
        if audio_np is None:
            audio_np = audio_tensor.squeeze(0).cpu().numpy() if audio_tensor.ndim > 1 else audio_tensor.cpu().numpy()
            if audio_np.dtype != np.float32:
                audio_np = audio_np.astype(np.float32, copy=False)
        
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()
        enhanced_np = audio_np
    else:
        inference_start = time.perf_counter()
        try:
            enhanced = enhance(model, df_state, audio_tensor)
        except Exception as e:
            logger.exception("Audio enhancement failed: %s", e)
            raise RuntimeError(f"Audio enhancement failed: {e}") from e

        inference_time = time.perf_counter() - inference_start

        try:
            model_device = next(model.parameters()).device
            device_str = "GPU" if model_device.type == "cuda" else "CPU"
        except Exception:
            device_str = "CPU"

        if logger.isEnabledFor(logging.INFO):
            speed_ratio = (
                audio_duration / inference_time if inference_time > 0 else 0.0
            )
            logger.info(
                "DeepFilterNet inference: %.3fs (audio: %.2fs, speed: %.2fx real-time, device=%s)",
                inference_time,
                audio_duration,
                speed_ratio,
                device_str,
            )

        if isinstance(enhanced, torch.Tensor):
            enhanced_np = enhanced.cpu().numpy()
            if enhanced_np.ndim == 2 and enhanced_np.shape[0] == 1:
                enhanced_np = enhanced_np.squeeze(0)
            if enhanced_np.dtype != np.float32:
                enhanced_np = enhanced_np.astype(np.float32, copy=False)
        else:
            enhanced_np = np.asarray(enhanced, dtype=np.float32)

        if enhanced_np.size == 0:
            raise RuntimeError("Enhanced audio is empty")
        if np.any(~np.isfinite(enhanced_np)):
            logger.warning(
                "Enhanced audio contains non-finite values, clipping"
            )
            enhanced_np = np.clip(enhanced_np, -1.0, 1.0)

        try:
            model_device = next(model.parameters()).device
            if model_device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    # if enhanced_filepath:
    #     try:
    #         sf.write(enhanced_filepath, enhanced_np, target_sr)
    #         logger.info("Saved enhanced audio: %s", enhanced_filepath)
    #     except Exception:
    #         logger.exception("Failed to save enhanced audio")

    return enhanced_np, target_sr, raw_filepath, enhanced_filepath

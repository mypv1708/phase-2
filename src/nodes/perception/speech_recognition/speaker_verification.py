import logging
import time
from typing import Optional

import numpy as np
import torch
import torchaudio

from .config import ENROLL_FILE, SPEAKER_THRESHOLD, STT_SAMPLE_RATE

logger = logging.getLogger(__name__)


class SpeakerVerifier:
    def __init__(self, speaker_model, enroll_wavs: Optional[torch.Tensor] = None):
        self.speaker_model = speaker_model
        self.threshold = SPEAKER_THRESHOLD
        
        # Get model device and move enrollment audio to same device
        if enroll_wavs is not None:
            try:
                if hasattr(speaker_model, "device"):
                    model_device_raw = speaker_model.device
                    # Handle both string and torch.device types
                    if isinstance(model_device_raw, str):
                        model_device = torch.device(model_device_raw)
                    else:
                        model_device = model_device_raw
                else:
                    model_device = next(speaker_model.parameters()).device
                
                # Move enrollment audio to model device
                self.enroll_wavs = enroll_wavs.to(model_device)
                device_str = "GPU" if model_device.type == "cuda" else "CPU"
                logger.info(
                    "Enrollment audio moved to %s for speaker verification",
                    device_str,
                )
            except Exception as e:
                logger.warning(
                    "Failed to move enrollment audio to model device, "
                    "using CPU: %s",
                    e,
                )
                self.enroll_wavs = enroll_wavs
        else:
            self.enroll_wavs = None

    @classmethod
    def load_enrollment(cls, enroll_file: str = ENROLL_FILE) -> Optional[torch.Tensor]:
        try:
            wavs, sr = torchaudio.load(enroll_file)
            if wavs.shape[0] > 1:
                wavs = wavs.mean(dim=0, keepdim=True)
            if sr != STT_SAMPLE_RATE:
                wavs = torchaudio.functional.resample(wavs, sr, STT_SAMPLE_RATE)
            return wavs
        except Exception as e:
            logger.exception("Failed to load enrollment voice from %s", enroll_file)
            return None

    def verify(self, audio: np.ndarray, sample_rate: int) -> bool:
        if self.speaker_model is None or self.enroll_wavs is None:
            return False

        try:
            audio_tensor = torch.from_numpy(
                audio.astype(np.float32, copy=False)
            )
            if sample_rate != STT_SAMPLE_RATE:
                audio_tensor = audio_tensor.unsqueeze(0)
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor, sample_rate, STT_SAMPLE_RATE
                )
                audio_tensor = audio_tensor.squeeze(0)
            
            test_wavs = audio_tensor.unsqueeze(0)
            
            # Get model device and move test audio to same device
            try:
                if hasattr(self.speaker_model, "device"):
                    model_device_raw = self.speaker_model.device
                    # Handle both string and torch.device types
                    if isinstance(model_device_raw, str):
                        model_device = torch.device(model_device_raw)
                    else:
                        model_device = model_device_raw
                else:
                    model_device = next(self.speaker_model.parameters()).device
                
                test_wavs = test_wavs.to(model_device)
            except Exception as e:
                logger.warning(
                    "Failed to move test audio to model device: %s", e
                )
            
            inference_start = time.perf_counter()
            score, prediction = self.speaker_model.verify_batch(
                self.enroll_wavs, 
                test_wavs, 
                threshold=self.threshold,
            )
            inference_time = time.perf_counter() - inference_start
            
            # Extract score and prediction values
            if hasattr(score, "__len__") and len(score) > 0:
                score_val = float(score[0].item())
            else:
                score_val = float(score)
            if hasattr(prediction, "__len__") and len(prediction) > 0:
                is_match = bool(prediction[0].item())
            else:
                is_match = bool(prediction)
            
            audio_duration = len(audio) / sample_rate
            speed_ratio = (
                audio_duration / inference_time if inference_time > 0 else 0.0
            )

            # Get device info for logging
            try:
                if hasattr(self.speaker_model, "device"):
                    model_device_raw = self.speaker_model.device
                    # Handle both string and torch.device types
                    if isinstance(model_device_raw, str):
                        model_device = torch.device(model_device_raw)
                    else:
                        model_device = model_device_raw
                else:
                    model_device = next(self.speaker_model.parameters()).device
                device_str = "GPU" if model_device.type == "cuda" else "CPU"
            except Exception:
                device_str = "CPU"  # Default to CPU if can't determine
            
            if is_match:
                if logger.isEnabledFor(logging.INFO):
                    logger.info(
                        "Speaker verification PASSED: score=%.4f "
                        "(threshold=%.4f, inference=%.3fs, audio=%.2fs, "
                        "speed=%.2fx, device=%s)",
                        score_val,
                        self.threshold,
                        inference_time,
                        audio_duration,
                        speed_ratio,
                        device_str,
                    )
            else:
                logger.warning(
                    "Speaker verification FAILED: score=%.4f "
                    "(threshold=%.4f, inference=%.3fs, audio=%.2fs, "
                    "speed=%.2fx, device=%s)",
                    score_val,
                    self.threshold,
                    inference_time,
                    audio_duration,
                    speed_ratio,
                    device_str,
                )
            
            return is_match
            
        except Exception:
            logger.exception("Speaker verification error")
            return False
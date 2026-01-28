import logging
import time
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .config import (
    STT_LANGUAGE,
    STT_MAX_NEW_TOKENS,
    STT_MODEL_ID,
    STT_NUM_BEAMS,
    STT_SAMPLE_RATE,
    STT_TASK,
    STT_USE_FP16,
)

logger = logging.getLogger(__name__)


class SpeechToTextEngine:
    def __init__(self, device: str):
        self.device = torch.device(device)
        self.model: Optional[AutoModelForSpeechSeq2Seq] = None
        self.processor: Optional[AutoProcessor] = None
        self._initialized = False
        self._language = None if (STT_LANGUAGE or "").lower() == "auto" else STT_LANGUAGE
        self._use_fp16 = STT_USE_FP16 and self.device.type == "cuda"
        # Device string for logging
        self.device_type = "GPU" if self.device.type == "cuda" else "CPU"

    def _initialize(self) -> None:
        if self._initialized:
            return

        # Support both PhoWhisper (full model ID) and Whisper (prefix with openai/)
        if STT_MODEL_ID.startswith("vinai/") or STT_MODEL_ID.startswith("openai/"):
            model_name = STT_MODEL_ID
        else:
            # Legacy format: "base" -> "openai/whisper-base"
            model_name = f"openai/whisper-{STT_MODEL_ID}"
        
        logger.info(
            "Loading PhoWhisper/Whisper model '%s' (device=%s, fp16=%s)...",
            model_name,
            self.device_type,
            self._use_fp16,
        )
        start_time = time.perf_counter()
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(self.device)
            self.model.eval()
            
            if hasattr(torch, "compile") and self.device.type == "cuda":
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.debug("Model compiled with torch.compile")
                except Exception:
                    pass
            
            load_time = time.perf_counter() - start_time
            model_type = "PhoWhisper" if "PhoWhisper" in model_name else "Whisper"
            logger.info(
                "%s model loaded on %s in %.2f seconds",
                model_type,
                self.device_type,
                load_time,
            )
            self._initialized = True
        except Exception as e:
            logger.exception("Failed to load STT model")
            raise RuntimeError(f"STT model initialization failed: {e}") from e

    def preload(self) -> None:
        """Preload the model immediately to avoid first-use delay."""
        if not self._initialized:
            self._initialize()
        
        # Warm up the model with a dummy inference to ensure it's fully loaded
        if self._initialized and self.model is not None:
            try:
                # Create a more realistic dummy audio input for warm-up (white noise)
                # This helps prevent hallucination on first real input
                dummy_audio = np.random.normal(0, 0.01, (STT_SAMPLE_RATE * 2,)).astype(np.float32)  # 2 seconds
                with torch.inference_mode():
                    inputs = self.processor(
                        dummy_audio,
                        sampling_rate=STT_SAMPLE_RATE,
                        return_tensors="pt",
                        padding=True
                    )
                    input_features = inputs["input_features"].to(self.device)
                    # Run a forward pass to warm up
                    _ = self.model.generate(
                        input_features,
                        language=self._language,
                        task=STT_TASK,
                        num_beams=1,
                        max_new_tokens=10,
                    )
                logger.debug("STT model warmed up successfully")
            except Exception as e:
                logger.debug(f"STT model warm-up failed (non-critical): {e}")

    def transcribe_audio(self, audio: np.ndarray, sr: int) -> str:
        if audio is None or audio.size == 0:
            raise ValueError("audio is empty")
        if sr <= 0:
            raise ValueError(f"Invalid sample rate: {sr}")
        if self.model is None:
            raise RuntimeError("STT model not initialized")

        total_start = time.perf_counter()
        
        audio = self._preprocess_audio(audio, sr)
        audio_duration = len(audio) / STT_SAMPLE_RATE
        text = self._run_inference(audio)
        text = self._postprocess_segments(text)
        
        total_time = time.perf_counter() - total_start
        
        if logger.isEnabledFor(logging.INFO):
            speed_ratio = audio_duration / total_time if total_time > 0 else 0.0
            logger.info(
                "STT total: %.3fs (audio: %.2fs, speed: %.2fx real-time, device=%s)",
                total_time,
                audio_duration,
                speed_ratio,
                self.device_type,
            )
        
        if not text and logger.isEnabledFor(logging.WARNING):
            logger.warning("STT returned empty result (no speech detected or filtered out)")
        
        return text

    def _preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1 if audio.shape[1] <= 8 else 0)
        
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)

        if sr != STT_SAMPLE_RATE:
            raise ValueError(
                f"Expected sample_rate={STT_SAMPLE_RATE}, got {sr}. "
                "Audio should be resampled before calling transcribe_audio."
            )

        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
        
        return audio

    def _run_inference(self, audio: np.ndarray) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("STT model not initialized")
        
        try:
            with torch.inference_mode():
                inputs = self.processor(
                    audio,
                    sampling_rate=STT_SAMPLE_RATE,
                    return_tensors="pt",
                    padding=True
                )
                
                input_features = inputs["input_features"].to(self.device)
                
                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=self._use_fp16
                ):
                    generated_ids = self.model.generate(
                        input_features,
                        language=self._language,
                        task=STT_TASK,
                        num_beams=STT_NUM_BEAMS,
                        max_new_tokens=STT_MAX_NEW_TOKENS,
                    )
                
                transcription = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
                
                return transcription
        except Exception as e:
            logger.exception("STT model inference failed: %s", e)
            raise RuntimeError(f"STT transcription failed: {e}") from e

    def _postprocess_segments(self, text: str) -> str:
        return text.strip()
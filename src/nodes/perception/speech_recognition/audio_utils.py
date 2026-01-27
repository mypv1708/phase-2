import numpy as np
import torch
import torchaudio

from .config import STT_SAMPLE_RATE


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio is None or getattr(audio, "size", 0) == 0:
        raise ValueError("audio is empty")
    
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    
    if audio.ndim == 2:
        if audio.shape[1] <= 8:
            return audio.mean(axis=1).astype(np.float32, copy=False)
        return audio.mean(axis=0).astype(np.float32, copy=False)
    
    raise ValueError(f"Unsupported audio shape: {audio.shape}")


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = STT_SAMPLE_RATE) -> np.ndarray:
    if orig_sr == target_sr:
        return audio if audio.dtype == np.float32 else audio.astype(np.float32, copy=False)
    
    wav = torch.from_numpy(audio).float().unsqueeze(0)
    wav = torchaudio.functional.resample(wav, orig_sr, target_sr)
    return wav.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def prepare_audio_for_stt(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio_mono = to_mono_float32(audio)
    return resample_audio(audio_mono, sample_rate, STT_SAMPLE_RATE)

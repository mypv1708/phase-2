import os
import time
from typing import Tuple

from .config import (
    AUDIO_BASE_DIR,
    RAW_SUBDIR,
    PROCESSED_SUBDIR,
    TIMESTAMP_FMT,
    RAW_FILE_PREFIX,
    ENHANCED_FILE_PREFIX,
    MILLISECONDS_PER_SECOND, 
)


def ensure_output_paths(today: str) -> Tuple[str, str]:
    raw_folder = os.path.join(AUDIO_BASE_DIR, RAW_SUBDIR, today)
    proc_folder = os.path.join(AUDIO_BASE_DIR, PROCESSED_SUBDIR, today)
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(proc_folder, exist_ok=True)
    return raw_folder, proc_folder


def build_filenames(today: str) -> Tuple[str, str]:
    timestamp = time.strftime(TIMESTAMP_FMT)
    milliseconds = int((time.time() % 1) * MILLISECONDS_PER_SECOND)
    timestamp_name = f"{timestamp}_{milliseconds:03d}"
    raw_file = os.path.join(AUDIO_BASE_DIR, RAW_SUBDIR, today, f"{RAW_FILE_PREFIX}{timestamp_name}.wav")
    enh_file = os.path.join(AUDIO_BASE_DIR, PROCESSED_SUBDIR, today, f"{ENHANCED_FILE_PREFIX}{timestamp_name}.wav")
    return raw_file, enh_file
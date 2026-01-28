import logging
import time
from collections import deque
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch


from libdf import DF

from .config import (
    RATE,
    DATE_FMT,
    FRAME_SIZE,
    MAX_RECORDING_SECONDS,
    PRE_BUFFER_FRAMES,
    SILENCE_EXIT,
    SILENCE_LIMIT,
    SAMPLE_WIDTH,
)
from .audio import init_audio_stream
from .enhance import enhance_utterance
from .paths import build_filenames, ensure_output_paths

logger = logging.getLogger(__name__)

EXPECTED_FRAME_SIZE = FRAME_SIZE * SAMPLE_WIDTH
FRAMES_PER_SECOND = RATE / FRAME_SIZE
SILENCE_FRAMES_THRESHOLD = int(SILENCE_LIMIT * FRAMES_PER_SECOND)


def run_recording_loop(
    model: torch.nn.Module,
    df_state: "DF",
    target_sr: int,
    device: str,
    on_utterance: Optional[Callable[[np.ndarray, int], bool]] = None,
) -> Optional[Tuple[np.ndarray, int]]:
    pa = None
    vad = None
    stream = None

    try:
        pa, vad, stream = init_audio_stream()
    except Exception as e:
        logger.exception("Failed to initialize audio stream: %s", e)
        raise

    # Buffer audio before speech starts (pre-buffer)
    pre_buffer: deque = deque(maxlen=PRE_BUFFER_FRAMES)
    recorded_frames: List[bytes] = []
    recording = False
    speech_end_time = None
    recording_start_time = None
    last_result: Optional[Tuple[np.ndarray, int]] = None
    stop_requested = False
    consecutive_silence_frames = 0

    def _reset_recording_state() -> None:
        nonlocal pre_buffer, recorded_frames, recording
        nonlocal recording_start_time, consecutive_silence_frames
        pre_buffer.clear()
        recorded_frames.clear()
        recording = False
        recording_start_time = None
        consecutive_silence_frames = 0

    def _process_and_reset() -> None:
        nonlocal speech_end_time, last_result, stop_requested
        speech_end_time = time.time()
        try:
            if not recorded_frames:
                logger.warning("No frames recorded, skipping enhancement")
                _reset_recording_state()
                return

            today = time.strftime(DATE_FMT)
            ensure_output_paths(today)
            raw_file, enh_file = build_filenames(today)

            enhanced_audio, enhanced_sr, raw_file, enh_file = (
                enhance_utterance(
                    recorded_frames,
                    model,
                    df_state,
                    target_sr,
                    raw_file,
                    enh_file,
                    device=device,
                )
            )
            last_result = (enhanced_audio, enhanced_sr)
            if on_utterance is not None:
                if on_utterance(enhanced_audio, enhanced_sr):
                    stop_requested = True
        except Exception as e:
            logger.exception("Failed to enhance/save audio: %s", e)
        finally:
            _reset_recording_state()

    try:
        while True:
            frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
            if len(frame) != EXPECTED_FRAME_SIZE:
                continue

            current_time = time.time()

            is_speech = vad.is_speech(frame, RATE)

            if not recording:
                # Buffer audio before speech starts
                pre_buffer.append(frame)

            if is_speech:
                consecutive_silence_frames = 0
                if not recording:
                    recording = True
                    recording_start_time = current_time
                    # Include pre-buffered audio
                    recorded_frames.extend(pre_buffer)
                    speech_end_time = None
                    logger.debug("Recording started")

                if (
                    recording_start_time
                    and (current_time - recording_start_time)
                    > MAX_RECORDING_SECONDS
                ):
                    logger.warning(
                        ">> Max recording duration (%ss) reached, "
                        "forcing save",
                        MAX_RECORDING_SECONDS,
                    )
                    _process_and_reset()
                    if stop_requested:
                        return last_result
                    continue

                recorded_frames.append(frame)
            else:
                if recording:
                    consecutive_silence_frames += 1
                    if consecutive_silence_frames >= SILENCE_FRAMES_THRESHOLD:
                        recording_duration = (
                            current_time - recording_start_time
                            if recording_start_time
                            else 0
                        )
                        logger.info(
                            "Silence detected (%.1fs), stopping recording "
                            "(duration: %.2fs)",
                            SILENCE_LIMIT,
                            recording_duration,
                        )
                        _process_and_reset()
                        if stop_requested:
                            return last_result

            if not recording and speech_end_time is not None:
                # No speech for SILENCE_EXIT seconds: exit
                if current_time - speech_end_time > SILENCE_EXIT:
                    logger.info("Silence timeout, exiting recording loop")
                    return None

    except KeyboardInterrupt:
        return None
    except Exception as e:
        logger.exception("Recording loop error: %s", e)
        raise
    finally:
        if stream is not None:
            try:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
            except Exception as e:
                logger.warning("Error closing audio stream: %s", e)
        if pa is not None:
            try:
                pa.terminate()
            except Exception as e:
                logger.warning("Error terminating PyAudio: %s", e)


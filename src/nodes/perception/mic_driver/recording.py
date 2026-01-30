import logging
import time
from collections import deque
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from libdf import DF

from .config import (
    RATE,
    FRAME_SIZE,
    MAX_RECORDING_SECONDS,
    PRE_BUFFER_FRAMES,
    SILENCE_EXIT,
    SILENCE_LIMIT,
    SAMPLE_WIDTH,
    POST_RESUME_IGNORE_MS,
)
from .audio import init_audio_stream
from .enhance import enhance_utterance
from .recording_control import is_recording_paused, should_clear_buffer, pause_recording

logger = logging.getLogger(__name__)

EXPECTED_FRAME_SIZE = FRAME_SIZE * SAMPLE_WIDTH
FRAMES_PER_SECOND = RATE / FRAME_SIZE
SILENCE_FRAMES_THRESHOLD = int(SILENCE_LIMIT * FRAMES_PER_SECOND)
POST_RESUME_IGNORE_FRAMES = int(POST_RESUME_IGNORE_MS / 1000 * FRAMES_PER_SECOND)


def run_recording_loop(
    model: torch.nn.Module,
    df_state: "DF",
    target_sr: int,
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
        nonlocal recording, recording_start_time, consecutive_silence_frames
        pre_buffer.clear()
        recorded_frames.clear()
        recording = False
        recording_start_time = None
        consecutive_silence_frames = 0

    def _process_and_reset() -> None:
        nonlocal speech_end_time, last_result, stop_requested
        speech_end_time = time.time()
        
        # CRITICAL: Stop stream IMMEDIATELY to prevent capturing TTS audio
        # This must happen BEFORE any processing/callback
        pause_recording()
        if stream.is_active():
            try:
                stream.stop_stream()
                logger.debug("Audio stream stopped for processing")
            except Exception as e:
                logger.debug("Failed to stop stream: %s", e)
        
        try:
            if not recorded_frames:
                logger.warning("No frames recorded, skipping enhancement")
                _reset_recording_state()
                return

            enhanced_audio, enhanced_sr = enhance_utterance(
                recorded_frames,
                model,
                df_state,
                target_sr,
            )
            last_result = (enhanced_audio, enhanced_sr)
            if on_utterance is not None:
                # Callback handles TTS - will resume recording when done
                if on_utterance(enhanced_audio, enhanced_sr):
                    stop_requested = True
        except Exception as e:
            logger.exception("Failed to enhance/save audio: %s", e)
        finally:
            _reset_recording_state()

    try:
        was_paused = False
        ignore_frames_remaining = 0  # Counter for post-resume grace period
        
        while True:
            # Check pause state BEFORE reading (to avoid blocking while paused)
            if is_recording_paused():
                # Stop stream to prevent buffering audio during TTS playback
                if stream.is_active():
                    try:
                        stream.stop_stream()
                        logger.debug("Audio stream stopped during pause")
                    except Exception as e:
                        logger.debug("Failed to stop stream: %s", e)
                
                was_paused = True
                time.sleep(0.05)
                continue
            
            # Check if we just resumed from pause - clear all buffers and restart stream
            # Consume both flags to avoid double-triggering
            need_reset = was_paused
            clear_flag = should_clear_buffer()  # Always consume the flag
            need_reset = need_reset or clear_flag
            
            if need_reset:
                was_paused = False
                pre_buffer.clear()
                recorded_frames.clear()
                recording = False
                recording_start_time = None
                consecutive_silence_frames = 0
                speech_end_time = None
                
                # Set grace period to ignore audio immediately after resume
                # This prevents capturing echo/reverb from TTS playback
                ignore_frames_remaining = POST_RESUME_IGNORE_FRAMES
                
                # Restart stream to ensure clean state (no buffered audio from TTS playback)
                try:
                    if not stream.is_active():
                        stream.start_stream()
                        logger.debug("Audio stream restarted")
                    
                    # Flush any remaining buffer
                    available = stream.get_read_available()
                    if available > 0:
                        _ = stream.read(available, exception_on_overflow=False)
                        logger.debug("Flushed %d frames from audio buffer", available)
                except Exception as e:
                    logger.debug("Stream restart/flush: %s", e)
                
                logger.info("Recording resumed - ignoring %d frames (%.1fs grace period)", 
                           ignore_frames_remaining, POST_RESUME_IGNORE_MS / 1000)
                continue  # Skip this iteration to ensure clean start
            
            # Read audio frame
            try:
                frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
            except Exception as e:
                logger.warning("Stream read error: %s", e)
                time.sleep(0.1)
                continue
                
            if len(frame) != EXPECTED_FRAME_SIZE:
                continue

            # Grace period: ignore frames immediately after resume to avoid TTS echo
            if ignore_frames_remaining > 0:
                ignore_frames_remaining -= 1
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
                    # After processing, set was_paused to trigger buffer clear on next iteration
                    was_paused = True
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
                        # After processing, set was_paused to trigger buffer clear on next iteration
                        was_paused = True
                        continue

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
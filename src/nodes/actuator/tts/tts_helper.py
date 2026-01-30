"""
Text-to-Speech helper utilities for intent responses.
"""
import logging

from .model_loader import load_tts_synthesizer
from .audio_player import play_audio_bytes

logger = logging.getLogger(__name__)


def speak_text(text: str, verbose: bool = False) -> bool:
    """
    Synthesize and play text as speech.
    
    Args:
        text: Text to synthesize and speak
        verbose: Whether to log verbose messages (default: False)
    
    Returns:
        True if successful, False otherwise
    """
    if not text or not text.strip():
        if verbose:
            logger.warning("Empty text provided to speak_text")
        return False
    
    try:
        synthesizer = load_tts_synthesizer()
        audio_bytes = synthesizer.synthesize_to_bytes(text)
        play_audio_bytes(audio_bytes)
        
        if verbose:
            logger.info(f"✓ Spoke: {text[:50]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to speak text: {e}", exc_info=verbose)
        return False
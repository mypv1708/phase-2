"""
TTS Model Loader - Load and manage TTS models
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Union, TYPE_CHECKING, Any
from .synthesizer import TTSSynthesizer

if TYPE_CHECKING:
    from .synthesizer import TTSSynthesizer

from .config import (
    TTS_MODEL_DIR,
    TTS_DEFAULT_LANGUAGE,
    TTS_DEFAULT_MODEL_NAME,
    TTS_MODEL_SEARCH_PATHS,
)

logger = logging.getLogger(__name__)

_tts_model_cache: Dict[str, Any] = {}
_default_model_path: Optional[Path] = None


def find_tts_model() -> Optional[Path]:
    """Find default TTS model file in search paths."""
    model_name = TTS_DEFAULT_MODEL_NAME
    lang = TTS_DEFAULT_LANGUAGE
    
    for search_path in TTS_MODEL_SEARCH_PATHS:
        search_dir = Path(search_path)
        
        # Try language subdirectory first
        model_path = search_dir / lang / model_name
        if model_path.exists():
            return model_path
        
        # Try direct path
        model_path = search_dir / model_name
        if model_path.exists():
            return model_path
    
    return None


def get_default_model_path() -> Path:
    """Get default TTS model path (cached)."""
    global _default_model_path
    
    if _default_model_path is not None:
        return _default_model_path
    
    model_path = find_tts_model()
    if model_path is None:
        default_path = TTS_MODEL_DIR / TTS_DEFAULT_LANGUAGE / TTS_DEFAULT_MODEL_NAME
        raise FileNotFoundError(
            f"Default TTS model not found: {default_path}\n"
            f"Searched in: {TTS_MODEL_SEARCH_PATHS}"
        )
    
    _default_model_path = model_path
    return model_path


def load_tts_synthesizer(
    model_path: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
) -> "TTSSynthesizer":
    """Load TTS synthesizer with caching support."""
    
    # Determine model path
    if model_path is None:
        model_path = get_default_model_path()
    
    model_path = Path(model_path)
    cache_key = str(model_path)
    
    # Check cache
    if use_cache and cache_key in _tts_model_cache:
        return _tts_model_cache[cache_key]
    
    # Load synthesizer
    try:
        synthesizer = TTSSynthesizer(model_path=str(model_path))
        
        if use_cache:
            _tts_model_cache[cache_key] = synthesizer
        
        return synthesizer
        
    except Exception as e:
        raise RuntimeError(f"Failed to load TTS synthesizer: {e}") from e
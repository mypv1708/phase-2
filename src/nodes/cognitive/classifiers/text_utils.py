"""
Text processing utilities for Vietnamese.
"""
import re
import unicodedata
from typing import Optional, Set


def normalize_text(text: str) -> str:
    """Normalize Vietnamese text: lowercase, normalize unicode, strip."""
    text = unicodedata.normalize("NFC", text.lower().strip())
    text = re.sub(r'\s+', ' ', text)
    return text


def remove_ending_particles(text: str, particles: Set[str]) -> str:
    """Remove ending particles from text."""
    words = text.split()
    while words and words[-1] in particles:
        words.pop()
    return ' '.join(words)


def remove_all_particles(text: str, particles: Set[str]) -> str:
    """Remove all particles from text."""
    words = [w for w in text.split() if w not in particles]
    return ' '.join(words)


def clean_text(text: str) -> str:
    """Remove punctuation and extra whitespace."""
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def check_pattern_match(text: str, pattern: re.Pattern) -> Optional[str]:
    """Check if text matches pattern and return remaining text."""
    match = pattern.match(text)
    if match:
        return text[match.end():].strip()
    return None


def is_only_stop_words(text: str, stop_words: Set[str]) -> bool:
    """Check if text contains only stop words."""
    words = text.split()
    return bool(words) and all(w in stop_words for w in words)

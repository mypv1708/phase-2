"""
Regex patterns for greeting classification.
"""
import re
from typing import Dict, Any

from ..config.settings import (
    INTENT_GREETING, INTENT_NOT_GREETING, INTENT_NOISE,
    GREETING_WORDS, TIME_PHRASES, WISH_PHRASES,
    COMPANION_WORDS, ENDING_PARTICLES, NOISE_WORDS,
    CONFIDENCE_HIGH, CONFIDENCE_MEDIUM,
)


class GreetingPatterns:
    """Patterns and constants for greeting detection."""
    
    # Re-export from config for backward compatibility
    INTENT_GREETING = INTENT_GREETING
    INTENT_NOT_GREETING = INTENT_NOT_GREETING
    INTENT_NOISE = INTENT_NOISE
    CONFIDENCE_HIGH = "high"
    CONFIDENCE_MEDIUM = "medium"
    
    GREETING_WORDS = GREETING_WORDS
    TIME_PHRASES = TIME_PHRASES
    WISH_PHRASES = WISH_PHRASES
    COMPANION_WORDS = COMPANION_WORDS
    ENDING_PARTICLES = ENDING_PARTICLES
    NOISE_WORDS = NOISE_WORDS
    
    OTHER_GREETINGS = ["rất vui được gặp", "hân hạnh"]
    
    @classmethod
    def compile_patterns(cls) -> Dict[str, Any]:
        """Compile regex patterns for efficient matching."""
        greeting_patterns = [
            re.compile(rf'^{re.escape(word)}\b\s*', re.IGNORECASE)
            for word in cls.GREETING_WORDS
        ]
        
        time_greeting_patterns = [
            re.compile(rf'^chào\s+{re.escape(time)}\b\s*', re.IGNORECASE)
            for time in cls.TIME_PHRASES
        ]
        
        other_greeting_patterns = [
            re.compile(rf'^{re.escape(phrase)}\b\s*', re.IGNORECASE)
            for phrase in cls.OTHER_GREETINGS
        ]
        
        # Special patterns
        hey_robot = re.compile(r'^robot\s+ơi\b', re.IGNORECASE)
        time_wish = re.compile(
            r'^(buổi\s+)?(sáng|chiều|tối)\s+(vui\s+vẻ|tốt\s+lành|an\s+lành)\b',
            re.IGNORECASE
        )
        
        return {
            "greeting": greeting_patterns,
            "time_greeting": time_greeting_patterns,
            "other_greeting": other_greeting_patterns,
            "hey_robot_pattern": hey_robot,
            "time_wish": time_wish,
        }

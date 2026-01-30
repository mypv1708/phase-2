"""
Rule-based greeting classifier for Vietnamese.
"""
from typing import Dict, Optional, Any

from .patterns import GreetingPatterns
from .text_utils import (
    normalize_text, remove_ending_particles, remove_all_particles,
    clean_text, check_pattern_match, is_only_stop_words,
)

GreetingResult = Dict[str, Any]


class GreetingClassifier:
    """Classifier for detecting greeting intent in Vietnamese."""
    
    def __init__(self):
        self.patterns = GreetingPatterns.compile_patterns()
    
    def is_greeting(self, text: str) -> GreetingResult:
        """
        Check if text is a greeting.
        
        Returns:
            {
                "text": str,
                "intent": "greeting" | "not_greeting" | "noise",
                "is_greeting": bool,
                "confidence": "high" | "medium"
            }
        """
        original = text
        normalized = normalize_text(text)
        
        # Check noise first
        if is_only_stop_words(normalized, GreetingPatterns.NOISE_WORDS):
            return self._result(original, False, intent=GreetingPatterns.INTENT_NOISE)
        
        # Check special patterns
        if self._check_special_patterns(normalized):
            return self._result(original, True)
        
        # Check basic greeting
        result = self._check_basic_greeting(normalized)
        if result:
            return {**result, "text": original}
        
        # Check time greeting
        result = self._check_time_greeting(normalized)
        if result:
            return {**result, "text": original}
        
        return self._result(original, False)
    
    def _result(
        self, text: str, is_greeting: bool,
        confidence: str = GreetingPatterns.CONFIDENCE_HIGH,
        intent: Optional[str] = None
    ) -> GreetingResult:
        if intent is None:
            intent = GreetingPatterns.INTENT_GREETING if is_greeting else GreetingPatterns.INTENT_NOT_GREETING
        return {
            "text": text,
            "intent": intent,
            "is_greeting": is_greeting,
            "confidence": confidence,
        }
    
    def _check_special_patterns(self, normalized: str) -> bool:
        clean = remove_ending_particles(normalized, GreetingPatterns.ENDING_PARTICLES)
        
        if self.patterns["hey_robot_pattern"].match(clean):
            return True
        if self.patterns["time_wish"].match(clean):
            return True
        
        for pattern in self.patterns["other_greeting"]:
            remaining = check_pattern_match(clean, pattern)
            if remaining is not None:
                remaining_clean = remove_all_particles(remaining, GreetingPatterns.ENDING_PARTICLES)
                if not remaining_clean or remaining_clean in GreetingPatterns.COMPANION_WORDS:
                    return True
        return False
    
    def _check_basic_greeting(self, normalized: str) -> Optional[GreetingResult]:
        for pattern in self.patterns["greeting"]:
            remaining = check_pattern_match(normalized, pattern)
            if remaining is not None and self._is_valid_remaining(remaining):
                confidence = GreetingPatterns.CONFIDENCE_HIGH if not remaining else GreetingPatterns.CONFIDENCE_MEDIUM
                return {
                    "intent": GreetingPatterns.INTENT_GREETING,
                    "is_greeting": True,
                    "confidence": confidence,
                }
        return None
    
    def _check_time_greeting(self, normalized: str) -> Optional[GreetingResult]:
        for pattern in self.patterns["time_greeting"]:
            remaining = check_pattern_match(normalized, pattern)
            if remaining is not None:
                remaining = remove_ending_particles(remaining, GreetingPatterns.ENDING_PARTICLES)
                if not remaining or remaining in GreetingPatterns.COMPANION_WORDS:
                    return {
                        "intent": GreetingPatterns.INTENT_GREETING,
                        "is_greeting": True,
                        "confidence": GreetingPatterns.CONFIDENCE_HIGH,
                    }
        return None
    
    def _is_valid_remaining(self, remaining: str) -> bool:
        if not remaining:
            return True
        
        remaining = clean_text(remaining)
        if not remaining:
            return True
        
        remaining = remove_ending_particles(remaining, GreetingPatterns.ENDING_PARTICLES)
        if not remaining:
            return True
        
        remaining_clean = remove_all_particles(remaining, GreetingPatterns.ENDING_PARTICLES)
        
        # Single companion word
        if not remaining_clean:
            return True
        words = remaining_clean.split()
        if len(words) == 1 and words[0] in GreetingPatterns.COMPANION_WORDS:
            return True
        
        # Time phrase
        for phrase in GreetingPatterns.TIME_PHRASES:
            if remaining_clean.startswith(phrase):
                rest = remaining_clean[len(phrase):].strip()
                if not rest or rest in GreetingPatterns.COMPANION_WORDS:
                    return True
                if any(wish in rest for wish in GreetingPatterns.WISH_PHRASES):
                    return True
        
        return False

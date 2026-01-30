"""
Rule-based stop command classifier for Vietnamese.
"""
import re
from typing import Dict, Any, List

from ..config.settings import (
    INTENT_STOP, STOP_WORDS, STOP_PREFIXES, STOP_SUFFIXES,
    CONFIDENCE_HIGH, CONFIDENCE_MEDIUM,
)

StopResult = Dict[str, Any]


class StopClassifier:
    """Classifier for detecting stop intent in Vietnamese."""
    
    def __init__(self):
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for stop detection."""
        patterns = []
        
        # Sort by length (longer first) for greedy matching
        sorted_words = sorted(STOP_WORDS, key=len, reverse=True)
        
        for word in sorted_words:
            # Optional prefix + stop word + optional suffix
            prefix_pattern = r'(?:' + '|'.join(re.escape(p) for p in STOP_PREFIXES) + r')?\s*'
            suffix_pattern = r'\s*(?:' + '|'.join(re.escape(s) for s in STOP_SUFFIXES) + r')?'
            
            pattern = re.compile(
                rf'^{prefix_pattern}{re.escape(word)}{suffix_pattern}$',
                re.IGNORECASE
            )
            patterns.append(pattern)
        
        return patterns
    
    def is_stop(self, text: str) -> StopResult:
        """
        Check if text is a stop command.
        
        Returns:
            {
                "text": str,
                "intent": "stop" | "not_stop",
                "is_stop": bool,
                "confidence": "high" | "medium"
            }
        """
        original = text
        normalized = self._normalize(text)
        
        # Empty or too long → not stop
        if not normalized or len(normalized) > 50:
            return self._result(original, False)
        
        # Check exact patterns
        for pattern in self.patterns:
            if pattern.match(normalized):
                return self._result(original, True, confidence=CONFIDENCE_HIGH)
        
        # Check if contains stop word (partial match)
        for word in STOP_WORDS:
            if word in normalized:
                # Only if reasonably short (likely stop command)
                if len(normalized.split()) <= 6:
                    return self._result(original, True, confidence=CONFIDENCE_MEDIUM)
        
        return self._result(original, False)
    
    def _normalize(self, text: str) -> str:
        """Normalize text for matching."""
        text = text.lower().strip()
        # Remove punctuation
        text = re.sub(r'[!?.,:;]+', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _result(
        self, text: str, is_stop: bool,
        confidence: float = CONFIDENCE_HIGH
    ) -> StopResult:
        intent = INTENT_STOP if is_stop else "not_stop"
        return {
            "text": text,
            "intent": intent,
            "is_stop": is_stop,
            "confidence": confidence,
        }


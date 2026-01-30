"""
Confidence checking utilities.
"""
from typing import Dict, Any, Tuple

from ..config.settings import (
    CONFIDENCE_THRESHOLD,
    CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW,
)


def get_confidence_value(result: Dict[str, Any]) -> float:
    """Get confidence as float."""
    confidence = result.get("confidence", 0.0)
    
    if isinstance(confidence, str):
        if confidence == "high":
            return CONFIDENCE_HIGH
        elif confidence == "medium":
            return CONFIDENCE_MEDIUM
        return CONFIDENCE_LOW
    
    return float(confidence)


def check_confidence_with_value(
    result: Dict[str, Any], 
    threshold: float = CONFIDENCE_THRESHOLD
) -> Tuple[float, bool]:
    """
    Get confidence value and check threshold in one call.
    
    Returns:
        (confidence_value, passed_threshold)
    """
    value = get_confidence_value(result)
    return value, value >= threshold

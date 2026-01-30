"""
Intent Processor - Intent Classification + Routing + TTS

This module provides the main entry point for processing user input through
the cognitive pipeline: intent classification → routing → response generation.
"""
import time

from .intents import classify_intent, route_intent


def process_input(text: str, use_tts: bool = True) -> dict:
    """
    Process user input through the cognitive pipeline.
    
    Flow:
    1. Classify intent (rule-based + OpenAI API)
    2. Route → check confidence → execute handler
    3. TTS response (optional)
    
    Args:
        text: User input text to process
        use_tts: Whether to use TTS for response (default: True)
    
    Returns:
        {
            "intent": str,
            "confidence": float,
            "passed": bool,
            "route": str,
            "response": str,
            "command": str or None,
            "action_result": any or None,
            "classify_time_ms": float,
        }
    """
    # Measure classify time only
    start = time.perf_counter()
    result = classify_intent(text)
    classify_time = (time.perf_counter() - start) * 1000
    
    output = route_intent(result, use_tts=use_tts)
    output["classify_time_ms"] = round(classify_time, 1)
    return output

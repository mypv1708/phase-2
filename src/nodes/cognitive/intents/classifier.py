"""
Combined intent classifier: rule-based + OpenAI API.
"""
from typing import Dict, Any, Optional

from ..config.settings import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM
from ..classifiers import GreetingClassifier, StopClassifier
from ..clients.openai import call_openai, extract_text, parse_json_response

# Singletons
_greeting_classifier: Optional[GreetingClassifier] = None
_stop_classifier: Optional[StopClassifier] = None


def get_greeting_classifier() -> GreetingClassifier:
    global _greeting_classifier
    if _greeting_classifier is None:
        _greeting_classifier = GreetingClassifier()
    return _greeting_classifier


def get_stop_classifier() -> StopClassifier:
    global _stop_classifier
    if _stop_classifier is None:
        _stop_classifier = StopClassifier()
    return _stop_classifier


def classify_with_openai(text: str) -> Dict[str, Any]:
    """Classify intent using OpenAI API."""
    response = call_openai(text)
    response_text = extract_text(response)
    result = parse_json_response(response_text)
    
    if not result.get("raw_text"):
        result["raw_text"] = text
    
    return result


def classify_intent(text: str) -> Dict[str, Any]:
    """
    Classify intent combining rule-based and OpenAI API.
    
    Flow:
    1. Check greeting/noise with rule-based (fast, no API)
    2. Check stop command with rule-based (fast, no API)
    3. If greeting/stop → return
    4. If noise → return noise intent
    5. Otherwise → call OpenAI API
    
    Returns:
        {
            "intent": str,
            "confidence": float,
            "slots": dict,
            "response": str,
            "raw_text": str
        }
    """
    # 1. Check greeting
    greeting_classifier = get_greeting_classifier()
    greeting_result = greeting_classifier.is_greeting(text)
    
    if greeting_result.get("is_greeting"):
        return {
            "intent": "greeting",
            "confidence": CONFIDENCE_HIGH if greeting_result.get("confidence") == "high" else CONFIDENCE_MEDIUM,
            "slots": {},
            "response": "",
            "raw_text": text,
        }
    
    # 2. Check noise
    if greeting_result.get("intent") == "noise":
        return {
            "intent": "noise",
            "confidence": CONFIDENCE_HIGH,
            "slots": {},
            "response": "",
            "raw_text": text,
        }
    
    # 3. Check stop command (fast rule-based)
    stop_classifier = get_stop_classifier()
    stop_result = stop_classifier.is_stop(text)
    
    if stop_result.get("is_stop"):
        return {
            "intent": "stop",
            "confidence": stop_result.get("confidence", CONFIDENCE_HIGH),
            "slots": {},
            "response": "",
            "raw_text": text,
        }
    
    # 4. Other intents → OpenAI API
    return classify_with_openai(text)


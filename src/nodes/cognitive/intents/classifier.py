"""
Combined intent classifier: rule-based + OpenAI API.
"""
from typing import Dict, Any, Optional

from ..config.settings import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM
from ..classifiers import GreetingClassifier
from ..clients.openai import call_openai, extract_text, parse_json_response

# Singleton
_greeting_classifier: Optional[GreetingClassifier] = None


def get_greeting_classifier() -> GreetingClassifier:
    global _greeting_classifier
    if _greeting_classifier is None:
        _greeting_classifier = GreetingClassifier()
    return _greeting_classifier


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
    2. If greeting → return
    3. If noise → return noise intent
    4. Otherwise → call OpenAI API
    
    Returns:
        {
            "intent": str,
            "confidence": float,
            "slots": dict,
            "response": str,
            "raw_text": str
        }
    """
    classifier = get_greeting_classifier()
    result = classifier.is_greeting(text)
    
    # Greeting detected
    if result.get("is_greeting"):
        return {
            "intent": "greeting",
            "confidence": CONFIDENCE_HIGH if result.get("confidence") == "high" else CONFIDENCE_MEDIUM,
            "slots": {},
            "response": "",
            "raw_text": text,
        }
    
    # Noise detected
    if result.get("intent") == "noise":
        return {
            "intent": "noise",
            "confidence": CONFIDENCE_HIGH,
            "slots": {},
            "response": "",
            "raw_text": text,
        }
    
    # Other intents → OpenAI API
    return classify_with_openai(text)


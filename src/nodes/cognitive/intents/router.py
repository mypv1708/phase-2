"""
Intent router - routes intents to appropriate handlers.
"""
from typing import Dict, Any, Callable, Optional

from ..config.settings import (
    CONFIDENCE_THRESHOLD,
    COMMAND_INTENTS, RESPONSE_INTENTS, ACTION_INTENTS,
)
from ..utils.confidence import check_confidence_with_value
from ..commands.converter import convert_result
from ..responses.templates import get_response
from ..clients.tts import speak_text


class IntentRouter:
    """Router for intent handling."""
    
    def __init__(self, threshold: float = CONFIDENCE_THRESHOLD):
        self.threshold = threshold
        self._action_handlers: Dict[str, Callable] = {}
    
    def register_action(self, intent: str, handler: Callable):
        """Register handler for action intent."""
        self._action_handlers[intent] = handler
    
    def get_route_type(self, intent: str) -> str:
        """Determine route type for intent."""
        if intent in COMMAND_INTENTS:
            return "command"
        if intent in RESPONSE_INTENTS:
            return "response"
        if intent in ACTION_INTENTS:
            return "action"
        return "unknown"
    
    def route(self, result: Dict[str, Any], use_tts: bool = True) -> Dict[str, Any]:
        """Route intent to appropriate handler."""
        intent = result.get("intent", "unknown")
        confidence, passed = check_confidence_with_value(result, self.threshold)
        route_type = self.get_route_type(intent)
        
        output = {
            "intent": intent,
            "confidence": confidence,
            "passed": passed,
            "route": route_type,
            "response": "",
            "command": None,
            "action_result": None,
        }
        
        # Low confidence → fallback
        if not passed:
            output["response"] = "Xin lỗi, tôi không chắc chắn. Bạn nói lại được không?"
        
        # Route: command
        elif route_type == "command":
            result = convert_result(result)
            output["command"] = result.get("formatted_command")
            output["response"] = self._command_response(intent, result)
        
        # Route: response
        elif route_type == "response":
            output["response"] = self._text_response(intent, result)
        
        # Route: action
        elif route_type == "action":
            output["action_result"] = self._execute_action(intent, result)
            output["response"] = self._action_response(intent, result)
        
        # TTS
        if use_tts and output["response"]:
            speak_text(output["response"])
        
        return output
    
    def _command_response(self, intent: str, result: Dict[str, Any]) -> str:
        if intent == "navigate":
            return f"Đã nhận lệnh di chuyển với {len(result.get('slots', []))} hành động."
        if intent == "stop":
            return "Đã dừng lại."
        return ""
    
    def _text_response(self, intent: str, result: Dict[str, Any]) -> str:
        if intent == "greeting":
            return get_response("greeting", result.get("raw_text", ""))
        if intent == "noise":
            return get_response("noise", "")
        if intent == "conversation":
            return result.get("response", "Tôi không hiểu.")
        return "Xin lỗi, tôi không hiểu yêu cầu của bạn."
    
    def _action_response(self, intent: str, result: Dict[str, Any]) -> str:
        slots = result.get("slots", {})
        
        if intent == "tracking-person":
            return "Đang theo dõi bạn."
        if intent == "go_to_object":
            return f"Đang đi tới {slots.get('object', 'đó')}."
        if intent == "go_to_location":
            return f"Đang đi tới {slots.get('location', 'đó')}."
        return ""
    
    def _execute_action(self, intent: str, result: Dict[str, Any]) -> Optional[Any]:
        handler = self._action_handlers.get(intent)
        return handler(result) if handler else None


# Singleton
_router: Optional[IntentRouter] = None


def get_router() -> IntentRouter:
    global _router
    if _router is None:
        _router = IntentRouter()
    return _router


def route_intent(result: Dict[str, Any], use_tts: bool = True) -> Dict[str, Any]:
    """Shortcut to route intent."""
    return get_router().route(result, use_tts)

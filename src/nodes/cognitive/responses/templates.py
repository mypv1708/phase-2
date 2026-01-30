"""
Response templates for different intents.
"""
import random
from datetime import datetime

MORNING_RESPONSES = [
    "Chào buổi sáng! Mình có thể giúp gì cho bạn?",
    "Mình đây! Bạn cần mình hỗ trợ gì không?",
    "Chào buổi sáng! Mình sẵn sàng giúp bạn.",
    "Buổi sáng vui vẻ! Mình có thể làm gì cho bạn?",
]

AFTERNOON_RESPONSES = [
    "Chào buổi chiều! Mình có thể giúp gì cho bạn?",
    "Mình đây! Bạn cần mình hỗ trợ gì không?",
    "Chào buổi chiều! Mình sẵn sàng giúp bạn.",
    "Chúc bạn buổi chiều vui vẻ! Mình có thể giúp gì?",
]

EVENING_RESPONSES = [
    "Chào buổi tối! Mình có thể giúp gì cho bạn?",
    "Mình đây! Bạn cần mình hỗ trợ gì không?",
    "Chào buổi tối! Mình luôn sẵn sàng giúp bạn.",
    "Chúc bạn buổi tối vui vẻ! Mình có thể làm gì?",
]

CASUAL_RESPONSES = [
    "Chào bạn yêu! Bạn cần mình làm gì nào?",
]

NOISE_RESPONSES = [
    "Mình nghe chưa rõ, bạn nói lại giúp mình nhé?",
    "Xin lỗi, mình chưa nghe rõ. Bạn lặp lại được không?",
]


def _get_time_of_day() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    return "evening"


def get_greeting_response(text_input: str) -> str:
    """Generate greeting response based on input and time."""
    text_lower = text_input.lower().strip()
    
    if "robot ơi" in text_lower:
        return random.choice(CASUAL_RESPONSES)
    
    if "sáng" in text_lower:
        return random.choice(MORNING_RESPONSES)
    if "chiều" in text_lower:
        return random.choice(AFTERNOON_RESPONSES)
    if "tối" in text_lower:
        return random.choice(EVENING_RESPONSES)
    
    time = _get_time_of_day()
    if time == "morning":
        return random.choice(MORNING_RESPONSES)
    elif time == "afternoon":
        return random.choice(AFTERNOON_RESPONSES)
    return random.choice(EVENING_RESPONSES)


def get_response(intent: str, text_input: str = "") -> str:
    """Get response for intent."""
    if intent == "greeting":
        return get_greeting_response(text_input)
    if intent == "noise":
        return random.choice(NOISE_RESPONSES)
    return ""

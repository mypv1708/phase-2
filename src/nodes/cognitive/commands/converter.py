"""
Convert intent slots to robot command string.
"""
from typing import Dict, List, Any

from ..config.settings import (
    MOVE_COMMANDS, TURN_COMMANDS,
    DEFAULT_DISTANCE, DEFAULT_ANGLE,
)


def _format_value(value: float) -> str:
    """Format number: remove trailing zeros."""
    if value is None:
        raise ValueError("Value cannot be None")
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def normalize_slots(slots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize slots: set default values for null distance/angle."""
    normalized = []
    for slot in slots:
        slot_copy = dict(slot)
        slot_type = slot_copy.get("type")
        
        if slot_type == "move" and slot_copy.get("distance") is None:
            slot_copy["distance"] = DEFAULT_DISTANCE
        elif slot_type == "turn" and slot_copy.get("angle") is None:
            slot_copy["angle"] = DEFAULT_ANGLE
        
        normalized.append(slot_copy)
    return normalized


def slot_to_command(slot: Dict[str, Any]) -> str:
    """Convert a slot to command string."""
    slot_type = slot.get("type")
    direction = slot.get("direction")
    
    if slot_type == "move":
        cmd = MOVE_COMMANDS.get(direction)
        if not cmd:
            raise ValueError(f"Unknown move direction: {direction}")
        return f"{cmd},{_format_value(slot.get('distance'))}"
    
    elif slot_type == "turn":
        cmd = TURN_COMMANDS.get(direction)
        if not cmd:
            raise ValueError(f"Unknown turn direction: {direction}")
        return f"{cmd},{_format_value(slot.get('angle'))}"
    
    raise ValueError(f"Unknown slot type: {slot_type}")


def slots_to_command(slots: List[Dict[str, Any]]) -> str:
    """Convert slots array to command string."""
    if not slots:
        raise ValueError("Slots array is empty")
    
    normalized = normalize_slots(slots)
    commands = [slot_to_command(slot) for slot in normalized]
    return "$SEQ;" + ";".join(commands) + ";STOP\n"


def convert_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert intent result and add formatted_command."""
    intent = result.get("intent")
    result_with_cmd = dict(result)
    
    if intent == "navigate":
        slots = result.get("slots", [])
        try:
            result_with_cmd["formatted_command"] = slots_to_command(slots)
        except ValueError as e:
            result_with_cmd["formatted_command"] = None
            result_with_cmd["command_error"] = str(e)
    
    elif intent == "stop":
        result_with_cmd["formatted_command"] = "$SEQ,STOP\n"
    
    return result_with_cmd


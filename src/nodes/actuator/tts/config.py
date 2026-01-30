"""
TTS (Text-to-Speech) configuration.
"""
from pathlib import Path

# Model Configuration
TTS_MODEL_DIR = Path(__file__).parent / "models"
TTS_DEFAULT_LANGUAGE = "vi"
TTS_DEFAULT_MODEL_NAME = "vi_VN-vais1000-medium.onnx"

# Model Paths
TTS_MODEL_SEARCH_PATHS = [
    str(TTS_MODEL_DIR),
    "./models/tts",
]

# Audio Playback Configuration
AUDIO_PLAYBACK_TIMEOUT = 30

# System Audio Players (in order of preference)
AUDIO_PLAYERS = ["aplay", "paplay", "play"]

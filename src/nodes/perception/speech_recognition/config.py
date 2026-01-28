# Speech-to-Text (PhoWhisper - Vietnamese fine-tuned Whisper via Hugging Face Transformers)
#
# PhoWhisper models (Vietnamese optimized): 
#   "vinai/PhoWhisper-base" (recommended for balance)
#   "vinai/PhoWhisper-small" (faster)
#   "vinai/PhoWhisper-medium" (more accurate)
#   "vinai/PhoWhisper-large" (most accurate, slower)
# Original Whisper models: "openai/whisper-tiny", "openai/whisper-base", etc.
STT_MODEL_ID = "vinai/PhoWhisper-tiny"  # PhoWhisper base model optimized for Vietnamese
STT_SAMPLE_RATE = 16000
STT_DEVICE = None  # None = auto-detect (cuda if available, else cpu)
STT_LANGUAGE = "vi"  # "vi" for Vietnamese (PhoWhisper is optimized for Vietnamese)
STT_TASK = "transcribe"  # "transcribe" or "translate"
STT_NUM_BEAMS = 5  # Beam search size (1 = greedy/fastest, 5 = default/slower)
STT_MAX_NEW_TOKENS = 100  # Maximum tokens to generate
STT_USE_FP16 = True  # Use FP16 for GPU (faster, less memory)

# STT validation thresholds
STT_MIN_DURATION = 0.5  # Minimum audio duration in seconds to process (skip if shorter)
STT_SILENT_THRESHOLD = 0.005  # RMS threshold for silent detection (normalized, ~0.5% energy)
STT_MAX_SILENT_RATIO = 0.7  # Skip STT if > 70% of audio is silent
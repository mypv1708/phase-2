import pyaudio

# Mic-driver only config: VAD/recording, enhance

# Audio capture config
RATE = 48000
CHANNELS = 1
FORMAT = pyaudio.paInt16
SAMPLE_WIDTH = 2

# Frame config
FRAME_DURATION_MS = 30
MILLISECONDS_PER_SECOND = 1000
FRAME_SIZE = int(RATE * FRAME_DURATION_MS / MILLISECONDS_PER_SECOND)

# VAD config
VAD_MODE = 3

# Silence / timing thresholds
SILENCE_LIMIT = 0.8
PRE_BUFFER_MS = 500
PRE_BUFFER_FRAMES = PRE_BUFFER_MS // FRAME_DURATION_MS
SILENCE_EXIT = 35.0
MAX_RECORDING_SECONDS = 30.0

# Post-resume grace period: ignore audio for this duration after resuming
# This prevents capturing echo/reverb from TTS playback
POST_RESUME_IGNORE_MS = 300  # 0.5 seconds

# DeepFilterNet model config
DF_POST_FILTER = True
DF_LOG_LEVEL = "WARNING"

# Audio processing
INT16_MAX = 32767.0

# Enhancement skip thresholds
MIN_ENHANCE_DURATION = 1.0  # Skip enhancement if audio < 1 second
MAX_SILENT_RATIO = 0.7  # Skip enhancement if > 70% of audio is silent
SILENT_THRESHOLD = 0.005  # RMS threshold for silent detection (normalized, ~0.5% energy)
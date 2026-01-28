import pyaudio

# Mic-driver only config: VAD/recording, enhance

# Audio playback
AUDIO_PLAYBACK_CHUNK_SIZE = 1024

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
SILENCE_LIMIT = 1.0
PRE_BUFFER_MS = 500
PRE_BUFFER_FRAMES = PRE_BUFFER_MS // FRAME_DURATION_MS
SILENCE_EXIT = 35.0
MAX_RECORDING_SECONDS = 30.0

# Paths & naming
AUDIO_BASE_DIR = "assets"
RAW_SUBDIR = "raw"
PROCESSED_SUBDIR = "processed"
TIMESTAMP_FMT = "%Y%m%d_%H%M%S"
# File naming
RAW_FILE_PREFIX = "raw_"
ENHANCED_FILE_PREFIX = "enhanced_"
DATE_FMT = "%Y%m%d"

# DeepFilterNet model config
DF_POST_FILTER = True
DF_LOG_LEVEL = "WARNING"

# Audio response
PIP_SOUND_FILE = "assets/audio/pip.wav"

# Audio processing
INT16_MAX = 32767.0

# Enhancement skip thresholds
MIN_ENHANCE_DURATION = 1.0  # Skip enhancement if audio < 1 second
MAX_SILENT_RATIO = 0.7  # Skip enhancement if > 70% of audio is silent
SILENT_THRESHOLD = 0.01  # RMS threshold for silent detection (normalized)
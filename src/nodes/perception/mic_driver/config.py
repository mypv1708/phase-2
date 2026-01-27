import pyaudio

# Mic-driver only config: wake-word, VAD/recording, enhance

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

# Wake word audio stream config
HOTWORD_NAME = "hello_robot"
HOTWORD_REFERENCE_FILE = "assets/embedding-json/hello_robot_ref.json"
HOTWORD_RELAXATION_TIME = 1.0
HOTWORD_THRESHOLD = 0.68
HOTWORD_BLOCKSIZE = 512
HOTWORD_SLIDING_WINDOW_SECS = 0.75
HOTWORD_WINDOW_LENGTH_SECS = 1.5
WAKE_WORD_CHANNELS = 1
WAKE_WORD_DTYPE = "float32"
WAKE_WORD_MIN_AUDIO_LEVEL = 0.001
WAKE_WORD_SLEEP_MS = 100

# Audio response
PIP_SOUND_FILE = "assets/audio/pip.wav"

# Audio processing
INT16_MAX = 32767.0
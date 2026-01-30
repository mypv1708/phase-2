"""
TTS Service - FastAPI server for Text-to-Speech
Run with: uvicorn tts_service:app --host 0.0.0.0 --port 8001
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tts.model_loader import load_tts_synthesizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global TTS synthesizer (loaded once at startup)
tts_synthesizer = None


class SynthesizeRequest(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load TTS model at startup, cleanup at shutdown."""
    global tts_synthesizer
    
    logger.info("Loading TTS model...")
    tts_synthesizer = load_tts_synthesizer()
    # Warm up - load model vào RAM
    tts_synthesizer._ensure_loaded()
    logger.info("TTS Service ready!")
    
    yield
    
    logger.info("TTS Service shutting down...")


app = FastAPI(
    title="TTS Service",
    description="Text-to-Speech API using Piper TTS",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "tts"}


@app.post("/speak")
async def speak(request: SynthesizeRequest):
    """
    Synthesize and play audio on server.
    Blocks until audio playback is complete.
    
    Useful when TTS service runs on same machine as speakers.
    """
    import asyncio
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    def _synthesize_and_play():
        """Blocking function to run in thread pool."""
        from tts.audio_player import play_audio_bytes
        
        audio_bytes = tts_synthesizer.synthesize_to_bytes(request.text)
        play_audio_bytes(audio_bytes)
        return True
    
    try:
        # Run blocking audio playback in thread pool
        # This ensures response is only sent AFTER audio finishes playing
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _synthesize_and_play)
        
        logger.info("TTS playback completed: %s", request.text[:50])
        
        return {
            "success": True,
            "text": request.text,
            "playback_complete": True,
        }
    except Exception as e:
        logger.error(f"Speak failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
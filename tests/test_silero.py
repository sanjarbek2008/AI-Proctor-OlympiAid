import sys
import os
import numpy as np
import scipy.io.wavfile as wav
import logging

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ai_engine import analyze_audio, vad_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_wav(filename, duration=3.0, freq=440.0, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Generate a sine wave (harmonic, should be detected as speech-like or at least silence/noise depending on VAD)
    # Silero might not think a pure sine wave is speech, but let's try.
    # Actually, let's create two files: one silence, one noise.
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    wav.write(filename, sr, (audio * 32767).astype(np.int16))
    with open(filename, 'rb') as f:
        return f.read()

def test_silero_load():
    if vad_model is None:
        logger.error("❌ Silero VAD model failed to load!")
        sys.exit(1)
    else:
        logger.info("✅ Silero VAD model loaded successfully.")

def test_analyze_audio():
    # 1. Test Silence
    print("\n--- Testing Silence ---")
    silence = np.zeros(16000 * 3, dtype=np.int16) # 3 seconds of silence
    wav.write("silence.wav", 16000, silence)
    with open("silence.wav", "rb") as f:
        audio_bytes = f.read()
    
    result = analyze_audio(audio_bytes)
    print(f"Result for silence: {result}")
    
    # 2. Test Sine Wave (Synthetic Sound)
    print("\n--- Testing Sine Wave (Tone) ---")
    audio_bytes = create_dummy_wav("tone.wav")
    result = analyze_audio(audio_bytes)
    print(f"Result for tone: {result}")
    
    # Clean up
    if os.path.exists("silence.wav"): os.remove("silence.wav")
    if os.path.exists("tone.wav"): os.remove("tone.wav")

if __name__ == "__main__":
    test_silero_load()
    test_analyze_audio()

import subprocess
import sys
import sounddevice as sd
import soundfile as sf
from llm import clean_for_speech


def speak(text, voice="en_US-lessac-medium"):
    cleaned = clean_for_speech(text)
    if not cleaned:
        return
    subprocess.run([sys.executable, "-m", "piper", "-m", voice, "-f", "out.wav", "--", cleaned], check=True)
    data, rate = sf.read("out.wav")
    sd.play(data, rate)
    sd.wait()


if __name__ == "__main__":
    speak("NVIDIA is up 2.5% today at $125.53.")

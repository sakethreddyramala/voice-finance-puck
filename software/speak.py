import subprocess
import sys
import sounddevice as sd
import soundfile as sf

def speak(text, voice="en_US-lessac-medium"):
    subprocess.run([sys.executable, "-m", "piper", "-m", voice, "-f", "out.wav", "--", text], check=True)
    data, rate = sf.read("out.wav")
    sd.play(data, rate)
    sd.wait()

if __name__ == "__main__":
    speak("NVIDIA is up two percent today.")

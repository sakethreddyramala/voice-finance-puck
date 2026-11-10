import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

SECONDS = 5
RATE = 16000

print("Recording... speak now")
audio = sd.rec(int(SECONDS * RATE), samplerate=RATE, channels=1, dtype="float32")
sd.wait()
sf.write("clip.wav", audio, RATE)
print("Done recording. Transcribing...")

model = WhisperModel("small", device="cpu", compute_type="int8")
segments, info = model.transcribe(audio.flatten(), language="en")
print("You said:", " ".join(s.text.strip() for s in segments))

"""Voice finance assistant: mic -> faster-whisper (GPU) -> Ollama + finance tools -> Piper -> speaker.

Run from the repo root:
    python software/assistant.py            # hold the hotkey (default F9) to talk, release to send
    python software/assistant.py --enter    # old behaviour: Enter to start/stop recording
    python software/assistant.py --text     # type instead of speaking (debug the LLM/tools)
    python software/assistant.py --demo     # use sample data, no internet needed

The hotkey is global (works while the terminal is not focused), which is the laptop
stand-in for the puck's hold-to-talk button.

Environment variables (all optional):
    OLLAMA_MODEL   default llama3.1:8b
    WHISPER_SIZE   default small.en   (try medium.en if you have VRAM to spare)
    WHISPER_DEVICE auto | cuda | cpu  (default auto: GPU first, CPU fallback)
    PIPER_VOICE    default en_US-lessac-medium
    DEMO_MODE      1 to force demo data
    HOTKEY         push-to-talk key: a pynput key name (f9, f8, space, caps_lock,
                   scroll_lock, pause, ...) or a single character. Default f9.
"""
import os
import sys

if "--demo" in sys.argv:
    os.environ["DEMO_MODE"] = "1"  # must be set before finance_tools is imported

import glob
import queue
import subprocess
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf

import finance_tools
import llm

WHISPER_SIZE = os.environ.get("WHISPER_SIZE", "small.en")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "auto")
PIPER_VOICE = os.environ.get("PIPER_VOICE", "en_US-lessac-medium")
MIC_RATE = 16000
MAX_TURNS = 6  # remember the last N user/assistant exchanges
TEXT_MODE = "--text" in sys.argv
USE_HOTKEY = "--enter" not in sys.argv
HOTKEY = os.environ.get("HOTKEY", "f9").strip().lower()

# Helps Whisper spell tickers and company names correctly.
WHISPER_HINT = (
    "Stock market questions. Companies and tickers: NVIDIA, NVDA, Apple, AAPL, "
    "Microsoft, MSFT, Tesla, TSLA, Amazon, AMZN, Alphabet, Google, Meta, AMD, "
    "S&P 500, Nasdaq, Dow Jones, VIX."
)


# ---- GPU setup --------------------------------------------------------------
def add_cuda_lib_dirs():
    """pip's nvidia-cublas-cu12 / nvidia-cudnn-cu12 wheels drop their DLLs/.so files
    inside site-packages/nvidia/*. Make sure the process can find them."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        roots = list(spec.submodule_search_locations) if spec else []
    except Exception:
        roots = []
    for root in roots:
        for pkg in glob.glob(os.path.join(root, "*")):
            for sub in ("bin", "lib"):
                d = os.path.join(pkg, sub)
                if not os.path.isdir(d):
                    continue
                if os.name == "nt":
                    os.add_dll_directory(d)
                    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                else:
                    import ctypes
                    for so in glob.glob(os.path.join(d, "lib*.so*")):
                        try:
                            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                        except OSError:
                            pass


def load_whisper():
    from faster_whisper import WhisperModel

    if WHISPER_DEVICE in ("auto", "cuda"):
        try:
            add_cuda_lib_dirs()
            model = WhisperModel(WHISPER_SIZE, device="cuda", compute_type="float16")
            # Warm-up: missing CUDA libs often only fail on the first real inference.
            segs, _ = model.transcribe(np.zeros(MIC_RATE, dtype="float32"), language="en")
            list(segs)
            print(f"Whisper '{WHISPER_SIZE}' running on GPU (CUDA, float16).")
            return model
        except Exception as e:
            print(f"(GPU Whisper unavailable, falling back to CPU: {e})")
            if WHISPER_DEVICE == "cuda":
                raise
    model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    print(f"Whisper '{WHISPER_SIZE}' running on CPU (int8).")
    return model


# ---- speech to text ---------------------------------------------------------
def record_until_enter():
    chunks = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    input("\nPress Enter to talk...")
    with sd.InputStream(samplerate=MIC_RATE, channels=1, dtype="float32", callback=callback):
        input("Listening... press Enter to stop.")
    if not chunks:
        return np.zeros(0, dtype="float32")
    return np.concatenate(chunks).flatten()


def _key_matches(pressed, target):
    """True if a pynput key event is the target key (case-insensitive for characters)."""
    if pressed == target:
        return True
    a, b = getattr(pressed, "char", None), getattr(target, "char", None)
    return a is not None and b is not None and a.lower() == b.lower()


class HotkeyRecorder:
    """Hold-to-talk: record while the hotkey is held down, stop when it is released.

    A pynput listener thread watches the keyboard globally and only sets/clears
    events; all audio work stays on the main thread. Key presses that happen while
    the assistant is thinking or speaking are ignored, so a stray press can't
    start a recording mid-answer (and the mic never hears the speaker).
    """

    def __init__(self, key_name):
        from pynput import keyboard  # imported lazily so --enter / --text work without it

        self.key_name = key_name
        self.key = self._parse(keyboard, key_name)
        self._armed = False        # True only while we are waiting for the user to talk
        self._down = False         # True while the hotkey is physically held
        self._pressed = threading.Event()
        self._released = threading.Event()
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    @staticmethod
    def _parse(keyboard, name):
        key = getattr(keyboard.Key, name, None)  # named keys: f9, space, caps_lock, ...
        if key is not None:
            return key
        if len(name) == 1:
            return keyboard.KeyCode.from_char(name)
        raise ValueError(
            f"Unknown HOTKEY '{name}'. Use a key name like f9, f8, space, caps_lock, "
            "scroll_lock, pause, or a single character."
        )

    # -- runs on the pynput thread
    def _on_press(self, key):
        # Holding a key makes the OS repeat key-down events; _down filters those out.
        if self._armed and not self._down and _key_matches(key, self.key):
            self._down = True
            self._pressed.set()

    def _on_release(self, key):
        if self._down and _key_matches(key, self.key):
            self._down = False
            self._released.set()

    # -- runs on the main thread
    def _wait(self, event):
        # Poll instead of event.wait() so Ctrl+C still works on Windows.
        while not event.wait(0.1):
            pass

    def record(self):
        chunks = []

        def callback(indata, frames, time_info, status):
            chunks.append(indata.copy())

        self._pressed.clear()
        self._released.clear()
        self._armed = True
        print(f"\nHold {self.key_name.upper()} to talk...")
        self._wait(self._pressed)
        self._armed = False
        print("Listening... release to send.")
        with sd.InputStream(samplerate=MIC_RATE, channels=1, dtype="float32", callback=callback):
            self._wait(self._released)
        if not chunks:
            return np.zeros(0, dtype="float32")
        return np.concatenate(chunks).flatten()

    def close(self):
        self._listener.stop()


def make_recorder():
    """Return a zero-argument function that records one utterance and returns float32 audio."""
    if not USE_HOTKEY:
        return record_until_enter
    try:
        recorder = HotkeyRecorder(HOTKEY)
    except ImportError:
        print("(pynput is not installed, falling back to Enter-to-talk. "
              "Run: pip install pynput)")
        return record_until_enter
    except ValueError as e:
        print(f"({e} Falling back to Enter-to-talk.)")
        return record_until_enter
    except Exception as e:  # e.g. no display / missing OS permission for global key capture
        print(f"(Hotkey listener unavailable: {e}. Falling back to Enter-to-talk.)")
        return record_until_enter
    return recorder.record


def transcribe(stt, audio):
    if len(audio) < MIC_RATE * 0.3:
        return ""
    segments, _ = stt.transcribe(
        audio, language="en", vad_filter=True, initial_prompt=WHISPER_HINT
    )
    return " ".join(s.text.strip() for s in segments).strip()


# ---- text to speech ---------------------------------------------------------
# Piper is fast enough on CPU, so it stays there and leaves VRAM for Whisper + the LLM.
_voice = None


def init_piper():
    global _voice
    try:
        from piper import PiperVoice
        _voice = PiperVoice.load(f"{PIPER_VOICE}.onnx")
    except Exception as e:
        print(f"(Piper Python API unavailable: {e}; using CLI fallback, a bit slower)")


def synth_to_audio(text):
    if _voice is not None:
        pieces, rate = [], None
        for chunk in _voice.synthesize(text):
            rate = chunk.sample_rate
            pieces.append(chunk.audio_int16_array)
        if pieces:
            return np.concatenate(pieces), rate
    subprocess.run(
        [sys.executable, "-m", "piper", "-m", PIPER_VOICE, "-f", "out.wav", "--", text],
        check=True,
    )
    data, rate = sf.read("out.wav")
    return data, rate


speech_q = queue.Queue()


def speaker_worker():
    """Plays sentences in order as they arrive, so speech starts before the LLM finishes."""
    while True:
        sentence = speech_q.get()
        if sentence is None:
            speech_q.task_done()
            return
        try:
            audio, rate = synth_to_audio(sentence)
            sd.play(audio, rate)
            sd.wait()
        except Exception as e:
            print(f"[tts error] {e}")
        speech_q.task_done()


# ---- main loop --------------------------------------------------------------
def main():
    stt = None
    if not TEXT_MODE:
        print("Loading Whisper...")
        stt = load_whisper()
    init_piper()
    threading.Thread(target=speaker_worker, daemon=True).start()
    record = None if TEXT_MODE else make_recorder()

    demo = os.environ.get("DEMO_MODE") == "1"
    print(f"Ready. LLM: {llm.OLLAMA_MODEL} | data: {'DEMO' if demo else 'live yfinance'} | Ctrl+C to quit.")

    history = []
    try:
        while True:
            if TEXT_MODE:
                text = input("\nYou: ").strip()
            else:
                text = transcribe(stt, record())
                print(f"You: {text}" if text else "(didn't catch anything)")
            if not text:
                continue

            history.append({"role": "user", "content": text})
            print("Assistant: ", end="", flush=True)
            try:
                reply = llm.ask(
                    history,
                    on_sentence=speech_q.put,
                    tools=finance_tools.TOOL_SCHEMAS,
                    run_tool=finance_tools.run_tool,
                    on_tool=lambda n, a: print(f"\n[tool] {n}({a})", flush=True),
                )
            except Exception as e:
                print(f"\n[LLM error] {e}\nIs Ollama running and is '{llm.OLLAMA_MODEL}' pulled?")
                history.pop()
                continue
            # Keep only plain user/assistant text in memory (tool traces are per-turn).
            history.append({"role": "assistant", "content": reply})
            history = history[-MAX_TURNS * 2:]

            speech_q.join()  # finish talking before listening again (avoids mic echo)
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
    finally:
        speech_q.put(None)


if __name__ == "__main__":
    main()

# Setup

Tested target: Windows with PowerShell, an NVIDIA RTX 4060 (8 GB VRAM), Python 3.10+.

## 1. Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r software/requirements.txt
```

## 2. Piper voice

```powershell
python -m piper.download_voices en_US-lessac-medium
```

This puts `en_US-lessac-medium.onnx` in the current folder. **Run `assistant.py` from the same folder** (the repo root). The `.onnx` files are git-ignored.

## 3. Ollama and the model

1. Install Ollama from https://ollama.com and make sure it is running (it starts a local server on port 11434).
2. Pull a model that supports tool calling:

```powershell
ollama pull llama3.1:8b
```

Ollama uses the NVIDIA GPU automatically when a recent driver is installed. Confirm with `ollama ps` while a model is loaded: the PROCESSOR column should say `100% GPU`.

### Picking a model for 8 GB of VRAM

Everything shares the 4060's memory, so budget it:

| Piece | Approx. VRAM |
| --- | --- |
| Whisper `small.en` (float16) | about 1 GB |
| 8B LLM (4-bit, e.g. `llama3.1:8b`) | about 5 GB |
| Piper | CPU only, 0 |

That fits with some headroom. If you hit out-of-memory errors or slowdowns:

- use a smaller LLM: `OLLAMA_MODEL=llama3.2:3b`
- or keep Whisper small and don't move up to `medium.en`

If you have headroom, `WHISPER_SIZE=medium.en` improves ticker and name accuracy.

## 4. GPU for faster-whisper

faster-whisper runs on CUDA through CTranslate2, which needs the CUDA 12 cuBLAS and cuDNN 9 libraries:

```powershell
pip install -r software/requirements-gpu.txt
```

`assistant.py` finds those libraries automatically (it adds the pip package folders to the DLL search path) and prints one of:

- `Whisper 'small.en' running on GPU (CUDA, float16).`
- `(GPU Whisper unavailable, falling back to CPU: ...)` followed by a CPU line. Everything still works, just slower. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

Force a device with `WHISPER_DEVICE=cuda` (fail loudly if the GPU isn't usable) or `WHISPER_DEVICE=cpu`.

Piper stays on the CPU on purpose. It is already faster than real time there, and that leaves the GPU memory for Whisper and the LLM.

## 5. Run it

```powershell
python software/assistant.py
```

| Option | Effect |
| --- | --- |
| `--text` | Type questions instead of speaking |
| `--demo` | Use sample data, no internet |
| `OLLAMA_MODEL=...` | Choose the LLM |
| `WHISPER_SIZE=...` | Choose the Whisper model |
| `PIPER_VOICE=...` | Choose the Piper voice |

PowerShell example: `$env:OLLAMA_MODEL="qwen2.5:7b"; python software/assistant.py`

## 6. Suggested first tests

1. `python software/finance_tools.py` with `$env:DEMO_MODE="1"` prints sample quotes, which proves the tools load.
2. `python software/assistant.py --text --demo` checks Ollama and tool calling with no mic.
3. `python software/assistant.py --demo` adds Whisper and Piper.
4. `python software/assistant.py` uses live yfinance data.

Try: "How is Nvidia doing today?", "How has Tesla done this year?", "How's the market?", "What is a P/E ratio?", "Should I buy Apple?" (it should decline to give advice).

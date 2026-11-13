# Troubleshooting

## Whisper falls back to CPU

The startup line says `GPU Whisper unavailable, falling back to CPU` followed by an error.

- **`Could not load library cudnn...` or `cublas64_12.dll not found`**: install the CUDA libraries with `pip install -r software/requirements-gpu.txt` in the same venv.
- **Version mismatch errors**: the pip CUDA packages must match what your installed CTranslate2 build expects (CUDA 12 and cuDNN 9 for current CTranslate2 4.x). Check the CTranslate2 release notes for your version.
- **`CUDA failed` or no device**: update the NVIDIA driver and check `nvidia-smi` works.
- To see the real error instead of a silent fallback, run with `WHISPER_DEVICE=cuda`.

## Out of GPU memory or slow responses

Whisper and the LLM share the 4060's 8 GB. Try `OLLAMA_MODEL=llama3.2:3b`, keep `WHISPER_SIZE=small.en`, and close other GPU-heavy apps (games, browsers with hardware acceleration). `ollama ps` shows whether the model is fully on the GPU.

## `[LLM error]` / connection refused

Ollama isn't running, or the model isn't pulled. Start Ollama, then run `ollama pull llama3.1:8b` (or whatever `OLLAMA_MODEL` is set to).

## The model never calls tools or invents prices

Use a model that supports tool calling (`llama3.1:8b`, `qwen2.5:7b` and `llama3.2:3b` do). Very small models are unreliable at it. Test with `python software/assistant.py --text --demo` and watch for `[tool] ...` lines.

## Whisper mishears tickers or company names

Move up to `WHISPER_SIZE=medium.en` if VRAM allows. Add the names you care about to `WHISPER_HINT` in `assistant.py` and to `NAME_TO_TICKER` in `finance_tools.py`.

## yfinance returns errors or empty data

- Check your internet connection. Without it the tools fall back to demo data and say so.
- yfinance scrapes Yahoo and breaks occasionally: `pip install -U yfinance`.
- Quotes can be delayed up to 15 minutes and are empty when markets are closed for some fields.

## Piper messages

- `Piper Python API unavailable ... using CLI fallback`: it still works, just slower per sentence (the CLI reloads the voice each time). This usually means the `.onnx` file isn't in the folder you ran from. Run from the repo root or re-run `python -m piper.download_voices en_US-lessac-medium` there.
- No sound: check the default output device in Windows sound settings.

## It hears itself

The mic is idle while the reply plays. If you still get echo, use headphones until the puck's hardware handles this.

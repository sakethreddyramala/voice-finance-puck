# voice-finance-puck

A desk puck you can talk to: hold a button, ask about a stock or the market, and it answers out loud. The AI runs fully local on a laptop, with no cloud, and the puck is a custom ESP32-S3 circuit board with its own microphone and speaker.

## How it works

Voice -> speech-to-text (faster-whisper, GPU) -> local LLM (Ollama) with finance tools (yfinance) -> text-to-speech (Piper) -> speaker

The LLM's reply is streamed into Piper one sentence at a time, so the puck starts talking before the model has finished writing. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Status

- [x] Step 0: Setup and checks
- [x] Step 1: Talking assistant on the laptop
- [ ] Step 2: Finance tools and demo mode (code written; needs testing on the RTX 4060 with real mic, Ollama and yfinance)
- [ ] Step 3: Puck prototype on a dev board
- [ ] Step 4: Custom circuit board
- [ ] Step 5: Enclosure and demo prep

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r software/requirements.txt
pip install -r software/requirements-gpu.txt     # NVIDIA GPU only
python -m piper.download_voices en_US-lessac-medium
ollama pull llama3.1:8b
python software/assistant.py
```

Press Enter to start talking, Enter again to stop. Full instructions, including GPU setup: [docs/SETUP.md](docs/SETUP.md).

Useful flags:

| Command | What it does |
| --- | --- |
| `python software/assistant.py --text` | Type instead of speak. Good for testing the LLM and tools without a mic. |
| `python software/assistant.py --demo` | Use sample data from `software/demo_data.json`. No internet needed. |

## What it can answer

- Current price and today's change for a stock, ETF or index
- Performance over a period (5 days, 1 month, 3 months, 6 months, 1 year, year to date)
- Basic company info (sector, market cap, P/E)
- How the overall market is doing (S&P 500, Nasdaq, Dow, VIX)
- General finance concepts, explained without any data lookup

## Files

| File | Purpose |
| --- | --- |
| `software/assistant.py` | Main loop: mic, Whisper, LLM, Piper, speaker |
| `software/llm.py` | Ollama streaming chat and the tool-calling loop |
| `software/finance_tools.py` | yfinance tools, ticker lookup, demo-mode fallback |
| `software/demo_data.json` | Made-up sample prices for demo mode |
| `software/transcribe.py` | Original 5-second mic and Whisper test |
| `software/speak.py` | Original Piper test |

## Notes

The assistant only states prices that a tool just returned. It explains and summarizes. It does not give buy or sell advice.

Quotes from yfinance can be delayed up to 15 minutes. Demo mode data is made up and is always announced as demo data.

Problems? See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

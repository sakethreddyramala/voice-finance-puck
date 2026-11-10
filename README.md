# voice-finance-puck

A desk puck you can talk to: hold a button, ask about a stock or the market, and it answers out loud. The AI runs fully local on a laptop, with no cloud, and the puck is a custom ESP32-S3 circuit board with its own microphone and speaker.

## How it works

Voice -> speech-to-text (faster-whisper) -> local LLM (Ollama) with finance tools -> text-to-speech (Piper) -> speaker

## Status

- [x] Step 0: Setup and checks
- [x] Step 1: Talking assistant on the laptop
- [ ] Step 2: Finance tools and demo mode
- [ ] Step 3: Puck prototype on a dev board
- [ ] Step 4: Custom circuit board
- [ ] Step 5: Enclosure and demo prep

## Setup

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r software/requirements.txt
python -m piper.download_voices en_US-lessac-medium
python software/transcribe.py

## Notes

The assistant only states prices that a tool just returned. It explains and summarizes. It does not give buy or sell advice.

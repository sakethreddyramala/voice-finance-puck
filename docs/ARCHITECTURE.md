# Architecture

## Pipeline

```
mic -> faster-whisper (GPU) -> text
                                 |
                                 v
                        Ollama LLM (GPU) <-> finance tools (yfinance / demo data)
                                 |
                                 v  streamed tokens, split into sentences
                          Piper TTS (CPU) -> speaker
```

## Components

**`assistant.py`** owns the audio. It records from the mic (Enter to start and stop for now), transcribes with faster-whisper, hands the text to the LLM, and plays Piper audio from a queue on a background thread. It waits for playback to finish before listening again so the speaker doesn't feed back into the mic.

**`llm.py`** talks to Ollama's `/api/chat` endpoint with streaming on. Only the standard library is used. As tokens arrive they are split into sentences and passed to a callback, which feeds Piper. If the model asks for a tool, the loop runs it, appends the result, and calls the model again (up to 4 rounds) until it produces a final answer.

**`finance_tools.py`** defines four tools and their JSON schemas for Ollama:

| Tool | Use |
| --- | --- |
| `get_stock_quote` | Price and today's change |
| `get_price_history` | Percent change, high and low over a period |
| `get_company_info` | Name, sector, market cap, P/E |
| `get_market_overview` | S&P 500, Nasdaq, Dow, VIX |

Spoken company names ("Nvidia") are mapped to tickers ("NVDA") before lookup. Tools never raise. Errors come back as `{"error": ...}` so the LLM can say it couldn't get the data.

## Demo mode

If `--demo` / `DEMO_MODE=1` is set, or a live yfinance call fails (no internet, rate limited, unknown symbol), tools fall back to `software/demo_data.json`. Those results carry `"demo": true` and the system prompt tells the model to say the number is demo data. Demo prices are made up and never presented as real.

## Honesty rules

These live in the system prompt in `llm.py`:

1. Only state a number a tool returned in this conversation.
2. Call a tool for any price, performance, size or market question.
3. If a tool errors, say so instead of guessing.
4. Say when data is demo data.
5. No buy, sell or hold advice and no price predictions.

## Memory

Conversation memory keeps the last 6 user/assistant exchanges as plain text. Tool calls and results are used within a turn and then dropped, so old numbers are never reused as if they were current.

## Latency notes

- Whisper and the LLM stay loaded between turns (`keep_alive` is 30 minutes).
- Sentence-level streaming means the first sentence is spoken while the rest is still being generated.
- A tool call adds one extra LLM round trip plus the yfinance request.

## Planned: the puck (Steps 3 to 5)

The laptop keeps running Whisper, the LLM and Piper. The ESP32-S3 handles the button, microphone and speaker. The pieces still to design are the audio link between puck and laptop (likely Wi-Fi streaming) and hold-to-talk handling that replaces the Enter key.

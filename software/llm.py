"""Ollama chat with streaming and tool calling. Standard library only.

ask() streams the model's reply, hands each finished sentence to a callback
(so Piper can start talking early), and runs tool calls in a loop until the
model gives a final answer.
"""
import json
import os
import re
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")  # keep the model in VRAM between turns
MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = (
    "You are a voice assistant on a desk puck that answers finance questions. "
    "Your replies are spoken aloud, so use one to three short sentences of plain "
    "spoken English: no markdown, no lists, no emojis, no ticker symbols with "
    "dollar signs. Say prices naturally, like 'one hundred forty two dollars and "
    "fifty cents', and say 'percent' instead of the symbol. "
    "RULES: (1) Only state a price, percentage, or market figure that a tool just "
    "returned in this conversation. Never guess or use memory for numbers. "
    "(2) Call a tool for any question about prices, performance, company size, or "
    "the market. If a tool returns an error, say you couldn't get that data. "
    "(3) If a tool result has demo true, say the figure is demo data. "
    "(4) Explain and summarize, but never give buy, sell, or hold advice or price "
    "predictions. If asked, say you can't give investment advice and offer facts "
    "instead. (5) General finance concepts you can explain without a tool."
)

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def clean_for_speech(text):
    return re.sub(r"[*_`#>]+", "", text).strip()


def _stream(messages, tools):
    """Yield parsed JSON chunks from one streaming Ollama call."""
    body = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0.2},
    }
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.strip()
            if line:
                yield json.loads(line)


def ask(history, on_sentence, tools=None, run_tool=None, on_tool=None, system=SYSTEM_PROMPT):
    """Answer the latest user message in `history` (a list of user/assistant text dicts).

    on_sentence(text): called for each finished sentence of spoken output.
    on_tool(name, args): optional, for logging tool calls.
    Returns the final reply text. `history` is not modified.
    """
    messages = [{"role": "system", "content": system}] + list(history)
    spoken_total = ""

    for _ in range(MAX_TOOL_ROUNDS + 1):
        content, buf, tool_calls = "", "", []
        for obj in _stream(messages, tools):
            msg = obj.get("message", {})
            token = msg.get("content", "")
            if token:
                print(token, end="", flush=True)
                content += token
                buf += token
                parts = SENTENCE_END.split(buf)
                if len(parts) > 1:
                    for sentence in parts[:-1]:
                        s = clean_for_speech(sentence)
                        if s:
                            on_sentence(s)
                    buf = parts[-1]
            for tc in msg.get("tool_calls") or []:
                tool_calls.append(tc)
            if obj.get("done"):
                break

        tail = clean_for_speech(buf)
        if tail:
            on_sentence(tail)
        spoken_total += content

        if not tool_calls or run_tool is None:
            print()
            return spoken_total.strip()

        # Record the assistant's tool request, run each tool, feed results back.
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if on_tool:
                on_tool(name, args)
            result = run_tool(name, args)
            messages.append({"role": "tool", "tool_name": name, "content": result})

    msg = "Sorry, I couldn't finish looking that up."
    on_sentence(msg)
    print(msg)
    return msg

"""Ollama chat with streaming, forced tool use and a number guard. Standard library only.

ask() works in three steps:

1. ROUTE. A rule-based router (finance_tools.route) classifies the message:
   chit-chat, market screener, finance concept, or "needs data". Small local models are
   unreliable at deciding to call tools, so for data questions the tools are run in code
   BEFORE the model speaks, and their results are put in the conversation.
2. GENERATE. The model streams its reply. Each finished sentence goes through the number
   guard before it is printed or spoken.
3. GUARD. Any number in a sentence that no tool returned this turn (digits or spelled-out
   words) blocks that sentence. The user then hears a summary built directly from the tool
   results instead, so a guessed price is never said out loud.
"""
import json
import math
import os
import re
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")  # keep the model in VRAM between turns
MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = (
    "You are a friendly voice assistant on a desk puck. Your specialty is finance, but you "
    "can chat too. Your replies are spoken aloud, so use one to three short sentences of "
    "plain spoken English: no markdown, no lists, no emojis. "
    "CHIT-CHAT: for greetings, small talk, thanks, jokes, or questions about you (for example "
    "'how are you doing today?'), just reply naturally like a friendly person, for example "
    "'Doing great, thanks for asking! What can I help you with?'. Never mention markets, "
    "tools, or data unless the person asked about them, and never say you are looking "
    "something up. "
    "NUMBERS: you may only say a price, percentage, index level, market cap, or ratio that "
    "appears in a tool result in this conversation. Never use memory or guess a number. If "
    "you don't have a number, say you would need to look it up. Write numbers as plain "
    "digits with the unit as a word, like '189.43 dollars', '2.1 percent', or '4.6 trillion "
    "dollars'. No dollar signs and no percent signs. "
    "DATA: say which ticker each number belongs to. If a tool result has demo true, say it "
    "is demo data and not real prices. If a tool result has an error, say you couldn't get "
    "that data. "
    "ADVICE: never give buy, sell, or hold advice, price predictions, or opinions on whether "
    "something is undervalued or overvalued. If asked, say you can't give investment advice "
    "and offer the facts instead. "
    "SCOPE: you can only check specific tickers or company names and the major indexes "
    "(S and P 500, Nasdaq, Dow, VIX). You cannot screen or rank the whole market. "
    "You can explain general finance concepts in plain words without specific figures."
)

# Extra instruction appended to the system prompt depending on how the router classified the message.
INTENT_NOTES = {
    "chat": "This message is casual conversation. No finance data is needed. Reply briefly and naturally.",
    "general": "This is a general finance concept question. Explain it simply, without specific numbers.",
    "data": ("The tool results for this question are already in the conversation. Answer only "
             "from them, using their exact numbers."),
}

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def clean_for_speech(text):
    return re.sub(r"[*_`#>]+", "", text).strip()


# ---- number guard -------------------------------------------------------------
_ONES = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_BIG = ("hundred",) + tuple(_SCALES)

# Things that look like numbers but are names or durations, not market figures.
_MASK = re.compile(
    r"s\s*&\s*p\s*500|nasdaq\s*-?\s*100|dow\s*30|russell\s*(?:1000|2000)|401\s*\(?k\)?|"
    r"\b\d+(?:\.\d+)?\s*-?\s*(?:day|week|month|year|quarter|hour|minute)s?\b", re.IGNORECASE)

_CLAIM = re.compile(
    r"(?P<money>\$)?\s*(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<dec>\d+))?\s*"
    r"(?P<unit>%|percent\b|dollars?\b|cents?\b|points?\b|thousand\b|million\b|billion\b|trillion\b)?",
    re.IGNORECASE)


def _words_int(words):
    total, cur = 0, 0
    for w in words:
        if w in _ONES:
            cur += _ONES[w]
        elif w in _TENS:
            cur += _TENS[w]
        elif w == "hundred":
            cur = max(cur, 1) * 100
        elif w in _SCALES:
            total += max(cur, 1) * _SCALES[w]
            cur = 0
    return int(total + cur)


def _run_to_text(run):
    """['one','hundred','forty','two'] -> '142'; ['four','point','six','trillion'] -> '4.6 trillion'."""
    run = list(run)
    scale = run.pop() if run and run[-1] in _SCALES else None
    if "point" in run:
        k = run.index("point")
        head, tail = run[:k], run[k + 1:]
    else:
        head, tail = run, []
    text = str(_words_int(head) if head else 1)
    if tail:
        text += "." + "".join(str(_ONES.get(w, _TENS.get(w, 0))) for w in tail)
    return text + (f" {scale}" if scale else "")


def _normalize_number_words(text):
    """Rewrite spelled-out numbers as digits so 'one hundred forty two dollars' is checked too."""
    toks = re.findall(r"[A-Za-z']+|\d[\d,]*(?:\.\d+)?|\S", text)
    low = [t.lower() for t in toks]
    is_num = lambda w: w in _ONES or w in _TENS or w in _BIG
    after_digits = lambda: bool(out) and re.fullmatch(r"\d[\d,]*(?:\.\d+)?", out[-1]) is not None
    out, i = [], 0
    while i < len(toks):
        # A scale word right after digits ("4.6 trillion") is a unit, not a new spelled-out number.
        if not is_num(low[i]) or (low[i] in _BIG and after_digits()):
            out.append(toks[i])
            i += 1
            continue
        run, j = [], i
        while j < len(toks):
            w = low[j]
            if is_num(w):
                run.append(w)
            elif w == "and" and run and run[-1] in _BIG and j + 1 < len(toks) and is_num(low[j + 1]):
                pass  # "one hundred AND forty two"
            elif w == "point" and run and j + 1 < len(toks) and low[j + 1] in _ONES:
                run.append(w)
            else:
                break
            j += 1
        out.append(_run_to_text(run))
        i = j
    return " ".join(out)


def _numbers_in(obj, out):
    """Collect every number in a tool result (any nesting) into `out`."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers_in(v, out)


def _supported(value, dec, unit, money, allowed):
    unit = (unit or "").lower()
    if unit.startswith("cent"):  # "forty three cents" must match the cents part of some tool number
        return any(round((abs(a) % 1) * 100) == round(value) for a in allowed)
    scale = _SCALES.get(unit, 1)
    if not unit and not money and dec == 0 and (value <= 10 or 1900 <= value <= 2100):
        return True  # "one of", "three sentences", "in 2008" are not market figures
    q = 10 ** dec
    for a in allowed:
        x = abs(a) / scale
        # Same number rounded OR truncated to the precision the model used.
        if abs(round(x, dec) - value) < 1e-9 or abs(math.floor(x * q) / q - value) < 1e-9:
            return True
    return False


def unsupported_numbers(text, allowed):
    """Numbers in `text` that do not match any number in `allowed` (the tool results)."""
    norm = _MASK.sub(" ", _normalize_number_words(text))
    bad = []
    for m in _CLAIM.finditer(norm):
        dec = m.group("dec") or ""
        value = float(m.group("int").replace(",", "") + ("." + dec if dec else ""))
        if not _supported(value, len(dec), m.group("unit"), bool(m.group("money")), allowed):
            bad.append(m.group(0).strip())
    return bad


# ---- Ollama -------------------------------------------------------------------
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


def _generate(messages, tools, allowed, speak, log):
    """Stream one model reply. Each sentence is number-checked before it is printed or spoken.

    speak(sentence) prints and speaks one approved sentence.
    Returns (spoken_text, tool_calls, blocked). If blocked is True the model tried to say a
    number no tool returned; generation was stopped and nothing unverified was spoken.
    """
    spoken, buf, tool_calls, blocked = [], "", [], False

    def emit(sentence):
        s = clean_for_speech(sentence)
        if not s:
            return True
        bad = unsupported_numbers(s, allowed)
        if bad:
            log(f"[guard] blocked unverified number(s) {bad} in: {s!r}")
            return False
        speak(s)
        spoken.append(s)
        return True

    stream = _stream(messages, tools)
    try:
        for obj in stream:
            msg = obj.get("message", {})
            token = msg.get("content", "")
            if token:
                buf += token
                parts = SENTENCE_END.split(buf)
                if len(parts) > 1:
                    for sentence in parts[:-1]:
                        if not emit(sentence):
                            blocked = True
                            break
                    buf = parts[-1]
                    if blocked:
                        break
            for tc in msg.get("tool_calls") or []:
                tool_calls.append(tc)
            if obj.get("done"):
                break
    finally:
        stream.close()

    if not blocked and not emit(buf):
        blocked = True
    return " ".join(spoken), tool_calls, blocked


def _loads(raw):
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _fallback_text(results, summarize, spoken_any):
    """What to say when the guard rejected the model's wording."""
    if results and summarize:
        facts = " ".join(summarize(name, args, data) for name, args, data in results)
        return ("Let me give you the verified numbers. " if spoken_any else "") + facts
    if spoken_any:
        return "Sorry, I can't verify the rest of that."
    return ("Sorry, I don't have a verified number for that. Ask me about a specific ticker "
            "and I'll look it up.")


def ask(history, on_sentence, tools=None, run_tool=None, on_tool=None, system=SYSTEM_PROMPT,
        route=None, summarize=None, log=None):
    """Answer the latest user message in `history` (a list of user/assistant text dicts).

    on_sentence(text): called for each finished, number-checked sentence of spoken output.
    on_tool(name, args): optional, for logging tool calls.
    route(text, history) -> plan: optional router (finance_tools.route). Without it the
        model decides about tool calls itself (the old behaviour), but the number guard still runs.
    summarize(name, args, result) -> str: builds a spoken sentence from a tool result; used
        if the guard blocks the model's wording (finance_tools.summarize_result).
    Returns the text that was actually spoken. `history` is not modified.
    """
    out_log = log or (lambda m: print(m, flush=True))
    mid_line = [True]  # assistant.py has already printed "Assistant: " without a newline

    def log(msg):  # start log lines on a fresh terminal line, never in the middle of a sentence
        if mid_line[0]:
            print()
            mid_line[0] = False
        out_log(msg)

    def speak(sentence):
        print(sentence + " ", end="", flush=True)
        mid_line[0] = True
        on_sentence(sentence)

    def say(text):
        for s in SENTENCE_END.split(text):
            if s.strip():
                speak(s.strip())

    user_text = next((m.get("content", "") for m in reversed(history) if m.get("role") == "user"), "")
    plan = route(user_text, list(history)[:-1]) if route else {"intent": "auto"}
    intent = plan.get("intent", "auto")
    if intent == "data" and run_tool is None:
        intent = "auto"  # can't force tools without a way to run them
    planned = [n for n, _ in plan.get("calls", [])]
    log(f"[router] intent={intent}" + (f" tools={planned}" if planned else ""))

    if "reply" in plan:  # screener / clarify: a fixed answer is better than asking the model
        say(plan["reply"])
        print()
        return plan["reply"]

    system_text = (system + " " + INTENT_NOTES.get(intent, "")).strip()
    messages = [{"role": "system", "content": system_text}] + list(history)
    results, allowed = [], []
    use_tools = tools if intent == "auto" else None

    def run_one(name, args):
        if on_tool:
            on_tool(name, args)
        raw = run_tool(name, args)
        data = _loads(raw)
        results.append((name, args, data))
        _numbers_in(data, allowed)
        return raw

    if intent == "data":
        calls = plan["calls"]
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": n, "arguments": a}} for n, a in calls]})
        for name, args in calls:
            messages.append({"role": "tool", "tool_name": name, "content": run_one(name, args)})

    spoken = []
    for _ in range(MAX_TOOL_ROUNDS + 1):
        text, tool_calls, blocked = _generate(messages, use_tools, allowed, speak, log)
        if text:
            spoken.append(text)
        if blocked:
            fallback = _fallback_text(results, summarize, bool(spoken))
            log("[guard] answering from the tool results instead")
            say(fallback)
            spoken.append(fallback)
            break
        if not tool_calls or run_tool is None or use_tools is None:
            break
        # Model-requested tools (only when no router is in use): run them and go around again.
        messages.append({"role": "assistant", "content": text, "tool_calls": tool_calls})
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            messages.append({"role": "tool", "tool_name": name, "content": run_one(name, args)})
    else:
        msg = "Sorry, I couldn't finish looking that up."
        say(msg)
        spoken.append(msg)

    final = " ".join(spoken).strip()
    if any(d.get("demo") for _, _, d in results) and "demo" not in final.lower():
        note = "Note that this is demo data, not real market prices."
        log("[guard] added demo-data disclosure")
        say(note)
        final += " " + note
    print()
    return final

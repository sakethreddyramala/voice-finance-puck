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
    "NEVER SAY 'LET ME CHECK': all tool lookups have already finished before you speak. "
    "Never say 'let me check', 'one moment', 'now let me look up', or promise to check "
    "another stock. Answer only with what you have right now. "
    "CHIT-CHAT: for greetings, small talk, thanks, jokes, or questions about you (for example "
    "'how are you doing today?'), just reply naturally like a friendly person, for example "
    "'Doing great, thanks for asking! What can I help you with?'. Never mention markets, "
    "tools, or data unless the person asked about them. "
    "NUMBERS: you may only say a price, percentage, index level, market cap, or ratio that "
    "appears in a tool result in this conversation. Never use memory or guess a number. "
    "Write numbers in natural spoken form: say 'down 3.72 percent' instead of '-3.72 percent', "
    "'up 2.1 percent' instead of '+2.1%', and '20 dollars and 31 cents' instead of '$20.31'. "
    "Never output minus signs, plus signs, dollar signs, or percent symbols. "
    "DATA: say which ticker each number belongs to. If a tool result has demo true, say it "
    "is demo data and not real prices. If a tool result has an error, say you couldn't get "
    "that data. "
    "ADVICE: never give buy, sell, or hold advice, price predictions, or opinions on whether "
    "something is undervalued or overvalued. If asked, say you can't give investment advice "
    "and offer the facts instead. "
    "SCOPE: you can only check specific tickers or company names and the major indexes "
    "(S and P 500, Nasdaq, Dow, VIX). You cannot screen or rank the whole market. "
    "You can explain general finance concepts in plain words without specific market prices."
)

# Extra instruction appended to the system prompt depending on how the router classified the message.
INTENT_NOTES = {
    "chat": "This message is casual conversation. No finance data is needed. Reply briefly and naturally.",
    "general": "This is a general finance concept question. Explain it clearly and concisely without quoting live market prices.",
    "data": ("The tool results for this question are already in the conversation. Answer ONLY "
             "using the tool results provided. Do not promise to look up anything else."),
}

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

_DEFERRED_LOOKUP = re.compile(
    r"\b(?:now,?\s*)?(?:let me|i'll|i will)\s+(?:check|look up|pull up|fetch|see)\b",
    re.IGNORECASE,
)


def _format_money(m):
    dollars = m.group(1).replace(",", "")
    cents = m.group(2)
    d_word = "dollar" if dollars == "1" else "dollars"
    if cents and int(cents) > 0:
        c_val = int(cents)
        c_word = "cent" if c_val == 1 else "cents"
        return f"{dollars} {d_word} and {c_val} {c_word}"
    return f"{dollars} {d_word}"


def clean_for_speech(text):
    """Normalize symbols, negative signs, and currency so Piper TTS reads smoothly."""
    s = re.sub(r"[*_`#>]+", "", text)
    s = s.replace("&", " and ")
    # $20.31 -> 20 dollars and 31 cents
    s = re.sub(r"\$(\d[\d,]*)(?:\.(\d{2}))?\b", _format_money, s)
    # -3.72% or -3.72 percent -> down 3.72 percent
    s = re.sub(r"(?<!\w)-\s*(\d+(?:\.\d+)?)\s*(?:%|percent\b)", r"down \1 percent", s, flags=re.IGNORECASE)
    # +3.72% or +3.72 percent -> up 3.72 percent
    s = re.sub(r"(?<!\w)\+\s*(\d+(?:\.\d+)?)\s*(?:%|percent\b)", r"up \1 percent", s, flags=re.IGNORECASE)
    # Standalone 3.72% -> 3.72 percent
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1 percent", s)
    # Leading minus sign on other numbers: -20.31 -> minus 20.31
    s = re.sub(r"(?<!\w)-\s*(\d+(?:\.\d+)?)", r"minus \1", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


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
    r"s\s*(?:&|and)\s*p\s*500|nasdaq\s*-?\s*100|dow\s*30|russell\s*(?:1000|2000)|401\s*\(?k\)?|"
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
                pass
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


def _numbers_in_text(text):
    """Extract numeric values mentioned in the user's own prompt so they can be echoed safely."""
    norm = _normalize_number_words(text)
    vals = []
    for m in _CLAIM.finditer(norm):
        dec = m.group("dec") or ""
        vals.append(float(m.group("int").replace(",", "") + ("." + dec if dec else "")))
    return vals


def _supported(value, dec, unit, money, allowed, allow_general_counts=False):
    unit = (unit or "").lower()
    if unit.startswith("cent"):
        return any(round((abs(a) % 1) * 100) == round(value) for a in allowed)
    scale = _SCALES.get(unit, 1)
    if not unit and not money and dec == 0 and (value <= 10 or 1900 <= value <= 2100):
        return True
    # In general finance explanations ("an index of 500 companies"), allow plain unitless integers.
    if allow_general_counts and not unit and not money and dec == 0:
        return True
    q = 10 ** dec
    for a in allowed:
        x = abs(a) / scale
        if abs(round(x, dec) - value) < 1e-9 or abs(math.floor(x * q) / q - value) < 1e-9:
            return True
    return False


def unsupported_numbers(text, allowed, allow_general_counts=False):
    """Numbers in `text` that do not match any number in `allowed`."""
    norm = _MASK.sub(" ", _normalize_number_words(text))
    bad = []
    for m in _CLAIM.finditer(norm):
        dec = m.group("dec") or ""
        value = float(m.group("int").replace(",", "") + ("." + dec if dec else ""))
        if not _supported(value, len(dec), m.group("unit"), bool(m.group("money")), allowed, allow_general_counts):
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


def _generate(messages, tools, allowed, speak, log, allow_general_counts=False):
    """Stream one model reply. Each sentence is number-checked before it is printed or spoken."""
    spoken, buf, tool_calls, blocked = [], "", [], False

    def emit(sentence):
        s = clean_for_speech(sentence)
        if not s:
            return True
        # Drop trailing filler like "Now, let me check Micron." so the assistant doesn't hang.
        if _DEFERRED_LOOKUP.search(s):
            log(f"[guard] stripped deferred-lookup sentence: {s!r}")
            return True
        bad = unsupported_numbers(s, allowed, allow_general_counts=allow_general_counts)
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
        facts = " ".join(clean_for_speech(summarize(name, args, data)) for name, args, data in results)
        return ("Let me give you the verified numbers. " if spoken_any else "") + facts
    if spoken_any:
        return "Sorry, I can't verify the rest of that."
    return ("Sorry, I don't have a verified number for that. Ask me about a specific ticker "
            "and I'll look it up.")


def ask(history, on_sentence, tools=None, run_tool=None, on_tool=None, system=SYSTEM_PROMPT,
        route=None, summarize=None, log=None):
    """Answer the latest user message in `history` (a list of user/assistant text dicts)."""
    out_log = log or (lambda m: print(m, flush=True))
    mid_line = [True]

    def log(msg):
        if mid_line[0]:
            print()
            mid_line[0] = False
        out_log(msg)

    def speak(sentence):
        cleaned = clean_for_speech(sentence)
        if not cleaned:
            return
        print(cleaned + " ", end="", flush=True)
        mid_line[0] = True
        on_sentence(cleaned)

    def say(text):
        for s in SENTENCE_END.split(text):
            if s.strip():
                speak(s.strip())

    user_text = next((m.get("content", "") for m in reversed(history) if m.get("role") == "user"), "")
    plan = route(user_text, list(history)[:-1]) if route else {"intent": "auto"}
    intent = plan.get("intent", "auto")
    if intent == "data" and run_tool is None:
        intent = "auto"
    planned = [n for n, _ in plan.get("calls", [])]
    log(f"[router] intent={intent}" + (f" tools={planned}" if planned else ""))

    if "reply" in plan:
        say(plan["reply"])
        print()
        return clean_for_speech(plan["reply"])

    system_text = (system + " " + INTENT_NOTES.get(intent, "")).strip()
    messages = [{"role": "system", "content": system_text}] + list(history)
    # Seed allowed numbers with any numbers the user themselves said in their prompt.
    results, allowed = [], _numbers_in_text(user_text)
    use_tools = tools if intent == "auto" else None
    allow_general_counts = (intent == "general")

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
        text, tool_calls, blocked = _generate(
            messages, use_tools, allowed, speak, log, allow_general_counts=allow_general_counts
        )
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

"""Finance tools the local LLM can call, backed by yfinance.

Every tool returns a small JSON-serializable dict. If yfinance is missing, the
network is down, or DEMO_MODE=1 is set, tools fall back to sample data from
demo_data.json and mark the result with "demo": true so the assistant says so.

The assistant must only state numbers returned by these tools.

This file also holds route(), a small rule-based router that decides, BEFORE the
LLM runs, whether a message is chit-chat, a market screener request, a question
that needs live data (and exactly which tools to run), or a general finance
concept question. Small local models are unreliable at deciding to call tools,
so the data fetch is done in code instead of being left to the model.

Set FINANCE_LOG=0 to silence the [finance] log lines.
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path

DEMO_FILE = Path(__file__).with_name("demo_data.json")
FORCE_DEMO = os.environ.get("DEMO_MODE", "0") == "1"
LOG_TOOLS = os.environ.get("FINANCE_LOG", "1") != "0"
MAX_TICKERS = 5

# Spoken company names -> tickers. Whisper hears names, yfinance needs tickers.
NAME_TO_TICKER = {
    "nvidia": "NVDA", "apple": "AAPL", "microsoft": "MSFT", "tesla": "TSLA",
    "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL", "meta": "META",
    "facebook": "META", "netflix": "NFLX", "amd": "AMD", "intel": "INTC",
    "palantir": "PLTR", "coinbase": "COIN", "disney": "DIS", "boeing": "BA",
    "jpmorgan": "JPM", "jp morgan": "JPM", "goldman sachs": "GS",
    "berkshire": "BRK-B", "walmart": "WMT", "costco": "COST", "visa": "V",
    "mastercard": "MA", "coca cola": "KO", "coca-cola": "KO", "pepsi": "PEP",
    "nike": "NKE", "starbucks": "SBUX", "uber": "UBER", "broadcom": "AVGO",
    "tsmc": "TSM", "oracle": "ORCL", "salesforce": "CRM", "adobe": "ADBE",
    "bitcoin": "BTC-USD", "ethereum": "ETH-USD", "gold": "GC=F", "oil": "CL=F",
}

INDEX_SYMBOLS = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "VIX": "^VIX",
}

_demo_cache = None


def _log(msg):
    """Terminal log so you can see exactly when yfinance runs and what came back."""
    if LOG_TOOLS:
        print(f"[finance] {msg}", flush=True)


def _demo():
    global _demo_cache
    if _demo_cache is None:
        _demo_cache = json.loads(DEMO_FILE.read_text(encoding="utf-8"))
    return _demo_cache


def resolve_symbol(text):
    """'nvidia' -> 'NVDA', 'nvda' -> 'NVDA'."""
    t = (text or "").strip().lower().replace(".", "")
    if t in NAME_TO_TICKER:
        return NAME_TO_TICKER[t]
    return (text or "").strip().upper().replace(" ", "")


def _yf():
    """Import yfinance lazily so demo mode works without it."""
    import yfinance as yf
    return yf


def _round(x, n=2):
    return None if x is None else round(float(x), n)


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ---- tools ------------------------------------------------------------------
def get_stock_quote(symbol):
    sym = resolve_symbol(symbol)
    if not FORCE_DEMO:
        _log(f"yfinance: fetching quote for {sym} ...")
        try:
            fi = _yf().Ticker(sym).fast_info
            price = fi["last_price"]
            prev = fi["previous_close"]
            if price is None or prev is None:
                raise ValueError("no price data")
            out = {
                "symbol": sym,
                "price": _round(price),
                "previous_close": _round(prev),
                "change": _round(price - prev),
                "change_percent": _round((price - prev) / prev * 100),
                "currency": fi["currency"] if "currency" in fi else "USD",
                "as_of": _now(),
                "note": "Quotes can be delayed up to 15 minutes.",
            }
            _log(f"yfinance OK (LIVE): {sym} price={out['price']} change={out['change_percent']}%")
            return out
        except Exception as e:
            err = str(e)
            _log(f"yfinance FAILED for {sym}: {err}")
    else:
        err = "demo mode"
    q = _demo()["quotes"].get(sym)
    if q is None:
        _log(f"no data for {sym} (live failed, not in demo data)")
        return {"error": f"No data for {sym}. Live lookup failed ({err}) and it is not in demo data."}
    _log(f"using DEMO data for {sym} (made-up sample numbers, not real prices)")
    return {"symbol": sym, **q, "currency": "USD", "demo": True,
            "note": "DEMO DATA, not real market prices."}


def get_price_history(symbol, period="1mo"):
    """Percent change over a period: 5d, 1mo, 3mo, 6mo, 1y, ytd."""
    sym = resolve_symbol(symbol)
    allowed = {"5d", "1mo", "3mo", "6mo", "1y", "ytd"}
    if period not in allowed:
        period = "1mo"
    if not FORCE_DEMO:
        _log(f"yfinance: fetching {period} history for {sym} ...")
        try:
            hist = _yf().Ticker(sym).history(period=period)["Close"].dropna()
            if len(hist) < 2:
                raise ValueError("not enough history")
            start, end = float(hist.iloc[0]), float(hist.iloc[-1])
            out = {
                "symbol": sym, "period": period,
                "start_price": _round(start), "end_price": _round(end),
                "change_percent": _round((end - start) / start * 100),
                "high": _round(hist.max()), "low": _round(hist.min()),
                "as_of": _now(),
            }
            _log(f"yfinance OK (LIVE): {sym} {period} change={out['change_percent']}%")
            return out
        except Exception as e:
            err = str(e)
            _log(f"yfinance FAILED for {sym} history: {err}")
    else:
        err = "demo mode"
    h = _demo()["history"].get(sym, {}).get(period)
    if h is None:
        _log(f"no {period} history for {sym} (live failed, not in demo data)")
        return {"error": f"No {period} history for {sym}. Live lookup failed ({err})."}
    _log(f"using DEMO {period} history for {sym} (made-up sample numbers)")
    return {"symbol": sym, "period": period, **h, "demo": True,
            "note": "DEMO DATA, not real market prices."}


def get_company_info(symbol):
    sym = resolve_symbol(symbol)
    if not FORCE_DEMO:
        _log(f"yfinance: fetching company info for {sym} ...")
        try:
            info = _yf().Ticker(sym).info
            if not info or "shortName" not in info:
                raise ValueError("no company info")
            out = {
                "symbol": sym,
                "name": info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "trailing_pe": _round(info.get("trailingPE")),
                "forward_pe": _round(info.get("forwardPE")),
                "as_of": _now(),
            }
            _log(f"yfinance OK (LIVE): {sym} is {out['name']}")
            return out
        except Exception as e:
            err = str(e)
            _log(f"yfinance FAILED for {sym} info: {err}")
    else:
        err = "demo mode"
    c = _demo()["companies"].get(sym)
    if c is None:
        _log(f"no company info for {sym} (live failed, not in demo data)")
        return {"error": f"No company info for {sym}. Live lookup failed ({err})."}
    _log(f"using DEMO company info for {sym}")
    return {"symbol": sym, **c, "demo": True, "note": "DEMO DATA, not real."}


def get_market_overview():
    """S&P 500, Nasdaq, Dow and VIX levels with daily change."""
    out = []
    any_demo = False
    for name, sym in INDEX_SYMBOLS.items():
        q = get_stock_quote(sym)
        if "error" in q:
            out.append({"index": name, "error": q["error"]})
            continue
        any_demo = any_demo or q.get("demo", False)
        out.append({"index": name, "level": q["price"],
                    "change_percent": q["change_percent"]})
    result = {"indices": out, "as_of": _now()}
    if any_demo:
        result["demo"] = True
        result["note"] = "DEMO DATA, not real market prices."
    return result


# ---- registry + schemas for Ollama -----------------------------------------
TOOL_FUNCS = {
    "get_stock_quote": get_stock_quote,
    "get_price_history": get_price_history,
    "get_company_info": get_company_info,
    "get_market_overview": get_market_overview,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_stock_quote",
        "description": "Get the current price and today's change for a stock, ETF or index. Use for any question about a price right now or how a stock did today.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "Ticker like NVDA, or a company name like Nvidia."}},
            "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "get_price_history",
        "description": "Get the percent change, high and low of a stock over a past period. Use for 'how has X done this month/year'.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "Ticker or company name."},
            "period": {"type": "string", "enum": ["5d", "1mo", "3mo", "6mo", "1y", "ytd"],
                       "description": "Lookback period."}},
            "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "get_company_info",
        "description": "Get a company's name, sector, market cap and trailing and forward P/E ratios. Use for 'what is X' or 'how big is X'.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "Ticker or company name."}},
            "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "get_market_overview",
        "description": "Get the S&P 500, Nasdaq, Dow and VIX levels and today's percent change. Use for 'how is the market doing'.",
        "parameters": {"type": "object", "properties": {}}}},
]


def run_tool(name, arguments):
    """Execute a tool call from the LLM. Never raises; returns a JSON string."""
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool {name}"})
    try:
        return json.dumps(fn(**(arguments or {})))
    except TypeError as e:
        return json.dumps({"error": f"Bad arguments for {name}: {e}"})
    except Exception as e:
        return json.dumps({"error": f"{name} failed: {e}"})


# ---- spoken summaries (used when the LLM's wording can't be trusted) ---------
_PERIOD_WORDS = {
    "5d": "the past 5 days", "1mo": "the past month", "3mo": "the past 3 months",
    "6mo": "the past 6 months", "1y": "the past year", "ytd": "this year so far",
}


def _updown(x):
    return "up" if (x or 0) >= 0 else "down"


def _big_money(n):
    for div, word in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if abs(n) >= div:
            return f"{n / div:.2f} {word} dollars"
    return f"{n:.0f} dollars"


def summarize_result(name, args, data):
    """Turn one tool result into plain spoken sentences using only the tool's own numbers.

    llm.ask() falls back to this if the model's answer contains a number that no tool
    returned, so the user still hears a correct answer.
    """
    sym = data.get("symbol") or (args or {}).get("symbol") or "that"
    if "error" in data and name != "get_market_overview":
        return f"I couldn't get data for {sym}."
    demo = " That is demo data, not real prices." if data.get("demo") else ""

    if name == "get_stock_quote":
        pct = data.get("change_percent")
        move = f", {_updown(pct)} {abs(pct):.2f} percent today" if pct is not None else ""
        return f"{sym} is at {data['price']:.2f} dollars{move}.{demo}"

    if name == "get_price_history":
        pct = data.get("change_percent")
        return (f"{sym} is {_updown(pct)} {abs(pct):.2f} percent over "
                f"{_PERIOD_WORDS.get(data.get('period'), 'that period')}, from "
                f"{data['start_price']:.2f} to {data['end_price']:.2f} dollars.{demo}")

    if name == "get_company_info":
        bits = [f"{data.get('name') or sym}"]
        if data.get("sector"):
            bits.append(f"is in the {data['sector']} sector")
        sentence = " ".join(bits)
        extras = []
        if data.get("market_cap"):
            extras.append(f"a market cap of {_big_money(data['market_cap'])}")
        if data.get("trailing_pe") is not None:
            extras.append(f"a trailing P E of {data['trailing_pe']:.2f}")
        if data.get("forward_pe") is not None:
            extras.append(f"a forward P E of {data['forward_pe']:.2f}")
        if extras:
            sentence += (" with " if data.get("sector") else " has ") + ", ".join(extras)
        return sentence + "." + demo

    if name == "get_market_overview":
        parts = []
        for ix in data.get("indices", []):
            nm = ix["index"].replace("S&P", "S and P")
            if "error" in ix:
                parts.append(f"I couldn't get the {nm}.")
            else:
                parts.append(f"The {nm} is at {ix['level']:.2f}, "
                             f"{_updown(ix['change_percent'])} {abs(ix['change_percent']):.2f} percent.")
        return " ".join(parts) + demo

    return f"I couldn't summarize {name}."


# ---- router -------------------------------------------------------------------
SCREENER_REPLY = (
    "I can't scan the whole market for things like the most undervalued stocks or the "
    "lowest forward P E. I can only check specific tickers or the major indexes, so tell "
    "me which companies you want me to look at."
)
CLARIFY_REPLY = (
    "I can look up a specific stock or a major index. Which company or ticker do you mean?"
)

_I = re.IGNORECASE

# Greetings and small talk. Only trusted when no ticker or market word is present.
_CHAT_RE = re.compile(
    r"^\s*(?:hi|hello|hey|yo|sup|howdy|hiya|good\s+(?:morning|afternoon|evening|night)|"
    r"how\s+(?:are|r)\s+(?:you|u|things|ya)|how'?s\s+it\s+going|how\s+is\s+it\s+going|"
    r"what'?s\s+up|thanks?|thank\s+you|cheers|bye|goodbye|see\s+you|"
    r"who\s+are\s+you|what(?:'s|\s+is)\s+your\s+name|what\s+can\s+you\s+do|"
    r"nice\s+to\s+meet\s+you|tell\s+me\s+a\s+joke|are\s+you\s+(?:there|alive|real|ok))\b", _I)

_MARKET_RE = re.compile(
    r"\b(?:the\s+market|stock\s+market|markets?|wall\s+street|s\s*&\s*p(?:\s*500)?|sp\s*500|spx|"
    r"nasdaq|dow(?:\s+jones)?|vix|volatility\s+index|indexes|indices)\b", _I)

# Screener / "pick me stocks" requests the tools cannot answer.
_RANK_WORDS = r"(?:most|least|top|best|worst|cheapest|highest|lowest|biggest|largest|hottest|which)"
_RANK_TARGET = (r"(?:undervalued|overvalued|cheap|expensive|forward\s*p\s*/?\s*e|p\s*/?\s*e|"
                r"price[- ]to[- ]earnings|dividend\s+yield|gainers?|losers?|growth\s+stocks?|"
                r"value\s+stocks?|stocks?|shares?|etfs?)")
_SCREENER_RES = [
    re.compile(rf"\b{_RANK_WORDS}\b.{{0,40}}\b{_RANK_TARGET}\b", _I),
    re.compile(r"\b(?:stocks?|shares?|companies|etfs?)\s+(?:to\s+buy|to\s+invest\s+in|to\s+watch|"
               r"with|that|under|over|trading\s+(?:below|above))\b", _I),
    re.compile(r"\b(?:find|show|list|recommend|suggest|pick|name)\b.{0,30}\b(?:undervalued|overvalued|"
               r"cheap|best|top|good|dividend|growth|value)\b.{0,15}\b(?:stocks?|shares?|companies|etfs?)\b", _I),
    re.compile(r"\b(?:screen(?:er)?|rank(?:ing)?)\b", _I),
]
_VALUATION_WORD = re.compile(r"\b(?:undervalued|overvalued)\b", _I)

_CONCEPT_CUE = re.compile(
    r"\b(?:what\s+is|what\s+are|what\s+does|explain|define|definition|meaning\s+of|"
    r"why\s+do|how\s+does|difference\s+between)\b", _I)
_DATA_CUE = re.compile(
    r"\b(?:today|tonight|now|currently|current|at\s+the\s+moment|doing|did|trading|trade|level|"
    r"closed|closing|opened|up|down|how\s+is|how's|how\s+are|how\s+much|price|prices|quote|"
    r"worth|cost|performing|performance|at)\b", _I)
_PRICE_CUE = re.compile(
    r"\b(?:price|prices|quote|trading|trades|cost|how\s+much|worth|today|tonight|now|doing|up|down|"
    r"closed|close|open|opened|currently|current|at)\b", _I)
_HIST_CUE = re.compile(
    r"\b(?:(?:this|last|past|over\s+the|in\s+the\s+(?:last|past))\s+"
    r"(?:week|month|quarter|year|\d+\s+(?:days?|months?)|(?:three|six)\s+months?)|ytd|year\s+to\s+date|"
    r"performance|performed|performing|history|historically|trend|gained|lost|done|since)\b", _I)
_COMPANY_CUE = re.compile(
    r"\b(?:market\s+cap\w*|capitalization|how\s+big|sector|industry|p\s*/?\s*e|pe\s+ratio|"
    r"price[- ]to[- ]earnings|forward\s+p\s*/?\s*e|valuation|valued|undervalued|overvalued|"
    r"expensive|cheap)\b", _I)
_ABOUT_CUE = re.compile(r"\b(?:what\s+is|what\s+does|tell\s+me\s+about|who\s+is|what\s+company)\b", _I)
_PRONOUN = re.compile(r"\b(?:it|its|that\s+one|this\s+one|that\s+stock|the\s+stock|the\s+company|they|them)\b", _I)
_OTHER_ENTITY = re.compile(r"\b(?:of|for)\s+(?!it\b|that\b|this\b|them\b|the\s+stock\b|the\s+company\b)\w+", _I)
_FINANCE_WORDS = re.compile(
    r"\b(?:stocks?|shares?|equit(?:y|ies)|etfs?|funds?|invest\w*|portfolio|dividends?|earnings|eps|"
    r"p\s*/?\s*e|price[- ]to[- ]earnings|valuation|undervalued|overvalued|bonds?|treasur(?:y|ies)|yields?|inflation|"
    r"interest\s+rates?|recession|crypto\w*|options?|ipo|bull|bear|short\s+sell\w*|market\s+cap\w*|"
    r"ticker|trader?|trading|finance|financial|economy|economic|gdp|rate\s+hike)\b", _I)
_PRICE_ASK = re.compile(r"\b(?:price|quote|trading\s+at)\b", _I)

# All-caps words that look like tickers but aren't.
_NOT_TICKERS = {
    "AI", "API", "ATH", "ATM", "CEO", "CFO", "CPI", "CPU", "DOW", "EPS", "ESG", "EV", "FAQ",
    "FED", "FOMC", "GDP", "GPU", "HI", "IPO", "IRA", "LLM", "NASDAQ", "NYSE", "OK", "OKAY",
    "PE", "PM", "PPI", "REIT", "ROI", "SEC", "SPX", "TV", "UK", "US", "USA", "USD", "VIX", "YTD",
    "AM", "AN", "AND", "ARE", "FOR", "HEY", "HOW", "THE", "WHY", "YOU", "WHAT", "WHO",
}
# Lowercase ticker symbols we accept as-is (4+ letters; short ones are ordinary words).
_LOWER_TICKERS = {v.lower() for v in NAME_TO_TICKER.values()
                  if len(v) >= 4 and v.isalpha() and v != "COST"}


def _strip_market_cap(low):
    return re.sub(r"market\s+cap\w*", " ", low)


def find_tickers(text):
    """Tickers mentioned in `text`, in order: company names, $TICKER, ALL-CAPS symbols."""
    low = text.lower()
    hits = {}  # ticker -> position of first mention

    def add(ticker, pos):
        if ticker not in hits or pos < hits[ticker]:
            hits[ticker] = pos

    for name in sorted(NAME_TO_TICKER, key=len, reverse=True):
        for m in re.finditer(rf"(?<![\w-]){re.escape(name)}(?![\w-])", low):
            add(NAME_TO_TICKER[name], m.start())
    for m in re.finditer(r"(?<![\w$])\$?([A-Za-z]{1,5}(?:[.-][A-Za-z])?)(?![\w])", text):
        tok, has_dollar = m.group(1), m.group(0).startswith("$")
        up = tok.upper()
        if has_dollar:
            add(up, m.start())
        elif tok.isupper() and len(tok) >= 2 and not text.isupper():
            if up not in _NOT_TICKERS and tok.lower() not in NAME_TO_TICKER:
                add(up, m.start())
        elif tok.lower() in _LOWER_TICKERS:
            add(up, m.start())
    return [t for t, _ in sorted(hits.items(), key=lambda kv: kv[1])][:MAX_TICKERS]


def _period_for(low):
    if re.search(r"\b(?:ytd|year\s+to\s+date|this\s+year|so\s+far\s+this\s+year)\b", low):
        return "ytd"
    if re.search(r"\b(?:6\s+months?|six\s+months?|half\s+a\s+year|half[- ]year)\b", low):
        return "6mo"
    if re.search(r"\b(?:3\s+months?|three\s+months?|quarter)\b", low):
        return "3mo"
    if re.search(r"\b(?:year|12\s+months?|twelve\s+months?|annual)\b", low):
        return "1y"
    if re.search(r"\b(?:week|5\s+days?|five\s+days?)\b", low):
        return "5d"
    return "1mo"


def _last_tickers(history):
    for m in reversed(list(history or [])):
        if m.get("role") == "user":
            found = find_tickers(m.get("content", ""))
            if found:
                return found
    return []


def _is_screener(low, has_market):
    if any(r.search(low) for r in _SCREENER_RES):
        return True
    # A bare "overvalued" is a screener ask, unless it is a definition question or about the market.
    return bool(_VALUATION_WORD.search(low)) and not _CONCEPT_CUE.search(low) and not has_market


def route(text, history=()):
    """Decide how to handle one user message. Returns a dict:

    {"intent": "chat"}                          casual conversation, no tools
    {"intent": "general"}                       finance concept question, no tools
    {"intent": "screener", "reply": "..."}      can't screen the market; canned answer
    {"intent": "clarify", "reply": "..."}       wants a price but named nothing we know
    {"intent": "data", "calls": [(tool, args)]} fetch these in code, then summarize
    """
    t = " ".join((text or "").split())
    low = t.lower()
    tickers = find_tickers(t)
    market = bool(_MARKET_RE.search(_strip_market_cap(low)))

    if not tickers and not market and _CHAT_RE.match(low):
        return {"intent": "chat"}
    if not tickers and _is_screener(low, market):
        return {"intent": "screener", "reply": SCREENER_REPLY}

    if not tickers and not market:
        # Follow-ups like "what about this year?" or "how is it doing?" reuse the last ticker.
        follow = (_HIST_CUE.search(low) or _PRICE_ASK.search(low)
                  or (_PRONOUN.search(low) and (_DATA_CUE.search(low) or _COMPANY_CUE.search(low))))
        if follow and not _OTHER_ENTITY.search(low):
            tickers = _last_tickers(history)

    if not tickers and not market:
        if _PRICE_ASK.search(low):
            return {"intent": "clarify", "reply": CLARIFY_REPLY}
        if _FINANCE_WORDS.search(low):
            return {"intent": "general"}
        return {"intent": "chat"}

    if not tickers and _CONCEPT_CUE.search(low) and not _DATA_CUE.search(low):
        return {"intent": "general"}  # "what is the VIX?" is a definition, not a lookup

    want_hist = bool(_HIST_CUE.search(low))
    want_price_words = bool(_PRICE_CUE.search(low))
    want_co = bool(_COMPANY_CUE.search(low)) or (
        bool(_ABOUT_CUE.search(low)) and not want_price_words and not want_hist)
    want_quote = want_price_words or not (want_hist or want_co)

    calls = []
    if market:
        calls.append(("get_market_overview", {}))
    for sym in tickers:
        if want_quote:
            calls.append(("get_stock_quote", {"symbol": sym}))
        if want_hist:
            calls.append(("get_price_history", {"symbol": sym, "period": _period_for(low)}))
        if want_co:
            calls.append(("get_company_info", {"symbol": sym}))
    return {"intent": "data", "calls": calls}


if __name__ == "__main__":
    print(json.dumps(get_stock_quote("nvidia"), indent=2))
    print(json.dumps(get_market_overview(), indent=2))

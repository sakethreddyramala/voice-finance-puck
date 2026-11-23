"""Finance tools the local LLM can call, backed by yfinance and live web search.

Every tool returns a small JSON-serializable dict. If yfinance is missing, the
network is down, or DEMO_MODE=1 is set, tools fall back to sample data from
demo_data.json and mark the result with "demo": true so the assistant says so.

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

# Common spoken company names -> tickers for instant zero-latency resolution.
NAME_TO_TICKER = {
    "nvidia": "NVDA", "apple": "AAPL", "microsoft": "MSFT", "tesla": "TSLA",
    "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL", "meta": "META",
    "facebook": "META", "netflix": "NFLX", "amd": "AMD", "intel": "INTC",
    "micron": "MU", "palantir": "PLTR", "coinbase": "COIN", "disney": "DIS",
    "boeing": "BA", "jpmorgan": "JPM", "jp morgan": "JPM", "goldman sachs": "GS",
    "berkshire": "BRK-B", "walmart": "WMT", "costco": "COST", "visa": "V",
    "mastercard": "MA", "coca cola": "KO", "coca-cola": "KO", "pepsi": "PEP",
    "nike": "NKE", "starbucks": "SBUX", "uber": "UBER", "broadcom": "AVGO",
    "tsmc": "TSM", "oracle": "ORCL", "salesforce": "CRM", "adobe": "ADBE",
    "qualcomm": "QCOM", "arm": "ARM", "supermicro": "SMCI", "super micro": "SMCI",
    "crowdstrike": "CRWD", "snowflake": "SNOW", "shopify": "SHOP", "spotify": "SPOT",
    "robinhood": "HOOD", "sofi": "SOFI", "gamestop": "GME", "amc": "AMC",
    "spy": "SPY", "qqq": "QQQ", "iwm": "IWM",
    "bitcoin": "BTC-USD", "ethereum": "ETH-USD", "solana": "SOL-USD",
    "gold": "GC=F", "oil": "CL=F",
}

INDEX_SYMBOLS = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "VIX": "^VIX",
}

_demo_cache = None
_search_ticker_cache = {}


def _log(msg):
    """Terminal log so you can see exactly when tools run and what came back."""
    if LOG_TOOLS:
        print(f"[finance] {msg}", flush=True)


def _demo():
    global _demo_cache
    if _demo_cache is None:
        _demo_cache = json.loads(DEMO_FILE.read_text(encoding="utf-8"))
    return _demo_cache


def _yf():
    """Import yfinance lazily so demo mode works without it."""
    import yfinance as yf
    return yf


def resolve_symbol(text):
    """Resolve company name or ticker ('nvidia' -> 'NVDA', 'micron' -> 'MU', 'CrowdStrike' -> 'CRWD')."""
    raw = (text or "").strip()
    if not raw:
        return ""
    low = raw.lower().replace(".", "")
    if low in NAME_TO_TICKER:
        return NAME_TO_TICKER[low]
    if low in _search_ticker_cache:
        return _search_ticker_cache[low]

    # If it already looks like a clean ticker (1-5 uppercase letters, or ^/=/ - suffix), return it.
    cleaned = raw.upper().replace(" ", "")
    if re.fullmatch(r"\^?[A-Z]{1,5}(?:[.-][A-Z]{1,2}|=F|-USD)?", cleaned) and len(raw) <= 5 and raw.isupper():
        return cleaned

    # Dynamic Yahoo Finance search for any company name worldwide
    if not FORCE_DEMO:
        try:
            search = _yf().Search(raw, max_results=3)
            quotes = getattr(search, "quotes", []) or []
            for q in quotes:
                sym = q.get("symbol")
                qtype = (q.get("quoteType") or "").upper()
                if sym and qtype in {"EQUITY", "ETF", "INDEX", "CRYPTOCURRENCY", "FUTURE"}:
                    _search_ticker_cache[low] = sym
                    _log(f"resolved '{raw}' -> {sym}")
                    return sym
        except Exception as e:
            _log(f"ticker search failed for '{raw}': {e}")

    return cleaned


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
                "fifty_two_week_high": _round(info.get("fiftyTwoWeekHigh")),
                "fifty_two_week_low": _round(info.get("fiftyTwoWeekLow")),
                "dividend_yield_percent": _round((info.get("dividendYield") or 0) * 100) if info.get("dividendYield") else None,
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


def get_options_summary(symbol, expiration=None):
    """Nearest expiration ATM calls and puts (strike, last price, implied volatility, volume)."""
    sym = resolve_symbol(symbol)
    if FORCE_DEMO:
        return {"error": f"Options chains are only available in live mode, not demo mode."}
    _log(f"yfinance: fetching options chain for {sym} ...")
    try:
        t = _yf().Ticker(sym)
        expirations = t.options
        if not expirations:
            return {"error": f"No options available for {sym}."}
        exp = expiration if (expiration in expirations) else expirations[0]
        price = float(t.fast_info["last_price"])
        chain = t.option_chain(exp)

        def _atm_row(df):
            if df is None or df.empty:
                return None
            idx = (df["strike"] - price).abs().idxmin()
            row = df.loc[idx]
            return {
                "strike": _round(row.get("strike")),
                "last_price": _round(row.get("lastPrice")),
                "bid": _round(row.get("bid")),
                "ask": _round(row.get("ask")),
                "implied_volatility_percent": _round((row.get("impliedVolatility") or 0) * 100, 1),
                "volume": int(row["volume"]) if row.get("volume") == row.get("volume") else 0,
                "open_interest": int(row["openInterest"]) if row.get("openInterest") == row.get("openInterest") else 0,
            }

        out = {
            "symbol": sym,
            "underlying_price": _round(price),
            "expiration": exp,
            "available_expirations": list(expirations[:4]),
            "atm_call": _atm_row(chain.calls),
            "atm_put": _atm_row(chain.puts),
            "as_of": _now(),
        }
        _log(f"yfinance OK (LIVE): {sym} options exp={exp} ATM call={out['atm_call']}")
        return out
    except Exception as e:
        _log(f"yfinance FAILED for {sym} options: {e}")
        return {"error": f"Could not fetch options for {sym}: {e}"}


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


def web_search(query):
    """Search the live web for finance news, reasons behind stock moves, earnings, or general answers."""
    if FORCE_DEMO:
        return {"error": "Web search is disabled in demo mode."}
    _log(f"web_search: querying '{query}' ...")
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=4):
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                })
        if not results:
            return {"error": f"No web results found for '{query}'."}
        _log(f"web_search OK: returned {len(results)} snippets")
        return {"query": query, "results": results, "as_of": _now()}
    except Exception as e:
        _log(f"web_search FAILED: {e}")
        return {"error": f"Web search failed: {e}"}


# ---- registry + schemas for Ollama -----------------------------------------
TOOL_FUNCS = {
    "get_stock_quote": get_stock_quote,
    "get_price_history": get_price_history,
    "get_company_info": get_company_info,
    "get_options_summary": get_options_summary,
    "get_market_overview": get_market_overview,
    "web_search": web_search,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_stock_quote",
        "description": "Get the current price and today's change for any stock, ETF, crypto, or index.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "Ticker like NVDA or company name like Micron."}},
            "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "get_price_history",
        "description": "Get the percent change, high and low of a stock over a past period.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "Ticker or company name."},
            "period": {"type": "string", "enum": ["5d", "1mo", "3mo", "6mo", "1y", "ytd"],
                       "description": "Lookback period."}},
            "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "get_company_info",
        "description": "Get a company's name, sector, market cap, 52-week range, dividend yield, and P/E ratios.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "Ticker or company name."}},
            "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "get_options_summary",
        "description": "Get nearest-expiration ATM call and put option prices, strike, implied volatility, and volume.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "Ticker or company name."}},
            "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "get_market_overview",
        "description": "Get the S&P 500, Nasdaq, Dow and VIX levels and today's percent change.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the internet for stock news, reasons why a stock moved, earnings dates, or general finance questions.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query."}},
            "required": ["query"]}}},
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
    """Turn one tool result into plain spoken sentences using only the tool's own numbers."""
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

    if name == "get_options_summary":
        call = data.get("atm_call") or {}
        put = data.get("atm_put") or {}
        return (f"For {sym} expiring {data.get('expiration')}, the at-the-money {call.get('strike')} dollar call "
                f"is trading at {call.get('last_price')} dollars with {call.get('implied_volatility_percent')} percent "
                f"implied volatility, and the {put.get('strike')} dollar put is at {put.get('last_price')} dollars.")

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

    if name == "web_search":
        snippets = [r.get("snippet", "") for r in data.get("results", []) if r.get("snippet")]
        return snippets[0] if snippets else "I found web results, though none had a clear summary."

    return f"I couldn't summarize {name}."


# ---- router -------------------------------------------------------------------
CLARIFY_REPLY = (
    "I can look up any stock, option chain, or market index, or search the web. Which company or topic do you mean?"
)

_I = re.IGNORECASE

_CHAT_RE = re.compile(
    r"^\s*(?:hi|hello|hey|yo|sup|howdy|hiya|good\s+(?:morning|afternoon|evening|night)|"
    r"how\s+(?:are|r)\s+(?:you|u|things|ya)|how'?s\s+it\s+going|how\s+is\s+it\s+going|"
    r"what'?s\s+up|thanks?|thank\s+you|cheers|bye|goodbye|see\s+you|"
    r"who\s+are\s+you|what(?:'s|\s+is)\s+your\s+name|what\s+can\s+you\s+do|"
    r"nice\s+to\s+meet\s+you|tell\s+me\s+a\s+joke|are\s+you\s+(?:there|alive|real|ok))\s*[!?.]*\s*$", _I)

_MARKET_RE = re.compile(
    r"\b(?:the\s+market|stock\s+market|markets?|wall\s+street|s\s*&\s*p(?:\s*500)?|sp\s*500|spx|"
    r"nasdaq|dow(?:\s+jones)?|vix|volatility\s+index|indexes|indices)\b", _I)

_OPTIONS_CUE = re.compile(
    r"\b(?:options?|calls?|puts?|strike|expir(?:y|ation)|implied\s+volatility|iv|open\s+interest|greeks|chain)\b", _I)

_NEWS_WEB_CUE = re.compile(
    r"\b(?:why\s+(?:is|did|are|was|were)|news|headline|catalyst|earnings|announce\w*|report\w*|"
    r"fed|fomc|powell|interest\s+rate|cpi|inflation|rumor|lawsuit|merger|acquire|acquisition|"
    r"search|google|look\s+up\s+online|internet|what\s+happened|going\s+on\s+with)\b", _I)

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

_CONCEPT_CUE = re.compile(
    r"\b(?:what\s+is\s+(?:a|an|the)?\s*(?:p/?e|etf|option|call|put|short|dividend|yield|bond|index|vix|ipo|bull|bear)|"
    r"explain|define|definition|meaning\s+of|how\s+does\s+(?:a|an|the)?\s*(?:stock|option|etf|market)\s+work|"
    r"difference\s+between)\b", _I)
_DATA_CUE = re.compile(
    r"\b(?:today|tonight|now|currently|current|at\s+the\s+moment|doing|did|trading|trade|level|"
    r"closed|closing|opened|up|down|more|less|higher|lower|better|worse|compare|versus|vs|"
    r"how\s+is|how's|how\s+are|how\s+much|price|prices|quote|worth|cost|performing|performance|at)\b", _I)
_PRICE_CUE = re.compile(
    r"\b(?:price|prices|quote|trading|trades|cost|how\s+much|worth|today|tonight|now|doing|up|down|"
    r"more|less|higher|lower|closed|close|open|opened|currently|current|at)\b", _I)
_HIST_CUE = re.compile(
    r"\b(?:(?:this|last|past|over\s+the|in\s+the\s+(?:last|past))\s+"
    r"(?:week|month|quarter|year|\d+\s+(?:days?|months?)|(?:three|six)\s+months?)|ytd|year\s+to\s+date|"
    r"performance|performed|performing|history|historically|trend|gained|lost|done|since)\b", _I)
_COMPANY_CUE = re.compile(
    r"\b(?:market\s+cap\w*|capitalization|how\s+big|sector|industry|p\s*/?\s*e|pe\s+ratio|"
    r"price[- ]to[- ]earnings|forward\s+p\s*/?\s*e|valuation|valued|52[- ]week|dividend)\b", _I)
_ABOUT_CUE = re.compile(r"\b(?:what\s+is|what\s+does|tell\s+me\s+about|who\s+is|what\s+company)\b", _I)
_PRONOUN = re.compile(r"\b(?:it|its|that\s+one|this\s+one|that\s+stock|the\s+stock|the\s+company|they|them)\b", _I)
_OTHER_ENTITY = re.compile(r"\b(?:of|for)\s+(?!it\b|that\b|this\b|them\b|the\s+stock\b|the\s+company\b)\w+", _I)
_PRICE_ASK = re.compile(r"\b(?:price|quote|trading\s+at)\b", _I)

_NOT_TICKERS = {
    "AI", "API", "ATH", "ATM", "CEO", "CFO", "CPI", "CPU", "DOW", "EPS", "ESG", "EV", "FAQ",
    "FED", "FOMC", "GDP", "GPU", "HI", "IPO", "IRA", "IV", "LLM", "NASDAQ", "NYSE", "OK", "OKAY",
    "PE", "PM", "PPI", "REIT", "ROI", "SEC", "SPX", "TV", "UK", "US", "USA", "USD", "VIX", "YTD",
    "AM", "AN", "AND", "ARE", "DID", "DO", "FOR", "HEY", "HOW", "IS", "IT", "OR", "THE", "TO",
    "UP", "VS", "WHY", "YOU", "WHAT", "WHO",
}
_LOWER_TICKERS = {v.lower() for v in NAME_TO_TICKER.values()
                  if len(v) >= 4 and v.isalpha() and v != "COST"}

# Common English words that shouldn't trigger dynamic Yahoo company searches on their own.
_STOPWORDS = {
    "a", "about", "after", "all", "also", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "between", "both", "but", "by",
    "call", "calls", "can", "chain", "company", "compare", "could", "current", "currently",
    "day", "days", "did", "do", "does", "doing", "down", "during",
    "each", "earnings", "etf", "etfs", "explain", "for", "forward", "from",
    "get", "give", "go", "going", "good", "great",
    "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how",
    "i", "if", "in", "index", "info", "into", "is", "it", "its",
    "just", "know", "last", "like", "look", "looking", "lowest",
    "make", "many", "market", "markets", "me", "month", "months", "more", "most", "much", "my",
    "news", "next", "no", "not", "now", "of", "on", "one", "only", "option", "options", "or", "other", "our", "out", "over", "overvalued",
    "past", "pe", "percent", "performance", "performing", "price", "prices", "put", "puts",
    "quote", "ratio", "right", "say", "search", "see", "share", "shares", "she", "should", "show", "so", "some", "stock", "stocks",
    "tell", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "to", "today", "tonight", "trading",
    "undervalued", "up", "us", "value", "versus", "vs",
    "want", "was", "we", "week", "well", "were", "what", "when", "where", "which", "who", "why", "will", "with", "would",
    "year", "yes", "yet", "yo", "you", "your",
}


def _strip_market_cap(low):
    return re.sub(r"market\s+cap\w*", " ", low)


def find_tickers(text):
    """Find all tickers in `text`: known company names, $TICKER, ALL-CAPS symbols, and capitalized company names."""
    low = text.lower()
    hits = {}  # ticker -> position of first mention

    def add(ticker, pos):
        if ticker and (ticker not in hits or pos < hits[ticker]):
            hits[ticker] = pos

    # 1. Known spoken company names
    for name in sorted(NAME_TO_TICKER, key=len, reverse=True):
        for m in re.finditer(rf"(?<![\w-]){re.escape(name)}(?![\w-])", low):
            add(NAME_TO_TICKER[name], m.start())

    # 2. Explicit $TICKER or ALL-CAPS tickers
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

    # 3. Dynamic lookup for any capitalized proper noun (e.g., "Micron", "CrowdStrike", "Toyota")
    # Whisper capitalizes proper nouns mid-sentence and at the start of sentences.
    for m in re.finditer(r"\b([A-Z][a-zA-Z0-9&.-]{2,}(?:\s+[A-Z][a-zA-Z0-9&.-]{2,})?)\b", text):
        phrase = m.group(1).strip()
        words_low = [w.lower() for w in phrase.split()]
        if all(w in _STOPWORDS for w in words_low):
            continue
        # Strip leading question words if Whisper capitalized the start of a sentence ("Did Micron" -> "Micron")
        while words_low and words_low[0] in _STOPWORDS:
            phrase = " ".join(phrase.split()[1:])
            words_low.pop(0)
        if not phrase or phrase.lower() in NAME_TO_TICKER or phrase.lower() in _STOPWORDS:
            continue
        if phrase.upper() in _NOT_TICKERS:
            continue
        resolved = resolve_symbol(phrase)
        if resolved and len(resolved) <= 8:
            add(resolved, m.start())

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


def _is_screener(low):
    return any(r.search(low) for r in _SCREENER_RES)


def route(text, history=()):
    """Route 99.9% of prompts to live yfinance tools, options chains, web search, or chat."""
    t = " ".join((text or "").split())
    low = t.lower()

    if _CHAT_RE.match(low):
        return {"intent": "chat"}

    tickers = find_tickers(t)
    market = bool(_MARKET_RE.search(_strip_market_cap(low)))

    # Follow-up pronouns ("what about its options?", "how did it do this year?")
    if not tickers and not market:
        follow = (_HIST_CUE.search(low) or _PRICE_ASK.search(low) or _OPTIONS_CUE.search(low)
                  or (_PRONOUN.search(low) and (_DATA_CUE.search(low) or _COMPANY_CUE.search(low) or _NEWS_WEB_CUE.search(low))))
        if follow and not _OTHER_ENTITY.search(low):
            tickers = _last_tickers(history)

    # Pure concept explanations ("What is a forward P/E ratio?")
    if not tickers and not market and _CONCEPT_CUE.search(low) and not _DATA_CUE.search(low) and not _NEWS_WEB_CUE.search(low):
        return {"intent": "general"}

    # Screener/ranking questions ("What stock is the most undervalued right now?") -> live web search!
    if not tickers and _is_screener(low):
        return {"intent": "data", "calls": [("web_search", {"query": t})]}

    want_options = bool(_OPTIONS_CUE.search(low))
    want_news = bool(_NEWS_WEB_CUE.search(low))
    want_hist = bool(_HIST_CUE.search(low))
    want_price_words = bool(_PRICE_CUE.search(low))
    want_co = bool(_COMPANY_CUE.search(low)) or (
        bool(_ABOUT_CUE.search(low)) and not want_price_words and not want_hist and not want_options and not want_news)
    want_quote = want_price_words or not (want_hist or want_co or want_options or want_news)

    calls = []
    if market:
        calls.append(("get_market_overview", {}))

    for sym in tickers:
        if want_options:
            calls.append(("get_options_summary", {"symbol": sym}))
        if want_quote:
            calls.append(("get_stock_quote", {"symbol": sym}))
        if want_hist:
            calls.append(("get_price_history", {"symbol": sym, "period": _period_for(low)}))
        if want_co:
            calls.append(("get_company_info", {"symbol": sym}))

    # If the user asks "why" a stock moved or asks for news on a ticker, also search the web for context
    if want_news:
        calls.append(("web_search", {"query": t}))

    if calls:
        return {"intent": "data", "calls": calls}

    # Any other non-greeting question (general knowledge, macro events, Fed, crypto, etc.) -> live web search!
    if len(t.split()) >= 3:
        return {"intent": "data", "calls": [("web_search", {"query": t})]}

    return {"intent": "chat"}


if __name__ == "__main__":
    print(json.dumps(get_stock_quote("micron"), indent=2))
    print(json.dumps(get_options_summary("NVDA"), indent=2))
    print(json.dumps(web_search("why is nvidia stock moving today"), indent=2))

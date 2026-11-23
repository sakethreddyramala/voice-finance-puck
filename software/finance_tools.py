"""General + finance tools backed by yfinance and live web search."""
import json
import os
import re
from datetime import datetime
from pathlib import Path

DEMO_FILE = Path(__file__).with_name("demo_data.json")
FORCE_DEMO = os.environ.get("DEMO_MODE", "0") == "1"
LOG_TOOLS = os.environ.get("FINANCE_LOG", "1") != "0"
MAX_TICKERS = 5

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
    "toyota": "TM", "ford": "F", "gm": "GM", "general motors": "GM",
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


def _log(msg):
    if LOG_TOOLS:
        print(f"[finance] {msg}", flush=True)


def _demo():
    global _demo_cache
    if _demo_cache is None:
        _demo_cache = json.loads(DEMO_FILE.read_text(encoding="utf-8"))
    return _demo_cache


def _yf():
    import yfinance as yf
    return yf


def resolve_symbol(text):
    raw = (text or "").strip()
    if not raw:
        return ""
    low = raw.lower().replace(".", "")
    if low in NAME_TO_TICKER:
        return NAME_TO_TICKER[low]
    return raw.upper().replace(" ", "")


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
        return {"error": f"No data for {sym}."}
    return {"symbol": sym, **q, "currency": "USD", "demo": True}


def get_top_movers(direction="losers"):
    """Fetch top daily gainers or losers directly from Yahoo Finance screeners."""
    if FORCE_DEMO:
        return {"error": "Top movers requires live data."}
    screen_name = "day_losers" if direction == "losers" else "day_gainers"
    _log(f"yfinance: screening {screen_name} ...")
    try:
        res = _yf().screen(screen_name, count=5)
        quotes = (res or {}).get("quotes", [])[:5]
        movers = []
        for q in quotes:
            movers.append({
                "symbol": q.get("symbol"),
                "name": q.get("shortName") or q.get("symbol"),
                "price": _round(q.get("regularMarketPrice")),
                "change_percent": _round(q.get("regularMarketChangePercent")),
            })
        if not movers:
            raise ValueError("empty screener result")
        _log(f"yfinance OK (LIVE): top {direction} = {[m['symbol'] for m in movers]}")
        return {"direction": direction, "movers": movers, "as_of": _now()}
    except Exception as e:
        _log(f"yfinance screener FAILED: {e}")
        return web_search(f"top stock market {direction} today yahoo finance")


def get_price_history(symbol, period="1mo"):
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
            _log(f"yfinance FAILED for {sym} history: {e}")
    return {"error": f"No {period} history for {sym}."}


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
                "as_of": _now(),
            }
            _log(f"yfinance OK (LIVE): {sym} is {out['name']}")
            return out
        except Exception as e:
            _log(f"yfinance FAILED for {sym} info: {e}")
    return {"error": f"No company info for {sym}."}


def get_options_summary(symbol, expiration=None):
    sym = resolve_symbol(symbol)
    if FORCE_DEMO:
        return {"error": "Options chains are only available in live mode."}
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
                "implied_volatility_percent": _round((row.get("impliedVolatility") or 0) * 100, 1),
                "volume": int(row["volume"]) if row.get("volume") == row.get("volume") else 0,
            }

        out = {
            "symbol": sym,
            "underlying_price": _round(price),
            "expiration": exp,
            "atm_call": _atm_row(chain.calls),
            "atm_put": _atm_row(chain.puts),
            "as_of": _now(),
        }
        _log(f"yfinance OK (LIVE): {sym} options exp={exp}")
        return out
    except Exception as e:
        _log(f"yfinance FAILED for {sym} options: {e}")
        return {"error": f"Could not fetch options for {sym}: {e}"}


def get_market_overview():
    out = []
    for name, sym in INDEX_SYMBOLS.items():
        q = get_stock_quote(sym)
        if "error" in q:
            out.append({"index": name, "error": q["error"]})
        else:
            out.append({"index": name, "level": q["price"], "change_percent": q["change_percent"]})
    return {"indices": out, "as_of": _now()}


def web_search(query):
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
            for r in ddgs.text(query, max_results=5):
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


TOOL_FUNCS = {
    "get_stock_quote": get_stock_quote,
    "get_top_movers": get_top_movers,
    "get_price_history": get_price_history,
    "get_company_info": get_company_info,
    "get_options_summary": get_options_summary,
    "get_market_overview": get_market_overview,
    "web_search": web_search,
}

TOOL_SCHEMAS = []


def run_tool(name, arguments):
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool {name}"})
    try:
        return json.dumps(fn(**(arguments or {})))
    except Exception as e:
        return json.dumps({"error": f"{name} failed: {e}"})


def summarize_result(name, args, data):
    sym = data.get("symbol") or (args or {}).get("symbol") or "that"
    if "error" in data and name != "get_market_overview":
        return f"I couldn't get data for {sym}."
    if name == "get_stock_quote":
        pct = data.get("change_percent")
        direction = "up" if (pct or 0) >= 0 else "down"
        return f"{sym} is at {data['price']:.2f} dollars, {direction} {abs(pct):.2f} percent."
    if name == "get_top_movers":
        items = [f"{m['name']} ({m['symbol']}) at {m['price']} dollars, down {abs(m['change_percent']):.2f} percent"
                 for m in data.get("movers", [])[:3]]
        return "The top movers are: " + ", ".join(items) + "."
    if name == "web_search":
        snippets = [r.get("snippet", "") for r in data.get("results", []) if r.get("snippet")]
        return snippets[0] if snippets else "I found search results but no clear summary."
    return "Here is what I found."


# ---- router -------------------------------------------------------------------
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

_LOSERS_RE = re.compile(r"\b(?:falling|dropping|losing|losers?|down|worst|crash\w*|tank\w*)\s+(?:the\s+)?most\b|\btop\s+losers?\b|\bbiggest\s+losers?\b", _I)
_GAINERS_RE = re.compile(r"\b(?:rising|gaining|gainers?|up|best|soar\w*|surge\w*)\s+(?:the\s+)?most\b|\btop\s+gainers?\b|\bbiggest\s+gainers?\b", _I)
_OPTIONS_CUE = re.compile(r"\b(?:options?|calls?|puts?|strike|expir(?:y|ation)|implied\s+volatility|iv|open\s+interest|chain)\b", _I)
_NEWS_WEB_CUE = re.compile(
    r"\b(?:why\s+(?:is|did|are|was|were)|news|headline|earnings|announce\w*|report\w*|"
    r"fed|fomc|powell|interest\s+rate|cpi|inflation|search|google|look\s+up|internet|weather|score|game|president|who\s+won|latest|current|today)\b", _I)
_HIST_CUE = re.compile(r"\b(?:(?:this|last|past|over\s+the)\s+(?:week|month|quarter|year)|ytd|year\s+to\s+date|history)\b", _I)
_COMPANY_CUE = re.compile(r"\b(?:market\s+cap\w*|sector|industry|p\s*/?\s*e|pe\s+ratio|forward\s+p\s*/?\s*e|valuation|52[- ]week)\b", _I)
_PRONOUN = re.compile(r"\b(?:it|its|that\s+one|this\s+one|that\s+stock|the\s+stock|they|them|this|that)\b", _I)

_NOT_TICKERS = {
    "AI", "API", "ATH", "ATM", "CEO", "CFO", "CPI", "CPU", "DOW", "EPS", "ESG", "EV", "FAQ",
    "FED", "FOMC", "GDP", "GPU", "HI", "IPO", "IRA", "IV", "LLM", "NASDAQ", "NYSE", "OK", "OKAY",
    "PE", "PM", "PPI", "REIT", "ROI", "SEC", "SPX", "TV", "UK", "US", "USA", "USD", "VIX", "YTD",
    "AM", "AN", "AND", "ARE", "CAN", "DID", "DO", "FOR", "HEY", "HOW", "IS", "IT", "OR", "THE", "TO",
    "UP", "VS", "WHY", "YOU", "WHAT", "WHO", "TVC",
}


def find_tickers(text):
    """Extract only explicit company names, $TICKER, or ALL-CAPS tickers (never random words)."""
    low = text.lower()
    hits = {}

    def add(ticker, pos):
        if ticker and (ticker not in hits or pos < hits[ticker]):
            hits[ticker] = pos

    for name in sorted(NAME_TO_TICKER, key=len, reverse=True):
        for m in re.finditer(rf"(?<![\w-]){re.escape(name)}(?![\w-])", low):
            add(NAME_TO_TICKER[name], m.start())

    for m in re.finditer(r"(?<![\w$])\$?([A-Za-z]{2,5}(?:[.-][A-Za-z])?)(?![\w])", text):
        tok, has_dollar = m.group(1), m.group(0).startswith("$")
        up = tok.upper()
        if has_dollar:
            add(up, m.start())
        elif tok.isupper() and not text.isupper() and up not in _NOT_TICKERS and tok.lower() not in NAME_TO_TICKER:
            add(up, m.start())

    return [t for t, _ in sorted(hits.items(), key=lambda kv: kv[1])][:MAX_TICKERS]


def _last_user_query(history):
    for m in reversed(list(history or [])):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _last_tickers(history):
    for m in reversed(list(history or [])):
        if m.get("role") == "user":
            found = find_tickers(m.get("content", ""))
            if found:
                return found
    return []


def route(text, history=()):
    t = " ".join((text or "").split())
    low = t.lower()

    if _CHAT_RE.match(low):
        return {"intent": "chat"}

    # Check if user is asking for top losers/gainers directly (or following up on that exact question)
    prev_q = _last_user_query(history)
    combined_low = f"{prev_q.lower()} {low}" if (_PRONOUN.search(low) and len(t.split()) <= 14) else low

    if _LOSERS_RE.search(combined_low) and not find_tickers(t):
        return {"intent": "data", "calls": [("get_top_movers", {"direction": "losers"})]}
    if _GAINERS_RE.search(combined_low) and not find_tickers(t):
        return {"intent": "data", "calls": [("get_top_movers", {"direction": "gainers"})]}

    tickers = find_tickers(t)
    market = bool(_MARKET_RE.search(low))

    if not tickers and not market and _PRONOUN.search(low):
        tickers = _last_tickers(history)

    calls = []
    if market:
        calls.append(("get_market_overview", {}))

    want_options = bool(_OPTIONS_CUE.search(low))
    want_hist = bool(_HIST_CUE.search(low))
    want_co = bool(_COMPANY_CUE.search(low))
    want_news = bool(_NEWS_WEB_CUE.search(low))

    for sym in tickers:
        if want_options:
            calls.append(("get_options_summary", {"symbol": sym}))
        elif want_hist:
            calls.append(("get_price_history", {"symbol": sym}))
        elif want_co:
            calls.append(("get_company_info", {"symbol": sym}))
        else:
            calls.append(("get_stock_quote", {"symbol": sym}))

    if want_news or (not calls and len(t.split()) >= 3):
        # Expand vague pronoun follow-ups with previous question context for better web search
        search_q = f"{prev_q} {t}".strip() if (_PRONOUN.search(low) and prev_q and not tickers) else t
        calls.append(("web_search", {"query": search_q}))

    if calls:
        return {"intent": "data", "calls": calls}

    return {"intent": "general"}

"""Finance tools the local LLM can call, backed by yfinance.

Every tool returns a small JSON-serializable dict. If yfinance is missing, the
network is down, or DEMO_MODE=1 is set, tools fall back to sample data from
demo_data.json and mark the result with "demo": true so the assistant says so.

The assistant must only state numbers returned by these tools.
"""
import json
import os
from datetime import datetime
from pathlib import Path

DEMO_FILE = Path(__file__).with_name("demo_data.json")
FORCE_DEMO = os.environ.get("DEMO_MODE", "0") == "1"

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
}

INDEX_SYMBOLS = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "VIX": "^VIX",
}

_demo_cache = None


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
        try:
            fi = _yf().Ticker(sym).fast_info
            price = fi["last_price"]
            prev = fi["previous_close"]
            if price is None or prev is None:
                raise ValueError("no price data")
            return {
                "symbol": sym,
                "price": _round(price),
                "previous_close": _round(prev),
                "change": _round(price - prev),
                "change_percent": _round((price - prev) / prev * 100),
                "currency": fi["currency"] if "currency" in fi else "USD",
                "as_of": _now(),
                "note": "Quotes can be delayed up to 15 minutes.",
            }
        except Exception as e:
            err = str(e)
    else:
        err = "demo mode"
    q = _demo()["quotes"].get(sym)
    if q is None:
        return {"error": f"No data for {sym}. Live lookup failed ({err}) and it is not in demo data."}
    return {"symbol": sym, **q, "currency": "USD", "demo": True,
            "note": "DEMO DATA, not real market prices."}


def get_price_history(symbol, period="1mo"):
    """Percent change over a period: 5d, 1mo, 3mo, 6mo, 1y, ytd."""
    sym = resolve_symbol(symbol)
    allowed = {"5d", "1mo", "3mo", "6mo", "1y", "ytd"}
    if period not in allowed:
        period = "1mo"
    if not FORCE_DEMO:
        try:
            hist = _yf().Ticker(sym).history(period=period)["Close"].dropna()
            if len(hist) < 2:
                raise ValueError("not enough history")
            start, end = float(hist.iloc[0]), float(hist.iloc[-1])
            return {
                "symbol": sym, "period": period,
                "start_price": _round(start), "end_price": _round(end),
                "change_percent": _round((end - start) / start * 100),
                "high": _round(hist.max()), "low": _round(hist.min()),
                "as_of": _now(),
            }
        except Exception as e:
            err = str(e)
    else:
        err = "demo mode"
    h = _demo()["history"].get(sym, {}).get(period)
    if h is None:
        return {"error": f"No {period} history for {sym}. Live lookup failed ({err})."}
    return {"symbol": sym, "period": period, **h, "demo": True,
            "note": "DEMO DATA, not real market prices."}


def get_company_info(symbol):
    sym = resolve_symbol(symbol)
    if not FORCE_DEMO:
        try:
            info = _yf().Ticker(sym).info
            if not info or "shortName" not in info:
                raise ValueError("no company info")
            return {
                "symbol": sym,
                "name": info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "trailing_pe": _round(info.get("trailingPE")),
                "as_of": _now(),
            }
        except Exception as e:
            err = str(e)
    else:
        err = "demo mode"
    c = _demo()["companies"].get(sym)
    if c is None:
        return {"error": f"No company info for {sym}. Live lookup failed ({err})."}
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
        "description": "Get a company's name, sector, market cap and P/E ratio. Use for 'what is X' or 'how big is X'.",
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


if __name__ == "__main__":
    print(json.dumps(get_stock_quote("nvidia"), indent=2))
    print(json.dumps(get_market_overview(), indent=2))

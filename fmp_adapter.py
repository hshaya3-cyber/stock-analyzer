"""
FMP Data Adapter
=================
Replaces yfinance with Financial Modeling Prep (FMP) API for reliable
stock data fetching, especially from cloud environments (Streamlit Cloud).

Uses the NEW stable API base: https://financialmodelingprep.com/stable/
Free tier: 250 requests/day, 5 years of historical data.
Sign up at: https://site.financialmodelingprep.com/
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── API Key ──────────────────────────────────────────────────────────────────
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")

_BASE = "https://financialmodelingprep.com/stable"

def _get(endpoint: str, params: dict = None) -> dict | list | None:
    """Make a GET request to FMP stable API."""
    if not FMP_API_KEY:
        raise ValueError(
            "FMP_API_KEY not set. Get a free key at "
            "https://site.financialmodelingprep.com/ and add it to "
            "Streamlit secrets or environment variables."
        )
    params = params or {}
    params["apikey"] = FMP_API_KEY
    url = f"{_BASE}/{endpoint.lstrip('/')}"
    resp = requests.get(url, params=params, timeout=30)
    # Handle plan-restricted endpoints gracefully
    if resp.status_code in (402, 403):
        print(f"  [FMP] Endpoint restricted (HTTP {resp.status_code}): {endpoint}")
        return None
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "Error Message" in data:
        print(f"  [FMP] API error: {data['Error Message']}")
        return None
    return data


# ─────────────────────────────────────────────────────────────────────────────
#  Historical Price Data → pandas DataFrame (same format as yfinance)
# ─────────────────────────────────────────────────────────────────────────────

def _period_to_dates(period: str) -> tuple:
    """Convert yfinance-style period string to (from_date, to_date)."""
    today = datetime.now()
    mapping = {
        "1mo": 30, "3mo": 90, "6mo": 180,
        "1y": 365, "2y": 730, "5y": 1825, "10y": 3650,
    }
    days = mapping.get(period, 365)
    from_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")
    return from_date, to_date


def fetch_historical(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch daily OHLCV data from FMP stable API.
    Returns DataFrame with DatetimeIndex and columns: Open, High, Low, Close, Volume.
    """
    from_date, to_date = _period_to_dates(period)

    # New stable endpoint
    data = _get("historical-price-eod/full", {
        "symbol": ticker,
        "from": from_date,
        "to": to_date,
    })

    # Handle both response formats (list or dict with "historical" key)
    if data is None:
        raise ValueError(f"No historical data for '{ticker}' (endpoint restricted or no data)")

    if isinstance(data, list):
        hist = data
    elif isinstance(data, dict) and "historical" in data:
        hist = data["historical"]
    else:
        raise ValueError(f"Unexpected response format for '{ticker}': {type(data)}")

    if not hist:
        raise ValueError(f"No historical data for '{ticker}'")

    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    # Keep only the standard columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"
    return df


def fetch_historical_interval(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Fetch data for different intervals (weekly, monthly).
    FMP free tier: fetch daily and resample.
    """
    df = fetch_historical(ticker, period)
    if df.empty:
        return df

    if interval in ("1wk", "weekly"):
        df = df.resample("W").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna()
    elif interval in ("1mo", "monthly"):
        df = df.resample("ME").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna()
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Company Info → dict (mapped to yfinance .info keys)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_info(ticker: str) -> dict:
    """
    Fetch company profile + key metrics + ratios + analyst targets from FMP.
    Returns a dict with the same keys as yfinance's Ticker.info.
    Gracefully handles restricted endpoints on the free plan.
    """
    info = {}

    # ── Company Profile ──
    try:
        profile_data = _get("profile", {"symbol": ticker})
        if profile_data and len(profile_data) > 0:
            p = profile_data[0]
            info.update({
                "longName": p.get("companyName"),
                "shortName": p.get("companyName"),
                "sector": p.get("sector"),
                "industry": p.get("industry"),
                "marketCap": p.get("mktCap"),
                "beta": p.get("beta"),
                "price": p.get("price"),
                "exchange": p.get("exchangeShortName"),
                "currency": p.get("currency"),
                "description": p.get("description"),
                "fullTimeEmployees": p.get("fullTimeEmployees"),
                "website": p.get("website"),
                "averageVolume": p.get("volAvg"),
                "fiftyTwoWeekHigh": p.get("range", "").split("-")[-1].strip() if p.get("range") else None,
                "fiftyTwoWeekLow": p.get("range", "").split("-")[0].strip() if p.get("range") else None,
                "dcfValue": p.get("dcf"),
            })
            for k in ("fiftyTwoWeekHigh", "fiftyTwoWeekLow"):
                try:
                    info[k] = float(info[k]) if info[k] else None
                except (ValueError, TypeError):
                    info[k] = None
    except Exception as e:
        print(f"  [FMP] Profile fetch failed: {e}")

    # ── Key Metrics (TTM) ──
    try:
        time.sleep(0.3)
        metrics = _get("key-metrics-ttm", {"symbol": ticker})
        if metrics and len(metrics) > 0:
            m = metrics[0]
            info.update({
                "trailingPE": m.get("peRatioTTM"),
                "priceToSalesTrailing12Months": m.get("priceToSalesRatioTTM"),
                "priceToBook": m.get("pbRatioTTM"),
                "enterpriseToEbitda": m.get("enterpriseValueOverEBITDATTM"),
                "dividendYield": m.get("dividendYieldTTM"),
                "pegRatio": m.get("pegRatioTTM"),
                "currentRatio": m.get("currentRatioTTM"),
                "debtToEquity": (m.get("debtToEquityTTM") or 0) * 100 if m.get("debtToEquityTTM") else None,
                "returnOnEquity": m.get("roeTTM"),
                "returnOnAssets": m.get("roaTTM"),
                "freeCashflow": m.get("freeCashFlowPerShareTTM"),
            })
    except Exception as e:
        print(f"  [FMP] Key metrics fetch failed: {e}")

    # ── Ratios (TTM) ──
    try:
        time.sleep(0.3)
        ratios = _get("ratios-ttm", {"symbol": ticker})
        if ratios and len(ratios) > 0:
            r = ratios[0]
            info.update({
                "profitMargins": r.get("netProfitMarginTTM"),
                "operatingMargins": r.get("operatingProfitMarginTTM"),
            })
            if not info.get("forwardPE"):
                info["forwardPE"] = r.get("priceEarningsToGrowthRatioTTM")
    except Exception as e:
        print(f"  [FMP] Ratios fetch failed: {e}")

    # ── Income Statement for revenue ──
    try:
        time.sleep(0.3)
        income = _get("income-statement", {"symbol": ticker, "period": "annual", "limit": 2})
        if income and len(income) > 0:
            latest = income[0]
            info["totalRevenue"] = latest.get("revenue")
            info["interestIncome"] = latest.get("interestIncome")
            info["interestExpense"] = latest.get("interestExpense")
            if len(income) >= 2 and income[1].get("revenue") and income[1]["revenue"] > 0:
                info["revenueGrowth"] = (
                    (income[0]["revenue"] - income[1]["revenue"]) / income[1]["revenue"]
                )
    except Exception as e:
        print(f"  [FMP] Income statement fetch failed: {e}")

    # ── Balance Sheet for halal screening ──
    try:
        time.sleep(0.3)
        bs = _get("balance-sheet-statement", {"symbol": ticker, "period": "annual", "limit": 1})
        if bs and len(bs) > 0:
            b = bs[0]
            info["totalDebt"] = b.get("totalDebt")
            info["totalCash"] = b.get("cashAndCashEquivalents")
    except Exception as e:
        print(f"  [FMP] Balance sheet fetch failed: {e}")

    # ── Analyst Price Targets ──
    try:
        time.sleep(0.3)
        targets = _get("price-target-consensus", {"symbol": ticker})
        if targets and len(targets) > 0:
            t = targets[0]
            info["targetMeanPrice"] = t.get("targetConsensus")
            info["targetHighPrice"] = t.get("targetHigh")
            info["targetLowPrice"] = t.get("targetLow")
    except Exception as e:
        print(f"  [FMP] Price targets fetch failed: {e}")

    # ── Analyst Recommendations ──
    try:
        time.sleep(0.3)
        recs = _get("analyst-stock-recommendations", {"symbol": ticker})
        if recs and len(recs) > 0:
            info["numberOfAnalystOpinions"] = len(recs)
            buy_count = sum(1 for r in recs[:20] if r.get("recommendationKey", "").lower() in ("buy", "strong_buy", "strong buy"))
            hold_count = sum(1 for r in recs[:20] if r.get("recommendationKey", "").lower() in ("hold", "neutral"))
            sell_count = sum(1 for r in recs[:20] if r.get("recommendationKey", "").lower() in ("sell", "strong_sell", "strong sell", "underperform"))
            total = buy_count + hold_count + sell_count
            if total > 0:
                if buy_count / total > 0.5:
                    info["recommendationKey"] = "buy"
                elif sell_count / total > 0.3:
                    info["recommendationKey"] = "sell"
                else:
                    info["recommendationKey"] = "hold"
                info["_analyst_breakdown"] = {
                    "strongBuy": sum(1 for r in recs[:20] if "strong" in r.get("recommendationKey", "").lower() and "buy" in r.get("recommendationKey", "").lower()),
                    "buy": buy_count,
                    "hold": hold_count,
                    "sell": sell_count,
                }
    except Exception as e:
        print(f"  [FMP] Recommendations fetch failed: {e}")

    # ── FMP Rating ──
    try:
        time.sleep(0.3)
        rating = _get("rating", {"symbol": ticker})
        if rating and len(rating) > 0:
            r = rating[0]
            fmp_rating = r.get("ratingRecommendation", "").lower()
            if not info.get("recommendationKey") and fmp_rating:
                if "buy" in fmp_rating:
                    info["recommendationKey"] = "buy"
                elif "sell" in fmp_rating:
                    info["recommendationKey"] = "sell"
                else:
                    info["recommendationKey"] = "hold"
    except Exception as e:
        print(f"  [FMP] Rating fetch failed: {e}")

    return info


# ─────────────────────────────────────────────────────────────────────────────
#  Combined fetch (replaces yfinance's fetch_data)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_data(ticker: str, period: str = "6mo") -> tuple:
    """
    Drop-in replacement for the yfinance-based fetch_data().
    Returns (df, info) — same as the original.
    """
    print(f"  [FMP] Fetching historical data for {ticker} ({period})...")
    df = fetch_historical(ticker, period)

    print(f"  [FMP] Fetching company info for {ticker}...")
    info = fetch_info(ticker)

    return df, info


# ─────────────────────────────────────────────────────────────────────────────
#  "Stock" object mimic (for functions that call stock.history())
# ─────────────────────────────────────────────────────────────────────────────

class FMPStock:
    """
    Mimics yf.Ticker interface for functions that call stock.history()
    (like analyze_multi_timeframe_trend and compute_multi_timeframe_evaluation).
    """
    def __init__(self, ticker: str):
        self.ticker = ticker

    def history(self, period="1y", interval="1d", auto_adjust=True, **kwargs):
        """Fetch history, resampling for weekly/monthly as needed."""
        if interval in ("1wk",):
            return fetch_historical_interval(self.ticker, period, "1wk")
        elif interval in ("1mo",):
            return fetch_historical_interval(self.ticker, period, "1mo")
        elif interval in ("5m", "15m", "30m", "60m"):
            # FMP free tier doesn't support intraday — return empty
            return pd.DataFrame()
        else:
            return fetch_historical(self.ticker, period)

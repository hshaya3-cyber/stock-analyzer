"""
Alpha Vantage Data Adapter
============================
Replaces yfinance with Alpha Vantage API for reliable stock data fetching
from cloud environments (Streamlit Cloud).

Free tier: 25 requests/day (enough for ~1-2 full stock analyses per day).
Get your free API key at: https://www.alphavantage.co/support/#api-key

Endpoints used (all free):
  - TIME_SERIES_DAILY (full history, 20+ years) — 1 call
  - TIME_SERIES_WEEKLY — 1 call
  - TIME_SERIES_MONTHLY — 1 call
  - OVERVIEW (fundamentals) — 1 call
  - INCOME_STATEMENT — 1 call
  - BALANCE_SHEET — 1 call
  Total per stock: ~6-10 calls depending on multi-timeframe needs
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── API Key ──────────────────────────────────────────────────────────────────
AV_API_KEY = os.environ.get("AV_API_KEY", "")

_BASE = "https://www.alphavantage.co/query"

def _get(params: dict) -> dict | None:
    """Make a GET request to Alpha Vantage API."""
    if not AV_API_KEY:
        raise ValueError(
            "AV_API_KEY not set. Get a free key at "
            "https://www.alphavantage.co/support/#api-key and add it to "
            "Streamlit secrets or environment variables."
        )
    params["apikey"] = AV_API_KEY
    resp = requests.get(_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "Error Message" in data:
        print(f"  [AV] Error: {data['Error Message']}")
        return None
    if "Note" in data:
        print(f"  [AV] Rate limit: {data['Note']}")
        return None
    if "Information" in data:
        print(f"  [AV] Info: {data['Information']}")
        return None
    return data


def _period_to_days(period: str) -> int:
    mapping = {
        "1mo": 30, "3mo": 90, "6mo": 180,
        "1y": 365, "2y": 730, "5y": 1825, "10y": 3650,
    }
    return mapping.get(period, 365)


def _parse_daily_data(data: dict, period: str = "1y") -> pd.DataFrame:
    ts_key = "Time Series (Daily)"
    if ts_key not in data:
        return pd.DataFrame()
    ts = data[ts_key]
    rows = []
    for date_str, vals in ts.items():
        rows.append({
            "Date": date_str,
            "Open": float(vals["1. open"]),
            "High": float(vals["2. high"]),
            "Low": float(vals["3. low"]),
            "Close": float(vals["4. close"]),
            "Volume": int(vals["5. volume"]),
        })
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    cutoff = datetime.now() - timedelta(days=_period_to_days(period))
    df = df[df.index >= cutoff]
    return df


def fetch_historical(ticker: str, period: str = "1y") -> pd.DataFrame:
    days = _period_to_days(period)
    outputsize = "full" if days > 100 else "compact"
    data = _get({
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": outputsize,
    })
    if data is None:
        raise ValueError(f"No data returned for '{ticker}' — check ticker or API limit")
    df = _parse_daily_data(data, period)
    if df.empty:
        raise ValueError(f"No historical data for '{ticker}'")
    return df


def _parse_weekly_monthly(data: dict, ts_key: str, period: str) -> pd.DataFrame:
    if ts_key not in data:
        return pd.DataFrame()
    ts = data[ts_key]
    rows = []
    for date_str, vals in ts.items():
        rows.append({
            "Date": date_str,
            "Open": float(vals["1. open"]),
            "High": float(vals["2. high"]),
            "Low": float(vals["3. low"]),
            "Close": float(vals["4. close"]),
            "Volume": int(vals["5. volume"]),
        })
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    cutoff = datetime.now() - timedelta(days=_period_to_days(period))
    df = df[df.index >= cutoff]
    return df


def fetch_historical_interval(ticker: str, period: str, interval: str) -> pd.DataFrame:
    time.sleep(1)
    if interval in ("1wk", "weekly"):
        data = _get({"function": "TIME_SERIES_WEEKLY", "symbol": ticker})
        if data is None:
            return pd.DataFrame()
        return _parse_weekly_monthly(data, "Weekly Time Series", period)
    elif interval in ("1mo", "monthly"):
        data = _get({"function": "TIME_SERIES_MONTHLY", "symbol": ticker})
        if data is None:
            return pd.DataFrame()
        return _parse_weekly_monthly(data, "Monthly Time Series", period)
    else:
        return fetch_historical(ticker, period)


def fetch_info(ticker: str) -> dict:
    info = {}

    # ── Company Overview ──
    try:
        time.sleep(1)
        ov = _get({"function": "OVERVIEW", "symbol": ticker})
        if ov:
            def _f(key, fallback=None):
                val = ov.get(key)
                if val is None or val == "None" or val == "-" or val == "":
                    return fallback
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return fallback

            def _s(key, fallback="N/A"):
                val = ov.get(key)
                return val if val and val != "None" and val != "-" else fallback

            info.update({
                "longName": _s("Name"),
                "shortName": _s("Name"),
                "sector": _s("Sector"),
                "industry": _s("Industry"),
                "marketCap": _f("MarketCapitalization"),
                "beta": _f("Beta"),
                "trailingPE": _f("TrailingPE"),
                "forwardPE": _f("ForwardPE"),
                "pegRatio": _f("PEGRatio"),
                "priceToSalesTrailing12Months": _f("PriceToSalesRatioTTM"),
                "priceToBook": _f("PriceToBookRatio"),
                "enterpriseToEbitda": _f("EVToEBITDA"),
                "totalRevenue": _f("RevenueTTM"),
                "revenueGrowth": _f("QuarterlyRevenueGrowthYOY"),
                "profitMargins": _f("ProfitMargin"),
                "operatingMargins": _f("OperatingMarginTTM"),
                "returnOnEquity": _f("ReturnOnEquityTTM"),
                "returnOnAssets": _f("ReturnOnAssetsTTM"),
                "dividendYield": _f("DividendYield"),
                "fiftyTwoWeekHigh": _f("52WeekHigh"),
                "fiftyTwoWeekLow": _f("52WeekLow"),
                "targetMeanPrice": _f("AnalystTargetPrice"),
                "averageVolume": _f("SharesOutstanding"),
                "description": _s("Description"),
                "exchange": _s("Exchange"),
                "currency": _s("Currency"),
            })

            rec = _s("AnalystRatingStrongBuy", None)
            if rec is not None:
                sb = int(_f("AnalystRatingStrongBuy", 0) or 0)
                b = int(_f("AnalystRatingBuy", 0) or 0)
                h = int(_f("AnalystRatingHold", 0) or 0)
                s = int(_f("AnalystRatingSell", 0) or 0)
                ss = int(_f("AnalystRatingStrongSell", 0) or 0)
                total = sb + b + h + s + ss
                info["numberOfAnalystOpinions"] = total
                info["_analyst_breakdown"] = {
                    "strongBuy": sb, "buy": b, "hold": h,
                    "sell": s, "strongSell": ss, "total": total,
                }
                if total > 0:
                    buy_pct = (sb + b) / total
                    sell_pct = (s + ss) / total
                    if buy_pct >= 0.6:
                        info["recommendationKey"] = "strong_buy" if sb > b else "buy"
                    elif sell_pct >= 0.6:
                        info["recommendationKey"] = "strong_sell" if ss > s else "sell"
                    elif buy_pct > sell_pct:
                        info["recommendationKey"] = "buy"
                    elif sell_pct > buy_pct:
                        info["recommendationKey"] = "sell"
                    else:
                        info["recommendationKey"] = "hold"
    except Exception as e:
        print(f"  [AV] Overview fetch failed: {e}")

    # ── Balance Sheet ──
    try:
        time.sleep(1)
        bs = _get({"function": "BALANCE_SHEET", "symbol": ticker})
        if bs and "annualReports" in bs and len(bs["annualReports"]) > 0:
            b = bs["annualReports"][0]
            def _bf(key):
                val = b.get(key)
                if val and val != "None":
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        pass
                return None
            info["totalDebt"] = _bf("shortLongTermDebtTotal") or _bf("longTermDebt")
            info["totalCash"] = _bf("cashAndCashEquivalentsAtCarryingValue") or _bf("cash")
            ca = _bf("totalCurrentAssets")
            cl = _bf("totalCurrentLiabilities")
            if ca and cl and cl > 0:
                info["currentRatio"] = ca / cl
            te = _bf("totalShareholderEquity")
            td = info.get("totalDebt")
            if td and te and te > 0:
                info["debtToEquity"] = (td / te) * 100
    except Exception as e:
        print(f"  [AV] Balance sheet fetch failed: {e}")

    # ── Income Statement ──
    try:
        time.sleep(1)
        inc = _get({"function": "INCOME_STATEMENT", "symbol": ticker})
        if inc and "annualReports" in inc and len(inc["annualReports"]) > 0:
            latest = inc["annualReports"][0]
            def _if(key):
                val = latest.get(key)
                if val and val != "None":
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        pass
                return None
            info["interestIncome"] = _if("interestIncome")
            info["interestExpense"] = _if("interestExpense")
            rev = _if("totalRevenue")
            if rev:
                info["totalRevenue"] = rev
            fcf = _if("operatingIncome")
            if fcf:
                info["freeCashflow"] = fcf
    except Exception as e:
        print(f"  [AV] Income statement fetch failed: {e}")

    return info


def fetch_data(ticker: str, period: str = "6mo") -> tuple:
    print(f"  [AV] Fetching historical data for {ticker} ({period})...")
    df = fetch_historical(ticker, period)
    print(f"  [AV] Fetching company info for {ticker}...")
    info = fetch_info(ticker)
    return df, info


class FMPStock:
    """
    Mimics yf.Ticker interface. Named FMPStock for backward compatibility
    with stock_analyzer.py imports. Uses Alpha Vantage under the hood.
    """
    def __init__(self, ticker: str):
        self.ticker = ticker

    def history(self, period="1y", interval="1d", auto_adjust=True, **kwargs):
        try:
            if interval in ("1wk",):
                return fetch_historical_interval(self.ticker, period, "1wk")
            elif interval in ("1mo",):
                return fetch_historical_interval(self.ticker, period, "1mo")
            elif interval in ("5m", "15m", "30m", "60m"):
                return pd.DataFrame()
            else:
                return fetch_historical(self.ticker, period)
        except Exception as e:
            print(f"  [AV] History fetch failed ({interval}): {e}")
            return pd.DataFrame()

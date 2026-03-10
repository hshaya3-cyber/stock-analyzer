"""
Stock Analysis Tool
===================
A comprehensive technical and fundamental analysis tool.
Usage: python stock_analyzer.py AAPL
       python stock_analyzer.py AAPL --period 1y
       python stock_analyzer.py AAPL MSFT NVDA  (multi-stock comparison)

Dependencies: pip install yfinance pandas numpy ta matplotlib reportlab
"""

import sys
import os
import json
import warnings
import time
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter, MonthLocator, WeekdayLocator
import ta

warnings.filterwarnings('ignore')

# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#0f3460',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#a0a0a0',
    'ytick.color': '#a0a0a0',
    'grid.color': '#0f3460',
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
    'font.size': 9,
})

BULL_COLOR = '#00e676'
BEAR_COLOR = '#ff1744'
NEUTRAL_COLOR = '#ffd740'
ACCENT_BLUE = '#448aff'
ACCENT_PURPLE = '#b388ff'


def fetch_data(ticker: str, period: str = '6mo') -> tuple:
    """Fetch stock data and info from Yahoo Finance."""
    # Retry logic to handle Yahoo Finance rate limiting
    max_retries = 3
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, auto_adjust=True)
            if df.empty:
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 5
                    print(f"  [RETRY] Empty data for {ticker}, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                raise ValueError(f"No data found for ticker '{ticker}'")
            info = stock.info or {}
            break
        except Exception as e:
            if "Too Many Requests" in str(e) or "Rate" in str(e):
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 8
                    print(f"  [RETRY] Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                    continue
            raise

    # ── Enhance analyst data from multiple yfinance endpoints ──
    # yf.Ticker.info often returns None or 'none' for analyst fields.
    # Try dedicated endpoints as fallback for more reliable data.

    def _is_missing(val):
        """Check if a value is effectively missing."""
        return val is None or val == 'none' or val == 'None' or val == '' or val == 0

    print(f"  [DEBUG] info.targetMeanPrice = {info.get('targetMeanPrice')}")
    print(f"  [DEBUG] info.recommendationKey = {info.get('recommendationKey')}")
    print(f"  [DEBUG] info.numberOfAnalystOpinions = {info.get('numberOfAnalystOpinions')}")

    # Fallback 1: analyst_price_targets
    if _is_missing(info.get('targetMeanPrice')):
        try:
            apt = stock.analyst_price_targets
            print(f"  [DEBUG] analyst_price_targets type={type(apt).__name__}, value={apt}")
            if apt is not None:
                if isinstance(apt, dict):
                    info['targetMeanPrice'] = apt.get('mean') or apt.get('current')
                    info['targetHighPrice'] = apt.get('high')
                    info['targetLowPrice'] = apt.get('low')
                    n = apt.get('numberOfAnalystOpinions')
                    if n:
                        info['numberOfAnalystOpinions'] = n
                elif isinstance(apt, pd.DataFrame) and not apt.empty:
                    if 'mean' in apt.columns:
                        info['targetMeanPrice'] = apt['mean'].iloc[0]
                    elif 'current' in apt.columns:
                        info['targetMeanPrice'] = apt['current'].iloc[0]
                elif isinstance(apt, pd.Series):
                    info['targetMeanPrice'] = apt.get('mean') or apt.get('current')
                    info['targetHighPrice'] = apt.get('high')
                    info['targetLowPrice'] = apt.get('low')
            print(f"  [DEBUG] After apt: targetMeanPrice = {info.get('targetMeanPrice')}")
        except Exception as e:
            print(f"  [DEBUG] analyst_price_targets failed: {e}")

    # Fallback 2: recommendations_summary for rating & breakdown
    # Always try this to get the breakdown, even if we have a rating
    try:
        rec = stock.recommendations_summary
        print(f"  [DEBUG] recommendations_summary type={type(rec).__name__}")
        if rec is not None and isinstance(rec, pd.DataFrame) and not rec.empty:
            print(f"  [DEBUG] recommendations_summary columns={list(rec.columns)}")
            print(f"  [DEBUG] recommendations_summary:\n{rec.head()}")
            latest = rec.iloc[0]
            sb = int(latest.get('strongBuy', 0) or 0)
            b = int(latest.get('buy', 0) or 0)
            h = int(latest.get('hold', 0) or 0)
            s = int(latest.get('sell', 0) or 0)
            ss = int(latest.get('strongSell', 0) or 0)
            total = sb + b + h + s + ss
            if total > 0:
                if _is_missing(info.get('numberOfAnalystOpinions')):
                    info['numberOfAnalystOpinions'] = total
                buy_pct = (sb + b) / total
                sell_pct = (s + ss) / total
                if _is_missing(info.get('recommendationKey')):
                    if buy_pct >= 0.6:
                        info['recommendationKey'] = 'strong_buy' if sb > b else 'buy'
                    elif sell_pct >= 0.6:
                        info['recommendationKey'] = 'strong_sell' if ss > s else 'sell'
                    elif buy_pct > sell_pct:
                        info['recommendationKey'] = 'buy'
                    elif sell_pct > buy_pct:
                        info['recommendationKey'] = 'sell'
                    else:
                        info['recommendationKey'] = 'hold'
                info['_analyst_breakdown'] = {
                    'strongBuy': sb, 'buy': b, 'hold': h,
                    'sell': s, 'strongSell': ss, 'total': total
                }
                print(f"  [DEBUG] After rec_summary: rating={info.get('recommendationKey')}, total={total}")
    except Exception as e:
        print(f"  [DEBUG] recommendations_summary failed: {e}")

    # Fallback 3: recommendations (individual analyst actions) for count
    if _is_missing(info.get('numberOfAnalystOpinions')):
        try:
            recs = stock.recommendations
            print(f"  [DEBUG] recommendations type={type(recs).__name__}")
            if recs is not None and isinstance(recs, pd.DataFrame) and not recs.empty:
                from datetime import timedelta
                cutoff = pd.Timestamp.now() - timedelta(days=90)
                try:
                    if hasattr(recs.index, 'tz') and recs.index.tz is not None:
                        cutoff = cutoff.tz_localize(recs.index.tz)
                    recent = recs[recs.index >= cutoff]
                except Exception:
                    recent = recs.tail(20)
                info['numberOfAnalystOpinions'] = len(recent) if not recent.empty else len(recs)
                print(f"  [DEBUG] After recommendations: count={info['numberOfAnalystOpinions']}")
        except Exception as e:
            print(f"  [DEBUG] recommendations failed: {e}")

    # Fallback 4: upgrades_downgrades for additional context
    if _is_missing(info.get('numberOfAnalystOpinions')):
        try:
            ud = stock.upgrades_downgrades
            if ud is not None and isinstance(ud, pd.DataFrame) and not ud.empty:
                from datetime import timedelta
                cutoff = pd.Timestamp.now() - timedelta(days=90)
                try:
                    if hasattr(ud.index, 'tz') and ud.index.tz is not None:
                        cutoff = cutoff.tz_localize(ud.index.tz)
                    recent = ud[ud.index >= cutoff]
                except Exception:
                    recent = ud.tail(20)
                info['numberOfAnalystOpinions'] = len(recent) if not recent.empty else len(ud)
                print(f"  [DEBUG] After upgrades_downgrades: count={info['numberOfAnalystOpinions']}")
        except Exception as e:
            print(f"  [DEBUG] upgrades_downgrades failed: {e}")

    print(f"  [FINAL] targetMeanPrice={info.get('targetMeanPrice')}, "
          f"rating={info.get('recommendationKey')}, "
          f"analysts={info.get('numberOfAnalystOpinions')}")

    return df, info, stock


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators."""
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # ── Moving Averages ──
    df['SMA_20'] = ta.trend.sma_indicator(close, window=20)
    df['SMA_50'] = ta.trend.sma_indicator(close, window=50)
    df['SMA_200'] = ta.trend.sma_indicator(close, window=200)
    df['EMA_9'] = ta.trend.ema_indicator(close, window=9)
    df['EMA_21'] = ta.trend.ema_indicator(close, window=21)

    # ── Bollinger Bands ──
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Width'] = bb.bollinger_wband()
    df['BB_Pct'] = bb.bollinger_pband()

    # ── Keltner Channels (1.5× ATR, EMA-based — standard for TTM Squeeze) ──
    kc = ta.volatility.KeltnerChannel(high, low, close, window=20, window_atr=10,
                                       original_version=False, multiplier=1.5)
    df['KC_Upper'] = kc.keltner_channel_hband()
    df['KC_Lower'] = kc.keltner_channel_lband()
    df['KC_Middle'] = kc.keltner_channel_mband()

    # ── Squeeze Detection (BB inside KC) ──
    df['Squeeze'] = (df['BB_Upper'] < df['KC_Upper']) & (df['BB_Lower'] > df['KC_Lower'])

    # ── Squeeze Strength (BB width relative to KC width) ──
    df['KC_Width'] = df['KC_Upper'] - df['KC_Lower']
    df['BB_KC_Ratio'] = df['BB_Width'] / (df['KC_Width'] + 1e-10)  # How tight BB is vs KC
    # Tight squeeze: BB width is very narrow AND inside KC
    bb_width_pctl = df['BB_Width'].rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    df['BB_Width_Pctl'] = bb_width_pctl
    df['Squeeze_Tight'] = df['Squeeze'] & (df['BB_Width_Pctl'] < 0.2)  # Bottom 20% of BB width

    # ── RSI ──
    df['RSI'] = ta.momentum.rsi(close, window=14)

    # ── RSI (4) - Short-term ──
    df['RSI_4'] = ta.momentum.rsi(close, window=4)

    # ── CCI (20) and CCI (6) ──
    df['CCI_20'] = ta.trend.cci(high, low, close, window=20)
    df['CCI_6'] = ta.trend.cci(high, low, close, window=6)

    # ── Stochastic RSI ──
    stoch_rsi = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    df['StochRSI_K'] = stoch_rsi.stochrsi_k() * 100
    df['StochRSI_D'] = stoch_rsi.stochrsi_d() * 100

    # ── MACD ──
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()

    # ── ADX ──
    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    df['ADX'] = adx.adx()
    df['DI_Plus'] = adx.adx_pos()
    df['DI_Minus'] = adx.adx_neg()

    # ── ATR ──
    df['ATR'] = ta.volatility.average_true_range(high, low, close, window=14)
    df['ATR_Pct'] = (df['ATR'] / close) * 100

    # ── Volume ──
    df['Vol_SMA_20'] = ta.trend.sma_indicator(volume.astype(float), window=20)
    df['Vol_SMA_63'] = ta.trend.sma_indicator(volume.astype(float), window=63)
    df['OBV'] = ta.volume.on_balance_volume(close, volume)
    df['VWAP'] = ta.volume.volume_weighted_average_price(high, low, close, volume)

    # ── MFI (Money Flow Index) ──
    df['MFI'] = ta.volume.money_flow_index(high, low, close, volume, window=14)
    df['MFI_6'] = ta.volume.money_flow_index(high, low, close, volume, window=6)

    # ── Ichimoku ──
    ichi = ta.trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
    df['Ichi_Tenkan'] = ichi.ichimoku_conversion_line()
    df['Ichi_Kijun'] = ichi.ichimoku_base_line()
    df['Ichi_SpanA'] = ichi.ichimoku_a()
    df['Ichi_SpanB'] = ichi.ichimoku_b()

    return df


def detect_support_resistance(df: pd.DataFrame, lookback: int = 60) -> dict:
    """Detect key support and resistance levels using pivot points."""
    recent = df.tail(lookback)
    close = recent['Close'].values
    high = recent['High'].values
    low = recent['Low'].values
    current_price = close[-1]

    levels = []

    # Local highs and lows
    for i in range(2, len(recent) - 2):
        if low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i+1] and low[i] < low[i+2]:
            levels.append(('support', low[i]))
        if high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i+1] and high[i] > high[i+2]:
            levels.append(('resistance', high[i]))

    # Cluster nearby levels (within 1.5%)
    clustered = []
    used = set()
    sorted_levels = sorted(levels, key=lambda x: x[1])
    for i, (type1, price1) in enumerate(sorted_levels):
        if i in used:
            continue
        cluster = [price1]
        for j, (type2, price2) in enumerate(sorted_levels[i+1:], i+1):
            if j in used:
                continue
            if abs(price2 - price1) / price1 < 0.015:
                cluster.append(price2)
                used.add(j)
        avg_price = np.mean(cluster)
        level_type = 'support' if avg_price < current_price else 'resistance'
        strength = len(cluster)
        clustered.append({'type': level_type, 'price': avg_price, 'strength': strength})
        used.add(i)

    supports = sorted([l for l in clustered if l['type'] == 'support'], key=lambda x: x['price'], reverse=True)[:3]
    resistances = sorted([l for l in clustered if l['type'] == 'resistance'], key=lambda x: x['price'])[:3]

    return {'supports': supports, 'resistances': resistances}


def detect_divergences(df: pd.DataFrame, lookback: int = 60, swing_window: int = 5) -> list:
    """
    Detect divergences between RSI and Price, MACD, MFI, and OBV.

    Divergence types detected:
      - Regular Bullish: Price makes lower low, indicator makes higher low → reversal up
      - Regular Bearish: Price makes higher high, indicator makes lower high → reversal down
      - Hidden Bullish:  Price makes higher low, indicator makes lower low → continuation up
      - Hidden Bearish:  Price makes lower high, indicator makes higher high → continuation down

    Checks RSI against: Price (classic), MACD Histogram, MFI, OBV
    """
    recent = df.tail(lookback).copy()
    recent = recent.dropna(subset=['RSI', 'MACD_Hist', 'MFI', 'OBV', 'RSI_4', 'MFI_6'])
    if len(recent) < swing_window * 4:
        return []

    close = recent['Close'].values
    idx = recent.index

    def find_swing_lows(series, window=swing_window):
        """Find local minima."""
        lows = []
        for i in range(window, len(series) - window):
            if series[i] == min(series[i - window:i + window + 1]):
                lows.append(i)
        return lows

    def find_swing_highs(series, window=swing_window):
        """Find local maxima."""
        highs = []
        for i in range(window, len(series) - window):
            if series[i] == max(series[i - window:i + window + 1]):
                highs.append(i)
        return highs

    divergences = []

    # ── RSI vs Price (Classic Divergence) ──
    rsi = recent['RSI'].values
    price_lows = find_swing_lows(close)
    price_highs = find_swing_highs(close)
    rsi_lows = find_swing_lows(rsi)
    rsi_highs = find_swing_highs(rsi)

    # Regular Bullish: price lower low + RSI higher low
    for i in range(len(price_lows) - 1):
        p1, p2 = price_lows[-2], price_lows[-1]
        if close[p2] < close[p1]:  # Price made lower low
            # Find RSI lows near these price lows
            r1_candidates = [r for r in rsi_lows if abs(r - p1) <= swing_window]
            r2_candidates = [r for r in rsi_lows if abs(r - p2) <= swing_window]
            if r1_candidates and r2_candidates:
                r1 = min(r1_candidates, key=lambda r: abs(r - p1))
                r2 = min(r2_candidates, key=lambda r: abs(r - p2))
                if rsi[r2] > rsi[r1]:  # RSI made higher low
                    divergences.append({
                        'type': 'REGULAR BULLISH',
                        'pair': 'RSI vs Price',
                        'description': f'Price lower low (${close[p2]:.2f} < ${close[p1]:.2f}) but RSI higher low ({rsi[r2]:.1f} > {rsi[r1]:.1f})',
                        'signal': 'Potential reversal UP',
                        'strength': 'STRONG',
                        'date_start': str(idx[p1].date()),
                        'date_end': str(idx[p2].date()),
                    })
        break  # Only check most recent pair

    # Regular Bearish: price higher high + RSI lower high
    for i in range(len(price_highs) - 1):
        p1, p2 = price_highs[-2], price_highs[-1]
        if close[p2] > close[p1]:  # Price made higher high
            r1_candidates = [r for r in rsi_highs if abs(r - p1) <= swing_window]
            r2_candidates = [r for r in rsi_highs if abs(r - p2) <= swing_window]
            if r1_candidates and r2_candidates:
                r1 = min(r1_candidates, key=lambda r: abs(r - p1))
                r2 = min(r2_candidates, key=lambda r: abs(r - p2))
                if rsi[r2] < rsi[r1]:  # RSI made lower high
                    divergences.append({
                        'type': 'REGULAR BEARISH',
                        'pair': 'RSI vs Price',
                        'description': f'Price higher high (${close[p2]:.2f} > ${close[p1]:.2f}) but RSI lower high ({rsi[r2]:.1f} < {rsi[r1]:.1f})',
                        'signal': 'Potential reversal DOWN',
                        'strength': 'STRONG',
                        'date_start': str(idx[p1].date()),
                        'date_end': str(idx[p2].date()),
                    })
        break

    # Hidden Bullish: price higher low + RSI lower low (trend continuation)
    if len(price_lows) >= 2:
        p1, p2 = price_lows[-2], price_lows[-1]
        if close[p2] > close[p1]:  # Price higher low
            r1_candidates = [r for r in rsi_lows if abs(r - p1) <= swing_window]
            r2_candidates = [r for r in rsi_lows if abs(r - p2) <= swing_window]
            if r1_candidates and r2_candidates:
                r1 = min(r1_candidates, key=lambda r: abs(r - p1))
                r2 = min(r2_candidates, key=lambda r: abs(r - p2))
                if rsi[r2] < rsi[r1]:  # RSI lower low
                    divergences.append({
                        'type': 'HIDDEN BULLISH',
                        'pair': 'RSI vs Price',
                        'description': f'Price higher low (${close[p2]:.2f} > ${close[p1]:.2f}) but RSI lower low ({rsi[r2]:.1f} < {rsi[r1]:.1f})',
                        'signal': 'Uptrend continuation expected',
                        'strength': 'MODERATE',
                        'date_start': str(idx[p1].date()),
                        'date_end': str(idx[p2].date()),
                    })

    # Hidden Bearish: price lower high + RSI higher high (trend continuation)
    if len(price_highs) >= 2:
        p1, p2 = price_highs[-2], price_highs[-1]
        if close[p2] < close[p1]:  # Price lower high
            r1_candidates = [r for r in rsi_highs if abs(r - p1) <= swing_window]
            r2_candidates = [r for r in rsi_highs if abs(r - p2) <= swing_window]
            if r1_candidates and r2_candidates:
                r1 = min(r1_candidates, key=lambda r: abs(r - p1))
                r2 = min(r2_candidates, key=lambda r: abs(r - p2))
                if rsi[r2] > rsi[r1]:  # RSI higher high
                    divergences.append({
                        'type': 'HIDDEN BEARISH',
                        'pair': 'RSI vs Price',
                        'description': f'Price lower high (${close[p2]:.2f} < ${close[p1]:.2f}) but RSI higher high ({rsi[r2]:.1f} > {rsi[r1]:.1f})',
                        'signal': 'Downtrend continuation expected',
                        'strength': 'MODERATE',
                        'date_start': str(idx[p1].date()),
                        'date_end': str(idx[p2].date()),
                    })

    # ── RSI vs MACD Histogram Divergence ──
    macd_h = recent['MACD_Hist'].values
    macd_lows = find_swing_lows(macd_h)
    macd_highs = find_swing_highs(macd_h)

    if len(rsi_lows) >= 2 and len(macd_lows) >= 2:
        r1, r2 = rsi_lows[-2], rsi_lows[-1]
        m1_candidates = [m for m in macd_lows if abs(m - r1) <= swing_window]
        m2_candidates = [m for m in macd_lows if abs(m - r2) <= swing_window]
        if m1_candidates and m2_candidates:
            m1 = min(m1_candidates, key=lambda m: abs(m - r1))
            m2 = min(m2_candidates, key=lambda m: abs(m - r2))
            if rsi[r2] > rsi[r1] and macd_h[m2] < macd_h[m1]:
                divergences.append({
                    'type': 'DIVERGENCE',
                    'pair': 'RSI vs MACD',
                    'description': f'RSI rising ({rsi[r1]:.1f}→{rsi[r2]:.1f}) while MACD Hist falling ({macd_h[m1]:.4f}→{macd_h[m2]:.4f})',
                    'signal': 'Momentum conflict - RSI bullish, MACD weakening',
                    'strength': 'MODERATE',
                    'date_start': str(idx[r1].date()),
                    'date_end': str(idx[r2].date()),
                })
            elif rsi[r2] < rsi[r1] and macd_h[m2] > macd_h[m1]:
                divergences.append({
                    'type': 'DIVERGENCE',
                    'pair': 'RSI vs MACD',
                    'description': f'RSI falling ({rsi[r1]:.1f}→{rsi[r2]:.1f}) while MACD Hist rising ({macd_h[m1]:.4f}→{macd_h[m2]:.4f})',
                    'signal': 'Momentum conflict - RSI bearish, MACD strengthening',
                    'strength': 'MODERATE',
                    'date_start': str(idx[r1].date()),
                    'date_end': str(idx[r2].date()),
                })

    if len(rsi_highs) >= 2 and len(macd_highs) >= 2:
        r1, r2 = rsi_highs[-2], rsi_highs[-1]
        m1_candidates = [m for m in macd_highs if abs(m - r1) <= swing_window]
        m2_candidates = [m for m in macd_highs if abs(m - r2) <= swing_window]
        if m1_candidates and m2_candidates:
            m1 = min(m1_candidates, key=lambda m: abs(m - r1))
            m2 = min(m2_candidates, key=lambda m: abs(m - r2))
            if rsi[r2] > rsi[r1] and macd_h[m2] < macd_h[m1]:
                divergences.append({
                    'type': 'DIVERGENCE',
                    'pair': 'RSI vs MACD (highs)',
                    'description': f'RSI higher high ({rsi[r1]:.1f}→{rsi[r2]:.1f}) but MACD Hist lower high ({macd_h[m1]:.4f}→{macd_h[m2]:.4f})',
                    'signal': 'Bearish momentum divergence at highs',
                    'strength': 'STRONG',
                    'date_start': str(idx[r1].date()),
                    'date_end': str(idx[r2].date()),
                })

    # ── RSI vs MFI Divergence ──
    mfi = recent['MFI'].values
    mfi_lows = find_swing_lows(mfi)
    mfi_highs = find_swing_highs(mfi)

    if len(rsi_lows) >= 2 and len(mfi_lows) >= 2:
        r1, r2 = rsi_lows[-2], rsi_lows[-1]
        f1_candidates = [f for f in mfi_lows if abs(f - r1) <= swing_window]
        f2_candidates = [f for f in mfi_lows if abs(f - r2) <= swing_window]
        if f1_candidates and f2_candidates:
            f1 = min(f1_candidates, key=lambda f: abs(f - r1))
            f2 = min(f2_candidates, key=lambda f: abs(f - r2))
            if (rsi[r2] > rsi[r1]) != (mfi[f2] > mfi[f1]):
                direction = 'RSI bullish / MFI bearish' if rsi[r2] > rsi[r1] else 'RSI bearish / MFI bullish'
                divergences.append({
                    'type': 'DIVERGENCE',
                    'pair': 'RSI vs MFI',
                    'description': f'RSI lows ({rsi[r1]:.1f}→{rsi[r2]:.1f}) vs MFI lows ({mfi[f1]:.1f}→{mfi[f2]:.1f})',
                    'signal': f'Volume-price disagreement - {direction}',
                    'strength': 'STRONG' if abs(rsi[r2] - rsi[r1]) > 5 else 'MODERATE',
                    'date_start': str(idx[r1].date()),
                    'date_end': str(idx[r2].date()),
                })

    if len(rsi_highs) >= 2 and len(mfi_highs) >= 2:
        r1, r2 = rsi_highs[-2], rsi_highs[-1]
        f1_candidates = [f for f in mfi_highs if abs(f - r1) <= swing_window]
        f2_candidates = [f for f in mfi_highs if abs(f - r2) <= swing_window]
        if f1_candidates and f2_candidates:
            f1 = min(f1_candidates, key=lambda f: abs(f - r1))
            f2 = min(f2_candidates, key=lambda f: abs(f - r2))
            if (rsi[r2] > rsi[r1]) != (mfi[f2] > mfi[f1]):
                direction = 'RSI rising / MFI falling' if rsi[r2] > rsi[r1] else 'RSI falling / MFI rising'
                divergences.append({
                    'type': 'DIVERGENCE',
                    'pair': 'RSI vs MFI (highs)',
                    'description': f'RSI highs ({rsi[r1]:.1f}→{rsi[r2]:.1f}) vs MFI highs ({mfi[f1]:.1f}→{mfi[f2]:.1f})',
                    'signal': f'Volume-price disagreement - {direction}',
                    'strength': 'STRONG' if abs(rsi[r2] - rsi[r1]) > 5 else 'MODERATE',
                    'date_start': str(idx[r1].date()),
                    'date_end': str(idx[r2].date()),
                })

    # ── RSI vs OBV Divergence ──
    obv = recent['OBV'].values
    # Normalize OBV for comparison
    obv_norm = (obv - obv.min()) / (obv.max() - obv.min() + 1e-10) * 100
    obv_lows = find_swing_lows(obv_norm)
    obv_highs = find_swing_highs(obv_norm)

    if len(rsi_lows) >= 2 and len(obv_lows) >= 2:
        r1, r2 = rsi_lows[-2], rsi_lows[-1]
        o1_candidates = [o for o in obv_lows if abs(o - r1) <= swing_window]
        o2_candidates = [o for o in obv_lows if abs(o - r2) <= swing_window]
        if o1_candidates and o2_candidates:
            o1 = min(o1_candidates, key=lambda o: abs(o - r1))
            o2 = min(o2_candidates, key=lambda o: abs(o - r2))
            if (rsi[r2] > rsi[r1]) != (obv[o2] > obv[o1]):
                direction = 'RSI rising / OBV falling' if rsi[r2] > rsi[r1] else 'RSI falling / OBV rising'
                divergences.append({
                    'type': 'DIVERGENCE',
                    'pair': 'RSI vs OBV',
                    'description': f'RSI and OBV moving in opposite directions at swing lows',
                    'signal': f'Accumulation/distribution mismatch - {direction}',
                    'strength': 'MODERATE',
                    'date_start': str(idx[r1].date()),
                    'date_end': str(idx[r2].date()),
                })

    # ── RSI(4) vs MFI(6) Divergence (Short-term) ──
    rsi4 = recent['RSI_4'].values
    mfi6 = recent['MFI_6'].values
    rsi4_lows = find_swing_lows(rsi4)
    rsi4_highs = find_swing_highs(rsi4)
    mfi6_lows = find_swing_lows(mfi6)
    mfi6_highs = find_swing_highs(mfi6)

    # RSI(4) vs MFI(6) at lows - classic swing divergence
    if len(rsi4_lows) >= 2 and len(mfi6_lows) >= 2:
        r1, r2 = rsi4_lows[-2], rsi4_lows[-1]
        f1_candidates = [f for f in mfi6_lows if abs(f - r1) <= swing_window]
        f2_candidates = [f for f in mfi6_lows if abs(f - r2) <= swing_window]
        if f1_candidates and f2_candidates:
            f1 = min(f1_candidates, key=lambda f: abs(f - r1))
            f2 = min(f2_candidates, key=lambda f: abs(f - r2))
            if rsi4[r2] < rsi4[r1] and mfi6[f2] > mfi6[f1]:
                divergences.append({
                    'type': 'BULLISH DIVERGENCE',
                    'pair': 'RSI(4) vs MFI(6)',
                    'description': f'RSI(4) lower low ({rsi4[r1]:.1f}->{rsi4[r2]:.1f}) but MFI(6) higher low ({mfi6[f1]:.1f}->{mfi6[f2]:.1f})',
                    'signal': 'Smart money accumulating despite price weakness - short-term reversal up likely',
                    'strength': 'STRONG' if abs(rsi4[r2] - rsi4[r1]) > 8 else 'MODERATE',
                    'date_start': str(idx[r1].date()),
                    'date_end': str(idx[r2].date()),
                })
            elif rsi4[r2] > rsi4[r1] and mfi6[f2] < mfi6[f1]:
                divergences.append({
                    'type': 'BEARISH DIVERGENCE',
                    'pair': 'RSI(4) vs MFI(6)',
                    'description': f'RSI(4) higher low ({rsi4[r1]:.1f}->{rsi4[r2]:.1f}) but MFI(6) lower low ({mfi6[f1]:.1f}->{mfi6[f2]:.1f})',
                    'signal': 'Volume drying up despite price strength - short-term pullback likely',
                    'strength': 'STRONG' if abs(rsi4[r2] - rsi4[r1]) > 8 else 'MODERATE',
                    'date_start': str(idx[r1].date()),
                    'date_end': str(idx[r2].date()),
                })

    # RSI(4) vs MFI(6) at highs
    if len(rsi4_highs) >= 2 and len(mfi6_highs) >= 2:
        r1, r2 = rsi4_highs[-2], rsi4_highs[-1]
        f1_candidates = [f for f in mfi6_highs if abs(f - r1) <= swing_window]
        f2_candidates = [f for f in mfi6_highs if abs(f - r2) <= swing_window]
        if f1_candidates and f2_candidates:
            f1 = min(f1_candidates, key=lambda f: abs(f - r1))
            f2 = min(f2_candidates, key=lambda f: abs(f - r2))
            if rsi4[r2] > rsi4[r1] and mfi6[f2] < mfi6[f1]:
                divergences.append({
                    'type': 'BEARISH DIVERGENCE',
                    'pair': 'RSI(4) vs MFI(6) (highs)',
                    'description': f'RSI(4) higher high ({rsi4[r1]:.1f}->{rsi4[r2]:.1f}) but MFI(6) lower high ({mfi6[f1]:.1f}->{mfi6[f2]:.1f})',
                    'signal': 'Price pushing higher but money flow weakening - distribution at top',
                    'strength': 'STRONG' if abs(rsi4[r2] - rsi4[r1]) > 8 else 'MODERATE',
                    'date_start': str(idx[r1].date()),
                    'date_end': str(idx[r2].date()),
                })
            elif rsi4[r2] < rsi4[r1] and mfi6[f2] > mfi6[f1]:
                divergences.append({
                    'type': 'BULLISH DIVERGENCE',
                    'pair': 'RSI(4) vs MFI(6) (highs)',
                    'description': f'RSI(4) lower high ({rsi4[r1]:.1f}->{rsi4[r2]:.1f}) but MFI(6) higher high ({mfi6[f1]:.1f}->{mfi6[f2]:.1f})',
                    'signal': 'Money flowing in despite weaker price momentum - accumulation',
                    'strength': 'STRONG' if abs(rsi4[r2] - rsi4[r1]) > 8 else 'MODERATE',
                    'date_start': str(idx[r1].date()),
                    'date_end': str(idx[r2].date()),
                })

    # ── MFI(6) Rising / RSI(4) Lagging or Falling (Momentum Pace Analysis) ──
    # Look at the last 10 bars for recent directional pace mismatch
    if len(recent) >= 10:
        last_n = 10
        rsi4_recent = rsi4[-last_n:]
        mfi6_recent = mfi6[-last_n:]

        # Calculate slopes using linear regression
        x = np.arange(last_n)
        rsi4_slope = np.polyfit(x, rsi4_recent, 1)[0]
        mfi6_slope = np.polyfit(x, mfi6_recent, 1)[0]

        # MFI rising strongly, RSI flat or falling
        mfi6_rising = mfi6_slope > 1.0  # MFI gaining more than 1 point per bar
        rsi4_lagging = rsi4_slope < mfi6_slope * 0.3  # RSI pace less than 30% of MFI pace

        if mfi6_rising and rsi4_lagging:
            if rsi4_slope < 0:
                pace_desc = 'FALLING'
                pace_signal = 'MFI(6) rising strongly while RSI(4) declining - volume-driven move not confirmed by price momentum. Watch for breakout or fakeout.'
                pace_strength = 'STRONG'
            else:
                pace_desc = 'LAGGING'
                pace_signal = 'MFI(6) rising faster than RSI(4) - money flow leading price. Potential bullish acceleration ahead.'
                pace_strength = 'MODERATE'

            divergences.append({
                'type': f'PACE MISMATCH (RSI {pace_desc})',
                'pair': 'MFI(6) vs RSI(4)',
                'description': f'MFI(6) slope: +{mfi6_slope:.2f}/bar | RSI(4) slope: {rsi4_slope:+.2f}/bar over last {last_n} bars',
                'signal': pace_signal,
                'strength': pace_strength,
                'date_start': str(idx[-last_n].date()),
                'date_end': str(idx[-1].date()),
            })

        # Opposite: RSI rising strongly, MFI flat or falling
        rsi4_rising = rsi4_slope > 1.0
        mfi6_lagging = mfi6_slope < rsi4_slope * 0.3

        if rsi4_rising and mfi6_lagging:
            if mfi6_slope < 0:
                pace_desc = 'FALLING'
                pace_signal = 'RSI(4) rising but MFI(6) declining - price momentum without volume support. Rally may lack conviction.'
                pace_strength = 'STRONG'
            else:
                pace_desc = 'LAGGING'
                pace_signal = 'RSI(4) rising faster than MFI(6) - price leading volume. Move may be unsustainable without volume confirmation.'
                pace_strength = 'MODERATE'

            divergences.append({
                'type': f'PACE MISMATCH (MFI {pace_desc})',
                'pair': 'RSI(4) vs MFI(6)',
                'description': f'RSI(4) slope: +{rsi4_slope:.2f}/bar | MFI(6) slope: {mfi6_slope:+.2f}/bar over last {last_n} bars',
                'signal': pace_signal,
                'strength': pace_strength,
                'date_start': str(idx[-last_n].date()),
                'date_end': str(idx[-1].date()),
            })

    return divergences


def analyze_fibonacci(df: pd.DataFrame, lookback: int = 120) -> dict:
    """
    Analyze Fibonacci Retracement and Extension levels.

    Automatically detects the major swing high/low in the lookback period,
    determines if the current move is a retracement (pullback) or extension
    (breakout beyond prior range), and calculates the appropriate levels.

    Fibonacci Retracement: Used when price is pulling back from a trend.
      Levels: 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%

    Fibonacci Extension: Used when price breaks beyond the prior high/low.
      Levels: 100%, 123.6%, 138.2%, 150%, 161.8%, 200%, 261.8%
    """
    recent = df.tail(lookback).copy()
    if len(recent) < 20:
        return {'available': False, 'reason': 'Insufficient data'}

    close = recent['Close'].values
    high = recent['High'].values
    low = recent['Low'].values
    idx = recent.index
    current = close[-1]
    n = len(recent)

    # ── Find CONFIRMED swing highs and lows ──
    # A swing point needs bars on both sides to confirm it.
    # We exclude the last `margin` bars so we don't pick unconfirmed extremes
    # (e.g., today's low in an active selloff is NOT a confirmed swing low).
    margin = 5  # Require 5 bars after the point to confirm it

    # Find confirmed swing highs (local maxima with margin bars on each side)
    confirmed_swing_highs = []
    confirmed_swing_lows = []
    window = 10  # lookback window for swing detection

    for i in range(window, n - margin):
        # Swing high: highest in surrounding window
        left_start = max(0, i - window)
        right_end = min(n, i + window + 1)
        if high[i] == max(high[left_start:right_end]):
            confirmed_swing_highs.append((i, high[i]))
        # Swing low: lowest in surrounding window
        if low[i] == min(low[left_start:right_end]):
            confirmed_swing_lows.append((i, low[i]))

    # If no confirmed swings found, fall back to broader search but still exclude last margin bars
    if not confirmed_swing_highs:
        safe_end = n - margin
        if safe_end > 0:
            sh_idx = np.argmax(high[:safe_end])
            confirmed_swing_highs = [(sh_idx, high[sh_idx])]

    if not confirmed_swing_lows:
        safe_end = n - margin
        if safe_end > 0:
            sl_idx = np.argmin(low[:safe_end])
            confirmed_swing_lows = [(sl_idx, low[sl_idx])]

    if not confirmed_swing_highs or not confirmed_swing_lows:
        return {'available': False, 'reason': 'Could not identify confirmed swing points'}

    # Select the MAJOR (most significant) swing high and low
    # For swing high: the highest confirmed high
    # For swing low: the lowest confirmed low
    best_sh = max(confirmed_swing_highs, key=lambda x: x[1])
    best_sl = min(confirmed_swing_lows, key=lambda x: x[1])

    swing_high_idx = best_sh[0]
    swing_low_idx = best_sl[0]
    swing_high = best_sh[1]
    swing_low = best_sl[1]

    price_range = swing_high - swing_low
    if price_range < 0.01:
        return {'available': False, 'reason': 'Price range too narrow for Fibonacci analysis'}

    # Determine trend direction: did the high come before or after the low?
    if swing_low_idx < swing_high_idx:
        # Low came first → UPTREND (swing low → swing high)
        primary_trend = 'UPTREND'
    else:
        # High came first → DOWNTREND (swing high → swing low)
        primary_trend = 'DOWNTREND'

    # Determine what's currently happening relative to the swing range
    # Retracement: price pulled back into the range
    # Extension: price broke beyond the range

    fib_retrace_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    fib_extend_ratios = [1.0, 1.236, 1.382, 1.5, 1.618, 2.0, 2.618]
    fib_retrace_labels = ['0%', '23.6%', '38.2%', '50%', '61.8%', '78.6%', '100%']
    fib_extend_labels = ['100%', '123.6%', '138.2%', '150%', '161.8%', '200%', '261.8%']

    result = {
        'available': True,
        'primary_trend': primary_trend,
        'swing_high': swing_high,
        'swing_low': swing_low,
        'swing_high_date': str(idx[swing_high_idx].date()),
        'swing_low_date': str(idx[swing_low_idx].date()),
        'price_range': price_range,
        'current_price': current,
        'retracement': {},
        'extension': {},
        'active_tool': None,
        'active_levels': [],
        'current_zone': None,
        'key_support': None,
        'key_resistance': None,
    }

    # ── Calculate Retracement Levels ──
    if primary_trend == 'UPTREND':
        # Retracement from high: levels go DOWN from swing_high toward swing_low
        retrace_levels = []
        for ratio, label in zip(fib_retrace_ratios, fib_retrace_labels):
            level = swing_high - (price_range * ratio)
            dist_pct = ((current - level) / level) * 100
            retrace_levels.append({
                'label': label, 'ratio': ratio, 'price': level, 'dist_pct': dist_pct,
            })
        result['retracement'] = {
            'direction': 'Pullback from high (measuring down)',
            'from': swing_high, 'to': swing_low, 'levels': retrace_levels,
        }

        # Extension beyond the high
        extend_levels = []
        for ratio, label in zip(fib_extend_ratios, fib_extend_labels):
            level = swing_low + (price_range * ratio)
            dist_pct = ((current - level) / level) * 100
            extend_levels.append({
                'label': label, 'ratio': ratio, 'price': level, 'dist_pct': dist_pct,
            })
        result['extension'] = {
            'direction': 'Extension above high',
            'from': swing_low, 'projected_from': swing_high, 'levels': extend_levels,
        }

        # Which tool is currently active?
        if current < swing_high and current > swing_low:
            result['active_tool'] = 'RETRACEMENT'
            result['active_reason'] = f'Price is pulling back within the uptrend range (${swing_low:.2f} - ${swing_high:.2f}). Retracement levels show potential support zones where the pullback may reverse and resume the uptrend.'
            result['active_levels'] = retrace_levels
        elif current >= swing_high:
            result['active_tool'] = 'EXTENSION'
            result['active_reason'] = f'Price broke above the swing high ${swing_high:.2f}. Extension levels show potential resistance/target zones for the continued uptrend.'
            result['active_levels'] = extend_levels
        else:
            result['active_tool'] = 'RETRACEMENT'
            result['active_reason'] = f'Price dropped below the swing low ${swing_low:.2f}. The full retracement (100%) has been exceeded - trend may be reversing.'
            result['active_levels'] = retrace_levels

    else:  # DOWNTREND
        # Retracement from low: levels go UP from swing_low toward swing_high
        retrace_levels = []
        for ratio, label in zip(fib_retrace_ratios, fib_retrace_labels):
            level = swing_low + (price_range * ratio)
            dist_pct = ((current - level) / level) * 100
            retrace_levels.append({
                'label': label, 'ratio': ratio, 'price': level, 'dist_pct': dist_pct,
            })
        result['retracement'] = {
            'direction': 'Bounce from low (measuring up)',
            'from': swing_low, 'to': swing_high, 'levels': retrace_levels,
        }

        # Extension below the low
        extend_levels = []
        for ratio, label in zip(fib_extend_ratios, fib_extend_labels):
            level = swing_high - (price_range * ratio)
            dist_pct = ((current - level) / level) * 100
            extend_levels.append({
                'label': label, 'ratio': ratio, 'price': level, 'dist_pct': dist_pct,
            })
        result['extension'] = {
            'direction': 'Extension below low',
            'from': swing_high, 'projected_from': swing_low, 'levels': extend_levels,
        }

        if current > swing_low and current < swing_high:
            result['active_tool'] = 'RETRACEMENT'
            result['active_reason'] = f'Price is bouncing within the downtrend range (${swing_low:.2f} - ${swing_high:.2f}). Retracement levels show potential resistance zones where the bounce may fail and resume the downtrend.'
            result['active_levels'] = retrace_levels
        elif current <= swing_low:
            result['active_tool'] = 'EXTENSION'
            result['active_reason'] = f'Price broke below the swing low ${swing_low:.2f}. Extension levels show potential support/target zones for the continued downtrend.'
            result['active_levels'] = extend_levels
        else:
            result['active_tool'] = 'RETRACEMENT'
            result['active_reason'] = f'Price moved above the swing high ${swing_high:.2f}. The full retracement (100%) has been exceeded - trend may be reversing.'
            result['active_levels'] = retrace_levels

    # ── Find current zone (between which two Fib levels price sits) ──
    active = result['active_levels']
    sorted_levels = sorted(active, key=lambda x: x['price'])
    for i in range(len(sorted_levels) - 1):
        if sorted_levels[i]['price'] <= current <= sorted_levels[i+1]['price']:
            result['current_zone'] = f'Between {sorted_levels[i]["label"]} (${sorted_levels[i]["price"]:.2f}) and {sorted_levels[i+1]["label"]} (${sorted_levels[i+1]["price"]:.2f})'
            break

    # ── Key support/resistance from Fib ──
    supports = [l for l in active if l['price'] < current]
    resistances = [l for l in active if l['price'] > current]
    if supports:
        nearest_support = max(supports, key=lambda x: x['price'])
        result['key_support'] = nearest_support
    if resistances:
        nearest_resistance = min(resistances, key=lambda x: x['price'])
        result['key_resistance'] = nearest_resistance

    return result


def compute_multi_timeframe_fibonacci(ticker_symbol: str, daily_df=None) -> dict:
    """
    Compute Fibonacci analysis across Daily, Weekly, and Monthly timeframes.
    Uses yfinance to fetch weekly/monthly data independently.

    Returns dict with keys: 'Daily', 'Weekly', 'Monthly' each containing
    the result of analyze_fibonacci() plus '_df' key with the raw DataFrame.
    """
    import yfinance as yf

    results = {}

    # ── Daily (reuse existing df if provided) ──
    if daily_df is not None and len(daily_df) >= 20:
        results['Daily'] = analyze_fibonacci(daily_df, lookback=120)
        results['Daily']['_df'] = daily_df
    else:
        results['Daily'] = {'available': False, 'reason': 'No daily data'}

    # ── Weekly ──
    try:
        stock = yf.Ticker(ticker_symbol)
        weekly_df = stock.history(period='5y', interval='1wk', auto_adjust=True)
        if weekly_df is not None and len(weekly_df) >= 30:
            results['Weekly'] = analyze_fibonacci(weekly_df, lookback=104)
            results['Weekly']['_df'] = weekly_df
        else:
            results['Weekly'] = {'available': False, 'reason': f'Insufficient weekly data ({len(weekly_df) if weekly_df is not None else 0} bars)'}
    except Exception as e:
        results['Weekly'] = {'available': False, 'reason': f'Error: {str(e)[:80]}'}

    # ── Monthly ──
    try:
        stock = yf.Ticker(ticker_symbol)
        monthly_df = stock.history(period='10y', interval='1mo', auto_adjust=True)
        if monthly_df is not None and len(monthly_df) >= 20:
            results['Monthly'] = analyze_fibonacci(monthly_df, lookback=60)
            results['Monthly']['_df'] = monthly_df
        else:
            results['Monthly'] = {'available': False, 'reason': f'Insufficient monthly data ({len(monthly_df) if monthly_df is not None else 0} bars)'}
    except Exception as e:
        results['Monthly'] = {'available': False, 'reason': f'Error: {str(e)[:80]}'}

    return results


def detect_chart_patterns(df: pd.DataFrame, lookback: int = 120) -> list:
    """
    Detect fully formed chart patterns in price data.
    Only reports patterns that are completely formed.

    Patterns detected:
    - Head & Shoulders / Inverse Head & Shoulders
    - Double Top / Double Bottom
    - Triple Top / Triple Bottom
    - Rising Wedge / Falling Wedge
    - Bull Flag / Bear Flag
    - Pennants
    - Ascending / Descending / Symmetrical Triangles
    """
    recent = df.tail(lookback).copy()
    if len(recent) < 30:
        return []

    close = recent['Close'].values
    high = recent['High'].values
    low = recent['Low'].values
    idx = recent.index
    patterns = []

    # ── Helper: Find swing highs and lows ──
    def find_swings(data, order=5):
        """Find swing highs and lows with their indices."""
        highs = []
        lows = []
        for i in range(order, len(data) - order):
            if all(data[i] >= data[i-j] for j in range(1, order+1)) and \
               all(data[i] >= data[i+j] for j in range(1, order+1)):
                highs.append((i, data[i]))
            if all(data[i] <= data[i-j] for j in range(1, order+1)) and \
               all(data[i] <= data[i+j] for j in range(1, order+1)):
                lows.append((i, data[i]))
        return highs, lows

    def pct_diff(a, b):
        return abs(a - b) / ((a + b) / 2) * 100

    avg_range = np.mean(high - low)
    price_range = np.max(high) - np.min(low)

    swing_highs, swing_lows = find_swings(high, order=5)
    swing_highs_c, swing_lows_c = find_swings(close, order=5)

    # Also find swings with smaller order for flag/pennant detection
    small_highs, small_lows = find_swings(high, order=3)

    # ════════════════════════════════════════════════════════════════
    # HEAD & SHOULDERS / INVERSE HEAD & SHOULDERS
    # ════════════════════════════════════════════════════════════════
    # H&S: Three peaks where middle is highest, flanking peaks roughly equal
    if len(swing_highs) >= 3:
        for i in range(len(swing_highs) - 2):
            left_i, left_p = swing_highs[i]
            head_i, head_p = swing_highs[i + 1]
            right_i, right_p = swing_highs[i + 2]

            # Head must be higher than both shoulders
            if head_p > left_p and head_p > right_p:
                # Shoulders roughly equal (within 3%)
                if pct_diff(left_p, right_p) < 3.0:
                    # Head significantly higher than shoulders (at least 1.5%)
                    if ((head_p - left_p) / left_p) * 100 > 1.5:
                        # Find neckline from lows between peaks
                        lows_between = [l for l in swing_lows if left_i < l[0] < right_i]
                        if len(lows_between) >= 1:
                            neckline = np.mean([l[1] for l in lows_between])
                            # Pattern is complete if price has broken below neckline
                            if close[-1] < neckline and right_i < len(close) - 3:
                                patterns.append({
                                    'name': 'Head & Shoulders',
                                    'bias': 'BEARISH',
                                    'type': 'Reversal',
                                    'period': f'{str(idx[left_i].date())} to {str(idx[right_i].date())}',
                                    'detail': f'Left shoulder: ${left_p:.2f}, Head: ${head_p:.2f}, Right shoulder: ${right_p:.2f}, Neckline: ${neckline:.2f}. Price broke below neckline.',
                                    'target': f'${neckline - (head_p - neckline):.2f} (measured move)',
                                })

    # Inverse H&S: Three troughs where middle is lowest
    if len(swing_lows) >= 3:
        for i in range(len(swing_lows) - 2):
            left_i, left_p = swing_lows[i]
            head_i, head_p = swing_lows[i + 1]
            right_i, right_p = swing_lows[i + 2]

            if head_p < left_p and head_p < right_p:
                if pct_diff(left_p, right_p) < 3.0:
                    if ((left_p - head_p) / head_p) * 100 > 1.5:
                        highs_between = [h for h in swing_highs if left_i < h[0] < right_i]
                        if len(highs_between) >= 1:
                            neckline = np.mean([h[1] for h in highs_between])
                            if close[-1] > neckline and right_i < len(close) - 3:
                                patterns.append({
                                    'name': 'Inverse Head & Shoulders',
                                    'bias': 'BULLISH',
                                    'type': 'Reversal',
                                    'period': f'{str(idx[left_i].date())} to {str(idx[right_i].date())}',
                                    'detail': f'Left shoulder: ${left_p:.2f}, Head: ${head_p:.2f}, Right shoulder: ${right_p:.2f}, Neckline: ${neckline:.2f}. Price broke above neckline.',
                                    'target': f'${neckline + (neckline - head_p):.2f} (measured move)',
                                })

    # ════════════════════════════════════════════════════════════════
    # DOUBLE TOP / DOUBLE BOTTOM
    # ════════════════════════════════════════════════════════════════
    if len(swing_highs) >= 2:
        for i in range(len(swing_highs) - 1):
            p1_i, p1_p = swing_highs[i]
            p2_i, p2_p = swing_highs[i + 1]

            # Peaks roughly equal (within 2%)
            if pct_diff(p1_p, p2_p) < 2.0:
                # Minimum spacing between peaks
                if p2_i - p1_i >= 10:
                    # Find trough between peaks
                    troughs = [l for l in swing_lows if p1_i < l[0] < p2_i]
                    if troughs:
                        trough_val = min(t[1] for t in troughs)
                        # Trough must be meaningfully lower (at least 2%)
                        if ((p1_p - trough_val) / p1_p) * 100 > 2.0:
                            # Complete: price broke below the trough
                            if close[-1] < trough_val and p2_i < len(close) - 3:
                                patterns.append({
                                    'name': 'Double Top',
                                    'bias': 'BEARISH',
                                    'type': 'Reversal',
                                    'period': f'{str(idx[p1_i].date())} to {str(idx[p2_i].date())}',
                                    'detail': f'Peak 1: ${p1_p:.2f}, Peak 2: ${p2_p:.2f}, Support: ${trough_val:.2f}. Price broke below support.',
                                    'target': f'${trough_val - (p1_p - trough_val):.2f} (measured move)',
                                })

    if len(swing_lows) >= 2:
        for i in range(len(swing_lows) - 1):
            t1_i, t1_p = swing_lows[i]
            t2_i, t2_p = swing_lows[i + 1]

            if pct_diff(t1_p, t2_p) < 2.0:
                if t2_i - t1_i >= 10:
                    peaks = [h for h in swing_highs if t1_i < h[0] < t2_i]
                    if peaks:
                        peak_val = max(p[1] for p in peaks)
                        if ((peak_val - t1_p) / t1_p) * 100 > 2.0:
                            if close[-1] > peak_val and t2_i < len(close) - 3:
                                patterns.append({
                                    'name': 'Double Bottom',
                                    'bias': 'BULLISH',
                                    'type': 'Reversal',
                                    'period': f'{str(idx[t1_i].date())} to {str(idx[t2_i].date())}',
                                    'detail': f'Trough 1: ${t1_p:.2f}, Trough 2: ${t2_p:.2f}, Resistance: ${peak_val:.2f}. Price broke above resistance.',
                                    'target': f'${peak_val + (peak_val - t1_p):.2f} (measured move)',
                                })

    # ════════════════════════════════════════════════════════════════
    # TRIPLE TOP / TRIPLE BOTTOM
    # ════════════════════════════════════════════════════════════════
    if len(swing_highs) >= 3:
        for i in range(len(swing_highs) - 2):
            p1_i, p1_p = swing_highs[i]
            p2_i, p2_p = swing_highs[i + 1]
            p3_i, p3_p = swing_highs[i + 2]

            if pct_diff(p1_p, p2_p) < 2.0 and pct_diff(p2_p, p3_p) < 2.0:
                if p2_i - p1_i >= 8 and p3_i - p2_i >= 8:
                    troughs = [l for l in swing_lows if p1_i < l[0] < p3_i]
                    if len(troughs) >= 2:
                        support = min(t[1] for t in troughs)
                        if close[-1] < support and p3_i < len(close) - 3:
                            patterns.append({
                                'name': 'Triple Top',
                                'bias': 'BEARISH',
                                'type': 'Reversal',
                                'period': f'{str(idx[p1_i].date())} to {str(idx[p3_i].date())}',
                                'detail': f'Three peaks at ~${np.mean([p1_p,p2_p,p3_p]):.2f}, Support: ${support:.2f}. Price broke below support.',
                                'target': f'${support - (p1_p - support):.2f} (measured move)',
                            })

    if len(swing_lows) >= 3:
        for i in range(len(swing_lows) - 2):
            t1_i, t1_p = swing_lows[i]
            t2_i, t2_p = swing_lows[i + 1]
            t3_i, t3_p = swing_lows[i + 2]

            if pct_diff(t1_p, t2_p) < 2.0 and pct_diff(t2_p, t3_p) < 2.0:
                if t2_i - t1_i >= 8 and t3_i - t2_i >= 8:
                    peaks = [h for h in swing_highs if t1_i < h[0] < t3_i]
                    if len(peaks) >= 2:
                        resistance = max(p[1] for p in peaks)
                        if close[-1] > resistance and t3_i < len(close) - 3:
                            patterns.append({
                                'name': 'Triple Bottom',
                                'bias': 'BULLISH',
                                'type': 'Reversal',
                                'period': f'{str(idx[t1_i].date())} to {str(idx[t3_i].date())}',
                                'detail': f'Three troughs at ~${np.mean([t1_p,t2_p,t3_p]):.2f}, Resistance: ${resistance:.2f}. Price broke above resistance.',
                                'target': f'${resistance + (resistance - t1_p):.2f} (measured move)',
                            })

    # ════════════════════════════════════════════════════════════════
    # WEDGES (Rising / Falling)
    # ════════════════════════════════════════════════════════════════
    # Use swing points to fit trendlines and check convergence
    if len(swing_highs) >= 3 and len(swing_lows) >= 3:
        # Get last N swing points
        recent_highs = swing_highs[-4:] if len(swing_highs) >= 4 else swing_highs[-3:]
        recent_lows = swing_lows[-4:] if len(swing_lows) >= 4 else swing_lows[-3:]

        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            h_x = np.array([h[0] for h in recent_highs])
            h_y = np.array([h[1] for h in recent_highs])
            l_x = np.array([l[0] for l in recent_lows])
            l_y = np.array([l[1] for l in recent_lows])

            h_slope = np.polyfit(h_x, h_y, 1)[0]
            l_slope = np.polyfit(l_x, l_y, 1)[0]

            # Normalize slopes
            h_slope_pct = (h_slope / np.mean(h_y)) * 100
            l_slope_pct = (l_slope / np.mean(l_y)) * 100

            wedge_start = min(h_x[0], l_x[0])
            wedge_end = max(h_x[-1], l_x[-1])

            # Lines converging?
            converging = (h_slope_pct < l_slope_pct) if h_slope > 0 else (h_slope_pct > l_slope_pct)

            # Rising Wedge: both slopes up, converging (bearish)
            if h_slope_pct > 0.1 and l_slope_pct > 0.1 and converging:
                if wedge_end < len(close) - 2:
                    # Complete if price broke below lower trendline
                    lower_trendline_end = l_slope * (len(close)-1) + np.polyfit(l_x, l_y, 1)[1]
                    if close[-1] < lower_trendline_end:
                        patterns.append({
                            'name': 'Rising Wedge',
                            'bias': 'BEARISH',
                            'type': 'Reversal',
                            'period': f'{str(idx[int(wedge_start)].date())} to {str(idx[int(wedge_end)].date())}',
                            'detail': f'Both trendlines rising but converging. Upper slope: {h_slope_pct:.2f}%/bar, Lower slope: {l_slope_pct:.2f}%/bar. Price broke below lower trendline.',
                            'target': f'${close[-1] - (h_y[-1] - l_y[0]):.2f} (wedge height projection)',
                        })

            # Falling Wedge: both slopes down, converging (bullish)
            if h_slope_pct < -0.1 and l_slope_pct < -0.1 and converging:
                if wedge_end < len(close) - 2:
                    upper_trendline_end = h_slope * (len(close)-1) + np.polyfit(h_x, h_y, 1)[1]
                    if close[-1] > upper_trendline_end:
                        patterns.append({
                            'name': 'Falling Wedge',
                            'bias': 'BULLISH',
                            'type': 'Reversal',
                            'period': f'{str(idx[int(wedge_start)].date())} to {str(idx[int(wedge_end)].date())}',
                            'detail': f'Both trendlines falling but converging. Upper slope: {h_slope_pct:.2f}%/bar, Lower slope: {l_slope_pct:.2f}%/bar. Price broke above upper trendline.',
                            'target': f'${close[-1] + (h_y[0] - l_y[-1]):.2f} (wedge height projection)',
                        })

    # ════════════════════════════════════════════════════════════════
    # TRIANGLES (Ascending / Descending / Symmetrical)
    # ════════════════════════════════════════════════════════════════
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs[-2:]
        recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows[-2:]

        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            h_x = np.array([h[0] for h in recent_highs])
            h_y = np.array([h[1] for h in recent_highs])
            l_x = np.array([l[0] for l in recent_lows])
            l_y = np.array([l[1] for l in recent_lows])

            h_slope = np.polyfit(h_x, h_y, 1)[0]
            l_slope = np.polyfit(l_x, l_y, 1)[0]
            h_slope_pct = (h_slope / np.mean(h_y)) * 100
            l_slope_pct = (l_slope / np.mean(l_y)) * 100

            tri_start = min(h_x[0], l_x[0])
            tri_end = max(h_x[-1], l_x[-1])
            tri_height = h_y[0] - l_y[0] if h_y[0] > l_y[0] else l_y[0] - h_y[0]

            # Ascending Triangle: flat top, rising lows
            if abs(h_slope_pct) < 0.15 and l_slope_pct > 0.15:
                resistance = np.mean(h_y)
                if close[-1] > resistance and tri_end < len(close) - 2:
                    patterns.append({
                        'name': 'Ascending Triangle',
                        'bias': 'BULLISH',
                        'type': 'Continuation',
                        'period': f'{str(idx[int(tri_start)].date())} to {str(idx[int(tri_end)].date())}',
                        'detail': f'Flat resistance at ~${resistance:.2f} with rising lows. Price broke above resistance.',
                        'target': f'${resistance + tri_height:.2f} (triangle height projection)',
                    })

            # Descending Triangle: flat bottom, falling highs
            elif abs(l_slope_pct) < 0.15 and h_slope_pct < -0.15:
                support = np.mean(l_y)
                if close[-1] < support and tri_end < len(close) - 2:
                    patterns.append({
                        'name': 'Descending Triangle',
                        'bias': 'BEARISH',
                        'type': 'Continuation',
                        'period': f'{str(idx[int(tri_start)].date())} to {str(idx[int(tri_end)].date())}',
                        'detail': f'Flat support at ~${support:.2f} with falling highs. Price broke below support.',
                        'target': f'${support - tri_height:.2f} (triangle height projection)',
                    })

            # Symmetrical Triangle: converging trendlines (both narrowing)
            elif h_slope_pct < -0.1 and l_slope_pct > 0.1:
                if tri_end < len(close) - 2:
                    upper_end = h_slope * (len(close)-1) + np.polyfit(h_x, h_y, 1)[1]
                    lower_end = l_slope * (len(close)-1) + np.polyfit(l_x, l_y, 1)[1]
                    if close[-1] > upper_end:
                        patterns.append({
                            'name': 'Symmetrical Triangle (Bullish Breakout)',
                            'bias': 'BULLISH',
                            'type': 'Continuation',
                            'period': f'{str(idx[int(tri_start)].date())} to {str(idx[int(tri_end)].date())}',
                            'detail': f'Converging trendlines with bullish breakout above upper trendline.',
                            'target': f'${close[-1] + tri_height:.2f} (triangle height projection)',
                        })
                    elif close[-1] < lower_end:
                        patterns.append({
                            'name': 'Symmetrical Triangle (Bearish Breakout)',
                            'bias': 'BEARISH',
                            'type': 'Continuation',
                            'period': f'{str(idx[int(tri_start)].date())} to {str(idx[int(tri_end)].date())}',
                            'detail': f'Converging trendlines with bearish breakout below lower trendline.',
                            'target': f'${close[-1] - tri_height:.2f} (triangle height projection)',
                        })

    # ════════════════════════════════════════════════════════════════
    # FLAGS (Bull Flag / Bear Flag) and PENNANTS
    # ════════════════════════════════════════════════════════════════
    # Flags: strong move (pole) followed by gentle counter-trend channel
    # Pennants: strong move followed by small symmetrical triangle

    # Look for a strong pole in the last 40 bars
    for pole_len in range(8, 25):
        if pole_len >= len(close) - 10:
            continue

        pole_start = -(pole_len + 15)
        pole_end = -15
        if abs(pole_start) > len(close):
            continue

        pole_move = close[pole_end] - close[pole_start]
        pole_pct = (pole_move / close[pole_start]) * 100

        # Need a strong pole (>5% move)
        if abs(pole_pct) < 5:
            continue

        # Consolidation phase (last 15 bars after pole)
        consol = close[-15:]
        consol_high = high[-15:]
        consol_low = low[-15:]
        consol_range = np.max(consol_high) - np.min(consol_low)
        pole_range = abs(pole_move)

        # Consolidation should be tight relative to pole (< 50% of pole)
        if consol_range > pole_range * 0.5:
            continue

        # Fit trendlines to consolidation
        x_consol = np.arange(len(consol))
        if len(x_consol) < 5:
            continue

        # Use small swings for flag/pennant
        consol_sh, consol_sl = find_swings(consol, order=2)
        if len(consol_sh) < 2 or len(consol_sl) < 2:
            # Fallback: use linear fit on highs and lows
            h_fit = np.polyfit(x_consol, consol_high, 1)
            l_fit = np.polyfit(x_consol, consol_low, 1)
            h_slope_c = h_fit[0]
            l_slope_c = l_fit[0]
        else:
            ch_x = np.array([s[0] for s in consol_sh])
            ch_y = np.array([s[1] for s in consol_sh])
            cl_x = np.array([s[0] for s in consol_sl])
            cl_y = np.array([s[1] for s in consol_sl])
            h_slope_c = np.polyfit(ch_x, ch_y, 1)[0]
            l_slope_c = np.polyfit(cl_x, cl_y, 1)[0]

        flag_start_idx = len(close) + pole_end
        flag_end_idx = len(close) - 1

        if pole_pct > 5:  # Bullish pole
            # Bull Flag: parallel downward channel
            if h_slope_c < 0 and l_slope_c < 0 and abs(h_slope_c - l_slope_c) < avg_range * 0.3:
                if close[-1] > np.max(consol_high) * 0.995:
                    patterns.append({
                        'name': 'Bull Flag',
                        'bias': 'BULLISH',
                        'type': 'Continuation',
                        'period': f'{str(idx[flag_start_idx].date())} to {str(idx[flag_end_idx].date())}',
                        'detail': f'Strong upward pole ({pole_pct:.1f}%) followed by downward-sloping consolidation. Price breaking above flag.',
                        'target': f'${close[-1] + pole_range:.2f} (pole height projection)',
                    })
                    break

            # Bullish Pennant: converging consolidation after up pole
            if h_slope_c < 0 and l_slope_c > 0:
                if close[-1] > np.max(consol_high) * 0.995:
                    patterns.append({
                        'name': 'Bullish Pennant',
                        'bias': 'BULLISH',
                        'type': 'Continuation',
                        'period': f'{str(idx[flag_start_idx].date())} to {str(idx[flag_end_idx].date())}',
                        'detail': f'Strong upward pole ({pole_pct:.1f}%) followed by symmetrical narrowing consolidation. Price breaking above pennant.',
                        'target': f'${close[-1] + pole_range:.2f} (pole height projection)',
                    })
                    break

        elif pole_pct < -5:  # Bearish pole
            # Bear Flag: parallel upward channel
            if h_slope_c > 0 and l_slope_c > 0 and abs(h_slope_c - l_slope_c) < avg_range * 0.3:
                if close[-1] < np.min(consol_low) * 1.005:
                    patterns.append({
                        'name': 'Bear Flag',
                        'bias': 'BEARISH',
                        'type': 'Continuation',
                        'period': f'{str(idx[flag_start_idx].date())} to {str(idx[flag_end_idx].date())}',
                        'detail': f'Strong downward pole ({pole_pct:.1f}%) followed by upward-sloping consolidation. Price breaking below flag.',
                        'target': f'${close[-1] - pole_range:.2f} (pole height projection)',
                    })
                    break

            # Bearish Pennant
            if h_slope_c < 0 and l_slope_c > 0:
                if close[-1] < np.min(consol_low) * 1.005:
                    patterns.append({
                        'name': 'Bearish Pennant',
                        'bias': 'BEARISH',
                        'type': 'Continuation',
                        'period': f'{str(idx[flag_start_idx].date())} to {str(idx[flag_end_idx].date())}',
                        'detail': f'Strong downward pole ({pole_pct:.1f}%) followed by symmetrical narrowing consolidation. Price breaking below pennant.',
                        'target': f'${close[-1] - pole_range:.2f} (pole height projection)',
                    })
                    break

    return patterns


def detect_eccentric_patterns(df, lookback=120, swing_window=5, low_tol=10, up_tol=10, ab_ratio=100, bc_ratio=30, be_ratio=40):
    """
    Chart pattern detection based on theEccentricTrader's Pine Script methodology.
    Credit: theEccentricTrader / Ozan Kaplanbasoglu (Pine Script v5)

    Detects patterns using swing high/low pivot points and retracement ratios:
    - Ascending/Descending/Symmetric Broadening
    - Double Bottom/Top (with tolerance)
    - Triple Bottom/Top (with tolerance)
    - Bull/Bear Elliott Wave (5-wave + ABC correction)
    - Bull/Bear Flag (with AB/BC/BE ratios)
    - Bull/Bear Alternate Flag
    - Bull/Bear Pennant
    - Bull/Bear Ascending Head & Shoulders
    - Bull/Bear Descending Head & Shoulders
    - Bull/Bear Head & Shoulders (classic)
    - Ascending/Descending/Symmetric Wedge
    """
    import numpy as np
    import pandas as pd

    recent = df.tail(lookback).copy()
    if len(recent) < 30:
        return []

    high = recent['High'].values
    low = recent['Low'].values
    close = recent['Close'].values
    idx = recent.index
    n = len(recent)

    # ── Detect Swing Highs and Swing Lows ──
    swing_highs = []  # (index_pos, price, bar_index_in_recent)
    swing_lows = []

    for i in range(swing_window, n - swing_window):
        # Swing High: high[i] is highest in window
        if high[i] == max(high[i - swing_window:i + swing_window + 1]):
            swing_highs.append({'pos': i, 'price': high[i], 'date': str(idx[i].date()) if hasattr(idx[i], 'date') else str(idx[i])})
        # Swing Low: low[i] is lowest in window
        if low[i] == min(low[i - swing_window:i + swing_window + 1]):
            swing_lows.append({'pos': i, 'price': low[i], 'date': str(idx[i].date()) if hasattr(idx[i], 'date') else str(idx[i])})

    patterns = []
    current = close[-1]

    # Helper: retracement ratio between two swings
    def retrace_ratio(move_from, move_to, reference_range):
        if reference_range == 0:
            return 0
        return abs(move_to - move_from) / abs(reference_range) * 100

    # Helper: check tolerance (are two prices within tol% of each other?)
    def within_tolerance(p1, p2, low_t=low_tol, up_t=up_tol):
        if p1 == 0:
            return False
        pct_diff = abs(p1 - p2) / p1 * 100
        return pct_diff <= max(low_t, up_t)

    # ══════════════════════════════════════
    # BROADENING PATTERNS (need 2 SH + 2 SL)
    # ══════════════════════════════════════
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        sh0, sh1 = swing_highs[-1], swing_highs[-2]
        sl0, sl1 = swing_lows[-1], swing_lows[-2]

        highs_expanding = sh0['price'] > sh1['price']
        highs_contracting = sh0['price'] < sh1['price']
        lows_expanding = sl0['price'] < sl1['price']
        lows_contracting = sl0['price'] > sl1['price']

        # Ascending Broadening: higher highs AND lower lows, but highs rise faster
        if highs_expanding and lows_expanding:
            high_range = sh0['price'] - sh1['price']
            low_range = sl1['price'] - sl0['price']
            if high_range > low_range:
                patterns.append({
                    'name': 'Ascending Broadening', 'bias': 'Neutral', 'type': 'Continuation',
                    'description': f'Expanding range with higher highs rising faster than lower lows. Range widening from ${sh1["price"]:.2f}-${sl1["price"]:.2f} to ${sh0["price"]:.2f}-${sl0["price"]:.2f}.',
                    'neckline': None, 'target': 'Range expansion expected',
                })

        # Descending Broadening: higher highs AND lower lows, but lows drop faster
        if highs_expanding and lows_expanding:
            high_range = sh0['price'] - sh1['price']
            low_range = sl1['price'] - sl0['price']
            if low_range > high_range:
                patterns.append({
                    'name': 'Descending Broadening', 'bias': 'Neutral', 'type': 'Continuation',
                    'description': f'Expanding range with lower lows dropping faster. Range widening from ${sh1["price"]:.2f}-${sl1["price"]:.2f} to ${sh0["price"]:.2f}-${sl0["price"]:.2f}.',
                    'neckline': None, 'target': 'Range expansion expected',
                })

        # Symmetric Broadening: higher highs AND lower lows, roughly equal expansion
        if highs_expanding and lows_expanding:
            high_range = sh0['price'] - sh1['price']
            low_range = sl1['price'] - sl0['price']
            if high_range > 0 and low_range > 0:
                ratio = high_range / low_range if low_range > 0 else 999
                if 0.5 <= ratio <= 2.0:
                    patterns.append({
                        'name': 'Broadening (Symmetric)', 'bias': 'Neutral', 'type': 'Reversal',
                        'description': f'Symmetric expanding range. Highs expanding ${high_range:.2f}, lows expanding ${low_range:.2f}.',
                        'neckline': None, 'target': 'Breakout direction determines bias',
                    })

    # ══════════════════════════════════════
    # DOUBLE BOTTOM / DOUBLE TOP
    # ══════════════════════════════════════
    if len(swing_lows) >= 2 and len(swing_highs) >= 1:
        sl0, sl1 = swing_lows[-1], swing_lows[-2]
        # Find a swing high between the two lows
        middle_highs = [sh for sh in swing_highs if sl1['pos'] < sh['pos'] < sl0['pos']]
        if middle_highs and within_tolerance(sl0['price'], sl1['price']):
            neckline = max(mh['price'] for mh in middle_highs)
            height = neckline - min(sl0['price'], sl1['price'])
            target = neckline + height
            if current < neckline * 1.03:  # not already far above neckline
                patterns.append({
                    'name': 'Double Bottom', 'bias': 'Bullish', 'type': 'Reversal',
                    'description': f'Two lows at ${sl1["price"]:.2f} ({sl1["date"]}) and ${sl0["price"]:.2f} ({sl0["date"]}) within {low_tol}% tolerance. Neckline at ${neckline:.2f}.',
                    'neckline': f'${neckline:.2f}', 'target': f'${target:.2f} (height projection)',
                })

    if len(swing_highs) >= 2 and len(swing_lows) >= 1:
        sh0, sh1 = swing_highs[-1], swing_highs[-2]
        middle_lows = [sl for sl in swing_lows if sh1['pos'] < sl['pos'] < sh0['pos']]
        if middle_lows and within_tolerance(sh0['price'], sh1['price']):
            neckline = min(ml['price'] for ml in middle_lows)
            height = max(sh0['price'], sh1['price']) - neckline
            target = neckline - height
            if current > neckline * 0.97:
                patterns.append({
                    'name': 'Double Top', 'bias': 'Bearish', 'type': 'Reversal',
                    'description': f'Two highs at ${sh1["price"]:.2f} ({sh1["date"]}) and ${sh0["price"]:.2f} ({sh0["date"]}) within {up_tol}% tolerance. Neckline at ${neckline:.2f}.',
                    'neckline': f'${neckline:.2f}', 'target': f'${target:.2f} (height projection)',
                })

    # ══════════════════════════════════════
    # TRIPLE BOTTOM / TRIPLE TOP
    # ══════════════════════════════════════
    if len(swing_lows) >= 3 and len(swing_highs) >= 2:
        sl0, sl1, sl2 = swing_lows[-1], swing_lows[-2], swing_lows[-3]
        if within_tolerance(sl0['price'], sl1['price']) and within_tolerance(sl1['price'], sl2['price']):
            middle_highs = [sh for sh in swing_highs if sl2['pos'] < sh['pos'] < sl0['pos']]
            if middle_highs:
                neckline = max(mh['price'] for mh in middle_highs)
                height = neckline - min(sl0['price'], sl1['price'], sl2['price'])
                target = neckline + height
                patterns.append({
                    'name': 'Triple Bottom', 'bias': 'Bullish', 'type': 'Reversal',
                    'description': f'Three lows at ${sl2["price"]:.2f}, ${sl1["price"]:.2f}, ${sl0["price"]:.2f} within tolerance. Neckline ${neckline:.2f}.',
                    'neckline': f'${neckline:.2f}', 'target': f'${target:.2f}',
                })

    if len(swing_highs) >= 3 and len(swing_lows) >= 2:
        sh0, sh1, sh2 = swing_highs[-1], swing_highs[-2], swing_highs[-3]
        if within_tolerance(sh0['price'], sh1['price']) and within_tolerance(sh1['price'], sh2['price']):
            middle_lows = [sl for sl in swing_lows if sh2['pos'] < sl['pos'] < sh0['pos']]
            if middle_lows:
                neckline = min(ml['price'] for ml in middle_lows)
                height = max(sh0['price'], sh1['price'], sh2['price']) - neckline
                target = neckline - height
                patterns.append({
                    'name': 'Triple Top', 'bias': 'Bearish', 'type': 'Reversal',
                    'description': f'Three highs at ${sh2["price"]:.2f}, ${sh1["price"]:.2f}, ${sh0["price"]:.2f} within tolerance. Neckline ${neckline:.2f}.',
                    'neckline': f'${neckline:.2f}', 'target': f'${target:.2f}',
                })

    # ══════════════════════════════════════
    # HEAD AND SHOULDERS VARIANTS
    # ══════════════════════════════════════
    # Bear H&S: SH2 < SH1 > SH0 (head higher than shoulders)
    if len(swing_highs) >= 3 and len(swing_lows) >= 2:
        sh0, sh1, sh2 = swing_highs[-1], swing_highs[-2], swing_highs[-3]
        sl0, sl1 = swing_lows[-1], swing_lows[-2]

        head = sh1['price']
        l_shoulder = sh2['price']
        r_shoulder = sh0['price']

        # Classic Bear H&S: head > both shoulders, shoulders roughly equal
        if head > l_shoulder and head > r_shoulder and within_tolerance(l_shoulder, r_shoulder, 15, 15):
            neckline = min(sl0['price'], sl1['price'])
            height = head - neckline
            target = neckline - height
            patterns.append({
                'name': 'Bear Head & Shoulders', 'bias': 'Bearish', 'type': 'Reversal',
                'description': f'Head ${head:.2f} ({sh1["date"]}), L-Shoulder ${l_shoulder:.2f}, R-Shoulder ${r_shoulder:.2f}. Neckline ${neckline:.2f}.',
                'neckline': f'${neckline:.2f}', 'target': f'${target:.2f}',
            })

        # Bear Ascending H&S: each high is higher (SH2 < SH1, SH0 could be between)
        if sh2['price'] < sh1['price'] and sh0['price'] < sh1['price'] and sh0['price'] > sh2['price']:
            neckline = min(sl0['price'], sl1['price'])
            patterns.append({
                'name': 'Bear Asc. Head & Shoulders', 'bias': 'Bearish', 'type': 'Reversal',
                'description': f'Ascending H&S: highs at ${sh2["price"]:.2f} → ${sh1["price"]:.2f} → ${sh0["price"]:.2f}. Rising neckline at ${neckline:.2f}.',
                'neckline': f'${neckline:.2f}', 'target': f'${neckline - (sh1["price"] - neckline):.2f}',
            })

        # Bear Descending H&S: highs descending but middle still highest
        if sh2['price'] > sh0['price'] and sh1['price'] > sh2['price']:
            neckline = min(sl0['price'], sl1['price'])
            patterns.append({
                'name': 'Bear Desc. Head & Shoulders', 'bias': 'Bearish', 'type': 'Reversal',
                'description': f'Descending H&S: highs at ${sh2["price"]:.2f} → ${sh1["price"]:.2f} → ${sh0["price"]:.2f}. Neckline ${neckline:.2f}.',
                'neckline': f'${neckline:.2f}', 'target': f'${neckline - (sh1["price"] - neckline):.2f}',
            })

    # Bull H&S (Inverse): SL2 > SL1 < SL0 (head lower than shoulders)
    if len(swing_lows) >= 3 and len(swing_highs) >= 2:
        sl0, sl1, sl2 = swing_lows[-1], swing_lows[-2], swing_lows[-3]
        sh0, sh1 = swing_highs[-1], swing_highs[-2]

        head = sl1['price']
        l_shoulder = sl2['price']
        r_shoulder = sl0['price']

        # Classic Bull H&S (Inverse)
        if head < l_shoulder and head < r_shoulder and within_tolerance(l_shoulder, r_shoulder, 15, 15):
            neckline = max(sh0['price'], sh1['price'])
            height = neckline - head
            target = neckline + height
            patterns.append({
                'name': 'Bull Head & Shoulders', 'bias': 'Bullish', 'type': 'Reversal',
                'description': f'Inverse H&S: Head ${head:.2f} ({sl1["date"]}), L-Shoulder ${l_shoulder:.2f}, R-Shoulder ${r_shoulder:.2f}. Neckline ${neckline:.2f}.',
                'neckline': f'${neckline:.2f}', 'target': f'${target:.2f}',
            })

        # Bull Ascending H&S
        if sl2['price'] > sl1['price'] and sl0['price'] > sl1['price'] and sl0['price'] < sl2['price']:
            neckline = max(sh0['price'], sh1['price'])
            patterns.append({
                'name': 'Bull Asc. Head & Shoulders', 'bias': 'Bullish', 'type': 'Reversal',
                'description': f'Asc. Inverse H&S: lows at ${sl2["price"]:.2f} → ${sl1["price"]:.2f} → ${sl0["price"]:.2f}. Neckline ${neckline:.2f}.',
                'neckline': f'${neckline:.2f}', 'target': f'${neckline + (neckline - sl1["price"]):.2f}',
            })

        # Bull Descending H&S
        if sl2['price'] < sl0['price'] and sl1['price'] < sl2['price']:
            neckline = max(sh0['price'], sh1['price'])
            patterns.append({
                'name': 'Bull Desc. Head & Shoulders', 'bias': 'Bullish', 'type': 'Reversal',
                'description': f'Desc. Inverse H&S: lows at ${sl2["price"]:.2f} → ${sl1["price"]:.2f} → ${sl0["price"]:.2f}. Neckline ${neckline:.2f}.',
                'neckline': f'${neckline:.2f}', 'target': f'${neckline + (neckline - sl1["price"]):.2f}',
            })

    # ══════════════════════════════════════
    # ELLIOTT WAVE (5+2 swings needed)
    # ══════════════════════════════════════
    if len(swing_highs) >= 4 and len(swing_lows) >= 4:
        # Bull Elliott: 5 up waves (alternating SL, SH) where wave3 > wave1 > wave5, wave4 > wave1 low
        sls = swing_lows[-4:]
        shs = swing_highs[-4:]

        # Bull: SL3 < SL2 (wave1 up), SH2 < SH1 (wave3 higher), SL1 > SL3 (wave4 above wave1)
        if (sls[-4]['price'] > sls[-3]['price'] < sls[-2]['price'] and  # wave 2 pullback
            shs[-3]['price'] < shs[-2]['price'] and  # wave 3 higher than wave 1
            sls[-2]['price'] > sls[-4]['price'] and   # wave 4 above wave 1 start
            shs[-1]['price'] < shs[-2]['price']):     # wave 5 or A lower (correction starting)
            patterns.append({
                'name': 'Bull Elliott Wave', 'bias': 'Bearish', 'type': 'Reversal',
                'description': f'5-wave bullish impulse completed. Wave 3 peak at ${shs[-2]["price"]:.2f}. ABC correction may be underway. Current at wave A/B.',
                'neckline': None, 'target': 'ABC correction target: 38.2%-61.8% of entire wave',
            })

        # Bear: SH3 > SH2 > SH1 (descending highs), SL descending
        if (shs[-4]['price'] < shs[-3]['price'] > shs[-2]['price'] and
            sls[-3]['price'] > sls[-2]['price'] and
            shs[-2]['price'] < shs[-4]['price'] and
            sls[-1]['price'] > sls[-2]['price']):
            patterns.append({
                'name': 'Bear Elliott Wave', 'bias': 'Bullish', 'type': 'Reversal',
                'description': f'5-wave bearish impulse completed. Wave 3 trough at ${sls[-2]["price"]:.2f}. ABC correction upward may be underway.',
                'neckline': None, 'target': 'ABC correction target: 38.2%-61.8% of entire wave',
            })

    # ══════════════════════════════════════
    # FLAG / ALT FLAG / PENNANT
    # ══════════════════════════════════════
    if len(swing_highs) >= 3 and len(swing_lows) >= 2:
        sh0, sh1, sh2 = swing_highs[-1], swing_highs[-2], swing_highs[-3]
        sl0, sl1 = swing_lows[-1], swing_lows[-2]

        # Bull Flag: strong move up (pole), then consolidation down
        pole_range = sh1['price'] - sl1['price'] if len(swing_lows) >= 2 else 0
        if pole_range > 0:
            ab_pct = (sh1['price'] - sl1['price']) / sl1['price'] * 100 if sl1['price'] > 0 else 0
            # Consolidation: recent swings are contained, making lower highs and lower lows
            if (sh0['price'] < sh1['price'] and sl0['price'] > sl1['price'] and
                sh0['price'] > sl0['price']):
                bc_pct = (sh1['price'] - sl0['price']) / pole_range * 100
                if bc_pct <= bc_ratio and ab_pct >= ab_ratio / 10:  # relaxed for stocks
                    # Check if channel is parallel (flag) or converging (pennant)
                    high_slope = sh0['price'] - sh1['price']
                    low_slope = sl0['price'] - sl1['price']

                    if high_slope < 0 and low_slope > 0:
                        # Converging = pennant
                        patterns.append({
                            'name': 'Bull Pennant', 'bias': 'Bullish', 'type': 'Continuation',
                            'description': f'Strong pole up ${sl1["price"]:.2f}→${sh1["price"]:.2f}, then converging consolidation. Breakout target: ${sh1["price"] + pole_range:.2f}.',
                            'neckline': f'${sh0["price"]:.2f}', 'target': f'${sh1["price"] + pole_range:.2f}',
                        })
                    else:
                        # Parallel or down-sloping = flag
                        patterns.append({
                            'name': 'Bull Flag', 'bias': 'Bullish', 'type': 'Continuation',
                            'description': f'Pole: ${sl1["price"]:.2f}→${sh1["price"]:.2f} (+{ab_pct:.1f}%). Flag consolidation with BC retracement {bc_pct:.1f}%.',
                            'neckline': f'${sh0["price"]:.2f}', 'target': f'${sl0["price"] + pole_range:.2f}',
                        })

    if len(swing_lows) >= 3 and len(swing_highs) >= 2:
        sl0, sl1, sl2 = swing_lows[-1], swing_lows[-2], swing_lows[-3]
        sh0, sh1 = swing_highs[-1], swing_highs[-2]

        # Bear Flag: strong move down (pole), then consolidation up
        pole_range = sh1['price'] - sl1['price'] if sh1['price'] > sl1['price'] else 0
        if pole_range > 0:
            ab_pct = pole_range / sh1['price'] * 100
            if (sl0['price'] > sl1['price'] and sh0['price'] < sh1['price'] and
                sh0['price'] > sl0['price']):
                bc_pct = (sh0['price'] - sl1['price']) / pole_range * 100
                if bc_pct <= bc_ratio * 2:
                    high_slope = sh0['price'] - sh1['price']
                    low_slope = sl0['price'] - sl1['price']

                    if high_slope < 0 and low_slope > 0:
                        patterns.append({
                            'name': 'Bear Pennant', 'bias': 'Bearish', 'type': 'Continuation',
                            'description': f'Pole down ${sh1["price"]:.2f}→${sl1["price"]:.2f}, converging consolidation. Breakdown target: ${sl1["price"] - pole_range:.2f}.',
                            'neckline': f'${sl0["price"]:.2f}', 'target': f'${sl1["price"] - pole_range:.2f}',
                        })
                    else:
                        patterns.append({
                            'name': 'Bear Flag', 'bias': 'Bearish', 'type': 'Continuation',
                            'description': f'Pole: ${sh1["price"]:.2f}→${sl1["price"]:.2f} (-{ab_pct:.1f}%). Flag consolidation upward.',
                            'neckline': f'${sl0["price"]:.2f}', 'target': f'${sh0["price"] - pole_range:.2f}',
                        })

    # Alt Flag (2-swing version: strong move + single retracement)
    if len(swing_highs) >= 2 and len(swing_lows) >= 1:
        sh0, sh1 = swing_highs[-1], swing_highs[-2]
        sl0 = swing_lows[-1]
        # Bull Alt Flag: big drop to SL0, bounce to SH0 (small retrace of prior drop)
        if sl0['pos'] > sh1['pos'] and sh0['pos'] > sl0['pos']:
            drop = sh1['price'] - sl0['price']
            bounce = sh0['price'] - sl0['price']
            if drop > 0 and bounce > 0:
                retrace = bounce / drop * 100
                if retrace <= bc_ratio:
                    patterns.append({
                        'name': 'Bear Alt. Flag', 'bias': 'Bearish', 'type': 'Continuation',
                        'description': f'Sharp drop ${sh1["price"]:.2f}→${sl0["price"]:.2f}, weak bounce to ${sh0["price"]:.2f} ({retrace:.1f}% retrace). Continuation lower expected.',
                        'neckline': f'${sl0["price"]:.2f}', 'target': f'${sl0["price"] - drop:.2f}',
                    })

    if len(swing_lows) >= 2 and len(swing_highs) >= 1:
        sl0, sl1 = swing_lows[-1], swing_lows[-2]
        sh0 = swing_highs[-1]
        if sh0['pos'] > sl1['pos'] and sl0['pos'] > sh0['pos']:
            rally = sh0['price'] - sl1['price']
            pullback = sh0['price'] - sl0['price']
            if rally > 0 and pullback > 0:
                retrace = pullback / rally * 100
                if retrace <= bc_ratio:
                    patterns.append({
                        'name': 'Bull Alt. Flag', 'bias': 'Bullish', 'type': 'Continuation',
                        'description': f'Sharp rally ${sl1["price"]:.2f}→${sh0["price"]:.2f}, shallow pullback to ${sl0["price"]:.2f} ({retrace:.1f}% retrace). Continuation higher expected.',
                        'neckline': f'${sh0["price"]:.2f}', 'target': f'${sh0["price"] + rally:.2f}',
                    })

    # ══════════════════════════════════════
    # WEDGE PATTERNS
    # ══════════════════════════════════════
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        sh0, sh1 = swing_highs[-1], swing_highs[-2]
        sl0, sl1 = swing_lows[-1], swing_lows[-2]

        highs_rising = sh0['price'] > sh1['price']
        lows_rising = sl0['price'] > sl1['price']
        highs_falling = sh0['price'] < sh1['price']
        lows_falling = sl0['price'] < sl1['price']

        # Range contracting?
        range_old = sh1['price'] - sl1['price']
        range_new = sh0['price'] - sl0['price']
        contracting = range_new < range_old * 0.85  # at least 15% narrower

        if contracting:
            if highs_rising and lows_rising:
                patterns.append({
                    'name': 'Ascending Wedge', 'bias': 'Bearish', 'type': 'Reversal',
                    'description': f'Rising highs and lows with contracting range. Range narrowed from ${range_old:.2f} to ${range_new:.2f}. Typically breaks down.',
                    'neckline': f'${sl0["price"]:.2f}', 'target': f'${sl0["price"] - range_old:.2f}',
                })
            elif highs_falling and lows_falling:
                patterns.append({
                    'name': 'Descending Wedge', 'bias': 'Bullish', 'type': 'Reversal',
                    'description': f'Falling highs and lows with contracting range. Range narrowed from ${range_old:.2f} to ${range_new:.2f}. Typically breaks up.',
                    'neckline': f'${sh0["price"]:.2f}', 'target': f'${sh0["price"] + range_old:.2f}',
                })
            elif (highs_rising and lows_falling) or (highs_falling and lows_rising):
                patterns.append({
                    'name': 'Symmetric Wedge', 'bias': 'Neutral', 'type': 'Continuation',
                    'description': f'Converging trendlines with range contracting from ${range_old:.2f} to ${range_new:.2f}. Breakout direction determines bias.',
                    'neckline': None, 'target': f'Measured move = ${range_old:.2f} from breakout',
                })

    return patterns


def generate_signals(df: pd.DataFrame) -> dict:
    """Generate trading signals based on indicator confluence."""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    signals = {}

    # ── Trend ──
    trend_score = 0
    trend_details = []
    if latest['Close'] > latest['SMA_50']:
        trend_score += 1
        trend_details.append(('Price > SMA 50', '+1', True))
    else:
        trend_score -= 1
        trend_details.append(('Price < SMA 50', '-1', False))
    if latest['Close'] > latest['SMA_200']:
        trend_score += 1
        trend_details.append(('Price > SMA 200', '+1', True))
    else:
        trend_score -= 1
        trend_details.append(('Price < SMA 200', '-1', False))
    if latest['SMA_50'] > latest['SMA_200']:
        trend_score += 1
        trend_details.append(('SMA 50 > SMA 200 (Golden Cross)', '+1', True))
    else:
        trend_score -= 1
        trend_details.append(('SMA 50 < SMA 200 (Death Cross)', '-1', False))
    if latest['EMA_9'] > latest['EMA_21']:
        trend_score += 1
        trend_details.append(('EMA 9 > EMA 21', '+1', True))
    else:
        trend_score -= 1
        trend_details.append(('EMA 9 < EMA 21', '-1', False))
    if latest['ADX'] > 25:
        if latest['DI_Plus'] > latest['DI_Minus']:
            trend_score += 1
            trend_details.append(('ADX > 25 & +DI > -DI (Strong uptrend)', '+1', True))
        else:
            trend_score -= 1
            trend_details.append(('ADX > 25 & +DI < -DI (Strong downtrend)', '-1', False))
    else:
        trend_details.append(('ADX < 25 (Weak/no trend)', '0', None))

    if trend_score >= 3:
        signals['trend'] = ('BULLISH', trend_score)
    elif trend_score <= -3:
        signals['trend'] = ('BEARISH', trend_score)
    else:
        signals['trend'] = ('NEUTRAL', trend_score)
    signals['trend_details'] = trend_details

    # ── Momentum ──
    mom_score = 0
    mom_details = []
    if latest['RSI'] > 50:
        mom_score += 1
        mom_details.append(('RSI > 50', '+1', True))
    elif latest['RSI'] < 50:
        mom_score -= 1
        mom_details.append(('RSI < 50', '-1', False))
    if latest['RSI'] > 70:
        mom_score -= 1  # Overbought penalty
        mom_details.append(('RSI > 70 (Overbought penalty)', '-1', False))
    elif latest['RSI'] < 30:
        mom_score += 1  # Oversold bonus
        mom_details.append(('RSI < 30 (Oversold bonus)', '+1', True))

    if latest['MACD'] > latest['MACD_Signal']:
        mom_score += 1
        mom_details.append(('MACD > Signal', '+1', True))
    else:
        mom_score -= 1
        mom_details.append(('MACD < Signal', '-1', False))
    if latest['MACD_Hist'] > prev['MACD_Hist']:
        mom_score += 1
        mom_details.append(('MACD Histogram expanding', '+1', True))
    else:
        mom_score -= 1
        mom_details.append(('MACD Histogram contracting', '-1', False))

    if latest['StochRSI_K'] > latest['StochRSI_D']:
        mom_score += 1
        mom_details.append(('StochRSI K > D', '+1', True))
    else:
        mom_score -= 1
        mom_details.append(('StochRSI K < D', '-1', False))

    # MFI contribution
    if latest['MFI'] > 50:
        mom_score += 1
        mom_details.append(('MFI > 50', '+1', True))
    elif latest['MFI'] < 50:
        mom_score -= 1
        mom_details.append(('MFI < 50', '-1', False))
    if latest['MFI'] > 80:
        mom_score -= 1  # Overbought penalty
        mom_details.append(('MFI > 80 (Overbought penalty)', '-1', False))
    elif latest['MFI'] < 20:
        mom_score += 1  # Oversold bonus
        mom_details.append(('MFI < 20 (Oversold bonus)', '+1', True))

    if mom_score >= 2:
        signals['momentum'] = ('BULLISH', mom_score)
    elif mom_score <= -2:
        signals['momentum'] = ('BEARISH', mom_score)
    else:
        signals['momentum'] = ('NEUTRAL', mom_score)
    signals['mom_details'] = mom_details

    # ── Volatility ──
    vol_notes = []
    if latest['Squeeze']:
        vol_notes.append('SQUEEZE ACTIVE - Expect breakout')
    if latest['BB_Pct'] > 1:
        vol_notes.append('Price above upper BB - Extended')
    elif latest['BB_Pct'] < 0:
        vol_notes.append('Price below lower BB - Oversold')
    if latest['ATR_Pct'] > df['ATR_Pct'].rolling(20).mean().iloc[-1] * 1.5:
        vol_notes.append('High volatility expansion')
    signals['volatility'] = vol_notes

    # ── Volume ──
    vol_signal = 'NORMAL'
    if latest['Volume'] > latest['Vol_SMA_20'] * 1.5:
        vol_signal = 'HIGH VOLUME'
    elif latest['Volume'] < latest['Vol_SMA_20'] * 0.5:
        vol_signal = 'LOW VOLUME'
    signals['volume'] = vol_signal

    # ── Overall Score ──
    total = trend_score + mom_score
    if total >= 5:
        signals['overall'] = 'STRONG BUY'
    elif total >= 2:
        signals['overall'] = 'BUY'
    elif total <= -5:
        signals['overall'] = 'STRONG SELL'
    elif total <= -2:
        signals['overall'] = 'SELL'
    else:
        signals['overall'] = 'NEUTRAL'
    signals['total_score'] = total

    return signals


def analyze_multi_timeframe_trend(stock) -> dict:
    """
    Analyze trend direction across Daily, Weekly, and Monthly timeframes.
    Uses SMA 20/50 crossover, price vs SMA 50, and higher-highs/lower-lows.
    """
    timeframes = {
        'Daily': {'period': '6mo', 'interval': '1d'},
        'Weekly': {'period': '2y', 'interval': '1wk'},
        'Monthly': {'period': '5y', 'interval': '1mo'},
    }

    results = {}

    for tf_name, params in timeframes.items():
        try:
            df = stock.history(period=params['period'], interval=params['interval'], auto_adjust=True)
            if df.empty or len(df) < 50:
                results[tf_name] = {
                    'trend': 'N/A',
                    'detail': f'Insufficient data ({len(df)} bars available, 50 required)',
                    'available': False,
                }
                continue

            close = df['Close']
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()

            current_price = close.iloc[-1]
            current_sma20 = sma_20.iloc[-1]
            current_sma50 = sma_50.iloc[-1]

            # Trend scoring
            score = 0
            details = []

            # 1. Price vs SMA 50
            if pd.notna(current_sma50):
                if current_price > current_sma50:
                    score += 1
                    details.append('Price above SMA 50')
                else:
                    score -= 1
                    details.append('Price below SMA 50')

            # 2. SMA 20 vs SMA 50 (golden/death cross)
            if pd.notna(current_sma20) and pd.notna(current_sma50):
                if current_sma20 > current_sma50:
                    score += 1
                    details.append('SMA 20 > SMA 50')
                else:
                    score -= 1
                    details.append('SMA 20 < SMA 50')

            # 3. SMA 50 slope (rising or falling over last 10 bars)
            if pd.notna(sma_50.iloc[-1]) and pd.notna(sma_50.iloc[-10]):
                if sma_50.iloc[-1] > sma_50.iloc[-10]:
                    score += 1
                    details.append('SMA 50 rising')
                else:
                    score -= 1
                    details.append('SMA 50 falling')

            # 4. Higher highs / lower lows (last 20 bars)
            recent = df.tail(20)
            mid = len(recent) // 2
            first_half_high = recent['High'].iloc[:mid].max()
            second_half_high = recent['High'].iloc[mid:].max()
            first_half_low = recent['Low'].iloc[:mid].min()
            second_half_low = recent['Low'].iloc[mid:].min()

            if second_half_high > first_half_high and second_half_low > first_half_low:
                score += 1
                details.append('Higher highs & higher lows')
            elif second_half_high < first_half_high and second_half_low < first_half_low:
                score -= 1
                details.append('Lower highs & lower lows')
            else:
                details.append('Mixed swing structure')

            # Determine trend
            if score >= 3:
                trend = 'STRONG UPTREND'
            elif score >= 1:
                trend = 'UPTREND'
            elif score <= -3:
                trend = 'STRONG DOWNTREND'
            elif score <= -1:
                trend = 'DOWNTREND'
            else:
                trend = 'SIDEWAYS'

            results[tf_name] = {
                'trend': trend,
                'score': score,
                'detail': ' | '.join(details),
                'price': current_price,
                'sma20': current_sma20 if pd.notna(current_sma20) else None,
                'sma50': current_sma50 if pd.notna(current_sma50) else None,
                'available': True,
            }

        except Exception as e:
            results[tf_name] = {
                'trend': 'N/A',
                'detail': f'Error fetching data: {str(e)}',
                'available': False,
            }

    return results


def get_fundamental_summary(info: dict) -> dict:
    """Extract key fundamental metrics."""
    def safe_get(key, default='N/A'):
        val = info.get(key, default)
        return val if val is not None else default

    fundamentals = {
        'Company': safe_get('longName', safe_get('shortName')),
        'Sector': safe_get('sector'),
        'Industry': safe_get('industry'),
        'Market Cap': safe_get('marketCap'),
        'P/E (Trailing)': safe_get('trailingPE'),
        'P/E (Forward)': safe_get('forwardPE'),
        'PEG Ratio': safe_get('pegRatio'),
        'P/S Ratio': safe_get('priceToSalesTrailing12Months'),
        'P/B Ratio': safe_get('priceToBook'),
        'EV/EBITDA': safe_get('enterpriseToEbitda'),
        'Revenue': safe_get('totalRevenue'),
        'Revenue Growth': safe_get('revenueGrowth'),
        'Profit Margin': safe_get('profitMargins'),
        'Operating Margin': safe_get('operatingMargins'),
        'ROE': safe_get('returnOnEquity'),
        'ROA': safe_get('returnOnAssets'),
        'Debt/Equity': safe_get('debtToEquity'),
        'Current Ratio': safe_get('currentRatio'),
        'Free Cash Flow': safe_get('freeCashflow'),
        'Dividend Yield': safe_get('dividendYield'),
        'Beta': safe_get('beta'),
        '52W High': safe_get('fiftyTwoWeekHigh'),
        '52W Low': safe_get('fiftyTwoWeekLow'),
        'Avg Volume': safe_get('averageVolume'),
        'Target Price': safe_get('targetMeanPrice'),
        'Analyst Rating': safe_get('recommendationKey') if safe_get('recommendationKey') not in ('none', 'None') else 'N/A',
        'Num Analysts': safe_get('numberOfAnalystOpinions'),
    }
    return fundamentals


def screen_halal_compliance(info: dict) -> dict:
    """
    Screen stock for Shariah (Halal) compliance using AAOIFI-based criteria.

    Criteria:
    1. Business Activity Screen - Exclude haram industries
    2. Debt Ratio - Total Debt / Market Cap < 33%
    3. Cash & Interest-Bearing Securities / Market Cap < 33%
    4. Haram Income / Total Revenue < 5%

    Returns dict with status, score, and details for each criterion.
    """
    results = {
        'status': 'UNKNOWN',
        'details': [],
        'pass_count': 0,
        'total_checks': 4,
    }

    sector = str(info.get('sector', '')).lower()
    industry = str(info.get('industry', '')).lower()

    # ── 1. Business Activity Screen ──
    haram_sectors = ['financial services']
    haram_industries = [
        'alcohol', 'beer', 'wine', 'spirits', 'brewing', 'distill', 'liquor',
        'gambling', 'casino', 'betting', 'lottery',
        'tobacco', 'cigarette', 'smoking',
        'pork', 'swine',
        'weapons', 'defense', 'ammunition', 'firearms', 'arms',
        'adult entertainment', 'pornography',
        'banks', 'banking', 'insurance', 'mortgage', 'lending',
        'credit services', 'consumer lending',
    ]

    sector_fail = sector in haram_sectors
    industry_fail = any(h in industry for h in haram_industries)

    if sector_fail or industry_fail:
        reason = f'Sector: {info.get("sector", "N/A")} | Industry: {info.get("industry", "N/A")}'
        results['details'].append({
            'criterion': 'Business Activity',
            'status': 'FAIL',
            'value': reason,
            'note': 'Industry involved in non-permissible activities',
        })
    else:
        results['details'].append({
            'criterion': 'Business Activity',
            'status': 'PASS',
            'value': f'Sector: {info.get("sector", "N/A")} | Industry: {info.get("industry", "N/A")}',
            'note': 'No haram business activity detected',
        })
        results['pass_count'] += 1

    # ── 2. Debt Ratio: Total Debt / Market Cap < 33% ──
    total_debt = info.get('totalDebt', None)
    market_cap = info.get('marketCap', None)

    if total_debt is not None and market_cap is not None and market_cap > 0:
        debt_ratio = (total_debt / market_cap) * 100
        if debt_ratio < 33:
            results['details'].append({
                'criterion': 'Debt Ratio',
                'status': 'PASS',
                'value': f'{debt_ratio:.1f}% (< 33%)',
                'note': f'Total Debt: {format_large_number(total_debt)} / Market Cap: {format_large_number(market_cap)}',
            })
            results['pass_count'] += 1
        else:
            results['details'].append({
                'criterion': 'Debt Ratio',
                'status': 'FAIL',
                'value': f'{debt_ratio:.1f}% (>= 33%)',
                'note': f'Total Debt: {format_large_number(total_debt)} / Market Cap: {format_large_number(market_cap)}',
            })
    else:
        results['details'].append({
            'criterion': 'Debt Ratio',
            'status': 'N/A',
            'value': 'Data not available',
            'note': 'Cannot determine - missing debt or market cap data',
        })

    # ── 3. Cash & Interest-Bearing Securities / Market Cap < 33% ──
    total_cash = info.get('totalCash', None)

    if total_cash is not None and market_cap is not None and market_cap > 0:
        cash_ratio = (total_cash / market_cap) * 100
        if cash_ratio < 33:
            results['details'].append({
                'criterion': 'Cash & Securities Ratio',
                'status': 'PASS',
                'value': f'{cash_ratio:.1f}% (< 33%)',
                'note': f'Total Cash: {format_large_number(total_cash)} / Market Cap: {format_large_number(market_cap)}',
            })
            results['pass_count'] += 1
        else:
            results['details'].append({
                'criterion': 'Cash & Securities Ratio',
                'status': 'FAIL',
                'value': f'{cash_ratio:.1f}% (>= 33%)',
                'note': f'Total Cash: {format_large_number(total_cash)} / Market Cap: {format_large_number(market_cap)}',
            })
    else:
        results['details'].append({
            'criterion': 'Cash & Securities Ratio',
            'status': 'N/A',
            'value': 'Data not available',
            'note': 'Cannot determine - missing cash data',
        })

    # ── 4. Haram Income / Total Revenue < 5% ──
    # yfinance doesn't provide haram income breakdown directly
    # We check if interest income is available as a proxy
    interest_income = info.get('interestIncome', None) or info.get('interestExpense', None)
    total_revenue = info.get('totalRevenue', None)

    if total_revenue is not None and total_revenue > 0 and interest_income is not None:
        haram_pct = (abs(interest_income) / total_revenue) * 100
        if haram_pct < 5:
            results['details'].append({
                'criterion': 'Haram Income Ratio',
                'status': 'PASS',
                'value': f'{haram_pct:.2f}% (< 5%)',
                'note': 'Interest-based income within permissible threshold',
            })
            results['pass_count'] += 1
        else:
            results['details'].append({
                'criterion': 'Haram Income Ratio',
                'status': 'FAIL',
                'value': f'{haram_pct:.2f}% (>= 5%)',
                'note': 'Interest-based income exceeds permissible threshold',
            })
    else:
        # If no interest data, and it's not a financial sector, likely pass
        if not sector_fail and not industry_fail:
            results['details'].append({
                'criterion': 'Haram Income Ratio',
                'status': 'LIKELY PASS',
                'value': 'No interest income data found',
                'note': 'Non-financial sector - likely compliant but verify independently',
            })
            results['pass_count'] += 1
        else:
            results['details'].append({
                'criterion': 'Haram Income Ratio',
                'status': 'N/A',
                'value': 'Data not available',
                'note': 'Cannot determine - financial sector requires manual review',
            })

    # ── Overall Status ──
    fail_count = sum(1 for d in results['details'] if d['status'] == 'FAIL')
    na_count = sum(1 for d in results['details'] if d['status'] in ('N/A',))

    if fail_count > 0:
        results['status'] = 'NON-COMPLIANT'
    elif na_count >= 2:
        results['status'] = 'UNCERTAIN'
    elif results['pass_count'] >= 3:
        results['status'] = 'HALAL'
    else:
        results['status'] = 'UNCERTAIN'

    return results


def format_large_number(num):
    """Format large numbers into readable strings."""
    if num == 'N/A' or num is None:
        return 'N/A'
    try:
        num = float(num)
    except (ValueError, TypeError):
        return str(num)
    if abs(num) >= 1e12:
        return f"${num/1e12:.2f}T"
    elif abs(num) >= 1e9:
        return f"${num/1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"${num/1e6:.2f}M"
    else:
        return f"${num:,.0f}"


def format_pct(val):
    """Format percentage values."""
    if val == 'N/A' or val is None:
        return 'N/A'
    try:
        return f"{float(val)*100:.1f}%"
    except (ValueError, TypeError):
        return str(val)


def format_ratio(val):
    """Format ratio values."""
    if val == 'N/A' or val is None:
        return 'N/A'
    try:
        return f"{float(val):.2f}"
    except (ValueError, TypeError):
        return str(val)


def plot_chart(df: pd.DataFrame, ticker: str, signals: dict, sr_levels: dict,
               divergences: list, output_dir: str) -> str:
    """Generate comprehensive technical analysis chart."""
    # Use last 120 days for chart clarity
    plot_df = df.tail(120).copy()

    fig = plt.figure(figsize=(18, 26))
    gs = gridspec.GridSpec(7, 1, height_ratios=[3, 1, 1, 1, 1, 1, 1], hspace=0.12)

    dates = plot_df.index
    x = np.arange(len(plot_df))  # Integer x-axis to eliminate weekend/holiday gaps

    # Custom formatter: map integer index → date label
    def format_date(val, pos):
        idx = int(round(val))
        if 0 <= idx < len(dates):
            return dates[idx].strftime('%b %d')
        return ''

    from matplotlib.ticker import FuncFormatter
    date_formatter = FuncFormatter(format_date)

    # ════════════════════════════════════════════════════════════════
    # Panel 1: Price + MAs + Bollinger + S/R
    # ════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0])

    # Candlestick-style coloring
    up = plot_df['Close'] >= plot_df['Open']
    down = ~up
    ax1.bar(x[up], (plot_df['Close'] - plot_df['Open'])[up], bottom=plot_df['Open'][up],
            width=0.6, color=BULL_COLOR, alpha=0.9)
    ax1.bar(x[up], (plot_df['High'] - plot_df['Close'])[up], bottom=plot_df['Close'][up],
            width=0.1, color=BULL_COLOR, alpha=0.9)
    ax1.bar(x[up], (plot_df['Open'] - plot_df['Low'])[up], bottom=plot_df['Low'][up],
            width=0.1, color=BULL_COLOR, alpha=0.9)
    ax1.bar(x[down], (plot_df['Close'] - plot_df['Open'])[down], bottom=plot_df['Open'][down],
            width=0.6, color=BEAR_COLOR, alpha=0.9)
    ax1.bar(x[down], (plot_df['High'] - plot_df['Open'])[down], bottom=plot_df['Open'][down],
            width=0.1, color=BEAR_COLOR, alpha=0.9)
    ax1.bar(x[down], (plot_df['Close'] - plot_df['Low'])[down], bottom=plot_df['Low'][down],
            width=0.1, color=BEAR_COLOR, alpha=0.9)

    # Moving Averages
    ax1.plot(x, plot_df['SMA_20'], color='#ffab40', linewidth=1, label='SMA 20', alpha=0.8)
    ax1.plot(x, plot_df['SMA_50'], color='#ffffff', linewidth=1.2, label='SMA 50', alpha=0.8)
    if plot_df['SMA_200'].notna().any():
        ax1.plot(x, plot_df['SMA_200'], color=ACCENT_PURPLE, linewidth=1.5, label='SMA 200', alpha=0.8)

    # Bollinger Bands
    ax1.fill_between(x, plot_df['BB_Upper'], plot_df['BB_Lower'],
                     alpha=0.08, color=ACCENT_BLUE, label='BB (20,2)')
    ax1.plot(x, plot_df['BB_Upper'], color=ACCENT_BLUE, linewidth=0.7, alpha=0.5)
    ax1.plot(x, plot_df['BB_Lower'], color=ACCENT_BLUE, linewidth=0.7, alpha=0.5)

    # Keltner Channel
    ax1.fill_between(x, plot_df['KC_Upper'], plot_df['KC_Lower'],
                     alpha=0.05, color='#ff9100', label='KC (20,10)')
    ax1.plot(x, plot_df['KC_Upper'], color='#ff9100', linewidth=0.7, alpha=0.5, linestyle='--')
    ax1.plot(x, plot_df['KC_Lower'], color='#ff9100', linewidth=0.7, alpha=0.5, linestyle='--')

    # Highlight squeeze zones (BB inside KC) on price panel
    squeeze_active = plot_df['Squeeze'] == True
    if squeeze_active.any():
        ax1.fill_between(x, plot_df['BB_Upper'], plot_df['BB_Lower'],
                         where=squeeze_active, alpha=0.15, color=BULL_COLOR, label='Squeeze Zone')

    # Support / Resistance lines
    price_range = plot_df['High'].max() - plot_df['Low'].min()
    for level in sr_levels.get('supports', []):
        if plot_df['Low'].min() - price_range*0.05 < level['price'] < plot_df['High'].max() + price_range*0.05:
            ax1.axhline(y=level['price'], color=BULL_COLOR, linestyle='--', alpha=0.5, linewidth=0.8)
            ax1.text(x[-1], level['price'], f" S: ${level['price']:.2f}", fontsize=7,
                     color=BULL_COLOR, va='center', alpha=0.8)
    for level in sr_levels.get('resistances', []):
        if plot_df['Low'].min() - price_range*0.05 < level['price'] < plot_df['High'].max() + price_range*0.05:
            ax1.axhline(y=level['price'], color=BEAR_COLOR, linestyle='--', alpha=0.5, linewidth=0.8)
            ax1.text(x[-1], level['price'], f" R: ${level['price']:.2f}", fontsize=7,
                     color=BEAR_COLOR, va='center', alpha=0.8)

    # Squeeze markers on price
    squeeze_mask = plot_df['Squeeze'] == True
    if squeeze_mask.any():
        ax1.scatter(x[squeeze_mask], plot_df['Low'][squeeze_mask] * 0.998,
                    marker='^', color=NEUTRAL_COLOR, s=15, alpha=0.7, zorder=5)

    # Signal badge
    overall = signals.get('overall', 'NEUTRAL')
    badge_color = BULL_COLOR if 'BUY' in overall else BEAR_COLOR if 'SELL' in overall else NEUTRAL_COLOR
    ax1.text(0.01, 0.97, f" {overall} ", transform=ax1.transAxes,
             fontsize=14, fontweight='bold', color='#1a1a2e',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=badge_color, alpha=0.9),
             va='top', ha='left')

    current_price = plot_df['Close'].iloc[-1]
    prev_close = plot_df['Close'].iloc[-2]
    pct_change = (current_price - prev_close) / prev_close * 100
    price_color = BULL_COLOR if pct_change >= 0 else BEAR_COLOR
    ax1.set_title(f'{ticker.upper()}  |  ${current_price:.2f}  ({pct_change:+.2f}%)',
                  fontsize=16, fontweight='bold', color=price_color, pad=15)
    ax1.legend(loc='upper right', fontsize=7, framealpha=0.3)
    ax1.set_ylabel('Price ($)')
    ax1.grid(True, alpha=0.15)
    ax1.tick_params(labelbottom=False)

    # ════════════════════════════════════════════════════════════════
    # Panel 2: Volume
    # ════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    vol_colors = [BULL_COLOR if c >= o else BEAR_COLOR for c, o in zip(plot_df['Close'], plot_df['Open'])]
    ax2.bar(x, plot_df['Volume'], color=vol_colors, alpha=0.6, width=0.6)
    ax2.plot(x, plot_df['Vol_SMA_20'], color=NEUTRAL_COLOR, linewidth=1, label='Vol SMA 20')
    ax2.set_ylabel('Volume')
    ax2.legend(loc='upper right', fontsize=7, framealpha=0.3)
    ax2.grid(True, alpha=0.15)
    ax2.tick_params(labelbottom=False)

    # ════════════════════════════════════════════════════════════════
    # Panel 3: RSI (4)
    # ════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(x, plot_df['RSI_4'], color='#ff9100', linewidth=1.2)
    ax3.axhline(y=80, color=BEAR_COLOR, linestyle='--', alpha=0.5, linewidth=0.7)
    ax3.axhline(y=20, color=BULL_COLOR, linestyle='--', alpha=0.5, linewidth=0.7)
    ax3.axhline(y=50, color='#555', linestyle='-', alpha=0.3, linewidth=0.5)
    ax3.fill_between(x, 80, plot_df['RSI_4'], where=plot_df['RSI_4'] >= 80,
                     color=BEAR_COLOR, alpha=0.2)
    ax3.fill_between(x, 20, plot_df['RSI_4'], where=plot_df['RSI_4'] <= 20,
                     color=BULL_COLOR, alpha=0.2)
    ax3.set_ylabel('RSI (4)')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.15)
    ax3.tick_params(labelbottom=False)
    rsi4_chart_val = plot_df['RSI_4'].iloc[-1]
    rsi4_chart_clr = BEAR_COLOR if rsi4_chart_val > 80 else BULL_COLOR if rsi4_chart_val < 20 else '#e0e0e0'
    ax3.text(0.99, 0.95, f'{rsi4_chart_val:.1f}', transform=ax3.transAxes,
             fontsize=9, color=rsi4_chart_clr, va='top', ha='right', fontweight='bold')

    # ════════════════════════════════════════════════════════════════
    # Panel 4: MFI (6)
    # ════════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(x, plot_df['MFI_6'], color='#18ffff', linewidth=1.2)
    ax4.axhline(y=80, color=BEAR_COLOR, linestyle='--', alpha=0.5, linewidth=0.7)
    ax4.axhline(y=20, color=BULL_COLOR, linestyle='--', alpha=0.5, linewidth=0.7)
    ax4.axhline(y=50, color='#555', linestyle='-', alpha=0.3, linewidth=0.5)
    ax4.fill_between(x, 80, plot_df['MFI_6'], where=plot_df['MFI_6'] >= 80,
                     color=BEAR_COLOR, alpha=0.2)
    ax4.fill_between(x, 20, plot_df['MFI_6'], where=plot_df['MFI_6'] <= 20,
                     color=BULL_COLOR, alpha=0.2)
    ax4.set_ylabel('MFI (6)')
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.15)
    ax4.tick_params(labelbottom=False)
    mfi6_chart_val = plot_df['MFI_6'].iloc[-1]
    mfi6_chart_clr = BEAR_COLOR if mfi6_chart_val > 80 else BULL_COLOR if mfi6_chart_val < 20 else '#e0e0e0'
    ax4.text(0.99, 0.95, f'{mfi6_chart_val:.1f}', transform=ax4.transAxes,
             fontsize=9, color=mfi6_chart_clr, va='top', ha='right', fontweight='bold')

    # ════════════════════════════════════════════════════════════════
    # Panel 5: RSI (14)
    # ════════════════════════════════════════════════════════════════
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.plot(x, plot_df['RSI'], color=ACCENT_BLUE, linewidth=1.2)
    ax5.axhline(y=70, color=BEAR_COLOR, linestyle='--', alpha=0.5, linewidth=0.7)
    ax5.axhline(y=30, color=BULL_COLOR, linestyle='--', alpha=0.5, linewidth=0.7)
    ax5.axhline(y=50, color='#555', linestyle='-', alpha=0.3, linewidth=0.5)
    ax5.fill_between(x, 70, plot_df['RSI'], where=plot_df['RSI'] >= 70,
                     color=BEAR_COLOR, alpha=0.2)
    ax5.fill_between(x, 30, plot_df['RSI'], where=plot_df['RSI'] <= 30,
                     color=BULL_COLOR, alpha=0.2)
    ax5.set_ylabel('RSI (14)')
    ax5.set_ylim(10, 90)
    ax5.grid(True, alpha=0.15)
    ax5.tick_params(labelbottom=False)
    rsi_chart_val = plot_df['RSI'].iloc[-1]
    rsi_chart_clr = BEAR_COLOR if rsi_chart_val > 70 else BULL_COLOR if rsi_chart_val < 30 else '#e0e0e0'
    ax5.text(0.99, 0.95, f'{rsi_chart_val:.1f}', transform=ax5.transAxes,
             fontsize=9, color=rsi_chart_clr, va='top', ha='right', fontweight='bold')

    # ════════════════════════════════════════════════════════════════
    # Panel 6: MFI (14)
    # ════════════════════════════════════════════════════════════════
    ax6 = fig.add_subplot(gs[5], sharex=ax1)
    ax6.plot(x, plot_df['MFI'], color='#18ffff', linewidth=1.2)
    ax6.axhline(y=80, color=BEAR_COLOR, linestyle='--', alpha=0.5, linewidth=0.7)
    ax6.axhline(y=20, color=BULL_COLOR, linestyle='--', alpha=0.5, linewidth=0.7)
    ax6.axhline(y=50, color='#555', linestyle='-', alpha=0.3, linewidth=0.5)
    ax6.fill_between(x, 80, plot_df['MFI'], where=plot_df['MFI'] >= 80,
                     color=BEAR_COLOR, alpha=0.2)
    ax6.fill_between(x, 20, plot_df['MFI'], where=plot_df['MFI'] <= 20,
                     color=BULL_COLOR, alpha=0.2)
    ax6.set_ylabel('MFI (14)')
    ax6.set_ylim(0, 100)
    ax6.grid(True, alpha=0.15)
    ax6.tick_params(labelbottom=False)
    mfi_chart_val = plot_df['MFI'].iloc[-1]
    mfi_chart_clr = BEAR_COLOR if mfi_chart_val > 80 else BULL_COLOR if mfi_chart_val < 20 else '#e0e0e0'
    ax6.text(0.99, 0.95, f'{mfi_chart_val:.1f}', transform=ax6.transAxes,
             fontsize=9, color=mfi_chart_clr, va='top', ha='right', fontweight='bold')

    # ════════════════════════════════════════════════════════════════
    # Panel 7: MACD
    # ════════════════════════════════════════════════════════════════
    ax7 = fig.add_subplot(gs[6], sharex=ax1)
    ax7.plot(x, plot_df['MACD'], color=ACCENT_BLUE, linewidth=1, label='MACD')
    ax7.plot(x, plot_df['MACD_Signal'], color=BEAR_COLOR, linewidth=1, label='Signal')
    hist_colors = [BULL_COLOR if v >= 0 else BEAR_COLOR for v in plot_df['MACD_Hist']]
    ax7.bar(x, plot_df['MACD_Hist'], color=hist_colors, alpha=0.5, width=0.6)
    ax7.axhline(y=0, color='#555', linewidth=0.5)
    ax7.set_ylabel('MACD')
    ax7.legend(loc='upper right', fontsize=7, framealpha=0.3)
    ax7.grid(True, alpha=0.15)
    ax7.set_xlabel('Date')
    ax7.xaxis.set_major_formatter(date_formatter)
    # Show ~10 evenly spaced date labels
    tick_spacing = max(1, len(x) // 10)
    ax7.set_xticks(x[::tick_spacing])
    plt.xticks(rotation=45)

    # ── Divergence annotations on price chart ──
    if divergences:
        for div in divergences:
            if 'Price' in div['pair']:
                div_color = BULL_COLOR if 'BULLISH' in div['type'] else BEAR_COLOR
                ax1.text(0.01, 0.88, f"[!] {div['type']} DIVERGENCE ({div['pair']})",
                         transform=ax1.transAxes, fontsize=8, fontweight='bold',
                         color=div_color,
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a2e',
                                   edgecolor=div_color, alpha=0.9),
                         va='top', ha='left')
                break

    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    chart_path = os.path.join(output_dir, f'{ticker.upper()}_{timestamp}_chart.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    return chart_path


def plot_fibonacci_chart(df: pd.DataFrame, ticker: str, fib_result: dict, output_dir: str) -> str:
    """Generate a Fibonacci retracement/extension chart overlaid on price."""
    if not fib_result.get('available', False):
        return None

    plot_df = df.tail(120).copy()
    dates = plot_df.index
    x = np.arange(len(plot_df))  # Integer x-axis to eliminate gaps
    close = plot_df['Close']
    high = plot_df['High']
    low = plot_df['Low']
    current = close.iloc[-1]

    from matplotlib.ticker import FuncFormatter
    def format_date(val, pos):
        idx = int(round(val))
        if 0 <= idx < len(dates):
            return dates[idx].strftime('%b %d')
        return ''
    date_formatter = FuncFormatter(format_date)

    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    # Candlestick bars
    up = close >= plot_df['Open']
    down = ~up
    ax.bar(x[up], (close - plot_df['Open'])[up], bottom=plot_df['Open'][up],
            width=0.6, color=BULL_COLOR, alpha=0.9)
    ax.bar(x[up], (high - close)[up], bottom=close[up],
            width=0.1, color=BULL_COLOR, alpha=0.9)
    ax.bar(x[up], (plot_df['Open'] - low)[up], bottom=low[up],
            width=0.1, color=BULL_COLOR, alpha=0.9)
    ax.bar(x[down], (close - plot_df['Open'])[down], bottom=plot_df['Open'][down],
            width=0.6, color=BEAR_COLOR, alpha=0.9)
    ax.bar(x[down], (high - plot_df['Open'])[down], bottom=plot_df['Open'][down],
            width=0.1, color=BEAR_COLOR, alpha=0.9)
    ax.bar(x[down], (close - low)[down], bottom=low[down],
            width=0.1, color=BEAR_COLOR, alpha=0.9)

    # Fibonacci level colors (golden tones for key levels)
    fib_colors = {
        '0%': '#ffffff',
        '23.6%': '#b0bec5',
        '38.2%': '#ffab40',
        '50%': '#ff9100',
        '61.8%': '#ff6d00',
        '78.6%': '#ff3d00',
        '100%': '#ffffff',
        '123.6%': '#18ffff',
        '138.2%': '#00e5ff',
        '150%': '#00bcd4',
        '161.8%': '#00acc1',
        '200%': '#0097a7',
        '261.8%': '#00838f',
    }

    active_tool = fib_result.get('active_tool', 'RETRACEMENT')
    active_levels = fib_result.get('active_levels', [])
    trend = fib_result['primary_trend']

    # Draw Fibonacci levels
    y_min = float('inf')
    y_max = float('-inf')
    for lv in active_levels:
        price = lv['price']
        label = lv['label']
        color = fib_colors.get(label, '#b0bec5')
        y_min = min(y_min, price)
        y_max = max(y_max, price)

        # Horizontal line across chart
        ax.axhline(y=price, color=color, linestyle='--', alpha=0.6, linewidth=1.0)

        # Label on right side
        dist_pct = lv['dist_pct']
        side_clr = BULL_COLOR if current > price else BEAR_COLOR
        ax.text(x[-1], price,
                f'  {label}  \\${price:.2f}  ({dist_pct:+.1f}%)',
                fontsize=8, color=color, va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a2e', alpha=0.8, edgecolor=color, linewidth=0.5))

    # Fill zones between key levels with subtle colors
    sorted_levels = sorted(active_levels, key=lambda x: x['price'])
    for i in range(len(sorted_levels) - 1):
        lower = sorted_levels[i]['price']
        upper = sorted_levels[i + 1]['price']
        # Highlight the zone where current price sits
        if lower <= current <= upper:
            ax.axhspan(lower, upper, alpha=0.12, color='#ffab40', zorder=0)

    # Mark swing high and low
    swing_high = fib_result['swing_high']
    swing_low = fib_result['swing_low']

    # Draw swing markers if they're in the visible date range
    swing_h_date = pd.Timestamp(fib_result['swing_high_date'])
    swing_l_date = pd.Timestamp(fib_result['swing_low_date'])

    # Match timezone if dates are tz-aware
    if dates.tz is not None:
        swing_h_date = swing_h_date.tz_localize(dates.tz)
        swing_l_date = swing_l_date.tz_localize(dates.tz)

    # Find integer x position for swing dates
    if swing_h_date >= dates[0]:
        sh_idx = dates.get_indexer([swing_h_date], method='nearest')[0]
        ax.annotate(f'Swing High\n\\${swing_high:.2f}', xy=(sh_idx, swing_high),
                    xytext=(sh_idx, swing_high + (y_max - y_min) * 0.04),
                    fontsize=8, color=BEAR_COLOR, fontweight='bold', ha='center',
                    arrowprops=dict(arrowstyle='->', color=BEAR_COLOR, lw=1.5))

    if swing_l_date >= dates[0]:
        sl_idx = dates.get_indexer([swing_l_date], method='nearest')[0]
        ax.annotate(f'Swing Low\n\\${swing_low:.2f}', xy=(sl_idx, swing_low),
                    xytext=(sl_idx, swing_low - (y_max - y_min) * 0.04),
                    fontsize=8, color=BULL_COLOR, fontweight='bold', ha='center',
                    arrowprops=dict(arrowstyle='->', color=BULL_COLOR, lw=1.5))

    # Current price marker
    ax.axhline(y=current, color='#ffffff', linestyle='-', alpha=0.4, linewidth=0.8)
    ax.text(x[0], current, f'  Current: \\${current:.2f}  ',
            fontsize=9, color='#ffffff', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor='#ffffff', alpha=0.9, linewidth=1))

    # Trend arrow
    trend_color = BULL_COLOR if 'UP' in trend else BEAR_COLOR
    trend_text = f'{trend} - Fibonacci {active_tool}'

    # Title
    pct_change = ((current - close.iloc[-2]) / close.iloc[-2]) * 100
    price_color = BULL_COLOR if pct_change >= 0 else BEAR_COLOR
    ax.set_title(f'{ticker.upper()}  |  Fibonacci {active_tool}  |  \\${current:.2f}  ({pct_change:+.2f}%)',
                 fontsize=14, fontweight='bold', color=price_color, pad=15)

    # Tool badge
    badge_color = BULL_COLOR if active_tool == 'EXTENSION' else '#ffab40'
    ax.text(0.01, 0.97, f' Fib {active_tool} ({trend}) ',
            transform=ax.transAxes, fontsize=11, fontweight='bold', color='#1a1a2e',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=badge_color, alpha=0.9),
            va='top', ha='left')

    # Current zone info
    if fib_result.get('current_zone'):
        zone_text = fib_result['current_zone'].replace('$', '\\$')
        ax.text(0.01, 0.90, zone_text,
                transform=ax.transAxes, fontsize=8, color='#ffab40',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a2e', edgecolor='#ffab40', alpha=0.9),
                va='top', ha='left')

    # Pad y-axis
    pad = (y_max - y_min) * 0.08
    ax.set_ylim(min(y_min, low.min()) - pad, max(y_max, high.max()) + pad)

    ax.set_ylabel('Price ($)', color='#e0e0e0')
    ax.set_xlabel('Date', color='#e0e0e0')
    ax.xaxis.set_major_formatter(date_formatter)
    tick_spacing = max(1, len(x) // 10)
    ax.set_xticks(x[::tick_spacing])
    ax.tick_params(colors='#e0e0e0')
    ax.grid(True, alpha=0.15)
    plt.xticks(rotation=45)

    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fib_chart_path = os.path.join(output_dir, f'{ticker.upper()}_{timestamp}_fib_chart.png')
    plt.savefig(fib_chart_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    return fib_chart_path


def plot_fibonacci_chart_tf(tf_df: pd.DataFrame, ticker: str, fib_result: dict,
                            timeframe: str, output_dir: str) -> str:
    """Generate a Fibonacci chart for weekly or monthly timeframe.

    Args:
        tf_df: OHLCV DataFrame (weekly or monthly bars)
        ticker: Ticker symbol
        fib_result: Output from analyze_fibonacci() for this timeframe
        timeframe: 'Weekly' or 'Monthly'
        output_dir: Directory for saving chart
    Returns:
        Path to saved chart image, or None if not available
    """
    if not fib_result.get('available', False):
        return None

    # Use more bars for bigger timeframes
    tail_n = 104 if timeframe == 'Weekly' else 60
    plot_df = tf_df.tail(tail_n).copy()
    if len(plot_df) < 10:
        return None

    dates = plot_df.index
    x = np.arange(len(plot_df))
    close = plot_df['Close']
    high = plot_df['High']
    low = plot_df['Low']
    current = close.iloc[-1]

    from matplotlib.ticker import FuncFormatter
    # Use month + year for weekly, just year/month for monthly
    def format_date(val, pos):
        idx = int(round(val))
        if 0 <= idx < len(dates):
            if timeframe == 'Monthly':
                return dates[idx].strftime('%b %Y')
            else:
                return dates[idx].strftime('%b %d \'%y')
        return ''
    date_formatter = FuncFormatter(format_date)

    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    # Candlestick bars
    up = close >= plot_df['Open']
    down = ~up
    bar_w = 0.6
    wick_w = 0.1
    ax.bar(x[up], (close - plot_df['Open'])[up], bottom=plot_df['Open'][up],
           width=bar_w, color=BULL_COLOR, alpha=0.9)
    ax.bar(x[up], (high - close)[up], bottom=close[up],
           width=wick_w, color=BULL_COLOR, alpha=0.9)
    ax.bar(x[up], (plot_df['Open'] - low)[up], bottom=low[up],
           width=wick_w, color=BULL_COLOR, alpha=0.9)
    ax.bar(x[down], (close - plot_df['Open'])[down], bottom=plot_df['Open'][down],
           width=bar_w, color=BEAR_COLOR, alpha=0.9)
    ax.bar(x[down], (high - plot_df['Open'])[down], bottom=plot_df['Open'][down],
           width=wick_w, color=BEAR_COLOR, alpha=0.9)
    ax.bar(x[down], (close - low)[down], bottom=low[down],
           width=wick_w, color=BEAR_COLOR, alpha=0.9)

    # Fibonacci level colors
    fib_colors = {
        '0%': '#ffffff', '23.6%': '#b0bec5', '38.2%': '#ffab40',
        '50%': '#ff9100', '61.8%': '#ff6d00', '78.6%': '#ff3d00',
        '100%': '#ffffff', '123.6%': '#18ffff', '138.2%': '#00e5ff',
        '150%': '#00bcd4', '161.8%': '#00acc1', '200%': '#0097a7', '261.8%': '#00838f',
    }

    active_tool = fib_result.get('active_tool', 'RETRACEMENT')
    active_levels = fib_result.get('active_levels', [])
    trend = fib_result['primary_trend']

    # Draw Fibonacci levels
    y_min = float('inf')
    y_max = float('-inf')
    for lv in active_levels:
        price = lv['price']
        label = lv['label']
        color = fib_colors.get(label, '#b0bec5')
        y_min = min(y_min, price)
        y_max = max(y_max, price)
        ax.axhline(y=price, color=color, linestyle='--', alpha=0.6, linewidth=1.0)
        dist_pct = lv['dist_pct']
        ax.text(x[-1], price,
                f'  {label}  \\${price:.2f}  ({dist_pct:+.1f}%)',
                fontsize=8, color=color, va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a2e', alpha=0.8,
                          edgecolor=color, linewidth=0.5))

    # Fill the zone where current price sits
    sorted_levels = sorted(active_levels, key=lambda lv: lv['price'])
    for i in range(len(sorted_levels) - 1):
        lo_p = sorted_levels[i]['price']
        hi_p = sorted_levels[i + 1]['price']
        if lo_p <= current <= hi_p:
            ax.axhspan(lo_p, hi_p, alpha=0.12, color='#ffab40', zorder=0)

    # Mark swing high and low
    swing_high = fib_result['swing_high']
    swing_low = fib_result['swing_low']
    swing_h_date = pd.Timestamp(fib_result['swing_high_date'])
    swing_l_date = pd.Timestamp(fib_result['swing_low_date'])

    if dates.tz is not None:
        if swing_h_date.tz is None:
            swing_h_date = swing_h_date.tz_localize(dates.tz)
        if swing_l_date.tz is None:
            swing_l_date = swing_l_date.tz_localize(dates.tz)

    if y_min == float('inf'):
        y_min = low.min()
    if y_max == float('-inf'):
        y_max = high.max()
    y_span = y_max - y_min if y_max > y_min else 1

    if swing_h_date >= dates[0]:
        sh_idx = dates.get_indexer([swing_h_date], method='nearest')[0]
        ax.annotate(f'Swing High\n\\${swing_high:.2f}', xy=(sh_idx, swing_high),
                    xytext=(sh_idx, swing_high + y_span * 0.04),
                    fontsize=8, color=BEAR_COLOR, fontweight='bold', ha='center',
                    arrowprops=dict(arrowstyle='->', color=BEAR_COLOR, lw=1.5))

    if swing_l_date >= dates[0]:
        sl_idx = dates.get_indexer([swing_l_date], method='nearest')[0]
        ax.annotate(f'Swing Low\n\\${swing_low:.2f}', xy=(sl_idx, swing_low),
                    xytext=(sl_idx, swing_low - y_span * 0.04),
                    fontsize=8, color=BULL_COLOR, fontweight='bold', ha='center',
                    arrowprops=dict(arrowstyle='->', color=BULL_COLOR, lw=1.5))

    # Current price marker
    ax.axhline(y=current, color='#ffffff', linestyle='-', alpha=0.4, linewidth=0.8)
    ax.text(x[0], current, f'  Current: \\${current:.2f}  ',
            fontsize=9, color='#ffffff', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                      edgecolor='#ffffff', alpha=0.9, linewidth=1))

    # Title & badge
    if len(close) >= 2:
        pct_change = ((current - close.iloc[-2]) / close.iloc[-2]) * 100
    else:
        pct_change = 0
    price_color = BULL_COLOR if pct_change >= 0 else BEAR_COLOR
    ax.set_title(
        f'{ticker.upper()}  |  {timeframe} Fibonacci {active_tool}  |  \\${current:.2f}  ({pct_change:+.2f}%)',
        fontsize=14, fontweight='bold', color=price_color, pad=15)

    badge_color = BULL_COLOR if active_tool == 'EXTENSION' else '#ffab40'
    ax.text(0.01, 0.97, f' {timeframe} Fib {active_tool} ({trend}) ',
            transform=ax.transAxes, fontsize=11, fontweight='bold', color='#1a1a2e',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=badge_color, alpha=0.9),
            va='top', ha='left')

    if fib_result.get('current_zone'):
        zone_text = fib_result['current_zone'].replace('$', '\\$')
        ax.text(0.01, 0.90, zone_text,
                transform=ax.transAxes, fontsize=8, color='#ffab40',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a2e',
                          edgecolor='#ffab40', alpha=0.9),
                va='top', ha='left')

    # Y-axis padding
    pad = y_span * 0.08
    ax.set_ylim(min(y_min, low.min()) - pad, max(y_max, high.max()) + pad)

    ax.set_ylabel('Price ($)', color='#e0e0e0')
    ax.set_xlabel('Date', color='#e0e0e0')
    ax.xaxis.set_major_formatter(date_formatter)
    tick_spacing = max(1, len(x) // 10)
    ax.set_xticks(x[::tick_spacing])
    ax.tick_params(colors='#e0e0e0')
    ax.grid(True, alpha=0.15)
    plt.xticks(rotation=45)

    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    chart_path = os.path.join(output_dir, f'{ticker.upper()}_{timestamp}_fib_{timeframe.lower()}.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    return chart_path


def compute_external_technical_evaluation(df, info=None):
    """
    Compute TradingView-style technical evaluation from dataframe.
    Returns a dict with all sub-sections for the PDF report.
    """
    import numpy as np
    import pandas as pd

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    close = df['Close']
    high = df['High']
    low = df['Low']
    current = latest['Close']

    # ══════════════════════════════════════════════════════
    # 1. MOVING AVERAGES DETAIL TABLE + Buy/Sell Counts
    # ══════════════════════════════════════════════════════
    ma_results = []

    # SMA periods
    for period in [10, 20, 30, 50, 100, 200]:
        col = f'SMA_{period}' if f'SMA_{period}' in df.columns else None
        if col and pd.notna(latest.get(col)):
            val = latest[col]
        elif len(df) >= period:
            val = close.tail(period).mean()
        else:
            continue
        signal = 'Buy' if current > val else 'Sell'
        ma_results.append({'name': f'SMA ({period})', 'value': val, 'signal': signal})

    # EMA periods
    for period in [10, 20, 30, 50, 100, 200]:
        col = f'EMA_{period}' if f'EMA_{period}' in df.columns else None
        if col and pd.notna(latest.get(col)):
            val = latest[col]
        elif len(df) >= period:
            val = close.ewm(span=period, adjust=False).mean().iloc[-1]
        else:
            continue
        signal = 'Buy' if current > val else 'Sell'
        ma_results.append({'name': f'EMA ({period})', 'value': val, 'signal': signal})

    # Ichimoku Baseline
    if pd.notna(latest.get('Ichi_Kijun')):
        val = latest['Ichi_Kijun']
        ma_results.append({'name': 'Ichimoku Base', 'value': val, 'signal': 'Buy' if current > val else 'Sell'})

    # VWAP
    if pd.notna(latest.get('VWAP')):
        val = latest['VWAP']
        ma_results.append({'name': 'VWAP', 'value': val, 'signal': 'Buy' if current > val else 'Sell'})

    # Hull MA (9)
    if len(df) >= 9:
        wma_half = close.rolling(5).apply(lambda x: np.average(x, weights=range(1, 6))).iloc[-1]
        wma_full = close.rolling(9).apply(lambda x: np.average(x, weights=range(1, 10))).iloc[-1]
        hull_raw = 2 * wma_half - wma_full
        ma_results.append({'name': 'Hull MA (9)', 'value': hull_raw, 'signal': 'Buy' if current > hull_raw else 'Sell'})

    ma_buy = sum(1 for m in ma_results if m['signal'] == 'Buy')
    ma_sell = sum(1 for m in ma_results if m['signal'] == 'Sell')
    ma_neutral = sum(1 for m in ma_results if m['signal'] == 'Neutral')
    ma_total = len(ma_results)

    if ma_buy > ma_sell + ma_neutral:
        if ma_buy >= ma_total * 0.8:
            ma_summary = 'STRONG BUY'
        else:
            ma_summary = 'BUY'
    elif ma_sell > ma_buy + ma_neutral:
        if ma_sell >= ma_total * 0.8:
            ma_summary = 'STRONG SELL'
        else:
            ma_summary = 'SELL'
    else:
        ma_summary = 'NEUTRAL'

    # ══════════════════════════════════════════════════════
    # 2. OSCILLATORS DETAIL TABLE + Buy/Sell Counts
    # ══════════════════════════════════════════════════════
    osc_results = []

    # RSI (14)
    rsi = latest.get('RSI')
    if pd.notna(rsi):
        if rsi < 30: sig = 'Buy'
        elif rsi > 70: sig = 'Sell'
        else: sig = 'Neutral'
        osc_results.append({'name': 'RSI (14)', 'value': f'{rsi:.2f}', 'signal': sig})

    # Stochastic %K (14,3,3)
    stoch_k = latest.get('StochRSI_K')
    stoch_d = latest.get('StochRSI_D')
    if pd.notna(stoch_k) and pd.notna(stoch_d):
        if stoch_k < 20 and stoch_k > stoch_d: sig = 'Buy'
        elif stoch_k > 80 and stoch_k < stoch_d: sig = 'Sell'
        else: sig = 'Neutral'
        osc_results.append({'name': 'Stoch %K', 'value': f'{stoch_k:.2f}', 'signal': sig})

    # CCI (20)
    cci = latest.get('CCI_20')
    if pd.notna(cci):
        prev_cci = prev.get('CCI_20', cci)
        if cci < -100 and cci > prev_cci: sig = 'Buy'
        elif cci > 100 and cci < prev_cci: sig = 'Sell'
        else: sig = 'Neutral'
        osc_results.append({'name': 'CCI (20)', 'value': f'{cci:.2f}', 'signal': sig})

    # ADX (14)
    adx = latest.get('ADX')
    di_plus = latest.get('DI_Plus')
    di_minus = latest.get('DI_Minus')
    if pd.notna(adx) and pd.notna(di_plus) and pd.notna(di_minus):
        if adx > 20 and di_plus > di_minus: sig = 'Buy'
        elif adx > 20 and di_plus < di_minus: sig = 'Sell'
        else: sig = 'Neutral'
        osc_results.append({'name': 'ADX (14)', 'value': f'{adx:.2f}', 'signal': sig})

    # Awesome Oscillator (AO)
    if len(df) >= 34:
        midpoint = (high + low) / 2
        ao = midpoint.rolling(5).mean().iloc[-1] - midpoint.rolling(34).mean().iloc[-1]
        ao_prev = midpoint.rolling(5).mean().iloc[-2] - midpoint.rolling(34).mean().iloc[-2]
        if ao > 0 and ao > ao_prev: sig = 'Buy'
        elif ao < 0 and ao < ao_prev: sig = 'Sell'
        else: sig = 'Neutral'
        osc_results.append({'name': 'Awesome Osc', 'value': f'{ao:.4f}', 'signal': sig})

    # MACD (12,26,9)
    macd = latest.get('MACD')
    macd_sig = latest.get('MACD_Signal')
    if pd.notna(macd) and pd.notna(macd_sig):
        if macd > macd_sig: sig = 'Buy'
        elif macd < macd_sig: sig = 'Sell'
        else: sig = 'Neutral'
        osc_results.append({'name': 'MACD (12,26)', 'value': f'{macd:.4f}', 'signal': sig})

    # Momentum (10)
    if len(df) >= 11:
        mom = current - close.iloc[-11]
        if mom > 0: sig = 'Buy'
        elif mom < 0: sig = 'Sell'
        else: sig = 'Neutral'
        osc_results.append({'name': 'Momentum (10)', 'value': f'{mom:.4f}', 'signal': sig})

    # Williams %R (14)
    if len(df) >= 14:
        h14 = high.tail(14).max()
        l14 = low.tail(14).min()
        wr = ((h14 - current) / (h14 - l14)) * -100 if (h14 - l14) > 0 else -50
        if wr < -80: sig = 'Buy'
        elif wr > -20: sig = 'Sell'
        else: sig = 'Neutral'
        osc_results.append({'name': 'Williams %R', 'value': f'{wr:.2f}', 'signal': sig})

    # Bull/Bear Power
    if pd.notna(latest.get('EMA_21')) and len(df) >= 13:
        ema13 = close.ewm(span=13, adjust=False).mean().iloc[-1]
        bull_power = high.iloc[-1] - ema13
        bear_power = low.iloc[-1] - ema13
        if bull_power > 0 and bull_power > (high.iloc[-2] - close.ewm(span=13, adjust=False).mean().iloc[-2]):
            sig = 'Buy'
        elif bear_power < 0 and bear_power < (low.iloc[-2] - close.ewm(span=13, adjust=False).mean().iloc[-2]):
            sig = 'Sell'
        else:
            sig = 'Neutral'
        osc_results.append({'name': 'Bull Bear Power', 'value': f'{bull_power:.4f} / {bear_power:.4f}', 'signal': sig})

    # Ultimate Oscillator
    if len(df) >= 28:
        bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
        uo = ((avg7.iloc[-1] * 4 + avg14.iloc[-1] * 2 + avg28.iloc[-1]) / 7) * 100
        if uo < 30: sig = 'Buy'
        elif uo > 70: sig = 'Sell'
        else: sig = 'Neutral'
        osc_results.append({'name': 'Ultimate Osc', 'value': f'{uo:.2f}', 'signal': sig})

    # MFI
    mfi = latest.get('MFI')
    if pd.notna(mfi):
        if mfi < 20: sig = 'Buy'
        elif mfi > 80: sig = 'Sell'
        else: sig = 'Neutral'
        osc_results.append({'name': 'MFI (14)', 'value': f'{mfi:.2f}', 'signal': sig})

    osc_buy = sum(1 for o in osc_results if o['signal'] == 'Buy')
    osc_sell = sum(1 for o in osc_results if o['signal'] == 'Sell')
    osc_neutral = sum(1 for o in osc_results if o['signal'] == 'Neutral')

    if osc_buy > osc_sell + osc_neutral:
        osc_summary = 'STRONG BUY' if osc_buy >= len(osc_results) * 0.7 else 'BUY'
    elif osc_sell > osc_buy + osc_neutral:
        osc_summary = 'STRONG SELL' if osc_sell >= len(osc_results) * 0.7 else 'SELL'
    else:
        osc_summary = 'NEUTRAL'

    # ══════════════════════════════════════════════════════
    # 3. OVERALL SUMMARY
    # ══════════════════════════════════════════════════════
    total_buy = ma_buy + osc_buy
    total_sell = ma_sell + osc_sell
    total_neutral = ma_neutral + osc_neutral
    total_all = total_buy + total_sell + total_neutral

    if total_buy > total_sell + total_neutral:
        if total_buy >= total_all * 0.75:
            overall_summary = 'STRONG BUY'
        else:
            overall_summary = 'BUY'
    elif total_sell > total_buy + total_neutral:
        if total_sell >= total_all * 0.75:
            overall_summary = 'STRONG SELL'
        else:
            overall_summary = 'SELL'
    else:
        overall_summary = 'NEUTRAL'

    # ══════════════════════════════════════════════════════
    # 4. BARCHART-STYLE OPINION (Short/Medium/Long)
    # ══════════════════════════════════════════════════════
    # Short-term (EMA9 vs EMA21, RSI, StochRSI, MACD hist direction)
    st_score = 0
    if pd.notna(latest.get('EMA_9')) and pd.notna(latest.get('EMA_21')):
        st_score += 1 if latest['EMA_9'] > latest['EMA_21'] else -1
    if pd.notna(rsi):
        st_score += 1 if rsi > 50 else -1
    if pd.notna(stoch_k) and pd.notna(stoch_d):
        st_score += 1 if stoch_k > stoch_d else -1
    if pd.notna(latest.get('MACD_Hist')) and pd.notna(prev.get('MACD_Hist')):
        st_score += 1 if latest['MACD_Hist'] > prev['MACD_Hist'] else -1

    if st_score >= 3: st_signal = 'STRONG BUY'
    elif st_score >= 1: st_signal = 'BUY'
    elif st_score <= -3: st_signal = 'STRONG SELL'
    elif st_score <= -1: st_signal = 'SELL'
    else: st_signal = 'NEUTRAL'

    # Medium-term (SMA20 vs SMA50, price vs SMA50, ADX trend)
    mt_score = 0
    if pd.notna(latest.get('SMA_20')) and pd.notna(latest.get('SMA_50')):
        mt_score += 1 if latest['SMA_20'] > latest['SMA_50'] else -1
    mt_score += 1 if current > latest.get('SMA_50', current) else -1
    if pd.notna(adx) and pd.notna(di_plus) and pd.notna(di_minus):
        if adx > 25:
            mt_score += 1 if di_plus > di_minus else -1
    if pd.notna(latest.get('MACD')) and pd.notna(latest.get('MACD_Signal')):
        mt_score += 1 if latest['MACD'] > latest['MACD_Signal'] else -1

    if mt_score >= 3: mt_signal = 'STRONG BUY'
    elif mt_score >= 1: mt_signal = 'BUY'
    elif mt_score <= -3: mt_signal = 'STRONG SELL'
    elif mt_score <= -1: mt_signal = 'SELL'
    else: mt_signal = 'NEUTRAL'

    # Long-term (SMA50 vs SMA200, price vs SMA200)
    lt_score = 0
    if pd.notna(latest.get('SMA_50')) and pd.notna(latest.get('SMA_200')):
        lt_score += 1 if latest['SMA_50'] > latest['SMA_200'] else -1
    if pd.notna(latest.get('SMA_200')):
        lt_score += 1 if current > latest['SMA_200'] else -1
    if pd.notna(latest.get('SMA_50')):
        lt_score += 1 if current > latest['SMA_50'] else -1

    if lt_score >= 2: lt_signal = 'STRONG BUY'
    elif lt_score >= 1: lt_signal = 'BUY'
    elif lt_score <= -2: lt_signal = 'STRONG SELL'
    elif lt_score <= -1: lt_signal = 'SELL'
    else: lt_signal = 'NEUTRAL'

    barchart_opinion = {
        'short_term': {'signal': st_signal, 'score': st_score},
        'medium_term': {'signal': mt_signal, 'score': mt_score},
        'long_term': {'signal': lt_signal, 'score': lt_score},
    }

    # ══════════════════════════════════════════════════════
    # 5. PIVOT POINTS (Classic, Fibonacci, Woodie, Camarilla)
    # ══════════════════════════════════════════════════════
    h = high.iloc[-1]
    l = low.iloc[-1]
    c = close.iloc[-1]

    # Classic
    pp = (h + l + c) / 3
    classic = {
        'PP': pp,
        'R1': 2 * pp - l, 'R2': pp + (h - l), 'R3': h + 2 * (pp - l),
        'S1': 2 * pp - h, 'S2': pp - (h - l), 'S3': l - 2 * (h - pp),
    }

    # Fibonacci
    fib_pp = pp
    fib_range = h - l
    fibonacci = {
        'PP': fib_pp,
        'R1': fib_pp + 0.382 * fib_range, 'R2': fib_pp + 0.618 * fib_range, 'R3': fib_pp + 1.0 * fib_range,
        'S1': fib_pp - 0.382 * fib_range, 'S2': fib_pp - 0.618 * fib_range, 'S3': fib_pp - 1.0 * fib_range,
    }

    # Woodie
    woodie_pp = (h + l + 2 * c) / 4
    woodie = {
        'PP': woodie_pp,
        'R1': 2 * woodie_pp - l, 'R2': woodie_pp + (h - l), 'R3': h + 2 * (woodie_pp - l),
        'S1': 2 * woodie_pp - h, 'S2': woodie_pp - (h - l), 'S3': l - 2 * (h - woodie_pp),
    }

    # Camarilla
    camarilla = {
        'PP': pp,
        'R1': c + 1.1 * (h - l) / 12, 'R2': c + 1.1 * (h - l) / 6, 'R3': c + 1.1 * (h - l) / 4,
        'S1': c - 1.1 * (h - l) / 12, 'S2': c - 1.1 * (h - l) / 6, 'S3': c - 1.1 * (h - l) / 4,
    }

    pivots = {
        'Classic': classic,
        'Fibonacci': fibonacci,
        'Woodie': woodie,
        'Camarilla': camarilla,
    }

    return {
        'ma_results': ma_results,
        'ma_summary': ma_summary,
        'ma_buy': ma_buy, 'ma_sell': ma_sell, 'ma_neutral': ma_neutral,
        'osc_results': osc_results,
        'osc_summary': osc_summary,
        'osc_buy': osc_buy, 'osc_sell': osc_sell, 'osc_neutral': osc_neutral,
        'overall_summary': overall_summary,
        'total_buy': total_buy, 'total_sell': total_sell, 'total_neutral': total_neutral,
        'barchart_opinion': barchart_opinion,
        'pivots': pivots,
    }


def compute_multi_timeframe_evaluation(ticker_symbol: str, daily_df=None) -> dict:
    """
    Compute TradingView/Investing.com-style technical evaluation across multiple timeframes.
    Timeframes: 5min, 15min, 30min, 4 hours, Daily, Weekly, Monthly.

    Uses yfinance to fetch data for each interval, computes indicators, and runs evaluation.
    Falls back gracefully if a timeframe has insufficient data.
    """
    import yfinance as yf

    timeframes = [
        {'label': '5 Min',   'interval': '5m',  'period': '60d',  'min_bars': 50},
        {'label': '15 Min',  'interval': '15m', 'period': '60d',  'min_bars': 50},
        {'label': '30 Min',  'interval': '30m', 'period': '60d',  'min_bars': 50},
        {'label': '4 Hours', 'interval': '60m', 'period': '730d', 'min_bars': 50},
        {'label': 'Daily',   'interval': '1d',  'period': '2y',   'min_bars': 200},
        {'label': 'Weekly',  'interval': '1wk', 'period': '5y',   'min_bars': 50},
        {'label': 'Monthly', 'interval': '1mo', 'period': '10y',  'min_bars': 30},
    ]

    results = {}
    stock = yf.Ticker(ticker_symbol)

    for tf in timeframes:
        try:
            if tf['label'] == 'Daily' and daily_df is not None and len(daily_df) >= tf['min_bars']:
                # Reuse the daily dataframe we already have (with indicators computed)
                df_tf = daily_df.copy()
            else:
                df_tf = stock.history(period=tf['period'], interval=tf['interval'])
                if df_tf is None or len(df_tf) < tf['min_bars']:
                    results[tf['label']] = {'available': False, 'reason': f'Insufficient data ({len(df_tf) if df_tf is not None else 0} bars)'}
                    continue

                # For 4 Hours: aggregate 60min bars into 4-hour bars
                if tf['label'] == '4 Hours':
                    df_tf = df_tf.resample('4h').agg({
                        'Open': 'first', 'High': 'max', 'Low': 'min',
                        'Close': 'last', 'Volume': 'sum'
                    }).dropna()
                    if len(df_tf) < tf['min_bars']:
                        results[tf['label']] = {'available': False, 'reason': f'Insufficient 4H bars ({len(df_tf)})'}
                        continue

                # Compute indicators
                df_tf = compute_indicators(df_tf)

            eval_result = compute_external_technical_evaluation(df_tf)
            eval_result['available'] = True
            results[tf['label']] = eval_result

        except Exception as e:
            results[tf['label']] = {'available': False, 'reason': str(e)[:80]}

    return results


def generate_report(ticker: str, df: pd.DataFrame, info: dict, signals: dict,
                    sr_levels: dict, divergences: list, mtf_trends: dict,
                    chart_patterns: list, eccentric_patterns: list, halal_result: dict, fib_result: dict,
                    mtf_fib: dict, mtf_eval: dict,
                    chart_path: str, fib_chart_path: str,
                    fib_chart_weekly: str, fib_chart_monthly: str,
                    output_dir: str) -> str:
    """Generate a comprehensive color-coded PDF report with chart."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    fundamentals = get_fundamental_summary(info)
    current = latest['Close']
    chg = current - prev['Close']
    chg_pct = (chg / prev['Close']) * 100

    # ── Colors ──
    BG_DARK = HexColor('#1a1a2e')
    BG_SECTION = HexColor('#16213e')
    BG_ROW_ALT = HexColor('#1c2a4a')
    BORDER = HexColor('#0f3460')
    BULL = HexColor('#00e676')
    BEAR = HexColor('#ff1744')
    NEUTRAL_C = HexColor('#9e9e9e')
    WARN = HexColor('#ffd740')
    WHITE = HexColor('#e0e0e0')
    MUTED = HexColor('#888888')
    BLUE = HexColor('#448aff')

    # ── Styles ──
    s_title = ParagraphStyle('Title', fontName='Helvetica-Bold', fontSize=22, textColor=BULL if chg >= 0 else BEAR, spaceAfter=2)
    s_subtitle = ParagraphStyle('Sub', fontName='Helvetica', fontSize=9, textColor=MUTED, spaceAfter=2)
    s_section = ParagraphStyle('Section', fontName='Helvetica-Bold', fontSize=12, textColor=BLUE, spaceBefore=16, spaceAfter=6)
    s_label = ParagraphStyle('Label', fontName='Helvetica', fontSize=8.5, textColor=MUTED)
    s_val = ParagraphStyle('Val', fontName='Helvetica-Bold', fontSize=8.5, textColor=WHITE)
    s_disclaimer = ParagraphStyle('Disc', fontName='Helvetica', fontSize=7, textColor=MUTED, alignment=TA_CENTER, spaceBefore=20)

    def colored(text, color):
        return f'<font color="{color}">{text}</font>'

    def b(text):
        return f'<b>{text}</b>'

    def val_style(text, color=WHITE):
        return Paragraph(f'<font color="{color}"><b>{text}</b></font>', s_val)

    def label_p(text):
        return Paragraph(text, s_label)

    def signal_clr(sig):
        s = sig.upper()
        if 'STRONG BUY' in s or 'STRONG BULLISH' in s: return BULL
        if 'BUY' in s or 'BULLISH' in s: return BULL
        if 'STRONG SELL' in s or 'STRONG BEARISH' in s: return BEAR
        if 'SELL' in s or 'BEARISH' in s: return BEAR
        return NEUTRAL_C

    def make_table(data, col_widths=None):
        if col_widths is None:
            col_widths = [2.2*inch, 4.3*inch]
        t = Table(data, colWidths=col_widths, hAlign='LEFT')
        style_cmds = [
            ('BACKGROUND', (0, 0), (-1, -1), BG_SECTION),
            ('TEXTCOLOR', (0, 0), (-1, -1), WHITE),
            ('FONTSIZE', (0, 0), (-1, -1), 8.5),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (0, -1), 10),
            ('LEFTPADDING', (1, 0), (1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.25, BORDER),
            ('ROUNDEDCORNERS', [4, 4, 4, 4]),
        ]
        for i in range(len(data)):
            if i % 2 == 1:
                style_cmds.append(('BACKGROUND', (0, i), (-1, i), BG_ROW_ALT))
        t.setStyle(TableStyle(style_cmds))
        return t

    # ── Build story ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f'{ticker.upper()}_{timestamp}_report.pdf')
    doc = SimpleDocTemplate(report_path, pagesize=letter,
                            topMargin=0.5*inch, bottomMargin=0.5*inch,
                            leftMargin=0.6*inch, rightMargin=0.6*inch)

    story = []

    # ── Chart Page ──
    if chart_path and os.path.exists(chart_path):
        page_w = letter[0] - 1.2 * inch  # available width with margins
        page_h = letter[1] - 1.2 * inch  # available height with margins (extra buffer)
        # Get image dimensions to scale proportionally
        from reportlab.lib.utils import ImageReader
        img_reader = ImageReader(chart_path)
        img_w, img_h = img_reader.getSize()
        aspect = img_h / img_w
        # Fit to page: try width first, then constrain by height
        display_w = page_w
        display_h = display_w * aspect
        if display_h > page_h:
            display_h = page_h
            display_w = display_h / aspect
        story.append(Image(chart_path, width=display_w, height=display_h))
        story.append(PageBreak())

    # ── Fibonacci Chart (Page 2) ──
    if fib_chart_path and os.path.exists(fib_chart_path):
        from reportlab.lib.utils import ImageReader as ImgR2
        page_w2 = letter[0] - 1.2 * inch
        page_h2 = letter[1] - 1.2 * inch
        img_r2 = ImgR2(fib_chart_path)
        iw2, ih2 = img_r2.getSize()
        aspect2 = ih2 / iw2
        dw2 = page_w2
        dh2 = dw2 * aspect2
        if dh2 > page_h2:
            dh2 = page_h2
            dw2 = dh2 / aspect2
        story.append(Image(fib_chart_path, width=dw2, height=dh2))
        story.append(PageBreak())

    # ── Weekly Fibonacci Chart ──
    if fib_chart_weekly and os.path.exists(fib_chart_weekly):
        from reportlab.lib.utils import ImageReader as ImgRW
        page_ww = letter[0] - 1.2 * inch
        page_hw = letter[1] - 1.2 * inch
        img_rw = ImgRW(fib_chart_weekly)
        iww, ihw = img_rw.getSize()
        aspect_w = ihw / iww
        dww = page_ww
        dhw = dww * aspect_w
        if dhw > page_hw:
            dhw = page_hw
            dww = dhw / aspect_w
        story.append(Image(fib_chart_weekly, width=dww, height=dhw))
        story.append(PageBreak())

    # ── Monthly Fibonacci Chart ──
    if fib_chart_monthly and os.path.exists(fib_chart_monthly):
        from reportlab.lib.utils import ImageReader as ImgRM
        page_wm = letter[0] - 1.2 * inch
        page_hm = letter[1] - 1.2 * inch
        img_rm = ImgRM(fib_chart_monthly)
        iwm, ihm = img_rm.getSize()
        aspect_m = ihm / iwm
        dwm = page_wm
        dhm = dwm * aspect_m
        if dhm > page_hm:
            dhm = page_hm
            dwm = dhm / aspect_m
        story.append(Image(fib_chart_monthly, width=dwm, height=dhm))
        story.append(PageBreak())

    # ── Header ──
    price_clr = BULL if chg >= 0 else BEAR
    s_ticker = ParagraphStyle('Ticker', fontName='Helvetica-Bold', fontSize=20, textColor=price_clr, spaceAfter=2, leading=24)
    s_price_line = ParagraphStyle('PriceLine', fontName='Helvetica-Bold', fontSize=13, textColor=price_clr, spaceAfter=6, leading=16)

    story.append(Paragraph(f'{ticker.upper()}', s_ticker))
    story.append(Paragraph(f'${current:.2f} &nbsp;&nbsp; ({chg:+.2f} / {chg_pct:+.2f}%)', s_price_line))
    story.append(Paragraph(f'{fundamentals["Company"]}', s_subtitle))
    story.append(Paragraph(f'{fundamentals["Sector"]} &nbsp;|&nbsp; {fundamentals["Industry"]}', s_subtitle))
    story.append(Paragraph(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', s_subtitle))
    story.append(Spacer(1, 8))

    # ── Signal Summary ──
    overall = signals.get('overall', 'NEUTRAL')
    score = signals.get('total_score', 0)
    trend_sig, trend_score = signals.get('trend', ('NEUTRAL', 0))
    mom_sig, mom_score = signals.get('momentum', ('NEUTRAL', 0))
    vol_signal = signals.get('volume', 'NORMAL')
    vol_notes = signals.get('volatility', [])

    overall_clr = signal_clr(overall)
    # ── Shariah (Halal) Compliance ──
    story.append(Paragraph('SHARIAH (HALAL) COMPLIANCE', s_section))
    halal_status = halal_result.get('status', 'UNKNOWN')
    if halal_status == 'HALAL':
        status_clr = BULL
        status_icon = 'HALAL'
    elif halal_status == 'NON-COMPLIANT':
        status_clr = BEAR
        status_icon = 'NON-COMPLIANT (HARAM)'
    else:
        status_clr = WARN
        status_icon = 'UNCERTAIN - Requires Manual Review'

    halal_data = [
        [label_p('Status'), val_style(status_icon, status_clr)],
        [label_p('Score'), val_style(f'{halal_result.get("pass_count", 0)}/{halal_result.get("total_checks", 4)} criteria passed',
            BULL if halal_result.get("pass_count", 0) >= 3 else BEAR if halal_result.get("pass_count", 0) <= 1 else WARN)],
    ]

    for detail in halal_result.get('details', []):
        d_status = detail['status']
        d_clr = BULL if d_status == 'PASS' else BEAR if d_status == 'FAIL' else WARN if d_status == 'LIKELY PASS' else NEUTRAL_C
        halal_data.append([label_p(f'  {detail["criterion"]}'), val_style(f'{d_status} - {detail["value"]}', d_clr)])
        if detail.get('note'):
            halal_data.append([label_p(f''), Paragraph(f'<font color="{MUTED}" size="7">{detail["note"]}</font>', s_val)])

    halal_data.append([label_p('  Note'), Paragraph(
        f'<font color="{MUTED}" size="7">Based on AAOIFI screening criteria. For authoritative rulings, consult a qualified Shariah scholar or use certified platforms (Musaffa, Zoya).</font>', s_val)])

    story.append(make_table(halal_data))

    # ── Analyst Consensus ──
    story.append(Paragraph('ANALYST CONSENSUS', s_section))
    rating = fundamentals['Analyst Rating']
    # Normalize 'none' / None / 'N/A' → proper N/A
    if rating in (None, 'none', 'None', 'N/A', ''):
        rating = 'N/A'
    rating_clr = BULL if rating in ('buy', 'strong_buy') else BEAR if rating in ('sell', 'strong_sell') else NEUTRAL_C
    analyst_data = [
        [label_p('Rating'), val_style(str(rating).upper().replace('_', ' ') if rating != 'N/A' else 'N/A', rating_clr)],
        [label_p('Target Price'), val_style(f'${fundamentals["Target Price"]}' if fundamentals["Target Price"] != 'N/A' else 'N/A')],
        [label_p('# of Analysts'), val_style(str(fundamentals['Num Analysts']))],
    ]
    if fundamentals['Target Price'] != 'N/A':
        try:
            target = float(fundamentals['Target Price'])
            upside = ((target - current) / current) * 100
            analyst_data.append([label_p('Implied Upside'), val_style(f'{upside:+.1f}%', BULL if upside > 0 else BEAR)])
        except (ValueError, TypeError):
            pass
    # Show detailed breakdown if available
    breakdown = info.get('_analyst_breakdown')
    if breakdown:
        total = breakdown['total']
        bd_text = (
            f'<font color="{BULL}"><b>Strong Buy: {breakdown["strongBuy"]}</b></font>'
            f' &nbsp; <font color="{BULL}"><b>Buy: {breakdown["buy"]}</b></font>'
            f' &nbsp; <font color="{NEUTRAL_C}"><b>Hold: {breakdown["hold"]}</b></font>'
            f' &nbsp; <font color="{BEAR}"><b>Sell: {breakdown["sell"]}</b></font>'
            f' &nbsp; <font color="{BEAR}"><b>Strong Sell: {breakdown["strongSell"]}</b></font>'
        )
        analyst_data.append([label_p('Breakdown'), Paragraph(bd_text, s_val)])
    # Target high/low if available
    tgt_high = info.get('targetHighPrice')
    tgt_low = info.get('targetLowPrice')
    if tgt_high and tgt_low:
        analyst_data.append([label_p('Target Range'), val_style(f'${tgt_low:.2f} — ${tgt_high:.2f}')])
    story.append(make_table(analyst_data))

    # ── Price Summary ──
    story.append(Paragraph('PRICE SUMMARY', s_section))
    price_data = [
        [label_p('Current Price'), val_style(f'${current:.2f} ({chg:+.2f} / {chg_pct:+.2f}%)', price_clr)],
    ]
    if fundamentals['52W High'] != 'N/A' and fundamentals['52W Low'] != 'N/A':
        try:
            high52 = float(fundamentals['52W High'])
            low52 = float(fundamentals['52W Low'])
            pct_h = ((current - high52) / high52) * 100
            pct_l = ((current - low52) / low52) * 100
            h_clr = BEAR if pct_h < 0 else BULL
            l_clr = BULL if pct_l > 0 else BEAR
            price_data.append([label_p('52W High'), Paragraph(
                f'<font color="{WHITE}"><b>${high52:.2f}</b></font>'
                f' &nbsp; <font color="{h_clr}"><b>{pct_h:+.2f}%</b></font>', s_val)])
            price_data.append([label_p('52W Low'), Paragraph(
                f'<font color="{WHITE}"><b>${low52:.2f}</b></font>'
                f' &nbsp; <font color="{l_clr}"><b>{pct_l:+.2f}%</b></font>', s_val)])
        except (ValueError, TypeError):
            price_data.append([label_p('52-Week High'), val_style(f'${fundamentals["52W High"]}')])
            price_data.append([label_p('52-Week Low'), val_style(f'${fundamentals["52W Low"]}')])
    else:
        price_data.append([label_p('52-Week High'), val_style(f'${fundamentals["52W High"]}')])
        price_data.append([label_p('52-Week Low'), val_style(f'${fundamentals["52W Low"]}')])
    story.append(make_table(price_data))


    story.append(Paragraph('SIGNAL SUMMARY', s_section))

    # Overall badge row
    overall_range = 'Trend ({t}) + Momentum ({m}) = {total}'.format(
        t=trend_score, m=mom_score, total=score)
    badge_data = [[
        Paragraph(f'<font color="{BG_DARK}" size="14"><b>&nbsp; {overall} &nbsp;</b></font>', s_val),
        Paragraph(f'<font color="{MUTED}">Confluence Score: {score} &nbsp; [{overall_range}]</font>', s_val)
    ]]
    badge_t = Table(badge_data, colWidths=[2.5*inch, 4*inch], hAlign='LEFT')
    badge_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), overall_clr),
        ('BACKGROUND', (1, 0), (1, 0), BG_SECTION),
        ('ROUNDEDCORNERS', [6, 6, 6, 6]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(badge_t)
    story.append(Spacer(1, 2))

    # Confluence scale
    scale_text = 'STRONG BUY >= 5 | BUY >= 2 | NEUTRAL -1 to 1 | SELL <= -2 | STRONG SELL <= -5'
    story.append(Paragraph(f'<font color="{MUTED}" size="7">{scale_text}</font>', s_val))
    story.append(Spacer(1, 4))

    vol_clr = BULL if vol_signal == 'HIGH VOLUME' else BEAR if vol_signal == 'LOW VOLUME' else NEUTRAL_C
    vol_text = '; '.join(vol_notes) if vol_notes else 'Normal'
    vol_text_clr = WARN if vol_notes else NEUTRAL_C

    # Volume comparison
    current_vol = latest['Volume']
    avg_vol = latest['Vol_SMA_63'] if pd.notna(latest.get('Vol_SMA_63')) else latest['Vol_SMA_20']
    vol_ratio = (current_vol / avg_vol) if avg_vol > 0 else 0
    vol_pct = (vol_ratio - 1) * 100

    if vol_ratio >= 2.0:
        vol_desc = f'VERY HIGH - {vol_ratio:.1f}x avg ({vol_pct:+.0f}%)'
        vol_detail_clr = BULL
    elif vol_ratio >= 1.5:
        vol_desc = f'HIGH - {vol_ratio:.1f}x avg ({vol_pct:+.0f}%)'
        vol_detail_clr = BULL
    elif vol_ratio >= 0.8:
        vol_desc = f'NORMAL - {vol_ratio:.1f}x avg ({vol_pct:+.0f}%)'
        vol_detail_clr = NEUTRAL_C
    elif vol_ratio >= 0.5:
        vol_desc = f'LOW - {vol_ratio:.1f}x avg ({vol_pct:+.0f}%)'
        vol_detail_clr = BEAR
    else:
        vol_desc = f'VERY LOW - {vol_ratio:.1f}x avg ({vol_pct:+.0f}%)'
        vol_detail_clr = BEAR

    data = [
        [label_p('Trend'), val_style(f'{trend_sig} (Score: {trend_score})', signal_clr(trend_sig))],
        [label_p('Momentum'), val_style(f'{mom_sig} (Score: {mom_score})', signal_clr(mom_sig))],
        [label_p('Volume'), val_style(vol_desc, vol_detail_clr)],
        [label_p('  Current Vol'), val_style(f'{current_vol:,.0f}')],
        [label_p('  Avg Vol (3M)'), val_style(f'{avg_vol:,.0f}')],
        [label_p('Volatility'), val_style(vol_text, vol_text_clr)],
    ]
    story.append(make_table(data))

    # Trend Score Breakdown
    trend_details = signals.get('trend_details', [])
    if trend_details:
        trend_bd = [[label_p('Trend Breakdown'), val_style(f'Score: {trend_score} (Bullish >= 3 | Bearish <= -3)', signal_clr(trend_sig))]]
        for desc, pts, is_bull in trend_details:
            d_clr = BULL if is_bull else BEAR if is_bull is False else NEUTRAL_C
            trend_bd.append([label_p(f''), Paragraph(f'<font color="{d_clr}" size="7.5">{pts} &nbsp; {desc}</font>', s_val)])
        story.append(make_table(trend_bd))

    # Momentum Score Breakdown
    mom_details = signals.get('mom_details', [])
    if mom_details:
        mom_bd = [[label_p('Momentum Breakdown'), val_style(f'Score: {mom_score} (Bullish >= 2 | Bearish <= -2)', signal_clr(mom_sig))]]
        for desc, pts, is_bull in mom_details:
            d_clr = BULL if is_bull else BEAR if is_bull is False else NEUTRAL_C
            mom_bd.append([label_p(f''), Paragraph(f'<font color="{d_clr}" size="7.5">{pts} &nbsp; {desc}</font>', s_val)])
        story.append(make_table(mom_bd))

    # ── Multi-Timeframe Trend ──
    story.append(Paragraph('MULTI-TIMEFRAME TREND', s_section))
    mtf_data = []
    for tf_name in ['Daily', 'Weekly', 'Monthly']:
        tf = mtf_trends.get(tf_name, {})
        if tf.get('available', False):
            trend = tf['trend']
            trend_clr = signal_clr(trend.replace('UPTREND', 'BULLISH').replace('DOWNTREND', 'BEARISH').replace('STRONG ', 'STRONG ').replace('SIDEWAYS', 'NEUTRAL'))
            detail = tf.get('detail', '')
            sma_info = ''
            if tf.get('sma20') is not None and tf.get('sma50') is not None:
                sma_info = f'  (SMA20: ${tf["sma20"]:.2f} | SMA50: ${tf["sma50"]:.2f})'
            mtf_data.append([label_p(tf_name), val_style(trend, trend_clr)])
            mtf_data.append([label_p(f'  Detail'), Paragraph(f'<font color="{MUTED}" size="7.5">{detail}{sma_info}</font>', s_val)])
        else:
            detail = tf.get('detail', 'Data not available')
            mtf_data.append([label_p(tf_name), val_style('N/A', NEUTRAL_C)])
            mtf_data.append([label_p(f'  Note'), Paragraph(f'<font color="{MUTED}" size="7.5">{detail}</font>', s_val)])

    # Trend alignment check
    available_trends = [mtf_trends[t]['trend'] for t in ['Daily', 'Weekly', 'Monthly'] if mtf_trends.get(t, {}).get('available', False)]
    if len(available_trends) >= 2:
        all_up = all('UPTREND' in t for t in available_trends)
        all_down = all('DOWNTREND' in t for t in available_trends)
        if all_up:
            mtf_data.append([label_p('Alignment'), val_style('ALL TIMEFRAMES BULLISH - Strong confluence', BULL)])
        elif all_down:
            mtf_data.append([label_p('Alignment'), val_style('ALL TIMEFRAMES BEARISH - Strong confluence', BEAR)])
        else:
            mtf_data.append([label_p('Alignment'), val_style('MIXED - Timeframes not aligned, exercise caution', WARN)])

    story.append(make_table(mtf_data))

    story.append(Paragraph('TECHNICAL EVALUATION - MULTI-TIMEFRAME', s_section))

    def summary_clr(sig):
        if 'BUY' in sig: return BULL
        elif 'SELL' in sig: return BEAR
        return NEUTRAL_C

    # ── Multi-Timeframe Summary Table (Investing.com style) ──
    tf_order = ['5 Min', '15 Min', '30 Min', '4 Hours', 'Daily', 'Weekly', 'Monthly']

    # Header
    mtf_header_row = [label_p('')]
    for tf_label in tf_order:
        mtf_header_row.append(Paragraph(f'<font color="{WHITE}" size="7"><b>{tf_label}</b></font>', s_val))
    mtf_table_data = [mtf_header_row]

    # Summary row
    summary_row = [label_p('Summary')]
    for tf_label in tf_order:
        ev = mtf_eval.get(tf_label, {})
        if ev.get('available'):
            sig = ev['overall_summary']
            summary_row.append(Paragraph(f'<font color="{summary_clr(sig)}" size="7"><b>{sig}</b></font>', s_val))
        else:
            summary_row.append(Paragraph(f'<font color="{MUTED}" size="7">N/A</font>', s_val))
    mtf_table_data.append(summary_row)

    # MA row
    ma_row = [label_p('Moving Avg')]
    for tf_label in tf_order:
        ev = mtf_eval.get(tf_label, {})
        if ev.get('available'):
            sig = ev['ma_summary']
            ma_row.append(Paragraph(
                f'<font color="{summary_clr(sig)}" size="6.5"><b>{sig}</b></font><br/>'
                f'<font color="{MUTED}" size="6">B:{ev["ma_buy"]} N:{ev["ma_neutral"]} S:{ev["ma_sell"]}</font>', s_val))
        else:
            ma_row.append(Paragraph(f'<font color="{MUTED}" size="7">N/A</font>', s_val))
    mtf_table_data.append(ma_row)

    # Oscillators row
    osc_row = [label_p('Oscillators')]
    for tf_label in tf_order:
        ev = mtf_eval.get(tf_label, {})
        if ev.get('available'):
            sig = ev['osc_summary']
            osc_row.append(Paragraph(
                f'<font color="{summary_clr(sig)}" size="6.5"><b>{sig}</b></font><br/>'
                f'<font color="{MUTED}" size="6">B:{ev["osc_buy"]} N:{ev["osc_neutral"]} S:{ev["osc_sell"]}</font>', s_val))
        else:
            osc_row.append(Paragraph(f'<font color="{MUTED}" size="7">N/A</font>', s_val))
    mtf_table_data.append(osc_row)

    # Buy/Sell total row
    bns_row = [label_p('Total B/N/S')]
    for tf_label in tf_order:
        ev = mtf_eval.get(tf_label, {})
        if ev.get('available'):
            bns_row.append(Paragraph(
                f'<font color="{BULL}" size="6.5"><b>{ev["total_buy"]}</b></font>'
                f'<font color="{MUTED}" size="6.5"> / </font>'
                f'<font color="{NEUTRAL_C}" size="6.5"><b>{ev["total_neutral"]}</b></font>'
                f'<font color="{MUTED}" size="6.5"> / </font>'
                f'<font color="{BEAR}" size="6.5"><b>{ev["total_sell"]}</b></font>', s_val))
        else:
            bns_row.append(Paragraph(f'<font color="{MUTED}" size="7">N/A</font>', s_val))
    mtf_table_data.append(bns_row)

    col_w = (letter[0] - 1.4*inch) / 7  # evenly distribute across 7 timeframes
    mtf_t = Table(mtf_table_data, colWidths=[0.9*inch] + [col_w]*7, hAlign='LEFT')
    mtf_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_SECTION),
        ('TEXTCOLOR', (0, 0), (-1, -1), WHITE),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#333355')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2a2a4a')),
        ('BACKGROUND', (0, 0), (0, -1), HexColor('#2a2a4a')),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(mtf_t)
    story.append(Spacer(1, 6))

    # ── Barchart-Style Technical Opinion ──
    bco_eval = compute_external_technical_evaluation(df, info)
    story.append(Paragraph('TECHNICAL OPINION (Barchart-Style)', s_section))
    bco = bco_eval['barchart_opinion']
    bco_data = []
    for tf_key, tf_label in [('short_term', 'Short-Term'), ('medium_term', 'Medium-Term'), ('long_term', 'Long-Term')]:
        tf = bco[tf_key]
        bco_data.append([label_p(tf_label), val_style(f'{tf["signal"]} (Score: {tf["score"]})', summary_clr(tf['signal']))])

    bc_signals = [bco['short_term']['signal'], bco['medium_term']['signal'], bco['long_term']['signal']]
    bc_buy_count = sum(1 for s in bc_signals if 'BUY' in s)
    bc_sell_count = sum(1 for s in bc_signals if 'SELL' in s)
    if bc_buy_count >= 2:
        bc_overall = 'BUY' if bc_buy_count == 2 else 'STRONG BUY'
    elif bc_sell_count >= 2:
        bc_overall = 'SELL' if bc_sell_count == 2 else 'STRONG SELL'
    else:
        bc_overall = 'NEUTRAL'
    bco_data.insert(0, [label_p('Barchart Opinion'), val_style(bc_overall, summary_clr(bc_overall))])
    story.append(make_table(bco_data))


    # ── Key Technical Levels ──
    story.append(Paragraph('KEY TECHNICAL LEVELS', s_section))

    def level_row(label, value):
        """Create a row with price, Above/Below, and % difference."""
        diff_pct = ((current - value) / value) * 100
        above = current > value
        dir_text = 'Above' if above else 'Below'
        dir_clr = BULL if above else BEAR
        pct_clr = BULL if above else BEAR
        return [label_p(label), Paragraph(
            f'<font color="{WHITE}"><b>${value:.2f}</b></font>'
            f' &nbsp; <font color="{dir_clr}"><b>{dir_text}</b></font>'
            f' &nbsp; <font color="{pct_clr}"><b>({diff_pct:+.2f}%)</b></font>', s_val)]

    tech_data = [
        level_row('SMA 20', latest['SMA_20']),
        level_row('SMA 50', latest['SMA_50']),
    ]
    if pd.notna(latest['SMA_200']):
        tech_data.append(level_row('SMA 200', latest['SMA_200']))

    ema_bull = latest['EMA_9'] > latest['EMA_21']
    ema9_diff = ((current - latest['EMA_9']) / latest['EMA_9']) * 100
    ema21_diff = ((current - latest['EMA_21']) / latest['EMA_21']) * 100
    tech_data.append([label_p('EMA 9'), Paragraph(
        f'<font color="{WHITE}"><b>${latest["EMA_9"]:.2f}</b></font>'
        f' &nbsp; <font color="{BULL if current > latest["EMA_9"] else BEAR}"><b>{"Above" if current > latest["EMA_9"] else "Below"} ({ema9_diff:+.2f}%)</b></font>', s_val)])
    tech_data.append([label_p('EMA 21'), Paragraph(
        f'<font color="{WHITE}"><b>${latest["EMA_21"]:.2f}</b></font>'
        f' &nbsp; <font color="{BULL if current > latest["EMA_21"] else BEAR}"><b>{"Above" if current > latest["EMA_21"] else "Below"} ({ema21_diff:+.2f}%)</b></font>', s_val)])
    tech_data.append([label_p('EMA Cross'), val_style(f'EMA 9 {">" if ema_bull else "<"} EMA 21  -  {"Bullish" if ema_bull else "Bearish"}', BULL if ema_bull else BEAR)])

    bb_upper_diff = ((current - latest['BB_Upper']) / latest['BB_Upper']) * 100
    bb_lower_diff = ((current - latest['BB_Lower']) / latest['BB_Lower']) * 100
    tech_data.append([label_p('BB Upper'), Paragraph(
        f'<font color="{WHITE}"><b>${latest["BB_Upper"]:.2f}</b></font>'
        f' &nbsp; <font color="{BEAR}"><b>({bb_upper_diff:+.2f}%)</b></font>', s_val)])
    tech_data.append([label_p('BB Lower'), Paragraph(
        f'<font color="{WHITE}"><b>${latest["BB_Lower"]:.2f}</b></font>'
        f' &nbsp; <font color="{BULL}"><b>({bb_lower_diff:+.2f}%)</b></font>', s_val)])

    bb_pct = latest['BB_Pct']
    bb_clr = BEAR if bb_pct > 1 else BULL if bb_pct < 0 else NEUTRAL_C
    tech_data.append([label_p('BB %B'), val_style(f'{bb_pct:.2f}', bb_clr)])

    vwap_above = current > latest['VWAP']
    vwap_diff = ((current - latest['VWAP']) / latest['VWAP']) * 100
    tech_data.append([label_p('VWAP'), Paragraph(
        f'<font color="{WHITE}"><b>${latest["VWAP"]:.2f}</b></font>'
        f' &nbsp; <font color="{BULL if vwap_above else BEAR}"><b>{"Above" if vwap_above else "Below"} ({vwap_diff:+.2f}%)</b></font>', s_val)])
    story.append(make_table(tech_data))

    # ── Momentum Indicators ──
    story.append(Paragraph('MOMENTUM INDICATORS', s_section))
    rsi_val = latest['RSI']
    rsi_clr = BEAR if rsi_val > 70 else BULL if rsi_val < 30 else NEUTRAL_C
    rsi_note = '  OVERBOUGHT' if rsi_val > 70 else '  OVERSOLD' if rsi_val < 30 else ''

    rsi4_val = latest['RSI_4']
    rsi4_clr = BEAR if rsi4_val > 80 else BULL if rsi4_val < 20 else NEUTRAL_C
    rsi4_note = '  OVERBOUGHT' if rsi4_val > 80 else '  OVERSOLD' if rsi4_val < 20 else ''

    cci20_val = latest['CCI_20']
    cci20_clr = BEAR if cci20_val > 100 else BULL if cci20_val < -100 else NEUTRAL_C
    cci20_note = '  OVERBOUGHT' if cci20_val > 100 else '  OVERSOLD' if cci20_val < -100 else ''

    cci6_val = latest['CCI_6']
    cci6_clr = BEAR if cci6_val > 100 else BULL if cci6_val < -100 else NEUTRAL_C
    cci6_note = '  OVERBOUGHT' if cci6_val > 100 else '  OVERSOLD' if cci6_val < -100 else ''

    stoch_bull = latest['StochRSI_K'] > latest['StochRSI_D']
    macd_bull = latest['MACD'] > latest['MACD_Signal']
    macd_hist = latest['MACD_Hist']
    macd_expanding = abs(macd_hist) > abs(prev['MACD_Hist'])
    adx_val = latest['ADX']

    mom_data = [
        [label_p('CCI (20)'), val_style(f'{cci20_val:.1f}{cci20_note}', cci20_clr)],
        [label_p('CCI (6)'), val_style(f'{cci6_val:.1f}{cci6_note}', cci6_clr)],
        [label_p('StochRSI K/D'), val_style(f'{latest["StochRSI_K"]:.1f} / {latest["StochRSI_D"]:.1f}  -  {"Bullish" if stoch_bull else "Bearish"}', BULL if stoch_bull else BEAR)],
        [label_p('ADX'), val_style(f'{adx_val:.1f}  -  {"Strong Trend" if adx_val > 25 else "Weak/No Trend"}', BULL if adx_val > 25 else NEUTRAL_C)],
        [label_p('+DI / -DI'), val_style(f'{latest["DI_Plus"]:.1f} / {latest["DI_Minus"]:.1f}', BULL if latest['DI_Plus'] > latest['DI_Minus'] else BEAR)],
        [label_p('ATR (14)'), val_style(f'${latest["ATR"]:.2f} ({latest["ATR_Pct"]:.2f}%)')],
    ]
    story.append(make_table(mom_data))

    # ── Keltner Channel / Bollinger / Squeeze ──
    story.append(Paragraph('KELTNER CHANNEL / BOLLINGER / SQUEEZE', s_section))
    squeeze = latest['Squeeze']
    squeeze_tight = latest['Squeeze_Tight'] if pd.notna(latest.get('Squeeze_Tight')) else False
    bb_width_pctl = latest['BB_Width_Pctl'] if pd.notna(latest.get('BB_Width_Pctl')) else None

    kc_data = [
        [label_p('KC Width'), val_style(f'${latest["KC_Width"]:.2f}')],
        [label_p('BB Width'), val_style(f'{latest["BB_Width"]:.4f}')],
    ]

    if bb_width_pctl is not None:
        pctl_display = f'{bb_width_pctl*100:.0f}th percentile'
        pctl_clr = BULL if bb_width_pctl < 0.2 else NEUTRAL_C
        kc_data.append([label_p('BB Width Rank'), val_style(pctl_display, pctl_clr)])

    # Squeeze status with enhanced logic
    if squeeze_tight:
        kc_data.append([label_p('Squeeze Status'), val_style('VERY STRONG BULLISH - BB tight and inside KC (explosive breakout imminent)', BULL)])
    elif squeeze:
        kc_data.append([label_p('Squeeze Status'), val_style('BULLISH - BB inside KC (energy building, breakout expected)', BULL)])
    else:
        # Check if BB is close to fitting inside KC
        bb_margin_upper = latest['KC_Upper'] - latest['BB_Upper']
        bb_margin_lower = latest['BB_Lower'] - latest['KC_Lower']
        if bb_margin_upper > 0 or bb_margin_lower > 0:
            kc_data.append([label_p('Squeeze Status'), val_style('OFF - BB expanding outside KC (trend in motion)', NEUTRAL_C)])
        else:
            kc_data.append([label_p('Squeeze Status'), val_style('OFF - No squeeze', NEUTRAL_C)])

    # Price position within channels
    if current > latest['KC_Upper']:
        kc_data.append([label_p('Price vs KC'), val_style('ABOVE KC Upper - Strong bullish momentum', BULL)])
    elif current < latest['KC_Lower']:
        kc_data.append([label_p('Price vs KC'), val_style('BELOW KC Lower - Strong bearish momentum', BEAR)])
    elif current > latest['KC_Middle']:
        kc_data.append([label_p('Price vs KC'), val_style('Above KC Middle - Mild bullish', BULL)])
    else:
        kc_data.append([label_p('Price vs KC'), val_style('Below KC Middle - Mild bearish', BEAR)])

    story.append(make_table(kc_data))

    # ── MFI / OBV ──
    story.append(Paragraph('MFI / OBV', s_section))
    mfi_val = latest['MFI']
    mfi_clr = BEAR if mfi_val > 80 else BULL if mfi_val < 20 else NEUTRAL_C
    mfi_note = '  OVERBOUGHT' if mfi_val > 80 else '  OVERSOLD' if mfi_val < 20 else ''

    mfi6_val = latest['MFI_6']
    mfi6_clr = BEAR if mfi6_val > 80 else BULL if mfi6_val < 20 else NEUTRAL_C
    mfi6_note = '  OVERBOUGHT' if mfi6_val > 80 else '  OVERSOLD' if mfi6_val < 20 else ''

    obv_val = latest['OBV']
    obv_5_ago = df.iloc[-6]['OBV'] if len(df) > 6 else prev['OBV']
    obv_20_ago = df.iloc[-21]['OBV'] if len(df) > 21 else prev['OBV']
    obv_short = 'Rising' if obv_val > obv_5_ago else 'Falling'
    obv_long = 'Rising' if obv_val > obv_20_ago else 'Falling'
    price_trend = 'up' if current > df.iloc[-21]['Close'] else 'down'

    mfi_obv_data = []

    # MFI vs RSI - always show
    if (rsi_val > 70 and mfi_val < 50) or (rsi_val < 30 and mfi_val > 50):
        mfi_obv_data.append([label_p('MFI/RSI'), val_style(f'CONFLICT - RSI={rsi_val:.1f} vs MFI={mfi_val:.1f} (volume disagrees with price)', WARN)])
    elif (rsi_val > 60 and mfi_val > 60):
        mfi_obv_data.append([label_p('MFI/RSI'), val_style(f'CONFIRMED BULLISH - RSI={rsi_val:.1f}, MFI={mfi_val:.1f} (both elevated)', BULL)])
    elif (rsi_val < 40 and mfi_val < 40):
        mfi_obv_data.append([label_p('MFI/RSI'), val_style(f'CONFIRMED BEARISH - RSI={rsi_val:.1f}, MFI={mfi_val:.1f} (both depressed)', BEAR)])
    elif abs(rsi_val - mfi_val) > 20:
        mfi_obv_data.append([label_p('MFI/RSI'), val_style(f'DIVERGING - RSI={rsi_val:.1f} vs MFI={mfi_val:.1f} (gap: {abs(rsi_val - mfi_val):.1f}pts)', WARN)])
    else:
        mfi_obv_data.append([label_p('MFI/RSI'), val_style(f'NEUTRAL - RSI={rsi_val:.1f}, MFI={mfi_val:.1f} (aligned, no extremes)', NEUTRAL_C)])

    mfi_obv_data.append([label_p('OBV'), val_style(f'{obv_val:,.0f}')])
    mfi_obv_data.append([label_p('OBV Trend (5d)'), val_style(obv_short, BULL if obv_short == 'Rising' else BEAR)])
    mfi_obv_data.append([label_p('OBV Trend (20d)'), val_style(obv_long, BULL if obv_long == 'Rising' else BEAR)])

    if price_trend == 'up' and obv_long == 'Falling':
        mfi_obv_data.append([label_p('OBV/Price'), val_style('BEARISH WARNING - Price rising but OBV falling', BEAR)])
    elif price_trend == 'down' and obv_long == 'Rising':
        mfi_obv_data.append([label_p('OBV/Price'), val_style('BULLISH SIGNAL - Price falling but OBV rising (accumulation)', BULL)])
    elif price_trend == 'up' and obv_long == 'Rising':
        mfi_obv_data.append([label_p('OBV/Price'), val_style('CONFIRMED - Price and OBV both rising (healthy)', BULL)])
    elif price_trend == 'down' and obv_long == 'Falling':
        mfi_obv_data.append([label_p('OBV/Price'), val_style('CONFIRMED - Price and OBV both falling (distribution)', BEAR)])

    story.append(make_table(mfi_obv_data))

    # ── Support / Resistance ──
    story.append(Paragraph('SUPPORT &amp; RESISTANCE', s_section))
    sr_data = []
    for s in sr_levels.get('resistances', []):
        dist = ((s['price'] - current) / current) * 100
        sr_data.append([label_p('Resistance'), val_style(f'${s["price"]:.2f}  ({dist:+.1f}%)  [Strength: {s["strength"]}]', BEAR)])
    sr_data.append([
        Paragraph(f'<b><font color="{WHITE}">  > Current</font></b>', s_label),
        val_style(f'${current:.2f}', WHITE)
    ])
    for s in sr_levels.get('supports', []):
        dist = ((s['price'] - current) / current) * 100
        sr_data.append([label_p('Support'), val_style(f'${s["price"]:.2f}  ({dist:+.1f}%)  [Strength: {s["strength"]}]', BULL)])
    story.append(make_table(sr_data))

    # ── Fibonacci Analysis ──
    story.append(Paragraph('FIBONACCI ANALYSIS', s_section))
    if fib_result.get('available', False):
        active_tool = fib_result.get('active_tool', 'RETRACEMENT')
        trend = fib_result['primary_trend']
        trend_clr = BULL if 'UP' in trend else BEAR

        fib_data = [
            [label_p('Primary Trend'), val_style(trend, trend_clr)],
            [label_p('Swing High'), Paragraph(
                f'<font color="{WHITE}"><b>${fib_result["swing_high"]:.2f}</b></font>'
                f' &nbsp; <font color="{MUTED}">({fib_result["swing_high_date"]})</font>', s_val)],
            [label_p('Swing Low'), Paragraph(
                f'<font color="{WHITE}"><b>${fib_result["swing_low"]:.2f}</b></font>'
                f' &nbsp; <font color="{MUTED}">({fib_result["swing_low_date"]})</font>', s_val)],
            [label_p('Range'), val_style(f'${fib_result["price_range"]:.2f}')],
            [label_p('Active Tool'), val_style(f'Fibonacci {active_tool}', BULL if active_tool == 'EXTENSION' else WARN)],
        ]

        # Why this tool is active
        if fib_result.get('active_reason'):
            fib_data.append([label_p('Reason'), Paragraph(
                f'<font color="{MUTED}" size="7.5">{fib_result["active_reason"]}</font>', s_val)])

        # Current zone
        if fib_result.get('current_zone'):
            fib_data.append([label_p('Current Zone'), val_style(fib_result['current_zone'])])

        # Key support/resistance from Fib
        if fib_result.get('key_support'):
            ks = fib_result['key_support']
            fib_data.append([label_p('Fib Support'), Paragraph(
                f'<font color="{BULL}"><b>${ks["price"]:.2f}</b></font>'
                f' &nbsp; <font color="{BULL}"><b>({ks["label"]}) ({ks["dist_pct"]:+.2f}%)</b></font>', s_val)])
        if fib_result.get('key_resistance'):
            kr = fib_result['key_resistance']
            fib_data.append([label_p('Fib Resistance'), Paragraph(
                f'<font color="{BEAR}"><b>${kr["price"]:.2f}</b></font>'
                f' &nbsp; <font color="{BEAR}"><b>({kr["label"]}) ({kr["dist_pct"]:+.2f}%)</b></font>', s_val)])

        story.append(make_table(fib_data))

        # Active levels table
        active_levels = fib_result.get('active_levels', [])
        if active_levels:
            story.append(Spacer(1, 4))
            levels_header = f'Fibonacci {active_tool} Levels'
            level_data = [[label_p(levels_header), val_style('')]]
            for lv in active_levels:
                lv_above = current > lv['price']
                lv_clr = BULL if lv_above else BEAR
                marker = '  >>>' if abs(lv['dist_pct']) < 2 else ''
                level_data.append([label_p(f'  {lv["label"]}'), Paragraph(
                    f'<font color="{WHITE}"><b>${lv["price"]:.2f}</b></font>'
                    f' &nbsp; <font color="{lv_clr}"><b>({lv["dist_pct"]:+.2f}%)</b></font>'
                    f'<font color="{WARN}"><b>{marker}</b></font>', s_val)])
            story.append(make_table(level_data))
    else:
        reason = fib_result.get('reason', 'Data not available')
        fib_data = [[label_p('Status'), val_style(f'N/A - {reason}', NEUTRAL_C)]]
        story.append(make_table(fib_data))

    # ── Multi-Timeframe Fibonacci (Weekly + Monthly) ──
    if mtf_fib:
        for tf_name in ['Weekly', 'Monthly']:
            tf_fib = mtf_fib.get(tf_name, {})
            if tf_fib.get('available', False):
                story.append(Paragraph(f'FIBONACCI ANALYSIS — {tf_name.upper()}', s_section))
                tf_tool = tf_fib.get('active_tool', 'RETRACEMENT')
                tf_trend = tf_fib['primary_trend']
                tf_trend_clr = BULL if 'UP' in tf_trend else BEAR

                tf_data = [
                    [label_p('Primary Trend'), val_style(tf_trend, tf_trend_clr)],
                    [label_p('Swing High'), Paragraph(
                        f'<font color="{WHITE}"><b>${tf_fib["swing_high"]:.2f}</b></font>'
                        f' &nbsp; <font color="{MUTED}">({tf_fib["swing_high_date"]})</font>', s_val)],
                    [label_p('Swing Low'), Paragraph(
                        f'<font color="{WHITE}"><b>${tf_fib["swing_low"]:.2f}</b></font>'
                        f' &nbsp; <font color="{MUTED}">({tf_fib["swing_low_date"]})</font>', s_val)],
                    [label_p('Range'), val_style(f'${tf_fib["price_range"]:.2f}')],
                    [label_p('Active Tool'), val_style(f'Fibonacci {tf_tool}', BULL if tf_tool == 'EXTENSION' else WARN)],
                ]

                if tf_fib.get('active_reason'):
                    tf_data.append([label_p('Reason'), Paragraph(
                        f'<font color="{MUTED}" size="7.5">{tf_fib["active_reason"]}</font>', s_val)])

                if tf_fib.get('current_zone'):
                    tf_data.append([label_p('Current Zone'), val_style(tf_fib['current_zone'])])

                if tf_fib.get('key_support'):
                    ks = tf_fib['key_support']
                    tf_data.append([label_p('Fib Support'), Paragraph(
                        f'<font color="{BULL}"><b>${ks["price"]:.2f}</b></font>'
                        f' &nbsp; <font color="{BULL}"><b>({ks["label"]}) ({ks["dist_pct"]:+.2f}%)</b></font>', s_val)])
                if tf_fib.get('key_resistance'):
                    kr = tf_fib['key_resistance']
                    tf_data.append([label_p('Fib Resistance'), Paragraph(
                        f'<font color="{BEAR}"><b>${kr["price"]:.2f}</b></font>'
                        f' &nbsp; <font color="{BEAR}"><b>({kr["label"]}) ({kr["dist_pct"]:+.2f}%)</b></font>', s_val)])

                story.append(make_table(tf_data))

                # Levels table
                tf_levels = tf_fib.get('active_levels', [])
                if tf_levels:
                    story.append(Spacer(1, 4))
                    tf_levels_header = f'Fibonacci {tf_tool} Levels ({tf_name})'
                    tf_level_data = [[label_p(tf_levels_header), val_style('')]]
                    for lv in tf_levels:
                        lv_above = current > lv['price']
                        lv_clr = BULL if lv_above else BEAR
                        marker = '  >>>' if abs(lv['dist_pct']) < 2 else ''
                        tf_level_data.append([label_p(f'  {lv["label"]}'), Paragraph(
                            f'<font color="{WHITE}"><b>${lv["price"]:.2f}</b></font>'
                            f' &nbsp; <font color="{lv_clr}"><b>({lv["dist_pct"]:+.2f}%)</b></font>'
                            f'<font color="{WARN}"><b>{marker}</b></font>', s_val)])
                    story.append(make_table(tf_level_data))

    # ── Chart Patterns ──
    story.append(Paragraph('CHART PATTERNS', s_section))
    if chart_patterns:
        for i, pat in enumerate(chart_patterns, 1):
            bias = pat['bias'].upper()
            pat_clr = BULL if bias == 'BULLISH' else BEAR if bias == 'BEARISH' else NEUTRAL_C
            type_clr = WARN if pat['type'] == 'Reversal' else BLUE

            pat_data = [
                [label_p(f'[{i}] Pattern'), val_style(pat['name'], pat_clr)],
                [label_p('    Bias'), val_style(bias, pat_clr)],
                [label_p('    Type'), val_style(pat['type'], type_clr)],
                [label_p('    Period'), val_style(pat['period'])],
                [label_p('    Detail'), Paragraph(f'<font color="{WHITE}" size="8">{pat["detail"]}</font>', s_val)],
                [label_p('    Target'), val_style(pat.get('target', 'N/A'), pat_clr)],
            ]
            story.append(make_table(pat_data))
            story.append(Spacer(1, 4))
    else:
        no_pat = [[label_p('Status'), val_style('No fully formed chart patterns detected in recent price action.', NEUTRAL_C)]]
        story.append(make_table(no_pat))

    # ── Chart Patterns (theEccentricTrader Method) ──
    story.append(Paragraph('CHART PATTERNS — theEccentricTrader Method', s_section))
    story.append(Paragraph(
        '<font color="#888888" size="7"><i>Swing-based pattern detection adapted from '
        'theEccentricTrader / Ozan Kaplanbasoglu (Pine Script v5). '
        'Uses pivot swing highs/lows with retracement ratios and tolerance matching.</i></font>', s_val))
    story.append(Spacer(1, 4))

    if eccentric_patterns:
        for i, pat in enumerate(eccentric_patterns, 1):
            bias = pat['bias'].upper()
            pat_clr = BULL if bias == 'BULLISH' else BEAR if bias == 'BEARISH' else NEUTRAL_C
            type_clr = WARN if pat['type'] == 'Reversal' else BLUE

            ep_data = [
                [label_p(f'[{i}] Pattern'), val_style(pat['name'], pat_clr)],
                [label_p('    Bias'), val_style(bias, pat_clr)],
                [label_p('    Type'), val_style(pat['type'], type_clr)],
                [label_p('    Detail'), Paragraph(f'<font color="{WHITE}" size="8">{pat["description"]}</font>', s_val)],
            ]
            if pat.get('neckline'):
                ep_data.append([label_p('    Neckline'), val_style(pat['neckline'], WHITE)])
            if pat.get('target'):
                ep_data.append([label_p('    Target'), val_style(pat['target'], pat_clr)])
            story.append(make_table(ep_data))
            story.append(Spacer(1, 4))
    else:
        no_ep = [[label_p('Status'), val_style('No patterns detected via theEccentricTrader swing method.', NEUTRAL_C)]]
        story.append(make_table(no_ep))

    # ── Divergences ──
    story.append(Paragraph('DIVERGENCE ANALYSIS', s_section))
    if divergences:
        for i, div in enumerate(divergences, 1):
            div_type = div['type'].upper()
            is_bull = 'BULLISH' in div_type
            is_bear = 'BEARISH' in div_type
            div_clr = BULL if is_bull else BEAR if is_bear else WARN
            strength_clr = BEAR if div['strength'] == 'STRONG' else WARN

            div_data = [
                [label_p(f'[{i}] Type'), val_style(div['type'], div_clr)],
                [label_p('    Pair'), val_style(div['pair'])],
                [label_p('    Strength'), val_style(div['strength'], strength_clr)],
                [label_p('    Period'), val_style(f'{div["date_start"]}  ->  {div["date_end"]}')],
                [label_p('    Detail'), Paragraph(f'<font color="{WHITE}" size="8">{div["description"]}</font>', s_val)],
                [label_p('    Signal'), val_style(div['signal'], div_clr)],
            ]
            story.append(make_table(div_data))
            story.append(Spacer(1, 4))
    else:
        no_div = [[label_p('Status'), val_style('No significant divergences detected. RSI, MACD, MFI, and OBV are generally aligned.', NEUTRAL_C)]]
        story.append(make_table(no_div))

    # ── Ichimoku Cloud ──
    story.append(Paragraph('ICHIMOKU CLOUD', s_section))
    tenkan = latest['Ichi_Tenkan']
    kijun = latest['Ichi_Kijun']
    span_a = latest['Ichi_SpanA']
    span_b = latest['Ichi_SpanB']

    ichi_data = []
    if pd.notna(tenkan):
        tenkan_pct = ((current - tenkan) / tenkan) * 100
        ichi_data.append([label_p('Tenkan-sen (9)'), Paragraph(
            f'<font color="{WHITE}"><b>${tenkan:.2f}</b></font>'
            f' &nbsp; <font color="{BULL if current > tenkan else BEAR}"><b>{"Above" if current > tenkan else "Below"} ({tenkan_pct:+.2f}%)</b></font>', s_val)])
    if pd.notna(kijun):
        kijun_pct = ((current - kijun) / kijun) * 100
        ichi_data.append([label_p('Kijun-sen (26)'), Paragraph(
            f'<font color="{WHITE}"><b>${kijun:.2f}</b></font>'
            f' &nbsp; <font color="{BULL if current > kijun else BEAR}"><b>{"Above" if current > kijun else "Below"} ({kijun_pct:+.2f}%)</b></font>', s_val)])
    if pd.notna(span_a):
        span_a_pct = ((current - span_a) / span_a) * 100
        ichi_data.append([label_p('Senkou Span A'), Paragraph(
            f'<font color="{WHITE}"><b>${span_a:.2f}</b></font>'
            f' &nbsp; <font color="{BULL if current > span_a else BEAR}"><b>({span_a_pct:+.2f}%)</b></font>', s_val)])
    if pd.notna(span_b):
        span_b_pct = ((current - span_b) / span_b) * 100
        ichi_data.append([label_p('Senkou Span B'), Paragraph(
            f'<font color="{WHITE}"><b>${span_b:.2f}</b></font>'
            f' &nbsp; <font color="{BULL if current > span_b else BEAR}"><b>({span_b_pct:+.2f}%)</b></font>', s_val)])

    if pd.notna(span_a) and pd.notna(span_b):
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        cloud_bull = span_a > span_b
        ichi_data.append([label_p('Cloud'), val_style('GREEN (bullish)' if cloud_bull else 'RED (bearish)', BULL if cloud_bull else BEAR)])
        if current > cloud_top:
            cloud_pct = ((current - cloud_top) / cloud_top) * 100
            ichi_data.append([label_p('Price vs Cloud'), val_style(f'ABOVE cloud (+{cloud_pct:.2f}%) - Bullish', BULL)])
        elif current < cloud_bottom:
            cloud_pct = ((cloud_bottom - current) / cloud_bottom) * 100
            ichi_data.append([label_p('Price vs Cloud'), val_style(f'BELOW cloud (-{cloud_pct:.2f}%) - Bearish', BEAR)])
        else:
            ichi_data.append([label_p('Price vs Cloud'), val_style('INSIDE cloud - Neutral / Transitioning', NEUTRAL_C)])

    if pd.notna(tenkan) and pd.notna(kijun):
        prev_tenkan = prev['Ichi_Tenkan'] if pd.notna(prev['Ichi_Tenkan']) else tenkan
        prev_kijun = prev['Ichi_Kijun'] if pd.notna(prev['Ichi_Kijun']) else kijun
        if prev_tenkan <= prev_kijun and tenkan > kijun:
            ichi_data.append([label_p('TK Cross'), val_style('BULLISH CROSS (Tenkan crossed above Kijun)', BULL)])
        elif prev_tenkan >= prev_kijun and tenkan < kijun:
            ichi_data.append([label_p('TK Cross'), val_style('BEARISH CROSS (Tenkan crossed below Kijun)', BEAR)])
        elif tenkan > kijun:
            ichi_data.append([label_p('TK Cross'), val_style('Tenkan above Kijun (Bullish bias)', BULL)])
        else:
            ichi_data.append([label_p('TK Cross'), val_style('Tenkan below Kijun (Bearish bias)', BEAR)])

    ichi_score = 0
    if pd.notna(tenkan) and current > tenkan: ichi_score += 1
    if pd.notna(kijun) and current > kijun: ichi_score += 1
    if pd.notna(span_a) and pd.notna(span_b) and current > max(span_a, span_b): ichi_score += 1
    if pd.notna(tenkan) and pd.notna(kijun) and tenkan > kijun: ichi_score += 1
    if pd.notna(span_a) and pd.notna(span_b) and span_a > span_b: ichi_score += 1
    ichi_sig = "STRONG BULLISH" if ichi_score >= 4 else "BULLISH" if ichi_score >= 3 else "BEARISH" if ichi_score <= 1 else "NEUTRAL"
    ichi_data.append([label_p('Ichimoku Signal'), val_style(f'{ichi_sig} ({ichi_score}/5)', signal_clr(ichi_sig))])

    if ichi_data:
        story.append(make_table(ichi_data))

    # ── Daily Detail: Overall Summary ──
    ext_eval = compute_external_technical_evaluation(df, info)
    story.append(Paragraph('DAILY DETAIL', s_section))
    ov = ext_eval['overall_summary']
    ov_data = [
        [label_p('Overall Summary'), val_style(ov, summary_clr(ov))],
        [label_p('  Buy / Neutral / Sell'), Paragraph(
            f'<font color="{BULL}"><b>{ext_eval["total_buy"]}</b></font>'
            f' &nbsp; / &nbsp; <font color="{NEUTRAL_C}"><b>{ext_eval["total_neutral"]}</b></font>'
            f' &nbsp; / &nbsp; <font color="{BEAR}"><b>{ext_eval["total_sell"]}</b></font>', s_val)],
    ]
    story.append(make_table(ov_data))

    # ── Moving Averages Summary ──
    story.append(Paragraph('MOVING AVERAGES (Daily)', s_section))
    ma_sum = ext_eval['ma_summary']
    ma_header = [
        [label_p('MA Summary'), val_style(ma_sum, summary_clr(ma_sum))],
        [label_p('  Buy / Neutral / Sell'), Paragraph(
            f'<font color="{BULL}"><b>{ext_eval["ma_buy"]}</b></font>'
            f' &nbsp; / &nbsp; <font color="{NEUTRAL_C}"><b>{ext_eval["ma_neutral"]}</b></font>'
            f' &nbsp; / &nbsp; <font color="{BEAR}"><b>{ext_eval["ma_sell"]}</b></font>', s_val)],
    ]
    for ma in ext_eval['ma_results']:
        pct = ((current - ma['value']) / ma['value']) * 100
        s_clr = BULL if ma['signal'] == 'Buy' else BEAR if ma['signal'] == 'Sell' else NEUTRAL_C
        ma_header.append([label_p(f'  {ma["name"]}'), Paragraph(
            f'<font color="{WHITE}"><b>${ma["value"]:.2f}</b></font>'
            f' &nbsp; <font color="{s_clr}"><b>{ma["signal"]} ({pct:+.2f}%)</b></font>', s_val)])
    story.append(make_table(ma_header))

    # ── Oscillators Summary ──
    story.append(Paragraph('OSCILLATORS (Daily)', s_section))
    osc_sum = ext_eval['osc_summary']
    osc_header = [
        [label_p('Oscillator Summary'), val_style(osc_sum, summary_clr(osc_sum))],
        [label_p('  Buy / Neutral / Sell'), Paragraph(
            f'<font color="{BULL}"><b>{ext_eval["osc_buy"]}</b></font>'
            f' &nbsp; / &nbsp; <font color="{NEUTRAL_C}"><b>{ext_eval["osc_neutral"]}</b></font>'
            f' &nbsp; / &nbsp; <font color="{BEAR}"><b>{ext_eval["osc_sell"]}</b></font>', s_val)],
    ]
    for osc in ext_eval['osc_results']:
        s_clr = BULL if osc['signal'] == 'Buy' else BEAR if osc['signal'] == 'Sell' else NEUTRAL_C
        osc_header.append([label_p(f'  {osc["name"]}'), Paragraph(
            f'<font color="{WHITE}"><b>{osc["value"]}</b></font>'
            f' &nbsp; <font color="{s_clr}"><b>{osc["signal"]}</b></font>', s_val)])
    story.append(make_table(osc_header))

    # ── Pivot Points ──
    story.append(Paragraph('PIVOT POINTS', s_section))
    pivots = ext_eval['pivots']
    piv_header = [[
        label_p('Level'),
        Paragraph(f'<font color="{WHITE}"><b>Classic</b></font>', s_val),
        Paragraph(f'<font color="{WHITE}"><b>Fibonacci</b></font>', s_val),
        Paragraph(f'<font color="{WHITE}"><b>Woodie</b></font>', s_val),
        Paragraph(f'<font color="{WHITE}"><b>Camarilla</b></font>', s_val),
    ]]
    for level in ['R3', 'R2', 'R1', 'PP', 'S1', 'S2', 'S3']:
        row_clr = BEAR if 'R' in level else BULL if 'S' in level else WHITE
        row = [label_p(level)]
        for ptype in ['Classic', 'Fibonacci', 'Woodie', 'Camarilla']:
            val = pivots[ptype][level]
            pct = ((current - val) / val) * 100
            p_clr = BULL if current > val else BEAR
            row.append(Paragraph(f'<font color="{WHITE}" size="7.5">${val:.2f} <font color="{p_clr}">({pct:+.1f}%)</font></font>', s_val))
        piv_header.append(row)

    piv_t = Table(piv_header, colWidths=[0.8*inch, 1.45*inch, 1.45*inch, 1.45*inch, 1.45*inch], hAlign='LEFT')
    piv_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_SECTION),
        ('TEXTCOLOR', (0, 0), (-1, -1), WHITE),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#333355')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2a2a4a')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    story.append(piv_t)


    # ── Fundamentals ──
    story.append(Paragraph('FUNDAMENTAL OVERVIEW', s_section))
    fund_data = [
        [label_p('Market Cap'), val_style(format_large_number(fundamentals['Market Cap']))],
        [label_p('P/E (Trailing)'), val_style(format_ratio(fundamentals['P/E (Trailing)']))],
        [label_p('P/E (Forward)'), val_style(format_ratio(fundamentals['P/E (Forward)']))],
        [label_p('PEG Ratio'), val_style(format_ratio(fundamentals['PEG Ratio']))],
        [label_p('P/S Ratio'), val_style(format_ratio(fundamentals['P/S Ratio']))],
        [label_p('P/B Ratio'), val_style(format_ratio(fundamentals['P/B Ratio']))],
        [label_p('EV/EBITDA'), val_style(format_ratio(fundamentals['EV/EBITDA']))],
        [label_p('Revenue'), val_style(format_large_number(fundamentals['Revenue']))],
        [label_p('Revenue Growth'), val_style(format_pct(fundamentals['Revenue Growth']),
            BULL if fundamentals['Revenue Growth'] != 'N/A' and fundamentals['Revenue Growth'] is not None and float(fundamentals['Revenue Growth']) > 0 else BEAR if fundamentals['Revenue Growth'] != 'N/A' and fundamentals['Revenue Growth'] is not None and float(fundamentals['Revenue Growth']) < 0 else NEUTRAL_C)],
        [label_p('Profit Margin'), val_style(format_pct(fundamentals['Profit Margin']))],
        [label_p('Operating Margin'), val_style(format_pct(fundamentals['Operating Margin']))],
        [label_p('ROE'), val_style(format_pct(fundamentals['ROE']))],
        [label_p('ROA'), val_style(format_pct(fundamentals['ROA']))],
        [label_p('Debt/Equity'), val_style(format_ratio(fundamentals['Debt/Equity']))],
        [label_p('Current Ratio'), val_style(format_ratio(fundamentals['Current Ratio']))],
        [label_p('Free Cash Flow'), val_style(format_large_number(fundamentals['Free Cash Flow']))],
        [label_p('Dividend Yield'), val_style(format_pct(fundamentals['Dividend Yield']))],
        [label_p('Beta'), val_style(format_ratio(fundamentals['Beta']))],
    ]
    story.append(make_table(fund_data))

    # ══════════════════════════════════════════════════════════════
    # EXTERNAL TECHNICAL EVALUATION (TradingView / Barchart Style)
    # ══════════════════════════════════════════════════════════════
    ext_eval = compute_external_technical_evaluation(df, info)


    # ── Disclaimer ──
    story.append(Paragraph('DISCLAIMER: This analysis is for informational purposes only. '
                           'It does not constitute financial advice. Always do your own '
                           'due diligence before making investment decisions.', s_disclaimer))

    # ── Build PDF ──
    def on_page(canvas_obj, doc_obj):
        canvas_obj.saveState()
        canvas_obj.setFillColor(BG_DARK)
        canvas_obj.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
        canvas_obj.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)

    # Also generate a plain text summary for terminal output
    # Build MTF summary for terminal
    mtf_summary_parts = []
    for tf in ['Daily', 'Weekly', 'Monthly']:
        t = mtf_trends.get(tf, {})
        mtf_summary_parts.append(f'{tf}: {t.get("trend", "N/A")}')
    mtf_line = ' | '.join(mtf_summary_parts)


    report_text = f"""
{'='*70}
  STOCK ANALYSIS REPORT: {ticker.upper()}
  {fundamentals['Company']}
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*70}
  * OVERALL SIGNAL: {overall} (Score: {score})
  Price: ${current:.2f} ({chg:+.2f} / {chg_pct:+.2f}%)
  Trend: {trend_sig} | Momentum: {mom_sig} | Volume: {vol_signal}
  Multi-TF: {mtf_line}
  RSI: {rsi_val:.1f} | MFI: {mfi_val:.1f} | ADX: {adx_val:.1f}
  Ichimoku: {ichi_sig} ({ichi_score}/5)
  Divergences: {len(divergences)} found
  Chart Patterns: {len(chart_patterns)} detected{' - ' + ', '.join(p['name'] + ' (' + p['bias'] + ')' for p in chart_patterns) if chart_patterns else ''}
{'='*70}
  Full color report saved as PDF.
"""

    return report_path, report_text


def analyze_stock(ticker: str, period: str = '1y', output_dir: str = None):
    """Main analysis function for a single stock."""
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[*] Fetching data for {ticker.upper()}...")
    df, info, stock = fetch_data(ticker, period)

    print(f"[*] Computing technical indicators...")
    df = compute_indicators(df)

    print(f"[*] Detecting support & resistance levels...")
    sr_levels = detect_support_resistance(df)

    print(f"[*] Generating trading signals...")
    signals = generate_signals(df)

    print(f"[*] Detecting divergences...")
    divergences = detect_divergences(df)

    print(f"[*] Detecting chart patterns...")
    chart_patterns = detect_chart_patterns(df)

    print(f"[*] Detecting chart patterns (theEccentricTrader method)...")
    eccentric_patterns = detect_eccentric_patterns(df)

    print(f"[*] Analyzing Fibonacci levels...")
    fib_result = analyze_fibonacci(df)

    print(f"[*] Analyzing multi-timeframe Fibonacci (Weekly + Monthly)...")
    mtf_fib = compute_multi_timeframe_fibonacci(ticker, daily_df=df)

    print(f"[*] Screening Halal compliance...")
    halal_result = screen_halal_compliance(info)

    print(f"[*] Analyzing multi-timeframe trends...")
    mtf_trends = analyze_multi_timeframe_trend(stock)

    print(f"[*] Creating chart...")
    chart_path = plot_chart(df, ticker, signals, sr_levels, divergences, output_dir)

    print(f"[*] Creating Fibonacci chart...")
    fib_chart_path = plot_fibonacci_chart(df, ticker, fib_result, output_dir)

    # Create weekly/monthly Fibonacci charts
    fib_chart_weekly = None
    fib_chart_monthly = None
    weekly_fib = mtf_fib.get('Weekly', {})
    monthly_fib = mtf_fib.get('Monthly', {})
    if weekly_fib.get('available') and weekly_fib.get('_df') is not None:
        print(f"[*] Creating Weekly Fibonacci chart...")
        fib_chart_weekly = plot_fibonacci_chart_tf(weekly_fib['_df'], ticker, weekly_fib, 'Weekly', output_dir)
    if monthly_fib.get('available') and monthly_fib.get('_df') is not None:
        print(f"[*] Creating Monthly Fibonacci chart...")
        fib_chart_monthly = plot_fibonacci_chart_tf(monthly_fib['_df'], ticker, monthly_fib, 'Monthly', output_dir)

    print(f"[*] Computing multi-timeframe technical evaluation...")
    mtf_eval = compute_multi_timeframe_evaluation(ticker, daily_df=df)

    print(f"[*] Generating report...")
    report_path, report_text = generate_report(ticker, df, info, signals, sr_levels, divergences, mtf_trends, chart_patterns, eccentric_patterns, halal_result, fib_result, mtf_fib, mtf_eval, chart_path, fib_chart_path, fib_chart_weekly, fib_chart_monthly, output_dir)

    # Clean up standalone chart images since they're now embedded in the PDF
    if os.path.exists(chart_path) and os.path.exists(report_path):
        os.remove(chart_path)
    if fib_chart_path and os.path.exists(fib_chart_path) and os.path.exists(report_path):
        os.remove(fib_chart_path)
    if fib_chart_weekly and os.path.exists(fib_chart_weekly) and os.path.exists(report_path):
        os.remove(fib_chart_weekly)
    if fib_chart_monthly and os.path.exists(fib_chart_monthly) and os.path.exists(report_path):
        os.remove(fib_chart_monthly)

    # Print report safely for Windows terminals
    try:
        print(report_text)
    except UnicodeEncodeError:
        print(report_text.encode('ascii', errors='replace').decode('ascii'))

    print(f"\n[OK] Report saved: {report_path}")

    return {
        'report': report_path,
        'signals': signals,
        'sr_levels': sr_levels,
        'latest_data': df.iloc[-1].to_dict(),
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python stock_analyzer.py TICKER [--period PERIOD]")
        print("  e.g. python stock_analyzer.py AAPL")
        print("       python stock_analyzer.py MSFT --period 1y")
        sys.exit(1)

    tickers = []
    period = '1y'

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--period' and i + 1 < len(args):
            period = args[i + 1]
            i += 2
        else:
            tickers.append(args[i])
            i += 1

    for t in tickers:
        try:
            analyze_stock(t, period=period)
        except Exception as e:
            print(f"❌ Error analyzing {t}: {e}")

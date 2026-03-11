"""
Stock Analyzer Pro — Professional Web Report
==============================================
Bloomberg-inspired dark terminal aesthetic with tabbed layout.
All analysis sections preserved, organized into logical tabs.
"""
import streamlit as st
import os, sys, io, tempfile, contextlib
from datetime import datetime
import pandas as pd
import numpy as np

if "AV_API_KEY" in st.secrets:
    os.environ["AV_API_KEY"] = st.secrets["AV_API_KEY"]

st.set_page_config(page_title="Stock Analyzer Pro", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# ═══════════════════════════════════════════════════════════════════════════════
# BLOOMBERG TERMINAL-INSPIRED CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-card: #111827;
    --bg-card-alt: #0f1520;
    --bg-hover: #1a2332;
    --border: #1e293b;
    --border-accent: #22c55e;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --green: #22c55e;
    --green-dim: rgba(34,197,94,0.15);
    --red: #ef4444;
    --red-dim: rgba(239,68,68,0.15);
    --amber: #f59e0b;
    --amber-dim: rgba(245,158,11,0.15);
    --blue: #3b82f6;
    --blue-dim: rgba(59,130,246,0.15);
}

.stApp { background-color: var(--bg-primary); }
section[data-testid="stSidebar"] { background-color: var(--bg-card); border-right: 1px solid var(--border); }

/* Header */
.pro-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid var(--border);
    border-bottom: 2px solid var(--green);
    padding: 1.2rem 1.8rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.pro-header .ticker-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.5px;
}
.pro-header .company-info {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-secondary);
    font-size: 0.85rem;
}
.pro-header .price-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    margin-left: auto;
}

/* Section headers */
.sec-h {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--green);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0.5rem 0;
    margin: 1rem 0 0.6rem 0;
    border-bottom: 1px solid var(--border);
}

/* Data cards */
.data-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
}
.data-card .dc-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.2rem;
}
.data-card .dc-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* Signal badges */
.sig-buy { background: var(--green-dim); color: var(--green); border: 1px solid var(--green); padding: 0.2rem 0.6rem; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 600; display: inline-block; }
.sig-sell { background: var(--red-dim); color: var(--red); border: 1px solid var(--red); padding: 0.2rem 0.6rem; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 600; display: inline-block; }
.sig-neutral { background: var(--amber-dim); color: var(--amber); border: 1px solid var(--amber); padding: 0.2rem 0.6rem; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 600; display: inline-block; }
.sig-info { background: var(--blue-dim); color: var(--blue); border: 1px solid var(--blue); padding: 0.2rem 0.6rem; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 600; display: inline-block; }

/* Halal badges */
.halal-pass { background: var(--green-dim); color: var(--green); border: 1px solid var(--green); padding: 0.3rem 1rem; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.85rem; }
.halal-fail { background: var(--red-dim); color: var(--red); border: 1px solid var(--red); padding: 0.3rem 1rem; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.85rem; }
.halal-warn { background: var(--amber-dim); color: var(--amber); border: 1px solid var(--amber); padding: 0.3rem 1rem; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.85rem; }

/* Metric rows */
.metric-row {
    font-family: 'JetBrains Mono', monospace;
    display: flex;
    justify-content: space-between;
    padding: 0.35rem 0;
    border-bottom: 1px solid rgba(30,41,59,0.5);
    font-size: 0.82rem;
}
.metric-row .mr-label { color: var(--text-secondary); }
.metric-row .mr-value { color: var(--text-primary); font-weight: 500; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    padding: 0.7rem 1.2rem;
    color: var(--text-muted);
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] { color: var(--green) !important; border-bottom-color: var(--green) !important; }

/* Dataframe styling */
.stDataFrame { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def sec(title):
    st.markdown(f'<div class="sec-h">{title}</div>', unsafe_allow_html=True)

def card(label, value):
    st.markdown(f'<div class="data-card"><div class="dc-label">{label}</div><div class="dc-value">{value}</div></div>', unsafe_allow_html=True)

def sig(text):
    t = str(text).upper()
    if any(w in t for w in ["BUY","BULLISH","UPTREND","PASS","HALAL","STRONG BUY"]): return f'<span class="sig-buy">{t}</span>'
    elif any(w in t for w in ["SELL","BEARISH","DOWNTREND","FAIL","NON-COMPLIANT"]): return f'<span class="sig-sell">{t}</span>'
    return f'<span class="sig-neutral">{t}</span>'

def mrow(label, value):
    st.markdown(f'<div class="metric-row"><span class="mr-label">{label}</span><span class="mr-value">{value}</span></div>', unsafe_allow_html=True)

def fmt(val, prefix="", suffix="", d=2):
    if val is None or val == "N/A": return "N/A"
    try:
        v = float(val)
        if np.isnan(v): return "N/A"
        if abs(v) >= 1e12: return f"{prefix}{v/1e12:.{d}f}T{suffix}"
        if abs(v) >= 1e9: return f"{prefix}{v/1e9:.{d}f}B{suffix}"
        if abs(v) >= 1e6: return f"{prefix}{v/1e6:.{d}f}M{suffix}"
        return f"{prefix}{v:,.{d}f}{suffix}"
    except: return str(val)

def fmt_pct(val):
    if val is None or val == "N/A": return "N/A"
    try:
        v = float(val)
        return f"{v*100:.2f}%" if abs(v) < 1 else f"{v:.2f}%"
    except: return str(val)

def safe(d, k, fb=0):
    v = d.get(k, fb)
    if v is None or (isinstance(v, float) and np.isnan(v)): return fb
    return v

def level_str(name, val, current):
    if val is None or (isinstance(val, float) and np.isnan(val)): return f"{name}: N/A"
    pct = ((current - val)/val)*100 if val != 0 else 0
    pos = "Above" if current > val else "Below"
    return f"{name}: ${val:,.2f} — {pos} ({pct:+.2f}%)"

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    ticker_input = st.text_input("Ticker(s)", value="AAPL", placeholder="AAPL, MSFT, NVDA")
    period = st.selectbox("Period", options=["3mo","6mo","1y","2y","5y"], index=2)
    st.markdown("---")
    st.caption("Stock Analyzer Pro v2.0")
    st.caption("Data: yfinance via Google Colab")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
run = st.button("▶ RUN ANALYSIS", use_container_width=True, type="primary")

if not run:
    st.markdown("""
    <div class="pro-header">
        <div>
            <div class="ticker-name">STOCK ANALYZER PRO</div>
            <div class="company-info">Comprehensive Technical & Fundamental Analysis Terminal</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.info("Enter ticker(s) and click **RUN ANALYSIS**")

if run:
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    if not tickers: st.error("Enter at least one ticker."); st.stop()

    sys.path.insert(0, os.path.dirname(__file__))
    from stock_analyzer import analyze_stock_web

    for ticker in tickers:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_buf = io.StringIO()
            prog = st.progress(0, text=f"Analyzing {ticker}...")
            try:
                with contextlib.redirect_stdout(log_buf):
                    R = analyze_stock_web(ticker, period=period, output_dir=tmpdir)
                prog.progress(100, text=f"✅ {ticker} complete")
            except Exception as e:
                prog.progress(100, text=f"❌ {ticker} failed")
                st.error(str(e))
                with st.expander("Debug Log"): st.code(log_buf.getvalue())
                continue

            df=R['df']; info=R['info']; signals=R['signals']; L=R['latest_data']
            sr=R['sr_levels']; divs=R['divergences']; cp=R['chart_patterns']; ep=R['eccentric_patterns']
            fib=R['fib_result']; mtf_fib=R['mtf_fib']; halal=R['halal_result']
            mtf_t=R['mtf_trends']; mtf_e=R['mtf_eval']; fund=R['fundamentals']; bco=R['barchart_opinion']
            current = safe(L,'Close')
            overall = signals.get('overall','NEUTRAL')

            # ═══ HEADER ═══
            company = info.get('longName', info.get('shortName', ticker))
            sector = info.get('sector', ''); industry = info.get('industry', '')
            pct_chg = ((current - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100) if len(df) >= 2 else 0
            chg_color = "var(--green)" if pct_chg >= 0 else "var(--red)"
            st.markdown(f"""
            <div class="pro-header">
                <div>
                    <div class="ticker-name">{ticker} <span style="font-size:0.9rem;color:var(--text-muted);font-weight:400;">{company}</span></div>
                    <div class="company-info">{sector} {'·' if sector and industry else ''} {industry}</div>
                </div>
                <div class="price-tag" style="color:{chg_color}">
                    ${current:,.2f} <span style="font-size:0.9rem;">({pct_chg:+.2f}%)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ═══ QUICK STATS ROW ═══
            q1,q2,q3,q4,q5,q6 = st.columns(6)
            q1.markdown(f'<div class="data-card"><div class="dc-label">Signal</div><div class="dc-value">{sig(overall)}</div></div>', unsafe_allow_html=True)
            q2.markdown(f'<div class="data-card"><div class="dc-label">Score</div><div class="dc-value">{signals.get("total_score",0)}</div></div>', unsafe_allow_html=True)
            q3.markdown(f'<div class="data-card"><div class="dc-label">RSI</div><div class="dc-value">{safe(L,"RSI"):.1f}</div></div>', unsafe_allow_html=True)
            q4.markdown(f'<div class="data-card"><div class="dc-label">Volume</div><div class="dc-value">{fmt(safe(L,"Volume"))}</div></div>', unsafe_allow_html=True)
            rating = str(info.get('recommendationKey','N/A')).replace('_',' ').title()
            q5.markdown(f'<div class="data-card"><div class="dc-label">Analyst</div><div class="dc-value">{rating}</div></div>', unsafe_allow_html=True)
            h_status = halal.get('status','?')
            h_cls = "halal-pass" if h_status=="HALAL" else "halal-fail" if h_status=="NON-COMPLIANT" else "halal-warn"
            q6.markdown(f'<div class="data-card"><div class="dc-label">Halal</div><div class="dc-value"><span class="{h_cls}">{h_status}</span></div></div>', unsafe_allow_html=True)

            with st.expander("📜 Analysis Log"): st.code(log_buf.getvalue())

            # ═══════════════════════════════════════════════════════════════
            # TABS
            # ═══════════════════════════════════════════════════════════════
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "⚡ Technical", "📐 Patterns & Levels", "🔄 Multi-Timeframe", "💰 Fundamentals"])

            # ─────────────────────────────────────────────────────────────
            # TAB 1: OVERVIEW
            # ─────────────────────────────────────────────────────────────
            with tab1:
                # Charts
                for pk in ['chart_path','fib_chart_path','fib_chart_weekly','fib_chart_monthly']:
                    if R.get(pk) and os.path.exists(R[pk]):
                        st.image(R[pk], use_container_width=True)

                sec("PRICE SUMMARY")
                p1,p2,p3,p4,p5,p6 = st.columns(6)
                for col, lbl, val in [(p1,"Close",f"${current:,.2f}"),(p2,"Volume",fmt(safe(L,'Volume'))),(p3,"RSI",f"{safe(L,'RSI'):.1f}"),(p4,"MACD",f"{safe(L,'MACD'):.4f}"),(p5,"52W High",fmt(info.get('fiftyTwoWeekHigh'),prefix="$")),(p6,"52W Low",fmt(info.get('fiftyTwoWeekLow'),prefix="$"))]:
                    col.markdown(f'<div class="data-card"><div class="dc-label">{lbl}</div><div class="dc-value">{val}</div></div>', unsafe_allow_html=True)
                avg_vol = safe(L,'Vol_SMA_20',0)
                vol_ratio = safe(L,'Volume',0)/avg_vol if avg_vol>0 else 0
                st.caption(f"Volume: {fmt(safe(L,'Volume'))} | Avg(20d): {fmt(avg_vol)} | Ratio: {vol_ratio:.1f}x")

                sec("SIGNAL SUMMARY")
                trend_sig=signals.get('trend','NEUTRAL'); trend_sc=signals.get('trend_score',0)
                mom_sig=signals.get('momentum','NEUTRAL'); mom_sc=signals.get('mom_score',0)
                score=signals.get('total_score',0)
                st.markdown(f"{sig(overall)} &nbsp;&nbsp; Confluence Score: **{score}** &nbsp;(Trend {trend_sc} + Momentum {mom_sc})", unsafe_allow_html=True)
                st.caption("STRONG BUY ≥5 | BUY ≥2 | NEUTRAL -1 to 1 | SELL ≤-2 | STRONG SELL ≤-5")
                s1,s2,s3 = st.columns(3)
                s1.markdown(f"**Trend:** {sig(trend_sig)} Score: {trend_sc}", unsafe_allow_html=True)
                s2.markdown(f"**Momentum:** {sig(mom_sig)} Score: {mom_sc}", unsafe_allow_html=True)
                s3.markdown(f"**Volume:** {signals.get('volume','NORMAL')}")
                for label, key in [("Trend Breakdown","trend_details"),("Momentum Breakdown","mom_details")]:
                    details = signals.get(key, [])
                    if details:
                        with st.expander(label):
                            for desc, pts, is_bull in details:
                                icon = "🟢" if is_bull else "🔴" if is_bull is False else "⚪"
                                st.markdown(f"{icon} **{pts}** {desc}")
                vol_notes = signals.get('volatility', [])
                if vol_notes: st.markdown("**Volatility:** " + " | ".join(vol_notes))

                sec("ANALYST CONSENSUS")
                a1,a2,a3,a4 = st.columns(4)
                a1.markdown(f'<div class="data-card"><div class="dc-label">Rating</div><div class="dc-value">{rating}</div></div>', unsafe_allow_html=True)
                a2.markdown(f'<div class="data-card"><div class="dc-label">Target</div><div class="dc-value">{fmt(info.get("targetMeanPrice"),prefix="$")}</div></div>', unsafe_allow_html=True)
                a3.markdown(f'<div class="data-card"><div class="dc-label">Analysts</div><div class="dc-value">{info.get("numberOfAnalystOpinions","N/A")}</div></div>', unsafe_allow_html=True)
                tp=info.get('targetMeanPrice')
                try: upside = f"{((float(tp)-current)/current)*100:+.1f}%"
                except: upside = "N/A"
                a4.markdown(f'<div class="data-card"><div class="dc-label">Upside</div><div class="dc-value">{upside}</div></div>', unsafe_allow_html=True)
                bd = info.get('_analyst_breakdown')
                if bd: st.markdown(f"🟢 Strong Buy: **{bd.get('strongBuy',0)}** · Buy: **{bd.get('buy',0)}** · ⚪ Hold: **{bd.get('hold',0)}** · 🔴 Sell: **{bd.get('sell',0)}** · Strong Sell: **{bd.get('strongSell',0)}**")

            # ─────────────────────────────────────────────────────────────
            # TAB 2: TECHNICAL
            # ─────────────────────────────────────────────────────────────
            with tab2:
                if bco and bco.get('barchart_opinion'):
                    sec("BARCHART-STYLE TECHNICAL OPINION")
                    bo = bco['barchart_opinion']
                    bc_sigs = [bo['short_term']['signal'], bo['medium_term']['signal'], bo['long_term']['signal']]
                    bc_buy=sum(1 for s in bc_sigs if 'BUY' in s); bc_sell=sum(1 for s in bc_sigs if 'SELL' in s)
                    bc_ov = 'BUY' if bc_buy>bc_sell else 'SELL' if bc_sell>bc_buy else 'NEUTRAL'
                    st.markdown(f"**Overall:** {sig(bc_ov)}", unsafe_allow_html=True)
                    b1,b2,b3 = st.columns(3)
                    b1.markdown(f"**Short:** {sig(bo['short_term']['signal'])} ({bo['short_term']['score']})", unsafe_allow_html=True)
                    b2.markdown(f"**Medium:** {sig(bo['medium_term']['signal'])} ({bo['medium_term']['score']})", unsafe_allow_html=True)
                    b3.markdown(f"**Long:** {sig(bo['long_term']['signal'])} ({bo['long_term']['score']})", unsafe_allow_html=True)

                sec("KEY TECHNICAL LEVELS")
                k1,k2 = st.columns(2)
                with k1:
                    for k in ['SMA_20','SMA_50','SMA_200','EMA_9','EMA_21']:
                        mrow(k.replace('_',' '), level_str(k, safe(L,k,None), current).split(': ',1)[-1])
                    cross = "EMA 9 > EMA 21 ✅ Bullish" if safe(L,'EMA_9',0)>safe(L,'EMA_21',0) else "EMA 9 < EMA 21 ❌ Bearish"
                    mrow("EMA Cross", cross)
                with k2:
                    mrow("BB Upper", level_str("BB", safe(L,'BB_Upper',None), current).split(': ',1)[-1])
                    mrow("BB Lower", level_str("BB", safe(L,'BB_Lower',None), current).split(': ',1)[-1])
                    mrow("BB %B", f"{safe(L,'BB_PctB',0):.2f}")
                    mrow("VWAP", level_str("VWAP", safe(L,'VWAP',None), current).split(': ',1)[-1])

                sec("MOMENTUM INDICATORS")
                m1,m2,m3,m4 = st.columns(4)
                for col,lbl,val in [(m1,"RSI (14)",f"{safe(L,'RSI'):.1f}"),(m2,"CCI (20)",f"{safe(L,'CCI'):.1f}"),(m3,"StochRSI K/D",f"{safe(L,'StochRSI_K'):.1f} / {safe(L,'StochRSI_D'):.1f}"),(m4,"MFI",f"{safe(L,'MFI'):.1f}")]:
                    col.markdown(f'<div class="data-card"><div class="dc-label">{lbl}</div><div class="dc-value">{val}</div></div>', unsafe_allow_html=True)
                m5,m6,m7,m8 = st.columns(4)
                atr_pct = f"({safe(L,'ATR',0)/current*100:.1f}%)" if current>0 else ""
                for col,lbl,val in [(m5,"ADX",f"{safe(L,'ADX'):.1f}"),(m6,"+DI / -DI",f"{safe(L,'Plus_DI'):.1f} / {safe(L,'Minus_DI'):.1f}"),(m7,"ATR (14)",f"${safe(L,'ATR'):.2f} {atr_pct}"),(m8,"MACD",f"{safe(L,'MACD'):.4f}")]:
                    col.markdown(f'<div class="data-card"><div class="dc-label">{lbl}</div><div class="dc-value">{val}</div></div>', unsafe_allow_html=True)

                sec("KELTNER / BOLLINGER / SQUEEZE")
                sq = "🔴 SQUEEZE ACTIVE" if L.get('Squeeze') else "🟢 No Squeeze"
                st.markdown(f"**Squeeze:** {sq}")
                kb1,kb2,kb3 = st.columns(3)
                kb1.markdown(f"**KC Width:** ${safe(L,'KC_Width'):.2f}")
                kb2.markdown(f"**BB Width:** {safe(L,'BB_Width'):.4f}")
                kc_u=safe(L,'KC_Upper',0); kc_l=safe(L,'KC_Lower',0); kc_m=safe(L,'KC_Middle',0)
                kc_pos = "ABOVE Upper ↑" if current>kc_u else "BELOW Lower ↓" if current<kc_l else "Above Mid ↗" if current>kc_m else "Below Mid ↘"
                kb3.markdown(f"**Price vs KC:** {kc_pos}")

                sec("MFI / OBV")
                rsi_v=safe(L,'RSI'); mfi_v=safe(L,'MFI')
                if rsi_v>70 and mfi_v>80: mfr="OVERBOUGHT"
                elif rsi_v<30 and mfi_v<20: mfr="OVERSOLD"
                elif abs(rsi_v-mfi_v)>20: mfr="DIVERGENCE"
                else: mfr="NEUTRAL"
                mrow("MFI/RSI Status", f"{mfr} (RSI={rsi_v:.1f}, MFI={mfi_v:.1f})")
                obv=safe(L,'OBV',0)
                mrow("OBV", fmt(obv))
                if len(df)>=20 and 'OBV' in df.columns:
                    mrow("OBV Trend (5d)", "Rising" if obv>df['OBV'].iloc[-5] else "Falling")
                    mrow("OBV Trend (20d)", "Rising" if obv>df['OBV'].iloc[-20] else "Falling")

                sec("ICHIMOKU CLOUD")
                tenkan=safe(L,'Ichimoku_Tenkan',None); kijun=safe(L,'Ichimoku_Kijun',None)
                spanA=safe(L,'Ichimoku_SpanA',None); spanB=safe(L,'Ichimoku_SpanB',None)
                i1,i2 = st.columns(2)
                with i1:
                    if tenkan: mrow("Tenkan-sen (9)", f"${tenkan:,.2f}")
                    if kijun: mrow("Kijun-sen (26)", f"${kijun:,.2f}")
                    if spanA is not None: mrow("Senkou Span A", f"${spanA:,.2f}")
                    if spanB is not None: mrow("Senkou Span B", f"${spanB:,.2f}")
                with i2:
                    if spanA is not None and spanB is not None:
                        mrow("Cloud", "GREEN ✅" if spanA>spanB else "RED ❌")
                        if current>max(spanA,spanB): mrow("Price vs Cloud", "ABOVE ↑ Bullish")
                        elif current<min(spanA,spanB): mrow("Price vs Cloud", "BELOW ↓ Bearish")
                        else: mrow("Price vs Cloud", "INSIDE ↔ Transitioning")
                    if tenkan and kijun:
                        mrow("TK Cross", "Bullish ✅" if tenkan>kijun else "Bearish ❌")
                        ichi_sc=sum([tenkan>kijun, current>tenkan, current>kijun, spanA is not None and spanB is not None and current>max(spanA,spanB), spanA is not None and spanB is not None and spanA>spanB])
                        ichi_s="BULLISH" if ichi_sc>=3 else "BEARISH" if ichi_sc<=1 else "NEUTRAL"
                        st.markdown(f"**Signal:** {sig(ichi_s)} ({ichi_sc}/5)", unsafe_allow_html=True)

            # ─────────────────────────────────────────────────────────────
            # TAB 3: PATTERNS & LEVELS
            # ─────────────────────────────────────────────────────────────
            with tab3:
                sec("SUPPORT & RESISTANCE")
                sc1,sc2 = st.columns(2)
                with sc2:
                    st.markdown("**Resistance**")
                    for lvl in sr.get('resistance', []):
                        p=lvl.get('price',lvl.get('level',lvl)) if isinstance(lvl,dict) else lvl
                        s_str=f" [Str: {lvl.get('strength',1)}]" if isinstance(lvl,dict) else ""
                        st.markdown(f"🔴 ${p:,.2f} ({((p-current)/current)*100:+.1f}%){s_str}")
                with sc1:
                    st.markdown(f"**▸ Current: ${current:,.2f}**")
                    st.markdown("**Support**")
                    for lvl in sr.get('support', []):
                        p=lvl.get('price',lvl.get('level',lvl)) if isinstance(lvl,dict) else lvl
                        s_str=f" [Str: {lvl.get('strength',1)}]" if isinstance(lvl,dict) else ""
                        st.markdown(f"🟢 ${p:,.2f} ({((p-current)/current)*100:+.1f}%){s_str}")

                sec("FIBONACCI ANALYSIS")
                if fib.get('available'):
                    f1,f2 = st.columns([2,1])
                    with f1:
                        mrow("Trend", fib.get('primary_trend','N/A'))
                        mrow("Swing High", f"${fib.get('swing_high',0):,.2f}")
                        mrow("Swing Low", f"${fib.get('swing_low',0):,.2f}")
                        mrow("Range", f"${fib.get('swing_high',0)-fib.get('swing_low',0):,.2f}")
                        mrow("Active Tool", f"Fibonacci {fib.get('active_tool','RETRACEMENT').upper()}")
                        if fib.get('current_zone'): mrow("Current Zone", fib['current_zone'])
                    with f2:
                        if fib.get('reason'): st.caption(fib['reason'])
                    levels = fib.get('levels', {})
                    if levels:
                        rows = []
                        for k,v in levels.items():
                            pct = ((v-current)/current)*100
                            marker = " ◄" if abs(pct) < 2 else ""
                            rows.append({"Level":k,"Price":f"${v:,.2f}","Distance":f"{pct:+.1f}%{marker}"})
                        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                else:
                    st.info(f"Fibonacci: {fib.get('reason','No data')}")
                for tf in ['Weekly','Monthly']:
                    tf_fib = mtf_fib.get(tf, {})
                    if tf_fib.get('available'):
                        with st.expander(f"Fibonacci — {tf}"):
                            mrow("Trend", tf_fib.get('primary_trend','N/A'))
                            levels = tf_fib.get('levels', {})
                            if levels:
                                rows = [{"Level":k,"Price":f"${v:,.2f}","Dist":f"{((v-current)/current)*100:+.1f}%"} for k,v in levels.items()]
                                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

                sec("CHART PATTERNS")
                if cp:
                    for p in cp:
                        icon = "📈" if p.get('direction')=='BULLISH' else "📉" if p.get('direction')=='BEARISH' else "📊"
                        st.markdown(f"{icon} **{p.get('pattern','?')}** — {sig(p.get('direction','N/A'))} — Confidence: {p.get('confidence','N/A')}", unsafe_allow_html=True)
                        if p.get('description'): st.caption(p['description'])
                else:
                    st.info("No chart patterns detected in current data.")
                if ep:
                    with st.expander("theEccentricTrader Method"):
                        for p in ep:
                            st.markdown(f"**{p.get('pattern','?')}** — {p.get('direction','N/A')}")
                else:
                    with st.expander("theEccentricTrader Method"):
                        st.info("No patterns detected.")

                sec("DIVERGENCE ANALYSIS")
                if divs:
                    for d in divs:
                        icon = "🟢" if d.get('type')=='BULLISH' else "🔴"
                        st.markdown(f"{icon} **{d.get('type','?')} {d.get('indicator','?')}** — {d.get('description','')}")
                else:
                    st.info("No divergences detected.")

            # ─────────────────────────────────────────────────────────────
            # TAB 4: MULTI-TIMEFRAME
            # ─────────────────────────────────────────────────────────────
            with tab4:
                sec("MULTI-TIMEFRAME TREND")
                for tf_name in ['Daily','Weekly','Monthly']:
                    tf = mtf_t.get(tf_name, {})
                    if tf.get('available', False):
                        trend = tf.get('trend','N/A')
                        mapped = trend.replace('UPTREND','BULLISH').replace('DOWNTREND','BEARISH').replace('SIDEWAYS','NEUTRAL')
                        st.markdown(f"**{tf_name}:** {sig(mapped)} — {tf.get('detail','')}", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{tf_name}:** N/A — {tf.get('detail', tf.get('reason',''))}")

                if mtf_e:
                    sec("TECHNICAL EVALUATION — MULTI-TIMEFRAME")
                    eval_rows = []
                    for tf_label in ['5 Min','15 Min','30 Min','4 Hours','Daily','Weekly','Monthly']:
                        ev = mtf_e.get(tf_label, {})
                        if ev.get('available'):
                            eval_rows.append({"TF":tf_label,"Overall":ev.get('overall_summary','N/A'),"MA":ev.get('ma_summary','N/A'),"Osc":ev.get('osc_summary','N/A'),"Buy":ev.get('total_buy',0),"Neut":ev.get('total_neutral',0),"Sell":ev.get('total_sell',0)})
                        else:
                            eval_rows.append({"TF":tf_label,"Overall":"—","MA":"—","Osc":"—","Buy":"—","Neut":"—","Sell":"—"})
                    st.dataframe(pd.DataFrame(eval_rows), hide_index=True, use_container_width=True)

                if bco and bco.get('overall_summary'):
                    sec("DAILY DETAIL")
                    st.markdown(f"**Overall:** {sig(bco['overall_summary'])} — Buy: {bco.get('total_buy',0)} / Neutral: {bco.get('total_neutral',0)} / Sell: {bco.get('total_sell',0)}", unsafe_allow_html=True)

                    sec("MOVING AVERAGES (Daily)")
                    st.markdown(f"**Summary:** {sig(bco.get('ma_summary','N/A'))} — Buy: {bco.get('ma_buy',0)} / Neutral: {bco.get('ma_neutral',0)} / Sell: {bco.get('ma_sell',0)}", unsafe_allow_html=True)
                    ma_r = bco.get('ma_results', [])
                    if ma_r:
                        ma_rows = [{"Indicator":m.get('name',''),"Value":f"${m.get('value',0):,.2f}" if m.get('value') else "N/A","Signal":m.get('signal','N/A'),"Dist":f"{m.get('pct_diff',0):+.2f}%"} for m in ma_r]
                        st.dataframe(pd.DataFrame(ma_rows), hide_index=True, use_container_width=True)

                    sec("OSCILLATORS (Daily)")
                    st.markdown(f"**Summary:** {sig(bco.get('osc_summary','N/A'))} — Buy: {bco.get('osc_buy',0)} / Neutral: {bco.get('osc_neutral',0)} / Sell: {bco.get('osc_sell',0)}", unsafe_allow_html=True)
                    osc_r = bco.get('osc_results', [])
                    if osc_r:
                        osc_rows = []
                        for o in osc_r:
                            try: ov = f"{float(o.get('value',0)):.2f}"
                            except (ValueError, TypeError): ov = str(o.get('value','N/A'))
                            osc_rows.append({"Indicator":o.get('name',''),"Value":ov,"Signal":o.get('signal','N/A')})
                        st.dataframe(pd.DataFrame(osc_rows), hide_index=True, use_container_width=True)

                if bco and bco.get('pivots'):
                    sec("PIVOT POINTS")
                    pivots = bco['pivots']
                    piv_rows = []
                    for level in ['R3','R2','R1','PP','S1','S2','S3']:
                        row = {"Level":level}
                        for method in ['Classic','Fibonacci','Woodie','Camarilla']:
                            v = pivots.get(method, {}).get(level, 0)
                            pct = ((v-current)/current)*100 if current>0 else 0
                            row[method] = f"${v:,.2f} ({pct:+.1f}%)"
                        piv_rows.append(row)
                    st.dataframe(pd.DataFrame(piv_rows), hide_index=True, use_container_width=True)

            # ─────────────────────────────────────────────────────────────
            # TAB 5: FUNDAMENTALS
            # ─────────────────────────────────────────────────────────────
            with tab5:
                sec("SHARIAH (HALAL) COMPLIANCE")
                h_cls2 = "halal-pass" if h_status=="HALAL" else "halal-fail" if h_status=="NON-COMPLIANT" else "halal-warn"
                st.markdown(f'<span class="{h_cls2}">{h_status}</span> &nbsp; ({halal.get("pass_count",0)}/{halal.get("total_checks",4)} criteria passed)', unsafe_allow_html=True)
                for d in halal.get('details', []):
                    icon = "✅" if d['status']=='PASS' else "❌" if d['status']=='FAIL' else "⚠️"
                    st.markdown(f"{icon} **{d['criterion']}** — {d['status']} — {d['value']}")
                    if d.get('note'): st.caption(d['note'])

                sec("FUNDAMENTAL OVERVIEW")
                fc1,fc2 = st.columns(2)
                with fc1:
                    for k in ['Company','Sector','Industry','Market Cap','P/E (Trailing)','P/E (Forward)','PEG Ratio','P/S Ratio','P/B Ratio','EV/EBITDA']:
                        v = fund.get(k, 'N/A')
                        if k=='Market Cap': v=fmt(v,prefix="$")
                        elif isinstance(v,float) and not np.isnan(v): v=f"{v:.2f}"
                        mrow(k, v)
                with fc2:
                    for k in ['Revenue','Revenue Growth','Profit Margin','Operating Margin','ROE','ROA','Debt/Equity','Current Ratio','Free Cash Flow','Dividend Yield','Beta','Target Price','Analyst Rating','Num Analysts']:
                        v = fund.get(k, 'N/A')
                        if k in ('Revenue','Free Cash Flow'): v=fmt(v,prefix="$")
                        elif k in ('52W High','52W Low','Target Price'): v=fmt(v,prefix="$")
                        elif k in ('Revenue Growth','Profit Margin','Operating Margin','ROE','ROA','Dividend Yield'): v=fmt_pct(v)
                        elif isinstance(v,float) and not np.isnan(v): v=f"{v:.2f}"
                        mrow(k, v)

            st.markdown("---")
            st.caption(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} · Data: yfinance · DISCLAIMER: For informational purposes only. Not financial advice.")

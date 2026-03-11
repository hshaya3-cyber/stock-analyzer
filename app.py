"""
Stock Analyzer Pro — Professional Terminal Report
===================================================
Bloomberg/TradingView-inspired dark terminal aesthetic.
Tabbed layout with all 22 analysis sections preserved.
"""
import streamlit as st
import os, sys, io, tempfile, contextlib
from datetime import datetime
import pandas as pd
import numpy as np

if "AV_API_KEY" in st.secrets:
    os.environ["AV_API_KEY"] = st.secrets["AV_API_KEY"]

st.set_page_config(page_title="Stock Analyzer Pro", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

# ═══════════════════════════════════════════════════════════════════════════════
# CSS — DARK TERMINAL AESTHETIC
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');
:root{--bg:#0a0e17;--card:#111827;--card2:#0d1117;--border:#1e293b;--accent:#22c55e;--red:#ef4444;--amber:#f59e0b;--blue:#3b82f6;--t1:#e2e8f0;--t2:#94a3b8;--t3:#64748b;--gdim:rgba(34,197,94,.12);--rdim:rgba(239,68,68,.12);--adim:rgba(245,158,11,.12);--bdim:rgba(59,130,246,.12)}
.stApp{background:var(--bg)}

/* Hide sidebar completely */
section[data-testid="stSidebar"]{display:none}
button[kind="header"]{display:none}
[data-testid="collapsedControl"]{display:none}

/* Mobile responsive */
@media(max-width:768px){
    .hdr{padding:.8rem 1rem;flex-direction:column;align-items:flex-start;gap:.5rem}
    .hdr .tk{font-size:1.2rem}
    .hdr .pr{margin-left:0;font-size:1.1rem}
    .dc{padding:.5rem .6rem}
    .dc .v{font-size:.85rem}
    .stTabs [data-baseweb="tab"]{padding:.5rem .6rem;font-size:.6rem}
    .mr{font-size:.72rem}
}

/* Header bar */
.hdr{background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid var(--border);border-left:3px solid var(--accent);padding:1rem 1.5rem;border-radius:6px;margin-bottom:.8rem;display:flex;align-items:center;gap:1rem;flex-wrap:wrap}
.hdr .tk{font-family:'JetBrains Mono',monospace;font-size:1.6rem;font-weight:700;color:var(--t1)}
.hdr .co{font-family:'DM Sans',sans-serif;color:var(--t3);font-size:.8rem}
.hdr .pr{font-family:'JetBrains Mono',monospace;font-size:1.4rem;font-weight:600;margin-left:auto}
.hdr .period-tag{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:var(--accent);background:var(--gdim);border:1px solid var(--accent);padding:.15rem .5rem;border-radius:3px;margin-left:.5rem}

/* Section headers */
.sh{font-family:'JetBrains Mono',monospace;font-size:.7rem;font-weight:600;color:var(--accent);letter-spacing:1.5px;text-transform:uppercase;padding:.4rem 0;margin:.8rem 0 .5rem;border-bottom:1px solid var(--border)}

/* Cards */
.dc{background:var(--card);border:1px solid var(--border);border-radius:5px;padding:.6rem .8rem;margin-bottom:.4rem}
.dc .l{font-family:'JetBrains Mono',monospace;font-size:.6rem;color:var(--t3);text-transform:uppercase;letter-spacing:.8px;margin-bottom:.15rem}
.dc .v{font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:600;color:var(--t1)}

/* Signals */
.sb{padding:.15rem .5rem;border-radius:3px;font-family:'JetBrains Mono',monospace;font-size:.7rem;font-weight:600;display:inline-block}
.sb-b{background:var(--gdim);color:var(--accent);border:1px solid var(--accent)}
.sb-s{background:var(--rdim);color:var(--red);border:1px solid var(--red)}
.sb-n{background:var(--adim);color:var(--amber);border:1px solid var(--amber)}
.sb-i{background:var(--bdim);color:var(--blue);border:1px solid var(--blue)}

/* Halal */
.h-pass{background:var(--gdim);color:var(--accent);border:1px solid var(--accent);padding:.2rem .7rem;border-radius:3px;font-family:'JetBrains Mono',monospace;font-weight:700;font-size:.8rem}
.h-fail{background:var(--rdim);color:var(--red);border:1px solid var(--red);padding:.2rem .7rem;border-radius:3px;font-family:'JetBrains Mono',monospace;font-weight:700;font-size:.8rem}
.h-warn{background:var(--adim);color:var(--amber);border:1px solid var(--amber);padding:.2rem .7rem;border-radius:3px;font-family:'JetBrains Mono',monospace;font-weight:700;font-size:.8rem}

/* Metric rows */
.mr{font-family:'JetBrains Mono',monospace;display:flex;justify-content:space-between;padding:.3rem 0;border-bottom:1px solid rgba(30,41,59,.4);font-size:.78rem}
.mr .ml{color:var(--t2)}.mr .mv{color:var(--t1);font-weight:500}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{gap:0;border-bottom:1px solid var(--border)}
.stTabs [data-baseweb="tab"]{font-family:'JetBrains Mono',monospace;font-size:.7rem;letter-spacing:.5px;text-transform:uppercase;padding:.6rem 1rem;color:var(--t3);border-bottom:2px solid transparent}
.stTabs [aria-selected="true"]{color:var(--accent)!important;border-bottom-color:var(--accent)!important}

/* Dataframes */
.stDataFrame{font-family:'JetBrains Mono',monospace;font-size:.75rem}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def sh(t): st.markdown(f'<div class="sh">{t}</div>', unsafe_allow_html=True)
def dc(l,v): st.markdown(f'<div class="dc"><div class="l">{l}</div><div class="v">{v}</div></div>', unsafe_allow_html=True)

def sg(t):
    t=str(t).upper()
    if any(w in t for w in ["BUY","BULLISH","UPTREND","PASS","HALAL","STRONG BUY","STRONG BULLISH"]): return f'<span class="sb sb-b">{t}</span>'
    elif any(w in t for w in ["SELL","BEARISH","DOWNTREND","FAIL","NON-COMPLIANT"]): return f'<span class="sb sb-s">{t}</span>'
    return f'<span class="sb sb-n">{t}</span>'

def mr(l,v): st.markdown(f'<div class="mr"><span class="ml">{l}</span><span class="mv">{v}</span></div>', unsafe_allow_html=True)

def fmt(v,p="",s="",d=2):
    if v is None or v=="N/A": return "N/A"
    try:
        f=float(v)
        if np.isnan(f): return "N/A"
        if abs(f)>=1e12: return f"{p}{f/1e12:.{d}f}T{s}"
        if abs(f)>=1e9: return f"{p}{f/1e9:.{d}f}B{s}"
        if abs(f)>=1e6: return f"{p}{f/1e6:.{d}f}M{s}"
        return f"{p}{f:,.{d}f}{s}"
    except: return str(v)

def fp(v):
    if v is None or v=="N/A": return "N/A"
    try:
        f=float(v)
        return f"{f*100:.2f}%" if abs(f)<1 else f"{f:.2f}%"
    except: return str(v)

def sf(d,k,fb=0):
    v=d.get(k,fb)
    if v is None or (isinstance(v,float) and np.isnan(v)): return fb
    return v

def lvl(name,val,cur):
    if val is None or (isinstance(val,float) and np.isnan(val)): return "N/A"
    pct=((cur-val)/val)*100 if val!=0 else 0
    pos="Above" if cur>val else "Below"
    return f"${val:,.2f} — {pos} ({pct:+.2f}%)"

# ═══════════════════════════════════════════════════════════════════════════════
# INPUTS — Main page (mobile-friendly, no sidebar)
# ═══════════════════════════════════════════════════════════════════════════════
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False

# Landing / Input section
if not st.session_state.analysis_running:
    st.markdown('''<div class="hdr">
        <div><div class="tk">STOCK ANALYZER PRO</div>
        <div class="co">Comprehensive Technical & Fundamental Analysis Terminal</div></div>
    </div>''', unsafe_allow_html=True)

    # Input row — compact for mobile
    c1, c2 = st.columns([3, 1])
    with c1:
        ticker_input = st.text_input("Ticker(s)", value="AAPL", placeholder="AAPL, MSFT, NVDA", label_visibility="collapsed")
    with c2:
        period = st.selectbox("Period", options=["3mo","6mo","1y","2y","5y"], index=2, label_visibility="collapsed")

    run = st.button("▶  RUN ANALYSIS", use_container_width=True, type="primary")

    if not run:
        st.markdown("")
        # Feature cards
        f1, f2, f3 = st.columns(3)
        f1.markdown(f'''<div class="dc" style="text-align:center;padding:1rem">
            <div class="l">📊 Technical</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:.8rem;color:var(--t2);margin-top:.3rem">20+ indicators · Patterns · Fibonacci · Ichimoku</div>
        </div>''', unsafe_allow_html=True)
        f2.markdown(f'''<div class="dc" style="text-align:center;padding:1rem">
            <div class="l">🔄 Multi-Timeframe</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:.8rem;color:var(--t2);margin-top:.3rem">Daily · Weekly · Monthly · 7 intervals</div>
        </div>''', unsafe_allow_html=True)
        f3.markdown(f'''<div class="dc" style="text-align:center;padding:1rem">
            <div class="l">💰 Fundamentals</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:.8rem;color:var(--t2);margin-top:.3rem">Halal screening · Analyst consensus · Ratios</div>
        </div>''', unsafe_allow_html=True)
        st.caption("Enter one or more tickers separated by commas · Select analysis period · Click RUN")
        st.caption("Stock Analyzer Pro v2.0 · Powered by yfinance via Google Colab")
        st.stop()
else:
    # When returning from analysis, show inputs at top
    c1, c2 = st.columns([3, 1])
    with c1:
        ticker_input = st.text_input("Ticker(s)", value="AAPL", placeholder="AAPL, MSFT, NVDA", label_visibility="collapsed")
    with c2:
        period = st.selectbox("Period", options=["3mo","6mo","1y","2y","5y"], index=2, label_visibility="collapsed")
    run = st.button("▶  RUN ANALYSIS", use_container_width=True, type="primary")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if run:
    tickers=[t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    if not tickers: st.error("Enter a ticker."); st.stop()
    sys.path.insert(0,os.path.dirname(__file__))
    from stock_analyzer import analyze_stock_web

    for ticker in tickers:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_buf=io.StringIO(); prog=st.progress(0,text=f"Analyzing {ticker}...")
            try:
                with contextlib.redirect_stdout(log_buf): R=analyze_stock_web(ticker,period=period,output_dir=tmpdir)
                prog.progress(100,text=f"✅ {ticker}")
            except Exception as e:
                prog.progress(100,text=f"❌ {ticker}"); st.error(str(e))
                with st.expander("Log"): st.code(log_buf.getvalue())
                continue

            df=R['df'];info=R['info'];signals=R['signals'];L=R['latest_data']
            sr=R['sr_levels'];divs=R['divergences'];cp=R['chart_patterns'];ep=R['eccentric_patterns']
            fib=R['fib_result'];mtf_fib=R['mtf_fib'];halal=R['halal_result']
            mtf_t=R['mtf_trends'];mtf_e=R['mtf_eval'];fund=R['fundamentals'];bco=R['barchart_opinion']
            cur=sf(L,'Close')
            overall=signals.get('overall','NEUTRAL')
            company=info.get('longName',info.get('shortName',ticker))
            sector=info.get('sector','');industry=info.get('industry','')
            pct_chg=((cur-df['Close'].iloc[-2])/df['Close'].iloc[-2]*100) if len(df)>=2 else 0
            chg_c="var(--accent)" if pct_chg>=0 else "var(--red)"
            h_status=halal.get('status','?')
            rating=str(info.get('recommendationKey','N/A')).replace('_',' ').title()

            # ═══ HEADER ═══
            st.markdown(f'''<div class="hdr">
                <div><div class="tk">{ticker} <span style="font-size:.85rem;color:var(--t3);font-weight:400">{company}</span></div>
                <div class="co">{sector}{" · " if sector and industry else ""}{industry}</div></div>
                <div class="pr" style="color:{chg_c}">${cur:,.2f} ({pct_chg:+.2f}%)</div>
                <span class="period-tag">Period: {period.upper()}</span>
            </div>''',unsafe_allow_html=True)

            # Quick stats
            q1,q2,q3,q4,q5,q6=st.columns(6)
            q1.markdown(f'<div class="dc"><div class="l">Signal</div><div class="v">{sg(overall)}</div></div>',unsafe_allow_html=True)
            q2.markdown(f'<div class="dc"><div class="l">Score</div><div class="v">{signals.get("total_score",0)}</div></div>',unsafe_allow_html=True)
            q3.markdown(f'<div class="dc"><div class="l">RSI (14)</div><div class="v">{sf(L,"RSI"):.1f}</div></div>',unsafe_allow_html=True)
            q4.markdown(f'<div class="dc"><div class="l">Volume</div><div class="v">{fmt(sf(L,"Volume"))}</div></div>',unsafe_allow_html=True)
            q5.markdown(f'<div class="dc"><div class="l">Analyst</div><div class="v">{rating}</div></div>',unsafe_allow_html=True)
            h_cls="h-pass" if h_status=="HALAL" else "h-fail" if h_status=="NON-COMPLIANT" else "h-warn"
            q6.markdown(f'<div class="dc"><div class="l">Halal</div><div class="v"><span class="{h_cls}">{h_status}</span></div></div>',unsafe_allow_html=True)

            with st.expander("📜 Analysis Log"): st.code(log_buf.getvalue())

            # ═══ TABS ═══
            tab1,tab2,tab3,tab4,tab5=st.tabs(["📊 Overview","⚡ Technical","📐 Patterns & Levels","🔄 Multi-Timeframe","💰 Fundamentals"])

            # ═══════════ TAB 1: OVERVIEW ═══════════
            with tab1:
                for pk in ['chart_path','fib_chart_path','fib_chart_weekly','fib_chart_monthly']:
                    if R.get(pk) and os.path.exists(R[pk]): st.image(R[pk],use_container_width=True)

                sh("PRICE SUMMARY")
                p1,p2,p3,p4,p5,p6=st.columns(6)
                for c,l,v in [(p1,"Close",f"${cur:,.2f}"),(p2,"Volume",fmt(sf(L,'Volume'))),(p3,"RSI (14)",f"{sf(L,'RSI'):.1f}"),(p4,"MACD",f"{sf(L,'MACD'):.4f}"),(p5,"52W High",fmt(info.get('fiftyTwoWeekHigh'),p="$")),(p6,"52W Low",fmt(info.get('fiftyTwoWeekLow'),p="$"))]:
                    c.markdown(f'<div class="dc"><div class="l">{l}</div><div class="v">{v}</div></div>',unsafe_allow_html=True)
                avg_vol=sf(L,'Vol_SMA_20',0); vol_r=sf(L,'Volume',0)/avg_vol if avg_vol>0 else 0
                st.caption(f"Current Vol: {fmt(sf(L,'Volume'))} · Avg Vol (20d): {fmt(avg_vol)} · Ratio: {vol_r:.1f}x · Period: {period}")

                sh("SIGNAL SUMMARY")
                ts=signals.get('trend','NEUTRAL');tsc=signals.get('trend_score',0)
                ms=signals.get('momentum','NEUTRAL');msc=signals.get('mom_score',0)
                sc=signals.get('total_score',0)
                st.markdown(f"{sg(overall)} &nbsp; Score: **{sc}** (Trend {tsc} + Momentum {msc})",unsafe_allow_html=True)
                st.caption("STRONG BUY≥5 | BUY≥2 | NEUTRAL -1~1 | SELL≤-2 | STRONG SELL≤-5")
                s1,s2,s3=st.columns(3)
                s1.markdown(f"**Trend:** {sg(ts)} Score: {tsc}",unsafe_allow_html=True)
                s2.markdown(f"**Momentum:** {sg(ms)} Score: {msc}",unsafe_allow_html=True)
                s3.markdown(f"**Volume:** {signals.get('volume','NORMAL')}")
                for lb,ky in [("Trend Breakdown","trend_details"),("Momentum Breakdown","mom_details")]:
                    dd=signals.get(ky,[])
                    if dd:
                        with st.expander(lb):
                            for desc,pts,ib in dd:
                                ic="🟢" if ib else "🔴" if ib is False else "⚪"
                                st.markdown(f"{ic} **{pts}** {desc}")
                vn=signals.get('volatility',[])
                if vn: st.markdown("**Volatility:** "+" | ".join(vn))

                sh("ANALYST CONSENSUS")
                a1,a2,a3,a4=st.columns(4)
                for c,l,v in [(a1,"Rating",rating),(a2,"Target",fmt(info.get('targetMeanPrice'),p="$")),(a3,"Analysts",str(info.get('numberOfAnalystOpinions','N/A')))]:
                    c.markdown(f'<div class="dc"><div class="l">{l}</div><div class="v">{v}</div></div>',unsafe_allow_html=True)
                tp=info.get('targetMeanPrice')
                try: ups=f"{((float(tp)-cur)/cur)*100:+.1f}%"
                except: ups="N/A"
                a4.markdown(f'<div class="dc"><div class="l">Upside</div><div class="v">{ups}</div></div>',unsafe_allow_html=True)
                bd=info.get('_analyst_breakdown')
                if bd: st.markdown(f"🟢 StrongBuy: **{bd.get('strongBuy',0)}** · Buy: **{bd.get('buy',0)}** · ⚪ Hold: **{bd.get('hold',0)}** · 🔴 Sell: **{bd.get('sell',0)}** · StrongSell: **{bd.get('strongSell',0)}**")

            # ═══════════ TAB 2: TECHNICAL ═══════════
            with tab2:
                if bco and bco.get('barchart_opinion'):
                    sh("BARCHART TECHNICAL OPINION")
                    bo=bco['barchart_opinion']
                    bcs=[bo['short_term']['signal'],bo['medium_term']['signal'],bo['long_term']['signal']]
                    bcb=sum(1 for s in bcs if 'BUY' in s);bce=sum(1 for s in bcs if 'SELL' in s)
                    bco_ov='BUY' if bcb>bce else 'SELL' if bce>bcb else 'NEUTRAL'
                    st.markdown(f"**Overall:** {sg(bco_ov)}",unsafe_allow_html=True)
                    b1,b2,b3=st.columns(3)
                    b1.markdown(f"**Short-Term:** {sg(bo['short_term']['signal'])} (Score: {bo['short_term']['score']})",unsafe_allow_html=True)
                    b2.markdown(f"**Medium-Term:** {sg(bo['medium_term']['signal'])} (Score: {bo['medium_term']['score']})",unsafe_allow_html=True)
                    b3.markdown(f"**Long-Term:** {sg(bo['long_term']['signal'])} (Score: {bo['long_term']['score']})",unsafe_allow_html=True)

                sh("KEY TECHNICAL LEVELS")
                k1,k2=st.columns(2)
                with k1:
                    for k in ['SMA_20','SMA_50','SMA_200','EMA_9','EMA_21']:
                        mr(k.replace('_',' '),lvl(k,sf(L,k,None),cur))
                    cross="EMA 9 > EMA 21 ✅ Bullish" if sf(L,'EMA_9',0)>sf(L,'EMA_21',0) else "EMA 9 < EMA 21 ❌ Bearish"
                    mr("EMA Cross",cross)
                with k2:
                    mr("BB Upper",lvl("BB",sf(L,'BB_Upper',None),cur))
                    mr("BB Lower",lvl("BB",sf(L,'BB_Lower',None),cur))
                    mr("BB %B",f"{sf(L,'BB_Pct',0):.2f}")
                    mr("VWAP",lvl("VWAP",sf(L,'VWAP',None),cur))

                sh("MOMENTUM INDICATORS")
                m1,m2,m3,m4=st.columns(4)
                for c,l,v in [(m1,"RSI (14)",f"{sf(L,'RSI'):.1f}"),(m2,"CCI (20)",f"{sf(L,'CCI_20'):.1f}"),(m3,"StochRSI K/D",f"{sf(L,'StochRSI_K'):.1f} / {sf(L,'StochRSI_D'):.1f}"),(m4,"MFI",f"{sf(L,'MFI'):.1f}")]:
                    c.markdown(f'<div class="dc"><div class="l">{l}</div><div class="v">{v}</div></div>',unsafe_allow_html=True)
                m5,m6,m7,m8=st.columns(4)
                atr_p=f"({sf(L,'ATR',0)/cur*100:.1f}%)" if cur>0 else ""
                for c,l,v in [(m5,"ADX",f"{sf(L,'ADX'):.1f}"),(m6,"+DI / -DI",f"{sf(L,'DI_Plus'):.1f} / {sf(L,'DI_Minus'):.1f}"),(m7,"ATR (14)",f"${sf(L,'ATR'):.2f} {atr_p}"),(m8,"MACD",f"{sf(L,'MACD'):.4f}")]:
                    c.markdown(f'<div class="dc"><div class="l">{l}</div><div class="v">{v}</div></div>',unsafe_allow_html=True)

                sh("KELTNER / BOLLINGER / SQUEEZE")
                sq="🔴 SQUEEZE ACTIVE — Expect breakout" if L.get('Squeeze') else "🟢 No Squeeze"
                st.markdown(f"**Squeeze:** {sq}")
                kb1,kb2,kb3=st.columns(3)
                kb1.markdown(f"**KC Width:** ${sf(L,'KC_Width'):.2f}")
                kb2.markdown(f"**BB Width:** {sf(L,'BB_Width'):.4f}")
                kcu=sf(L,'KC_Upper',0);kcl=sf(L,'KC_Lower',0);kcm=sf(L,'KC_Middle',0)
                kcp="ABOVE Upper ↑ Strong bullish" if cur>kcu else "BELOW Lower ↓ Strong bearish" if cur<kcl else "Above Middle ↗ Mild bullish" if cur>kcm else "Below Middle ↘ Mild bearish"
                kb3.markdown(f"**Price vs KC:** {kcp}")

                sh("MFI / OBV")
                rv=sf(L,'RSI');mv=sf(L,'MFI')
                if rv>70 and mv>80: mfr="OVERBOUGHT"
                elif rv<30 and mv<20: mfr="OVERSOLD"
                elif abs(rv-mv)>20: mfr="DIVERGENCE"
                else: mfr="NEUTRAL"
                mr("MFI/RSI",f"{mfr} (RSI={rv:.1f}, MFI={mv:.1f})")
                obv=sf(L,'OBV',0)
                mr("OBV",fmt(obv))
                if len(df)>=20 and 'OBV' in df.columns:
                    mr("OBV Trend (5d)","Rising" if obv>df['OBV'].iloc[-5] else "Falling")
                    mr("OBV Trend (20d)","Rising" if obv>df['OBV'].iloc[-20] else "Falling")
                    price_trend='up' if cur>df['Close'].iloc[-21] else 'down'
                    obv_long="Rising" if obv>df['OBV'].iloc[-20] else "Falling"
                    if price_trend=='up' and obv_long=='Falling': mr("OBV/Price","⚠️ DIVERGENCE — Price up but OBV falling")
                    elif price_trend=='down' and obv_long=='Rising': mr("OBV/Price","⚠️ DIVERGENCE — Price down but OBV rising")
                    elif price_trend=='up' and obv_long=='Rising': mr("OBV/Price","✅ CONFIRMED — Both rising")
                    else: mr("OBV/Price","✅ CONFIRMED — Both falling")

                sh("ICHIMOKU CLOUD")
                tenkan=sf(L,'Ichi_Tenkan',None);kijun=sf(L,'Ichi_Kijun',None)
                spA=sf(L,'Ichi_SpanA',None);spB=sf(L,'Ichi_SpanB',None)
                i1,i2=st.columns(2)
                with i1:
                    if tenkan: mr("Tenkan-sen (9)",lvl("T",tenkan,cur))
                    if kijun: mr("Kijun-sen (26)",lvl("K",kijun,cur))
                    if spA is not None: mr("Senkou Span A",f"${spA:,.2f}")
                    if spB is not None: mr("Senkou Span B",f"${spB:,.2f}")
                with i2:
                    if spA is not None and spB is not None:
                        mr("Cloud","GREEN (bullish) ✅" if spA>spB else "RED (bearish) ❌")
                        if cur>max(spA,spB): mr("Price vs Cloud","ABOVE ↑ Bullish")
                        elif cur<min(spA,spB): mr("Price vs Cloud","BELOW ↓ Bearish")
                        else: mr("Price vs Cloud","INSIDE ↔ Transitioning")
                    if tenkan and kijun:
                        mr("TK Cross","Tenkan > Kijun ✅" if tenkan>kijun else "Tenkan < Kijun ❌")
                        isc=sum([tenkan>kijun,cur>tenkan,cur>kijun,spA is not None and spB is not None and cur>max(spA,spB),spA is not None and spB is not None and spA>spB])
                        isig="BULLISH" if isc>=3 else "BEARISH" if isc<=1 else "NEUTRAL"
                        st.markdown(f"**Ichimoku Signal:** {sg(isig)} ({isc}/5)",unsafe_allow_html=True)
                if not tenkan and not kijun:
                    st.info("Ichimoku data not available — insufficient history for calculation.")

            # ═══════════ TAB 3: PATTERNS & LEVELS ═══════════
            with tab3:
                sh("SUPPORT & RESISTANCE")
                sc1,sc2=st.columns(2)
                with sc1:
                    st.markdown(f"**▸ Current: ${cur:,.2f}**")
                    st.markdown("**Support Levels**")
                    sups=sr.get('supports',[])
                    if sups:
                        for s in sups:
                            p=s.get('price',0);st_r=s.get('strength',1)
                            st.markdown(f"🟢 **${p:,.2f}** ({((p-cur)/cur)*100:+.1f}%) [Strength: {st_r}]")
                    else:
                        st.caption("No support levels detected")
                with sc2:
                    st.markdown("**Resistance Levels**")
                    ress=sr.get('resistances',[])
                    if ress:
                        for r in ress:
                            p=r.get('price',0);st_r=r.get('strength',1)
                            st.markdown(f"🔴 **${p:,.2f}** ({((p-cur)/cur)*100:+.1f}%) [Strength: {st_r}]")
                    else:
                        st.caption("No resistance levels detected")

                sh("FIBONACCI ANALYSIS")
                if fib.get('available'):
                    f1,f2=st.columns([2,1])
                    with f1:
                        mr("Trend",fib.get('primary_trend','N/A'))
                        mr("Swing High",f"${fib.get('swing_high',0):,.2f} ({fib.get('swing_high_date','N/A')})")
                        mr("Swing Low",f"${fib.get('swing_low',0):,.2f} ({fib.get('swing_low_date','N/A')})")
                        mr("Range",f"${fib.get('swing_high',0)-fib.get('swing_low',0):,.2f}")
                        mr("Active Tool",f"Fibonacci {fib.get('active_tool','RETRACEMENT').upper()}")
                        if fib.get('current_zone'): mr("Current Zone",fib['current_zone'])
                    with f2:
                        if fib.get('reason'): st.caption(fib['reason'])
                    levels=fib.get('levels',{})
                    if levels:
                        rows=[]
                        for k,v in levels.items():
                            pct=((v-cur)/cur)*100
                            marker=" ◄" if abs(pct)<2 else ""
                            rows.append({"Level":k,"Price":f"${v:,.2f}","Distance":f"{pct:+.1f}%{marker}"})
                        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
                else:
                    st.info(f"Fibonacci: {fib.get('reason','Insufficient data')}")
                for tf in ['Weekly','Monthly']:
                    tff=mtf_fib.get(tf,{})
                    if tff.get('available'):
                        with st.expander(f"Fibonacci — {tf}"):
                            mr("Trend",tff.get('primary_trend','N/A'))
                            lvls=tff.get('levels',{})
                            if lvls:
                                rows=[{"Level":k,"Price":f"${v:,.2f}","Dist":f"{((v-cur)/cur)*100:+.1f}%"} for k,v in lvls.items()]
                                st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)

                sh("CHART PATTERNS")
                if cp:
                    for p in cp:
                        ic="📈" if p.get('direction')=='BULLISH' else "📉" if p.get('direction')=='BEARISH' else "📊"
                        st.markdown(f"{ic} **{p.get('pattern','?')}** — {sg(p.get('direction','N/A'))} — Confidence: {p.get('confidence','N/A')}",unsafe_allow_html=True)
                        if p.get('description'): st.caption(p['description'])
                else:
                    st.info("No chart patterns detected.")
                if ep:
                    with st.expander("theEccentricTrader Method"):
                        for p in ep: st.markdown(f"**{p.get('pattern','?')}** — {p.get('direction','N/A')}")
                else:
                    with st.expander("theEccentricTrader Method"): st.info("No patterns detected.")

                sh("DIVERGENCE ANALYSIS")
                if divs:
                    for d in divs:
                        ic="🟢" if d.get('type')=='BULLISH' else "🔴"
                        st.markdown(f"{ic} **{d.get('type','?')} {d.get('indicator','?')}** — {d.get('description','')}")
                else:
                    st.info("No divergences detected. RSI, MACD, MFI, and OBV are generally aligned.")

            # ═══════════ TAB 4: MULTI-TIMEFRAME ═══════════
            with tab4:
                sh("MULTI-TIMEFRAME TREND")
                for tn in ['Daily','Weekly','Monthly']:
                    tf=mtf_t.get(tn,{})
                    if tf.get('available',False):
                        trend=tf.get('trend','N/A')
                        mapped=trend.replace('UPTREND','BULLISH').replace('DOWNTREND','BEARISH').replace('SIDEWAYS','NEUTRAL')
                        for w in ['STRONG ']: mapped=mapped.replace(w,w)
                        st.markdown(f"**{tn}:** {sg(mapped)} — {tf.get('detail','')}",unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{tn}:** N/A — {tf.get('detail',tf.get('reason',''))}")

                if mtf_e:
                    sh("TECHNICAL EVALUATION — MULTI-TIMEFRAME")
                    er=[]
                    for tl in ['5 Min','15 Min','30 Min','4 Hours','Daily','Weekly','Monthly']:
                        ev=mtf_e.get(tl,{})
                        if ev.get('available'):
                            er.append({"TF":tl,"Overall":ev.get('overall_summary','N/A'),"MA":ev.get('ma_summary','N/A'),"Osc":ev.get('osc_summary','N/A'),"Buy":ev.get('total_buy',0),"Neut":ev.get('total_neutral',0),"Sell":ev.get('total_sell',0)})
                        else:
                            er.append({"TF":tl,"Overall":"—","MA":"—","Osc":"—","Buy":"—","Neut":"—","Sell":"—"})
                    st.dataframe(pd.DataFrame(er),hide_index=True,use_container_width=True)

                if bco and bco.get('overall_summary'):
                    sh("DAILY DETAIL")
                    st.markdown(f"**Overall:** {sg(bco['overall_summary'])} — Buy: {bco.get('total_buy',0)} / Neutral: {bco.get('total_neutral',0)} / Sell: {bco.get('total_sell',0)}",unsafe_allow_html=True)

                    sh("MOVING AVERAGES (Daily)")
                    st.markdown(f"**Summary:** {sg(bco.get('ma_summary','N/A'))} — B:{bco.get('ma_buy',0)} / N:{bco.get('ma_neutral',0)} / S:{bco.get('ma_sell',0)}",unsafe_allow_html=True)
                    mar=bco.get('ma_results',[])
                    if mar:
                        rows=[{"Indicator":m.get('name',''),"Value":f"${m.get('value',0):,.2f}" if m.get('value') else "N/A","Signal":m.get('signal','N/A'),"Dist":f"{m.get('pct_diff',0):+.2f}%"} for m in mar]
                        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)

                    sh("OSCILLATORS (Daily)")
                    st.markdown(f"**Summary:** {sg(bco.get('osc_summary','N/A'))} — B:{bco.get('osc_buy',0)} / N:{bco.get('osc_neutral',0)} / S:{bco.get('osc_sell',0)}",unsafe_allow_html=True)
                    osr=bco.get('osc_results',[])
                    if osr:
                        rows=[]
                        for o in osr:
                            try: ov=f"{float(o.get('value',0)):.2f}"
                            except (ValueError,TypeError): ov=str(o.get('value','N/A'))
                            rows.append({"Indicator":o.get('name',''),"Value":ov,"Signal":o.get('signal','N/A')})
                        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)

                if bco and bco.get('pivots'):
                    sh("PIVOT POINTS")
                    pivots=bco['pivots'];prows=[]
                    for lv in ['R3','R2','R1','PP','S1','S2','S3']:
                        row={"Level":lv}
                        for mt in ['Classic','Fibonacci','Woodie','Camarilla']:
                            v=pivots.get(mt,{}).get(lv,0)
                            pct=((v-cur)/cur)*100 if cur>0 else 0
                            row[mt]=f"${v:,.2f} ({pct:+.1f}%)"
                        prows.append(row)
                    st.dataframe(pd.DataFrame(prows),hide_index=True,use_container_width=True)

            # ═══════════ TAB 5: FUNDAMENTALS ═══════════
            with tab5:
                sh("SHARIAH (HALAL) COMPLIANCE")
                hc2="h-pass" if h_status=="HALAL" else "h-fail" if h_status=="NON-COMPLIANT" else "h-warn"
                st.markdown(f'<span class="{hc2}">{h_status}</span> &nbsp; ({halal.get("pass_count",0)}/{halal.get("total_checks",4)} criteria)',unsafe_allow_html=True)
                for d in halal.get('details',[]):
                    ic="✅" if d['status']=='PASS' else "❌" if d['status']=='FAIL' else "⚠️"
                    st.markdown(f"{ic} **{d['criterion']}** — {d['status']} — {d['value']}")
                    if d.get('note'): st.caption(d['note'])
                st.caption("Based on AAOIFI screening. Consult a Shariah scholar or certified platform (Musaffa, Zoya) for authoritative rulings.")

                sh("FUNDAMENTAL OVERVIEW")
                fc1,fc2=st.columns(2)
                with fc1:
                    for k in ['Company','Sector','Industry','Market Cap','P/E (Trailing)','P/E (Forward)','PEG Ratio','P/S Ratio','P/B Ratio','EV/EBITDA']:
                        v=fund.get(k,'N/A')
                        if k=='Market Cap': v=fmt(v,p="$")
                        elif isinstance(v,float) and not np.isnan(v): v=f"{v:.2f}"
                        mr(k,v)
                with fc2:
                    for k in ['Revenue','Revenue Growth','Profit Margin','Operating Margin','ROE','ROA','Debt/Equity','Current Ratio','Free Cash Flow','Dividend Yield','Beta','Avg Volume','52W High','52W Low','Target Price','Analyst Rating','Num Analysts']:
                        v=fund.get(k,'N/A')
                        if k in ('Revenue','Free Cash Flow'): v=fmt(v,p="$")
                        elif k in ('52W High','52W Low','Target Price'): v=fmt(v,p="$")
                        elif k in ('Revenue Growth','Profit Margin','Operating Margin','ROE','ROA','Dividend Yield'): v=fp(v)
                        elif isinstance(v,float) and not np.isnan(v): v=f"{v:.2f}"
                        mr(k,v)

            st.markdown("---")
            st.caption(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} · Period: {period} · Data: yfinance · For informational purposes only. Not financial advice.")

"""
Stock Analyzer Pro — Streamlit Web App (Full Report)
======================================================
Renders ALL analysis sections directly in the browser,
matching the PDF report content 1:1.
"""
import streamlit as st
import os, sys, io, tempfile, contextlib
from datetime import datetime
import pandas as pd
import numpy as np

if "AV_API_KEY" in st.secrets:
    os.environ["AV_API_KEY"] = st.secrets["AV_API_KEY"]

st.set_page_config(page_title="Stock Analyzer Pro", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);padding:1.5rem 2rem;border-radius:12px;margin-bottom:1.5rem;border:1px solid #30363d}
.main-header h1{color:#e0e0e0;margin:0;font-size:1.8rem}
.main-header p{color:#8b949e;margin:.3rem 0 0 0;font-size:.95rem}
.section-header{background:#161b22;border-left:4px solid #238636;padding:.6rem 1rem;border-radius:0 8px 8px 0;margin:1.2rem 0 .8rem 0;font-size:1.05rem;font-weight:700;color:#e0e0e0}
.badge-buy{background:#00e676;color:#1a1a2e;padding:.3rem .8rem;border-radius:6px;font-weight:700;font-size:.95rem;display:inline-block}
.badge-sell{background:#ff1744;color:#fff;padding:.3rem .8rem;border-radius:6px;font-weight:700;font-size:.95rem;display:inline-block}
.badge-neutral{background:#ffd740;color:#1a1a2e;padding:.3rem .8rem;border-radius:6px;font-weight:700;font-size:.95rem;display:inline-block}
.badge-halal{background:#00e676;color:#1a1a2e;padding:.3rem .8rem;border-radius:6px;font-weight:700}
.badge-haram{background:#ff1744;color:#fff;padding:.3rem .8rem;border-radius:6px;font-weight:700}
.badge-uncertain{background:#ffd740;color:#1a1a2e;padding:.3rem .8rem;border-radius:6px;font-weight:700}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>📈 Stock Analyzer Pro</h1><p>Comprehensive Technical & Fundamental Analysis</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Settings")
    ticker_input = st.text_input("Ticker Symbol(s)", value="AAPL", placeholder="AAPL, MSFT")
    period = st.selectbox("Analysis Period", options=["3mo","6mo","1y","2y","5y"], index=2)
    st.markdown("---")
    st.caption("Powered by yfinance via Google Colab")

def badge(text, kind="neutral"):
    return f'<span class="badge-{kind}">{text}</span>'

def signal_badge(sig):
    sig = str(sig).upper()
    if any(w in sig for w in ["BUY","BULLISH","UPTREND","PASS","HALAL","STRONG BUY","STRONG BULLISH"]):
        return badge(sig, "buy")
    elif any(w in sig for w in ["SELL","BEARISH","DOWNTREND","FAIL","NON-COMPLIANT","HARAM"]):
        return badge(sig, "sell")
    return badge(sig, "neutral")

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def fmt(val, prefix="", suffix="", d=2):
    if val is None or val == "N/A": return "N/A"
    try:
        v = float(val)
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

def safe(d, k, fallback=0):
    v = d.get(k, fallback)
    if v is None or (isinstance(v, float) and np.isnan(v)): return fallback
    return v

if not os.environ.get("AV_API_KEY"):
    st.warning("AV_API_KEY not found. If running on Colab with yfinance, this is fine.")

run = st.button("🚀  Run Analysis", use_container_width=True)

if run:
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    if not tickers: st.error("Enter at least one ticker."); st.stop()

    sys.path.insert(0, os.path.dirname(__file__))
    from stock_analyzer import analyze_stock_web

    for ticker in tickers:
        st.markdown("---")
        st.markdown(f"## 🔍 {ticker}")
        with tempfile.TemporaryDirectory() as tmpdir:
            log_buf = io.StringIO()
            progress = st.progress(0, text=f"Analyzing {ticker}...")
            try:
                with contextlib.redirect_stdout(log_buf):
                    R = analyze_stock_web(ticker, period=period, output_dir=tmpdir)
                progress.progress(100, text=f"✅ {ticker} Done!")
            except Exception as e:
                progress.progress(100, text=f"❌ {ticker} Error")
                st.error(f"Error: {e}")
                with st.expander("Log"): st.code(log_buf.getvalue())
                continue

            with st.expander("📜 Analysis Log"): st.code(log_buf.getvalue())

            df = R['df']; info = R['info']; signals = R['signals']; L = R['latest_data']
            sr = R['sr_levels']; divs = R['divergences']; cp = R['chart_patterns']; ep = R['eccentric_patterns']
            fib = R['fib_result']; mtf_fib = R['mtf_fib']; halal = R['halal_result']
            mtf_t = R['mtf_trends']; mtf_e = R['mtf_eval']; fund = R['fundamentals']; bco = R['barchart_opinion']
            current = safe(L, 'Close')

            # ═══ CHARTS ═══
            for p_key in ['chart_path','fib_chart_path','fib_chart_weekly','fib_chart_monthly']:
                if R.get(p_key) and os.path.exists(R[p_key]):
                    st.image(R[p_key], use_container_width=True)

            # ═══ PRICE SUMMARY ═══
            section("PRICE SUMMARY")
            pc1,pc2,pc3,pc4,pc5,pc6 = st.columns(6)
            pc1.metric("Close", f"${current:,.2f}")
            pc2.metric("Volume", fmt(safe(L,'Volume')))
            pc3.metric("RSI", f"{safe(L,'RSI'):.1f}")
            pc4.metric("MACD", f"{safe(L,'MACD'):.4f}")
            pc5.metric("52W High", fmt(info.get('fiftyTwoWeekHigh'), prefix="$"))
            pc6.metric("52W Low", fmt(info.get('fiftyTwoWeekLow'), prefix="$"))
            # Volume detail
            avg_vol = safe(L, 'Vol_SMA_20', 0)
            vol_ratio = safe(L,'Volume',0) / avg_vol if avg_vol > 0 else 0
            st.caption(f"Volume: {fmt(safe(L,'Volume'))} | Avg (20d): {fmt(avg_vol)} | Ratio: {vol_ratio:.1f}x")

            # ═══ SIGNAL SUMMARY ═══
            section("SIGNAL SUMMARY")
            overall = signals.get('overall','NEUTRAL')
            score = signals.get('total_score', 0)
            trend_sig = signals.get('trend','NEUTRAL')
            trend_sc = signals.get('trend_score', 0)
            mom_sig = signals.get('momentum','NEUTRAL')
            mom_sc = signals.get('mom_score', 0)
            vol_signal = signals.get('volume','NORMAL')

            st.markdown(f"{signal_badge(overall)} &nbsp; Confluence Score: **{score}** (Trend {trend_sc} + Momentum {mom_sc})", unsafe_allow_html=True)
            st.caption("STRONG BUY >= 5 | BUY >= 2 | NEUTRAL -1 to 1 | SELL <= -2 | STRONG SELL <= -5")
            tc1,tc2,tc3 = st.columns(3)
            tc1.markdown(f"**Trend:** {signal_badge(trend_sig)} Score: {trend_sc}", unsafe_allow_html=True)
            tc2.markdown(f"**Momentum:** {signal_badge(mom_sig)} Score: {mom_sc}", unsafe_allow_html=True)
            tc3.markdown(f"**Volume:** {vol_signal}")

            for label, key in [("Trend Breakdown","trend_details"),("Momentum Breakdown","mom_details")]:
                details = signals.get(key, [])
                if details:
                    with st.expander(label):
                        for desc, pts, is_bull in details:
                            icon = "🟢" if is_bull else "🔴" if is_bull is False else "⚪"
                            st.markdown(f"{icon} **{pts}** {desc}")

            vol_notes = signals.get('volatility', [])
            if vol_notes:
                st.markdown("**Volatility:** " + " | ".join(vol_notes))

            # ═══ HALAL ═══
            section("SHARIAH (HALAL) COMPLIANCE")
            h_status = halal.get('status','UNKNOWN')
            h_cls = "halal" if h_status=="HALAL" else "haram" if h_status=="NON-COMPLIANT" else "uncertain"
            st.markdown(f'<span class="badge-{h_cls}">{h_status}</span> ({halal.get("pass_count",0)}/{halal.get("total_checks",4)} criteria passed)', unsafe_allow_html=True)
            for d in halal.get('details', []):
                icon = "✅" if d['status']=='PASS' else "❌" if d['status']=='FAIL' else "⚠️"
                st.markdown(f"{icon} **{d['criterion']}** — {d['status']} — {d['value']}")
                if d.get('note'): st.caption(d['note'])

            # ═══ ANALYST CONSENSUS ═══
            section("ANALYST CONSENSUS")
            ac1,ac2,ac3,ac4 = st.columns(4)
            ac1.metric("Rating", str(info.get('recommendationKey','N/A')).replace('_',' ').title())
            ac2.metric("Target Price", fmt(info.get('targetMeanPrice'), prefix="$"))
            ac3.metric("# Analysts", str(info.get('numberOfAnalystOpinions','N/A')))
            tp = info.get('targetMeanPrice')
            if tp and current > 0:
                try: ac4.metric("Implied Upside", f"{((float(tp)-current)/current)*100:+.1f}%")
                except: ac4.metric("Implied Upside", "N/A")
            bd = info.get('_analyst_breakdown')
            if bd:
                st.markdown(f"🟢 Strong Buy: **{bd.get('strongBuy',0)}** · Buy: **{bd.get('buy',0)}** · ⚪ Hold: **{bd.get('hold',0)}** · 🔴 Sell: **{bd.get('sell',0)}** · Strong Sell: **{bd.get('strongSell',0)}**")

            # ═══ BARCHART-STYLE OPINION ═══
            if bco and bco.get('barchart_opinion'):
                section("TECHNICAL OPINION (Barchart-Style)")
                bo = bco['barchart_opinion']
                # Overall
                bc_signals = [bo['short_term']['signal'], bo['medium_term']['signal'], bo['long_term']['signal']]
                bc_buy = sum(1 for s in bc_signals if 'BUY' in s)
                bc_sell = sum(1 for s in bc_signals if 'SELL' in s)
                bc_overall = 'BUY' if bc_buy > bc_sell else 'SELL' if bc_sell > bc_buy else 'NEUTRAL'
                st.markdown(f"**Barchart Opinion:** {signal_badge(bc_overall)}", unsafe_allow_html=True)
                bc1,bc2,bc3 = st.columns(3)
                bc1.markdown(f"**Short-Term:** {signal_badge(bo['short_term']['signal'])} (Score: {bo['short_term']['score']})", unsafe_allow_html=True)
                bc2.markdown(f"**Medium-Term:** {signal_badge(bo['medium_term']['signal'])} (Score: {bo['medium_term']['score']})", unsafe_allow_html=True)
                bc3.markdown(f"**Long-Term:** {signal_badge(bo['long_term']['signal'])} (Score: {bo['long_term']['score']})", unsafe_allow_html=True)

            # ═══ KEY TECHNICAL LEVELS ═══
            section("KEY TECHNICAL LEVELS")
            def level_row(name, val):
                if val is None or (isinstance(val, float) and np.isnan(val)): return f"**{name}:** N/A"
                pct = ((current - val)/val)*100 if val != 0 else 0
                pos = "Above" if current > val else "Below"
                return f"**{name}:** ${val:,.2f} — {pos} ({pct:+.2f}%)"
            kc1,kc2 = st.columns(2)
            with kc1:
                for k in ['SMA_20','SMA_50','SMA_200','EMA_9','EMA_21']:
                    st.markdown(level_row(k.replace('_',' '), safe(L,k,None)))
                cross = "EMA 9 > EMA 21 (Bullish)" if safe(L,'EMA_9',0) > safe(L,'EMA_21',0) else "EMA 9 < EMA 21 (Bearish)"
                st.markdown(f"**EMA Cross:** {cross}")
            with kc2:
                st.markdown(level_row("BB Upper", safe(L,'BB_Upper',None)))
                st.markdown(level_row("BB Lower", safe(L,'BB_Lower',None)))
                st.markdown(f"**BB %B:** {safe(L,'BB_PctB',0):.2f}")
                st.markdown(level_row("VWAP", safe(L,'VWAP',None)))

            # ═══ MOMENTUM INDICATORS ═══
            section("MOMENTUM INDICATORS")
            mc1,mc2,mc3,mc4 = st.columns(4)
            mc1.metric("RSI (14)", f"{safe(L,'RSI'):.1f}")
            mc2.metric("CCI (20)", f"{safe(L,'CCI'):.1f}")
            mc3.metric("StochRSI K/D", f"{safe(L,'StochRSI_K'):.1f} / {safe(L,'StochRSI_D'):.1f}")
            mc4.metric("MFI", f"{safe(L,'MFI'):.1f}")
            mc5,mc6,mc7,mc8 = st.columns(4)
            mc5.metric("ADX", f"{safe(L,'ADX'):.1f}")
            mc6.metric("+DI / -DI", f"{safe(L,'Plus_DI'):.1f} / {safe(L,'Minus_DI'):.1f}")
            mc7.metric("ATR (14)", f"${safe(L,'ATR'):.2f} ({safe(L,'ATR',0)/current*100:.2f}%)" if current > 0 else "N/A")
            mc8.metric("MACD", f"{safe(L,'MACD'):.4f}")

            # ═══ KELTNER / BOLLINGER / SQUEEZE ═══
            section("KELTNER CHANNEL / BOLLINGER / SQUEEZE")
            sq = "🔴 SQUEEZE ACTIVE - Expect breakout" if L.get('Squeeze') else "🟢 No Squeeze"
            st.markdown(f"**Squeeze Status:** {sq}")
            kb1,kb2,kb3 = st.columns(3)
            kb1.markdown(f"**KC Width:** ${safe(L,'KC_Width'):.2f}")
            kb2.markdown(f"**BB Width:** {safe(L,'BB_Width'):.4f}")
            kc_upper = safe(L,'KC_Upper',0); kc_lower = safe(L,'KC_Lower',0); kc_mid = safe(L,'KC_Middle',0)
            if current > kc_upper: kc_pos = "ABOVE KC Upper - Strong bullish"
            elif current < kc_lower: kc_pos = "BELOW KC Lower - Strong bearish"
            elif current > kc_mid: kc_pos = "Above KC Middle - Mild bullish"
            else: kc_pos = "Below KC Middle - Mild bearish"
            kb3.markdown(f"**Price vs KC:** {kc_pos}")

            # ═══ MFI / OBV ═══
            section("MFI / OBV")
            ob1,ob2 = st.columns(2)
            rsi_val = safe(L,'RSI'); mfi_val = safe(L,'MFI')
            if rsi_val > 70 and mfi_val > 80: mfi_rsi = "OVERBOUGHT"
            elif rsi_val < 30 and mfi_val < 20: mfi_rsi = "OVERSOLD"
            elif abs(rsi_val - mfi_val) > 20: mfi_rsi = "DIVERGENCE"
            else: mfi_rsi = "NEUTRAL"
            ob1.markdown(f"**MFI/RSI:** {mfi_rsi} (RSI={rsi_val:.1f}, MFI={mfi_val:.1f})")
            obv = safe(L,'OBV',0)
            ob2.markdown(f"**OBV:** {fmt(obv)}")
            # OBV trend
            if len(df) >= 20:
                obv_5 = df['OBV'].iloc[-5] if 'OBV' in df.columns else 0
                obv_20 = df['OBV'].iloc[-20] if 'OBV' in df.columns else 0
                obv_trend_5 = "Rising" if obv > obv_5 else "Falling"
                obv_trend_20 = "Rising" if obv > obv_20 else "Falling"
                st.markdown(f"**OBV Trend (5d):** {obv_trend_5} | **OBV Trend (20d):** {obv_trend_20}")

            # ═══ SUPPORT & RESISTANCE ═══
            section("SUPPORT & RESISTANCE")
            sc1,sc2 = st.columns(2)
            with sc2:
                st.markdown("**Resistance Levels**")
                for lvl in sr.get('resistance', []):
                    if isinstance(lvl, dict):
                        p = lvl.get('price', lvl.get('level', 0)); s = lvl.get('strength', 1)
                        st.markdown(f"🔴 ${p:,.2f} ({((p-current)/current)*100:+.1f}%) [Strength: {s}]")
                    else:
                        st.markdown(f"🔴 ${lvl:,.2f} ({((lvl-current)/current)*100:+.1f}%)")
            with sc1:
                st.markdown(f"**Current: ${current:,.2f}**")
                st.markdown("**Support Levels**")
                for lvl in sr.get('support', []):
                    if isinstance(lvl, dict):
                        p = lvl.get('price', lvl.get('level', 0)); s = lvl.get('strength', 1)
                        st.markdown(f"🟢 ${p:,.2f} ({((p-current)/current)*100:+.1f}%) [Strength: {s}]")
                    else:
                        st.markdown(f"🟢 ${lvl:,.2f} ({((lvl-current)/current)*100:+.1f}%)")

            # ═══ FIBONACCI ═══
            section("FIBONACCI ANALYSIS")
            if fib.get('available'):
                st.markdown(f"**Trend:** {fib.get('primary_trend','N/A')} | **Swing High:** ${fib.get('swing_high',0):,.2f} | **Swing Low:** ${fib.get('swing_low',0):,.2f} | **Range:** ${fib.get('swing_high',0)-fib.get('swing_low',0):,.2f}")
                st.markdown(f"**Active Tool:** Fibonacci {fib.get('active_tool','RETRACEMENT').upper()}")
                if fib.get('reason'): st.caption(fib['reason'])
                if fib.get('current_zone'): st.markdown(f"**Current Zone:** {fib['current_zone']}")
                levels = fib.get('levels', {})
                if levels:
                    rows = []
                    for k,v in levels.items():
                        pct = ((v-current)/current)*100
                        marker = " >>>" if abs(pct) < 2 else ""
                        rows.append({"Level": k, "Price": f"${v:,.2f}", "Distance": f"{pct:+.1f}%{marker}"})
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            else:
                st.info(f"Fibonacci: {fib.get('reason','No data')}")

            for tf in ['Weekly','Monthly']:
                tf_fib = mtf_fib.get(tf, {})
                if tf_fib.get('available'):
                    with st.expander(f"Fibonacci - {tf}"):
                        st.markdown(f"**Trend:** {tf_fib.get('primary_trend','N/A')}")
                        levels = tf_fib.get('levels', {})
                        if levels:
                            rows = [{"Level":k,"Price":f"${v:,.2f}","Dist":f"{((v-current)/current)*100:+.1f}%"} for k,v in levels.items()]
                            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            # ═══ CHART PATTERNS ═══
            section("CHART PATTERNS")
            if cp:
                for p in cp:
                    icon = "📈" if p.get('direction')=='BULLISH' else "📉" if p.get('direction')=='BEARISH' else "📊"
                    st.markdown(f"{icon} **{p.get('pattern','?')}** — {p.get('direction','N/A')} — Confidence: {p.get('confidence','N/A')}")
                    if p.get('description'): st.caption(p['description'])
            else:
                st.info("No chart patterns detected.")
            if ep:
                with st.expander("Chart Patterns - theEccentricTrader Method"):
                    if ep:
                        for p in ep:
                            st.markdown(f"**{p.get('pattern','?')}** — {p.get('direction','N/A')}")
                    else:
                        st.info("No patterns detected.")

            # ═══ DIVERGENCES ═══
            section("DIVERGENCE ANALYSIS")
            if divs:
                for d in divs:
                    icon = "🟢" if d.get('type')=='BULLISH' else "🔴"
                    st.markdown(f"{icon} **{d.get('type','?')} {d.get('indicator','?')}** — {d.get('description','')}")
            else:
                st.info("No divergences detected.")

            # ═══ ICHIMOKU ═══
            section("ICHIMOKU CLOUD")
            ichi_cols = st.columns(2)
            tenkan = safe(L,'Ichimoku_Tenkan',None); kijun = safe(L,'Ichimoku_Kijun',None)
            spanA = safe(L,'Ichimoku_SpanA',None); spanB = safe(L,'Ichimoku_SpanB',None)
            with ichi_cols[0]:
                if tenkan: st.markdown(level_row("Tenkan-sen (9)", tenkan))
                if kijun: st.markdown(level_row("Kijun-sen (26)", kijun))
                if spanA is not None: st.markdown(f"**Senkou Span A:** ${spanA:,.2f}")
                if spanB is not None: st.markdown(f"**Senkou Span B:** ${spanB:,.2f}")
            with ichi_cols[1]:
                if spanA is not None and spanB is not None:
                    cloud_color = "GREEN (bullish)" if spanA > spanB else "RED (bearish)"
                    st.markdown(f"**Cloud:** {cloud_color}")
                    if current > max(spanA, spanB): st.markdown("**Price vs Cloud:** ABOVE cloud - Bullish")
                    elif current < min(spanA, spanB): st.markdown("**Price vs Cloud:** BELOW cloud - Bearish")
                    else: st.markdown("**Price vs Cloud:** INSIDE cloud - Neutral/Transitioning")
                if tenkan and kijun:
                    tk = "Tenkan above Kijun (Bullish)" if tenkan > kijun else "Tenkan below Kijun (Bearish)"
                    st.markdown(f"**TK Cross:** {tk}")
                    # Ichimoku score
                    ichi_score = 0
                    if tenkan and tenkan > kijun: ichi_score += 1
                    if current > tenkan: ichi_score += 1
                    if current > kijun: ichi_score += 1
                    if spanA and spanB and current > max(spanA, spanB): ichi_score += 1
                    if spanA and spanB and spanA > spanB: ichi_score += 1
                    ichi_sig = "BULLISH" if ichi_score >= 3 else "BEARISH" if ichi_score <= 1 else "NEUTRAL"
                    st.markdown(f"**Ichimoku Signal:** {signal_badge(ichi_sig)} ({ichi_score}/5)", unsafe_allow_html=True)

            # ═══ MULTI-TIMEFRAME TREND ═══
            section("MULTI-TIMEFRAME TREND")
            for tf_name in ['Daily','Weekly','Monthly']:
                tf = mtf_t.get(tf_name, {})
                if tf.get('available', False):
                    trend = tf.get('trend','N/A')
                    mapped = trend.replace('UPTREND','BULLISH').replace('DOWNTREND','BEARISH').replace('SIDEWAYS','NEUTRAL')
                    st.markdown(f"**{tf_name}:** {signal_badge(mapped)} — {tf.get('detail','')}", unsafe_allow_html=True)
                else:
                    st.markdown(f"**{tf_name}:** N/A — {tf.get('detail', tf.get('reason','No data'))}")

            # ═══ MTF TECHNICAL EVALUATION ═══
            if mtf_e:
                section("TECHNICAL EVALUATION - MULTI-TIMEFRAME")
                eval_rows = []
                for tf_label in ['5 Min','15 Min','30 Min','4 Hours','Daily','Weekly','Monthly']:
                    ev = mtf_e.get(tf_label, {})
                    if ev.get('available'):
                        eval_rows.append({"Timeframe":tf_label,"Overall":ev.get('overall_summary','N/A'),"MA":ev.get('ma_summary','N/A'),"Oscillators":ev.get('osc_summary','N/A'),"Buy":ev.get('total_buy',0),"Neutral":ev.get('total_neutral',0),"Sell":ev.get('total_sell',0)})
                    else:
                        eval_rows.append({"Timeframe":tf_label,"Overall":"N/A","MA":"N/A","Oscillators":"N/A","Buy":"-","Neutral":"-","Sell":"-"})
                if eval_rows:
                    st.dataframe(pd.DataFrame(eval_rows), hide_index=True, use_container_width=True)

            # ═══ DAILY DETAIL ═══
            if bco and bco.get('overall_summary'):
                section("DAILY DETAIL")
                st.markdown(f"**Overall:** {signal_badge(bco['overall_summary'])} — Buy: {bco.get('total_buy',0)} / Neutral: {bco.get('total_neutral',0)} / Sell: {bco.get('total_sell',0)}", unsafe_allow_html=True)

                # Moving Averages
                section("MOVING AVERAGES (Daily)")
                st.markdown(f"**MA Summary:** {signal_badge(bco.get('ma_summary','N/A'))} — Buy: {bco.get('ma_buy',0)} / Neutral: {bco.get('ma_neutral',0)} / Sell: {bco.get('ma_sell',0)}", unsafe_allow_html=True)
                ma_results = bco.get('ma_results', [])
                if ma_results:
                    ma_rows = []
                    for m in ma_results:
                        ma_rows.append({"Indicator":m.get('name',''),"Value":f"${m.get('value',0):,.2f}" if m.get('value') else "N/A","Signal":m.get('signal','N/A'),"Distance":f"{m.get('pct_diff',0):+.2f}%"})
                    st.dataframe(pd.DataFrame(ma_rows), hide_index=True, use_container_width=True)

                # Oscillators
                section("OSCILLATORS (Daily)")
                st.markdown(f"**Oscillator Summary:** {signal_badge(bco.get('osc_summary','N/A'))} — Buy: {bco.get('osc_buy',0)} / Neutral: {bco.get('osc_neutral',0)} / Sell: {bco.get('osc_sell',0)}", unsafe_allow_html=True)
                osc_results = bco.get('osc_results', [])
                if osc_results:
                    osc_rows = []
                    for o in osc_results:
                        osc_rows.append({"Indicator":o.get('name',''),"Value":f"{o.get('value',0):.2f}" if o.get('value') is not None else "N/A","Signal":o.get('signal','N/A')})
                    st.dataframe(pd.DataFrame(osc_rows), hide_index=True, use_container_width=True)

            # ═══ PIVOT POINTS ═══
            if bco and bco.get('pivots'):
                section("PIVOT POINTS")
                pivots = bco['pivots']
                piv_rows = []
                for level in ['R3','R2','R1','PP','S1','S2','S3']:
                    row = {"Level": level}
                    for method in ['Classic','Fibonacci','Woodie','Camarilla']:
                        v = pivots.get(method, {}).get(level, 0)
                        pct = ((v-current)/current)*100 if current > 0 else 0
                        row[method] = f"${v:,.2f} ({pct:+.1f}%)"
                    piv_rows.append(row)
                st.dataframe(pd.DataFrame(piv_rows), hide_index=True, use_container_width=True)

            # ═══ FUNDAMENTALS ═══
            section("FUNDAMENTAL OVERVIEW")
            fc1,fc2 = st.columns(2)
            with fc1:
                for k in ['Company','Sector','Industry','Market Cap','P/E (Trailing)','P/E (Forward)','PEG Ratio','P/S Ratio','P/B Ratio','EV/EBITDA']:
                    v = fund.get(k, 'N/A')
                    if k == 'Market Cap': v = fmt(v, prefix="$")
                    elif isinstance(v, float) and v != 0: v = f"{v:.2f}"
                    st.markdown(f"**{k}:** {v}")
            with fc2:
                for k in ['Revenue','Revenue Growth','Profit Margin','Operating Margin','ROE','ROA','Debt/Equity','Current Ratio','Free Cash Flow','Dividend Yield','Beta','52W High','52W Low','Target Price','Analyst Rating','Num Analysts']:
                    v = fund.get(k, 'N/A')
                    if k in ('Revenue','Free Cash Flow'): v = fmt(v, prefix="$")
                    elif k in ('52W High','52W Low','Target Price'): v = fmt(v, prefix="$")
                    elif k in ('Revenue Growth','Profit Margin','Operating Margin','ROE','ROA','Dividend Yield'): v = fmt_pct(v)
                    elif isinstance(v, float) and v != 0: v = f"{v:.2f}"
                    st.markdown(f"**{k}:** {v}")

            st.markdown("---")
            st.caption(f"Analysis generated at {datetime.now().strftime('%Y-%m-%d %H:%M')} | Data: yfinance")
            st.caption("DISCLAIMER: This analysis is for informational purposes only. It does not constitute financial advice.")

elif not run:
    st.info("Enter a ticker in the sidebar, then click **Run Analysis**.")

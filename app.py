"""
Stock Analyzer Pro — Streamlit Web App
========================================
Renders the full analysis report directly in the browser.
Data source: Alpha Vantage (free tier: 25 req/day).
"""
import streamlit as st
import os, sys, io, tempfile, contextlib
from datetime import datetime
import pandas as pd

# ── Alpha Vantage API Key ────────────────────────────────────────────────────
if "AV_API_KEY" in st.secrets:
    os.environ["AV_API_KEY"] = st.secrets["AV_API_KEY"]

st.set_page_config(page_title="Stock Analyzer Pro", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);padding:1.5rem 2rem;border-radius:12px;margin-bottom:1.5rem;border:1px solid #30363d}
.main-header h1{color:#e0e0e0;margin:0;font-size:1.8rem}
.main-header p{color:#8b949e;margin:.3rem 0 0 0;font-size:.95rem}
.section-header{background:#161b22;border-left:4px solid #238636;padding:.6rem 1rem;border-radius:0 8px 8px 0;margin:1.2rem 0 .8rem 0;font-size:1.05rem;font-weight:700;color:#e0e0e0}
.metric-card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:.8rem 1rem;text-align:center}
.metric-card .label{color:#8b949e;font-size:.75rem;text-transform:uppercase;letter-spacing:.5px}
.metric-card .value{color:#e0e0e0;font-size:1.3rem;font-weight:700}
.badge-buy{background:#00e676;color:#1a1a2e;padding:.3rem .8rem;border-radius:6px;font-weight:700;font-size:.95rem;display:inline-block}
.badge-sell{background:#ff1744;color:#fff;padding:.3rem .8rem;border-radius:6px;font-weight:700;font-size:.95rem;display:inline-block}
.badge-neutral{background:#ffd740;color:#1a1a2e;padding:.3rem .8rem;border-radius:6px;font-weight:700;font-size:.95rem;display:inline-block}
.badge-halal{background:#00e676;color:#1a1a2e;padding:.3rem .8rem;border-radius:6px;font-weight:700}
.badge-haram{background:#ff1744;color:#fff;padding:.3rem .8rem;border-radius:6px;font-weight:700}
.badge-uncertain{background:#ffd740;color:#1a1a2e;padding:.3rem .8rem;border-radius:6px;font-weight:700}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header"><h1>📈 Stock Analyzer Pro</h1><p>Comprehensive Technical & Fundamental Analysis</p></div>', unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    ticker_input = st.text_input("Ticker Symbol(s)", value="AAPL", help="Comma-separated (e.g. AAPL, MSFT)", placeholder="AAPL")
    period = st.selectbox("Analysis Period", options=["3mo","6mo","1y","2y","5y"], index=0, help="Free tier: ~5 months max data")
    st.markdown("---")
    st.markdown("### 📋 Sections")
    st.markdown("Signal Summary · Halal Screening · Analyst Consensus · Price Summary · Technical Levels · Momentum · Keltner/Bollinger/Squeeze · MFI/OBV · Support & Resistance · Fibonacci · Chart Patterns · Divergences · Ichimoku · Multi-Timeframe · Fundamentals")
    st.markdown("---")
    st.caption("Free tier: ~2 stocks/day · Powered by Alpha Vantage")

# ── Helpers ───────────────────────────────────────────────────────────────────
def badge(text, kind="neutral"):
    cls = f"badge-{kind}"
    return f'<span class="{cls}">{text}</span>'

def signal_badge(sig):
    sig = str(sig).upper()
    if any(w in sig for w in ["BUY","BULLISH","UPTREND","PASS","HALAL"]):
        return badge(sig, "buy")
    elif any(w in sig for w in ["SELL","BEARISH","DOWNTREND","FAIL","NON-COMPLIANT","HARAM"]):
        return badge(sig, "sell")
    return badge(sig, "neutral")

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def fmt_num(val, prefix="", suffix="", decimals=2):
    if val is None or val == "N/A":
        return "N/A"
    try:
        v = float(val)
        if abs(v) >= 1e12: return f"{prefix}{v/1e12:.{decimals}f}T{suffix}"
        if abs(v) >= 1e9: return f"{prefix}{v/1e9:.{decimals}f}B{suffix}"
        if abs(v) >= 1e6: return f"{prefix}{v/1e6:.{decimals}f}M{suffix}"
        return f"{prefix}{v:,.{decimals}f}{suffix}"
    except (ValueError, TypeError):
        return str(val)

def fmt_pct(val):
    if val is None or val == "N/A": return "N/A"
    try: return f"{float(val)*100:.2f}%" if abs(float(val)) < 1 else f"{float(val):.2f}%"
    except: return str(val)

# ── Main ─────────────────────────────────────────────────────────────────────
if not os.environ.get("AV_API_KEY"):
    st.warning("⚠️ **Alpha Vantage API Key not found.** Add `AV_API_KEY` in Streamlit secrets.")

run = st.button("🚀  Run Analysis", use_container_width=True)

if run:
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    if not tickers:
        st.error("Enter at least one ticker.")
        st.stop()

    sys.path.insert(0, os.path.dirname(__file__))
    from stock_analyzer import analyze_stock_web

    for ticker in tickers:
        st.markdown("---")
        st.markdown(f"## 🔍 {ticker}")

        with tempfile.TemporaryDirectory() as tmpdir:
            log_buf = io.StringIO()
            progress = st.progress(0, text=f"Analyzing {ticker}…")

            try:
                with contextlib.redirect_stdout(log_buf):
                    R = analyze_stock_web(ticker, period=period, output_dir=tmpdir)
                progress.progress(100, text=f"✅ {ticker} — Done!")
            except Exception as e:
                progress.progress(100, text=f"❌ {ticker} — Error")
                st.error(f"Error: {e}")
                with st.expander("📜 Log"):
                    st.code(log_buf.getvalue())
                continue

            with st.expander("📜 Analysis Log"):
                st.code(log_buf.getvalue())

            df = R['df']
            info = R['info']
            signals = R['signals']
            latest = R['latest_data']
            sr = R['sr_levels']
            divs = R['divergences']
            cp = R['chart_patterns']
            ep = R['eccentric_patterns']
            fib = R['fib_result']
            mtf_fib = R['mtf_fib']
            halal = R['halal_result']
            mtf_t = R['mtf_trends']
            mtf_e = R['mtf_eval']
            fund = R['fundamentals']
            bco = R['barchart_opinion']

            # ── Charts ──
            if R.get('chart_path') and os.path.exists(R['chart_path']):
                st.image(R['chart_path'], use_container_width=True)
            if R.get('fib_chart_path') and os.path.exists(R['fib_chart_path']):
                st.image(R['fib_chart_path'], use_container_width=True)
            if R.get('fib_chart_weekly') and os.path.exists(R['fib_chart_weekly']):
                st.image(R['fib_chart_weekly'], use_container_width=True)
            if R.get('fib_chart_monthly') and os.path.exists(R['fib_chart_monthly']):
                st.image(R['fib_chart_monthly'], use_container_width=True)

            # ── Price Summary ──
            section("PRICE SUMMARY")
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("Close", f"${latest.get('Close',0):,.2f}")
            c2.metric("Volume", fmt_num(latest.get('Volume',0)))
            c3.metric("RSI", f"{latest.get('RSI',0):.1f}" if latest.get('RSI') else "N/A")
            c4.metric("MACD", f"{latest.get('MACD',0):.3f}" if latest.get('MACD') else "N/A")
            c5.metric("52W High", fmt_num(info.get('fiftyTwoWeekHigh'), prefix="$"))
            c6.metric("52W Low", fmt_num(info.get('fiftyTwoWeekLow'), prefix="$"))

            # ── Signal Summary ──
            section("SIGNAL SUMMARY")
            overall = signals.get('overall','NEUTRAL')
            score = signals.get('total_score', 0)
            trend_sig = signals.get('trend','NEUTRAL')
            trend_sc = signals.get('trend_score', 0)
            mom_sig = signals.get('momentum','NEUTRAL')
            mom_sc = signals.get('mom_score', 0)

            st.markdown(f"{signal_badge(overall)} &nbsp; Confluence Score: **{score}** (Trend {trend_sc} + Momentum {mom_sc})", unsafe_allow_html=True)
            st.caption("STRONG BUY ≥ 5 | BUY ≥ 2 | NEUTRAL -1 to 1 | SELL ≤ -2 | STRONG SELL ≤ -5")

            tc1,tc2,tc3 = st.columns(3)
            tc1.markdown(f"**Trend:** {signal_badge(trend_sig)} Score: {trend_sc}", unsafe_allow_html=True)
            tc2.markdown(f"**Momentum:** {signal_badge(mom_sig)} Score: {mom_sc}", unsafe_allow_html=True)
            tc3.markdown(f"**Volume:** {signals.get('volume','NORMAL')}")

            # Trend breakdown
            td = signals.get('trend_details', [])
            if td:
                with st.expander("Trend Breakdown"):
                    for desc, pts, is_bull in td:
                        icon = "🟢" if is_bull else "🔴" if is_bull is False else "⚪"
                        st.markdown(f"{icon} **{pts}** — {desc}")
            md = signals.get('mom_details', [])
            if md:
                with st.expander("Momentum Breakdown"):
                    for desc, pts, is_bull in md:
                        icon = "🟢" if is_bull else "🔴" if is_bull is False else "⚪"
                        st.markdown(f"{icon} **{pts}** — {desc}")

            # ── Halal Screening ──
            section("SHARIAH (HALAL) COMPLIANCE")
            h_status = halal.get('status','UNKNOWN')
            h_cls = "halal" if h_status == "HALAL" else "haram" if h_status == "NON-COMPLIANT" else "uncertain"
            st.markdown(f'<span class="badge-{h_cls}">{h_status}</span> &nbsp; ({halal.get("pass_count",0)}/{halal.get("total_checks",4)} criteria passed)', unsafe_allow_html=True)
            for d in halal.get('details', []):
                icon = "✅" if d['status']=='PASS' else "❌" if d['status']=='FAIL' else "⚠️"
                st.markdown(f"{icon} **{d['criterion']}** — {d['status']} — {d['value']}")
                if d.get('note'):
                    st.caption(f"   {d['note']}")

            # ── Analyst Consensus ──
            section("ANALYST CONSENSUS")
            ac1,ac2,ac3 = st.columns(3)
            ac1.metric("Target Price", fmt_num(info.get('targetMeanPrice'), prefix="$"))
            ac2.metric("Rating", str(info.get('recommendationKey','N/A')).replace('_',' ').title())
            ac3.metric("# Analysts", str(info.get('numberOfAnalystOpinions','N/A')))
            bd = info.get('_analyst_breakdown')
            if bd:
                st.markdown(f"🟢 Strong Buy: **{bd.get('strongBuy',0)}** · Buy: **{bd.get('buy',0)}** · ⚪ Hold: **{bd.get('hold',0)}** · 🔴 Sell: **{bd.get('sell',0)}** · Strong Sell: **{bd.get('strongSell',0)}**")

            # ── Key Technical Levels ──
            section("KEY TECHNICAL LEVELS")
            kc1,kc2,kc3,kc4 = st.columns(4)
            kc1.metric("SMA 20", f"${latest.get('SMA_20',0):,.2f}" if latest.get('SMA_20') else "N/A")
            kc2.metric("SMA 50", f"${latest.get('SMA_50',0):,.2f}" if latest.get('SMA_50') else "N/A")
            kc3.metric("EMA 12", f"${latest.get('EMA_12',0):,.2f}" if latest.get('EMA_12') else "N/A")
            kc4.metric("SMA 200", f"${latest.get('SMA_200',0):,.2f}" if latest.get('SMA_200') else "N/A")

            # ── Momentum Indicators ──
            section("MOMENTUM INDICATORS")
            mc1,mc2,mc3,mc4 = st.columns(4)
            mc1.metric("RSI (14)", f"{latest.get('RSI',0):.1f}" if latest.get('RSI') else "N/A")
            mc2.metric("MACD", f"{latest.get('MACD',0):.4f}" if latest.get('MACD') else "N/A")
            mc3.metric("Stoch RSI K", f"{latest.get('StochRSI_K',0):.1f}" if latest.get('StochRSI_K') else "N/A")
            mc4.metric("MFI", f"{latest.get('MFI',0):.1f}" if latest.get('MFI') else "N/A")

            # ── Keltner / Bollinger / Squeeze ──
            section("KELTNER / BOLLINGER / SQUEEZE")
            sq = "🔴 SQUEEZE ACTIVE" if latest.get('Squeeze') else "🟢 No Squeeze"
            bc1,bc2 = st.columns(2)
            bc1.markdown(f"**Squeeze:** {sq}")
            bc2.markdown(f"**BB Width:** {latest.get('BB_Width',0):.4f}" if latest.get('BB_Width') else "**BB Width:** N/A")

            # ── Support & Resistance ──
            section("SUPPORT & RESISTANCE")
            sc1,sc2 = st.columns(2)
            with sc1:
                st.markdown("**Support Levels**")
                for s in sr.get('support', []):
                    st.markdown(f"🟢 ${s:,.2f}")
            with sc2:
                st.markdown("**Resistance Levels**")
                for r_val in sr.get('resistance', []):
                    st.markdown(f"🔴 ${r_val:,.2f}")

            # ── Fibonacci ──
            section("FIBONACCI ANALYSIS")
            if fib.get('available'):
                st.markdown(f"**Trend:** {fib.get('primary_trend','N/A')} | **Swing High:** ${fib.get('swing_high',0):,.2f} | **Swing Low:** ${fib.get('swing_low',0):,.2f}")
                fib_levels = fib.get('levels', {})
                if fib_levels:
                    fib_df = pd.DataFrame([{"Level": k, "Price": f"${v:,.2f}"} for k,v in fib_levels.items()])
                    st.dataframe(fib_df, hide_index=True, use_container_width=True)
            else:
                st.info(f"Fibonacci not available: {fib.get('reason','No data')}")

            # MTF Fibonacci
            for tf in ['Weekly', 'Monthly']:
                tf_fib = mtf_fib.get(tf, {})
                if tf_fib.get('available'):
                    with st.expander(f"Fibonacci — {tf}"):
                        st.markdown(f"**Trend:** {tf_fib.get('primary_trend','N/A')}")
                        levels = tf_fib.get('levels', {})
                        if levels:
                            fdf = pd.DataFrame([{"Level": k, "Price": f"${v:,.2f}"} for k,v in levels.items()])
                            st.dataframe(fdf, hide_index=True, use_container_width=True)

            # ── Chart Patterns ──
            section("CHART PATTERNS")
            if cp:
                for p in cp:
                    icon = "📈" if p.get('direction') == 'BULLISH' else "📉" if p.get('direction') == 'BEARISH' else "📊"
                    st.markdown(f"{icon} **{p.get('pattern','Unknown')}** — {p.get('direction','N/A')} — Confidence: {p.get('confidence','N/A')}")
                    if p.get('description'):
                        st.caption(p['description'])
            else:
                st.info("No chart patterns detected in the current data.")

            if ep:
                with st.expander("Chart Patterns — theEccentricTrader Method"):
                    for p in ep:
                        st.markdown(f"**{p.get('pattern','Unknown')}** — {p.get('direction','N/A')}")
                        if p.get('description'):
                            st.caption(p['description'])

            # ── Divergences ──
            section("DIVERGENCE ANALYSIS")
            if divs:
                for d in divs:
                    icon = "🟢" if d.get('type') == 'BULLISH' else "🔴"
                    st.markdown(f"{icon} **{d.get('type','?')} {d.get('indicator','?')}** divergence detected — {d.get('description','')}")
            else:
                st.info("No divergences detected.")

            # ── Multi-Timeframe Trend ──
            section("MULTI-TIMEFRAME TREND")
            for tf_name in ['Daily','Weekly','Monthly']:
                tf = mtf_t.get(tf_name, {})
                if tf.get('available', False):
                    trend = tf.get('trend','N/A')
                    st.markdown(f"**{tf_name}:** {signal_badge(trend.replace('UPTREND','BULLISH').replace('DOWNTREND','BEARISH').replace('SIDEWAYS','NEUTRAL'))} — {tf.get('detail','')}", unsafe_allow_html=True)
                else:
                    st.markdown(f"**{tf_name}:** N/A")

            # ── Multi-Timeframe Technical Evaluation ──
            if mtf_e:
                section("TECHNICAL EVALUATION — MULTI-TIMEFRAME")
                eval_rows = []
                for tf_label in ['5 Min','15 Min','30 Min','4 Hours','Daily','Weekly','Monthly']:
                    ev = mtf_e.get(tf_label, {})
                    if ev.get('available'):
                        eval_rows.append({
                            "Timeframe": tf_label,
                            "Overall": ev.get('overall_summary','N/A'),
                            "MA": ev.get('ma_summary','N/A'),
                            "Oscillators": ev.get('osc_summary','N/A'),
                            "Buy": ev.get('total_buy',0),
                            "Neutral": ev.get('total_neutral',0),
                            "Sell": ev.get('total_sell',0),
                        })
                if eval_rows:
                    st.dataframe(pd.DataFrame(eval_rows), hide_index=True, use_container_width=True)

            # ── Fundamental Overview ──
            section("FUNDAMENTAL OVERVIEW")
            fc1,fc2 = st.columns(2)
            left_keys = ['Company','Sector','Industry','Market Cap','P/E (Trailing)','P/E (Forward)','PEG Ratio','P/S Ratio','P/B Ratio','EV/EBITDA']
            right_keys = ['Revenue','Revenue Growth','Profit Margin','Operating Margin','ROE','ROA','Debt/Equity','Current Ratio','Free Cash Flow','Dividend Yield','Beta','Target Price','Analyst Rating','Num Analysts']
            with fc1:
                for k in left_keys:
                    v = fund.get(k, 'N/A')
                    if k == 'Market Cap': v = fmt_num(v, prefix="$")
                    elif k == 'Revenue': v = fmt_num(v, prefix="$")
                    elif 'Growth' in k or 'Margin' in k or 'Yield' in k: v = fmt_pct(v)
                    st.markdown(f"**{k}:** {v}")
            with fc2:
                for k in right_keys:
                    v = fund.get(k, 'N/A')
                    if k == 'Revenue': v = fmt_num(v, prefix="$")
                    elif k == 'Free Cash Flow': v = fmt_num(v, prefix="$")
                    elif 'Growth' in k or 'Margin' in k or 'Yield' in k or k in ('ROE','ROA'): v = fmt_pct(v)
                    elif k == 'Target Price': v = fmt_num(v, prefix="$")
                    st.markdown(f"**{k}:** {v}")

            st.markdown("---")
            st.caption(f"Analysis generated at {datetime.now().strftime('%Y-%m-%d %H:%M')} · Data from Alpha Vantage")

elif not run:
    st.info("👈 Enter a ticker in the sidebar, then click **Run Analysis**.")
    c1,c2,c3 = st.columns(3)
    c1.markdown("#### 📊 Technical\n20+ indicators, patterns, divergences, Fibonacci")
    c2.markdown("#### 🔬 Patterns\nH&S, Wedges, Triangles, Harmonics, Elliott")
    c3.markdown("#### 📑 Live Report\nAll sections rendered directly in the browser")

"""
Stock Analyzer — Streamlit App
================================
Web interface for the comprehensive stock analysis tool.
Deploy on Streamlit Community Cloud via GitHub.
Data source: Financial Modeling Prep (FMP) — free tier: 250 requests/day.
"""

import streamlit as st
import os
import sys
import io
import tempfile
import contextlib
from datetime import datetime

# ── Alpha Vantage API Key: load from Streamlit secrets or environment ────────
if "AV_API_KEY" in st.secrets:
    os.environ["AV_API_KEY"] = st.secrets["AV_API_KEY"]

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container */
    .stApp {
        background-color: #0e1117;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #30363d;
    }
    .main-header h1 {
        color: #e0e0e0;
        margin: 0;
        font-size: 1.8rem;
    }
    .main-header p {
        color: #8b949e;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }
    /* Status cards */
    .status-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    /* Log area */
    .log-area {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.82rem;
        color: #8b949e;
        max-height: 400px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%);
        box-shadow: 0 0 15px rgba(46, 160, 67, 0.3);
    }
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        width: 100%;
    }
    /* Info/warning boxes */
    .stAlert {
        border-radius: 8px;
    }
    /* Text input */
    .stTextInput input {
        background-color: #0d1117;
        border: 1px solid #30363d;
        color: #e0e0e0;
        border-radius: 6px;
    }
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #0d1117;
        border-color: #30363d;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📈 Stock Analyzer Pro</h1>
    <p>Comprehensive Technical & Fundamental Analysis — with PDF reports</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    ticker_input = st.text_input(
        "Ticker Symbol(s)",
        value="AAPL",
        help="Enter one or more tickers separated by commas (e.g. AAPL, MSFT, NVDA)",
        placeholder="AAPL, MSFT, NVDA",
    )

    period = st.selectbox(
        "Analysis Period",
        options=["3mo", "6mo", "1y", "2y", "5y"],
        index=2,
        help="How far back to look for price history",
    )

    st.markdown("---")
    st.markdown("### 📋 What you get")
    st.markdown("""
    - **Technical indicators** — SMA, EMA, MACD, RSI, Bollinger, Keltner, Squeeze
    - **Chart patterns** — Head & Shoulders, Wedges, Triangles, Channels, etc.
    - **Fibonacci analysis** — Daily, Weekly, Monthly retracements
    - **Support / Resistance** — Auto-detected levels
    - **Divergences** — RSI, MACD, OBV, MFI
    - **Multi-timeframe** — Daily, Weekly, Monthly trend consensus
    - **Fundamental snapshot** — Valuation, profitability, analyst targets
    - **Halal screening** — Shariah compliance check
    - **Full PDF report** — Downloadable, dark-themed, chart-embedded
    """)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#8b949e; font-size:0.8rem;'>"
        "Powered by FMP · ta · matplotlib · reportlab"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

# Check for Alpha Vantage API key
if not os.environ.get("AV_API_KEY"):
    st.warning(
        "⚠️ **Alpha Vantage API Key not found.** Add your free API key to run analysis.\n\n"
        "1. Get a free key at [alphavantage.co](https://www.alphavantage.co/support/#api-key)\n"
        "2. In Streamlit Cloud: click **Manage app** → **Settings** → **Secrets** → add:\n"
        "```\nAV_API_KEY = \"your_key_here\"\n```\n"
        "3. Click **Save** — the app will restart automatically.\n\n"
        "**Free tier:** 25 requests/day (~1-2 full stock analyses per day)"
    )
run_button = st.button("🚀  Run Analysis", use_container_width=True)

if run_button:
    # Parse tickers
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    if not tickers:
        st.error("Please enter at least one ticker symbol.")
        st.stop()

    # Import the analyzer
    sys.path.insert(0, os.path.dirname(__file__))
    from stock_analyzer import analyze_stock  # noqa: E402

    for ticker in tickers:
        st.markdown(f"---")
        st.markdown(f"## 🔍  {ticker}")

        # Create a temp dir for this ticker's output
        with tempfile.TemporaryDirectory() as tmpdir:
            # Capture console output
            log_buffer = io.StringIO()
            progress_bar = st.progress(0, text=f"Analyzing {ticker}…")
            log_placeholder = st.empty()

            try:
                with contextlib.redirect_stdout(log_buffer):
                    result = analyze_stock(ticker, period=period, output_dir=tmpdir)

                progress_bar.progress(100, text=f"✅ {ticker} — Done!")

                # Show console log in expander
                with st.expander("📜 Analysis Log", expanded=False):
                    st.markdown(
                        f"<div class='log-area'>{log_buffer.getvalue()}</div>",
                        unsafe_allow_html=True,
                    )

                # Locate the PDF
                report_path = result.get("report")
                if report_path and os.path.exists(report_path):
                    # Display latest price info
                    latest = result.get("latest_data", {})
                    signals = result.get("signals", {})

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        close_price = latest.get("Close", 0)
                        st.metric("Close", f"${close_price:,.2f}")
                    with col2:
                        vol = latest.get("Volume", 0)
                        st.metric("Volume", f"{vol:,.0f}")
                    with col3:
                        trend = signals.get("trend", "N/A")
                        st.metric("Trend", trend)
                    with col4:
                        rsi = signals.get("rsi_value", 0)
                        st.metric("RSI", f"{rsi:.1f}")

                    # Offer PDF download
                    with open(report_path, "rb") as f:
                        pdf_bytes = f.read()

                    st.download_button(
                        label=f"📥  Download {ticker} Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

                    # Show charts from tmpdir if any PNGs were left around
                    for fname in sorted(os.listdir(tmpdir)):
                        if fname.lower().endswith(".png"):
                            st.image(
                                os.path.join(tmpdir, fname),
                                caption=fname.replace("_", " ").replace(".png", ""),
                                use_container_width=True,
                            )
                else:
                    st.warning(f"Analysis ran but no PDF was generated for {ticker}.")

            except Exception as e:
                progress_bar.progress(100, text=f"❌ {ticker} — Error")
                st.error(f"Error analyzing **{ticker}**: {e}")
                with st.expander("📜 Log before error", expanded=True):
                    st.markdown(
                        f"<div class='log-area'>{log_buffer.getvalue()}</div>",
                        unsafe_allow_html=True,
                    )

elif not run_button:
    # Landing state
    st.info("👈 Enter one or more tickers in the sidebar, then click **Run Analysis**.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="status-card">
            <h4>📊 Technical</h4>
            <p style="color:#8b949e; font-size:0.9rem;">
            20+ indicators including MACD, RSI, Bollinger Bands, Keltner Channels, TTM Squeeze, Volume Profile
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="status-card">
            <h4>🔬 Patterns</h4>
            <p style="color:#8b949e; font-size:0.9rem;">
            Head & Shoulders, Wedges, Triangles, Double Top/Bottom, Harmonic Patterns, Elliott Waves
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="status-card">
            <h4>📑 Reports</h4>
            <p style="color:#8b949e; font-size:0.9rem;">
            Dark-themed PDF with embedded charts, Fibonacci levels, multi-timeframe analysis, and Halal screening
            </p>
        </div>
        """, unsafe_allow_html=True)

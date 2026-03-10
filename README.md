# 📈 Stock Analyzer Pro

A comprehensive **technical & fundamental stock analysis** tool with a **Streamlit** web interface and **PDF report generation**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?logo=streamlit&logoColor=white)

---

## Features

| Category | Details |
|----------|---------|
| **Technical Indicators** | SMA, EMA, MACD, RSI, Stochastic, Bollinger Bands, Keltner Channels, TTM Squeeze, Volume Profile, OBV, MFI |
| **Chart Patterns** | Head & Shoulders, Double Top/Bottom, Wedges, Triangles, Channels, Cup & Handle |
| **Harmonic Patterns** | Gartley, Butterfly, Bat, Crab, Shark |
| **Fibonacci Analysis** | Daily / Weekly / Monthly retracements with auto-detected swing points |
| **Support & Resistance** | Algorithmically detected price levels |
| **Divergences** | RSI, MACD, OBV, MFI divergence detection |
| **Multi-Timeframe** | Daily → Weekly → Monthly trend consensus |
| **Fundamentals** | Valuation ratios, profitability, analyst targets |
| **Halal Screening** | Shariah compliance check (debt, interest, revenue filters) |
| **PDF Reports** | Dark-themed, chart-embedded, downloadable reports |

---

## Quick Start

### Run Locally

```bash
# Clone
git clone https://github.com/<YOUR_USERNAME>/stock-analyzer-app.git
cd stock-analyzer-app

# Install
pip install -r requirements.txt

# Launch
streamlit run app.py
```

### Deploy on Streamlit Community Cloud

1. **Push to GitHub** — push this repo to your GitHub account
2. **Go to** [share.streamlit.io](https://share.streamlit.io)
3. **New app** → select your repo → set **Main file** to `app.py`
4. **Deploy** — it will install requirements and launch automatically

---

## Project Structure

```
stock-analyzer-app/
├── app.py                  # Streamlit web interface
├── stock_analyzer.py       # Core analysis engine (4500+ lines)
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Dark theme & server config
├── .gitignore
└── README.md
```

---

## CLI Usage

You can also run the analyzer directly from the command line:

```bash
python stock_analyzer.py AAPL
python stock_analyzer.py AAPL --period 1y
python stock_analyzer.py AAPL MSFT NVDA
```

This generates a PDF report in the current directory.

---

## License

MIT

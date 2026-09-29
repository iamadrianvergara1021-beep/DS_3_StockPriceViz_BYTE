import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Stock Price & Volume Anomaly Detector",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
ACCENT = "#4f8cff"
ACCENT_SOFT = "#9db8ff"
GOOD = "#2ecc9c"
BAD = "#ff5d73"
BG = "#0d1117"
CARD = "#141a24"
BORDER = "#232b38"
TEXT_MAIN = "#e8ecf1"
TEXT_MUTED = "#8a94a6"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .block-container {{
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }}

    /* Header */
    .dash-title {{
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: {TEXT_MAIN};
        margin-bottom: 0.15rem;
    }}
    .dash-subtitle {{
        color: {TEXT_MUTED};
        font-size: 0.92rem;
        font-weight: 500;
        margin-bottom: 1.6rem;
    }}
    .pill {{
        display: inline-block;
        background: rgba(79, 140, 255, 0.12);
        color: {ACCENT_SOFT};
        border: 1px solid rgba(79, 140, 255, 0.35);
        border-radius: 999px;
        padding: 2px 11px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background: {CARD};
        border: 1px solid {BORDER};
        padding: 18px 20px 14px 20px;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.28);
    }}
    div[data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED} !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    div[data-testid="stMetricValue"] {{
        color: {TEXT_MAIN} !important;
        font-weight: 700 !important;
    }}

    /* Section headers */
    h3 {{
        color: {TEXT_MAIN} !important;
        font-weight: 650 !important;
        letter-spacing: -0.01em;
        border-left: 3px solid {ACCENT};
        padding-left: 10px;
        margin-top: 1.6rem !important;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {CARD};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] h3 {{
        border-left: none;
        padding-left: 0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {TEXT_MUTED} !important;
    }}

    /* Dataframe */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 12px;
        overflow: hidden;
    }}

    hr {{ border-color: {BORDER} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": CARD,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": TEXT_MAIN,
    "text.color": TEXT_MAIN,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "grid.color": BORDER,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
})

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="dash-title">📈 Stock Price & Volume Anomaly Terminal</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dash-subtitle">'
    '<span class="pill">LIVE MARKET DATA</span>'
    '<span class="pill">DUAL MOVING AVERAGES</span>'
    '<span class="pill">VOLUME SPIKE DETECTION</span>'
    '<span class="pill">RETURN DISTRIBUTION</span>'
    '&nbsp; Powered by Yahoo Finance via yfinance'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Asset & Horizon")
ticker = st.sidebar.selectbox("Ticker Symbol", ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN"], index=0)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

st.sidebar.divider()
st.sidebar.markdown("### Model Parameters")
sma_short = st.sidebar.slider("Short Moving Average (Days)", 5, 40, 20)
sma_long = st.sidebar.slider("Long Moving Average (Days)", 30, 100, 50)
vol_spike_threshold = st.sidebar.slider("Volume Spike Threshold (% Above Avg)", 100, 800, 500, 50)


# ---------------------------------------------------------------------------
# Real data fetch (replaces the old random-seed generator)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Fetching live market data...")
def load_data(ticker_sym, start_dt, end_dt, short_w, long_w, vol_thresh):
    raw = yf.download(ticker_sym, start=start_dt, end=end_dt, progress=False)
    if raw.empty:
        return pd.DataFrame()

    df = raw.reset_index()[["Date", "Close", "Volume"]].copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df["Daily_Return_Pct"] = (df["Close"].pct_change().fillna(0) * 100).round(2)
    df[f"SMA_{short_w}"] = df["Close"].rolling(window=short_w, min_periods=1).mean().round(2)
    df[f"SMA_{long_w}"] = df["Close"].rolling(window=long_w, min_periods=1).mean().round(2)

    avg_vol = df["Volume"].mean()
    df["Avg_Volume"] = int(avg_vol)
    df["Volume_Spike_Pct"] = (((df["Volume"] - avg_vol) / avg_vol) * 100).round(1)
    df["Is_Volume_Anomaly"] = df["Volume_Spike_Pct"] >= vol_thresh
    return df


df = load_data(ticker, start_date, end_date, sma_short, sma_long, vol_spike_threshold)

if df.empty:
    st.error("No data returned — check the ticker and date range (markets are closed on weekends/holidays).")
else:
    anomalies = df[df["Is_Volume_Anomaly"]]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Selected Asset", ticker)
    c2.metric("Latest Close", f"${df['Close'].iloc[-1]:.2f}")
    c3.metric(f"{sma_short}-Day SMA", f"${df[f'SMA_{sma_short}'].iloc[-1]:.2f}")
    c4.metric(f"{sma_long}-Day SMA", f"${df[f'SMA_{sma_long}'].iloc[-1]:.2f}")
    c5.metric(f"Volume Anomalies (>={vol_spike_threshold}%)", f"{len(anomalies)} Days")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"### Price Trajectory & Moving Average Crossovers ({sma_short}D vs {sma_long}D)")
    fig_price, ax_price = plt.subplots(figsize=(14, 4.2))
    ax_price.plot(df["Date"], df["Close"], color=ACCENT, linewidth=1.8, label="Close Price ($)", alpha=0.95)
    ax_price.plot(df["Date"], df[f"SMA_{sma_short}"], color="#f0883e", linewidth=1.6, linestyle="--", label=f"{sma_short}-Day SMA")
    ax_price.plot(df["Date"], df[f"SMA_{sma_long}"], color=GOOD, linewidth=1.6, linestyle="-.", label=f"{sma_long}-Day SMA")
    ax_price.scatter(anomalies["Date"], anomalies["Close"], color=BAD, s=70, zorder=5,
                      edgecolors="white", linewidth=0.7, label=f"Volume Spike (>={vol_spike_threshold}%)")
    ax_price.set_ylabel("Price (USD)")
    ax_price.grid(True)
    ax_price.spines["top"].set_visible(False)
    ax_price.spines["right"].set_visible(False)
    ax_price.legend(frameon=True, facecolor=CARD, edgecolor=BORDER, loc="upper left")
    plt.tight_layout()
    st.pyplot(fig_price)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f"### Trading Volume Spike Detection (>={vol_spike_threshold}% Above Avg)")
        fig1, ax1 = plt.subplots(figsize=(7, 4.0))
        bar_colors = [BAD if a else BORDER for a in df["Is_Volume_Anomaly"]]
        ax1.bar(df["Date"], df["Volume"] / 1e6, color=bar_colors, width=1.6)
        avg_v = df["Avg_Volume"].iloc[0] / 1e6
        thresh_v = avg_v * (1 + vol_spike_threshold / 100.0)
        ax1.axhline(avg_v, color=ACCENT, linewidth=1.4, label=f"Baseline Avg ({avg_v:.1f}M)")
        ax1.axhline(thresh_v, color=BAD, linestyle="--", linewidth=1.4, label=f"+{vol_spike_threshold}% Spike Threshold")
        ax1.set_ylabel("Volume (Millions)")
        ax1.set_xlabel("Date")
        ax1.grid(True)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.legend(frameon=True, facecolor=CARD, edgecolor=BORDER, loc="upper left")
        plt.tight_layout()
        st.pyplot(fig1)

    with col_right:
        st.markdown("### Daily Percentage Returns Distribution")
        fig2, ax2 = plt.subplots(figsize=(7, 4.0))
        sns.histplot(df["Daily_Return_Pct"], bins=28, kde=True, color=ACCENT,
                     edgecolor=CARD, alpha=0.8, ax=ax2)
        ax2.axvline(df["Daily_Return_Pct"].mean(), color=BAD, linestyle="--", linewidth=1.5,
                    label=f"Mean Return ({df['Daily_Return_Pct'].mean():.2f}%)")
        ax2.set_xlabel("Daily Return (%)")
        ax2.set_ylabel("Frequency (Trading Days)")
        ax2.grid(True)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.legend(frameon=True, facecolor=CARD, edgecolor=BORDER, loc="upper right")
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### Flagged Volume Anomalies (Volume Spiked >= {vol_spike_threshold}% Above Average)")
    display_df = anomalies[["Date", "Close", "Volume", "Avg_Volume", "Volume_Spike_Pct", "Daily_Return_Pct"]].copy()
    display_df["Date"] = pd.to_datetime(display_df["Date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📝 Interpretation (for README / notebook)"):
        vol = df["Daily_Return_Pct"].std()
        trend = "an uptrend" if df["Close"].iloc[-1] > df["Close"].iloc[0] else "a downtrend"
        st.write(
            f"Over the selected window, **{ticker}** moved through {trend}, with a daily return "
            f"standard deviation (volatility) of **{vol:.2f}%**. The detector flagged "
            f"**{len(anomalies)} day(s)** where volume spiked {vol_spike_threshold}%+ above its "
            f"trailing average — worth cross-checking against that stock's news calendar for "
            f"earnings releases or major announcements around those dates."
        )
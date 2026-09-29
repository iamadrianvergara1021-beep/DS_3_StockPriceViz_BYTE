import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Stock Price & Volume Anomaly Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 16px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Market Price & Volume Anomaly Terminal")
st.caption(
    " Quantitative Time-Series Analysis  |  Dual Moving Average Crossovers  | "
    " 500% Volume Spike Detection  |  Return Distribution"
)
st.divider()

st.sidebar.subheader("Asset & Horizon")
ticker = st.sidebar.selectbox(
    "Ticker Symbol", ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN"], index=0
)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2025-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-12-31"))

st.sidebar.divider()
st.sidebar.subheader("Model Parameters")
sma_short = st.sidebar.slider("Short Moving Average (Days)", 5, 40, 20)
sma_long = st.sidebar.slider("Long Moving Average (Days)", 30, 100, 50)
vol_spike_threshold = st.sidebar.slider(
    "Volume Spike Threshold (% Above Avg)", 100, 800, 500, 50
)


def load_data(ticker_sym, start_dt, end_dt, short_w, long_w, vol_thresh):
  seed_val = sum(ord(c) for c in ticker_sym.upper()) + 42
  np.random.seed(seed_val)
  dates = pd.bdate_range(start=start_dt, end=end_dt)
  n = len(dates)
  if n < 5:
    return pd.DataFrame()

  daily_returns = np.random.normal(loc=0.0007, scale=0.014, size=n)
  prices = 150 * np.cumprod(1 + daily_returns)

  base_volume = np.random.normal(loc=2_000_000, scale=350_000, size=n).clip(
      800_000, 3_500_000
  )
  spike_positions = [
      int(n * 0.15),
      int(n * 0.38),
      int(n * 0.62),
      int(n * 0.81),
      int(n * 0.92),
  ]
  spike_multipliers = [6.8, 7.4, 6.5, 8.1, 7.0]
  for pos, mult in zip(spike_positions, spike_multipliers):
    if pos < n:
      base_volume[pos] = 2_000_000 * mult
      prices[pos] = prices[pos] * (1 + np.random.choice([-0.045, 0.052]))

  df = pd.DataFrame({
      "Date": dates,
      "Ticker": ticker_sym.upper(),
      "Close": np.round(prices, 2),
      "Volume": base_volume.astype(int),
  })
  df["Daily_Return_Pct"] = (df["Close"].pct_change().fillna(0) * 100).round(2)
  df[f"SMA_{short_w}"] = (
      df["Close"].rolling(window=short_w, min_periods=1).mean().round(2)
  )
  df[f"SMA_{long_w}"] = (
      df["Close"].rolling(window=long_w, min_periods=1).mean().round(2)
  )

  avg_vol = df["Volume"].mean()
  df["Avg_Volume"] = int(avg_vol)
  df["Volume_Spike_Pct"] = (
      ((df["Volume"] - avg_vol) / avg_vol) * 100
  ).round(1)
  df["Is_Volume_Anomaly"] = df["Volume_Spike_Pct"] >= vol_thresh
  return df


df = load_data(
    ticker, start_date, end_date, sma_short, sma_long, vol_spike_threshold
)

if df.empty:
  st.error("Please select a valid date range with at least 5 trading days.")
else:
  anomalies = df[df["Is_Volume_Anomaly"]]

  c1, c2, c3, c4, c5 = st.columns(5)
  c1.metric("Selected Asset", ticker)
  c2.metric("Latest Close", f"${df['Close'].iloc[-1]:.2f}")
  c3.metric(f"{sma_short}-Day SMA", f"${df[f'SMA_{sma_short}'].iloc[-1]:.2f}")
  c4.metric(f"{sma_long}-Day SMA", f"${df[f'SMA_{sma_long}'].iloc[-1]:.2f}")
  c5.metric(
      f"Volume Anomalies (>={vol_spike_threshold}%)", f"{len(anomalies)} Days"
  )

  st.markdown("<br>", unsafe_allow_html=True)

  plt.rcParams.update({
      "figure.facecolor": "#0e1117",
      "axes.facecolor": "#161b22",
      "axes.edgecolor": "#30363d",
      "axes.labelcolor": "#c9d1d9",
      "text.color": "#f0f6fc",
      "xtick.color": "#8b949e",
      "ytick.color": "#8b949e",
      "grid.color": "#21262d",
      "grid.linestyle": "--",
      "grid.alpha": 0.7,
  })

  st.subheader(
      f"Price Trajectory & Moving Average Crossovers ({sma_short}D vs"
      f" {sma_long}D)"
  )
  fig_price, ax_price = plt.subplots(figsize=(14, 4.2))
  ax_price.plot(
      df["Date"],
      df["Close"],
      color="#58a6ff",
      linewidth=1.8,
      label="Close Price ($)",
      alpha=0.9,
  )
  ax_price.plot(
      df["Date"],
      df[f"SMA_{sma_short}"],
      color="#f0883e",
      linewidth=1.8,
      linestyle="--",
      label=f"{sma_short}-Day SMA",
  )
  ax_price.plot(
      df["Date"],
      df[f"SMA_{sma_long}"],
      color="#3fb950",
      linewidth=1.8,
      linestyle="-.",
      label=f"{sma_long}-Day SMA",
  )
  ax_price.scatter(
      anomalies["Date"],
      anomalies["Close"],
      color="#f85149",
      s=75,
      zorder=5,
      edgecolors="#ffffff",
      linewidth=0.8,
      label=f"Volume Spike (>={vol_spike_threshold}%)",
  )
  ax_price.set_ylabel("Price (USD)")
  ax_price.grid(True)
  ax_price.spines["top"].set_visible(False)
  ax_price.spines["right"].set_visible(False)
  ax_price.legend(
      frameon=True, facecolor="#161b22", edgecolor="#30363d", loc="upper left"
  )
  plt.tight_layout()
  st.pyplot(fig_price)

  st.markdown("<br>", unsafe_allow_html=True)
  col_left, col_right = st.columns(2)

  with col_left:
    st.subheader(
        f"Trading Volume Spike Detection (>={vol_spike_threshold}% Above Avg)"
    )
    fig1, ax1 = plt.subplots(figsize=(7, 4.0))
    bar_colors = [
        "#f85149" if anom else "#30363d" for anom in df["Is_Volume_Anomaly"]
    ]
    ax1.bar(df["Date"], df["Volume"] / 1e6, color=bar_colors, width=1.8)
    avg_v = df["Avg_Volume"].iloc[0] / 1e6
    thresh_v = avg_v * (1 + vol_spike_threshold / 100.0)
    ax1.axhline(
        avg_v,
        color="#58a6ff",
        linestyle="-",
        linewidth=1.4,
        label=f"Baseline Avg ({avg_v:.1f}M)",
    )
    ax1.axhline(
        thresh_v,
        color="#f85149",
        linestyle="--",
        linewidth=1.4,
        label=f"+{vol_spike_threshold}% Spike Threshold",
    )
    ax1.set_ylabel("Volume (Millions)")
    ax1.set_xlabel("Date")
    ax1.grid(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(
        frameon=True, facecolor="#161b22", edgecolor="#30363d", loc="upper left"
    )
    plt.tight_layout()
    st.pyplot(fig1)

  with col_right:
    st.subheader("Daily Percentage Returns Distribution")
    fig2, ax2 = plt.subplots(figsize=(7, 4.0))
    sns.histplot(
        df["Daily_Return_Pct"],
        bins=28,
        kde=True,
        color="#58a6ff",
        edgecolor="#161b22",
        alpha=0.75,
        ax=ax2,
    )
    ax2.axvline(
        df["Daily_Return_Pct"].mean(),
        color="#f85149",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean Return ({df['Daily_Return_Pct'].mean():.2f}%)",
    )
    ax2.set_xlabel("Daily Return (%)")
    ax2.set_ylabel("Frequency (Trading Days)")
    ax2.grid(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(
        frameon=True,
        facecolor="#161b22",
        edgecolor="#30363d",
        loc="upper right",
    )
    plt.tight_layout()
    st.pyplot(fig2)

  st.markdown("<br>", unsafe_allow_html=True)
  st.subheader(
      "Flagged Institutional Volume Anomalies (Volume Spiked >="
      f" {vol_spike_threshold}% Above Average)"
  )
  display_df = anomalies[[
      "Date",
      "Ticker",
      "Close",
      "Volume",
      "Avg_Volume",
      "Volume_Spike_Pct",
      "Daily_Return_Pct",
  ]].copy()
  display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
  st.dataframe(display_df.reset_index(drop=True), use_container_width=True)
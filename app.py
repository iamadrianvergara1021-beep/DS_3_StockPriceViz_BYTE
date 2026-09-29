import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Stock Price & Volume Anomaly Detector",
    page_icon="📈",
    layout="wide",
)
sns.set_theme(style="darkgrid")

st.title("📈 Stock Price Visualization & 500% Volume Anomaly Detector")
st.write(
    "Interactive time-series analysis featuring **20-Day & 50-Day Moving"
    " Averages**, **Daily Returns Histogram**, and **Trading Volume Spike"
    " Detection (>500% Above Average)**."
)

st.sidebar.header("1. Ticker & Date Range")
ticker = st.sidebar.selectbox(
    "Select Stock Ticker", ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN"], index=0
)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2025-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-12-31"))

st.sidebar.header("2. Moving Average & Anomaly Controls")
sma_short = st.sidebar.slider("Short Moving Average (Days)", 5, 40, 20)
sma_long = st.sidebar.slider("Long Moving Average (Days)", 30, 100, 50)
vol_spike_threshold = st.sidebar.slider(
    "Volume Spike Threshold (% Above Average)", 100, 800, 500, 50
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
  c1.metric("Ticker", ticker)
  c2.metric("Latest Close", f"${df['Close'].iloc[-1]:.2f}")
  c3.metric(f"{sma_short}-Day SMA", f"${df[f'SMA_{sma_short}'].iloc[-1]:.2f}")
  c4.metric(f"{sma_long}-Day SMA", f"${df[f'SMA_{sma_long}'].iloc[-1]:.2f}")
  c5.metric(
      f"Volume Spikes (>={vol_spike_threshold}%)", f"{len(anomalies)} Days"
  )

  st.subheader(
      f"1. {ticker} Price Trend with {sma_short}-Day & {sma_long}-Day Moving"
      " Averages"
  )
  chart_cols = ["Close", f"SMA_{sma_short}", f"SMA_{sma_long}"]
  st.line_chart(df.set_index("Date")[chart_cols], height=320)

  col_left, col_right = st.columns(2)

  with col_left:
    st.subheader(
        f"2. Volume Anomaly Detector (>={vol_spike_threshold}% Above Avg)"
    )
    fig1, ax1 = plt.subplots(figsize=(7, 3.8))
    bar_colors = [
        "#ef4444" if anom else "#64748b" for anom in df["Is_Volume_Anomaly"]
    ]
    ax1.bar(df["Date"], df["Volume"] / 1e6, color=bar_colors, width=1.6)
    avg_v = df["Avg_Volume"].iloc[0] / 1e6
    thresh_v = avg_v * (1 + vol_spike_threshold / 100.0)
    ax1.axhline(
        avg_v,
        color="#3b82f6",
        linestyle="-",
        linewidth=1.5,
        label=f"Avg Volume ({avg_v:.1f}M)",
    )
    ax1.axhline(
        thresh_v,
        color="#ef4444",
        linestyle="--",
        linewidth=1.5,
        label=f"+{vol_spike_threshold}% Spike Threshold",
    )
    ax1.set_ylabel("Volume (Millions)")
    ax1.set_xlabel("Date")
    ax1.legend(loc="upper left")
    plt.tight_layout()
    st.pyplot(fig1)

  with col_right:
    st.subheader("3. Distribution of Daily Returns (Histogram)")
    fig2, ax2 = plt.subplots(figsize=(7, 3.8))
    sns.histplot(
        df["Daily_Return_Pct"], bins=30, kde=True, color="#3b82f6", ax=ax2
    )
    ax2.axvline(
        df["Daily_Return_Pct"].mean(),
        color="#ef4444",
        linestyle="--",
        label="Mean Return",
    )
    ax2.set_xlabel("Daily Return (%)")
    ax2.set_ylabel("Frequency (Days)")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)

  st.subheader(
      f"4. Flagged Trading Volume Anomalies (Volume Spiked >="
      f" {vol_spike_threshold}% Above Average)"
  )
  st.dataframe(
      anomalies[[
          "Date",
          "Ticker",
          "Close",
          "Volume",
          "Avg_Volume",
          "Volume_Spike_Pct",
          "Daily_Return_Pct",
      ]].reset_index(drop=True),
      use_container_width=True,
  )
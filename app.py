import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Stock Price Anomaly Detector", page_icon="📈", layout="wide"
)

st.title("📈 Stock Price Visualization & Anomaly Detector UI")
st.write(
    "Adjust the Moving Average window and Z-Score threshold to detect unusual stock price shocks in real time."
)

st.sidebar.header("Model Parameters")
sma_window = st.sidebar.slider("Moving Average Window (Days)", 5, 60, 20)
z_threshold = st.sidebar.slider(
    "Anomaly Z-Score Threshold (σ)", 1.5, 3.5, 2.5, 0.1
)

np.random.seed(42)
dates = pd.bdate_range(start="2025-01-01", periods=252)
daily_returns = np.random.normal(loc=0.0008, scale=0.015, size=len(dates))
daily_returns[[45, 110, 175, 220]] = [0.065, -0.058, 0.072, -0.061]
prices = 150 * np.cumprod(1 + daily_returns)

df = pd.DataFrame({"Date": dates, "Close": np.round(prices, 2)})
df["SMA"] = (
    df["Close"].rolling(window=sma_window, min_periods=1).mean().round(2)
)
df["Daily_Return_Pct"] = df["Close"].pct_change().fillna(0) * 100

mean_ret = df["Daily_Return_Pct"].mean()
std_ret = df["Daily_Return_Pct"].std()
df["Z_Score"] = ((df["Daily_Return_Pct"] - mean_ret) / std_ret).round(2)
df["Anomaly"] = df["Z_Score"].abs() > z_threshold

anomalies = df[df["Anomaly"]]
col1, col2, col3 = st.columns(3)
col1.metric("Latest Close Price", f"${df['Close'].iloc[-1]:.2f}")
col2.metric(f"{sma_window}-Day Moving Average", f"${df['SMA'].iloc[-1]:.2f}")
col3.metric("Detected Anomalies", f"{len(anomalies)} Days")

st.subheader("Price Trend vs. Moving Average")
chart_df = df.set_index("Date")[["Close", "SMA"]]
st.line_chart(chart_df)

st.subheader(f"Flagged Market Anomalies (|Z-Score| > {z_threshold})")
st.dataframe(
    anomalies[
        ["Date", "Close", "Daily_Return_Pct", "Z_Score"]
    ].reset_index(drop=True),
    use_container_width=True,
)
# DS_3_StockPriceViz_BYTE

## Overview
Task 3 submission for the **B.Y.T.E by Arithmatrix Virtual Internship Program (AVIP) 2026**: Stock Price Visualization, Dual Moving Averages (20-Day & 50-Day SMA), Daily Return Distribution, and the **"Insider" 500% Trading Volume Anomaly Detector**.

## Ticker & Data Source
- **Default Ticker Symbol:** `GME` (GameStop Corp.) and `AAPL` (Apple Inc.) — dynamically configurable in the Streamlit dashboard and notebook for `SMCI`, `NVDA`, `TSLA`, `MSFT`, `AMZN`, or any valid date range.
- **Data Source:** Real historical OHLCV market data fetched directly from **Yahoo Finance** via the `yfinance` Python API (`yf.download()`).

## Volatility, Return & "Insider" Volume Anomaly Interpretation (118 Words)
Across the 2024 trading horizon (`2024-01-01` to `2024-12-31`), **GME** exhibited an overall upward trajectory accompanied by extreme right-skewed tail volatility, recording a mean daily return of **+0.65%** and a daily return standard deviation of **10.89%**. The 20-day Simple Moving Average (`SMA_20`) crossing above the 50-day SMA (`SMA_50`) in May 2024 confirmed a sharp bullish momentum breakout. 

Using the **500% above-average volume threshold** (baseline average of **18.8M shares/day**), the "Insider" Anomaly Detector flagged **9 trading days** in bright red between May 13 and June 12, 2024, where volume surged between **+597.0%** and **+1,386.6%** (`279.1M` shares on `2024-06-07`). These extreme institutional/retail volume shocks directly coincided with the widest single-day price swings (`+74.4%` and `-39.4%`).

## Deliverables Included
- `DS_3_StockPriceViz_BYTE.ipynb`: Jupyter Notebook fetching real historical stock data via `yfinance`, computing 20-day and 50-day Simple Moving Averages, distribution of daily returns, and 500%+ volume spike anomaly detection.
- `app.py`: Interactive Streamlit dashboard powered by live Yahoo Finance data with custom date-range and parameter controls.
- `stock_data_cleaned.csv`: Cleaned historical stock dataset with moving averages and volume anomaly flags.
- Exported Visualization PNGs:
  - `stock_price_moving_averages.png`
  - `stock_price_anomalies.png`
  - `daily_returns_distribution.png`

## How to Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
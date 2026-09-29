# DS_3_StockPriceViz_BYTE

## Overview
Task 3 submission for the **B.Y.T.E by Arithmatrix Virtual Internship Program (AVIP) 2026**: Stock Price Visualization, Dual Moving Averages (20-Day & 50-Day SMA), Daily Return Distribution, and 500% Trading Volume Anomaly Detector.

## Ticker & Data Source
- **Default Ticker Symbol:** `AAPL` (Apple Inc.) — configurable in the dashboard and notebook for `NVDA`, `TSLA`, `MSFT`, `AMZN`, or any valid equity ticker.
- **Data Source:** Real historical OHLCV market data fetched directly from **Yahoo Finance** via the `yfinance` Python API (`yf.download()`).

## Deliverables Included
- `DS_3_StockPriceViz_BYTE.ipynb`: Jupyter Notebook fetching real historical stock data via `yfinance`, computing 20-day and 50-day Simple Moving Averages, distribution of daily returns, and 500%+ volume spike anomaly detection.
- `app.py`: Interactive Streamlit dashboard powered by live Yahoo Finance data.
- `stock_data_cleaned.csv`: Cleaned historical stock dataset with moving averages and volume anomaly indicators.
- Exported Visualization PNGs:
  - `stock_price_moving_averages.png`
  - `stock_price_anomalies.png`
  - `daily_returns_distribution.png`

## How to Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
# DS_3_StockPriceViz_BYTE

## Overview
This repository contains Task 3 for the **B.Y.T.E by Arithmatrix Virtual Internship Program (AVIP) 2026**: Stock Price Visualization, Moving Averages, and Statistical Anomaly Detection UI.

## Deliverables Included
- `DS_3_StockPriceViz_BYTE.ipynb`: Jupyter Notebook with time-series processing, Simple Moving Averages (`SMA_20`, `SMA_50`), Z-score anomaly detection, and analysis report.
- `app.py`: Interactive Streamlit Stock Price & Anomaly Detector dashboard.
- `stock_data_cleaned.csv`: Processed time-series dataset with engineered features and anomaly flags.
- Exported Visualization PNGs:
  - `stock_price_moving_averages.png`
  - `stock_price_anomalies.png`
  - `daily_returns_distribution.png`

## How to Run Locally
1. Run all cells in `DS_3_StockPriceViz_BYTE.ipynb` to generate the cleaned dataset and PNG charts.
2. Launch the Streamlit web app:
   ```bash
   streamlit run app.py
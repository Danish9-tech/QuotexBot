"""
download_real_data.py — Downloads real market forex historical candle data from Yahoo Finance
and saves to load_csv-compatible CSV files:
- backtest/real_data_1m.csv
- backtest/real_data_15m.csv
"""

import os
import yfinance as yf
import pandas as pd


def fetch_and_save():
    print("Downloading real forex market data from Yahoo Finance...")

    # Fetch 1-minute interval candles (7-day window limit from Yahoo)
    print("Fetching EURUSD=X (1m interval)...")
    df_1m = yf.download("EURUSD=X", period="7d", interval="1m", progress=False)

    if isinstance(df_1m.columns, pd.MultiIndex):
        df_1m.columns = df_1m.columns.get_level_values(0)

    df_1m = df_1m.reset_index()
    df_1m.columns = [str(c).strip().lower() for c in df_1m.columns]

    rename_map = {
        "datetime": "timestamp", "date": "timestamp",
        "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"
    }
    df_1m = df_1m.rename(columns=rename_map)

    required = ["timestamp", "open", "high", "low", "close"]
    df_1m = df_1m[required].dropna()

    out_1m = os.path.join("backtest", "real_data_1m.csv")
    df_1m.to_csv(out_1m, index=False)
    print(f"Saved {len(df_1m)} rows of 1-minute data to {out_1m}")

    # Fetch 15-minute interval candles (60-day window)
    print("Fetching EURUSD=X (15m interval)...")
    df_15m = yf.download("EURUSD=X", period="60d", interval="15m", progress=False)

    if isinstance(df_15m.columns, pd.MultiIndex):
        df_15m.columns = df_15m.columns.get_level_values(0)

    df_15m = df_15m.reset_index()
    df_15m.columns = [str(c).strip().lower() for c in df_15m.columns]
    df_15m = df_15m.rename(columns=rename_map)
    df_15m = df_15m[required].dropna()

    out_15m = os.path.join("backtest", "real_data_15m.csv")
    df_15m.to_csv(out_15m, index=False)
    print(f"Saved {len(df_15m)} rows of 15-minute data to {out_15m}")


if __name__ == "__main__":
    fetch_and_save()

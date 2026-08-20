"""
data.py — get OHLC candle data into a standard DataFrame the engine can use.

Standard format: pandas DataFrame, indexed 0..N-1, columns:
    timestamp (datetime), open, high, low, close, volume (volume optional)
"""

import pandas as pd
import numpy as np


def load_csv(path: str) -> pd.DataFrame:
    """Load OHLC data from CSV. Expects at least: timestamp,open,high,low,close
    Column names are matched case-insensitively and common aliases are handled."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    aliases = {
        "time": "timestamp", "date": "timestamp", "datetime": "timestamp",
        "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if "volume" not in df.columns:
        df["volume"] = np.nan

    return df[["timestamp", "open", "high", "low", "close", "volume"]] if "timestamp" in df.columns \
        else df[["open", "high", "low", "close", "volume"]]


def generate_synthetic(n: int = 5000, start_price: float = 100.0,
                        vol: float = 0.0006, seed: int = 42) -> pd.DataFrame:
    """Generate a random-walk candle series purely for TESTING the engine itself
    (checking your code runs and math is right)."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, vol, n)
    close = start_price * np.exp(np.cumsum(returns))

    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]

    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))

    timestamps = pd.date_range("2026-01-01", periods=n, freq="min")

    return pd.DataFrame({
        "timestamp": timestamps, "open": open_, "high": high,
        "low": low, "close": close, "volume": rng.integers(50, 500, n),
    })


def fetch_binance(symbol: str = "BTCUSDT", interval: str = "1m", limit: int = 1000) -> pd.DataFrame:
    """Pull real crypto candles from Binance's public REST API."""
    import requests
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def fetch_yfinance(ticker: str = "EURUSD=X", period: str = "7d", interval: str = "1m") -> pd.DataFrame:
    """Pull real forex candles via yfinance."""
    import yfinance as yf
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df = df.reset_index().rename(columns={
        "Datetime": "timestamp", "Date": "timestamp", "Open": "open",
        "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
    })
    return df[["timestamp", "open", "high", "low", "close", "volume"]]

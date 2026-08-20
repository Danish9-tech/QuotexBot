"""
strategies.py — strategy functions for backtesting.

CONTRACT: Every strategy function takes `df` (a DataFrame containing ONLY data
up to and including the current candle — the engine enforces this structurally)
and returns one of: "call", "put", or None (no trade).
"""

import pandas as pd
import numpy as np


def random_baseline(df: pd.DataFrame, rng=np.random.default_rng(1)) -> str:
    """Pure coin flip baseline for edge validation."""
    return "call" if rng.random() < 0.5 else "put"


def ema_crossover(df: pd.DataFrame, fast: int = 5, slow: int = 20) -> str | None:
    """Classic trend-following EMA crossover signal."""
    if len(df) < slow + 1:
        return None
    close = df["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    prev_diff = ema_fast.iloc[-2] - ema_slow.iloc[-2]
    curr_diff = ema_fast.iloc[-1] - ema_slow.iloc[-1]

    if prev_diff <= 0 and curr_diff > 0:
        return "call"
    if prev_diff >= 0 and curr_diff < 0:
        return "put"
    return None


def rsi_reversal(df: pd.DataFrame, period: int = 14, low_th: float = 30, high_th: float = 70) -> str | None:
    """Mean-reversion signal using RSI 14."""
    if len(df) < period + 1:
        return None
    close = df["close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]

    if avg_loss == 0:
        return None
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    if rsi < low_th:
        return "call"
    if rsi > high_th:
        return "put"
    return None


def bollinger_reversal(df: pd.DataFrame, period: int = 20, std_mult: float = 2.0) -> str | None:
    """Bollinger Bands mean-reversion signal."""
    if len(df) < period + 1:
        return None
    close = df["close"]
    mid = close.rolling(period).mean().iloc[-1]
    std = close.rolling(period).std().iloc[-1]
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    price = close.iloc[-1]

    if price <= lower:
        return "call"
    if price >= upper:
        return "put"
    return None


def sureshot_quant_pro(df: pd.DataFrame) -> str | None:
    """
    PORTED BOT STRATEGY: Sureshot Quant Pro Engine
    Exact quantitative indicator logic ported from live bot (groq_analyzer.py).
    Takes a DataFrame containing ONLY history through the current candle (no look-ahead).
    Calculates Donchian 24, RSI 14, Stochastic (5,3,3), and Dual EMAs (20/50/100).
    """
    if len(df) < 24:
        return None

    # Slice max 150 candles for 100x faster indicator calculation without look-ahead
    if len(df) > 150:
        df = df.iloc[-150:]

    # Standardize column names to lowercase
    close_series = df["close"]
    open_series = df["open"]
    high_series = df["high"]
    low_series = df["low"]

    # Calculate Moving Averages
    ema_20_series = close_series.ewm(span=20, adjust=False).mean()
    ema_50_series = close_series.ewm(span=min(50, len(df)), adjust=False).mean()
    ema_100_series = close_series.ewm(span=min(100, len(df)), adjust=False).mean()

    # Calculate RSI 14 using Exponential Moving Average
    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi_14_series = 100.0 - (100.0 / (1.0 + rs))

    # Calculate Stochastic Oscillator (5, 3, 3)
    low_5 = low_series.rolling(window=min(5, len(df))).min()
    high_5 = high_series.rolling(window=min(5, len(df))).max()
    stoch_k_series = 100.0 * ((close_series - low_5) / ((high_5 - low_5).replace(0, 1e-9)))

    # Calculate Donchian Channel (Period 24)
    dc_upper_series = high_series.rolling(window=min(24, len(df))).max()
    dc_lower_series = low_series.rolling(window=min(24, len(df))).min()

    # Current candle metrics
    c_open = float(open_series.iloc[-1])
    c_close = float(close_series.iloc[-1])
    c_high = float(high_series.iloc[-1])
    c_low = float(low_series.iloc[-1])

    # Previous candles
    p_open = float(open_series.iloc[-2]) if len(df) >= 2 else c_open
    p_close = float(close_series.iloc[-2]) if len(df) >= 2 else c_close
    p2_open = float(open_series.iloc[-3]) if len(df) >= 3 else p_open
    p2_close = float(close_series.iloc[-3]) if len(df) >= 3 else p_close

    total_range = max(0.000001, c_high - c_low)
    body_size = abs(c_close - c_open)
    upper_wick = c_high - max(c_open, c_close)
    lower_wick = min(c_open, c_close) - c_low

    upper_wick_ratio = upper_wick / total_range
    lower_wick_ratio = lower_wick / total_range

    ema_20 = float(ema_20_series.iloc[-1])
    ema_50 = float(ema_50_series.iloc[-1])
    rsi_14 = float(rsi_14_series.iloc[-1])
    stoch_k = float(stoch_k_series.iloc[-1])
    dc_upper = float(dc_upper_series.iloc[-1])
    dc_lower = float(dc_lower_series.iloc[-1])

    is_uptrend = c_close >= ema_20
    is_downtrend = c_close < ema_20

    is_touching_dc_lower = (c_low <= dc_lower) or (abs(c_close - dc_lower) <= total_range * 0.15)
    is_touching_dc_upper = (c_high >= dc_upper) or (abs(c_close - dc_upper) <= total_range * 0.15)

    gap = c_open - p_close
    gap_threshold = max(0.000001, total_range * 0.15)
    is_gap_up = gap > gap_threshold
    is_gap_down = gap < -gap_threshold

    # Skip flat/tied price candles (price unchanged from previous candle)
    if c_close == p_close:
        return None

    # --- Sureshot Pro Quant Rules ---
    # Rule 1: Triple Extreme Confluence Reversal
    if (is_touching_dc_lower or rsi_14 <= 35) and stoch_k <= 20:
        return "call"
    elif (is_touching_dc_upper or rsi_14 >= 65) and stoch_k >= 80:
        return "put"

    # Rule 2: Structural Wick Rejection Bounce
    elif lower_wick_ratio >= 0.15 or is_gap_down:
        return "call"
    elif upper_wick_ratio >= 0.15 or is_gap_up:
        return "put"

    # Rule 3: Dual EMA Confluence Trend Impulse
    elif (c_close > ema_20) and (ema_20 > ema_50) and (c_close >= c_open) and stoch_k > 45:
        return "call"
    elif (c_close < ema_20) and (ema_20 < ema_50) and (c_close <= c_open) and stoch_k < 55:
        return "put"

    # Rule 4: 3-Bar Momentum Expansion
    elif is_uptrend and (p2_close > p2_open) and (p_close > p_open) and (c_close > c_open):
        return "call"
    elif is_downtrend and (p2_close < p2_open) and (p_close < p_open) and (c_close < c_open):
        return "put"

    # Default: Return None when no clear quantitative setup is present
    else:
        return None

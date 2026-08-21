"""
strategies_researched.py — real, named strategies sourced from a published
trading guide (TradingRage / CrypOption Hub, "Quotex Strategy: The Math Of
Profitability And 6 Strategies", accessed 2026), NOT invented or guessed.

Each strategy below follows the article's exact stated indicator settings and
entry rules as closely as translatable into code. The article itself is
upfront that these are frameworks to test, not verified/backtested results —
which is exactly the point of running them through this engine.

One strategy from the source (ZigZag + DeMarker reversal) was deliberately
EXCLUDED here: the source itself states ZigZag repaints (it redraws the most
recent swing point as new candles form), which makes it structurally
impossible to backtest without look-ahead bias — the "signal" a human sees on
a live chart didn't exist yet at the time it appears to have fired. Including
it would reintroduce the exact bug this project already found and fixed once.

Every function below follows the same contract as strategies.py: only sees
data up to and including the current candle, returns "call"/"put"/None.
"""

import pandas as pd
import numpy as np


# ---------- shared indicator helpers ----------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0
    minus_dm[(minus_dm - plus_dm) < 0] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    return macd_line, signal_line


def _recent_support_resistance(df: pd.DataFrame, lookback: int = 50, window: int = 3):
    """Simple, strictly-causal swing high/low detector: a candle at index i-window
    is a swing point if it's the max/min within a window on BOTH sides — using
    only data already visible (no future candles needed since we only look at
    candles that are `window` bars old relative to the current end of df)."""
    if len(df) < lookback + window * 2 + 1:
        return None, None
    recent = df.iloc[-lookback:]
    highs, lows = [], []
    vals_h = recent["high"].values
    vals_l = recent["low"].values
    for i in range(window, len(recent) - window):
        seg_h = vals_h[i - window: i + window + 1]
        seg_l = vals_l[i - window: i + window + 1]
        if vals_h[i] == seg_h.max():
            highs.append(vals_h[i])
        if vals_l[i] == seg_l.min():
            lows.append(vals_l[i])
    resistance = max(highs) if highs else None
    support = min(lows) if lows else None
    return support, resistance


# ---------- Strategy 1: EMA(9,21) crossover + RSI(14) filter, 1-min ----------

def strategy_ema9_21_rsi_filter(df: pd.DataFrame) -> str | None:
    """Source rules: 9/21 EMA cross triggers direction; RSI(14) > 50 confirms
    call, < 50 confirms put. Enter on the candle close where both align."""
    if len(df) < 22:
        return None
    close = df["close"]
    ema9 = _ema(close, 9)
    ema21 = _ema(close, 21)
    rsi = _rsi(close, 14)

    prev_diff = ema9.iloc[-2] - ema21.iloc[-2]
    curr_diff = ema9.iloc[-1] - ema21.iloc[-1]
    rsi_now = rsi.iloc[-1]
    if pd.isna(rsi_now):
        return None

    crossed_up = prev_diff <= 0 and curr_diff > 0
    crossed_down = prev_diff >= 0 and curr_diff < 0

    if crossed_up and rsi_now > 50:
        return "call"
    if crossed_down and rsi_now < 50:
        return "put"
    return None


# ---------- Strategy 2: Bollinger Band reversion, ADX-filtered ----------

def strategy_bollinger_reversion_adx(df: pd.DataFrame, period: int = 20, std_mult: float = 2.0,
                                      adx_threshold: float = 20.0) -> str | None:
    """Source rules: only trade when ADX < 20 (no trend). Put when price pierces
    the upper band and closes back inside; call on the lower band mirror."""
    if len(df) < max(period, 28) + 1:
        return None
    close = df["close"]
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std

    adx = _adx(df, 14)
    adx_now = adx.iloc[-1]
    if pd.isna(adx_now) or adx_now >= adx_threshold:
        return None  # trending market -- source says skip this strategy entirely

    prev_high, prev_close_ = df["high"].iloc[-2], close.iloc[-2]
    curr_close = close.iloc[-1]
    upper_now, lower_now = upper.iloc[-1], lower.iloc[-1]

    # "touched or pushed slightly outside, then closed back inside"
    touched_upper = df["high"].iloc[-1] >= upper_now
    touched_lower = df["low"].iloc[-1] <= lower_now
    closed_back_inside_upper = curr_close < upper_now
    closed_back_inside_lower = curr_close > lower_now

    if touched_upper and closed_back_inside_upper:
        return "put"
    if touched_lower and closed_back_inside_lower:
        return "call"
    return None


# ---------- Strategy 4: MACD + Support/Resistance, 5-min ----------

def strategy_macd_support_resistance(df: pd.DataFrame, level_tolerance: float = 0.0006) -> str | None:
    """Source rules: price approaches a marked S/R level; MACD histogram/line
    crossing confirms direction off that level. `level_tolerance` is a fractional
    distance (e.g. 0.0006 = 0.06%) used to define 'approaching' a level, since we
    don't have a human eyeballing a chart."""
    if len(df) < 60:
        return None
    close = df["close"]
    macd_line, signal_line = _macd(close)
    if pd.isna(macd_line.iloc[-1]) or pd.isna(signal_line.iloc[-1]):
        return None

    support, resistance = _recent_support_resistance(df, lookback=50, window=3)
    if support is None or resistance is None:
        return None

    price = close.iloc[-1]
    near_support = abs(price - support) / support <= level_tolerance
    near_resistance = abs(price - resistance) / resistance <= level_tolerance

    prev_macd_diff = macd_line.iloc[-2] - signal_line.iloc[-2]
    curr_macd_diff = macd_line.iloc[-1] - signal_line.iloc[-1]
    bullish_flip = prev_macd_diff <= 0 and curr_macd_diff > 0
    bearish_flip = prev_macd_diff >= 0 and curr_macd_diff < 0

    if near_support and bullish_flip:
        return "call"
    if near_resistance and bearish_flip:
        return "put"
    return None


# ---------- Strategy 5: Pin bar at a level, pure price action ----------

def strategy_pinbar_at_level(df: pd.DataFrame, level_tolerance: float = 0.0006,
                              wick_body_ratio: float = 2.0) -> str | None:
    """Source rules: price reaches a marked S/R level, a pin bar forms there
    (small body, long wick rejecting the level). Enter on the candle AFTER the
    pin bar, in the direction of the rejection -- i.e. the pin bar itself is
    at index -2, and we're deciding the trade at index -1 (already the 'next
    candle' relative to the pin bar), matching the source's stated entry rule."""
    if len(df) < 55:
        return None

    support, resistance = _recent_support_resistance(df.iloc[:-1], lookback=50, window=3)
    if support is None or resistance is None:
        return None

    pin = df.iloc[-2]  # the candle we're checking for a pin bar shape
    o, h, l, c = pin["open"], pin["high"], pin["low"], pin["close"]
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    if body == 0:
        return None

    near_support = abs(l - support) / support <= level_tolerance if support else False
    near_resistance = abs(h - resistance) / resistance <= level_tolerance if resistance else False

    is_bullish_pin = lower_wick >= wick_body_ratio * body and near_support
    is_bearish_pin = upper_wick >= wick_body_ratio * body and near_resistance

    if is_bullish_pin:
        return "call"
    if is_bearish_pin:
        return "put"
    return None

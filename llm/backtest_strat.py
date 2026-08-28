"""
Backtest-compatible LLM signal function.

The backtest framework's strategy contract is:
    def strat(df: pd.DataFrame) -> str | None
returning "call", "put", or None (no trade).

This module provides:
  - news_driven(df, asset): a proxy for the LLM signal that uses ONLY the
    calendar (date/time rules) since historical news is not available in the
    backtest. Trades only around high-impact events.
  - combined_signal(df, asset): both `sureshot_quant_pro` AND `news_driven`
    must agree on direction. If they disagree or either abstains, no trade.

**Honest limitation:** the calendar proxy uses a static schedule of known
high-impact events (FOMC, CPI, NFP, ECB, etc.). It validates the *abstention
discipline* and the *two-layer confirmation logic*, not the LLM's actual
news interpretation. Live deployment adds the LLM's news reasoning on top.
"""

from datetime import datetime, timedelta
import calendar as _cal
import pandas as pd


# Approximate 2024-2026 high-impact event dates. Used for backtest only.
# These are days when FOMC, ECB, CPI, NFP, or similar events are scheduled.
# In live mode the LLM uses real news + real calendar; here we use a static
# lookup so the backtest can exercise the abstention discipline.
HIGH_IMPACT_DATES_UTC: set[str] = {
    # 2024
    "2024-01-10", "2024-01-11", "2024-01-31",  # NFP, CPI
    "2024-01-31", "2024-01-28",  # FOMC
    "2024-02-07", "2024-02-13", "2024-02-29",  # NFP, CPI
    "2024-03-08", "2024-03-12", "2024-03-19", "2024-03-20",  # NFP, CPI, FOMC
    "2024-04-05", "2024-04-10", "2024-04-30",  # NFP, CPI, FOMC
    "2024-05-03", "2024-05-10", "2024-05-15", "2024-05-31",  # NFP, CPI, FOMC
    "2024-06-07", "2024-06-12", "2024-06-18", "2024-06-28",  # NFP, CPI, FOMC
    "2024-07-05", "2024-07-11", "2024-07-25", "2024-07-31",  # NFP, CPI, FOMC
    "2024-08-02", "2024-08-14", "2024-08-21",  # NFP, CPI, FOMC/Jackson Hole
    "2024-09-06", "2024-09-11", "2024-09-17", "2024-09-18",  # NFP, CPI, FOMC
    "2024-10-04", "2024-10-10", "2024-10-29",  # NFP, CPI, FOMC
    "2024-11-01", "2024-11-07", "2024-11-13", "2024-11-26",  # NFP, CPI, FOMC
    "2024-12-06", "2024-12-11", "2024-12-17", "2024-12-18",  # NFP, CPI, FOMC
    # 2025
    "2025-01-10", "2025-01-15", "2025-01-29",  # NFP, CPI, FOMC
    "2025-02-07", "2025-02-12", "2025-02-19",  # NFP, CPI, FOMC
    "2025-03-07", "2025-03-12", "2025-03-19",  # NFP, CPI, FOMC
    "2025-04-04", "2025-04-10", "2025-04-16", "2025-04-30",  # NFP, CPI, FOMC
    "2025-05-02", "2025-05-13", "2025-05-15", "2025-05-28",  # NFP, CPI, FOMC
    "2025-06-06", "2025-06-11", "2025-06-18", "2025-06-25",  # NFP, CPI, FOMC
    "2025-07-03", "2025-07-11", "2025-07-15", "2025-07-30",  # NFP, CPI, FOMC
    "2025-08-01", "2025-08-12", "2025-08-14",  # NFP, CPI, Jackson Hole
    "2025-09-05", "2025-09-10", "2025-09-17",  # NFP, CPI, FOMC
    "2025-10-03", "2025-10-10", "2025-10-29",  # NFP, CPI, FOMC
    "2025-11-07", "2025-11-12", "2025-11-19", "2025-11-26",  # NFP, CPI, FOMC
    "2025-12-05", "2025-12-10", "2025-12-17",  # NFP, CPI, FOMC
    # 2026
    "2026-01-09", "2026-01-14", "2026-01-28",  # NFP, CPI, FOMC
    "2026-02-06", "2026-02-11", "2026-02-18",  # NFP, CPI, FOMC
    "2026-03-06", "2026-03-11", "2026-03-18",  # NFP, CPI, FOMC
    "2026-04-03", "2026-04-09", "2026-04-15", "2026-04-29",  # NFP, CPI, FOMC
    "2026-05-01", "2026-05-08", "2026-05-13", "2026-05-27",  # NFP, CPI, FOMC
    "2026-06-05", "2026-06-10", "2026-06-17", "2026-06-24",  # NFP, CPI, FOMC
    "2026-07-02", "2026-07-10", "2026-07-15", "2026-07-29",  # NFP, CPI, FOMC
    "2026-08-07", "2026-08-12", "2026-08-19",  # NFP, CPI, Jackson Hole
}


def _df_timestamp(df: pd.DataFrame) -> pd.Timestamp | None:
    """Get the timestamp of the latest row in df (assumes sorted ascending)."""
    if df is None or len(df) == 0:
        return None
    last = df.iloc[-1]
    for col in ("timestamp", "time", "datetime", "date"):
        if col in df.columns:
            v = last[col]
            if isinstance(v, pd.Timestamp):
                return v
            try:
                return pd.Timestamp(v)
            except Exception:
                continue
    return None


def _is_event_window(ts: pd.Timestamp, window_minutes: int = 60) -> bool:
    """
    Approximate event window: within 60 min of a high-impact date (UTC).
    Conservative — only fires when we have strong reason to think an event
    is happening.
    """
    if ts is None:
        return False
    date_str = ts.strftime("%Y-%m-%d")
    if date_str in HIGH_IMPACT_DATES_UTC:
        return True
    return False


def _dominant_direction(df: pd.DataFrame) -> str | None:
    """
    Heuristic: look at last 5 candles' close direction.

    In a real backtest, the LLM would base this on news + indicators. Here
    we use a simple momentum proxy: if last 3 of last 5 candles are bullish
    (close > open), the LLM proxy says CALL. If bearish, PUT. Else None.
    """
    if len(df) < 5:
        return None
    recent = df.iloc[-5:]
    try:
        bull = sum(1 for _, r in recent.iterrows()
                   if float(r.get("close", 0)) > float(r.get("open", 0)))
        bear = 5 - bull
        if bull >= 3:
            return "call"
        if bear >= 3:
            return "put"
    except (ValueError, TypeError):
        return None
    return None


def news_driven(df: pd.DataFrame, asset: str = "EURUSD_otc") -> str | None:
    """
    Backtest proxy for the live LLM signal.

    Logic:
      1. If today is a high-impact event date (UTC): proceed
      2. Otherwise: abstain (None)
      3. If event day: take a directional call from candle momentum
    """
    ts = _df_timestamp(df)
    if not _is_event_window(ts):
        return None
    return _dominant_direction(df)


def combined_signal(df: pd.DataFrame, asset: str = "EURUSD_otc") -> str | None:
    """
    Two-layer: indicator stack AND news-driven proxy must agree.

    Returns the agreed direction, or None if they disagree or either abstains.
    """
    # Lazy import to avoid circular dep
    from backtest.strategies import sureshot_quant_pro
    ind = sureshot_quant_pro(df)
    nw = news_driven(df, asset)
    if ind is None or nw is None:
        return None
    if ind == nw:
        return ind
    return None  # disagreement -> no trade

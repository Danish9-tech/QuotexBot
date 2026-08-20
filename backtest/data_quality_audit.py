"""
data_quality_audit.py — Comprehensive data quality audit on real_data_1m.csv & real_data_15m.csv:
1. Count consecutive identical close prices (stale candles).
2. Audit timestamps for gaps, duplicates, and non-sequential ordering.
3. Analyze time-of-day / session clustering of flat candles.
4. Re-run strategy and inversion diagnostic on cleaned (de-duplicated) data.
"""

import os
import pandas as pd
import numpy as np
from data import load_csv
from metrics import breakeven_win_rate


def sureshot_quant_pro_with_subrule_tag(df: pd.DataFrame):
    """Identical to sureshot_quant_pro in strategies.py, returning (signal, subrule_tag)."""
    if len(df) < 24:
        return None, None

    if len(df) > 150:
        df = df.iloc[-150:]

    close_series = df["close"]
    open_series = df["open"]
    high_series = df["high"]
    low_series = df["low"]

    ema_20_series = close_series.ewm(span=20, adjust=False).mean()
    ema_50_series = close_series.ewm(span=min(50, len(df)), adjust=False).mean()

    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi_14_series = 100.0 - (100.0 / (1.0 + rs))

    low_5 = low_series.rolling(window=min(5, len(df))).min()
    high_5 = high_series.rolling(window=min(5, len(df))).max()
    stoch_k_series = 100.0 * ((close_series - low_5) / ((high_5 - low_5).replace(0, 1e-9)))

    dc_upper_series = high_series.rolling(window=min(24, len(df))).max()
    dc_lower_series = low_series.rolling(window=min(24, len(df))).min()

    c_open = float(open_series.iloc[-1])
    c_close = float(close_series.iloc[-1])
    c_high = float(high_series.iloc[-1])
    c_low = float(low_series.iloc[-1])

    p_open = float(open_series.iloc[-2]) if len(df) >= 2 else c_open
    p_close = float(close_series.iloc[-2]) if len(df) >= 2 else c_close
    p2_open = float(open_series.iloc[-3]) if len(df) >= 3 else p_open
    p2_close = float(close_series.iloc[-3]) if len(df) >= 3 else p_close

    total_range = max(0.000001, c_high - c_low)
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

    if c_close == p_close:
        return None, None

    if (is_touching_dc_lower or rsi_14 <= 35) and stoch_k <= 20:
        return "call", "Rule 1 (Oversold - CALL)"
    elif (is_touching_dc_upper or rsi_14 >= 65) and stoch_k >= 80:
        return "put", "Rule 1 (Overbought - PUT)"
    elif lower_wick_ratio >= 0.15 or is_gap_down:
        return "call", "Rule 2 (Lower Wick Bounce - CALL)"
    elif upper_wick_ratio >= 0.15 or is_gap_up:
        return "put", "Rule 2 (Upper Wick Rejection - PUT)"
    elif (c_close > ema_20) and (ema_20 > ema_50) and (c_close >= c_open) and stoch_k > 45:
        return "call", "Rule 3 (Bullish Impulse - CALL)"
    elif (c_close < ema_20) and (ema_20 < ema_50) and (c_close <= c_open) and stoch_k < 55:
        return "put", "Rule 3 (Bearish Impulse - PUT)"
    elif is_uptrend and (p2_close > p2_open) and (p_close > p_open) and (c_close > c_open):
        return "call", "Rule 4 (3 Green Expansion - CALL)"
    elif is_downtrend and (p2_close < p2_open) and (p_close < p_open) and (c_close < c_open):
        return "put", "Rule 4 (3 Red Expansion - PUT)"
    else:
        return None, None


def audit_data(csv_filename: str, label: str):
    file_path = os.path.join("backtest", csv_filename)
    if not os.path.exists(file_path):
        print(f"Error: {file_path} missing.")
        return

    df = load_csv(file_path)
    total_rows = len(df)

    print("=" * 90)
    print(f"DATA QUALITY AUDIT & CLEANED DIAGNOSTIC: {label}")
    print(f"File: {csv_filename} ({total_rows} total rows)")
    print("=" * 90)

    # 1. Consecutive identical close prices (stale candles)
    is_consec_flat = df["close"] == df["close"].shift(1)
    n_flat = is_consec_flat.sum()
    pct_flat = (n_flat / total_rows) * 100

    print(f"\n1. CONSECUTIVE DUPLICATE CLOSE AUDIT:")
    print(f"   Total Candles:                   {total_rows}")
    print(f"   Stale Flat Candles (Close == Prev Close): {n_flat} ({pct_flat:.2f}% of dataset)")

    # 2. Timestamp Audit
    print(f"\n2. TIMESTAMP INTEGRITY AUDIT:")
    df["dt"] = pd.to_datetime(df["timestamp"])
    dup_ts = df["dt"].duplicated().sum()
    is_sorted = df["dt"].is_monotonic_increasing

    time_diffs = df["dt"].diff()
    gaps = time_diffs[time_diffs > pd.Timedelta(minutes=5)]

    print(f"   Duplicate Timestamps:            {dup_ts}")
    print(f"   Strict Monotonic (Sequential):   {is_sorted}")
    print(f"   Significant Time Gaps (>5 min):  {len(gaps)} gaps found")

    # 3. Session & Hour Clustering of Flat Stretches
    flat_df = df[is_consec_flat].copy()
    flat_df["hour"] = flat_df["dt"].dt.hour
    flat_df["day_name"] = flat_df["dt"].dt.day_name()

    print(f"\n3. FLAT CANDLE CLUSTERING ANALYSIS (TIME-OF-DAY / DAY-OF-WEEK):")
    print(f"   Top Flat Hours (UTC):")
    top_hours = flat_df["hour"].value_counts().head(5)
    for hr, cnt in top_hours.items():
        print(f"     Hour {hr:02d}:00 UTC: {cnt:4d} flat candles ({cnt/n_flat*100:5.2f}% of all flat candles)")

    print(f"   Flat Days of Week:")
    top_days = flat_df["day_name"].value_counts()
    for dy, cnt in top_days.items():
        print(f"     {dy:10s}: {cnt:4d} flat candles ({cnt/n_flat*100:5.2f}%)")

    # 4. Cleaned Diagnostic (Filter out duplicate consecutive rows entirely)
    # Remove rows where close price is identical to previous row's close price
    df_clean = df[~is_consec_flat].reset_index(drop=True)
    clean_rows = len(df_clean)

    print(f"\n4. DIAGNOSTIC ON CLEANED DATASET (Stale candles purged from source dataset):")
    print(f"   Original Rows: {total_rows} -> Cleaned Rows: {clean_rows} ({total_rows - clean_rows} stale rows removed)")

    payout_pct = 0.85
    needed = breakeven_win_rate(payout_pct)
    total_trades = 0
    normal_wins = 0
    inverted_wins = 0
    tied_expiries = 0

    subrule_stats = {}

    for i in range(len(df_clean)):
        expiry_i = i + 1
        if expiry_i >= len(df_clean):
            break

        visible = df_clean.iloc[: i + 1]
        signal, subrule = sureshot_quant_pro_with_subrule_tag(visible)

        if signal in ("call", "put"):
            entry_price = df_clean.iloc[i]["close"]
            expiry_price = df_clean.iloc[expiry_i]["close"]
            total_trades += 1

            if expiry_price == entry_price:
                tied_expiries += 1
                win_normal = False
                win_inverted = False
            else:
                win_normal = (expiry_price > entry_price) if signal == "call" else (expiry_price < entry_price)
                inverted_signal = "put" if signal == "call" else "call"
                win_inverted = (expiry_price > entry_price) if inverted_signal == "call" else (expiry_price < entry_price)

            if win_normal:
                normal_wins += 1
            if win_inverted:
                inverted_wins += 1

            if subrule not in subrule_stats:
                subrule_stats[subrule] = {"trades": 0, "wins": 0}
            subrule_stats[subrule]["trades"] += 1
            if win_normal:
                subrule_stats[subrule]["wins"] += 1

    overall_normal_wr = (normal_wins / total_trades * 100) if total_trades > 0 else 0.0
    overall_inverted_wr = (inverted_wins / total_trades * 100) if total_trades > 0 else 0.0
    tie_pct = (tied_expiries / total_trades * 100) if total_trades > 0 else 0.0

    non_tied_trades = total_trades - tied_expiries
    non_tied_normal_wr = (normal_wins / non_tied_trades * 100) if non_tied_trades > 0 else 0.0
    non_tied_inverted_wr = (inverted_wins / non_tied_trades * 100) if non_tied_trades > 0 else 0.0

    print(f"   Cleaned Strategy Results (Breakeven required: {needed*100:.2f}%):")
    print(f"     Trades Taken on Cleaned Data: {total_trades}")
    print(f"     Original Strategy Win Rate:   {overall_normal_wr:.2f}% ({normal_wins}/{total_trades})")
    print(f"     Inverted Strategy Win Rate:   {overall_inverted_wr:.2f}% ({inverted_wins}/{total_trades})")
    print(f"     Tied Expiries on Clean Data:  {tied_expiries} ({tie_pct:.2f}%)")
    print(f"     Non-Tied Directional Win Rate:{non_tied_normal_wr:.2f}% ({normal_wins}/{non_tied_trades})")
    print("=" * 90 + "\n")


def main():
    audit_data("real_data_1m.csv", "EURUSD 1-Minute Real Candles (7-Day History)")
    audit_data("real_data_15m.csv", "EURUSD 15-Minute Real Candles (60-Day History)")


if __name__ == "__main__":
    main()

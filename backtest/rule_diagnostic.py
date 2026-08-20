"""
rule_diagnostic.py — Forensic diagnostic script:
1. Sub-rule win rate breakdown (Rule 1 CALL/PUT, Rule 2 CALL/PUT, Rule 3 CALL/PUT, Rule 4 CALL/PUT)
2. Inverted signal test (flipping every signal CALL <-> PUT) with explicit tie breakdown.
Run on both real_data_1m.csv and real_data_15m.csv without changing live strategy code.
"""

import os
import pandas as pd
from data import load_csv
from engine import BacktestEngine
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

    # Skip flat entry candles
    if c_close == p_close:
        return None, None

    # Rule 1
    if (is_touching_dc_lower or rsi_14 <= 35) and stoch_k <= 20:
        return "call", "Rule 1 (Oversold - CALL)"
    elif (is_touching_dc_upper or rsi_14 >= 65) and stoch_k >= 80:
        return "put", "Rule 1 (Overbought - PUT)"

    # Rule 2
    elif lower_wick_ratio >= 0.15 or is_gap_down:
        return "call", "Rule 2 (Lower Wick Bounce - CALL)"
    elif upper_wick_ratio >= 0.15 or is_gap_up:
        return "put", "Rule 2 (Upper Wick Rejection - PUT)"

    # Rule 3
    elif (c_close > ema_20) and (ema_20 > ema_50) and (c_close >= c_open) and stoch_k > 45:
        return "call", "Rule 3 (Bullish Impulse - CALL)"
    elif (c_close < ema_20) and (ema_20 < ema_50) and (c_close <= c_open) and stoch_k < 55:
        return "put", "Rule 3 (Bearish Impulse - PUT)"

    # Rule 4
    elif is_uptrend and (p2_close > p2_open) and (p_close > p_open) and (c_close > c_open):
        return "call", "Rule 4 (3 Green Expansion - CALL)"
    elif is_downtrend and (p2_close < p2_open) and (p_close < p_open) and (c_close < c_open):
        return "put", "Rule 4 (3 Red Expansion - PUT)"

    else:
        return None, None


def analyze_dataset(csv_filename: str, label: str):
    file_path = os.path.join("backtest", csv_filename)
    if not os.path.exists(file_path):
        print(f"Error: {file_path} missing.")
        return

    df = load_csv(file_path)
    payout_pct = 0.85
    needed = breakeven_win_rate(payout_pct)

    print("=" * 85)
    print(f"PER-RULE WIN RATE & INVERTED SIGNAL DIAGNOSTIC: {label}")
    print(f"Dataset File: {csv_filename} ({len(df)} candles)")
    print("=" * 85)

    subrule_stats = {}
    total_trades = 0
    normal_wins = 0
    inverted_wins = 0
    tied_expiries = 0

    for i in range(len(df)):
        expiry_i = i + 1
        if expiry_i >= len(df):
            break

        visible = df.iloc[: i + 1]
        signal, subrule = sureshot_quant_pro_with_subrule_tag(visible)

        if signal in ("call", "put"):
            entry_price = df.iloc[i]["close"]
            expiry_price = df.iloc[expiry_i]["close"]
            total_trades += 1

            if expiry_price == entry_price:
                # Exact price tie at expiry: both normal and inverted signals lose under binary rules
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

    # 1. Print Per-Rule Breakdown
    print(f"\n1. PER-RULE WIN RATE BREAKDOWN (Breakeven needed: {needed*100:.2f}%):")
    print(f"{'Sub-Rule Name':42s} | {'Trades':7s} | {'Wins':6s} | {'Win Rate':9s} | {'Edge vs BE':10s}")
    print("-" * 85)

    sorted_rules = sorted(subrule_stats.keys())
    for r in sorted_rules:
        t_cnt = subrule_stats[r]["trades"]
        w_cnt = subrule_stats[r]["wins"]
        wr = (w_cnt / t_cnt * 100) if t_cnt > 0 else 0.0
        edge = wr - (needed * 100)
        print(f"{r:42s} | {t_cnt:7d} | {w_cnt:6d} | {wr:8.2f}% | {edge:+9.2f} pts")

    print("-" * 85)

    # 2. Print Inverted Signal & Tie Math Diagnostic
    overall_normal_wr = (normal_wins / total_trades * 100) if total_trades > 0 else 0.0
    overall_inverted_wr = (inverted_wins / total_trades * 100) if total_trades > 0 else 0.0
    tie_pct = (tied_expiries / total_trades * 100) if total_trades > 0 else 0.0

    non_tied_trades = total_trades - tied_expiries
    non_tied_normal_wr = (normal_wins / non_tied_trades * 100) if non_tied_trades > 0 else 0.0
    non_tied_inverted_wr = (inverted_wins / non_tied_trades * 100) if non_tied_trades > 0 else 0.0

    print(f"\n2. INVERTED SIGNAL DIAGNOSTIC & MATHEMATICAL PROOF:")
    print(f"   Total Executed Trades:            {total_trades}")
    print(f"   Tied Expiries (Both CALL/PUT Lose): {tied_expiries} ({tie_pct:.2f}% of trades)")
    print(f"   Non-Tied Trades:                  {non_tied_trades} ({100 - tie_pct:.2f}% of trades)")
    print("-" * 85)
    print(f"   ALL TRADES (Including Ties):")
    print(f"     Original Strategy Win Rate:     {overall_normal_wr:.2f}% ({normal_wins}/{total_trades})")
    print(f"     Inverted Strategy Win Rate:     {overall_inverted_wr:.2f}% ({inverted_wins}/{total_trades})")
    print(f"     Tie Expiry Loss Rate:           {tie_pct:.2f}% ({tied_expiries}/{total_trades})")
    print(f"     SUM (Original + Inverted + Ties): {overall_normal_wr + overall_inverted_wr + tie_pct:.2f}% (PROVES 100% MATH INTEGRITY)")
    print("-" * 85)
    print(f"   EXCLUDING TIES (Pure Non-Flat Expiries):")
    print(f"     Original Non-Tied Win Rate:     {non_tied_normal_wr:.2f}% ({normal_wins}/{non_tied_trades})")
    print(f"     Inverted Non-Tied Win Rate:     {non_tied_inverted_wr:.2f}% ({inverted_wins}/{non_tied_trades})")
    print(f"     SUM (Original + Inverted Non-Tied): {non_tied_normal_wr + non_tied_inverted_wr:.2f}% (EXACT 100% FLIP PROOF)")
    print("=" * 85 + "\n")


def main():
    analyze_dataset("real_data_1m.csv", "EURUSD 1-Minute Real Candles (7-Day History)")
    analyze_dataset("real_data_15m.csv", "EURUSD 15-Minute Real Candles (60-Day History)")


if __name__ == "__main__":
    main()

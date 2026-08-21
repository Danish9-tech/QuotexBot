"""
ema_rsi_inversion_diagnostic.py — Inversion & tie diagnostic specifically for strategy_ema9_21_rsi_filter:
1. Calculates original vs inverted (CALL <-> PUT) win rates on real_data_1m.csv.
2. Checks exact tie count and tie percentage (where expiry_price == entry_price).
3. Evaluates pure non-tied directional accuracy.
"""

import os
import pandas as pd
from data import load_csv
from engine import BacktestEngine
from strategies_researched import strategy_ema9_21_rsi_filter
from metrics import breakeven_win_rate

PAYOUT_PCT = 0.85
BREAKEVEN_WR = breakeven_win_rate(PAYOUT_PCT) * 100  # 54.05%


def audit_ema_rsi_strategy(csv_filename: str, label: str, expiry_periods: int):
    file_path = os.path.join("backtest", csv_filename)
    if not os.path.exists(file_path):
        print(f"Error: {file_path} missing.")
        return

    df = load_csv(file_path)
    engine = BacktestEngine(payout_pct=PAYOUT_PCT, tie_is_loss=True)

    result = engine.run(
        df,
        strategy_ema9_21_rsi_filter,
        expiry_periods=expiry_periods,
        start_balance=1000.0,
        stake_mode="fixed_pct",
        stake_value=0.01,
        cooldown_periods=0
    )

    trades_df = result.to_df()
    total_trades = len(trades_df)

    if total_trades == 0:
        print(f"No trades executed for {label}")
        return

    normal_wins = 0
    inverted_wins = 0
    tied_expiries = 0

    for idx, row in trades_df.iterrows():
        entry_price = row["entry_price"]
        expiry_price = row["expiry_price"]
        direction = row["direction"]

        if expiry_price == entry_price:
            tied_expiries += 1
        else:
            win_norm = (expiry_price > entry_price) if direction == "call" else (expiry_price < entry_price)
            inv_dir = "put" if direction == "call" else "call"
            win_inv = (expiry_price > entry_price) if inv_dir == "call" else (expiry_price < entry_price)

            if win_norm:
                normal_wins += 1
            if win_inv:
                inverted_wins += 1

    overall_norm_wr = (normal_wins / total_trades) * 100
    overall_inv_wr = (inverted_wins / total_trades) * 100
    tie_pct = (tied_expiries / total_trades) * 100

    non_tied_trades = total_trades - tied_expiries
    non_tied_norm_wr = (normal_wins / non_tied_trades * 100) if non_tied_trades > 0 else 0.0
    non_tied_inv_wr = (inverted_wins / non_tied_trades * 100) if non_tied_trades > 0 else 0.0

    print("=" * 85)
    print(f"EMA(9,21) + RSI INVERSION & TIE DIAGNOSTIC: {label}")
    print(f"Dataset File: {csv_filename} ({len(df)} candles) | Expiry: {expiry_periods} period(s)")
    print("=" * 85)
    print(f"  Total Trades Executed:              {total_trades}")
    print(f"  Tied Expiries (Both Lose):         {tied_expiries} ({tie_pct:.2f}% of trades)")
    print(f"  Non-Tied Trades (Price Moved):      {non_tied_trades} ({100 - tie_pct:.2f}% of trades)")
    print("-" * 85)
    print(f"  ALL TRADES (Including Ties):")
    print(f"    Original Strategy Win Rate:       {overall_norm_wr:.2f}% ({normal_wins}/{total_trades})")
    print(f"    Inverted Strategy Win Rate:       {overall_inv_wr:.2f}% ({inverted_wins}/{total_trades})")
    print(f"    Tie Loss Rate:                    {tie_pct:.2f}% ({tied_expiries}/{total_trades})")
    print(f"    SUM (Original + Inverted + Ties): {overall_norm_wr + overall_inv_wr + tie_pct:.2f}% (PROVES 100% PARITY)")
    print("-" * 85)
    print(f"  EXCLUDING TIES (Pure Non-Flat Expiries):")
    print(f"    Original Non-Tied Win Rate:       {non_tied_norm_wr:.2f}% ({normal_wins}/{non_tied_trades})")
    print(f"    Inverted Non-Tied Win Rate:       {non_tied_inv_wr:.2f}% ({inverted_wins}/{non_tied_trades})")
    print(f"    SUM (Original + Inverted Non-Tied): {non_tied_norm_wr + non_tied_inv_wr:.2f}% (EXACT 100% FLIP PROOF)")
    print("=" * 85 + "\n")


def main():
    audit_ema_rsi_strategy("real_data_1m.csv", "EURUSD 1-Minute Real Candles", expiry_periods=1)
    audit_ema_rsi_strategy("real_data_15m.csv", "EURUSD 15-Minute Real Candles", expiry_periods=1)


if __name__ == "__main__":
    main()

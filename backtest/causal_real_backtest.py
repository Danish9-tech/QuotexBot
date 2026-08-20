"""
causal_real_backtest.py — Strictly causally-valid backtest on full uncompressed real datasets:
1. Keeps full, un-purged CSV history (real_data_1m.csv & real_data_15m.csv).
2. Uses live causal filter in strategy (skips trade if entry candle c_close == p_close).
3. Evaluates fixed expiry at exactly 1 real candle later on the full timeline.
4. Counts exact price ties at expiry as LOSSES (tie_is_loss=True).
"""

import os
import pandas as pd
from data import load_csv
from engine import BacktestEngine
from strategies import sureshot_quant_pro
from metrics import compute_metrics, print_report, breakeven_win_rate


def run_causal_backtest(csv_filename: str, label: str):
    file_path = os.path.join("backtest", csv_filename)
    if not os.path.exists(file_path):
        print(f"Error: {file_path} missing.")
        return

    df = load_csv(file_path)
    total_candles = len(df)
    payout_pct = 0.85
    needed = breakeven_win_rate(payout_pct)

    print("=" * 85)
    print(f"CAUSALLY-VALID REAL MARKET BACKTEST: {label}")
    print(f"Dataset File: {csv_filename} ({total_candles} full uncompressed candles)")
    print("=" * 85)

    engine = BacktestEngine(payout_pct=payout_pct, tie_is_loss=True)
    result = engine.run(
        df,
        sureshot_quant_pro,
        expiry_periods=1,
        start_balance=1000.0,
        stake_mode="fixed_pct",
        stake_value=0.01,
        cooldown_periods=0
    )

    trades_df = result.to_df()
    metrics = compute_metrics(trades_df, 1000.0, payout_pct)
    print_report(metrics)

    # Inversion & Tie Analysis on Full Timeline
    normal_wins = 0
    inverted_wins = 0
    tied_expiries = 0
    total_trades = len(trades_df)

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

    overall_wr = metrics.get("win_rate", 0.0) * 100
    inverted_wr = (inverted_wins / total_trades * 100) if total_trades > 0 else 0.0
    tie_pct = (tied_expiries / total_trades * 100) if total_trades > 0 else 0.0

    edge = overall_wr - (needed * 100)

    print(f"CAUSAL SELECTIVITY & TIE ANALYSIS:")
    print(f"  Total Dataset Candles:           {total_candles}")
    print(f"  Trades Taken:                    {total_trades} ({(total_trades / (total_candles - 24))*100:.2f}% of candles)")
    print(f"  Candles Returning None:          {total_candles - 24 - total_trades} ({((total_candles - 24 - total_trades) / (total_candles - 24))*100:.2f}% SKIPPED)")
    print(f"  Tied Expiries (Both Lose):       {tied_expiries} ({tie_pct:.2f}% of trades)")
    print("-" * 85)
    print(f"  Original Strategy Win Rate:      {overall_wr:.2f}%")
    print(f"  Inverted Strategy Win Rate:      {inverted_wr:.2f}%")
    print(f"  Breakeven Win Rate Needed:       {needed*100:.2f}%")
    print(f"  NET EDGE VS BREAKEVEN:           {edge:+6.2f} percentage points -> {'PROFITABLE' if edge > 0 else 'LOSING'}")
    print("=" * 85 + "\n")


def main():
    run_causal_backtest("real_data_1m.csv", "EURUSD 1-Minute Real Candles (Full 7-Day Timeline)")
    run_causal_backtest("real_data_15m.csv", "EURUSD 15-Minute Real Candles (Full 60-Day Timeline)")


if __name__ == "__main__":
    main()

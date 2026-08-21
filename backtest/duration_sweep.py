"""
duration_sweep.py — Expiry duration sweep script across real market datasets:
Tests sureshot_quant_pro, ema_crossover, rsi_reversal, bollinger_reversal:
1. real_data_1m.csv with expiry_periods = [1, 2, 3, 5, 10, 15] (1m, 2m, 3m, 5m, 10m, 15m expiries)
2. real_data_15m.csv with expiry_periods = [1, 2, 4] (15m, 30m, 60m expiries)

Note: Sub-1-minute durations (5s / 15s / 30s) cannot be backtested as no independent real market data exists for sub-minute ticks.
"""

import os
import pandas as pd
from data import load_csv
from engine import BacktestEngine
from strategies import sureshot_quant_pro, ema_crossover, rsi_reversal, bollinger_reversal
from metrics import compute_metrics, breakeven_win_rate

PAYOUT_PCT = 0.85
BREAKEVEN_WR = breakeven_win_rate(PAYOUT_PCT) * 100  # 54.05%


def run_sweep_for_dataset(csv_filename: str, label: str, durations: list[tuple[int, str]]):
    file_path = os.path.join("backtest", csv_filename)
    if not os.path.exists(file_path):
        print(f"Error: {file_path} missing.")
        return

    df = load_csv(file_path)
    engine = BacktestEngine(payout_pct=PAYOUT_PCT, tie_is_loss=True)

    strategies = [
        ("sureshot_quant_pro", sureshot_quant_pro),
        ("ema_crossover", ema_crossover),
        ("rsi_reversal", rsi_reversal),
        ("bollinger_reversal", bollinger_reversal),
    ]

    print("=" * 95)
    print(f"EXPIRY DURATION SWEEP: {label}")
    print(f"Dataset File: {csv_filename} ({len(df)} candles) | Breakeven Line: {BREAKEVEN_WR:.2f}%")
    print("=" * 95)
    print(f"{'Strategy Name':22s} | {'Expiry Period':15s} | {'Trades':7s} | {'Win Rate':9s} | {'Edge vs 54.05% BE':18s} | {'Verdict'}")
    print("-" * 95)

    for strat_name, strat_fn in strategies:
        for periods, duration_label in durations:
            result = engine.run(
                df,
                strat_fn,
                expiry_periods=periods,
                start_balance=1000.0,
                stake_mode="fixed_pct",
                stake_value=0.01,
                cooldown_periods=0
            )

            trades_df = result.to_df()
            if len(trades_df) == 0:
                print(f"{strat_name:22s} | {duration_label:15s} | {'0':7s} | {'0.00%':9s} | {'N/A':18s} | NO TRADES")
                continue

            metrics = compute_metrics(trades_df, 1000.0, PAYOUT_PCT)
            wr = metrics.get("win_rate", 0.0) * 100
            n_trades = metrics.get("n_trades", 0)
            edge = wr - BREAKEVEN_WR

            verdict = "PROFITABLE" if edge > 0 else "LOSING"
            print(f"{strat_name:22s} | {duration_label:15s} | {n_trades:7d} | {wr:8.2f}% | {edge:+17.2f} pts | {verdict}")

        print("-" * 95)

    print("=" * 95 + "\n")


def main():
    print("\n" + "#" * 95)
    print(" [NOTICE] SUB-1-MINUTE EXPIRY DISCLAIMER:")
    print(" Sub-1-minute durations (e.g. 5s, 15s, 30s) CANNOT be backtested against real market data.")
    print("No independent public exchange feed records historical tick-level 5s/15s/30s binary data.")
    print("#" * 95 + "\n")

    m1_durations = [
        (1, "1 min (1 pd)"),
        (2, "2 min (2 pds)"),
        (3, "3 min (3 pds)"),
        (5, "5 min (5 pds)"),
        (10, "10 min (10 pds)"),
        (15, "15 min (15 pds)"),
    ]
    run_sweep_for_dataset("real_data_1m.csv", "EURUSD 1-Minute Real Candles", m1_durations)

    m15_durations = [
        (1, "15 min (1 pd)"),
        (2, "30 min (2 pds)"),
        (4, "60 min (4 pds)"),
    ]
    run_sweep_for_dataset("real_data_15m.csv", "EURUSD 15-Minute Real Candles", m15_durations)


if __name__ == "__main__":
    main()

"""
run_researched_backtest.py — Backtests published strategies_researched.py
using causally-valid methodology on uncompressed real_data_1m.csv and real_data_15m.csv.
"""

import os
import pandas as pd
from data import load_csv
from engine import BacktestEngine
from metrics import compute_metrics, breakeven_win_rate
from strategies_researched import (
    strategy_ema9_21_rsi_filter,
    strategy_bollinger_reversion_adx,
    strategy_macd_support_resistance,
    strategy_pinbar_at_level,
)

PAYOUT_PCT = 0.85
BREAKEVEN_WR = breakeven_win_rate(PAYOUT_PCT) * 100  # 54.05%


def run_researched_test(csv_filename: str, label: str, config: list):
    file_path = os.path.join("backtest", csv_filename)
    if not os.path.exists(file_path):
        print(f"Error: {file_path} missing.")
        return

    df = load_csv(file_path)
    engine = BacktestEngine(payout_pct=PAYOUT_PCT, tie_is_loss=True)

    print("=" * 95)
    print(f"RESEARCHED STRATEGIES REAL BACKTEST: {label}")
    print(f"Dataset File: {csv_filename} ({len(df)} candles) | Breakeven Line: {BREAKEVEN_WR:.2f}%")
    print("=" * 95)
    print(f"{'Strategy Name':35s} | {'Expiry Period':15s} | {'Trades':7s} | {'Win Rate':9s} | {'Edge vs 54.05% BE':18s} | {'Verdict'}")
    print("-" * 95)

    for strat_name, strat_fn, periods, duration_label in config:
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
            print(f"{strat_name:35s} | {duration_label:15s} | {'0':7s} | {'0.00%':9s} | {'N/A':18s} | NO TRADES")
            continue

        metrics = compute_metrics(trades_df, 1000.0, PAYOUT_PCT)
        wr = metrics.get("win_rate", 0.0) * 100
        n_trades = metrics.get("n_trades", 0)
        edge = wr - BREAKEVEN_WR

        verdict = "PROFITABLE" if edge > 0 else "LOSING"
        print(f"{strat_name:35s} | {duration_label:15s} | {n_trades:7d} | {wr:8.2f}% | {edge:+17.2f} pts | {verdict}")

    print("=" * 95 + "\n")


def main():
    m1_config = [
        ("strategy_ema9_21_rsi_filter", strategy_ema9_21_rsi_filter, 1, "1 min (1 pd)"),
        ("strategy_bollinger_reversion_adx", strategy_bollinger_reversion_adx, 1, "1 min (1 pd)"),
        ("strategy_macd_support_resistance", strategy_macd_support_resistance, 5, "5 min (5 pds)"),
        ("strategy_pinbar_at_level", strategy_pinbar_at_level, 1, "1 min (1 pd)"),
    ]
    run_researched_test("real_data_1m.csv", "EURUSD 1-Minute Real Candles", m1_config)

    m15_config = [
        ("strategy_ema9_21_rsi_filter", strategy_ema9_21_rsi_filter, 1, "15 min (1 pd)"),
        ("strategy_bollinger_reversion_adx", strategy_bollinger_reversion_adx, 1, "15 min (1 pd)"),
        ("strategy_macd_support_resistance", strategy_macd_support_resistance, 1, "15 min (1 pd)"),
        ("strategy_pinbar_at_level", strategy_pinbar_at_level, 1, "15 min (1 pd)"),
    ]
    run_researched_test("real_data_15m.csv", "EURUSD 15-Minute Real Candles", m15_config)


if __name__ == "__main__":
    main()

"""
run_real_backtest.py — Backtests sureshot_quant_pro on real forex market data:
1. backtest/real_data_1m.csv (1-minute candles, 1-candle expiry)
2. backtest/real_data_15m.csv (15-minute candles, 1-candle expiry)
"""

import os
from data import load_csv
from engine import BacktestEngine
from strategies import sureshot_quant_pro
from metrics import compute_metrics, print_report

PAYOUT_PCT = 0.85  # 85% payout on Quotex
START_BALANCE = 1000.0


def run_test_on_file(csv_filename: str, label: str):
    file_path = os.path.join("backtest", csv_filename)
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist. Run download_real_data.py first.")
        return

    df = load_csv(file_path)
    print(f"\n{'='*60}")
    print(f"REAL MARKET DATA BACKTEST: {label}")
    print(f"Dataset File: {csv_filename} ({len(df)} candles)")
    print(f"{'='*60}")

    engine = BacktestEngine(payout_pct=PAYOUT_PCT, tie_is_loss=True)
    result = engine.run(
        df,
        sureshot_quant_pro,
        expiry_periods=1,
        start_balance=START_BALANCE,
        stake_mode="fixed_pct",
        stake_value=0.01,
        cooldown_periods=0
    )

    trades_df = result.to_df()
    metrics = compute_metrics(trades_df, START_BALANCE, PAYOUT_PCT)
    print_report(metrics)

    # Sample size reliability check
    n_trades = metrics.get("n_trades", 0)
    if n_trades < 200:
        print(f"[WARNING] SAMPLE SIZE SMALL: Only {n_trades} trades executed. Samples under 200 trades may be statistically too small.")
    else:
        print(f"[VALID] SAMPLE SIZE VALID: {n_trades} trades executed (>= 200 threshold). Sample size is statistically reliable.")

    return metrics


def main():
    m1 = run_test_on_file("real_data_1m.csv", "EURUSD 1-Minute Real Candles (7-Day History)")
    m15 = run_test_on_file("real_data_15m.csv", "EURUSD 15-Minute Real Candles (60-Day History)")


if __name__ == "__main__":
    main()

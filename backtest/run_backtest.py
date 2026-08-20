"""
run_backtest.py — main entry point for offline backtesting.

Runs offline backtests on synthetic or historical market data.
"""

from data import generate_synthetic, load_csv
from engine import BacktestEngine
from strategies import sureshot_quant_pro, ema_crossover, rsi_reversal, bollinger_reversal, random_baseline
from metrics import compute_metrics, print_report
from report import plot_equity_curve
import functools

# ============================== CONFIG ==============================
USE_SYNTHETIC_DATA = True          # Set False once you have a real CSV file
CSV_PATH = "your_data.csv"         # used only if USE_SYNTHETIC_DATA = False

STRATEGY_NAME = "sureshot_pro"     # "sureshot_pro" | "ema_crossover" | "rsi_reversal" | "bollinger_reversal" | "random_baseline"
EXPIRY_PERIODS = 1                 # 1 candle expiry (matches 1m candle @ 1m trade duration or 5s candle @ 5s trade duration)
PAYOUT_PCT = 0.85                  # 85% payout on Quotex
START_BALANCE = 1000.0
STAKE_MODE = "fixed_pct"           # "fixed_pct" | "fixed_amount" | "martingale"
STAKE_VALUE = 0.01                 # 1% of balance per trade
COOLDOWN_PERIODS = 0
# ======================================================================

STRATEGIES = {
    "sureshot_pro": sureshot_quant_pro,
    "ema_crossover": functools.partial(ema_crossover, fast=5, slow=20),
    "rsi_reversal": functools.partial(rsi_reversal, period=14, low_th=30, high_th=70),
    "bollinger_reversal": functools.partial(bollinger_reversal, period=20, std_mult=2.0),
    "random_baseline": random_baseline,
}


def main():
    df = generate_synthetic(n=5000) if USE_SYNTHETIC_DATA else load_csv(CSV_PATH)
    print(f"Loaded {len(df)} candles.")

    strategy_fn = STRATEGIES[STRATEGY_NAME]
    engine = BacktestEngine(payout_pct=PAYOUT_PCT, tie_is_loss=True)

    result = engine.run(
        df, strategy_fn,
        expiry_periods=EXPIRY_PERIODS,
        start_balance=START_BALANCE,
        stake_mode=STAKE_MODE,
        stake_value=STAKE_VALUE,
        cooldown_periods=COOLDOWN_PERIODS,
    )

    trades_df = result.to_df()
    metrics = compute_metrics(trades_df, START_BALANCE, PAYOUT_PCT)
    print_report(metrics)

    if not trades_df.empty:
        out_png = f"equity_curve_{STRATEGY_NAME}.png"
        plot_equity_curve(trades_df, START_BALANCE, out_png, title=f"{STRATEGY_NAME} — {metrics['n_trades']} trades")
        print(f"\nSaved chart: {out_png}")


if __name__ == "__main__":
    main()

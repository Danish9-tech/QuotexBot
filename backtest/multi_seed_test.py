"""
multi_seed_test.py — Statistical edge validation across multiple independent random datasets.

Runs strategies across 20 independent datasets and reports the average win rate,
expectancy, and percentage of net-profitable runs.
"""

import statistics
import functools
from data import generate_synthetic
from engine import BacktestEngine
from strategies import sureshot_quant_pro, ema_crossover, rsi_reversal, bollinger_reversal
from metrics import compute_metrics, breakeven_win_rate

N_SEEDS = 20
PAYOUT_PCT = 0.85  # 85% payout ratio on Quotex
STRATEGIES = {
    "sureshot_pro": sureshot_quant_pro,
    "ema_crossover": functools.partial(ema_crossover, fast=5, slow=20),
    "rsi_reversal": functools.partial(rsi_reversal, period=14, low_th=30, high_th=70),
    "bollinger_reversal": functools.partial(bollinger_reversal, period=20, std_mult=2.0),
}


def main():
    engine = BacktestEngine(payout_pct=PAYOUT_PCT)
    need = breakeven_win_rate(PAYOUT_PCT)
    print(f"Breakeven win rate at {PAYOUT_PCT*100:.0f}% payout: {need*100:.2f}%\n")

    for name, fn in STRATEGIES.items():
        win_rates, returns = [], []
        for seed in range(N_SEEDS):
            df = generate_synthetic(n=5000, seed=seed)
            result = engine.run(df, fn, expiry_periods=1, start_balance=1000,
                                 stake_mode="fixed_pct", stake_value=0.01)
            trades_df = result.to_df()
            if trades_df.empty:
                continue
            m = compute_metrics(trades_df, 1000, PAYOUT_PCT)
            win_rates.append(m["win_rate"])
            returns.append(m["total_return_pct"])

        if not win_rates:
            print(f"{name}: no trades generated across any seed.")
            continue

        mean_wr = statistics.mean(win_rates)
        std_wr = statistics.stdev(win_rates) if len(win_rates) > 1 else 0.0
        mean_ret = statistics.mean(returns)
        pct_profitable_runs = sum(r > 0 for r in returns) / len(returns) * 100

        print(f"--- {name} ({N_SEEDS} datasets) ---")
        print(f"  mean win rate:        {mean_wr*100:.2f}%  (std across runs: {std_wr*100:.2f} pts)")
        print(f"  mean return:          {mean_ret:+.2f}%")
        print(f"  runs that were net-profitable: {pct_profitable_runs:.0f}% of {len(returns)}")
        verdict = "no real edge shown" if mean_wr < need else "shows an edge — investigate further"
        print(f"  verdict: {verdict}\n")


if __name__ == "__main__":
    main()

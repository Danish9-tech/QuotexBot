"""
LLM signal layer validation runner.

Compares three strategies on real EURUSD 1-minute data:
  1. sureshot_quant_pro  (indicator stack — baseline)
  2. news_driven         (LLM proxy: calendar-gated, momentum direction)
  3. combined_signal     (indicator AND LLM agree)

Prints a comparison report and exits 0 only if combined_signal beats the
indicator-only baseline by a meaningful margin. This is the gate before
the LLM layer is wired into the live bot.

Usage:
  python -m backtest.llm_validation
"""

import sys
import os
from pathlib import Path

# Make project root importable so `from llm.backtest_strat import ...` works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.data import load_csv
from backtest.engine import BacktestEngine
from backtest.strategies import sureshot_quant_pro
from backtest.metrics import compute_metrics
from llm.backtest_strat import news_driven, combined_signal


CSV_PATH = str(PROJECT_ROOT / "backtest" / "real_data_1m.csv")
PAYOUT_PCT = 0.85
START_BALANCE = 1000.0
STAKE_VALUE = 0.01
COOLDOWN_PERIODS = 0
EXPIRY_PERIODS = 1

# Decision thresholds for go/no-go
COMBINED_WR_MIN = 0.53       # combined_signal must hit at least 53% WR
COMBINED_TRADES_MIN = 30     # on at least 30 trades
EDGE_IMPROVEMENT_MIN = 0.02  # combined WR must beat baseline WR by ≥ 2pp


def run_one(df, name, strategy_fn):
    engine = BacktestEngine(payout_pct=PAYOUT_PCT, tie_is_loss=True)
    result = engine.run(
        df, strategy_fn,
        expiry_periods=EXPIRY_PERIODS,
        start_balance=START_BALANCE,
        stake_mode="fixed_pct",
        stake_value=STAKE_VALUE,
        cooldown_periods=COOLDOWN_PERIODS,
    )
    trades_df = result.to_df()
    metrics = compute_metrics(trades_df, START_BALANCE, PAYOUT_PCT)
    return {"name": name, "metrics": metrics, "trades": len(trades_df)}


def fmt_pct(x): return f"{x*100:.2f}%" if x is not None else "—"
def fmt_dollar(x): return f"${x:+.2f}" if x is not None else "—"


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run backtest/download_real_data.py first.")
        sys.exit(1)

    print("=" * 78)
    print("  LLM SIGNAL LAYER VALIDATION")
    print("=" * 78)
    print(f"  Data: {CSV_PATH}")
    print(f"  Payout: {PAYOUT_PCT*100:.0f}%   Starting balance: ${START_BALANCE:.0f}")
    print(f"  Stake: {STAKE_VALUE*100:.1f}% of balance   Cooldown: {COOLDOWN_PERIODS} candles")
    print()

    df = load_csv(CSV_PATH)
    print(f"  Loaded {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()

    strategies = [
        ("sureshot_quant_pro", "Indicator stack (baseline)", sureshot_quant_pro),
        ("news_driven",        "LLM proxy (calendar + momentum)", news_driven),
        ("combined_signal",    "Indicator + LLM (both must agree)", combined_signal),
    ]

    results = []
    for name, desc, fn in strategies:
        try:
            r = run_one(df, name, fn)
            results.append((name, desc, r))
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")
            results.append((name, desc, None))

    # Print comparison table
    print("-" * 78)
    print(f"  {'Strategy':<40s} {'Trades':>7s} {'Win rate':>10s} {'PnL $':>10s} {'Max DD %':>10s}")
    print("-" * 78)
    for name, desc, r in results:
        if r is None:
            print(f"  {desc:<40s} {'ERR':>7s} {'-':>10s} {'-':>10s} {'-':>10s}")
            continue
        m = r["metrics"]
        n = m.get("n_trades", 0)
        wr = m.get("win_rate")
        pnl = (m.get("final_balance", 0) - m.get("start_balance", 0)) if n > 0 else 0
        dd = m.get("max_drawdown_pct")
        print(f"  {desc:<40s} {n:>7d} "
              f"{fmt_pct(wr):>10s} "
              f"{('+' if pnl >= 0 else '')}{pnl:>8.2f} "
              f"{(str(dd) + '%') if dd is not None else '-':>10s}")
    print("-" * 78)
    print()

    # Decision logic
    by_name = {r[0]: r[2] for r in results}
    base = by_name.get("sureshot_quant_pro")
    comb = by_name.get("combined_signal")

    if base is None or comb is None:
        print("ERROR: missing baseline or combined result; cannot decide.")
        sys.exit(1)

    base_wr = base["metrics"].get("win_rate") or 0.0
    comb_wr = comb["metrics"].get("win_rate") or 0.0
    comb_n = comb["trades"]

    print("GO/NO-GO DECISION")
    print("-" * 78)
    print(f"  Combined WR:        {fmt_pct(comb_wr)}  (min {fmt_pct(COMBINED_WR_MIN)})")
    print(f"  Combined trades:    {comb_n}  (min {COMBINED_TRADES_MIN})")
    print(f"  Baseline WR:        {fmt_pct(base_wr)}")
    print(f"  Edge over baseline: {fmt_pct(comb_wr - base_wr)}  (min {fmt_pct(EDGE_IMPROVEMENT_MIN)})")
    print()

    ok_wr = comb_wr >= COMBINED_WR_MIN
    ok_n = comb_n >= COMBINED_TRADES_MIN
    ok_edge = (comb_wr - base_wr) >= EDGE_IMPROVEMENT_MIN

    if ok_wr and ok_n and ok_edge:
        print("  [PASS] combined signal meets all thresholds")
        print("  Safe to enable LLM signal layer in bot.py (set LLM_REQUIRED=true).")
        sys.exit(0)
    else:
        reasons = []
        if not ok_wr: reasons.append(f"WR {fmt_pct(comb_wr)} < {fmt_pct(COMBINED_WR_MIN)}")
        if not ok_n:   reasons.append(f"trades {comb_n} < {COMBINED_TRADES_MIN}")
        if not ok_edge: reasons.append(f"edge {fmt_pct(comb_wr - base_wr)} < {fmt_pct(EDGE_IMPROVEMENT_MIN)}")
        print(f"  [FAIL] {'; '.join(reasons)}")
        print("  Do NOT enable LLM signal layer live. The abstention + confirmation logic")
        print("  does not produce measurable edge on this dataset.")
        sys.exit(1)


if __name__ == "__main__":
    main()

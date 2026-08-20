# backtest — honest binary-option strategy backtester

A Python framework for testing whether a Quotex-style (call/put, fixed expiry) strategy actually has an edge, before you risk any money on it.

## Quick start

```bash
python backtest/run_backtest.py          # runs on synthetic data out of the box
python backtest/multi_seed_test.py       # THE important one — test across multiple seeds
```

## Files

| File | What it does |
|---|---|
| `data.py` | Load CSV data, generate synthetic test data, or fetch real data (Binance/yfinance helpers included) |
| `engine.py` | The simulator. Enforces no look-ahead, applies realistic win/loss payout asymmetry |
| `strategies.py` | Strategy functions including `sureshot_quant_pro` (ported from live bot), EMA crossover, RSI, Bollinger |
| `metrics.py` | Win rate, breakeven win rate, expectancy, drawdown, profit factor |
| `report.py` | Equity curve chart generator |
| `run_backtest.py` | Single-run CLI entry point |
| `multi_seed_test.py` | Multi-dataset statistical validation across 20 seeds |

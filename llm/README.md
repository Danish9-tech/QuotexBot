# LLM Signal Layer

Real LLM-based signal layer for the QuotexAutoTradeBot. **Not currently wired into the live bot.** See "Status" below.

## What it is

A two-layer signal system:

1. **Layer 1**: the existing indicator stack (`groq_analyzer.get_groq_trading_signal`)
2. **Layer 2**: this LLM signal layer (`llm.signal.get_llm_trading_signal`)

Both must agree on direction for a trade to be placed. The LLM explicitly abstains when there's no information edge.

## How the LLM signal works

1. Fetches upcoming **economic calendar** events for the asset's currencies (Forex Factory RSS, free, 1-hour cache).
2. Fetches **recent news headlines** (NewsAPI free tier, 100 req/day, 15-min cache).
3. **Hard abstain** if no high-impact event is in the next 15 minutes.
4. Otherwise, builds a prompt with: last 5 candles, simple indicators (close, SMA5, SMA20), news headlines + sentiment, upcoming events.
5. Sends prompt to **Groq** (model: `llama-3.3-70b-versatile`, default; configurable via `LLM_MODEL`).
6. Parses the LLM's one-line response (CALL/PUT/ABSTAIN + confidence).
7. Returns `{signal, confidence, reason, events_nearby, news_count}`.

The prompt teaches the LLM to default to ABSTAIN. The system prompt is:

> "ABSTAIN is the correct answer most of the time. Only return CALL or PUT when there is a clear, scheduled catalyst and you have at least 60% confidence."

## Files

| File | Purpose |
|---|---|
| `calendar.py` | Forex Factory RSS fetcher with TTL cache |
| `news.py` | NewsAPI fetcher with TTL cache + coarse sentiment |
| `prompts.py` | Prompt template + response parser |
| `signal.py` | Live signal function (callable from bot.py) |
| `backtest_strat.py` | Backtest-compatible version |

## Status: NOT WIRED INTO BOT.PY

The backtest validation gate (`backtest/llm_validation.py`) was run on the existing `backtest/real_data_1m.csv` and **FAILED** on 2026-08-29.

```
Strategy                                  Trades   Win rate
Indicator stack (baseline)                  3944     33.16%
LLM proxy (calendar + momentum)             2874     19.55%
Indicator + LLM (both must agree)            588     31.80%   <- 1.36pp WORSE than baseline
```

The combined signal produced 588 trades at 31.80% WR, which is below the 54.05% breakeven AND below the indicator-only baseline. The plan's go/no-go rule was: combined WR must be ≥ 53% AND edge over baseline ≥ 2pp. Neither was met.

**Conclusion:** the LLM signal layer is committed as a working module but not enabled. Re-enable only after:
- Better market data (the current dataset has long flat segments)
- A proper historical news archive (NewsAPI free tier is current-only)
- Re-running the validation and passing the gate

## To enable later (when validation passes)

1. Add to `.env`:
   ```
   NEWSAPI_KEY=<your-newsapi-key>     # free tier at newsapi.org
   LLM_MODEL=llama-3.3-70b-versatile
   LLM_TIMEOUT_SECONDS=10
   LLM_REQUIRED=false                  # start in shadow mode
   LLM_MIN_CONFIDENCE=60
   ```
2. Wire into `bot.py` per the approved plan in `~/.claude/plans/zany-humming-crayon.md`.
3. Re-run `python -m backtest.llm_validation` after every data refresh.

## Honest limits

- NewsAPI free tier: 100 requests/day. At 15-min cache and ~20 assets, this is roughly 1,920 fetches per day worst case — over budget. Production should use a paid tier or RSS-only.
- Forex Factory RSS feed format can change. The XML parser is best-effort.
- The LLM has no information edge on tick-level price action. It only adds value when there's a scheduled event.
- The prompt parser accepts loose formats ("CALL with 72% confidence") but a sufficiently weird LLM response will trigger ABSTAIN.

## What did NOT happen

- `bot.py` was not modified. The live bot still uses the indicator stack only.
- `groq_analyzer.py` was not modified.
- `dashboard/` was not modified. No new LLM-specific tiles.
- The dashboard's Pause/Resume buttons still work as before.
- The auto kill switch and Telegram pings remain disabled.

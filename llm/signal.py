"""
Live LLM signal function.

Pipeline:
  1. Build asset context (currencies, news queries)
  2. Fetch events (cached) and news (cached) in parallel
  3. If no event in next 15 min -> abstain
  4. Build prompt
  5. Call Groq with timeout
  6. Parse response
  7. Return {signal, confidence, reason, ...}

All failures are fail-soft: returns ABSTAIN with a reason.
"""

import os
import asyncio
import logging
import time
from typing import Any

import aiohttp
from dotenv import load_dotenv

from . import calendar, news, prompts

load_dotenv()
logger = logging.getLogger("llm.signal")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "10"))
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

# Concurrency limit — don't hammer the API
_semaphore = asyncio.Semaphore(int(os.getenv("LLM_MAX_CONCURRENT", "5")))


def _summary_candles(candles: list[dict]) -> list[dict]:
    """Compress the full candle list to a 5-candle summary for the prompt."""
    out = []
    for c in candles[-5:]:
        if not isinstance(c, dict):
            continue
        out.append({
            "time": c.get("time") or c.get("timestamp") or "",
            "open": c.get("open") or c.get("o"),
            "high": c.get("high") or c.get("h"),
            "low": c.get("low") or c.get("l"),
            "close": c.get("close") or c.get("c"),
        })
    return out


def _extract_indicators(candles: list[dict]) -> dict:
    """
    Lightweight indicator summary for the prompt.

    We don't reimplement the full indicator stack here — the bot's main
    `get_groq_trading_signal` already computes them and the bot.py integration
    passes them in via `candles` keys if present, OR we compute cheap
    summaries (close vs simple moving average) here.
    """
    if not candles:
        return {}
    closes = []
    for c in candles:
        if isinstance(c, dict):
            v = c.get("close") or c.get("c")
            if v is not None:
                closes.append(float(v))
    if len(closes) < 5:
        return {"close_count": len(closes)}

    last = closes[-1]
    sma5 = sum(closes[-5:]) / 5
    sma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else None
    range_pct = (max(closes[-20:]) - min(closes[-20:])) / last * 100 if len(closes) >= 20 else None

    out = {
        "last_close": round(last, 5),
        "sma5": round(sma5, 5),
        "sma20": round(sma20, 5) if sma20 is not None else None,
        "above_sma5": last > sma5,
        "above_sma20": last > sma20 if sma20 is not None else None,
    }
    if range_pct is not None:
        out["recent_range_pct"] = round(range_pct, 3)
    return out


async def _call_groq(prompt: str) -> str:
    """Call Groq Chat Completions API and return the model's text content."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a conservative trading signal assistant. "
                    "ABSTAIN is the correct answer most of the time. "
                    "Only return CALL or PUT when there is a clear, scheduled catalyst "
                    "and you have at least 60% confidence."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,  # low temperature for deterministic abstention
        "max_tokens": 80,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    timeout = aiohttp.ClientTimeout(total=LLM_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(GROQ_CHAT_URL, json=payload, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Groq HTTP {resp.status}: {body[:200]}")
            data = await resp.json()

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Groq returned no choices")
    msg = choices[0].get("message") or {}
    return (msg.get("content") or "").strip()


async def get_llm_trading_signal(candles: list, asset_name: str) -> dict:
    """
    Compute an LLM-based trading signal for `asset_name`.

    Returns a dict with the same shape as the indicator-stack signal:
        {signal: "call"|"put"|"abstain", confidence: int, reason: str,
         events_nearby: int, news_count: int}

    `signal` is the LLM's call in {"call", "put", "abstain"}. "abstain" is
    the same as "doji" in the indicator stack — no trade.
    """
    # 1. Fetch calendar + news in parallel
    try:
        events_task = calendar.fetch_events(asset_name)
        news_task = news.fetch_news(asset_name)
        events, articles = await asyncio.gather(events_task, news_task, return_exceptions=True)
        if isinstance(events, Exception):
            events = []
        if isinstance(articles, Exception):
            articles = []
    except Exception as e:
        logger.warning(f"LLM data fetch failed for {asset_name}: {e}")
        events, articles = [], []

    # 2. Hard abstain if no event in next 15 min — this is the abstention discipline
    events_nearby = sum(1 for e in events if e.get("hours_away", 999) * 60 <= 15)
    if events_nearby == 0:
        return {
            "signal": "abstain",
            "confidence": 0,
            "reason": "no scheduled event in next 15 min",
            "events_nearby": 0,
            "news_count": len(articles),
        }

    # 3. Build prompt
    sentiment = news.aggregate_sentiment(articles)
    candle_summary = _summary_candles(candles)
    indicators = _extract_indicators(candles)
    prompt = prompts.build_signal_prompt(
        asset=asset_name,
        candles_summary=candle_summary,
        indicators=indicators,
        news_articles=articles,
        events=events,
        sentiment=sentiment,
    )

    # 4. Call LLM with timeout + concurrency control
    try:
        async with _semaphore:
            t0 = time.time()
            text = await _call_groq(prompt)
            elapsed = time.time() - t0
            logger.debug(f"LLM response in {elapsed:.2f}s: {text[:100]}")
    except asyncio.TimeoutError:
        return {
            "signal": "abstain",
            "confidence": 0,
            "reason": f"LLM timeout after {LLM_TIMEOUT_SECONDS}s",
            "events_nearby": events_nearby,
            "news_count": len(articles),
        }
    except Exception as e:
        logger.warning(f"LLM call failed for {asset_name}: {e}")
        return {
            "signal": "abstain",
            "confidence": 0,
            "reason": f"LLM error: {str(e)[:60]}",
            "events_nearby": events_nearby,
            "news_count": len(articles),
        }

    # 5. Parse
    parsed = prompts.parse_llm_response(text)

    return {
        "signal": parsed["signal"],
        "confidence": parsed["confidence"],
        "reason": parsed["reason"],
        "events_nearby": events_nearby,
        "news_count": len(articles),
    }

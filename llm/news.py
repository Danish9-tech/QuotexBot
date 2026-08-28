"""
News fetcher for the LLM signal layer.

Uses NewsAPI.org (free tier: 100 requests/day, current-only). If no API key
is set, returns an empty list — the LLM still works with the calendar alone.

Sentiment is a coarse keyword count, NOT a real ML model. Good enough for
the LLM prompt; the LLM is doing the real interpretation.

Asset -> query keywords mirror the calendar mapping. Each asset maps to a
short list of high-signal search terms.
"""

import os
import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

logger = logging.getLogger("llm.news")

NEWSAPI_URL = "https://newsapi.org/v2/everything"
CACHE_TTL_SECONDS = 15 * 60  # 15 min — news changes faster than calendar
FETCH_TIMEOUT_SECONDS = 10
NEWS_LOOKBACK_HOURS = 6  # only consider news from the last 6 hours
NEWS_MAX_RESULTS = 10

# Coarse sentiment lexicon. Counts are bounded to ±3.
POSITIVE_WORDS = {
    "rally", "surge", "jump", "climb", "rise", "gain", "beat", "beats",
    "strong", "growth", "expand", "expansion", "boost", "optimistic",
    "hawkish", "tighten", "hike", "raise", "upgrade", "record high",
}
NEGATIVE_WORDS = {
    "fall", "drop", "plunge", "sink", "slip", "decline", "miss", "misses",
    "weak", "soft", "slow", "shrink", "contraction", "recession", "cut",
    "dovish", "ease", "lower", "downgrade", "record low", "crisis",
}


ASSET_QUERIES: dict[str, list[str]] = {
    "EURUSD": ["EUR", "euro", "ECB", "European Central Bank", "eurozone"],
    "EURUSD_otc": ["EUR", "euro", "ECB", "European Central Bank", "eurozone"],
    "GBPUSD": ["GBP", "pound sterling", "Bank of England", "BoE", "UK economy"],
    "GBPUSD_otc": ["GBP", "pound sterling", "Bank of England", "BoE", "UK economy"],
    "USDJPY": ["USD", "dollar", "Federal Reserve", "Fed", "Bank of Japan", "BoJ", "yen"],
    "USDJPY_otc": ["USD", "dollar", "Federal Reserve", "Fed", "Bank of Japan", "BoJ", "yen"],
    "USDCHF": ["USD", "dollar", "Swiss National Bank", "SNB", "Swiss franc"],
    "USDCHF_otc": ["USD", "dollar", "Swiss National Bank", "SNB", "Swiss franc"],
    "AUDUSD": ["AUD", "Australian dollar", "RBA", "Reserve Bank of Australia"],
    "AUDUSD_otc": ["AUD", "Australian dollar", "RBA", "Reserve Bank of Australia"],
    "USDCAD": ["USD", "dollar", "Bank of Canada", "BoC", "Canadian dollar", "oil prices"],
    "USDCAD_otc": ["USD", "dollar", "Bank of Canada", "BoC", "Canadian dollar", "oil prices"],
    "NZDUSD": ["NZD", "kiwi", "RBNZ", "Reserve Bank of New Zealand"],
    "NZDUSD_otc": ["NZD", "kiwi", "RBNZ", "Reserve Bank of New Zealand"],
    "XAUUSD": ["gold", "bullion", "Federal Reserve", "Fed", "inflation", "CPI"],
    "XAUUSD_otc": ["gold", "bullion", "Federal Reserve", "Fed", "inflation", "CPI"],
    "Gold": ["gold", "bullion", "Federal Reserve", "Fed", "inflation", "CPI"],
    "Gold_otc": ["gold", "bullion", "Federal Reserve", "Fed", "inflation", "CPI"],
    "XAGUSD": ["silver", "bullion", "industrial metals", "Fed", "inflation"],
    "XAGUSD_otc": ["silver", "bullion", "industrial metals", "Fed", "inflation"],
    "Silver": ["silver", "bullion", "industrial metals", "Fed", "inflation"],
    "Silver_otc": ["silver", "bullion", "industrial metals", "Fed", "inflation"],
    "UKBrent": ["Brent crude", "oil prices", "OPEC", "Middle East"],
    "UKBrent_otc": ["Brent crude", "oil prices", "OPEC", "Middle East"],
    "USCrude": ["WTI crude", "oil prices", "OPEC", "Middle East", "EIA crude"],
    "USCrude_otc": ["WTI crude", "oil prices", "OPEC", "Middle East", "EIA crude"],
}


def _coarse_sentiment(text: str) -> int:
    """Return a bounded sentiment score in {-3, -2, -1, 0, 1, 2, 3}."""
    t = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in t)
    neg = sum(1 for w in NEGATIVE_WORDS if w in t)
    raw = pos - neg
    return max(-3, min(3, raw))


class _NewsCache:
    """One global cache; rate-limited by CACHE_TTL_SECONDS."""

    def __init__(self):
        self._by_query: dict[str, list[dict]] = {}
        self._fetched_at: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def get(self, query: str, ttl: int = CACHE_TTL_SECONDS) -> list[dict]:
        async with self._lock:
            now = time.time()
            last = self._fetched_at.get(query, 0)
            if query in self._fetched_at and (now - last) < ttl:
                return self._by_query.get(query, [])

            api_key = os.getenv("NEWSAPI_KEY", "")
            if not api_key:
                # Without an API key, return empty rather than spam log
                self._by_query[query] = []
                self._fetched_at[query] = now
                return []

            articles = await _fetch_newsapi(query, api_key)
            self._by_query[query] = articles
            self._fetched_at[query] = now
            return articles

    def invalidate(self):
        self._by_query.clear()
        self._fetched_at.clear()


_cache = _NewsCache()


async def _fetch_newsapi(query: str, api_key: str) -> list[dict]:
    """Call NewsAPI and return normalized articles."""
    from_iso = (datetime.now(timezone.utc) - timedelta(hours=NEWS_LOOKBACK_HOURS)).isoformat()
    params = {
        "q": query,
        "from": from_iso,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": str(NEWS_MAX_RESULTS),
    }
    headers = {"Authorization": api_key, "User-Agent": "QuotexBot/1.0"}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT_SECONDS)) as session:
            async with session.get(NEWSAPI_URL, params=params, headers=headers) as resp:
                if resp.status == 429:
                    logger.warning("NewsAPI rate-limited (429)")
                    return []
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(f"NewsAPI HTTP {resp.status}: {body[:200]}")
                    return []
                data = await resp.json()
    except Exception as e:
        logger.warning(f"NewsAPI fetch exception: {e}")
        return []

    articles = data.get("articles") or []
    out = []
    for a in articles:
        title = (a.get("title") or "").strip()
        if not title or "[Removed]" in title:
            continue
        published = a.get("publishedAt") or ""
        source = (a.get("source") or {}).get("name", "")
        out.append({
            "title": title,
            "source": source,
            "published_at": published,
            "sentiment": _coarse_sentiment(title + " " + (a.get("description") or "")),
        })
    return out


async def fetch_news(asset: str) -> list[dict]:
    """
    Return recent news articles affecting `asset` (deduplicated across queries).

    Returns a list of dicts: {title, source, published_at, sentiment}.
    Returns [] on any error or if NEWSAPI_KEY is not set.
    """
    queries = ASSET_QUERIES.get(asset, [])
    if not queries:
        return []

    # Fetch all queries in parallel
    results = await asyncio.gather(
        *[_cache.get(q) for q in queries],
        return_exceptions=True,
    )
    seen_titles: set[str] = set()
    out: list[dict] = []
    for r in results:
        if isinstance(r, Exception):
            continue
        for art in r:
            key = art["title"].lower()[:80]
            if key in seen_titles:
                continue
            seen_titles.add(key)
            out.append(art)
    out.sort(key=lambda a: a.get("published_at", ""), reverse=True)
    return out[:NEWS_MAX_RESULTS]


def aggregate_sentiment(articles: list[dict]) -> dict:
    """
    Aggregate sentiment from a list of news articles.

    Returns: {score: int (-3..3), n: int, breakdown: {pos, neg, neutral}}
    """
    if not articles:
        return {"score": 0, "n": 0, "breakdown": {"pos": 0, "neg": 0, "neutral": 0}}
    pos = sum(1 for a in articles if a.get("sentiment", 0) > 0)
    neg = sum(1 for a in articles if a.get("sentiment", 0) < 0)
    neutral = len(articles) - pos - neg
    avg = sum(a.get("sentiment", 0) for a in articles) / len(articles)
    return {
        "score": max(-3, min(3, round(avg))),
        "n": len(articles),
        "breakdown": {"pos": pos, "neg": neg, "neutral": neutral},
    }

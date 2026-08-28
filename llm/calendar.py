"""
Economic calendar fetcher.

Uses Forex Factory's free public XML feed (no API key required) to get
upcoming high-impact events. Filters to a per-asset currency list and
a time window around the current moment.

All fetches are cached in-process for `ttl_seconds` to avoid hammering
the feed. On any error returns an empty list — fail-soft.
"""

import os
import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

logger = logging.getLogger("llm.calendar")

FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
CACHE_TTL_SECONDS = 60 * 60  # 1 hour
FETCH_TIMEOUT_SECONDS = 10
MAX_WINDOW_HOURS = 4

# Asset name -> currencies that drive it.
# Default mapping covers the most common OTC pairs on Quotex.
ASSET_CURRENCIES: dict[str, list[str]] = {
    # Forex majors
    "EURUSD": ["EUR", "USD"], "EURUSD_otc": ["EUR", "USD"],
    "GBPUSD": ["GBP", "USD"], "GBPUSD_otc": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"], "USDJPY_otc": ["USD", "JPY"],
    "USDCHF": ["USD", "CHF"], "USDCHF_otc": ["USD", "CHF"],
    "AUDUSD": ["AUD", "USD"], "AUDUSD_otc": ["AUD", "USD"],
    "USDCAD": ["USD", "CAD"], "USDCAD_otc": ["USD", "CAD"],
    "NZDUSD": ["NZD", "USD"], "NZDUSD_otc": ["NZD", "USD"],
    "EURJPY": ["EUR", "JPY"], "EURJPY_otc": ["EUR", "JPY"],
    "EURGBP": ["EUR", "GBP"], "EURGBP_otc": ["EUR", "GBP"],
    "EURCAD": ["EUR", "CAD"], "EURCAD_otc": ["EUR", "CAD"],
    "GBPJPY": ["GBP", "JPY"], "GBPJPY_otc": ["GBP", "JPY"],
    "AUDJPY": ["AUD", "JPY"], "AUDJPY_otc": ["AUD", "JPY"],
    "AUDCHF": ["AUD", "CHF"], "AUDCHF_otc": ["AUD", "CHF"],
    "AUDCAD": ["AUD", "CAD"], "AUDCAD_otc": ["AUD", "CAD"],
    "CADJPY": ["CAD", "JPY"], "CADJPY_otc": ["CAD", "JPY"],
    "CADCHF": ["CAD", "CHF"], "CADCHF_otc": ["CAD", "CHF"],
    "CHFJPY": ["CHF", "JPY"], "CHFJPY_otc": ["CHF", "JPY"],
    "NZDCAD": ["NZD", "CAD"], "NZDCAD_otc": ["NZD", "CAD"],
    "NZDCHF": ["NZD", "CHF"], "NZDCHF_otc": ["NZD", "CHF"],
    "NZDJPY": ["NZD", "JPY"], "NZDJPY_otc": ["NZD", "JPY"],
    "EURAUD": ["EUR", "AUD"], "EURAUD_otc": ["EUR", "AUD"],
    "EURNZD": ["EUR", "NZD"], "EURNZD_otc": ["EUR", "NZD"],
    "GBPAUD": ["GBP", "AUD"], "GBPAUD_otc": ["GBP", "AUD"],
    "GBPCAD": ["GBP", "CAD"], "GBPCAD_otc": ["GBP", "CAD"],
    "GBPNZD": ["GBP", "NZD"], "GBPNZD_otc": ["GBP", "NZD"],
    # Commodities
    "XAUUSD": ["USD"], "XAUUSD_otc": ["USD"],  # gold
    "Gold": ["USD"], "Gold_otc": ["USD"],
    "XAGUSD": ["USD"], "XAGUSD_otc": ["USD"],  # silver
    "Silver": ["USD"], "Silver_otc": ["USD"],
    "UKBrent": ["USD"], "UKBrent_otc": ["USD"],
    "USCrude": ["USD"], "USCrude_otc": ["USD"],
    # Exotics (less reliable on Forex Factory but listed for completeness)
    "USDBRL": ["USD", "BRL"], "USDBRL_otc": ["USD", "BRL"],
    "USDZAR": ["USD", "ZAR"], "USDZAR_otc": ["USD", "ZAR"],
    "USDTRY": ["USD", "TRY"], "USDTRY_otc": ["USD", "TRY"],
    "USDMXN": ["USD", "MXN"], "USDMXN_otc": ["USD", "MXN"],
}


class _CalendarCache:
    """Single global cache; the bot is one process."""

    def __init__(self):
        self._events: list[dict] = []
        self._fetched_at: float = 0.0
        self._lock = asyncio.Lock()

    async def get(self, ttl: int = CACHE_TTL_SECONDS) -> list[dict]:
        async with self._lock:
            now = time.time()
            if self._events or (now - self._fetched_at) < ttl:
                return self._events
            self._events = await _fetch_ff_calendar()
            self._fetched_at = now
            return self._events

    def invalidate(self):
        self._fetched_at = 0.0
        self._events = []


_cache = _CalendarCache()


async def _fetch_ff_calendar() -> list[dict]:
    """Fetch the Forex Factory XML feed and parse it into a list of events."""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT_SECONDS)) as session:
            async with session.get(FF_URL, headers={"User-Agent": "QuotexBot/1.0"}) as resp:
                if resp.status != 200:
                    logger.warning(f"Forex Factory calendar HTTP {resp.status}")
                    return []
                body = await resp.text()
    except Exception as e:
        logger.warning(f"Forex Factory calendar fetch exception: {e}")
        return []

    return _parse_ff_xml(body)


def _parse_ff_xml(body: str) -> list[dict]:
    """
    Minimal XML parser for Forex Factory's calendar.

    The feed is structured like:
      <weeklyevents>
        <event>
          <title>...</title>
          <country>USD</country>
          <date>...</date>      (e.g. "08-29-2026" or ISO)
          <time>...</time>      (e.g. "8:30am" or ISO)
          <impact>High</impact>
          <forecast>...</forecast>
          <previous>...</previous>
        </event>
        ...
      </weeklyevents>
    """
    import xml.etree.ElementTree as ET
    events: list[dict] = []
    try:
        root = ET.fromstring(body)
    except ET.ParseError as e:
        logger.warning(f"Forex Factory XML parse error: {e}")
        return []

    for ev in root.findall("event"):
        try:
            title = (ev.findtext("title") or "").strip()
            country = (ev.findtext("country") or "").strip().upper()
            date_str = (ev.findtext("date") or "").strip()
            time_str = (ev.findtext("time") or "").strip()
            impact = (ev.findtext("impact") or "").strip()

            if not title or not country:
                continue

            dt = _parse_ff_datetime(date_str, time_str)
            if dt is None:
                continue

            events.append({
                "title": title,
                "currency": country,
                "time_utc": dt,
                "impact": impact,
            })
        except Exception:
            continue

    return events


def _parse_ff_datetime(date_str: str, time_str: str) -> datetime | None:
    """
    Parse Forex Factory date + time strings into a UTC datetime.

    FF uses several formats; handle the common ones:
      date: "08-29-2026" or "2026-08-29" or ISO
      time: "8:30am" or "13:30" or "All Day" (skip) or "" (skip)
    Times without a timezone suffix are assumed ET (FF's default).
    """
    if not time_str or "all day" in time_str.lower():
        return None  # all-day events are rarely market-moving

    # Date parsing
    dt = None
    for fmt in ("%m-%d-%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            dt = datetime.strptime(date_str, fmt)
            break
        except ValueError:
            continue
    if dt is None:
        return None

    # Time parsing
    t = None
    for fmt in ("%I:%M%p", "%H:%M", "%I:%M %p"):
        try:
            t = datetime.strptime(time_str.strip().lower(), fmt.lower()).time()
            break
        except ValueError:
            continue
    if t is None:
        return None

    # Assume ET (UTC-5 standard / UTC-4 DST). Approximate as UTC-4 (DST is in
    # effect during US summer which is when most market-moving events occur).
    # We accept ~1h imprecision for the LLM prompt.
    local = dt.combine(dt.date(), t)
    utc = local - timedelta(hours=4)
    return utc.replace(tzinfo=timezone.utc)


async def fetch_events(asset: str) -> list[dict]:
    """
    Return high-impact events affecting `asset` in the next MAX_WINDOW_HOURS hours.

    Returns a list of dicts with keys: title, currency, time_utc (datetime),
    impact, hours_away (float).
    """
    currencies = ASSET_CURRENCIES.get(asset)
    if not currencies:
        return []

    try:
        all_events = await _cache.get()
    except Exception as e:
        logger.warning(f"calendar cache get failed: {e}")
        return []

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=MAX_WINDOW_HOURS)

    out = []
    for ev in all_events:
        if ev["currency"] not in currencies:
            continue
        if ev["impact"].lower() != "high":
            continue
        if not (now <= ev["time_utc"] <= cutoff):
            continue
        out.append({
            **ev,
            "hours_away": round((ev["time_utc"] - now).total_seconds() / 3600, 2),
        })
    out.sort(key=lambda e: e["time_utc"])
    return out


async def has_nearby_event(asset: str, within_minutes: int = 15) -> dict | None:
    """
    Convenience: return the nearest upcoming high-impact event for `asset`
    within `within_minutes`, or None.
    """
    events = await fetch_events(asset)
    for ev in events:
        if ev["hours_away"] * 60 <= within_minutes:
            return ev
    return None

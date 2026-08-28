"""
Telegram alerts + kill-switch logic for the dashboard.

The kill switch is *polite* — it writes `service_status: False` to every `trade_settings`
doc in MongoDB. The live bot's existing `run_trading_loop_for_account` already checks this
field on every asset iteration (see bot.py:2352-2363) and stops itself. We don't import
anything from bot.py; we just write the same MongoDB state the bot already reads.
"""

import os
import asyncio
import logging
from typing import Any

import aiohttp
from dotenv import load_dotenv

from . import metrics

load_dotenv()
logger = logging.getLogger("dashboard.alerts")

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
OWNER_ID = os.getenv("OWNER_ID", "")

KILL_WR_THRESHOLD = float(os.getenv("KILL_WR_THRESHOLD", "52.0"))
KILL_STREAK_THRESHOLD = int(os.getenv("KILL_STREAK_THRESHOLD", "4"))
KILL_PNL_THRESHOLD = float(os.getenv("KILL_PNL_THRESHOLD", "-10.0"))
KILL_ROLLING_N = int(os.getenv("KILL_ROLLING_N", "50"))
KILL_STREAK_N = int(os.getenv("KILL_STREAK_N", "10"))


def evaluate_kill_switch(metrics_snapshot: dict) -> dict:
    """
    Decide whether the kill switch should fire. Pure function — no I/O.

    Returns: {triggered: bool, reasons: list[str], values: dict}
    """
    reasons: list[str] = []
    values: dict = {}

    r = metrics_snapshot.get("rolling_50", {})
    if r.get("n", 0) >= KILL_ROLLING_N:
        wr = r.get("win_rate", 0.0)
        values["rolling_winrate_50"] = wr
        if wr < KILL_WR_THRESHOLD:
            reasons.append(
                f"Win rate {wr:.1f}% < {KILL_WR_THRESHOLD:.1f}% over last {r['n']} trades"
            )

    s = metrics_snapshot.get("recent_10", {})
    if s.get("n", 0) >= KILL_STREAK_N:
        wins = s.get("wins", 0)
        values["last_10_wins"] = wins
        if wins < KILL_STREAK_THRESHOLD:
            reasons.append(
                f"Last {s['n']} trades have only {wins} wins (threshold {KILL_STREAK_THRESHOLD})"
            )

    p = metrics_snapshot.get("daily_pnl", {})
    pnl = p.get("net_pnl", 0.0)
    values["daily_pnl"] = pnl
    if pnl <= KILL_PNL_THRESHOLD:
        reasons.append(
            f"Daily PnL ${pnl:+.2f} <= ${KILL_PNL_THRESHOLD:+.2f}"
        )

    return {
        "triggered": bool(reasons),
        "reasons": reasons,
        "values": values,
        "thresholds": {
            "winrate_below": KILL_WR_THRESHOLD,
            "streak_below": KILL_STREAK_THRESHOLD,
            "pnl_below": KILL_PNL_THRESHOLD,
        },
    }


async def write_kill_switch(db, service_status: bool) -> int:
    """
    Flip `service_status` for every trade_settings doc. Returns count modified.

    bot.py:2352-2363 already uses this exact write. bot.py will pick it up on the next
    asset iteration of its trading loop.
    """
    trade_settings = db["trade_settings"]
    result = await trade_settings.update_many(
        {},
        {"$set": {"service_status": bool(service_status)}},
    )
    return result.modified_count


async def send_telegram_message(text: str) -> bool:
    """
    Send a Telegram message to OWNER_ID via the bot's BOT_TOKEN.

    Standalone HTTP — we don't share a connection with bot.py. This is intentional:
    the dashboard runner is a separate process, and bot.py's pyrogram client is
    bound to that process.
    """
    if not BOT_TOKEN or not OWNER_ID:
        logger.warning("BOT_TOKEN or OWNER_ID not set — skipping Telegram send")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": OWNER_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"Telegram send failed: {resp.status} {body[:200]}")
                    return False
                return True
    except Exception as e:
        logger.error(f"Telegram send exception: {e}")
        return False


def format_status_message(metrics_snapshot: dict, kill_state: dict) -> str:
    """Format a Telegram-ready status summary."""
    overall = metrics_snapshot.get("overall", {})
    r50 = metrics_snapshot.get("rolling_50", {})
    s10 = metrics_snapshot.get("recent_10", {})
    daily = metrics_snapshot.get("daily_pnl", {})
    dd = metrics_snapshot.get("drawdown", {})
    inv = metrics_snapshot.get("inversion", {})

    lines: list[str] = []
    if kill_state.get("triggered"):
        lines.append("🛑 *KILL SWITCH TRIGGERED*")
        for r in kill_state.get("reasons", []):
            lines.append(f"   • {r}")
        lines.append("")
    else:
        lines.append("✅ *Dashboard check — bot live, all green*")
        lines.append("")

    lines.append(f"*Win Rate (last 50)*: {r50.get('win_rate', 0):.2f}% "
                 f"(breakeven {r50.get('breakeven', 54.05):.2f}%, "
                 f"gap {r50.get('gap', 0):+.2f}pp)")
    lines.append(f"*Last 10*: {s10.get('wins', 0)}/{s10.get('n', 0)} wins")
    lines.append(f"*Daily PnL*: ${daily.get('net_pnl', 0):+.2f}")
    lines.append(f"*Drawdown*: ${dd.get('current_drawdown', 0):.2f} current / "
                 f"${dd.get('max_drawdown', 0):.2f} max")
    lines.append(f"*Net PnL (all-time)*: ${overall.get('net_pnl', 0):+.2f} "
                 f"({overall.get('wins', 0)}W / {overall.get('losses', 0)}L / "
                 f"{overall.get('ties', 0)}T, WR {overall.get('win_rate', 0):.2f}%)")
    lines.append("")
    edge = inv.get("edge_pct", 0.0)
    if inv.get("n", 0) >= 20:
        if abs(edge) < 2.0:
            edge_note = "≈ no edge (coin flip)"
        elif edge > 0:
            edge_note = f"+{edge:.2f}pp edge over inversion"
        else:
            edge_note = f"−{abs(edge):.2f}pp — strategy worse than flipping a coin"
        lines.append(f"*Inversion test* (last {inv['n']}): "
                     f"original {inv.get('original_winrate', 0):.2f}% vs "
                     f"inverted {inv.get('inverted_winrate', 0):.2f}% — {edge_note}")
    return "\n".join(lines)


def format_kill_switch_alert(kill_state: dict, metrics_snapshot: dict) -> str:
    """Format the Telegram alert sent the moment the kill switch fires."""
    lines = ["🛑🛑🛑 *KILL SWITCH TRIGGERED* 🛑🛑🛑", ""]
    lines.append("Bot has been paused automatically. Open the dashboard to resume.")
    lines.append("")
    lines.append("*Reasons:*")
    for r in kill_state.get("reasons", []):
        lines.append(f"   • {r}")
    lines.append("")
    overall = metrics_snapshot.get("overall", {})
    lines.append(f"*All-time:* {overall.get('wins', 0)}W / {overall.get('losses', 0)}L, "
                 f"WR {overall.get('win_rate', 0):.2f}%, "
                 f"Net PnL ${overall.get('net_pnl', 0):+.2f}")
    dd = metrics_snapshot.get("drawdown", {})
    lines.append(f"*Drawdown:* ${dd.get('current_drawdown', 0):.2f} current / "
                 f"${dd.get('max_drawdown', 0):.2f} max")
    return "\n".join(lines)

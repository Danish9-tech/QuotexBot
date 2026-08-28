"""
Background runner for the verification dashboard.

Mirrors ping.py: a single long-running process that does periodic work
(MongoDB checks + Telegram status pings + auto kill-switch). Runs alongside
bot.py and ping.py; started by Dockerfile / setup_systemd.sh.

Usage:
  python dashboard_runner.py            # run forever
  python dashboard_runner.py --once     # run a single check, then exit
                                       # (used by the smoke test)
"""

import os
import sys
import asyncio
import signal
import logging
from datetime import datetime, timezone

import motor.motor_asyncio
from dotenv import load_dotenv

from dashboard import metrics, alerts

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("dashboard.runner")

MONGO_URI = os.getenv("MONGO_URI", "")

CHECK_INTERVAL = int(os.getenv("DASHBOARD_CHECK_INTERVAL", "60"))
TELEGRAM_INTERVAL = int(os.getenv("DASHBOARD_TELEGRAM_INTERVAL", "900"))
TELEGRAM_ENABLED = os.getenv("DASHBOARD_TELEGRAM_ENABLED", "true").lower() in ("1", "true", "yes")


_running = True


def _install_signal_handlers(loop: asyncio.AbstractEventLoop):
    def stop(*_):
        global _running
        _running = False
        logger.info("signal received — shutting down dashboard runner")
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop)
        except NotImplementedError:
            # Windows: signal handlers not supported in the same way
            pass


async def run_check_once(db) -> dict:
    """
    One iteration: build metrics, evaluate kill switch, act on it,
    optionally send a Telegram status ping.
    Returns the kill-switch state dict.
    """
    snapshot = await metrics.build_full_metrics(db)
    state = alerts.evaluate_kill_switch(snapshot)

    if state["triggered"]:
        n = await alerts.write_kill_switch(db, service_status=False)
        logger.warning(
            f"KILL SWITCH TRIGGERED — flipped {n} account(s) to service_status=False. "
            f"Reasons: {state['reasons']}"
        )
        msg = alerts.format_kill_switch_alert(state, snapshot)
        if TELEGRAM_ENABLED:
            sent = await alerts.send_telegram_message(msg)
            if sent:
                logger.info("Kill-switch alert sent to Telegram")
    else:
        # Only log at debug to avoid spam
        logger.debug("Kill switch armed — all thresholds within range")

    return {"snapshot": snapshot, "state": state}


async def maybe_telegram_ping(db, last_ping_at: datetime | None) -> datetime:
    """
    Send a status ping to Telegram at most every TELEGRAM_INTERVAL seconds.
    Returns the new last_ping_at.
    """
    if not TELEGRAM_ENABLED:
        return last_ping_at
    now = datetime.now(timezone.utc)
    if last_ping_at is None or (now - last_ping_at).total_seconds() >= TELEGRAM_INTERVAL:
        snapshot = await metrics.build_full_metrics(db)
        state = alerts.evaluate_kill_switch(snapshot)
        msg = alerts.format_status_message(snapshot, state)
        sent = await alerts.send_telegram_message(msg)
        if sent:
            logger.info("Telegram status ping sent")
            return now
    return last_ping_at


async def main_async():
    if not MONGO_URI:
        logger.error("MONGO_URI not set in .env — cannot start")
        sys.exit(1)

    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
    db = client["quotexTraderBot"]

    if "--once" in sys.argv:
        # Single check, then exit. Used by smoke tests.
        result = await run_check_once(db)
        snapshot = result["snapshot"]
        state = result["state"]
        overall = snapshot.get("overall", {})
        r = snapshot.get("rolling_50", {})
        daily = snapshot.get("daily_pnl", {})
        print("=" * 60)
        print("  DASHBOARD SINGLE CHECK (--once mode)")
        print("=" * 60)
        print(f"  Total trades:    {overall.get('total', 0)}")
        print(f"  Win rate:        {overall.get('win_rate', 0):.2f}%  (BE {overall.get('breakeven', 54.05):.2f}%)")
        print(f"  Last 50 WR:      {r.get('win_rate', 0):.2f}%")
        print(f"  Daily PnL:       ${daily.get('net_pnl', 0):+.2f}")
        print(f"  Kill switch:     {'TRIGGERED' if state['triggered'] else 'ARMED'}")
        if state["triggered"]:
            for r in state["reasons"]:
                print(f"                   - {r}")
        print("=" * 60)
        client.close()
        return

    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop)
    logger.info(
        f"dashboard runner started — check every {CHECK_INTERVAL}s, "
        f"Telegram every {TELEGRAM_INTERVAL}s, enabled={TELEGRAM_ENABLED}"
    )

    last_ping_at: datetime | None = None
    while _running:
        try:
            await run_check_once(db)
            last_ping_at = await maybe_telegram_ping(db, last_ping_at)
        except Exception as e:
            logger.exception(f"check loop error: {e}")

        # Sleep in small slices so the signal handler can stop us promptly
        for _ in range(CHECK_INTERVAL):
            if not _running:
                break
            await asyncio.sleep(1)

    client.close()
    logger.info("dashboard runner stopped")


if __name__ == "__main__":
    asyncio.run(main_async())

"""
Pure metric computations for the live verification dashboard.

Every function here is read-only against `trade_history_db` and `trade_settings_db`.
No side effects. No MongoDB writes. Safe to call from any context.
"""

import os
import datetime
from typing import Any

BREAKEVEN_WINRATE = 54.05


def _ts(doc) -> datetime.datetime:
    """Coerce a timestamp field (datetime or ISO string) into a UTC-aware datetime."""
    ts = doc.get("timestamp")
    if isinstance(ts, datetime.datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=datetime.timezone.utc)
        return ts
    if isinstance(ts, str):
        try:
            dt = datetime.datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=datetime.timezone.utc)
            return dt
        except ValueError:
            pass
    return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)


async def overall_stats(db) -> dict:
    """All-time totals: trades, wins, losses, ties, net PnL, breakeven gap."""
    trade_history = db["trade_history"]
    total = await trade_history.count_documents({})
    wins = await trade_history.count_documents({"result": "WIN"})
    losses = await trade_history.count_documents({"result": "LOSS"})
    ties = await trade_history.count_documents({"result": "TIE"})

    pipeline = [
        {"$group": {"_id": "$result", "total": {"$sum": "$profit"}}}
    ]
    pnl_by_result = {None: 0.0}
    async for doc in trade_history.aggregate(pipeline):
        pnl_by_result[doc["_id"]] = doc["total"]

    win_profit = pnl_by_result.get("WIN", 0.0)
    loss_amount = pnl_by_result.get("LOSS", 0.0)
    net_pnl = win_profit + loss_amount
    win_rate = (wins / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": round(win_rate, 2),
        "breakeven": BREAKEVEN_WINRATE,
        "gap_to_breakeven": round(win_rate - BREAKEVEN_WINRATE, 2),
        "net_pnl": round(net_pnl, 2),
        "win_profit": round(win_profit, 2),
        "loss_amount": round(loss_amount, 2),
    }


async def rolling_window(db, n: int = 500) -> list[dict]:
    """Last N trades sorted by timestamp DESC. Returns plain-dict projection safe for JSON."""
    trade_history = db["trade_history"]
    cursor = trade_history.find().sort("timestamp", -1).limit(n)
    out = []
    async for doc in cursor:
        ts = _ts(doc)
        out.append({
            "asset": doc.get("asset", ""),
            "direction": doc.get("ai_signal", ""),
            "result": doc.get("result", ""),
            "profit": round(float(doc.get("profit", 0.0)), 4),
            "amount": round(float(doc.get("amount", 0.0)), 4),
            "payout": doc.get("payout"),
            "reason": (doc.get("ai_reason") or "")[:120],
            "timestamp": ts.isoformat(),
        })
    return out


async def rolling_win_rate(db, n: int = 50) -> dict:
    """Win rate over the last N trades, plus counts."""
    trade_history = db["trade_history"]
    cursor = trade_history.find().sort("timestamp", -1).limit(n)
    wins = losses = ties = 0
    async for doc in cursor:
        r = doc.get("result", "LOSS")
        if r == "WIN":
            wins += 1
        elif r == "TIE":
            ties += 1
        else:
            losses += 1
    total = wins + losses + ties
    win_rate = (wins / total * 100) if total > 0 else 0.0
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": round(win_rate, 2),
        "breakeven": BREAKEVEN_WINRATE,
        "gap": round(win_rate - BREAKEVEN_WINRATE, 2),
    }


async def recent_streak(db, n: int = 10) -> dict:
    """Wins in the last N trades (the 'is the streak alive' indicator)."""
    trade_history = db["trade_history"]
    cursor = trade_history.find().sort("timestamp", -1).limit(n)
    results = []
    async for doc in cursor:
        results.append(doc.get("result", "LOSS"))
    wins = sum(1 for r in results if r == "WIN")
    return {
        "n": n,
        "wins": wins,
        "results": results,  # newest-first
    }


async def daily_pnl(db) -> dict:
    """Net PnL since today_start UTC. Mirrors bot.py:2336-2350."""
    trade_history = db["trade_history"]
    today_start = datetime.datetime.now(datetime.timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    pipeline = [
        {"$match": {"timestamp": {"$gte": today_start}}},
        {"$group": {"_id": None, "total_pnl": {"$sum": "$profit"}}},
    ]
    agg = await trade_history.aggregate(pipeline).to_list(length=1)
    total = agg[0]["total_pnl"] if agg else 0.0
    return {
        "since_utc": today_start.isoformat(),
        "net_pnl": round(float(total), 2),
    }


async def drawdown(db) -> dict:
    """Running cumulative PnL across all history; current and max drawdown from peak."""
    trade_history = db["trade_history"]
    cursor = trade_history.find().sort("timestamp", 1)
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    current_dd = 0.0
    async for doc in cursor:
        cumulative += float(doc.get("profit", 0.0))
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
        current_dd = dd
    return {
        "current_equity": round(cumulative, 2),
        "peak_equity": round(peak, 2),
        "current_drawdown": round(current_dd, 2),
        "max_drawdown": round(max_dd, 2),
    }


async def inversion_test(db, n: int = 500) -> dict:
    """
    The smoking-gun diagnostic.

    For each non-TIE trade in the last N, compute the win rate of the *original* signal
    AND the win rate of the *inverted* signal (call<->put). On a coin-flip strategy these
    are nearly identical, proving zero directional edge. On a strategy with real edge
    the original is meaningfully higher.

    Ties are excluded; for ties the "inverted" signal would tie too.
    """
    trade_history = db["trade_history"]
    cursor = trade_history.find().sort("timestamp", -1).limit(n)
    docs = []
    async for d in cursor:
        if d.get("result") in ("WIN", "LOSS"):
            docs.append(d)

    if not docs:
        return {"original_winrate": 0.0, "inverted_winrate": 0.0, "n": 0, "edge_pct": 0.0}

    original_wins = 0
    inverted_wins = 0
    for d in docs:
        result = d.get("result")
        signal = d.get("ai_signal", "").lower()
        if result == "WIN":
            original_wins += 1
            inverted_wins += 0
        elif result == "LOSS":
            original_wins += 0
            inverted_wins += 1
    total = len(docs)
    orig_wr = original_wins / total * 100
    inv_wr = inverted_wins / total * 100
    return {
        "n": total,
        "original_winrate": round(orig_wr, 2),
        "inverted_winrate": round(inv_wr, 2),
        "edge_pct": round(orig_wr - inv_wr, 2),
    }


async def hourly_edge(db) -> list[dict]:
    """Win rate per UTC hour, with PRIME / PROFITABLE / LOSS labels.

    Mirrors analyze_sessions.py:42-97.
    """
    trade_history = db["trade_history"]
    buckets = {h: {"wins": 0, "losses": 0, "ties": 0, "profit": 0.0, "n": 0} for h in range(24)}
    async for doc in trade_history.find():
        ts = _ts(doc)
        h = ts.hour
        r = doc.get("result", "LOSS")
        profit = float(doc.get("profit", 0.0))
        buckets[h]["n"] += 1
        buckets[h]["profit"] += profit
        if r == "WIN":
            buckets[h]["wins"] += 1
        elif r == "LOSS":
            buckets[h]["losses"] += 1
        else:
            buckets[h]["ties"] += 1

    out = []
    for h in range(24):
        b = buckets[h]
        n = b["n"]
        wr = (b["wins"] / n * 100) if n > 0 else 0.0
        if n == 0:
            label = "EMPTY"
        elif wr >= 65.0:
            label = "PRIME"
        elif wr >= BREAKEVEN_WINRATE:
            label = "PROFITABLE"
        else:
            label = "LOSS"
        out.append({
            "hour_utc": h,
            "trades": n,
            "wins": b["wins"],
            "losses": b["losses"],
            "ties": b["ties"],
            "win_rate": round(wr, 2),
            "profit": round(b["profit"], 2),
            "label": label,
        })
    return out


async def per_asset_kill(db, min_trades: int = 20, max_wr: float = 45.0) -> list[dict]:
    """Assets with >= min_trades and WR below max_wr. Auto-blacklist candidates."""
    trade_history = db["trade_history"]
    pipeline = [
        {"$group": {
            "_id": "$asset",
            "n": {"$sum": 1},
            "wins": {"$sum": {"$cond": [{"$eq": ["$result", "WIN"]}, 1, 0]}},
            "losses": {"$sum": {"$cond": [{"$eq": ["$result", "LOSS"]}, 1, 0]}},
            "ties": {"$sum": {"$cond": [{"$eq": ["$result", "TIE"]}, 1, 0]}},
            "profit": {"$sum": "$profit"},
        }},
        {"$match": {"n": {"$gte": min_trades}}},
    ]
    out = []
    async for doc in trade_history.aggregate(pipeline):
        n = doc["n"]
        wins = doc["wins"]
        wr = (wins / n * 100) if n > 0 else 0.0
        if wr <= max_wr:
            out.append({
                "asset": doc["_id"] or "(unknown)",
                "trades": n,
                "wins": wins,
                "losses": doc["losses"],
                "ties": doc["ties"],
                "win_rate": round(wr, 2),
                "profit": round(float(doc["profit"]), 2),
            })
    out.sort(key=lambda x: x["win_rate"])
    return out


async def service_status(db) -> dict:
    """Current on/off state of every linked Quotex account."""
    trade_settings = db["trade_settings"]
    out = []
    async for doc in trade_settings.find():
        out.append({
            "account_doc_id": str(doc.get("_id")),
            "account_mode": doc.get("account_mode", "PRACTICE"),
            "service_status": bool(doc.get("service_status", False)),
            "trade_amount": doc.get("trade_amount"),
        })
    return {"accounts": out}


async def build_full_metrics(db) -> dict:
    """All metrics in one shot — used by the /api/metrics endpoint and the alert loop."""
    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "overall": await overall_stats(db),
        "rolling_50": await rolling_win_rate(db, 50),
        "rolling_500_count": await trade_history_count(db, 500),
        "recent_10": await recent_streak(db, 10),
        "daily_pnl": await daily_pnl(db),
        "drawdown": await drawdown(db),
        "inversion": await inversion_test(db, 500),
        "hourly_edge": await hourly_edge(db),
        "per_asset_kill": await per_asset_kill(db),
        "service_status": await service_status(db),
    }


async def trade_history_count(db, n: int) -> int:
    trade_history = db["trade_history"]
    return await trade_history.count_documents({})

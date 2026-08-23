import asyncio
import os
import datetime
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://root:12345@cluster0.p73ls.mongodb.net/quotex_bot?retryWrites=true&w=majority")

async def analyze_trading_sessions():
    """
    Analyzes MongoDB trade_history by UTC Hour, Day of Week, Trading Session, and Asset
    to identify peak profit hours and optimal win-rate windows for AI trading.
    """
    print(f"\n{Fore.CYAN}==========================================================================")
    print(f"{Fore.YELLOW}           QUANT AI TIMEZONE & TRADING SESSION PERFORMANCE ANALYZER         ")
    print(f"{Fore.CYAN}=========================================================================={Style.RESET_ALL}\n")

    client = AsyncIOMotorClient(MONGO_URI)
    db = client['quotex_bot']
    trade_history = db['trade_history']

    total_trades = await trade_history.count_documents({})
    if total_trades == 0:
        print(f"{Fore.RED}No trade history records found in MongoDB.{Style.RESET_ALL}")
        return

    wins = await trade_history.count_documents({"result": "WIN"})
    losses = await trade_history.count_documents({"result": "LOSS"})
    ties = await trade_history.count_documents({"result": "TIE"})
    overall_win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

    print(f"Total Trades Recorded: {total_trades}")
    print(f"Overall Wins: {Fore.GREEN}{wins}{Style.RESET_ALL} | Losses: {Fore.RED}{losses}{Style.RESET_ALL} | Ties: {Fore.YELLOW}{ties}{Style.RESET_ALL}")
    print(f"Overall Win Rate: {Fore.GREEN if overall_win_rate >= 54.05 else Fore.RED}{overall_win_rate:.2f}%{Style.RESET_ALL} (Quotex BE: 54.05%)\n")

    # 1. Hourly Breakdown (UTC 00:00 - 23:00)
    hourly_stats = {}
    for h in range(24):
        hourly_stats[h] = {"wins": 0, "losses": 0, "ties": 0, "profit": 0.0}

    async for doc in trade_history.find():
        ts = doc.get("timestamp")
        if not ts:
            continue
        if isinstance(ts, str):
            try:
                ts = datetime.datetime.fromisoformat(ts)
            except Exception:
                continue

        # Extract UTC Hour
        utc_hour = ts.hour if hasattr(ts, 'hour') else 0
        res = doc.get("result", "LOSS")
        prof = float(doc.get("profit", 0.0))

        if utc_hour in hourly_stats:
            if res == "WIN":
                hourly_stats[utc_hour]["wins"] += 1
                hourly_stats[utc_hour]["profit"] += prof
            elif res == "LOSS":
                hourly_stats[utc_hour]["losses"] += 1
                hourly_stats[utc_hour]["profit"] -= float(doc.get("amount", 5.0))
            else:
                hourly_stats[utc_hour]["ties"] += 1

    print(f"{Fore.CYAN}--- HOURLY PERFORMANCE BREAKDOWN (UTC TIMEZONE) ---{Style.RESET_ALL}")
    print(f"{'UTC Hour':10s} | {'Trades':8s} | {'Wins':6s} | {'Losses':7s} | {'Win Rate':10s} | {'Net Profit ($)':15s} | {'Performance'}")
    print("-" * 80)

    best_hours = []
    for h in range(24):
        w = hourly_stats[h]["wins"]
        l = hourly_stats[h]["losses"]
        t = hourly_stats[h]["ties"]
        tot = w + l + t
        p = hourly_stats[h]["profit"]

        if tot == 0:
            print(f"{h:02d}:00 UTC   | {'0':8s} | {'0':6s} | {'0':7s} | {'0.00%':10s} | {'$0.00':15s} | NO TRADES")
            continue

        wr = (w / tot) * 100
        if wr >= 65.0:
            status = f"{Fore.GREEN}★ PRIME WINNING HOUR ({wr:.1f}%){Style.RESET_ALL}"
            best_hours.append((h, wr, tot, p))
        elif wr >= 54.05:
            status = f"{Fore.GREEN}✓ PROFITABLE ({wr:.1f}%){Style.RESET_ALL}"
        else:
            status = f"{Fore.RED}✗ LOW WIN RATE ({wr:.1f}%){Style.RESET_ALL}"

        p_str = f"+${p:.2f}" if p >= 0 else f"-${abs(p):.2f}"
        print(f"{h:02d}:00 UTC   | {tot:8d} | {w:6d} | {l:7d} | {wr:9.1f}% | {p_str:15s} | {status}")

    print("\n" + "=" * 80)
    print(f"{Fore.YELLOW}                   TOP OPTIMAL TRADING SESSIONS (UTC)                       {Style.RESET_ALL}")
    print("=" * 80)

    # Global Session Definitions
    sessions = {
        "Asian Session (Tokyo/Sydney)": range(23, 8),
        "European Session (London)": range(7, 16),
        "US Session (New York)": range(12, 21),
        "London / NY Overlap (Peak Liquidity)": range(12, 16)
    }

    for s_name, hours_range in sessions.items():
        s_wins = sum(hourly_stats[h % 24]["wins"] for h in hours_range)
        s_losses = sum(hourly_stats[h % 24]["losses"] for h in hours_range)
        s_ties = sum(hourly_stats[h % 24]["ties"] for h in hours_range)
        s_tot = s_wins + s_losses + s_ties
        s_prof = sum(hourly_stats[h % 24]["profit"] for h in hours_range)
        s_wr = (s_wins / s_tot) * 100 if s_tot > 0 else 0

        p_fmt = f"+${s_prof:.2f}" if s_prof >= 0 else f"-${abs(s_prof):.2f}"
        color = Fore.GREEN if s_wr >= 54.05 else Fore.RED
        print(f"{s_name:38s} | Trades: {s_tot:4d} | Win Rate: {color}{s_wr:6.2f}%{Style.RESET_ALL} | Net Profit: {p_fmt}")

    if best_hours:
        best_hours.sort(key=lambda x: x[1], reverse=True)
        top_h = best_hours[0]
        print(f"\n{Fore.GREEN}🚀 AI RECOMMENDATION: Optimal trading window is at {top_h[0]:02d}:00 UTC ({top_h[1]:.1f}% Win Rate over {top_h[2]} trades)!{Style.RESET_ALL}\n")

if __name__ == "__main__":
    asyncio.run(analyze_trading_sessions())

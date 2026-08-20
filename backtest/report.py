"""report.py — equity curve + win/loss chart, saved to PNG."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(trades_df: pd.DataFrame, start_balance: float, out_path: str, title: str = "Equity Curve"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})

    balances = pd.concat([pd.Series([start_balance]), trades_df["balance_after"]], ignore_index=True)
    ax1.plot(balances.values, color="#2b6cb0", linewidth=1.5)
    ax1.axhline(start_balance, color="gray", linestyle="--", linewidth=1, label="Starting balance")
    ax1.set_title(title)
    ax1.set_ylabel("Balance")
    ax1.legend()
    ax1.grid(alpha=0.3)

    colors = trades_df["win"].map({True: "#2f855a", False: "#c53030"})
    ax2.bar(range(len(trades_df)), trades_df["pnl"], color=colors, width=1.0)
    ax2.set_ylabel("PnL / trade")
    ax2.set_xlabel("Trade #")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path

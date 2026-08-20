"""
engine.py — backtest simulator for binary-option style trading (call/put, fixed expiry).
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, List
import pandas as pd
import numpy as np


@dataclass
class Trade:
    entry_index: int
    expiry_index: int
    entry_time: object
    direction: str          # "call" or "put"
    entry_price: float
    expiry_price: float
    stake: float
    payout_pct: float
    win: bool
    pnl: float
    balance_after: float


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])


class BacktestEngine:
    def __init__(self, payout_pct: float = 0.85, tie_is_loss: bool = True):
        """
        payout_pct: fraction paid on a win, e.g. 0.85 = 85% profit on a winning trade.
        tie_is_loss: whether an exact-unchanged expiry counts as a loss (typical) or refund.
        """
        self.payout_pct = payout_pct
        self.tie_is_loss = tie_is_loss

    def run(
        self,
        df: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame], Optional[str]],
        expiry_periods: int = 1,
        start_balance: float = 1000.0,
        stake_mode: str = "fixed_pct",   # "fixed_pct" | "fixed_amount" | "martingale"
        stake_value: float = 0.01,        # 1% of balance, or fixed $ amount depending on mode
        martingale_multiplier: float = 2.2,
        cooldown_periods: int = 0,        # min candles between trades
    ) -> BacktestResult:
        n = len(df)
        result = BacktestResult()
        balance = start_balance
        next_allowed_index = 0
        current_martingale_stake = None

        for i in range(n):
            expiry_i = i + expiry_periods
            if expiry_i >= n:
                break
            if i < next_allowed_index:
                continue

            # STRICT no-lookahead: strategy only sees data through candle i
            visible = df.iloc[: i + 1]
            signal = strategy_fn(visible)
            if signal not in ("call", "put"):
                continue

            entry_price = df.iloc[i]["close"]
            expiry_price = df.iloc[expiry_i]["close"]

            if signal == "call":
                win = expiry_price > entry_price
            else:
                win = expiry_price < entry_price
            if expiry_price == entry_price:
                win = not self.tie_is_loss

            # position sizing
            if stake_mode == "fixed_pct":
                stake = balance * stake_value
            elif stake_mode == "fixed_amount":
                stake = stake_value
            elif stake_mode == "martingale":
                if current_martingale_stake is None or (result.trades and result.trades[-1].win):
                    current_martingale_stake = balance * stake_value
                stake = current_martingale_stake
            else:
                raise ValueError(f"Unknown stake_mode: {stake_mode}")

            stake = min(stake, balance)  # can't stake more than you have

            pnl = stake * self.payout_pct if win else -stake
            balance += pnl

            if stake_mode == "martingale" and not win:
                current_martingale_stake = min(current_martingale_stake * martingale_multiplier, balance) \
                    if balance > 0 else 0

            result.trades.append(Trade(
                entry_index=i, expiry_index=expiry_i,
                entry_time=df.iloc[i].get("timestamp", i),
                direction=signal, entry_price=entry_price, expiry_price=expiry_price,
                stake=stake, payout_pct=self.payout_pct, win=win, pnl=pnl,
                balance_after=balance,
            ))

            next_allowed_index = i + 1 + cooldown_periods

            if balance <= 0:
                break  # account blown

        return result

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from indicators.sma import simple_moving_average


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class SmaCrossoverConfig:
    fast_window: int = 20
    slow_window: int = 50


def generate_signals(close_prices: pd.Series, cfg: SmaCrossoverConfig = SmaCrossoverConfig()) -> pd.Series:
    """
    Generate trading signals using an SMA crossover.

    BUY when fast SMA crosses above slow SMA.
    SELL when fast SMA crosses below slow SMA.
    HOLD otherwise.
    """
    _ensure_valid_windows(cfg)

    fast = simple_moving_average(close_prices, cfg.fast_window)
    slow = simple_moving_average(close_prices, cfg.slow_window)

    # Crossover detection: compare today vs yesterday.
    crossed_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    crossed_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    signals = pd.Series(Signal.HOLD.value, index=close_prices.index, name="signal")
    signals.loc[crossed_up] = Signal.BUY.value
    signals.loc[crossed_down] = Signal.SELL.value

    # For rows where either SMA is NaN, force HOLD.
    signals.loc[fast.isna() | slow.isna()] = Signal.HOLD.value

    return signals


def _ensure_valid_windows(cfg: SmaCrossoverConfig) -> None:
    if cfg.fast_window <= 0 or cfg.slow_window <= 0:
        raise ValueError("SMA windows must be positive integers.")
    if cfg.fast_window >= cfg.slow_window:
        raise ValueError("fast_window must be smaller than slow_window to form a crossover strategy.")

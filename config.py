from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    ticker: str = "QQQ"
    start_date: str = "2018-01-01"
    interval: str = "1d"
    training_ratio: float = 0.60
    data_frequency: str = "1D"
    initial_cash: float = 100_000
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0005
    batch_size: int = 1000
    fast_ema_periods: List[int] = field(
        default_factory=lambda: list(range(4, 40, 3))
    )
    medium_ema_periods: List[int] = field(
        default_factory=lambda: list(range(80, 200, 3))
    )
    slow_ema_periods: List[int] = field(
        default_factory=lambda: list(range(100, 200, 3))
    )

from typing import List, Tuple



def build_ema_combinations(
    fast_ema_periods: List[int], medium_ema_periods: List[int], slow_ema_periods: List[int]
) -> List[Tuple[int, int, int]]:
    ema_combinations = []
    for fast_ema_period in fast_ema_periods:
        for medium_ema_period in medium_ema_periods:
            for slow_ema_period in slow_ema_periods:
                if fast_ema_period < medium_ema_period and fast_ema_period < slow_ema_period:
                    ema_combinations.append(
                        (fast_ema_period, medium_ema_period, slow_ema_period)
                    )
    return ema_combinations
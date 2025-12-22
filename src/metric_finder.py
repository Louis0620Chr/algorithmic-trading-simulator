from typing import List, Tuple

import pandas as pd
import vectorbt as vbt

from config import Config


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


def run_grid_search(
    training_close: pd.Series, ema_combinations: List[Tuple[int, int, int]], config: Config
) -> pd.DataFrame:
    if not ema_combinations:
        return pd.DataFrame()

    batch_size = config.batch_size
    total_combinations = len(ema_combinations)

    all_ema_periods = sorted({period for combo in ema_combinations for period in combo})
    ema_cache = vbt.MA.run(training_close, window=all_ema_periods, ewm=True)

    grid_search_results = []

    for batch_start in range(0, total_combinations, batch_size):
        batch_end = min(batch_start + batch_size, total_combinations)
        batch_combinations = ema_combinations[batch_start:batch_end]

        batch_entries = []
        batch_exits = []

        for fast_ema_period, medium_ema_period, slow_ema_period in batch_combinations:
            fast_ema = ema_cache.ma[fast_ema_period]
            medium_ema = ema_cache.ma[medium_ema_period]
            slow_ema = ema_cache.ma[slow_ema_period]

            entries_raw = (
                fast_ema.vbt.crossed_above(medium_ema)
                | fast_ema.vbt.crossed_above(slow_ema)
                | medium_ema.vbt.crossed_above(slow_ema)
            )
            exits_raw = (
                fast_ema.vbt.crossed_below(medium_ema)
                | fast_ema.vbt.crossed_below(slow_ema)
                | medium_ema.vbt.crossed_below(slow_ema)
            )

            # Shift signals to avoid lookahead bias
            entries = entries_raw.shift(1).fillna(False).astype(bool)
            exits = exits_raw.shift(1).fillna(False).astype(bool)

            batch_entries.append(entries)
            batch_exits.append(exits)

        entries_dataframe = pd.concat(batch_entries, axis=1)
        exits_dataframe = pd.concat(batch_exits, axis=1)

        portfolios = vbt.Portfolio.from_signals(
            close=training_close,
            entries=entries_dataframe,
            exits=exits_dataframe,
            init_cash=config.initial_cash,
            fees=config.fees,
            slippage=config.slippage,
            freq=config.frequency,
        )

        sharpe_ratios = portfolios.sharpe_ratio(freq=config.frequency)

        for index, (fast_ema_period, medium_ema_period, slow_ema_period) in enumerate(
            batch_combinations
        ):
            sharpe_ratio = float(
                sharpe_ratios.iloc[index] if hasattr(sharpe_ratios, "iloc") else sharpe_ratios
            )
            grid_search_results.append(
                {
                    "fast_ema_period": fast_ema_period,
                    "medium_ema_period": medium_ema_period,
                    "slow_ema_period": slow_ema_period,
                    "sharpe_ratio": sharpe_ratio,
                }
            )

    results_dataframe = pd.DataFrame(grid_search_results)
    return results_dataframe

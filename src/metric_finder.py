from typing import List, Tuple

import numpy as np
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
        empty_results = pd.DataFrame()
        return empty_results

    batch_size = config.batch_size
    total_combinations = len(ema_combinations)

    all_ema_periods = sorted({period for combo in ema_combinations for period in combo})
    ema_cache = vbt.MA.run(training_close, window=all_ema_periods, ewm=True)

    grid_search_results = []
    years = max(
        (training_close.index[-1] - training_close.index[0]).days / 365.25, 1e-9
    )

    for batch_start in range(0, total_combinations, batch_size):
        batch_end = min(batch_start + batch_size, total_combinations)
        batch_combinations = ema_combinations[batch_start:batch_end]
        batch_length = len(batch_combinations)

        batch_entries = []
        batch_exits = []

        for fast_ema_period, medium_ema_period, slow_ema_period in batch_combinations:
            try:
                fast_ema = pd.Series(
                    ema_cache.ma[fast_ema_period].values.flatten(),
                    index=training_close.index,
                )
                medium_ema = pd.Series(
                    ema_cache.ma[medium_ema_period].values.flatten(),
                    index=training_close.index,
                )
                slow_ema = pd.Series(
                    ema_cache.ma[slow_ema_period].values.flatten(),
                    index=training_close.index,
                )

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
                entries_shifted = entries_raw.shift(1)
                entries = pd.Series(
                    np.where(entries_shifted.isna(), False, entries_shifted),
                    index=training_close.index,
                    dtype=bool,
                )

                exits_shifted = exits_raw.shift(1)
                exits = pd.Series(
                    np.where(exits_shifted.isna(), False, exits_shifted),
                    index=training_close.index,
                    dtype=bool,
                )

                batch_entries.append(entries)
                batch_exits.append(exits)
            except Exception:
                batch_entries.append(
                    pd.Series(False, index=training_close.index, dtype=bool)
                )
                batch_exits.append(
                    pd.Series(False, index=training_close.index, dtype=bool)
                )
                pass

        entries_dataframe = pd.DataFrame(batch_entries).T
        exits_dataframe = pd.DataFrame(batch_exits).T

        try:
            portfolios = vbt.Portfolio.from_signals(
                close=training_close,
                entries=entries_dataframe,
                exits=exits_dataframe,
                init_cash=config.initial_cash,
                fees=config.fees,
                slippage=config.slippage,
                freq=config.frequency,
            )
        except Exception:
            pass
            continue

        sharpe_ratios = portfolios.sharpe_ratio(freq=config.frequency)

        for index, (fast_ema_period, medium_ema_period, slow_ema_period) in enumerate(
            batch_combinations
        ):
            try:
                sharpe_ratio = float(
                    sharpe_ratios.iloc[index]
                    if hasattr(sharpe_ratios, "iloc")
                    else sharpe_ratios
                )
                grid_search_results.append(
                    {
                        "fast_ema_period": fast_ema_period,
                        "medium_ema_period": medium_ema_period,
                        "slow_ema_period": slow_ema_period,
                        "sharpe_ratio": sharpe_ratio,
                    }
                )
            except Exception:
                pass

    results_dataframe = pd.DataFrame(grid_search_results)
    return results_dataframe

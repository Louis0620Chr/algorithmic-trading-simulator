from typing import List, Tuple
import time
import pandas
import vectorbt
from backtest import build_buy_and_sell_signals
from config import Config


def build_ema_combinations(fast_ema_periods: List[int], medium_ema_periods: List[int], slow_ema_periods: List[int]) -> List[Tuple[int, int, int]]:
    return [
        (fast_period, medium_period, slow_period)
        for fast_period in fast_ema_periods
        for medium_period in medium_ema_periods
        for slow_period in slow_ema_periods
        if fast_period < medium_period and fast_period < slow_period
    ]


def run_grid_search(training_close_prices: pandas.Series, ema_combinations: List[Tuple[int, int, int]], config: Config) -> pandas.DataFrame:
    if not ema_combinations:
        return pandas.DataFrame()

    batch_size = config.batch_size
    total_combinations = len(ema_combinations)
    print(f"Grid search: {total_combinations} combinations, batch size {batch_size}")

    periods_from_combinations = [
        period for combination in ema_combinations for period in combination
    ]
    all_periods = sorted(set(periods_from_combinations))
    ema_cache = vectorbt.MA.run(training_close_prices, window=all_periods, ewm=True)
    results = []
    years_in_sample = max((training_close_prices.index[-1] - training_close_prices.index[0]).days / 365.25, 1e-9)
    start_time = time.perf_counter()

    price_index = training_close_prices.index

    empty_signal_series = pandas.Series(False, index=price_index, dtype=bool)
    for batch_start in range(0, total_combinations, batch_size):
        batch_start_time = time.perf_counter()
        batch_end = min(batch_start + batch_size, total_combinations)
        current_batch_combinations = ema_combinations[batch_start:batch_end]

        batch_entry_signals = []
        batch_exit_signals = []

        for fast_period, medium_period, slow_period in current_batch_combinations:
            try:
                fast_ema_series = pandas.Series(
                    ema_cache.ma[fast_period].values.flatten(),
                    index=price_index,
                )
                medium_ema_series = pandas.Series(
                    ema_cache.ma[medium_period].values.flatten(),
                    index=price_index,
                )
                slow_ema_series = pandas.Series(
                    ema_cache.ma[slow_period].values.flatten(),
                    index=price_index,
                )

                entry_signals, exit_signals = build_buy_and_sell_signals(fast_ema_series, medium_ema_series, slow_ema_series, price_index)

                batch_entry_signals.append(entry_signals)
                batch_exit_signals.append(exit_signals)
            except Exception:
                batch_entry_signals.append(empty_signal_series)
                batch_exit_signals.append(empty_signal_series)

        entry_signals_dataframe = pandas.concat(batch_entry_signals, axis=1)
        exit_signals_dataframe = pandas.concat(batch_exit_signals, axis=1)

        try:
            batch_portfolios = vectorbt.Portfolio.from_signals(close=training_close_prices, entries=entry_signals_dataframe, exits=exit_signals_dataframe, init_cash=config.initial_cash, fees=config.fee_rate, slippage=config.slippage_rate, freq=config.data_frequency)
        except Exception:
            continue

        sharpe_ratio_values = batch_portfolios.sharpe_ratio(freq=config.data_frequency)

        for combination_index, (fast_period, medium_period, slow_period) in enumerate(
            current_batch_combinations
        ):
            try:
                sharpe_ratio = float(
                    sharpe_ratio_values.iloc[combination_index]
                    if hasattr(sharpe_ratio_values, "iloc")
                    else sharpe_ratio_values
                )

                portfolio_trades = (
                    batch_portfolios.trades
                    if len(current_batch_combinations) == 1
                    else batch_portfolios[combination_index].trades
                )

                total_trades = len(portfolio_trades)
                trades_per_year = total_trades / years_in_sample

                if trades_per_year < 2:
                    continue

                results.append({"fast_ema_period": fast_period, "medium_ema_period": medium_period, "slow_ema_period": slow_period, "sharpe_ratio": sharpe_ratio})
            except Exception:
                pass

        elapsed_total = time.perf_counter() - start_time
        batch_elapsed = time.perf_counter() - batch_start_time
        progress_percent = (batch_end / total_combinations) * 100.0
        print(
            "Grid search progress: {}/{} ({:.1f}%) | batch {:.2f}s | total {:.2f}s".format(
                batch_end, total_combinations, progress_percent, batch_elapsed, elapsed_total
            )
        )

    results_dataframe = pandas.DataFrame(results)
    total_elapsed = time.perf_counter() - start_time
    print("Grid search complete in {:.2f}s".format(total_elapsed))
    return results_dataframe

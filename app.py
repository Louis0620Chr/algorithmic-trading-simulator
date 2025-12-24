from config import Config
from data import load_stock_data, select_close_series
from metric_finder import build_ema_combinations, run_grid_search
from backtest import build_ema_signals, run_portfolio_backtest, compute_metrics
from visualization import plot_best_strategy

def main():
    config = Config()

    # Fetch and prepare close prices
    stock_data = load_stock_data(config.ticker, config.start_date, config.interval)
    close = select_close_series(stock_data, config.ticker)
    close.name = "price"

    # Training split for parameter search
    split_index = int(len(close) * config.training_ratio)
    training_close = close.iloc[:split_index].copy()

    # Grid search over EMA combinations
    
    ema_combinations = build_ema_combinations(
        config.fast_ema_periods,
        config.medium_ema_periods,
        config.slow_ema_periods
        )
    results_dataframe = run_grid_search(
        training_close, ema_combinations, config
    )
   
    best_result_index = results_dataframe["sharpe_ratio"].idxmax()
    best_result = results_dataframe.loc[best_result_index]
    fast_period = int(best_result["fast_ema_period"])
    medium_period = int(best_result["medium_ema_period"])
    slow_period = int(best_result["slow_ema_period"])

    (
        fast_ema_series,
        medium_ema_series,
        slow_ema_series,
        entry_signals,
        exit_signals,
    ) = build_ema_signals(close, fast_period, medium_period, slow_period)

    full_sample_portfolio = run_portfolio_backtest(
        close, entry_signals, exit_signals, config
    )

    performance_metrics = compute_metrics(full_sample_portfolio, config)

    plot_best_strategy(
        close_price_series=close,
        fast_period=fast_period,
        medium_period=medium_period,
        slow_period=slow_period,
        fast_ema_series=fast_ema_series,
        medium_ema_series=medium_ema_series,
        slow_ema_series=slow_ema_series,
        entry_signals=entry_signals,
        exit_signals=exit_signals,
        performance_metrics=performance_metrics,
    )

if __name__ == "__main__":
    main()

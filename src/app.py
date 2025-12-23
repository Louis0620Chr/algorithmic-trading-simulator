from config import Config
from data import load_stock_data, select_close_series
from metric_finder import build_ema_combinations, run_grid_search

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

if __name__ == "__main__":
    main()

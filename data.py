from pathlib import Path

import pandas as pd
import yfinance as yf


def load_stock_data(ticker: str, start_date: str, interval: str = "1d") -> pd.DataFrame:
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"{ticker}_{start_date}_{interval}.csv"

    if cache_path.exists():
        cached = pd.read_csv(cache_path, header=[0, 1], index_col=0, parse_dates=True)
        if isinstance(cached.columns, pd.MultiIndex) and "Close" in cached.columns.get_level_values(0):
            return cached
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    stock_data = yf.download(ticker, start=start_date, interval=interval)
    stock_data.to_csv(cache_path)
    return stock_data

def select_close_series(dataframe: pd.DataFrame, ticker: str) -> pd.Series:
    if isinstance(dataframe.columns, pd.MultiIndex):
        if ("Close", ticker) in dataframe.columns:
            close_series = dataframe[("Close", ticker)]
        else:
            close_columns = [column for column in dataframe.columns if "Close" in str(column)]
            if not close_columns:
                raise KeyError("Close not found")
            close_series = dataframe[close_columns[0]]
    else:
        close_series = dataframe["Close"]
    close_series = close_series.astype(float).squeeze()
    return close_series

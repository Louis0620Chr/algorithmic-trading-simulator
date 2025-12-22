import pandas as pd
import yfinance as yf


def load_stock_data(ticker: str, start_date: str, interval: str = "1d") -> pd.DataFrame:
    stock_data = yf.download(ticker, start=start_date, interval=interval)
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

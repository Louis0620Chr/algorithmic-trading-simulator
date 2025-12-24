import yfinance as yf
from config import Config

config = Config()

df = yf.download(config.ticker, config.start_date, config.interval).get(f"Close, {config.ticker}")

print(df.head())
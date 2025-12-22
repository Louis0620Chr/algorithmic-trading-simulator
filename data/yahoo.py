import yfinance as yf
import talib
TICKER = "QQQ"
START_DATE = '2018-01-01'
class DataFetcher:
    def fetch_data(self):
        df = yf.download(self.ticker, start=self.start_date, interval=self.interval, progress=False)
        if df.empty:
            raise ValueError("No data returned from Yahoo Finance")
        return df 
    def process_data(self):
        data = self.fetch_data()

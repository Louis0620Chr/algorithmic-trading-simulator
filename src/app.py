from config import Config
from data import load_stock_data, select_close_series

def main():
    config = Config()

    # Fetch and prepare close prices
    stock_data = load_stock_data(config.ticker, config.start_date, config.interval)
    close = select_close_series(stock_data, config.ticker)
    close.name = "price"
    
if __name__ == "__main__":
    main()


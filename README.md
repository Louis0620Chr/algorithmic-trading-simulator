# EMA Strategy Backtesting Project

This project implements an **Exponential Moving Average (EMA) crossover trading strategy** with a **grid search for parameter optimization**, followed by a **full-sample backtest, performance evaluation, and visualization**.

The main goal is to:
- Find the best EMA parameter combination based on historical data
- Evaluate the best strategy on the full dataset
- Visualize price action, EMAs, entry/exit signals, and performance metrics

---

## Project Structure

```text
├── app.py # Entry point of the application
├── config.py # Central configuration (ticker, dates, parameters)
├── data.py # Data loading and preprocessing
├── metric_finder.py # EMA grid search and parameter evaluation
├── backtest.py # Signal generation, portfolio backtesting, metrics
├── visualization.py # Plotting and visualization utilities
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```
---


## How the Project Works

### 1. Data Loading
- Historical price data is fetched using Yahoo Finance
- The close price series is extracted and used for all calculations

### 2. Training / Evaluation Split
- A configurable training ratio is used
- Only the training portion is used for grid search optimization

### 3. Grid Search (EMA Optimization)
- All combinations of fast, medium, and slow EMA periods are generated
- Each combination is backtested on the training data
- The best combination is selected based on Sharpe Ratio

### 4. Full-Sample Backtest
- The best EMA parameters are applied to the full dataset
- Entry and exit signals are generated
- A portfolio backtest is executed

### 5. Performance Metrics
Typical metrics include:
- Total return
- Sharpe ratio
- Drawdowns

### 6. Visualization
- Price series with EMA overlays
- Entry and exit signals
- Strategy performance summary

---


## Setup and Installation

### 1. Create a Virtual Environment

python -m venv .venv

### 2. Activate the Virtual Environment

macOS / Linux:

source .venv/bin/activate
<<<<<<< HEAD
python -m pip install -r requirements.txt

Run the app from the repo root:

python app.py
=======

Windows:

.venv\Scripts\activate

### 3. Install Dependencies

python -m pip install -r requirements.txt


---

## Running the Project

After installing dependencies, run:

python app.py


The script will:
1. Download market data
2. Perform EMA grid search
3. Backtest the best strategy
4. Compute performance metrics
5. Display visualizations

---

## Configuration

All key parameters can be adjusted in `config.py`, including:
- Ticker symbol
- Date range
- Data interval
- EMA period ranges
- Training ratio

This allows experimentation without modifying the core logic.

---

## Notes

- This project is intended for research and educational purposes
- It is not financial advice
- Performance on historical data does not guarantee future results

---

## Summary

This project demonstrates:
- Clean separation of concerns
- Parameter optimization via grid search
- Full-sample strategy evaluation
- A reproducible backtesting workflow

It is suitable as a software craftsmanship, quantitative trading, or backtesting architecture example.
>>>>>>> 745ec118924f5727879b1d22dcbe8217983fc9f4

import numpy
import pandas
import vectorbt

from config import Config


def build_ema_signals(close_price_series, fast_period: int, medium_period: int, slow_period: int):
    """
    Compute fast, medium and slow exponential moving averages and derive
    corresponding buy and sell signals based on EMA crossovers.
    """
    fast_ema_series = vectorbt.MA.run(close_price_series, fast_period, ewm=True).ma
    medium_ema_series = vectorbt.MA.run(close_price_series, medium_period, ewm=True).ma
    slow_ema_series = vectorbt.MA.run(close_price_series, slow_period, ewm=True).ma

    entry_signals, exit_signals = build_buy_and_sell_signals(
        fast_ema_series,
        medium_ema_series,
        slow_ema_series,
        close_price_series.index,
    )

    return (fast_ema_series, medium_ema_series, slow_ema_series, entry_signals, exit_signals)

def shift_and_align_signals(raw_signals, price_index):
    """
    Shift signals by one bar to avoid look-ahead bias and align them
    to the price index used for backtesting.
    """
    shifted_signals = raw_signals.shift(1)
    boolean_signals = shifted_signals.astype("boolean")
    filled_signals = boolean_signals.fillna(False)
    aligned_signals = filled_signals.reindex(price_index).astype("boolean").fillna(False)
    return aligned_signals.to_numpy(dtype=bool)

def build_buy_and_sell_signals(fast_ema_series, medium_ema_series, slow_ema_series, price_index):
    """
    Define entry and exit conditions using EMA crossover logic.
    Multiple crossover combinations are allowed to trigger signals.
    """
    fast_crosses_above_medium = fast_ema_series.vbt.crossed_above(medium_ema_series)
    fast_crosses_above_slow = fast_ema_series.vbt.crossed_above(slow_ema_series)
    medium_crosses_above_slow = medium_ema_series.vbt.crossed_above(slow_ema_series)
    entry_signals_raw = fast_crosses_above_medium | fast_crosses_above_slow | medium_crosses_above_slow

    fast_crosses_below_medium = fast_ema_series.vbt.crossed_below(medium_ema_series)
    fast_crosses_below_slow = fast_ema_series.vbt.crossed_below(slow_ema_series)
    medium_crosses_below_slow = medium_ema_series.vbt.crossed_below(slow_ema_series)
    exit_signals_raw = fast_crosses_below_medium | fast_crosses_below_slow | medium_crosses_below_slow

    entry_signals = shift_and_align_signals(entry_signals_raw, price_index)
    exit_signals = shift_and_align_signals(exit_signals_raw, price_index)
    entry_signals_series = pandas.Series(entry_signals, index=price_index, dtype=bool)
    exit_signals_series = pandas.Series(exit_signals, index=price_index, dtype=bool)
    return entry_signals_series, exit_signals_series

def run_portfolio_backtest(close_price_series, entry_signals, exit_signals, config: Config):
    """
    Execute a VectorBT signal-based portfolio backtest using
    predefined trading costs and capital settings.
    """
    portfolio = vectorbt.Portfolio.from_signals(
        close=close_price_series.to_numpy(dtype=float),
        entries=entry_signals.to_numpy(dtype=bool),
        exits=exit_signals.to_numpy(dtype=bool),
        init_cash=config.initial_cash,
        fees=config.fee_rate,
        slippage=config.slippage_rate,
        freq=config.data_frequency,
    )
    return portfolio

def compute_metrics(portfolio, config: Config) -> dict:
    """
    Compute key performance metrics for strategy evaluation.
    """
    total_return = float(portfolio.total_return())
    annualized_return = float(portfolio.annualized_return(freq=config.data_frequency))
    sharpe_ratio = float(portfolio.sharpe_ratio(freq=config.data_frequency))
    maximum_drawdown = float(portfolio.max_drawdown())
    volatility = float(portfolio.annualized_volatility(freq=config.data_frequency))

    trades = portfolio.trades
    win_rate = numpy.nan
    if len(trades) > 0:
        trade_returns = _extract_trade_returns(trades)
        if trade_returns.size > 0:
            win_rate = (trade_returns > 0).sum() / len(trade_returns)

    metrics = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "maximum_drawdown": maximum_drawdown,
        "volatility": volatility,
        "win_rate": win_rate,
    }
    return metrics

def _extract_trade_returns(trades):
    trade_returns = (
        trades.returns.values
        if hasattr(trades.returns, "values")
        else numpy.array(trades.returns)
    )
    return trade_returns

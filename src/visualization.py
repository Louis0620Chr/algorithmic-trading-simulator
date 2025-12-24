import matplotlib.pyplot as plt
import numpy


def _format_percentage(value: float) -> str:
    if value is None or numpy.isnan(value):
        return "n/a"
    return "{:.2f}%".format(value * 100.0)


def _format_number(value: float) -> str:
    if value is None or numpy.isnan(value):
        return "n/a"
    return "{:.2f}".format(value)

def plot_best_strategy(
    close_price_series,
    fast_period: int,
    medium_period: int,
    slow_period: int,
    fast_ema_series,
    medium_ema_series,
    slow_ema_series,
    entry_signals,
    exit_signals,
    performance_metrics: dict,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(
        close_price_series.index,
        close_price_series.values,
        label="Close",
        color="black",
        linewidth=1.5,
        alpha=0.7,
    )

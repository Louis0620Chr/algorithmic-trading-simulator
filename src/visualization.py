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

    moving_average_plot_definitions = [
        ("Fast Exponential Moving Average", fast_ema_series, fast_period, "blue"),
        ("Medium Exponential Moving Average", medium_ema_series, medium_period, "orange"),
        ("Slow Exponential Moving Average", slow_ema_series, slow_period, "purple"),
    ]

    for label_prefix, moving_average_series, period, line_color in moving_average_plot_definitions:
        ax.plot(
            moving_average_series.index,
            moving_average_series.values,
            label=f"{label_prefix} ({period})",
            color=line_color,
            alpha=0.8,
            linewidth=1.2,
        )

    signal_plot_definitions = [
        ("Buy", entry_signals, "^", "green"),
        ("Sell", exit_signals, "v", "red"),
    ]

    for signal_label, signal_mask, marker_style, marker_color in signal_plot_definitions:
        signal_index = close_price_series.index[signal_mask]
        ax.scatter(
            signal_index,
            close_price_series.reindex(signal_index).values,
            marker=marker_style,
            color=marker_color,
            s=80,
            label=signal_label,
            zorder=5,
        )

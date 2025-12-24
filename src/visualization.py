import matplotlib.pyplot as plt
import numpy


def _format_percentage(value: float) -> str:
    formatted_value = "n/a"
    if value is None or numpy.isnan(value):
        formatted_value = "n/a"
    else:
        formatted_value = "{:.2f}%".format(value * 100.0)
    return formatted_value


def _format_number(value: float) -> str:
    formatted_value = "n/a"
    if value is None or numpy.isnan(value):
        formatted_value = "n/a"
    else:
        formatted_value = "{:.2f}".format(value)
    return formatted_value

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

    metric_labels = [
        "Total Return",
        "Annualized Return",
        "Sharpe Ratio",
        "Maximum Drawdown",
        "Volatility",
        "Win Rate",
    ]

    metric_values = [
        _format_percentage(performance_metrics.get("total_return")),
        _format_percentage(performance_metrics.get("annualized_return")),
        _format_number(performance_metrics.get("sharpe_ratio")),
        _format_percentage(performance_metrics.get("maximum_drawdown")),
        _format_percentage(performance_metrics.get("volatility")),
        _format_percentage(performance_metrics.get("win_rate")),
    ]

    cell_text = [[label, value] for label, value in zip(metric_labels, metric_values)]

    table = ax.table(
        cellText=cell_text,
        colLabels=["Metric", "Value"],
        loc="lower right",
        bbox=[0.68, 0.02, 0.3, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    ax.set_title(
        "Best Triple Exponential Moving Average({},{},{}) - Full Sample Signals".format(
            fast_period, medium_period, slow_period
        ),
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

from __future__ import annotations

import pandas as pd


def simple_moving_average(values: pd.Series, window: int) -> pd.Series:
    """
    Compute a simple moving average over a fixed window.

    Returns a Series aligned with the input index.
    The first (window - 1) values are NaN by definition.
    """
    _ensure_valid_window(window)
    return values.rolling(window=window).mean()


def _ensure_valid_window(window: int) -> None:
    if window <= 0:
        raise ValueError(f"window must be a positive integer, got {window}.")

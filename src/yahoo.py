from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import pandas as pd

_REQUIRED_MARKET_DATA_COLUMNS: Final[tuple[str, ...]] = (
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
)

DataSource = Literal["yahoo", "cache_only"]


@dataclass(frozen=True)
class YahooConfig:
    symbol: str = "QQQ"
    period: str = "5y"
    interval: str = "1d"
    cache_dir: Path = Path("cache")


def load_price_history(
    cfg: YahooConfig = YahooConfig(),
    *,
    force_refresh: bool = False,
    source: DataSource = "yahoo",
) -> pd.DataFrame:
    """
    Return historical daily market price and volume data.

    The returned DataFrame:
    - uses a DateTimeIndex
    - contains open/high/low/close prices and traded volume
    - is validated for completeness and expected schema

    `source` controls how data is obtained:
    - "yahoo": download (if needed) and cache to CSV
    - "cache_only": read from cache only (useful when offline)
    """
    cache_path = _cache_path(cfg)

    if not force_refresh:
        cached = _try_load_cache(cache_path)
        if cached is not None:
            return _validate_and_clean(cached)

    if source == "cache_only":
        raise FileNotFoundError(
            f"Cache file not found at '{cache_path}'. "
            "Run once with source='yahoo' when internet/dependencies work."
        )

    downloaded = _download_from_yahoo(cfg)
    _write_cache(downloaded, cache_path)
    return _validate_and_clean(downloaded)


def _cache_path(cfg: YahooConfig) -> Path:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{cfg.symbol}_{cfg.period}_{cfg.interval}.csv"
    return cfg.cache_dir / filename


def _try_load_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date")


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out.index.name = "Date"  # stable round-trip name for CSV
    out.to_csv(path)


def _download_from_yahoo(cfg: YahooConfig) -> pd.DataFrame:
    try:
        import yfinance as yf  # external dependency at the boundary
    except ImportError as exc:
        raise RuntimeError(
            "Cannot download from Yahoo Finance because 'yfinance' is not installed. "
            "Install it with: pip install yfinance"
        ) from exc

    df = yf.download(
        cfg.symbol,
        period=cfg.period,
        interval=cfg.interval,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError(f"No market data returned for symbol='{cfg.symbol}'.")

    return df


def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce invariants:
    - required price and volume columns must exist
    - index must be a DateTimeIndex
    - rows must not contain missing values
    """
    missing = [c for c in _REQUIRED_MARKET_DATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing: {missing}. Available columns: {list(df.columns)}"
        )

    cleaned = df.loc[:, _REQUIRED_MARKET_DATA_COLUMNS].copy()

    if not isinstance(cleaned.index, pd.DatetimeIndex):
        raise TypeError(
            f"Expected time-based index (DatetimeIndex), got {type(cleaned.index).__name__}."
        )

    cleaned.index.name = "Date"
    return cleaned.dropna()

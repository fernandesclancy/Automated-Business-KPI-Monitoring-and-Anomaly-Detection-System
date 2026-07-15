"""
Utility functions for KPI Sentinel.

Contains helper functions used across multiple modules.
"""

import logging
from typing import Optional, Union
import numpy as np
import pandas as pd

from src.config import LOG_FORMAT, LOG_DATE_FORMAT


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level (default: INFO).
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def safe_divide(
    numerator: Union[pd.Series, np.ndarray, float],
    denominator: Union[pd.Series, np.ndarray, float],
    fill_value: float = np.nan
) -> Union[pd.Series, np.ndarray, float]:
    """
    Perform division with protection against division by zero.
    
    Args:
        numerator: The numerator value(s).
        denominator: The denominator value(s).
        fill_value: Value to use when denominator is zero (default: NaN).
    
    Returns:
        Result of division with zeros replaced by fill_value.
    """
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        result = numerator / denominator
        if isinstance(result, pd.Series):
            result = result.replace([np.inf, -np.inf], fill_value)
            result = result.fillna(fill_value)
        return result
    elif isinstance(numerator, np.ndarray) or isinstance(denominator, np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.true_divide(numerator, denominator)
            result[~np.isfinite(result)] = fill_value
        return result
    else:
        if denominator == 0:
            return fill_value
        return numerator / denominator


def rolling_mad(
    series: pd.Series,
    window: int,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Calculate rolling Median Absolute Deviation (MAD).
    
    MAD is a robust measure of variability, defined as:
    MAD = median(|x - median(x)|)
    
    Args:
        series: Input time series.
        window: Rolling window size.
        min_periods: Minimum number of observations required (default: window).
    
    Returns:
        Series of rolling MAD values.
    """
    if min_periods is None:
        min_periods = window
    
    def mad_func(x):
        if len(x) < min_periods:
            return np.nan
        median_val = np.nanmedian(x)
        return np.nanmedian(np.abs(x - median_val))
    
    return series.rolling(window=window, min_periods=min_periods).apply(mad_func, raw=True)


def build_segment_key(level: str, row: Optional[pd.Series] = None) -> str:
    """
    Build a string representation of a segment key.
    
    Args:
        level: Hierarchy level (global, state, store, category, department).
        row: DataFrame row containing segment identifiers (optional for global).
    
    Returns:
        String representation of the segment (e.g., "global", "state=CA", "store=CA_1").
    """
    if level == "global":
        return "global"
    elif level == "state":
        return f"state={row['state_id']}"
    elif level == "store":
        return f"store={row['store_id']}"
    elif level == "category":
        return f"category={row['cat_id']}"
    elif level == "department":
        return f"department={row['dept_id']}"
    elif level == "item":
        return f"item={row['item_id']}"
    else:
        return f"{level}=unknown"


def parse_segment_key(segment_key: str) -> dict:
    """
    Parse a segment key string into its components.
    
    Args:
        segment_key: String representation (e.g., "state=CA").
    
    Returns:
        Dictionary with level and value.
    """
    if segment_key == "global":
        return {"level": "global", "value": None}
    
    if "=" in segment_key:
        level, value = segment_key.split("=", 1)
        return {"level": level, "value": value}
    
    return {"level": "unknown", "value": segment_key}


def get_child_level(parent_level: str) -> Optional[str]:
    """
    Get the child level for a given parent level in the hierarchy.
    
    Hierarchy: global -> state -> store -> department -> item
    
    Args:
        parent_level: The parent hierarchy level.
    
    Returns:
        The child level, or None if at the lowest level.
    """
    hierarchy = {
        "global": "state",
        "state": "store",
        "store": "department",
        "category": "department",
        "department": "item"
    }
    return hierarchy.get(parent_level)


def get_segment_filter(level: str, segment_key: str, df: pd.DataFrame) -> pd.Series:
    """
    Create a boolean filter for a DataFrame based on segment key.
    
    Args:
        level: Hierarchy level.
        segment_key: Segment key string.
        df: DataFrame to filter.
    
    Returns:
        Boolean Series for filtering.
    """
    if level == "global":
        return pd.Series([True] * len(df), index=df.index)
    
    parsed = parse_segment_key(segment_key)
    value = parsed["value"]
    
    if level == "state":
        return df["state_id"] == value
    elif level == "store":
        return df["store_id"] == value
    elif level == "category":
        return df["cat_id"] == value
    elif level == "department":
        return df["dept_id"] == value
    elif level == "item":
        return df["item_id"] == value
    
    return pd.Series([False] * len(df), index=df.index)


def format_number(value: float, precision: int = 2) -> str:
    """
    Format a number for display with appropriate precision.
    
    Args:
        value: The number to format.
        precision: Decimal places (default: 2).
    
    Returns:
        Formatted string representation.
    """
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{precision}f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.{precision}f}K"
    return f"{value:.{precision}f}"


def format_percentage(value: float, precision: int = 1) -> str:
    """
    Format a decimal value as a percentage string.
    
    Args:
        value: The decimal value (e.g., 0.15 for 15%).
        precision: Decimal places (default: 1).
    
    Returns:
        Formatted percentage string (e.g., "15.0%").
    """
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{precision}f}%"

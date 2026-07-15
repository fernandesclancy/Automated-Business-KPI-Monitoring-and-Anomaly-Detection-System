"""
Baseline Module.

This module provides time series baseline computation using STL
(Seasonal-Trend decomposition using Loess) for anomaly detection.

The baseline captures the expected value of a metric based on:
- Long-term trend
- Weekly seasonal pattern (period=7)

Residuals (actual - baseline) are used for anomaly scoring.
"""

import logging
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import STL_PERIOD, MIN_HISTORY_DAYS
from src.utils import setup_logger

logger = setup_logger(__name__)


def compute_stl_baseline(
    series: pd.Series,
    period: int = STL_PERIOD,
    robust: bool = True
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute STL decomposition baseline for a time series.
    
    Args:
        series: Input time series (must be sorted by time).
        period: Seasonal period (default: 7 for weekly).
        robust: Whether to use robust fitting (default: True).
    
    Returns:
        Tuple of (baseline, trend, seasonal, residual) Series.
    """
    # Handle missing values by forward filling then backward filling
    series_clean = series.ffill().bfill()
    
    # If still have NaN (all NaN series), return NaN series
    if series_clean.isna().all():
        nan_series = pd.Series(np.nan, index=series.index)
        return nan_series, nan_series, nan_series, nan_series
    
    # Need at least 2 periods for STL
    if len(series_clean) < 2 * period:
        nan_series = pd.Series(np.nan, index=series.index)
        return nan_series, nan_series, nan_series, nan_series
    
    try:
        # Fit STL
        stl = STL(series_clean, period=period, robust=robust)
        result = stl.fit()
        
        # Extract components
        trend = pd.Series(result.trend, index=series.index)
        seasonal = pd.Series(result.seasonal, index=series.index)
        residual = pd.Series(result.resid, index=series.index)
        
        # Baseline = trend + seasonal
        baseline = trend + seasonal
        
        return baseline, trend, seasonal, residual
        
    except Exception as e:
        logger.warning(f"STL decomposition failed: {e}")
        nan_series = pd.Series(np.nan, index=series.index)
        return nan_series, nan_series, nan_series, nan_series


def compute_baseline_for_series(
    df: pd.DataFrame,
    value_col: str = "value",
    date_col: str = "date",
    min_history: int = MIN_HISTORY_DAYS
) -> pd.DataFrame:
    """
    Compute baseline and residuals for a single time series DataFrame.
    
    Args:
        df: DataFrame with date and value columns (must be sorted by date).
        value_col: Name of the value column.
        date_col: Name of the date column.
        min_history: Minimum days of history before computing baseline.
    
    Returns:
        DataFrame with added baseline, trend, seasonal, and residual columns.
    """
    df = df.copy()
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Get the value series
    values = df[value_col]
    
    # Compute STL baseline
    baseline, trend, seasonal, residual = compute_stl_baseline(values)
    
    # Add to DataFrame
    df["baseline"] = baseline
    df["trend"] = trend
    df["seasonal"] = seasonal
    df["residual"] = residual
    
    # Mask early period where we don't have enough history
    if min_history > 0:
        df.loc[:min_history-1, ["baseline", "trend", "seasonal", "residual"]] = np.nan
    
    return df


class BaselineModel:
    """
    Baseline model class for computing and storing STL decomposition results.
    
    This class provides a convenient interface for computing baselines
    and can be extended for different baseline methods.
    """
    
    def __init__(
        self,
        period: int = STL_PERIOD,
        robust: bool = True,
        min_history: int = MIN_HISTORY_DAYS
    ):
        """
        Initialize the baseline model.
        
        Args:
            period: Seasonal period for STL.
            robust: Whether to use robust fitting.
            min_history: Minimum history days before scoring.
        """
        self.period = period
        self.robust = robust
        self.min_history = min_history
        self._fitted = False
        self._baseline = None
        self._trend = None
        self._seasonal = None
        self._residual = None
    
    def fit(self, series: pd.Series) -> "BaselineModel":
        """
        Fit the baseline model to a time series.
        
        Args:
            series: Input time series.
        
        Returns:
            Self for method chaining.
        """
        self._baseline, self._trend, self._seasonal, self._residual = \
            compute_stl_baseline(series, self.period, self.robust)
        
        # Mask early period
        if self.min_history > 0 and len(series) > self.min_history:
            self._baseline.iloc[:self.min_history] = np.nan
            self._residual.iloc[:self.min_history] = np.nan
        
        self._fitted = True
        return self
    
    @property
    def baseline(self) -> Optional[pd.Series]:
        """Get the computed baseline."""
        return self._baseline
    
    @property
    def trend(self) -> Optional[pd.Series]:
        """Get the computed trend component."""
        return self._trend
    
    @property
    def seasonal(self) -> Optional[pd.Series]:
        """Get the computed seasonal component."""
        return self._seasonal
    
    @property
    def residual(self) -> Optional[pd.Series]:
        """Get the computed residuals."""
        return self._residual
    
    def get_components(self) -> dict:
        """
        Get all decomposition components as a dictionary.
        
        Returns:
            Dictionary with baseline, trend, seasonal, and residual.
        """
        if not self._fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        return {
            "baseline": self._baseline,
            "trend": self._trend,
            "seasonal": self._seasonal,
            "residual": self._residual
        }

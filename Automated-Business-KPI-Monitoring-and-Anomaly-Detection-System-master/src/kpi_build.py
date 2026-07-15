"""
KPI Build Module.

This module computes business KPIs from the item_daily data at multiple
hierarchy levels and outputs the metric_facts.parquet artifact.

KPIs computed:
- units: Sum of daily units sold
- revenue: Sum of daily revenue
- avg_price: Revenue-weighted average price (revenue / units)
- price_index: Current avg_price relative to 28-day rolling median
- zero_sales_rate: Proportion of series with zero sales
- demand_volatility: Rolling MAD of units over 28 days

Hierarchy levels:
- global: All data aggregated
- state: Aggregated by state_id
- store: Aggregated by store_id
- category: Aggregated by cat_id
- department: Aggregated by dept_id
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DATA_PROCESSED,
    ITEM_DAILY_FILE,
    METRIC_FACTS_FILE,
    DEFAULT_KPIS,
    DEFAULT_LEVELS,
    MAD_WINDOW,
)
from src.utils import setup_logger, safe_divide, rolling_mad, build_segment_key

logger = setup_logger(__name__)


def compute_base_aggregates(
    item_daily: pd.DataFrame,
    level: str,
    group_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute base aggregates (units, revenue, series count) for a hierarchy level.
    
    Args:
        item_daily: The item_daily DataFrame.
        level: Hierarchy level name.
        group_cols: Columns to group by (None for global).
    
    Returns:
        DataFrame with daily aggregates for the level.
    """
    if group_cols is None:
        group_cols = []
    
    # Always group by date
    full_group = ["date"] + group_cols
    
    # Aggregate
    agg_dict = {
        "units": "sum",
        "revenue": "sum",
        "series_id": "nunique"
    }
    
    # Count zero sales series
    item_daily_copy = item_daily.copy()
    item_daily_copy["is_zero"] = (item_daily_copy["units"] == 0).astype(int)
    agg_dict["is_zero"] = "sum"
    
    agg = item_daily_copy.groupby(full_group, as_index=False).agg(agg_dict)
    agg = agg.rename(columns={"series_id": "n_series", "is_zero": "zero_count"})
    
    # Add level identifier
    agg["level"] = level
    
    # Add segment columns based on level
    if level == "global":
        agg["state_id"] = None
        agg["store_id"] = None
        agg["cat_id"] = None
        agg["dept_id"] = None
    elif level == "state":
        agg["store_id"] = None
        agg["cat_id"] = None
        agg["dept_id"] = None
    elif level == "store":
        agg["cat_id"] = None
        agg["dept_id"] = None
    elif level == "category":
        agg["state_id"] = None
        agg["store_id"] = None
        agg["dept_id"] = None
    elif level == "department":
        agg["state_id"] = None
        agg["store_id"] = None
        agg["cat_id"] = None
    
    return agg


def compute_kpis(agg: pd.DataFrame, level: str) -> pd.DataFrame:
    """
    Compute all KPIs from base aggregates.
    
    Args:
        agg: DataFrame with base aggregates.
        level: Hierarchy level name.
    
    Returns:
        Long-format DataFrame with one row per metric per day per segment.
    """
    # Sort by segment and date for rolling calculations
    if level == "global":
        sort_cols = ["date"]
        segment_cols = []
    elif level == "state":
        sort_cols = ["state_id", "date"]
        segment_cols = ["state_id"]
    elif level == "store":
        sort_cols = ["store_id", "date"]
        segment_cols = ["store_id"]
    elif level == "category":
        sort_cols = ["cat_id", "date"]
        segment_cols = ["cat_id"]
    elif level == "department":
        sort_cols = ["dept_id", "date"]
        segment_cols = ["dept_id"]
    else:
        sort_cols = ["date"]
        segment_cols = []
    
    agg = agg.sort_values(sort_cols).reset_index(drop=True)
    
    # Compute derived metrics
    # avg_price = revenue / units (with safe divide)
    agg["avg_price"] = safe_divide(agg["revenue"], agg["units"], fill_value=np.nan)
    
    # zero_sales_rate = zero_count / n_series
    agg["zero_sales_rate"] = safe_divide(agg["zero_count"], agg["n_series"], fill_value=0)
    
    # For rolling metrics, we need to group by segment
    if segment_cols:
        # price_index: avg_price / rolling_median(avg_price, 28)
        agg["price_median_28"] = agg.groupby(segment_cols)["avg_price"].transform(
            lambda x: x.rolling(window=MAD_WINDOW, min_periods=1).median()
        )
        
        # demand_volatility: rolling MAD of units
        agg["demand_volatility"] = agg.groupby(segment_cols)["units"].transform(
            lambda x: rolling_mad(x, window=MAD_WINDOW, min_periods=7)
        )
    else:
        # Global level - no grouping needed
        agg["price_median_28"] = agg["avg_price"].rolling(window=MAD_WINDOW, min_periods=1).median()
        agg["demand_volatility"] = rolling_mad(agg["units"], window=MAD_WINDOW, min_periods=7)
    
    # price_index = avg_price / price_median_28
    agg["price_index"] = safe_divide(agg["avg_price"], agg["price_median_28"], fill_value=np.nan)
    
    # Melt to long format
    metric_cols = ["units", "revenue", "avg_price", "price_index", "zero_sales_rate", "demand_volatility"]
    id_cols = ["date", "level", "state_id", "store_id", "cat_id", "dept_id", "n_series"]
    
    # Keep only needed columns
    keep_cols = id_cols + metric_cols
    keep_cols = [c for c in keep_cols if c in agg.columns]
    agg_subset = agg[keep_cols]
    
    # Melt
    melted = agg_subset.melt(
        id_vars=[c for c in id_cols if c in agg_subset.columns],
        value_vars=metric_cols,
        var_name="metric_name",
        value_name="value"
    )
    
    # Add granularity
    melted["granularity"] = "day"
    
    return melted


def build_all_kpis(item_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Build KPIs for all hierarchy levels.
    
    Args:
        item_daily: The item_daily DataFrame.
    
    Returns:
        Complete metric_facts DataFrame in long format.
    """
    all_metrics = []
    
    # Level configurations
    level_configs = {
        "global": None,
        "state": ["state_id"],
        "store": ["store_id"],
        "category": ["cat_id"],
        "department": ["dept_id"]
    }
    
    for level, group_cols in level_configs.items():
        logger.info(f"Computing KPIs for level: {level}")
        
        # Compute base aggregates
        agg = compute_base_aggregates(item_daily, level, group_cols)
        
        # Compute KPIs
        metrics = compute_kpis(agg, level)
        
        all_metrics.append(metrics)
        logger.info(f"  Generated {len(metrics)} metric records")
    
    # Concatenate all levels
    metric_facts = pd.concat(all_metrics, ignore_index=True)
    
    # Ensure date is datetime
    metric_facts["date"] = pd.to_datetime(metric_facts["date"])
    
    # Sort
    metric_facts = metric_facts.sort_values(["metric_name", "level", "date"]).reset_index(drop=True)
    
    logger.info(f"Total metric facts: {len(metric_facts)} records")
    return metric_facts


def run_kpi_build() -> pd.DataFrame:
    """
    Execute the KPI build pipeline.
    
    Returns:
        The complete metric_facts DataFrame.
    """
    logger.info("Starting KPI build pipeline")
    
    # Load item_daily
    input_path = DATA_PROCESSED / ITEM_DAILY_FILE
    if not input_path.exists():
        raise FileNotFoundError(
            f"Item daily file not found: {input_path}. "
            "Please run m5_ingest.py first."
        )
    
    logger.info(f"Loading item_daily from {input_path}")
    item_daily = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(item_daily)} records")
    
    # Build KPIs
    metric_facts = build_all_kpis(item_daily)
    
    # Save
    output_path = DATA_PROCESSED / METRIC_FACTS_FILE
    logger.info(f"Saving metric_facts to {output_path}")
    metric_facts.to_parquet(output_path, index=False)
    
    logger.info("KPI build pipeline complete")
    return metric_facts


def main():
    """Main entry point for the KPI build module."""
    try:
        run_kpi_build()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f"KPI build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Root Cause Analysis Module.

This module implements hierarchical contribution analysis to identify
the drivers behind detected anomalies.

For each anomaly event, the RCA:
1. Identifies child segments in the hierarchy
2. Computes before/after metrics for each child
3. Calculates contribution shares and deltas
4. Ranks drivers by impact score
5. Attaches calendar context (events, SNAP, weekday)

Output: rca_drivers.parquet with ranked driver records per event.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DATA_PROCESSED,
    METRIC_FACTS_FILE,
    ANOMALIES_FILE,
    RCA_DRIVERS_FILE,
    ITEM_DAILY_FILE,
    BEFORE_WINDOW_DAYS,
    DELTA_VALUE_WEIGHT,
    DELTA_SHARE_WEIGHT,
)
from src.utils import setup_logger, safe_divide, parse_segment_key, get_child_level

logger = setup_logger(__name__)


def get_calendar_context(
    item_daily: pd.DataFrame,
    peak_date: pd.Timestamp,
    segment_key: str
) -> Dict:
    """
    Get calendar context for a specific date.
    
    Args:
        item_daily: Item daily DataFrame with calendar columns.
        peak_date: The date to get context for.
        segment_key: Segment key to determine relevant SNAP flag.
    
    Returns:
        Dictionary with calendar context information.
    """
    # Get a row for the peak date
    date_rows = item_daily[item_daily["date"] == peak_date]
    
    if len(date_rows) == 0:
        return {
            "weekday": None,
            "is_weekend": None,
            "event_name": None,
            "event_type": None,
            "snap_flag": None
        }
    
    row = date_rows.iloc[0]
    
    # Determine weekday
    weekday = row.get("weekday", peak_date.strftime("%A"))
    is_weekend = peak_date.weekday() >= 5
    
    # Get event info
    event_name = row.get("event_name_1", None)
    event_type = row.get("event_type_1", None)
    
    # Determine relevant SNAP flag based on segment
    snap_flag = None
    parsed = parse_segment_key(segment_key)
    
    if parsed["level"] == "state":
        state = parsed["value"]
        snap_col = f"snap_{state}"
        if snap_col in row:
            snap_flag = bool(row[snap_col])
    elif parsed["level"] == "store":
        # Extract state from store_id (e.g., CA_1 -> CA)
        store_id = parsed["value"]
        if "_" in store_id:
            state = store_id.split("_")[0]
            snap_col = f"snap_{state}"
            if snap_col in row:
                snap_flag = bool(row[snap_col])
    elif parsed["level"] == "global":
        # Check if any SNAP is active
        snap_cols = ["snap_CA", "snap_TX", "snap_WI"]
        for col in snap_cols:
            if col in row and row[col]:
                snap_flag = True
                break
    
    return {
        "weekday": weekday,
        "is_weekend": is_weekend,
        "event_name": event_name,
        "event_type": event_type,
        "snap_flag": snap_flag
    }


def get_child_segments(
    metric_facts: pd.DataFrame,
    parent_level: str,
    parent_segment_key: str,
    metric_name: str
) -> List[Tuple[str, str]]:
    """
    Get child segments for a parent segment.
    
    Args:
        metric_facts: Metric facts DataFrame.
        parent_level: Parent hierarchy level.
        parent_segment_key: Parent segment key.
        metric_name: Metric name to filter by.
    
    Returns:
        List of (child_level, child_segment_key) tuples.
    """
    child_level = get_child_level(parent_level)
    
    if child_level is None:
        return []
    
    # Filter metric facts for child level and metric
    child_df = metric_facts[
        (metric_facts["level"] == child_level) &
        (metric_facts["metric_name"] == metric_name)
    ]
    
    # Further filter based on parent segment
    parsed = parse_segment_key(parent_segment_key)
    
    if parent_level == "global":
        # All children at child level
        pass
    elif parent_level == "state":
        # Filter stores within state
        state_id = parsed["value"]
        # Stores are named like CA_1, CA_2, etc.
        child_df = child_df[child_df["store_id"].str.startswith(f"{state_id}_")]
    elif parent_level == "store":
        # For store, children are departments
        # All departments are available at each store
        pass
    elif parent_level == "category":
        # Departments within category
        cat_id = parsed["value"]
        # Departments are named like FOODS_1, FOODS_2, etc.
        child_df = child_df[child_df["dept_id"].str.startswith(f"{cat_id.split('_')[0]}_")]
    
    # Get unique child segments
    children = []
    
    if child_level == "state":
        for state_id in child_df["state_id"].dropna().unique():
            children.append((child_level, f"state={state_id}"))
    elif child_level == "store":
        for store_id in child_df["store_id"].dropna().unique():
            children.append((child_level, f"store={store_id}"))
    elif child_level == "department":
        for dept_id in child_df["dept_id"].dropna().unique():
            children.append((child_level, f"department={dept_id}"))
    elif child_level == "category":
        for cat_id in child_df["cat_id"].dropna().unique():
            children.append((child_level, f"category={cat_id}"))
    
    return children


def get_series_values(
    metric_facts: pd.DataFrame,
    metric_name: str,
    level: str,
    segment_key: str,
    dates: List[pd.Timestamp]
) -> pd.Series:
    """
    Get metric values for a specific series and dates.
    
    Args:
        metric_facts: Metric facts DataFrame.
        metric_name: Metric name.
        level: Hierarchy level.
        segment_key: Segment key.
        dates: List of dates to get values for.
    
    Returns:
        Series of values indexed by date.
    """
    # Filter for the series
    mask = (
        (metric_facts["metric_name"] == metric_name) &
        (metric_facts["level"] == level)
    )
    
    parsed = parse_segment_key(segment_key)
    
    if level == "state":
        mask &= metric_facts["state_id"] == parsed["value"]
    elif level == "store":
        mask &= metric_facts["store_id"] == parsed["value"]
    elif level == "category":
        mask &= metric_facts["cat_id"] == parsed["value"]
    elif level == "department":
        mask &= metric_facts["dept_id"] == parsed["value"]
    
    series_df = metric_facts[mask].copy()
    series_df = series_df.set_index("date")["value"]
    
    # Get values for specified dates
    values = series_df.reindex(dates)
    
    return values


def compute_contribution_analysis(
    metric_facts: pd.DataFrame,
    event: pd.Series,
    child_segments: List[Tuple[str, str]]
) -> List[Dict]:
    """
    Compute contribution analysis for an event.
    
    Args:
        metric_facts: Metric facts DataFrame.
        event: Event row from anomalies DataFrame.
        child_segments: List of (child_level, child_segment_key) tuples.
    
    Returns:
        List of driver dictionaries.
    """
    metric_name = event["metric_name"]
    parent_level = event["level"]
    parent_segment_key = event["segment_key"]
    peak_date = pd.Timestamp(event["peak_date"])
    
    # Define windows
    before_start = peak_date - pd.Timedelta(days=BEFORE_WINDOW_DAYS)
    before_end = peak_date - pd.Timedelta(days=1)
    before_dates = pd.date_range(before_start, before_end, freq="D")
    after_dates = [peak_date]
    
    # Get parent values
    parent_before = get_series_values(
        metric_facts, metric_name, parent_level, parent_segment_key, before_dates
    ).mean()
    
    parent_after = get_series_values(
        metric_facts, metric_name, parent_level, parent_segment_key, after_dates
    ).iloc[0] if len(after_dates) > 0 else np.nan
    
    drivers = []
    
    for child_level, child_segment_key in child_segments:
        # Get child values
        child_before = get_series_values(
            metric_facts, metric_name, child_level, child_segment_key, before_dates
        ).mean()
        
        child_after = get_series_values(
            metric_facts, metric_name, child_level, child_segment_key, after_dates
        ).iloc[0] if len(after_dates) > 0 else np.nan
        
        # Skip if no data
        if pd.isna(child_before) and pd.isna(child_after):
            continue
        
        # Compute shares
        share_before = safe_divide(child_before, parent_before + 1e-9, fill_value=0)
        share_after = safe_divide(child_after, parent_after + 1e-9, fill_value=0)
        
        # Compute deltas
        delta_value = child_after - child_before if not (pd.isna(child_after) or pd.isna(child_before)) else 0
        delta_share = share_after - share_before
        
        # Compute driver score
        driver_score = (
            DELTA_VALUE_WEIGHT * abs(delta_value) +
            DELTA_SHARE_WEIGHT * abs(delta_share) * (abs(parent_after) + 1e-9)
        )
        
        driver = {
            "event_id": event["event_id"],
            "child_level": child_level,
            "child_segment_key": child_segment_key,
            "child_before": child_before,
            "child_after": child_after,
            "delta_value": delta_value,
            "share_before": share_before,
            "share_after": share_after,
            "delta_share": delta_share,
            "driver_score": driver_score
        }
        
        drivers.append(driver)
    
    # Rank drivers
    drivers = sorted(drivers, key=lambda x: x["driver_score"], reverse=True)
    for rank, driver in enumerate(drivers, 1):
        driver["rank"] = rank
    
    return drivers


def analyze_price_decomposition(
    metric_facts: pd.DataFrame,
    event: pd.Series,
    child_level: str,
    child_segment_key: str
) -> Dict:
    """
    For avg_price anomalies, decompose into revenue vs units contribution.
    
    Args:
        metric_facts: Metric facts DataFrame.
        event: Event row.
        child_level: Child hierarchy level.
        child_segment_key: Child segment key.
    
    Returns:
        Dictionary with decomposition hints.
    """
    peak_date = pd.Timestamp(event["peak_date"])
    before_start = peak_date - pd.Timedelta(days=BEFORE_WINDOW_DAYS)
    before_end = peak_date - pd.Timedelta(days=1)
    before_dates = pd.date_range(before_start, before_end, freq="D")
    after_dates = [peak_date]
    
    # Get revenue and units for the child segment
    revenue_before = get_series_values(
        metric_facts, "revenue", child_level, child_segment_key, before_dates
    ).mean()
    revenue_after = get_series_values(
        metric_facts, "revenue", child_level, child_segment_key, after_dates
    ).iloc[0] if len(after_dates) > 0 else np.nan
    
    units_before = get_series_values(
        metric_facts, "units", child_level, child_segment_key, before_dates
    ).mean()
    units_after = get_series_values(
        metric_facts, "units", child_level, child_segment_key, after_dates
    ).iloc[0] if len(after_dates) > 0 else np.nan
    
    # Compute percentage changes
    revenue_pct_change = safe_divide(revenue_after - revenue_before, revenue_before + 1e-9, fill_value=0)
    units_pct_change = safe_divide(units_after - units_before, units_before + 1e-9, fill_value=0)
    
    # Determine primary driver
    if abs(revenue_pct_change) > abs(units_pct_change):
        primary_driver = "revenue"
    else:
        primary_driver = "units"
    
    return {
        "revenue_pct_change": revenue_pct_change,
        "units_pct_change": units_pct_change,
        "primary_driver": primary_driver
    }


def run_root_cause_analysis(
    anomalies: Optional[pd.DataFrame] = None,
    metric_facts: Optional[pd.DataFrame] = None,
    item_daily: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Run root cause analysis for all detected anomalies.
    
    Args:
        anomalies: Optional anomalies DataFrame.
        metric_facts: Optional metric facts DataFrame.
        item_daily: Optional item daily DataFrame (for calendar context).
    
    Returns:
        DataFrame of RCA drivers.
    """
    logger.info("Starting root cause analysis pipeline")
    
    # Load data if not provided
    if anomalies is None:
        anomalies_path = DATA_PROCESSED / ANOMALIES_FILE
        if not anomalies_path.exists():
            raise FileNotFoundError(
                f"Anomalies file not found: {anomalies_path}. "
                "Please run detect.py first."
            )
        logger.info(f"Loading anomalies from {anomalies_path}")
        anomalies = pd.read_parquet(anomalies_path)
    
    if metric_facts is None:
        metric_facts_path = DATA_PROCESSED / METRIC_FACTS_FILE
        if not metric_facts_path.exists():
            raise FileNotFoundError(
                f"Metric facts file not found: {metric_facts_path}. "
                "Please run kpi_build.py first."
            )
        logger.info(f"Loading metric_facts from {metric_facts_path}")
        metric_facts = pd.read_parquet(metric_facts_path)
    
    if item_daily is None:
        item_daily_path = DATA_PROCESSED / ITEM_DAILY_FILE
        if item_daily_path.exists():
            logger.info(f"Loading item_daily from {item_daily_path}")
            # Load only needed columns for calendar context
            item_daily = pd.read_parquet(
                item_daily_path,
                columns=["date", "weekday", "event_name_1", "event_type_1",
                        "snap_CA", "snap_TX", "snap_WI"]
            )
        else:
            item_daily = pd.DataFrame()
    
    # Ensure dates are datetime
    metric_facts["date"] = pd.to_datetime(metric_facts["date"])
    if "peak_date" in anomalies.columns:
        anomalies["peak_date"] = pd.to_datetime(anomalies["peak_date"])
    
    if "z_severity_peak" in anomalies.columns and len(anomalies) > 200:
        logger.info(f"Limiting RCA to top 200 events (out of {len(anomalies)})")
        anomalies = anomalies.nlargest(200, "z_severity_peak")

    all_drivers = []
    
    # Process each anomaly event
    for idx, event in anomalies.iterrows():
        event_id = event["event_id"]
        logger.info(f"Analyzing event {event_id}: {event['metric_name']} at {event['level']}")
        
        # Get child segments
        child_segments = get_child_segments(
            metric_facts,
            event["level"],
            event["segment_key"],
            event["metric_name"]
        )
        
        if not child_segments:
            logger.info(f"  No child segments found for event {event_id}")
            continue
        
        # Compute contribution analysis
        drivers = compute_contribution_analysis(metric_facts, event, child_segments)
        
        # Add calendar context to top drivers
        if len(item_daily) > 0:
            calendar_context = get_calendar_context(
                item_daily,
                pd.Timestamp(event["peak_date"]),
                event["segment_key"]
            )
            for driver in drivers:
                driver.update(calendar_context)
        
        # For avg_price, add decomposition hints to top drivers
        if event["metric_name"] == "avg_price":
            for driver in drivers[:5]:  # Top 5 only
                decomp = analyze_price_decomposition(
                    metric_facts, event,
                    driver["child_level"],
                    driver["child_segment_key"]
                )
                driver["price_decomp_hint"] = decomp["primary_driver"]
                driver["revenue_pct_change"] = decomp["revenue_pct_change"]
                driver["units_pct_change"] = decomp["units_pct_change"]
        
        all_drivers.extend(drivers)
        logger.info(f"  Found {len(drivers)} drivers")
    
    # Create DataFrame
    if all_drivers:
        rca_df = pd.DataFrame(all_drivers)
    else:
        rca_df = pd.DataFrame()
    
    logger.info(f"Total RCA drivers: {len(rca_df)}")
    
    # Save
    output_path = DATA_PROCESSED / RCA_DRIVERS_FILE
    logger.info(f"Saving RCA drivers to {output_path}")
    rca_df.to_parquet(output_path, index=False)
    
    logger.info("Root cause analysis pipeline complete")
    return rca_df


def main():
    """Main entry point for the RCA module."""
    try:
        run_root_cause_analysis()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f"RCA failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

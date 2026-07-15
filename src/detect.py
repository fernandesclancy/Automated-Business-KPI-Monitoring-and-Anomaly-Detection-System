"""
Anomaly Detection Module.

This module implements the anomaly detection pipeline using:
- STL-based baseline computation
- Robust scoring using rolling MAD
- Cooldown-based false positive suppression
- Event grouping for consecutive anomalies

Output: anomalies.parquet with event-level anomaly records.
"""

import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import uuid

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DATA_PROCESSED,
    METRIC_FACTS_FILE,
    ANOMALIES_FILE,
    GROUND_TRUTH_FILE,
    MAD_WINDOW,
    ANOMALY_Z,
    COOLDOWN_DAYS,
    MIN_HISTORY_DAYS,
    MAX_SEVERITY,
    Z_SEVERITY_MULTIPLIER,
    PCT_SEVERITY_MULTIPLIER,
)
from src.baseline import compute_stl_baseline
from src.utils import setup_logger, rolling_mad, build_segment_key

logger = setup_logger(__name__)


def compute_anomaly_scores(
    df: pd.DataFrame,
    value_col: str = "value"
) -> pd.DataFrame:
    """
    Compute anomaly scores for a single time series.
    
    Args:
        df: DataFrame with date and value columns (sorted by date).
        value_col: Name of the value column.
    
    Returns:
        DataFrame with baseline, residual, sigma, score, and severity columns.
    """
    df = df.copy()
    values = df[value_col]
    
    # Compute STL baseline
    baseline, trend, seasonal, residual = compute_stl_baseline(values)
    
    df["baseline"] = baseline
    df["residual"] = residual
    
    # Compute rolling MAD of residuals for robust sigma estimate
    df["sigma"] = rolling_mad(residual, window=MAD_WINDOW, min_periods=7)
    
    # Compute z-score
    df["score"] = df["residual"] / (df["sigma"] + 1e-9)
    
    # Flag anomalies
    df["anomaly_flag"] = df["score"].abs() >= ANOMALY_Z
    
    # Compute severities
    df["z_severity"] = np.minimum(MAX_SEVERITY, Z_SEVERITY_MULTIPLIER * df["score"].abs())
    
    # Percentage change from baseline
    df["pct_change"] = (df[value_col] - df["baseline"]) / (df["baseline"].abs() + 1e-9)
    df["pct_severity"] = np.minimum(MAX_SEVERITY, PCT_SEVERITY_MULTIPLIER * df["pct_change"].abs())
    
    # Direction
    df["direction"] = np.where(df[value_col] > df["baseline"], "spike", "drop")
    
    # Mask early period
    mask = df.index < MIN_HISTORY_DAYS
    df.loc[mask, ["score", "z_severity", "pct_severity"]] = np.nan
    df.loc[mask, "anomaly_flag"] = False
    
    return df


def apply_cooldown(df: pd.DataFrame, cooldown_days: int = COOLDOWN_DAYS) -> pd.DataFrame:
    """
    Apply cooldown suppression to anomaly flags.
    
    After a flagged anomaly, suppress the next cooldown_days unless
    severity increases by more than 25%.
    
    Args:
        df: DataFrame with anomaly_flag and z_severity columns.
        cooldown_days: Number of days to suppress after an anomaly.
    
    Returns:
        DataFrame with updated anomaly_flag column.
    """
    df = df.copy()
    df["anomaly_flag_original"] = df["anomaly_flag"].copy()
    
    last_anomaly_idx = -cooldown_days - 1
    last_severity = 0
    
    for idx in range(len(df)):
        if not df.loc[df.index[idx], "anomaly_flag"]:
            continue
        
        days_since_last = idx - last_anomaly_idx
        current_severity = df.loc[df.index[idx], "z_severity"]
        
        if pd.isna(current_severity):
            current_severity = 0
        
        if days_since_last <= cooldown_days:
            # Check if severity increased by more than 25%
            severity_increase = (current_severity - last_severity) / (last_severity + 1e-9)
            
            if severity_increase <= 0.25:
                # Suppress this anomaly
                df.loc[df.index[idx], "anomaly_flag"] = False
                continue
        
        # This is a valid anomaly
        last_anomaly_idx = idx
        last_severity = current_severity
    
    return df


def group_consecutive_anomalies(df: pd.DataFrame) -> List[dict]:
    """
    Group consecutive anomaly days into events.
    
    Args:
        df: DataFrame with anomaly_flag, date, and severity columns.
    
    Returns:
        List of event dictionaries.
    """
    events = []
    
    # Find anomaly rows
    anomaly_df = df[df["anomaly_flag"] == True].copy()
    
    if len(anomaly_df) == 0:
        return events
    
    # Group consecutive dates
    anomaly_df = anomaly_df.sort_values("date").reset_index(drop=True)
    anomaly_df["date_diff"] = anomaly_df["date"].diff().dt.days
    anomaly_df["group"] = (anomaly_df["date_diff"] > 1).cumsum()
    
    # Create events for each group
    for group_id, group_df in anomaly_df.groupby("group"):
        # Find peak (max z_severity)
        peak_idx = group_df["z_severity"].idxmax()
        peak_row = group_df.loc[peak_idx]
        
        event = {
            "event_id": str(uuid.uuid4())[:8],
            "start_date": group_df["date"].min(),
            "end_date": group_df["date"].max(),
            "duration_days": len(group_df),
            "peak_date": peak_row["date"],
            "direction": peak_row["direction"],
            "z_severity_peak": peak_row["z_severity"],
            "pct_severity_peak": peak_row["pct_severity"],
            "pct_change_peak": peak_row["pct_change"],
            "score_peak": peak_row["score"],
            "value_peak": peak_row["value"],
            "baseline_peak": peak_row["baseline"]
        }
        
        events.append(event)
    
    return events


def detect_anomalies_for_series(
    df: pd.DataFrame,
    metric_name: str,
    level: str,
    segment_key: str
) -> List[dict]:
    """
    Run full anomaly detection pipeline for a single time series.
    
    Args:
        df: DataFrame with date and value columns.
        metric_name: Name of the metric.
        level: Hierarchy level.
        segment_key: Segment identifier string.
    
    Returns:
        List of event dictionaries with metadata.
    """
    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)
    
    # Skip if not enough data
    if len(df) < MIN_HISTORY_DAYS + 14:
        return []
    
    # Compute scores
    scored_df = compute_anomaly_scores(df)
    
    # Apply cooldown
    scored_df = apply_cooldown(scored_df)
    
    # Group into events
    events = group_consecutive_anomalies(scored_df)
    
    # Add metadata to events
    for event in events:
        event["metric_name"] = metric_name
        event["level"] = level
        event["segment_key"] = segment_key
    
    return events


def build_segment_key_from_row(row: pd.Series, level: str) -> str:
    """
    Build segment key string from a metric_facts row.
    
    Args:
        row: DataFrame row with segment columns.
        level: Hierarchy level.
    
    Returns:
        Segment key string.
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
    return "unknown"


def run_detection(metric_facts: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Run anomaly detection on all time series in metric_facts.
    
    Args:
        metric_facts: Optional DataFrame (loads from file if not provided).
    
    Returns:
        DataFrame of anomaly events.
    """
    logger.info("Starting anomaly detection pipeline")
    
    # Load metric facts if not provided
    if metric_facts is None:
        input_path = DATA_PROCESSED / METRIC_FACTS_FILE
        if not input_path.exists():
            raise FileNotFoundError(
                f"Metric facts file not found: {input_path}. "
                "Please run kpi_build.py first."
            )
        logger.info(f"Loading metric_facts from {input_path}")
        metric_facts = pd.read_parquet(input_path)
    
    # Ensure date is datetime
    metric_facts["date"] = pd.to_datetime(metric_facts["date"])
    
    # Get unique series identifiers
    # Group by metric_name, level, and segment columns
    all_events = []
    
    # Process each metric and level combination
    for metric_name in metric_facts["metric_name"].unique():
        metric_df = metric_facts[metric_facts["metric_name"] == metric_name]
        
        for level in metric_df["level"].unique():
            level_df = metric_df[metric_df["level"] == level]
            
            # Determine grouping columns based on level
            if level == "global":
                # Single series for global
                segment_key = "global"
                series_df = level_df[["date", "value"]].copy()
                
                events = detect_anomalies_for_series(
                    series_df, metric_name, level, segment_key
                )
                all_events.extend(events)
                
            elif level == "state":
                for state_id in level_df["state_id"].dropna().unique():
                    segment_key = f"state={state_id}"
                    series_df = level_df[level_df["state_id"] == state_id][["date", "value"]].copy()
                    
                    events = detect_anomalies_for_series(
                        series_df, metric_name, level, segment_key
                    )
                    all_events.extend(events)
                    
            elif level == "store":
                for store_id in level_df["store_id"].dropna().unique():
                    segment_key = f"store={store_id}"
                    series_df = level_df[level_df["store_id"] == store_id][["date", "value"]].copy()
                    
                    events = detect_anomalies_for_series(
                        series_df, metric_name, level, segment_key
                    )
                    all_events.extend(events)
                    
            elif level == "category":
                for cat_id in level_df["cat_id"].dropna().unique():
                    segment_key = f"category={cat_id}"
                    series_df = level_df[level_df["cat_id"] == cat_id][["date", "value"]].copy()
                    
                    events = detect_anomalies_for_series(
                        series_df, metric_name, level, segment_key
                    )
                    all_events.extend(events)
                    
            elif level == "department":
                for dept_id in level_df["dept_id"].dropna().unique():
                    segment_key = f"department={dept_id}"
                    series_df = level_df[level_df["dept_id"] == dept_id][["date", "value"]].copy()
                    
                    events = detect_anomalies_for_series(
                        series_df, metric_name, level, segment_key
                    )
                    all_events.extend(events)
        
        logger.info(f"Processed metric: {metric_name}, found {len([e for e in all_events if e['metric_name'] == metric_name])} events")
    
    # Create events DataFrame
    if all_events:
        anomalies_df = pd.DataFrame(all_events)
        
        # Ensure proper column order
        col_order = [
            "event_id", "metric_name", "level", "segment_key",
            "start_date", "end_date", "duration_days", "peak_date",
            "direction", "z_severity_peak", "pct_severity_peak", "pct_change_peak",
            "score_peak", "value_peak", "baseline_peak"
        ]
        anomalies_df = anomalies_df[[c for c in col_order if c in anomalies_df.columns]]
    else:
        anomalies_df = pd.DataFrame()
    
    logger.info(f"Total anomaly events detected: {len(anomalies_df)}")
    
    # Save
    output_path = DATA_PROCESSED / ANOMALIES_FILE
    logger.info(f"Saving anomalies to {output_path}")
    anomalies_df.to_parquet(output_path, index=False)
    
    logger.info("Anomaly detection pipeline complete")
    return anomalies_df


def create_ground_truth_events() -> pd.DataFrame:
    """
    Create synthetic ground truth events for evaluation.
    
    This function defines a set of known anomaly injections that can be
    used to evaluate detection performance.
    
    Returns:
        DataFrame with ground truth event definitions.
    """
    ground_truth = [
        {
            "gt_id": "gt_001",
            "metric_name": "revenue",
            "level": "store",
            "segment_key": "store=CA_1",
            "injection_type": "drop",
            "magnitude": 0.30,
            "start_date": pd.Timestamp("2016-03-15"),
            "end_date": pd.Timestamp("2016-03-17"),
            "duration_days": 3
        },
        {
            "gt_id": "gt_002",
            "metric_name": "zero_sales_rate",
            "level": "department",
            "segment_key": "department=FOODS_1",
            "injection_type": "spike",
            "magnitude": 0.50,
            "start_date": pd.Timestamp("2016-04-01"),
            "end_date": pd.Timestamp("2016-04-07"),
            "duration_days": 7
        },
        {
            "gt_id": "gt_003",
            "metric_name": "avg_price",
            "level": "state",
            "segment_key": "state=CA",
            "injection_type": "spike",
            "magnitude": 0.20,
            "start_date": pd.Timestamp("2016-05-10"),
            "end_date": pd.Timestamp("2016-05-11"),
            "duration_days": 2
        },
        {
            "gt_id": "gt_004",
            "metric_name": "units",
            "level": "global",
            "segment_key": "global",
            "injection_type": "drop",
            "magnitude": 0.25,
            "start_date": pd.Timestamp("2016-06-20"),
            "end_date": pd.Timestamp("2016-06-22"),
            "duration_days": 3
        },
        {
            "gt_id": "gt_005",
            "metric_name": "revenue",
            "level": "state",
            "segment_key": "state=TX",
            "injection_type": "drop",
            "magnitude": 0.35,
            "start_date": pd.Timestamp("2016-07-04"),
            "end_date": pd.Timestamp("2016-07-06"),
            "duration_days": 3
        },
    ]
    
    gt_df = pd.DataFrame(ground_truth)
    
    # Save
    output_path = DATA_PROCESSED / GROUND_TRUTH_FILE
    gt_df.to_parquet(output_path, index=False)
    logger.info(f"Ground truth events saved to {output_path}")
    
    return gt_df


def main():
    """Main entry point for the detection module."""
    try:
        run_detection()
        create_ground_truth_events()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Detection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

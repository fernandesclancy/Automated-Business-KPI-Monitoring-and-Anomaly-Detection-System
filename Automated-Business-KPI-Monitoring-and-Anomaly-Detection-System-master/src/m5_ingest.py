"""
M5 Dataset Ingestion Module.

This module handles the loading, transformation, and joining of the M5 dataset
files to produce the base item_daily.parquet artifact.

The M5 dataset consists of:
- sales_train_evaluation.csv: Wide-format daily sales data
- calendar.csv: Date dimension with events and SNAP information
- sell_prices.csv: Weekly price data by store and item

The output is a long-format daily fact table with units, prices, and revenue.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DATA_RAW,
    DATA_PROCESSED,
    SALES_FILE,
    CALENDAR_FILE,
    PRICES_FILE,
    ITEM_DAILY_FILE,
)
from src.utils import setup_logger

logger = setup_logger(__name__)


def load_calendar(filepath: Path) -> pd.DataFrame:
    """
    Load and prepare the calendar dimension table.
    
    Args:
        filepath: Path to calendar.csv.
    
    Returns:
        DataFrame with calendar data including date, week, events, and SNAP flags.
    """
    logger.info(f"Loading calendar from {filepath}")
    
    calendar = pd.read_csv(filepath, parse_dates=["date"])
    
    # Select relevant columns
    calendar_cols = [
        "d", "date", "wm_yr_wk", "wday", "weekday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI"
    ]
    
    # Keep only columns that exist
    available_cols = [c for c in calendar_cols if c in calendar.columns]
    calendar = calendar[available_cols]
    
    logger.info(f"Calendar loaded: {len(calendar)} days")
    return calendar


def load_prices(filepath: Path) -> pd.DataFrame:
    """
    Load the sell prices fact table.
    
    Args:
        filepath: Path to sell_prices.csv.
    
    Returns:
        DataFrame with store_id, item_id, wm_yr_wk, and sell_price.
    """
    logger.info(f"Loading prices from {filepath}")
    
    prices = pd.read_csv(filepath)
    
    logger.info(f"Prices loaded: {len(prices)} records")
    return prices


def load_and_melt_sales(filepath: Path, chunksize: int = 5000) -> pd.DataFrame:
    """
    Load sales data and melt from wide to long format.
    
    The sales file has columns: id, item_id, dept_id, cat_id, store_id, state_id,
    followed by d_1, d_2, ..., d_N for daily sales.
    
    This function processes the data in chunks to manage memory usage.
    
    Args:
        filepath: Path to sales_train_evaluation.csv.
        chunksize: Number of rows to process at a time.
    
    Returns:
        Long-format DataFrame with one row per series per day.
    """
    logger.info(f"Loading and melting sales from {filepath}")
    
    # First, read just the header to identify columns
    header = pd.read_csv(filepath, nrows=0)
    all_cols = header.columns.tolist()
    
    # Identify id columns and day columns
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in all_cols if c.startswith("d_")]
    
    logger.info(f"Found {len(day_cols)} day columns")
    
    # Process in chunks
    chunks = []
    total_rows = 0
    
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        # Melt the chunk
        melted = chunk.melt(
            id_vars=id_cols,
            value_vars=day_cols,
            var_name="d",
            value_name="units"
        )
        
        # Rename 'id' to 'series_id' for clarity
        melted = melted.rename(columns={"id": "series_id"})
        
        chunks.append(melted)
        total_rows += len(chunk)
        
        if total_rows % 10000 == 0:
            logger.info(f"Processed {total_rows} series...")
    
    # Concatenate all chunks
    sales_long = pd.concat(chunks, ignore_index=True)
    
    logger.info(f"Sales melted: {len(sales_long)} records from {total_rows} series")
    return sales_long


def join_and_compute(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame
) -> pd.DataFrame:
    """
    Join sales with calendar and prices, then compute revenue.
    
    Args:
        sales: Long-format sales DataFrame.
        calendar: Calendar dimension DataFrame.
        prices: Prices fact DataFrame.
    
    Returns:
        Complete item_daily DataFrame with all columns.
    """
    logger.info("Joining sales with calendar...")
    
    # Join with calendar on 'd'
    item_daily = sales.merge(calendar, on="d", how="left")
    
    logger.info("Joining with prices...")
    
    # Join with prices on store_id, item_id, and wm_yr_wk
    item_daily = item_daily.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left"
    )
    
    logger.info("Computing revenue...")
    
    # Compute revenue = units * sell_price
    item_daily["revenue"] = item_daily["units"] * item_daily["sell_price"]
    
    # Handle missing prices (set revenue to NaN where price is missing)
    missing_prices = item_daily["sell_price"].isna().sum()
    if missing_prices > 0:
        logger.warning(f"Missing prices for {missing_prices} records ({missing_prices/len(item_daily)*100:.2f}%)")
    
    # Sort by series and date
    item_daily = item_daily.sort_values(["series_id", "date"]).reset_index(drop=True)
    
    logger.info(f"Item daily table complete: {len(item_daily)} records")
    return item_daily


def run_ingestion() -> pd.DataFrame:
    """
    Execute the full ingestion pipeline.
    
    Returns:
        The complete item_daily DataFrame.
    """
    logger.info("Starting M5 data ingestion pipeline")
    
    # Ensure output directory exists
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    # Check for input files
    sales_path = DATA_RAW / SALES_FILE
    calendar_path = DATA_RAW / CALENDAR_FILE
    prices_path = DATA_RAW / PRICES_FILE
    
    for path, name in [(sales_path, "Sales"), (calendar_path, "Calendar"), (prices_path, "Prices")]:
        if not path.exists():
            logger.error(f"{name} file not found: {path}")
            logger.info(f"Please place {path.name} in {DATA_RAW}")
            raise FileNotFoundError(f"{name} file not found: {path}")
    
    # Load data
    calendar = load_calendar(calendar_path)
    prices = load_prices(prices_path)
    sales = load_and_melt_sales(sales_path)
    
    # Join and compute
    item_daily = join_and_compute(sales, calendar, prices)
    
    # Save to parquet
    output_path = DATA_PROCESSED / ITEM_DAILY_FILE
    logger.info(f"Saving to {output_path}")
    item_daily.to_parquet(output_path, index=False)
    
    logger.info("Ingestion pipeline complete")
    return item_daily


def main():
    """Main entry point for the ingestion module."""
    try:
        run_ingestion()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

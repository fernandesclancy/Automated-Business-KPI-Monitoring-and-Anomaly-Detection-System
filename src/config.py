"""
Configuration module for KPI Sentinel.

Contains all global constants and default parameters used throughout the project.
"""

from pathlib import Path
from typing import List

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_REPORTS = PROJECT_ROOT / "outputs" / "reports"
OUTPUTS_FIGURES = PROJECT_ROOT / "outputs" / "figures"

# STL decomposition parameters
STL_PERIOD: int = 7  # Weekly seasonality

# Anomaly detection parameters
MAD_WINDOW: int = 28  # Rolling window for MAD calculation
ANOMALY_Z: float = 4.5  # Z-score threshold for anomaly flagging
COOLDOWN_DAYS: int = 2  # Days to suppress after flagged anomaly
BEFORE_WINDOW_DAYS: int = 28  # Days before peak for RCA comparison
MIN_HISTORY_DAYS: int = 35  # Minimum history required before scoring

# Default KPI metrics
DEFAULT_KPIS: List[str] = [
    "units",
    "revenue",
    "avg_price",
    "price_index",
    "zero_sales_rate",
    "demand_volatility"
]

# Default hierarchy levels
DEFAULT_LEVELS: List[str] = [
    "global",
    "state",
    "store",
    "category",
    "department"
]

# Data file names
SALES_FILE = "sales_train_evaluation.csv"
CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"

# Output file names
ITEM_DAILY_FILE = "item_daily.parquet"
METRIC_FACTS_FILE = "metric_facts.parquet"
ANOMALIES_FILE = "anomalies.parquet"
RCA_DRIVERS_FILE = "rca_drivers.parquet"
GROUND_TRUTH_FILE = "ground_truth_events.parquet"

# Severity calculation parameters
MAX_SEVERITY: float = 100.0
Z_SEVERITY_MULTIPLIER: float = 20.0
PCT_SEVERITY_MULTIPLIER: float = 100.0

# RCA driver score weights
DELTA_VALUE_WEIGHT: float = 0.7
DELTA_SHARE_WEIGHT: float = 0.3

# Logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

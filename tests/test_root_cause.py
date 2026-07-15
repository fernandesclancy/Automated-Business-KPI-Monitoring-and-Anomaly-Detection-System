"""
Unit tests for the root cause analysis module.

Tests cover:
- Driver ranking identifies known injected drivers
- Contribution analysis calculations
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.root_cause import (
    compute_contribution_analysis,
    get_child_segments,
    get_series_values,
)
from src.utils import get_child_level


class TestDriverRanking:
    """Tests for driver ranking logic."""
    
    def test_identifies_injected_driver(self):
        """Test that RCA identifies a known injected driver."""
        # Create synthetic metric_facts with a clear driver
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        
        # Create data for global level and two states
        records = []
        
        # Global level - normal pattern then anomaly
        for i, date in enumerate(dates):
            value = 1000 if i < 50 else 700  # 30% drop at day 50
            records.append({
                "date": date,
                "metric_name": "revenue",
                "level": "global",
                "value": value,
                "state_id": None,
                "store_id": None,
                "cat_id": None,
                "dept_id": None,
            })
        
        # State CA - this is the driver (drops more)
        for i, date in enumerate(dates):
            value = 600 if i < 50 else 300  # 50% drop - this is the driver
            records.append({
                "date": date,
                "metric_name": "revenue",
                "level": "state",
                "value": value,
                "state_id": "CA",
                "store_id": None,
                "cat_id": None,
                "dept_id": None,
            })
        
        # State TX - stable
        for i, date in enumerate(dates):
            value = 400 if i < 50 else 400  # No change
            records.append({
                "date": date,
                "metric_name": "revenue",
                "level": "state",
                "value": value,
                "state_id": "TX",
                "store_id": None,
                "cat_id": None,
                "dept_id": None,
            })
        
        metric_facts = pd.DataFrame(records)
        metric_facts["date"] = pd.to_datetime(metric_facts["date"])
        
        # Create a synthetic event
        event = pd.Series({
            "event_id": "test_001",
            "metric_name": "revenue",
            "level": "global",
            "segment_key": "global",
            "peak_date": pd.Timestamp("2024-02-20"),  # Day 50
            "direction": "drop",
        })
        
        # Get child segments
        child_segments = [("state", "state=CA"), ("state", "state=TX")]
        
        # Run contribution analysis
        drivers = compute_contribution_analysis(metric_facts, event, child_segments)
        
        # CA should be ranked first (it's the driver)
        assert len(drivers) == 2
        assert drivers[0]["child_segment_key"] == "state=CA"
        assert drivers[0]["rank"] == 1
        
        # CA should have larger delta_value (negative)
        assert abs(drivers[0]["delta_value"]) > abs(drivers[1]["delta_value"])
    
    def test_share_change_captured(self):
        """Test that share changes are correctly calculated."""
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        
        records = []
        
        # Global: 1000 before, 1000 after (no change in total)
        for i, date in enumerate(dates):
            records.append({
                "date": date,
                "metric_name": "units",
                "level": "global",
                "value": 1000,
                "state_id": None,
                "store_id": None,
                "cat_id": None,
                "dept_id": None,
            })
        
        # State CA: 500 before, 700 after (share increases)
        for i, date in enumerate(dates):
            value = 500 if i < 50 else 700
            records.append({
                "date": date,
                "metric_name": "units",
                "level": "state",
                "value": value,
                "state_id": "CA",
                "store_id": None,
                "cat_id": None,
                "dept_id": None,
            })
        
        # State TX: 500 before, 300 after (share decreases)
        for i, date in enumerate(dates):
            value = 500 if i < 50 else 300
            records.append({
                "date": date,
                "metric_name": "units",
                "level": "state",
                "value": value,
                "state_id": "TX",
                "store_id": None,
                "cat_id": None,
                "dept_id": None,
            })
        
        metric_facts = pd.DataFrame(records)
        metric_facts["date"] = pd.to_datetime(metric_facts["date"])
        
        event = pd.Series({
            "event_id": "test_002",
            "metric_name": "units",
            "level": "global",
            "segment_key": "global",
            "peak_date": pd.Timestamp("2024-02-20"),
            "direction": "spike",
        })
        
        child_segments = [("state", "state=CA"), ("state", "state=TX")]
        drivers = compute_contribution_analysis(metric_facts, event, child_segments)
        
        # Find CA and TX drivers
        ca_driver = next(d for d in drivers if d["child_segment_key"] == "state=CA")
        tx_driver = next(d for d in drivers if d["child_segment_key"] == "state=TX")
        
        # CA share should increase (from 0.5 to 0.7)
        assert ca_driver["delta_share"] > 0
        
        # TX share should decrease (from 0.5 to 0.3)
        assert tx_driver["delta_share"] < 0


class TestChildSegments:
    """Tests for child segment identification."""
    
    def test_global_children_are_states(self):
        """Global level should have state children."""
        child_level = get_child_level("global")
        assert child_level == "state"
    
    def test_state_children_are_stores(self):
        """State level should have store children."""
        child_level = get_child_level("state")
        assert child_level == "store"
    
    def test_store_children_are_departments(self):
        """Store level should have department children."""
        child_level = get_child_level("store")
        assert child_level == "department"
    
    def test_department_children_are_items(self):
        """Department level should have item children."""
        child_level = get_child_level("department")
        assert child_level == "item"


class TestSeriesValues:
    """Tests for series value retrieval."""
    
    def test_retrieves_correct_values(self):
        """Test that correct values are retrieved for a series."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        
        metric_facts = pd.DataFrame({
            "date": dates,
            "metric_name": ["revenue"] * 10,
            "level": ["state"] * 10,
            "value": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
            "state_id": ["CA"] * 10,
            "store_id": [None] * 10,
            "cat_id": [None] * 10,
            "dept_id": [None] * 10,
        })
        
        # Get values for specific dates
        query_dates = [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-05")]
        values = get_series_values(
            metric_facts,
            "revenue",
            "state",
            "state=CA",
            query_dates
        )
        
        assert len(values) == 2
        assert values.iloc[0] == 120  # Jan 3
        assert values.iloc[1] == 140  # Jan 5
    
    def test_handles_missing_dates(self):
        """Test that missing dates return NaN."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        
        metric_facts = pd.DataFrame({
            "date": dates,
            "metric_name": ["revenue"] * 5,
            "level": ["global"] * 5,
            "value": [100, 110, 120, 130, 140],
            "state_id": [None] * 5,
            "store_id": [None] * 5,
            "cat_id": [None] * 5,
            "dept_id": [None] * 5,
        })
        
        # Query a date that doesn't exist
        query_dates = [pd.Timestamp("2024-01-10")]
        values = get_series_values(
            metric_facts,
            "revenue",
            "global",
            "global",
            query_dates
        )
        
        assert len(values) == 1
        assert pd.isna(values.iloc[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

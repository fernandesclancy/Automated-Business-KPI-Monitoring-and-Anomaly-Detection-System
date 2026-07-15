"""
Unit tests for the anomaly detection module.

Tests cover:
- Event grouping logic for consecutive anomalies
- Cooldown suppression behavior
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detect import (
    group_consecutive_anomalies,
    apply_cooldown,
    compute_anomaly_scores,
)


class TestEventGrouping:
    """Tests for the event grouping logic."""
    
    def test_single_anomaly_creates_single_event(self):
        """A single anomaly day should create one event."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "value": [100] * 10,
            "anomaly_flag": [False, False, False, True, False, False, False, False, False, False],
            "z_severity": [0, 0, 0, 50, 0, 0, 0, 0, 0, 0],
            "pct_severity": [0, 0, 0, 40, 0, 0, 0, 0, 0, 0],
            "pct_change": [0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0],
            "direction": ["drop"] * 10,
            "score": [0, 0, 0, 4.0, 0, 0, 0, 0, 0, 0],
        })
        
        events = group_consecutive_anomalies(df)
        
        assert len(events) == 1
        assert events[0]["duration_days"] == 1
        assert events[0]["start_date"] == pd.Timestamp("2024-01-04")
        assert events[0]["end_date"] == pd.Timestamp("2024-01-04")
    
    def test_consecutive_anomalies_grouped(self):
        """Consecutive anomaly days should be grouped into one event."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "value": [100] * 10,
            "anomaly_flag": [False, False, True, True, True, False, False, False, False, False],
            "z_severity": [0, 0, 40, 60, 50, 0, 0, 0, 0, 0],
            "pct_severity": [0, 0, 35, 55, 45, 0, 0, 0, 0, 0],
            "pct_change": [0, 0, 0.2, 0.4, 0.3, 0, 0, 0, 0, 0],
            "direction": ["spike"] * 10,
            "score": [0, 0, 3.5, 5.0, 4.2, 0, 0, 0, 0, 0],
        })
        
        events = group_consecutive_anomalies(df)
        
        assert len(events) == 1
        assert events[0]["duration_days"] == 3
        assert events[0]["start_date"] == pd.Timestamp("2024-01-03")
        assert events[0]["end_date"] == pd.Timestamp("2024-01-05")
        # Peak should be the day with max z_severity (day 4, index 3)
        assert events[0]["peak_date"] == pd.Timestamp("2024-01-04")
        assert events[0]["z_severity_peak"] == 60
    
    def test_non_consecutive_anomalies_separate_events(self):
        """Non-consecutive anomalies should create separate events."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "value": [100] * 10,
            "anomaly_flag": [False, True, False, False, True, True, False, False, True, False],
            "z_severity": [0, 50, 0, 0, 45, 55, 0, 0, 40, 0],
            "pct_severity": [0, 45, 0, 0, 40, 50, 0, 0, 35, 0],
            "pct_change": [0, 0.3, 0, 0, 0.25, 0.35, 0, 0, 0.2, 0],
            "direction": ["spike"] * 10,
            "score": [0, 4.0, 0, 0, 3.8, 4.5, 0, 0, 3.6, 0],
        })
        
        events = group_consecutive_anomalies(df)
        
        assert len(events) == 3
        
        # First event: day 2
        assert events[0]["duration_days"] == 1
        assert events[0]["start_date"] == pd.Timestamp("2024-01-02")
        
        # Second event: days 5-6
        assert events[1]["duration_days"] == 2
        assert events[1]["start_date"] == pd.Timestamp("2024-01-05")
        assert events[1]["end_date"] == pd.Timestamp("2024-01-06")
        
        # Third event: day 9
        assert events[2]["duration_days"] == 1
        assert events[2]["start_date"] == pd.Timestamp("2024-01-09")
    
    def test_no_anomalies_returns_empty(self):
        """No anomalies should return empty list."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": [100] * 5,
            "anomaly_flag": [False] * 5,
            "z_severity": [0] * 5,
            "pct_severity": [0] * 5,
            "pct_change": [0] * 5,
            "direction": ["drop"] * 5,
            "score": [0] * 5,
        })
        
        events = group_consecutive_anomalies(df)
        
        assert len(events) == 0


class TestCooldownSuppression:
    """Tests for the cooldown suppression logic."""
    
    def test_cooldown_suppresses_nearby_anomalies(self):
        """Anomalies within cooldown period should be suppressed."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "value": [100] * 10,
            "anomaly_flag": [False, True, True, True, False, False, False, False, False, False],
            "z_severity": [0, 50, 45, 40, 0, 0, 0, 0, 0, 0],
        })
        df.index = range(len(df))
        
        result = apply_cooldown(df, cooldown_days=2)
        
        # First anomaly should remain, next two within cooldown should be suppressed
        # (unless severity increases by >25%)
        assert result.loc[1, "anomaly_flag"] == True
        assert result.loc[2, "anomaly_flag"] == False  # Suppressed (severity decreased)
        assert result.loc[3, "anomaly_flag"] == False  # Suppressed (severity decreased)
    
    def test_cooldown_allows_severity_increase(self):
        """Anomalies with >25% severity increase should not be suppressed."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "value": [100] * 10,
            "anomaly_flag": [False, True, True, False, False, False, False, False, False, False],
            "z_severity": [0, 40, 60, 0, 0, 0, 0, 0, 0, 0],  # 50% increase
        })
        df.index = range(len(df))
        
        result = apply_cooldown(df, cooldown_days=2)
        
        # Both should remain because severity increased by >25%
        assert result.loc[1, "anomaly_flag"] == True
        assert result.loc[2, "anomaly_flag"] == True
    
    def test_cooldown_resets_after_period(self):
        """Anomalies after cooldown period should not be suppressed."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "value": [100] * 10,
            "anomaly_flag": [False, True, False, False, False, True, False, False, False, False],
            "z_severity": [0, 50, 0, 0, 0, 45, 0, 0, 0, 0],
        })
        df.index = range(len(df))
        
        result = apply_cooldown(df, cooldown_days=2)
        
        # First anomaly at index 1
        # Second anomaly at index 5 (4 days later, outside cooldown)
        assert result.loc[1, "anomaly_flag"] == True
        assert result.loc[5, "anomaly_flag"] == True


class TestAnomalyScoring:
    """Tests for the anomaly scoring computation."""
    
    def test_scores_computed_correctly(self):
        """Verify that anomaly scores are computed."""
        # Create a simple series with a clear anomaly
        np.random.seed(42)
        n = 100
        values = np.random.normal(100, 5, n)
        values[80] = 200  # Clear spike
        
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "value": values
        })
        
        result = compute_anomaly_scores(df)
        
        # Check that required columns exist
        assert "baseline" in result.columns
        assert "residual" in result.columns
        assert "score" in result.columns
        assert "anomaly_flag" in result.columns
        assert "z_severity" in result.columns
        assert "pct_severity" in result.columns
        assert "direction" in result.columns
        
        # The spike should be detected as an anomaly
        # (after MIN_HISTORY_DAYS)
        assert result.loc[80, "direction"] == "spike"
    
    def test_early_period_masked(self):
        """Verify that early period is masked (no scores)."""
        n = 50
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "value": np.random.normal(100, 5, n)
        })
        
        result = compute_anomaly_scores(df)
        
        # First MIN_HISTORY_DAYS should have no anomaly flags
        from src.config import MIN_HISTORY_DAYS
        early_flags = result.loc[:MIN_HISTORY_DAYS-1, "anomaly_flag"]
        assert not early_flags.any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

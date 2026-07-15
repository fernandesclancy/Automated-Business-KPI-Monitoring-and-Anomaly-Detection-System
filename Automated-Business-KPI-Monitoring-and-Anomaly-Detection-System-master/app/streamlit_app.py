"""
KPI Sentinel Streamlit Dashboard.

A multi-page dashboard for exploring KPI anomalies and their root causes.

Pages:
1. Overview - KPI trends with anomaly markers
2. Root Cause Explorer - Drill down into anomaly drivers
3. Incident Report Generator - Generate and download reports
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from src.config import (
    DATA_PROCESSED,
    METRIC_FACTS_FILE,
    ANOMALIES_FILE,
    RCA_DRIVERS_FILE,
    DEFAULT_KPIS,
    DEFAULT_LEVELS,
)
from src.report import generate_report
from src.utils import format_number, format_percentage


# Page configuration
st.set_page_config(
    page_title="KPI Sentinel",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_metric_facts():
    """Load metric facts data."""
    path = DATA_PROCESSED / METRIC_FACTS_FILE
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_anomalies():
    """Load anomalies data."""
    path = DATA_PROCESSED / ANOMALIES_FILE
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "peak_date" in df.columns:
        df["peak_date"] = pd.to_datetime(df["peak_date"])
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"])
    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date"])
    return df


@st.cache_data
def load_rca_drivers():
    """Load RCA drivers data."""
    path = DATA_PROCESSED / RCA_DRIVERS_FILE
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def get_segment_options(metric_facts: pd.DataFrame, level: str) -> list:
    """Get available segment options for a level."""
    if level == "global":
        return ["global"]
    
    level_df = metric_facts[metric_facts["level"] == level]
    
    if level == "state":
        return sorted(level_df["state_id"].dropna().unique().tolist())
    elif level == "store":
        return sorted(level_df["store_id"].dropna().unique().tolist())
    elif level == "category":
        return sorted(level_df["cat_id"].dropna().unique().tolist())
    elif level == "department":
        return sorted(level_df["dept_id"].dropna().unique().tolist())
    
    return []


def filter_series(
    metric_facts: pd.DataFrame,
    metric_name: str,
    level: str,
    segment: str
) -> pd.DataFrame:
    """Filter metric facts for a specific series."""
    mask = (
        (metric_facts["metric_name"] == metric_name) &
        (metric_facts["level"] == level)
    )
    
    if level == "state":
        mask &= metric_facts["state_id"] == segment
    elif level == "store":
        mask &= metric_facts["store_id"] == segment
    elif level == "category":
        mask &= metric_facts["cat_id"] == segment
    elif level == "department":
        mask &= metric_facts["dept_id"] == segment
    
    return metric_facts[mask].sort_values("date")


def build_segment_key(level: str, segment: str) -> str:
    """Build segment key string."""
    if level == "global":
        return "global"
    return f"{level}={segment}"


# Sidebar navigation
st.sidebar.title("KPI Sentinel")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Root Cause Explorer", "Incident Report Generator"]
)

# Load data
metric_facts = load_metric_facts()
anomalies = load_anomalies()
rca_drivers = load_rca_drivers()

# Check if data exists
if len(metric_facts) == 0:
    st.error("No data found. Please run the data pipeline first:")
    st.code("""
python -m src.m5_ingest
python -m src.kpi_build
python -m src.detect
python -m src.root_cause
    """)
    st.stop()


# Page 1: Overview
if page == "Overview":
    st.title("KPI Overview Dashboard")
    st.markdown("Explore KPI trends and detected anomalies across the retail hierarchy.")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        metric = st.selectbox(
            "Metric",
            options=metric_facts["metric_name"].unique().tolist(),
            index=0
        )
    
    with col2:
        level = st.selectbox(
            "Level",
            options=metric_facts["level"].unique().tolist(),
            index=0
        )
    
    with col3:
        segments = get_segment_options(metric_facts, level)
        segment = st.selectbox(
            "Segment",
            options=segments,
            index=0 if segments else None
        )
    
    # Date range
    date_col1, date_col2 = st.columns(2)
    min_date = metric_facts["date"].min().date()
    max_date = metric_facts["date"].max().date()
    
    with date_col1:
        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    
    with date_col2:
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
    
    # Filter data
    series_df = filter_series(metric_facts, metric, level, segment)
    series_df = series_df[
        (series_df["date"].dt.date >= start_date) &
        (series_df["date"].dt.date <= end_date)
    ]
    
    if len(series_df) == 0:
        st.warning("No data available for the selected filters.")
    else:
        # Time series plot
        st.subheader(f"{metric} - {level}: {segment}")
        
        fig = go.Figure()
        
        # Main series
        fig.add_trace(go.Scatter(
            x=series_df["date"],
            y=series_df["value"],
            mode="lines",
            name="Actual",
            line=dict(color="blue", width=2)
        ))
        
        # Get anomalies for this series
        segment_key = build_segment_key(level, segment)
        series_anomalies = anomalies[
            (anomalies["metric_name"] == metric) &
            (anomalies["level"] == level) &
            (anomalies["segment_key"] == segment_key)
        ]
        
        # Add anomaly markers
        if len(series_anomalies) > 0:
            for _, anomaly in series_anomalies.iterrows():
                peak_date = anomaly["peak_date"]
                peak_value = series_df[series_df["date"] == peak_date]["value"].values
                
                if len(peak_value) > 0:
                    color = "red" if anomaly["direction"] == "spike" else "orange"
                    fig.add_trace(go.Scatter(
                        x=[peak_date],
                        y=[peak_value[0]],
                        mode="markers",
                        name=f"Anomaly ({anomaly['direction']})",
                        marker=dict(color=color, size=12, symbol="diamond"),
                        showlegend=False
                    ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=metric,
            hovermode="x unified",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly events table
        st.subheader("Detected Anomaly Events")
        
        if len(series_anomalies) > 0:
            display_cols = [
                "event_id", "peak_date", "direction", "duration_days",
                "z_severity_peak", "pct_change_peak"
            ]
            display_cols = [c for c in display_cols if c in series_anomalies.columns]
            
            display_df = series_anomalies[display_cols].copy()
            display_df["peak_date"] = display_df["peak_date"].dt.strftime("%Y-%m-%d")
            display_df["z_severity_peak"] = display_df["z_severity_peak"].round(1)
            display_df["pct_change_peak"] = (display_df["pct_change_peak"] * 100).round(1).astype(str) + "%"
            
            display_df.columns = ["Event ID", "Peak Date", "Direction", "Duration", "Severity", "Change %"]
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No anomalies detected for this series.")


# Page 2: Root Cause Explorer
elif page == "Root Cause Explorer":
    st.title("Root Cause Explorer")
    st.markdown("Investigate the drivers behind detected anomalies.")
    
    if len(anomalies) == 0:
        st.warning("No anomalies detected. Run the detection pipeline first.")
        st.stop()
    
    # Event selector
    event_options = anomalies["event_id"].tolist()
    
    # Create display labels
    event_labels = []
    for _, row in anomalies.iterrows():
        label = f"{row['event_id']} - {row['metric_name']} ({row['level']}: {row['segment_key']})"
        event_labels.append(label)
    
    selected_label = st.selectbox("Select Event", options=event_labels, index=0)
    selected_event_id = selected_label.split(" - ")[0]
    
    # Get event details
    event = anomalies[anomalies["event_id"] == selected_event_id].iloc[0]
    
    # Event summary card
    st.subheader("Event Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Metric", event["metric_name"])
    
    with col2:
        st.metric("Direction", event["direction"].upper())
    
    with col3:
        st.metric("Severity", f"{event.get('z_severity_peak', 0):.1f}")
    
    with col4:
        st.metric("Duration", f"{event.get('duration_days', 1)} days")
    
    # Calendar context
    st.subheader("Calendar Context")
    
    event_drivers = rca_drivers[rca_drivers["event_id"] == selected_event_id] if len(rca_drivers) > 0 else pd.DataFrame()
    
    if len(event_drivers) > 0 and "weekday" in event_drivers.columns:
        first_driver = event_drivers.iloc[0]
        
        ctx_col1, ctx_col2, ctx_col3, ctx_col4 = st.columns(4)
        
        with ctx_col1:
            st.metric("Weekday", first_driver.get("weekday", "N/A"))
        
        with ctx_col2:
            weekend = "Yes" if first_driver.get("is_weekend", False) else "No"
            st.metric("Weekend", weekend)
        
        with ctx_col3:
            event_name = first_driver.get("event_name", None)
            st.metric("Calendar Event", event_name if event_name else "None")
        
        with ctx_col4:
            snap = "Yes" if first_driver.get("snap_flag", False) else "No"
            st.metric("SNAP Active", snap)
    else:
        st.info("Calendar context not available.")
    
    # Top drivers
    st.subheader("Top Drivers")
    
    if len(event_drivers) > 0:
        # Sort by rank and take top 10
        top_drivers = event_drivers.sort_values("rank").head(10)
        
        # Bar chart
        fig = px.bar(
            top_drivers,
            x="driver_score",
            y="child_segment_key",
            orientation="h",
            title="Driver Scores by Segment",
            labels={"driver_score": "Driver Score", "child_segment_key": "Segment"}
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.subheader("Driver Details")
        
        display_cols = [
            "rank", "child_segment_key", "child_before", "child_after",
            "delta_value", "delta_share", "driver_score"
        ]
        display_cols = [c for c in display_cols if c in top_drivers.columns]
        
        display_df = top_drivers[display_cols].copy()
        display_df["child_before"] = display_df["child_before"].apply(lambda x: format_number(x))
        display_df["child_after"] = display_df["child_after"].apply(lambda x: format_number(x))
        display_df["delta_value"] = display_df["delta_value"].apply(lambda x: format_number(x))
        display_df["delta_share"] = display_df["delta_share"].apply(lambda x: format_percentage(x))
        display_df["driver_score"] = display_df["driver_score"].apply(lambda x: format_number(x))
        
        display_df.columns = ["Rank", "Segment", "Before", "After", "Delta", "Share Change", "Score"]
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Drilldown option
        st.subheader("Drilldown")
        
        drilldown_options = top_drivers["child_segment_key"].tolist()
        selected_drilldown = st.selectbox(
            "Select a segment to drill down",
            options=drilldown_options,
            index=0
        )
        
        st.info(f"To drill down into {selected_drilldown}, filter the Overview page with this segment.")
        
    else:
        st.info("No driver analysis available for this event.")


# Page 3: Incident Report Generator
elif page == "Incident Report Generator":
    st.title("Incident Report Generator")
    st.markdown("Generate detailed incident reports for anomaly events.")
    
    if len(anomalies) == 0:
        st.warning("No anomalies detected. Run the detection pipeline first.")
        st.stop()
    
    # Event selector
    event_options = anomalies["event_id"].tolist()
    
    event_labels = []
    for _, row in anomalies.iterrows():
        label = f"{row['event_id']} - {row['metric_name']} ({row['level']}: {row['segment_key']})"
        event_labels.append(label)
    
    selected_label = st.selectbox("Select Event for Report", options=event_labels, index=0)
    selected_event_id = selected_label.split(" - ")[0]
    
    # Generate button
    if st.button("Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            try:
                report_content = generate_report(
                    selected_event_id,
                    anomalies,
                    rca_drivers
                )
                
                st.success("Report generated successfully!")
                
                # Display preview
                st.subheader("Report Preview")
                st.markdown(report_content)
                
                # Download button
                st.download_button(
                    label="Download Report (Markdown)",
                    data=report_content,
                    file_name=f"event_{selected_event_id}.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"Failed to generate report: {str(e)}")
    
    # Quick stats
    st.subheader("Event Quick Stats")
    
    event = anomalies[anomalies["event_id"] == selected_event_id].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Event Details**")
        st.write(f"- Metric: {event['metric_name']}")
        st.write(f"- Level: {event['level']}")
        st.write(f"- Segment: {event['segment_key']}")
        st.write(f"- Direction: {event['direction']}")
    
    with col2:
        st.markdown("**Timing and Severity**")
        st.write(f"- Peak Date: {event.get('peak_date', 'N/A')}")
        st.write(f"- Duration: {event.get('duration_days', 1)} days")
        st.write(f"- Z-Severity: {event.get('z_severity_peak', 0):.1f}")
        st.write(f"- Change: {event.get('pct_change_peak', 0)*100:.1f}%")


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**KPI Sentinel v1.0**")
st.sidebar.markdown("Retail KPI Anomaly Detection")

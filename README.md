# Automated Business KPI Monitoring and Root Cause Analysis System

Designed and built an automated business KPI monitoring system to detect anomalies and identify root causes across large-scale retail data, enabling proactive performance monitoring and faster decision-making.

---

## Demo

### Dashboard Overview
![Dashboard Overview](https://github.com/user-attachments/assets/fc98fd3b-3f46-474e-a459-7c8c8371b5b2)

### Root Cause Explorer
![Root Cause Explorer](https://github.com/user-attachments/assets/937d21c5-eda1-4b22-bc19-415e21f5ac4d)

### Walkthrough Video
[Watch Demo Video](https://github.com/user-attachments/assets/39b9592b-3ac3-41f1-a793-6aace36f80fb)

---

## Overview

This system simulates how real businesses monitor operational performance and detect issues early.

It provides:

1. Automated KPI computation across multiple business dimensions  
2. Statistical anomaly detection with seasonality awareness  
3. Hierarchical root cause analysis for issue diagnosis  
4. Interactive dashboard and incident reporting for decision-making  

This is not just a model. It is a complete monitoring and decision-support system designed for real-world analytics workflows.

---

## System Impact

1. Processes 30K+ product-level time series across multiple hierarchy levels  
2. Detects anomalies at daily granularity using robust statistical thresholds  
3. Reduces manual effort required for KPI monitoring and issue detection  
4. Enables faster root cause identification through structured analysis  
5. Improves operational visibility across business segments  

---

## Why This System Matters

Businesses rely on KPIs such as revenue, demand, and pricing to monitor performance, but manual monitoring often leads to delayed issue detection.

This system:

1. Automates KPI tracking across multiple business dimensions  
2. Detects anomalies early before they significantly impact performance  
3. Provides root cause insights to accelerate decision-making  
4. Reduces dependency on manual dashboard monitoring  

---

## Dataset

M5 Forecasting – Accuracy dataset:  
https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Required files in data/raw/:

1. sales_train_evaluation.csv  
2. calendar.csv  
3. sell_prices.csv  

---

## Data Hierarchy

Global  
State (CA, TX, WI)  
Store  
Department  
Item  

---

## Key KPIs

1. units: total units sold  
2. revenue: units multiplied by price  
3. avg_price: revenue divided by units  
4. price_index: price relative to recent history  
5. zero_sales_rate: proportion of zero-sale items  
6. demand_volatility: variability of demand  

---

## Methodology

### Baseline Modeling
Seasonality-aware baseline using STL decomposition with weekly patterns.

### Anomaly Detection
Residual-based scoring using Median Absolute Deviation to identify deviations from expected trends.

### Severity Measurement
1. Statistical deviation from baseline  
2. Percentage change relative to expected values  

### False Positive Control
1. Minimum history requirement  
2. Cooldown period after anomaly detection  
3. Grouping of consecutive anomaly events  

### Root Cause Analysis
1. Compare historical and anomaly periods  
2. Measure contribution changes across hierarchy  
3. Identify top drivers of performance changes  

---

## Evaluation

Synthetic anomaly injection is used:

1. Introduce controlled anomalies  
2. Measure detection performance using precision and recall  
3. Evaluate detection speed and reliability  

---

## System Architecture

Data Ingestion → KPI Computation → Baseline Modeling → Anomaly Detection → Root Cause Analysis → Reporting → Dashboard

---

## Repository Structure

kpi-monitoring-system/

README.md  
requirements.txt  

data/  
raw/  
processed/  

src/  
config.py  
utils.py  
m5_ingest.py  
kpi_build.py  
baseline.py  
detect.py  
root_cause.py  
report.py  

app/  
streamlit_app.py  

notebooks/  
01_data_prep_eda.ipynb  
02_detection_eval.ipynb  
03_rca_examples.ipynb  

outputs/  
reports/  
figures/  

tests/  
test_detect.py  
test_root_cause.py  

---

## Quickstart

### Install
```bash
pip install -r requirements.txt

python -m src.m5_ingest
python -m src.kpi_build
python -m src.detect
python -m src.root_cause

streamlit run app/streamlit_app.py

# Data Directory

This directory contains all data files and preprocessing scripts for the vocational training recommendation system.

## Contents

- `raw/` - Raw data files (employment statistics, training program details)
- `processed/` - Cleaned and preprocessed data ready for model training
- `preprocessing.py` - Scripts for data cleaning and feature engineering

## Data Sources

The following data sources should be collected and placed in the `raw/` directory:

1. Regional employment statistics (unemployment rates, job growth by sector)
2. Industry demand forecasts
3. Vocational training program details (duration, cost, skills taught)
4. Skills gap analysis reports

## Data Format

Data should be stored in CSV or JSON format for easy processing with pandas.
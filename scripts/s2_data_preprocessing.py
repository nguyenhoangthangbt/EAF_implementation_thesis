#!/usr/bin/env python
"""
Data Preprocessing Script
Cleans and prepares raw gold price data for feature engineering
"""

import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.data_utils import load_raw_data, clean_data,load_raw_data_mt5
from utils.config_utils import load_config

def run_data_preprocessing(load_mt5_data=False, extra_symbols=['EURUSD']):
    """Main function for data preprocessing"""
    print("Starting data preprocessing")
        
    # Load raw data
    raw_data_dir = project_root / "data" / "raw"
    processed_data_dir = project_root / "data" / "processed"
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the latest raw data file
    raw_files = list(raw_data_dir.glob("gold_price_*.csv"))
    if not raw_files:
        print("No raw data files found")
        return 1
    
    latest_file = max(raw_files, key=os.path.getctime)
    print(f"Loading raw data from {latest_file}")

    
    # Load and clean data
    if load_mt5_data:
        df = load_raw_data_mt5("XAUUSD") 
        Volume= load_raw_data(latest_file)['Volume']
        df['Volume']=Volume
    else: 
        df = load_raw_data(latest_file)
    
    df_cleaned = clean_data(df)

    if len(extra_symbols)>0:
        for symbol in extra_symbols:
            if symbol: df_cleaned[symbol] = load_raw_data_mt5(symbol)['Close'] # symbol is not None
    # df_cleaned = transform_to_price_differences(df_cleaned)
    df_cleaned.index.rename('Date',inplace=True) # Raw data 1st column name = Price, but correct name is Date
    
    # Save processed data
    filename = f"gold_price_processed{'_mt5' if load_mt5_data else ''}.csv"
    
    output_path = processed_data_dir / filename
    df_cleaned.to_csv(output_path)
    print(f"Processed data saved to {output_path}")
    
    # Log data statistics
    print(f"Original data shape: {df.shape}")
    print(f"Processed data shape: {df_cleaned.shape}")
    print(f"Number of missing values: {df_cleaned.isnull().sum().sum()}")
    print(f"Date range: {df_cleaned.index.min()} to {df_cleaned.index.max()}")
    
    print("Data preprocessing completed successfully")
    return df_cleaned
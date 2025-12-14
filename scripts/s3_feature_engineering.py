#!/usr/bin/env python
"""
Feature Engineering Script
Creates autoregressive features, lags, volatility measures, and technical indicators
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
from utils.config_utils import load_config
from utils.data_utils import load_processed_data
from utils.feature_utils import (
    create_Log_Return,
    create_date_features,
    create_lag_features,
    create_rolling_features,
    create_target_variable
)

from utils_strategies import signals_TAlib_utils

def run_feature_engineering(load_mt5_data:bool=False,base_features =['Close'],\
        target_isClose=False,features_filepath='' ):
    """Main function for feature engineering"""
    print("Starting feature engineering process")
    
    # Load configuration
    config = load_config()
    
    # Load processed data
    processed_data_dir = project_root / "data" / "processed"
    features_dir = project_root / "data" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the latest processed data file
    processed_files = list(processed_data_dir.glob(f"gold_price_processed*{'_mt5' if load_mt5_data else ''}.csv"))
    if not processed_files:
        print("No processed data files found")
        return 1
    
    latest_file = max(processed_files, key=os.path.getctime)
    print(f"Loading processed data from {latest_file}")
    
    df = load_processed_data(latest_file)

    print("Creating target variable")
    df=create_target_variable(df,target_isClose)
    
    df=create_date_features(df,cyclical_encoding=True,add_weekend=False,add_quarter=True,country='US')
    ##################################################################
    df=create_Log_Return(df,base_features,target_isClose)

    for price in base_features:# not include log return due to poor perf
        ### Generate lag, rolling #############################  
        if price:
            df=create_lag_features(df,price) #,feature_config['lags']
            df=create_rolling_features(df,price) #,feature_config['rolling_windows']

    ### Generate all TAlib features #############################
    
    print("Creating all TAlib features")
    print('df shape before TAlib features: ', df.shape)

    singal_gen = signals_TAlib_utils.TALibSignalGenerator(df)
    config_ta = load_config('ta_config.yaml')
    config_paras_dict = {
                            'include_momentum':True,
                            'include_volume':True,
                            'include_volatility':True,
                            'include_overlap':True,
                            'include_cycle':True,
                            'include_price':True,
                            'include_statistic':True}
    ta_short_term_paras = {'momentum_params':config_ta['short_term']['momentum'],
                            'volume_params':config_ta['short_term']['volume'],
                            'volatility_params':config_ta['short_term']['volatility'],
                            'overlap_params':config_ta['short_term']['overlap'],
                            'cycle_params':config_ta['short_term']['cycle'],
                            'price_params':config_ta['short_term']['price'],
                            'statistic_params':config_ta['short_term']['statistic']}
    ##generate_all_signals
    singals_short = singal_gen.generate_all_signals(column='Close', **config_paras_dict, **ta_short_term_paras)
    df_singals_short = pd.DataFrame(singals_short)
    df=pd.concat([df,df_singals_short],axis=1)     
    
    ################################
    print('df_concat shape: ', df.shape)
    # Remove rows with NaN values (from feature creation)
    df.index.rename('Date',inplace=True)
    initial_shape = df.shape
    df = df.dropna()
    print(f"Removed {initial_shape[0] - df.shape[0]} rows with NaN values")
    cur_date = datetime.now().strftime("%Y%m%d")
    # Save feature-engineered data
    df.to_csv(features_filepath)
    # print('First 20 features :\n',df.iloc[:,0:20].head().to_string())
    # print(df.target)
    print(f"Feature-engineered data saved to {features_filepath}")
    
    # Log feature statistics
    print(f"Total features created: {len(df.columns) - 1}")  # -1 for target
    print(f"Final data shape: {df.shape}")
    
    print("Feature engineering completed successfully")
    return df

def remove_multicollinear(df, threshold=0.9):
    """
    Remove multicollinearity using the same correlation-based strategy
    
    Parameters:
        df: pandas DataFrame (numeric features only)
        threshold: float (correlation threshold, e.g., 0.9)
    
    Returns:
        DataFrame with selected features.
    """

    df_corr = df.corr().abs()

    # Create a matrix that only considers upper triangle correlation pairs
    upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))

    # Loop while any correlation exceeds threshold
    while any(upper.max() > threshold):
        
        # Find the pair exceeding threshold
        feature_to_drop = upper.idxmax()[upper.max() > threshold][0]
        
        # Compute mean correlation of each feature to decide which one is more redundant
        mean_corr = df_corr.mean()
        
        # Find correlated features with feature_to_drop
        correlated_features = upper[feature_to_drop][upper[feature_to_drop] > threshold].index.tolist()

        # Among the correlated features + itself, remove the feature with highest average correlation
        drop_candidates = correlated_features + [feature_to_drop]
        redundant_feature = mean_corr[drop_candidates].idxmax()

        # Drop the redundant feature and recompute matrices
        df = df.drop(columns=[redundant_feature])
        df_corr = df.corr().abs()
        upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))

    return df

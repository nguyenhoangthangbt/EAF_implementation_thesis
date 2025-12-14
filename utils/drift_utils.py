"""
Drift Detection Utilities
Functions for concept drift detection and adaptive retraining
"""

from river import drift
import xgboost as xgb
import numpy as np
import logging, json
from .data_utils import get_retraining_data

def initialize_adwin(delta=0.002, clock=32):
    """Initialize ADWIN drift detector"""
    print(f"Initializing ADWIN with delta={delta}, clock={clock}")
    return drift.ADWIN(delta=delta, clock=clock)

def detect_drift(adwin, error):
    """Detect concept drift using ADWIN"""
    adwin.update(error)
    return adwin.drift_detected

def retrain_model(model, retrain_data, current_date):
    """Retrain model on appropriate historical window including pre-test data"""
    print(f"Retraining model using data up to {current_date}")
        
    # Extract features and target
    X_retrain = retrain_data.drop('target', axis=1)
    y_retrain = retrain_data['target']
    
    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_retrain, label=y_retrain)
    
    # Get current model parameters as dictionary
    params_str = model.save_config()
    params = json.loads(params_str)
    
    # Get the number of boosting rounds
    num_boost_round = model.num_boosted_rounds()
    
    # Retrain
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        verbose_eval=False
    )
    
    return model
"""
Evaluation Utilities
Functions for calculating performance metrics
"""

import numpy as np
import pandas as pd
import logging

def calculate_statistical_metrics(y_true, y_pred):
    """Calculate statistical metrics"""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }
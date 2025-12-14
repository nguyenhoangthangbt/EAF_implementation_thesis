"""
Model Utilities
Functions for model training and evaluation
"""
import random
import numpy as np
import xgboost as xgb
import logging,sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def objective(trial, X_train, y_train, X_val, y_val, model_config,GLOBAL_SEED,verbose):
    """Objective function for Optuna hyperparameter optimization"""
    # Define hyperparameter search space
    params = {
        'objective': model_config['objective'],
        'eval_metric': model_config['eval_metric'],
        'eta': trial.suggest_float('eta', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-8, 100.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 100.0, log=True),
        'tree_method': 'hist',
        'seed': GLOBAL_SEED,
        'random_state': GLOBAL_SEED,
    }
    
    # Train model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'validation')],
        early_stopping_rounds=model_config['early_stopping_rounds'],
        verbose_eval=verbose
    )
    
    # Evaluate
    val_pred = model.predict(dval)
    rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
    
    return rmse

def get_best_model(best_params, X_train, y_train, X_val, y_val,GLOBAL_SEED):
    """Train model with best hyperparameters"""
    # Add best parameters to config
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': best_params['eta'],
        'max_depth': best_params['max_depth'],
        'min_child_weight': best_params['min_child_weight'],
        'subsample': best_params['subsample'],
        'colsample_bytree': best_params['colsample_bytree'],
        'lambda': best_params['lambda'],
        'alpha': best_params['alpha'],
        'tree_method': 'hist',
        'seed': GLOBAL_SEED,
        'random_state': GLOBAL_SEED,
    }
    
    # Train model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'validation')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    return model

def load_model(model_path):
    """Load XGBoost model from file"""
    print(f"Loading model from {model_path}")
    model = xgb.Booster()
    model.load_model(str(model_path))
    return model

def save_model(model, model_path):
    """Save XGBoost model to file"""
    print(f"Saving model to {model_path}")
    model.save_model(str(model_path))
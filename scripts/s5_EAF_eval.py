#!/usr/bin/env python
"""
Gold Price Forecasting Evaluation Script
Implements the sequential evaluation of the Explainable Adaptive Framework (EAF)
following the methodology described in the research paper.
"""
import os,random
import sys
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import xgboost as xgb
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from utils.config_utils import load_config

# Import utility modules
from utils.data_utils import load_feature_data, prepare_time_series_data, get_retraining_data
from utils.model_utils import load_model, save_model
from utils.drift_utils import initialize_adwin, detect_drift,retrain_model
from utils.eval_utils import calculate_statistical_metrics
from utils_strategies.backtest_utils import backtest_strategy
# Configure logging
log_dir = project_root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def save_results(metrics, trading_results, results_dir, timestamp):
    """Save evaluation results with proper handling of different array lengths"""

    # Save main metrics
    metrics_path = results_dir / f"metrics_{timestamp}.yaml"
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f)
    
    # Create a copy without trade_details (which has different length)
    results_to_save = trading_results.copy()
    
    # Save trade details separately if it exists
    if 'trade_details' in results_to_save:
        trade_details = results_to_save.pop('trade_details')
        
        # Save trade details as JSON
        trade_details_path = results_dir / f"trade_details_{timestamp}.json"
        with open(trade_details_path, 'w') as f:
            json.dump(trade_details, f, indent=2)

    # Now all arrays in results_to_save have the same length
    trading_results_path = results_dir / f"trading_results_{timestamp}.csv"
    pd.DataFrame(results_to_save).to_csv(trading_results_path, index=False)

    return metrics_path, trading_results_path

def run_adpative(GLOBAL_SEED,target_isClose=False,features_filepath='',Long_allowed=True,Short_allowed=False,adwin_config='adwin_default',write_OHLCV_output=False):
        """Main function for model evaluation with proper sequential processing"""
        # Load configuration
        np.random.seed(GLOBAL_SEED)
        random.seed(GLOBAL_SEED)
        
        print(f'Run EAF evaluation | GLOBAL_SEED = {GLOBAL_SEED}...')
        config = load_config()
        drift_config = config['drift'][adwin_config]
        print(f'drift_config = {adwin_config} \n',drift_config)
        time_config = config['time_periods']
        
        # Setup directories
        results_dir = project_root / "results" / "metrics"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        df = load_feature_data(features_filepath)
        
        # Prepare time series data with historical context
        data_dict = prepare_time_series_data(df, time_config)
        full_dataset = data_dict['full']
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_val = data_dict['X_val']
        y_val = data_dict['y_val']
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        
        # Load initial model (trained on training+validation data)
        models_dir = project_root / "models" / "trained"
        ## load the latest static XGBM
        # model_files = list(models_dir.glob(f"xgboost_model_{'Close' if target_isClose else 'LogReturn'}*.json"))
        # model_path = max(model_files, key=os.path.getctime)

        model_path = models_dir / f"xgboost_model_{'Close' if target_isClose else 'LogReturn'}{GLOBAL_SEED}.json"
        model = load_model(model_path)

        list_results = []
        for period in ['training','validating','testing']: #,
            if period=='training':
                df_OHLCV=X_train.copy()
                true_values= y_train.copy()
            elif period=='validating':
                df_OHLCV=X_val.copy()
                true_values= y_val.copy()
            elif period=='testing':
                df_OHLCV=X_test.copy()
                true_values= y_test.copy()
            #############################################
            current_model = model  # Start with initial model
            # Initialize ADWIN drift detector
            adwin = initialize_adwin(
                delta=drift_config['delta'],
                clock=drift_config['clock']
            )

            # To store sequential evaluation results
            predictions = []
            errors = []
            drift_points = []  # Indices where drift was detected
            drift_dates = []   # Dates when drift was detected
            # Process test data sequentially (TRUE EAF EVALUATION)
            total_days = len(df_OHLCV)

            for i in range(total_days):
                current_date = df_OHLCV.index[i]
                progress = (i + 1) / total_days * 100
                
                # 1. Predict using CURRENT model
                dmatrix = xgb.DMatrix(df_OHLCV.iloc[[i]])
                y_pred = current_model.predict(dmatrix)[0]
                predictions.append(y_pred)
                
                # 2. Observe actual value and compute error
                y_true = true_values.iloc[i]
                error = abs(y_true - y_pred)
                errors.append(error)
                
                # 3. Check for drift using ADWIN
                drift_detected = detect_drift(adwin, error)
                
                # 4. Handle drift detection and retraining
                if drift_detected:#(i + 1)%5==0:# 
                    drift_points.append(i)
                    drift_dates.append(current_date)
                    # Get retraining data (includes historical data, not just test data)

                    retrain_data = get_retraining_data(
                        full_dataset, 
                        current_date,
                        drift_config['retrain_window']
                    )
                    
                    # Retrain the model
                    current_model = retrain_model(
                        current_model, 
                        retrain_data,
                        current_date
                    )
            
            print(period, ' - drift dates: \n',drift_dates)
            df_OHLCV_backtest=df_OHLCV.copy()
            if not target_isClose:
                # df['target'] = np.log(df['Close'].shift(-2) / df['Close'].shift(-1))*100
                predicted_pct_return = np.exp(np.array(predictions)/100)-1 
            
                #genrate signals
                signals = pd.Series(0, index=df_OHLCV.index)
                if Long_allowed:
                    signals[predicted_pct_return > 0]= 1
                if Short_allowed:
                    signals[predicted_pct_return < 0]= -1
                
                # df_OHLCV['prediction_label'] = predictions['prediction_label']
                
                df_OHLCV_backtest['predicted_pct_return'] = predicted_pct_return
                predicted_close=None
            else:
                predicted_close = pd.Series(predictions,index=df_OHLCV.index)
                signals=None
            
            # Run backtest using your existing backtester
            backtester1=backtest_strategy(pct_transaction_fee=0.0)
            result_dict = backtester1.run_backtest2_SL_TP(
                Long_allowed=Long_allowed,
                Short_allowed=Short_allowed,
                df_OHLCV=df_OHLCV_backtest,
                signals=signals,
                predicted_close=predicted_close,
                strategy_name="EAF",
                strategy_type='moc',
                execution_threshold=0.0,
                risk_free_rate=0.0,
                write_OHLCV_output=write_OHLCV_output
            )
            statistical_metrics = calculate_statistical_metrics(true_values.values, np.array(predictions))
            list_results.append({
                            'GLOBAL_SEED':GLOBAL_SEED,
                            'baseline_modelName':'baseline_EAF',
                            'target_isClose':target_isClose,
                            'period':period,
                            **statistical_metrics,
                            **result_dict['metrics'],
                            'at':str(datetime.now())
                            })
        return list_results,current_model,drift_dates,X_test, y_test
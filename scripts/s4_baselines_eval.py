#!/usr/bin/env python
"""
Model Training Script
Trains XGBoost model with Bayesian hyperparameter optimization using Optuna
"""
import random
import pandas as pd
import numpy as np
import sys
import optuna
import xgboost as xgb
from pathlib import Path
from datetime import datetime
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.eval_utils import calculate_statistical_metrics
from utils_strategies.backtest_utils import backtest_strategy
from utils.data_utils import load_feature_data,  prepare_time_series_data
from utils.model_utils import objective, get_best_model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

from utils.config_utils import load_config
import numpy as np

config = load_config()

# Turn off verbose trial logs
optuna.logging.set_verbosity(optuna.logging.ERROR)

def data_preparation(target_isClose,features_filepath=''):
    time_config = config['time_periods']
     # Load feature-engineered data
    df = load_feature_data(features_filepath)
    # Prepare time series data
    data_dict= prepare_time_series_data(
        df, time_config
    )
    X_train, y_train, X_val, y_val, X_test, y_test=data_dict['X_train'],data_dict['y_train'],data_dict['X_val'],data_dict['y_val'],data_dict['X_test'],data_dict['y_test']
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_sequences(X, y, timesteps=10):
    Xs, ys, idxs = [], [], []

    X_values = X.values

    # --- FIX HERE ---
    # y can be a Pandas Series OR a NumPy array
    if hasattr(y, "values"):
        y_values = y.values        # Pandas Series
    else:
        y_values = y               # NumPy array
        
    index_values = X.index  # ALWAYS Pandas index
    for i in range(len(X) - timesteps):
        Xs.append(X_values[i : i + timesteps])
        ys.append(y_values[i + timesteps])
        idxs.append(index_values[i + timesteps])

    return np.array(Xs), np.array(ys), idxs

def model_predict_and_backtest(GLOBAL_SEED,baseline_modelName ='baseline_XGBM',model_cls='',target_isClose=False,features_filepath='',Long_allowed=True,Short_allowed=False,write_OHLCV_output=False):
    import numpy as np
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)

    X_train, y_train, X_val, y_val, X_test, y_test = data_preparation(target_isClose,features_filepath)
    timesteps = 10

    list_results=[]
    list_periods = ['training','validating','testing']
    
    for period in list_periods:
        ### prepare data #############################
        if period=='training':
            df_OHLCV=X_train.copy()
            signals_idx = df_OHLCV.index
            true_values= y_train.copy()
            
        elif period=='validating':
            df_OHLCV=X_val.copy()
            signals_idx = df_OHLCV.index
            true_values= y_val.copy()
            
        elif period=='testing':
            df_OHLCV=X_test.copy()
            signals_idx = df_OHLCV.index
            true_values= y_test.copy()

        ### predict ##################################
        if baseline_modelName =='baseline_XGBM':
            predictions=model_cls.predict(xgb.DMatrix(df_OHLCV,label=true_values))
        elif baseline_modelName == 'baseline_LGBM':
            predictions=model_cls.predict(df_OHLCV)
        elif baseline_modelName == 'baseline_LSTM':
            # Unpack model + scalers
            model = model_cls["model"]
            scaler_X = model_cls["scaler_X"]
            scaler_y = model_cls["scaler_y"]
            timesteps = model_cls["timesteps"]
            # 1) Scale X for this period
            df_X_scaled = pd.DataFrame(
                scaler_X.transform(df_OHLCV),
                index=df_OHLCV.index,
                columns=df_OHLCV.columns)

            X_lstm, y_lstm, idx_lstm = create_sequences(df_X_scaled, true_values, timesteps)
            # 4) Predict in scaled space
            preds_scaled = model.predict(X_lstm).ravel()

            # 5) Inverse-transform Close predictions if needed
            if scaler_y is not None:
                predictions = scaler_y.inverse_transform(
                    preds_scaled.reshape(-1,1)
                ).ravel()
            else:
                predictions = preds_scaled

            # 6) Set correct indices
            signals_idx = idx_lstm
            true_values = y_lstm  # now correct length

        elif baseline_modelName =='baseline_BuyAndHold':
            # naive persistence baseline
            if target_isClose:
                predictions = np.repeat(max(true_values), len(true_values))
            else:
                predictions = np.zeros(len(true_values))   # predicts 0 return each day

        elif baseline_modelName == 'baseline_ARIMA':
            from pmdarima import auto_arima
            
            model_cls = auto_arima
            model = model_cls(
                    y_train.values,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    random_state=GLOBAL_SEED
                )
            # Forecast val+test as one sequence
            total_periods = len(y_val) + len(y_test)
            all_forecasts = model.predict(n_periods=total_periods)

            if period == 'training':
                print(f'started {period} ARIMA ...')
                predictions = model.predict_in_sample()
                
            elif period == 'validating':
                print(f'started {period} ARIMA ...')
                predictions = all_forecasts[:len(y_val)]
                
            else:
                print(f'started {period} ARIMA ...')
                predictions = all_forecasts[len(y_val):]
                  
        predictions = np.array(predictions)
        predictions = np.ravel(predictions)
        #### backtesting #################################################
        df_OHLCV = df_OHLCV.loc[signals_idx] # correct the index of df_OHLCV vs. signals
        df_OHLCV_backtest = df_OHLCV.copy()
        if not target_isClose:
            # df['target'] = np.log(df['Close'].shift(-2) / df['Close'].shift(-1))*100
            predicted_pct_return = np.exp(np.array(predictions)/100)-1 
            #genrate signals
            signals = pd.Series(0, index=signals_idx)

            if Long_allowed:
                signals[predicted_pct_return > 0]= 1
            if Short_allowed:
                signals[predicted_pct_return < 0]= -1
            df_OHLCV_backtest['predicted_pct_return'] = predicted_pct_return
            predicted_close=None
        else:
            predicted_close = pd.Series(predictions,index=signals_idx)
            signals=None

        if baseline_modelName =='baseline_BuyAndHold':
            signals = pd.Series(np.repeat(1,len(predictions)),index=signals_idx) # buy only
            predicted_close = None # backtesting signal=1 only, no price prediction
        
        # Run backtesting
        backtester=backtest_strategy(pct_transaction_fee=0.0)
        result_dict = backtester.run_backtest2_SL_TP(
            Long_allowed=Long_allowed,
            Short_allowed=Short_allowed,
            df_OHLCV=df_OHLCV_backtest,
            signals=signals,
            predicted_close=predicted_close,
            strategy_name=baseline_modelName,
            strategy_type='moc',
            execution_threshold=0.0,
            risk_free_rate=0.0,
            write_OHLCV_output=write_OHLCV_output
        )
        statistical_metrics = calculate_statistical_metrics(true_values, np.array(predictions))
        list_results.append({
                            'GLOBAL_SEED':GLOBAL_SEED,
                            'baseline_modelName':baseline_modelName,
                            'target_isClose':target_isClose,
                            'period':period,
                            **statistical_metrics,
                            **result_dict['metrics'],
                            'at':str(datetime.now())
                            })
    return  list_results

def run1_XGBM(GLOBAL_SEED,optuna_search=False,target_isClose=False,features_filepath='',Long_allowed=True,Short_allowed=False,write_OHLCV_output=False):
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    """Main function for model training"""
    models_dir = project_root / "models" / "trained"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_config = config['model']['xgboost']
    

    X_train, y_train, X_val, y_val, X_test, y_test = data_preparation(target_isClose,features_filepath)
      
    
    STUDY_NAME = f"XGBM_optuna_study_{'Close' if target_isClose else 'LogReturn'}{GLOBAL_SEED}"
    STORAGE_DIR = project_root / "models" / Path("optuna_results")
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    STORAGE_PATH = STORAGE_DIR / f"{STUDY_NAME}.db"
    STORAGE_URL = f"sqlite:///{STORAGE_PATH}"
        
    # ------------------------------------------------------------------
    # Sampler
    # ------------------------------------------------------------------
    sampler = optuna.samplers.TPESampler(
        seed=GLOBAL_SEED
    )
    # ------------------------------------------------------------------
    # Study Creation or Reload
    # ------------------------------------------------------------------
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="minimize",
        sampler=sampler,
        storage=STORAGE_URL,
        load_if_exists=True
    )
    import os
    if optuna_search or not os.path.exists(STORAGE_PATH):
        print(f"optuna_search = {optuna_search} | STORAGE_PATH exists ={os.path.exists(STORAGE_PATH)} | Run Optuna study")
        print(f"Number of existing trials: {len(study.trials)}")
        
        print('run_train_val_model: optuna optimizing ...')
        verbose=False
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val, model_config, GLOBAL_SEED,verbose),
            n_trials = model_config['n_trials'],
            n_jobs=1, ## must be 1 to avoid racing condition when using SQLite
            show_progress_bar=True
        )
    else:
        print('Skip this Optuna search, reload existing optimized XGBM...')

    # Get best model
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    best_model = get_best_model(study.best_params, X_train, y_train, X_val, y_val, GLOBAL_SEED)
    
    # Save model
    model_path = models_dir / f"xgboost_model_{'Close' if target_isClose else 'LogReturn'}{GLOBAL_SEED}.json"
    best_model.save_model(str(model_path))
    print(f"Model saved to {model_path}")
    
    # Save study results
    study_df = study.trials_dataframe()
    study_path = models_dir / f"optuna_study_{'Close' if target_isClose else 'LogReturn'}{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    study_df.to_csv(study_path, index=False)
    print(f"Optuna study results saved to {study_path}")
    print("Model training completed successfully")
    ## train, eval and backtest the model
    list_results = model_predict_and_backtest(GLOBAL_SEED,'baseline_XGBM', best_model,target_isClose,features_filepath,Long_allowed,Short_allowed,write_OHLCV_output)
    
    return list_results

def run2_ARIMA(GLOBAL_SEED, target_isClose=False,features_filepath='', Long_allowed=True, Short_allowed=False, write_OHLCV_output=False):
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    
    print("run2_ARIMA")
    list_results = model_predict_and_backtest(GLOBAL_SEED,'baseline_ARIMA', 'auto_arima',target_isClose,features_filepath,Long_allowed,Short_allowed,write_OHLCV_output)
    return list_results

def run3_LSTM(GLOBAL_SEED, target_isClose=False,features_filepath='', Long_allowed=True, Short_allowed=False, write_OHLCV_output=False):
    # Deep Learning (for LSTM)
    import tensorflow as tf
    from tensorflow.keras.models import Sequential # type: ignore
    from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping # type: ignore

    tf.random.set_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    print("run3_LSTM ..." )
    X_train, y_train, X_val, y_val, X_test, y_test  = data_preparation(target_isClose,features_filepath)
    # Reshape for LSTM: (samples, timesteps=1, features)
    timesteps = 10
    n_features=X_train.shape[1]
    X_train_lstm, y_train_lstm, idx= create_sequences(X_train, y_train, timesteps)

    # Build model
    model = Sequential([
        LSTM(64, input_shape=(timesteps, n_features)),
        Dropout(0.2),
        Dense(1)
    ])
    print('started training LSTM')
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_lstm, y_train_lstm, epochs=100 , batch_size=32,
            verbose=0)
    print('completed training LSTM')
## train, eval and backtest the model
    list_results = model_predict_and_backtest(GLOBAL_SEED,'baseline_LSTM', model,target_isClose,features_filepath,Long_allowed,Short_allowed,write_OHLCV_output)
    
    return list_results

def run3_LSTM_scaled(GLOBAL_SEED, target_isClose=False,features_filepath='',
              Long_allowed=True,Short_allowed=False,write_OHLCV_output=False):

    import tensorflow as tf
    from tensorflow.keras.models import Sequential # type: ignore
    from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping # type: ignore
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    tf.random.set_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)

    print("run3_LSTM ..." )
    X_train, y_train, X_val, y_val, X_test, y_test  = data_preparation(target_isClose,features_filepath)

    # ---------- SCALE X ----------
    scaler_X = StandardScaler()
    X_train_s = pd.DataFrame(scaler_X.fit_transform(X_train), index=X_train.index, columns=X_train.columns)

    # ---------- SCALE y (ONLY for Close) ----------
    if target_isClose:
        scaler_y = MinMaxScaler()
        y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1,1)).ravel()
    else:
        scaler_y = None
        y_train_s = y_train.values  # log returns -> no scaling needed

    # ---------- CREATE SEQUENCES FOR TRAIN ----------
    timesteps = 10
    X_train_lstm, y_train_lstm, _ = create_sequences(X_train_s, y_train_s, timesteps)
    n_features = X_train_lstm.shape[2]

    # ---------- BUILD & TRAIN MODEL ----------
    model = Sequential([
        LSTM(64, input_shape=(timesteps, n_features)),
        Dropout(0.2),
        Dense(1)
    ])
    print('started training LSTM')
    model.compile(optimizer='adam', loss='mse')
    model.fit(
        X_train_lstm, y_train_lstm,
        epochs=80, batch_size=32,
        verbose=0
    )
    print('completed training LSTM')

    # ---------- PACKAGE MODEL + SCALERS ----------
    model_pack = {
        "model": model,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "timesteps": timesteps,
    }

    # Use the common evaluation + backtest pipeline
    list_results = model_predict_and_backtest(
        GLOBAL_SEED,
        'baseline_LSTM',
        model_pack,          # <-- pass dict, not raw model
        target_isClose,
        features_filepath,
        Long_allowed,
        Short_allowed,
        write_OHLCV_output
    )
    return list_results

def run4_LGBM(GLOBAL_SEED, target_isClose=False,features_filepath='', Long_allowed=True, Short_allowed=False, write_OHLCV_output=False):
    import lightgbm as lgb 
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    print("run4_LightGBM ..." )
    # Use fixed params from config (no Optuna)
    lgb_params = config.get('model', {}).get('lightgbm_static', {
        'n_estimators': 800,
        'learning_rate': 0.03,
        'num_leaves': 63,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': GLOBAL_SEED
    })
    X_train, y_train, X_val, y_val, X_test, y_test  = data_preparation(target_isClose,features_filepath)
    print('started training LGBM')
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train)
    print('completed training LGBM')
## train, eval and backtest the model
    list_results = model_predict_and_backtest(GLOBAL_SEED,'baseline_LGBM', model,target_isClose,features_filepath,Long_allowed,Short_allowed,write_OHLCV_output)
    return list_results

def run5_BuyAndHold(GLOBAL_SEED, target_isClose=False,features_filepath='', Long_allowed=True, Short_allowed=False, write_OHLCV_output=False):
    ## train, eval and backtest the model
    print("run5_BuyAndHold ..." )
    list_results = model_predict_and_backtest(GLOBAL_SEED,'baseline_BuyAndHold', '',target_isClose,features_filepath,Long_allowed,Short_allowed,write_OHLCV_output)
    return list_results
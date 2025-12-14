"""
Feature Engineering Utilities
Functions for creating autoregressive features from OHLCV data
"""
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import MinimalFCParameters, EfficientFCParameters
import numpy as np, pandas as pd
import logging, shap,sys,pathlib,os

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))

def create_Log_Return(df,list_Price:list=['Close','Open','High','Low'], target_isClose=False): #
    # Compute log returns (more robust representation)
    if len(list_Price) > 0 and 'Close' not in list_Price:
        list_Price.insert(0,'Close')  

    if not target_isClose:
        pwr=3
        for p in list_Price:
            if p and 'Log_Return' not in p:
                name= 'Log_Return' if p == "Close" else f'Log_{p}_Return'
                df[name] = (df[p] - df[p].shift(1))**pwr # Log_Return
            
                df[name+'_change1'] = df[name]-df[name].shift(1)   #Log_Return_change1
                df[name+'_change2'] = df[name+'_change1'] - df[name+'_change1'].shift(1) #Log_Return_change2
                df[name+'_change3'] = df[name+'_change2'] - df[name+'_change2'].shift(1) #Log_Return_change3
                # # ## their interactions
                df[name+'12'] = df[name+'_change1']*df[name+'_change2']
                df[name+'13'] = df[name+'_change1']*df[name+'_change3']
                df[name+'23'] = df[name+'_change2']*df[name+'_change3']
                df[name+'123'] = df[name+'_change1']*df[name+'_change2']*df[name+'_change3']
    else:
        print("list_Price:",list_Price)
        for p in list_Price:
            if p and 'Log_Return' not in p:
                name= 'Log_Return' if p == "Close" else f'Log_{p}_Return'
                df[name] = np.log(df[p] / df[p].shift(1)) *100# Log_Return
                df[name+'_change1'] = df[name]-df[name].shift(1)   #Log_Return_change1
                df[name+'_change2'] = df[name+'_change1'] - df[name+'_change1'].shift(1) #Log_Return_change2
                df[name+'_change3'] = df[name+'_change2'] - df[name+'_change2'].shift(1) #Log_Return_change3
                # # ## their interactions
                df[name+'12'] = df[name+'_change1']*df[name+'_change2']
                df[name+'13'] = df[name+'_change1']*df[name+'_change3']
                df[name+'23'] = df[name+'_change2']*df[name+'_change3']
                df[name+'123'] = df[name+'_change1']*df[name+'_change2']*df[name+'_change3']

    df = df.dropna()
    return df

def create_date_features(df, 
                      cyclical_encoding=True,
                      add_weekend=False,
                      add_quarter=True,
                      country='US'):
   
    data = df.copy()
    
    # Validate index type
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Index must be a pandas DatetimeIndex.")
    
    # Extract basic date components
    data['day_of_week'] = data.index.dayofweek  # 0=Mon, 6=Sun
    data['day_of_month'] = data.index.day
    data['day_of_year'] = data.index.dayofyear
    data['month'] = data.index.month
    if add_quarter:
        data['quarter'] = data.index.quarter
    data['year'] = data.index.year
    
    # Binary flag: weekend
    if add_weekend:
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)  # Sat=5, Sun=6
    
    # Cyclical encoding: preserves continuity (e.g., Dec ~ Jan)
    if cyclical_encoding:
        # Day of week (7-day cycle)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        # Month (12-month cycle)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # Day of year (365-day cycle) - useful for annual seasonality
        data['day_of_year_norm'] = data['day_of_year'] / 365.25
        data['dayy_sin'] = np.sin(2 * np.pi * data['day_of_year_norm'])
        data['dayy_cos'] = np.cos(2 * np.pi * data['day_of_year_norm'])

    # Optional: Holiday flag (requires `holidays` package)
    try:
        import holidays
        years_list = data['year'].unique()
        country_holidays = holidays.CountryHoliday(country, years=years_list)
        data['is_holiday'] = np.isin(data.index.date,country_holidays.keys()).astype(int)
    except ImportError:
        print("Optional package 'holidays' not installed. Skipping holiday detection. "
              "Install with: pip install holidays")
        data['is_holiday'] = 0  # or skip this column

    return data

def create_lag_features(df,column='Close', windows=[5, 10, 20, 30, 60, 90, 120, 180, 250]): #, 120, 180, 250

    lags = [1, 2, 3] + windows
    print(f"Creating {len(lags)} lag features: {lags}")
    
    df = df.copy()
    # Create lag features for closing price
    for lag in lags:
        df[f'{column}_Lag_{lag}'] = df[column].shift(lag)
        # df[f'{column}_LogReturn_{lag}'] = np.log(df[column]/df[column].shift(lag))*100 # _LogReturn_Lag1, _LogReturn_Lag1, etc.
    return df

def create_rolling_features(df:pd.DataFrame,column='Close', windows= [5, 10, 20, 30, 60, 90, 180, 250]): #, 120, 180, 250
   
    print(f"Creating rolling features for windows: {windows}")
    
    df = df.copy()
    # Create rolling statistics
    for window in windows:
        df[f'{column}_Roll_Mean_{window}'] = df[column].rolling(window=window).mean()
        df[f'{column}_Roll_Std_{window}'] = df[column].rolling(window=window).std()
        df[f'{column}_Roll_Min_{window}'] = df[column].rolling(window=window).min()
        df[f'{column}_Roll_Max_{window}'] = df[column].rolling(window=window).max()

    return df
import hashlib

def create_tsfresh_features(
    df,
    target_isClose=False,
    columns=None,
    window_size=30,
    stride=1,
    feature_set='efficient',
    impute_nans=True,
    select_significant=False,
    target_column=None,
    cache_dir = '.././data/features',
    n_jobs=4,
    recompute = False
):
    # 🔴 ONLY ADDING CACHING BELOW - NO OTHER CHANGES
    
    os.makedirs(cache_dir, exist_ok=True)
    # Generate unique cache name from ALL parameters
    ts_col_str = "_".join(sorted(columns)) if columns else "all_numeric"
    cache_str = (
        f"{df.shape[0]}rows_{df.shape[1]}cols_IDmin{str(df.index.min())}_IDmax{str(df.index.max())}"
        f"_target{'Close' if target_isClose else 'LogReturn'}_label{target_column}"
        f"_tscols_{ts_col_str}_ws{window_size}_s{stride}_fs{feature_set}_imn{impute_nans}_sig{select_significant}"
    )
    print("TSfresh_cache_str:", cache_str)
    cache_name = hashlib.md5(cache_str.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_name}.parquet")

    # Check if cached version exists
    if not recompute and os.path.exists(cache_path):
        print(f"✅ recompute = {recompute} | Loading cached features")
        return pd.read_parquet(cache_path)
        
    # 🔴 YOUR ORIGINAL SCRIPT STARTS HERE (NO CHANGES)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")

    if columns is None or columns == []:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Validate columns
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    settings_map = {
        'minimal': MinimalFCParameters(),
        'efficient': EfficientFCParameters(),
        'comprehensive': None
    }
    if feature_set not in settings_map:
        raise ValueError("Invalid feature_set")
    settings = settings_map[feature_set]

    feature_dfs = []

    for col in columns:
        if col:
            print(f"🔄 Processing '{col}' with {window_size}-day windows...")

            series = df[col].dropna()
            if len(series) < window_size:
                print(f"⚠️ Not enough data in '{col}' ({len(series)} < {window_size})")
                continue

            windows = []
            for start in range(0, len(series) - window_size + 1, stride):
                end_idx = start + window_size
                window_data = series.iloc[start:end_idx].values
                window_id = series.index[end_idx - 1]

                for point_idx, value in enumerate(window_data):
                    windows.append({
                        'id': window_id,
                        'time': point_idx,
                        'value': value,
                        'kind': col
                    })

            if not windows:
                print(f"⚠️ No windows created for '{col}'")
                continue

            df_long = pd.DataFrame(windows)

            # Critical: Validate required columns exist and not NaN
            for col_name in ['id', 'time', 'value']:
                if col_name not in df_long.columns:
                    raise ValueError(f"Missing column: {col_name}")
                if df_long[col_name].isnull().any():
                    raise ValueError(f"Column '{col_name}' contains NaN")

            # Extract features
            try:
                X = extract_features(
                    df_long,
                    column_id='id',
                    column_sort='time',
                    column_value='value',
                    default_fc_parameters=settings,
                    n_jobs=n_jobs,
                    impute_function=impute if impute_nans else None
                )
            except Exception as e:
                print(f"❌ Failed to extract features for '{col}': {e}")
                continue

            # Rename to avoid conflicts
            X.columns = [f"{col}__{feat}" for feat in X.columns]
            feature_dfs.append(X)

    if not feature_dfs:
        raise ValueError("No features were extracted. Check input data.")

    X_full = pd.concat(feature_dfs, axis=1).sort_index()

    # Optional: feature selection
    if select_significant and target_column is not None:
        if target_column not in df:
            raise ValueError(f"Target column '{target_column}' not found")
        y = df[target_column].reindex(X_full.index)
        X_full = select_features(X_full, y)
        print(f"✅ Selected {X_full.shape[1]} significant features")

    print(f"✅ Feature extraction complete. Shape: {X_full.shape}")
    df = pd.concat([df, X_full], axis=1)
    
    # Save to cache
    df.to_parquet(cache_path)
    print(f"💾 Saved to cache: {cache_path}")
    return df

def create_tsfresh_features2(
    df=pd.DataFrame,
    columns=None,
    window_size=30,
    stride=1,
    feature_set='efficient',
    impute_nans=True,
    select_significant=False,
    target_column=None,
    n_jobs=4
):
    import os
    str_columns=str(columns).replace(",","_").replace("[","").replace("]","")
    file_path=str(project_root)+f'/data/features/tsfresh_{str_columns}.csv'
    if os.path.exists(file_path):
        df_preload = pd.read_csv(file_path,parse_dates=True,index_col=0)
        if (df.shape[0] == df_preload.shape[0]) and df.index.equals(df_preload.index) \
            and df[['target']].equals(df_preload[['target']]) and df.iloc[:,:df.shape[1]].columns.equals(df_preload.iloc[:,:df.shape[1]].columns):
            print("pre_load features are available. Loading ... !")
            return df_preload
            
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")

    if columns is None or columns == []:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Validate columns
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    settings_map = {
        'minimal': MinimalFCParameters(),
        'efficient': EfficientFCParameters(),
        'comprehensive': None
    }
    if feature_set not in settings_map:
        raise ValueError("Invalid feature_set")
    settings = settings_map[feature_set]

    feature_dfs = []

    for col in columns:
        if col:
            print(f"🔄 Processing '{col}' with {window_size}-day windows...")

            series = df[col].dropna()
            if len(series) < window_size:
                print(f"⚠️ Not enough data in '{col}' ({len(series)} < {window_size})")
                continue

            windows = []
            for start in range(0, len(series) - window_size + 1, stride):
                end_idx = start + window_size
                window_data = series.iloc[start:end_idx].values
                window_id = series.index[end_idx - 1]

                for point_idx, value in enumerate(window_data):
                    windows.append({
                        'id': window_id,
                        'time': point_idx,
                        'value': value,
                        'kind': col  # optional: use kind for advanced settings
                    })

            if not windows:
                print(f"⚠️ No windows created for '{col}'")
                continue

            df_long = pd.DataFrame(windows)

            # 🔴 Critical: Validate required columns exist and not NaN
            for col_name in ['id', 'time', 'value']:
                if col_name not in df_long.columns:
                    raise ValueError(f"Missing column: {col_name}")
                if df_long[col_name].isnull().any():
                    raise ValueError(f"Column '{col_name}' contains NaN")

            # Extract features
            try:
                X = extract_features(
                    df_long,
                    column_id='id',
                    column_sort='time',
                    column_value='value',
                    default_fc_parameters=settings,
                    n_jobs=n_jobs,
                    impute_function=impute if impute_nans else None
                )
            except Exception as e:
                print(f"❌ Failed to extract features for '{col}': {e}")
                continue

            # Rename to avoid conflicts
            X.columns = [f"{col}__{feat}" for feat in X.columns]
            feature_dfs.append(X)

    if not feature_dfs:
        raise ValueError("No features were extracted. Check input data.")

    X_full = pd.concat(feature_dfs, axis=1).sort_index()

    # Optional: feature selection
    if select_significant and target_column is not None:
        if target_column not in df:
            raise ValueError(f"Target column '{target_column}' not found")
        y = df[target_column].reindex(X_full.index)
        X_full = select_features(X_full, y)
        print(f"✅ Selected {X_full.shape[1]} significant features")

    print(f"✅ Feature extraction complete. Shape: {X_full.shape}")
    df=pd.concat([df,X_full],axis=1)
    df.to_csv(file_path)
    return df  

def create_target_variable(df:pd.DataFrame,target_isClose=False, horizon=1)->pd.DataFrame:
    print("Creating target variable (next day's closing price)")
    df = df.copy()
    print('create_target_variable target_isClose:',target_isClose)
    if target_isClose:
        df['target'] = df['Close'].shift(-horizon-1)
        df.dropna(inplace=True)  # Drop last row with NaN target
        return df
    else:
        df['target'] = np.log(df['Close'].shift(-horizon-1)/df['Close'].shift(-1))*100
        df.dropna(inplace=True)  # Drop last row with NaN target
        return df

def normalize_features(df, windows=[30]):
    
    print(f"Applying rolling Min-Max normalization with windows: {windows}")
    
    df = df.copy()
    
    # Identify feature columns (excluding target and raw price data)
    feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close'
                                                             , 'target']]
    
    for window in windows:
        for col in feature_cols:
            # Calculate rolling min and max
            rolling_min = df[col].rolling(window=window).min()
            rolling_max = df[col].rolling(window=window).max()
            
            # Apply Min-Max scaling
            range_val = rolling_max - rolling_min
            # Avoid division by zero
            range_val = np.where(range_val == 0, 1e-8, range_val)
            
            df[f'{col}_norm_{window}'] = (df[col] - rolling_min) / range_val
    
    # Drop original features (keep only normalized versions)
    df = df.drop(columns=feature_cols)
    
    return df

def handle_outliers(df, columns=None, factor=1.5):
   
    print(f"Handling outliers using IQR method (factor={factor})")
    
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in ['Open', 'High', 'Low', 'Close']:  # Skip raw price data
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # Replace outliers with boundary values
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    
    return df

def compute_rolling_correlations(X, window_size=90, threshold=0.9):
   
    print(f"Computing rolling correlations with window size={window_size} days")
    
    # Dictionary to track when features become highly correlated
    flagged_features = {}
    
    # Process in rolling windows
    for i in range(window_size, len(X), 30):  # Slide by 1 month
        window_start = i - window_size
        window_end = i
        X_window = X.iloc[window_start:window_end]
        
        # Compute correlation matrix
        corr_matrix = X_window.corr().abs()
        
        # Identify highly correlated feature pairs
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = []
        for column in upper.columns:
            if any(upper[column] > threshold):
                to_drop.append(column)
        
        # Record flagged features with timestamp
        for feature in to_drop:
            if feature not in flagged_features:
                flagged_features[feature] = []
            flagged_features[feature].append([str(X.index[window_end]), float(upper[upper[feature] > threshold][feature].max())])
    
    return flagged_features

def handle_feature_redundancy(X, y, model, window_size=90, correlation_threshold=0.9, shap_threshold=0.2):
    # 1. Correlation Analysis (rolling window)
    flagged_by_correlation = compute_rolling_correlations(X, window_size, correlation_threshold)
    
    # 2. SHAP-Based Pruning
    # First, get SHAP values from the model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Calculate mean absolute SHAP values
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    feature_importance = pd.Series(mean_shap, index=X.columns)
    
    # Identify bottom 20% features by importance
    shap_cutoff = feature_importance.quantile(shap_threshold)
    flagged_by_shap = feature_importance[feature_importance <= shap_cutoff].index.tolist()
    
    # 3. Combine results (features flagged by BOTH methods get highest priority)
    high_priority_flags = [f for f in flagged_by_correlation.keys() 
                          if f in flagged_by_shap and len(flagged_by_correlation[f]) > 5]
    
    medium_priority_flags = [f for f in flagged_by_correlation.keys() 
                            if len(flagged_by_correlation[f]) > 10]
    
    shap_only_flags = [f for f in flagged_by_shap if f not in flagged_by_correlation]
    
    # 4. Generate removal recommendations (not automatic removal)
    removal_recommendations = {
        'high_priority': high_priority_flags,
        'medium_priority': medium_priority_flags,
        'shap_only': shap_only_flags,
        'correlation_details': flagged_by_correlation
    }
    
    print(f"Feature redundancy analysis complete:")
    print(f"  - High priority removal candidates: {len(high_priority_flags)}")
    print(f"  - Medium priority removal candidates: {len(medium_priority_flags)}")
    print(f"  - SHAP-only removal candidates: {len(shap_only_flags)}")
    
    return removal_recommendations

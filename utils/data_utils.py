"""
Data Utilities
Functions for loading and processing data
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging, datetime, json, sys,os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def fetch_yahoo_data(symbol, start, end):
    """Fetch data from Yahoo Finance"""
    print(f"Fetching data for {symbol} from Yahoo Finance ({start} to {end})")
    df = yf.download(symbol, start=start, end=end).reset_index()
    print(df.head().to_string())
    return df

def fetch_alpha_vantage_data(symbol, start, end, api_key=None):
    """Fetch data from Alpha Vantage (placeholder - needs API key)"""
    # In a real implementation, this would use the Alpha Vantage API
    print(f"Fetching data for {symbol} from Alpha Vantage ({start} to {end})")
    # This is a placeholder - would need actual API implementation
    raise NotImplementedError("Alpha Vantage implementation requires API key")

def load_raw_data(file_path):
    """Load raw data from CSV"""
    print(f"Loading raw data from {file_path}")
    df = pd.read_csv(file_path, parse_dates=True)
    df = df.iloc[1:,1:] # skip 1st row and 1st col
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date',inplace=True)
    print('load_raw_data \n', df.head().to_string())
    return df

def load_raw_data_mt5(symbol="XAUUSD",filepath=r"G:\My Drive\1python_test\customizedPredictor\input\price_mt5"):
    """Load raw data from CSV"""
    file_path = os.path.join(filepath, symbol + ".csv")
    print(f"Loading raw data from {file_path}")
    df = pd.read_csv(file_path, parse_dates=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date',inplace=True)
    print('load_raw_data_mt5 \n', df.head().to_string())
    return df

def identify_and_log_data_errors(df, output_dir=None):
    """Identify potential data errors and create validation report"""
    print("Identifying potential data errors for manual validation")
    
    # Calculate IQR bounds (stricter threshold for data errors)
    Q1 = df['Close'].quantile(0.25)
    Q3 = df['Close'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR  # Stricter threshold for data errors
    upper_bound = Q3 + 3 * IQR
    
    # Flag potential data errors
    is_potential_error = (df['Close'] < lower_bound) | (df['Close'] > upper_bound)
    potential_errors = df[is_potential_error].copy()
    
    # Create output directory if needed
    if output_dir is None:
        output_dir = project_root / "data" / "error_validation"
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate detailed error report with context
    if not potential_errors.empty:
        # Add contextual information for manual validation
        potential_errors['Value'] = potential_errors['Close']
        potential_errors['IQR_Multiplier'] = (potential_errors['Close'] - Q1) / IQR
        potential_errors['Prev_Day_Return'] = np.log(potential_errors['Close'] / potential_errors['Close'].shift(1))
        potential_errors['Next_Day_Return'] = np.log(potential_errors['Close'].shift(-1) / potential_errors['Close'])
        
        # Flag known crisis periods for context
        crisis_dates = pd.DatetimeIndex([
            '2008-09-15', '2008-09-16', '2008-09-17',  # Lehman collapse
            '2020-03-16', '2020-03-17', '2020-03-18',  # COVID crash
            '2022-02-24', '2022-02-25', '2022-02-26',  # Ukraine invasion
            '2023-03-10', '2023-03-11', '2023-03-12'   # Banking crisis
        ])
        
        potential_errors['Is_Crisis_Period'] = potential_errors.index.isin(crisis_dates)
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"data_error_validation_{timestamp}.csv"
        potential_errors.to_csv(report_path)
        
        print(f"Identified {len(potential_errors)} potential data errors")
        print(f"Detailed validation report saved to {report_path}")
        
        # Create summary for quick review
        summary = {
            "total_potential_errors": len(potential_errors),
            "outside_upper_bound": int(sum(potential_errors['Close'] > upper_bound)),
            "outside_lower_bound": int(sum(potential_errors['Close'] < lower_bound)),
            "in_crisis_periods": int(sum(potential_errors['Is_Crisis_Period'])),
            "requires_manual_validation": int(len(potential_errors) - sum(potential_errors['Is_Crisis_Period']))
        }
        
        summary_path = output_dir / f"data_error_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Error summary saved to {summary_path}")
        
        return is_potential_error, report_path
    else:
        print("No potential data errors identified")
        return is_potential_error, None

# def transform_to_price_differences(df):
#     # Define the price columns
#     price_cols = ['Open', 'High', 'Low', 'Close']
    
#     # Create difference columns
#     diff_df = df[price_cols].diff()  # This computes current - previous
#     diff_df = diff_df[price_cols].apply(lambda x: x + 0.00001)  # avoid zero values
#     result_df = diff_df.dropna() # Drop the first row with NaNs
#     return result_df

def clean_data(df,valid_columns=['Close','High','Low','Open', 'Volume']):
    """Clean and preprocess raw data with proper data error handling"""
    print("Cleaning data")
    # drop NA values
    df = df.dropna()
    df.index.rename('Date',inplace = True)
    # if 'Tick_volume' in df.columns: 
    #     df['Volume'] = df['Tick_volume'] # in case of mt5 data
    df = df[valid_columns]
    df[valid_columns] = df[valid_columns].astype('float')
    # Ensure continuous date index (fill weekends/holidays)

    # Forward-fill missing values (including weekends)
    # df = df.asfreq(freq='D',method='ffill')
    is_potential_error, error_report_path = identify_and_log_data_errors(df)
    
    # Create manual validation file if needed
    if error_report_path:
        print("Manual validation required for potential data errors")
        print("Please review the error validation report before proceeding")
        
        # Create instruction file for manual validation
        instruction_path = error_report_path.parent / "MANUAL_VALIDATION_INSTRUCTIONS.txt"
        with open(instruction_path, 'w') as f:
            f.write("DATA ERROR VALIDATION INSTRUCTIONS\n")
            f.write("="*50 + "\n\n")
            f.write("1. Review 'data_error_validation_*.csv' for potential data errors\n")
            f.write("2. For each entry, determine if it's a true data error or legitimate market movement:\n")
            f.write("   - TRUE DATA ERROR: Value appears incorrect (e.g., typo, data corruption)\n")
            f.write("   - LEGITIMATE MOVEMENT: Occurs during known crisis periods or has news context\n")
            f.write("3. Update the 'is_valid' column in the CSV:\n")
            f.write("   - 1 = Valid (keep the data point)\n")
            f.write("   - 0 = Invalid (remove the data point)\n")
            f.write("4. Save the updated CSV and restart the preprocessing step\n\n")
            f.write("IMPORTANT: Only remove data points confirmed as errors!\n")
            f.write("Gold markets can have legitimate extreme movements during crises.\n")
        
        # Check if manual validation has been completed
        validation_complete = False
        for file in error_report_path.parent.glob("data_error_validation_*.validated.csv"):
            if file.exists():
                print(f"Found completed validation file: {file}")
                validated_errors = pd.read_csv(file, index_col=0, parse_dates=True)
                
                # Only remove confirmed data errors
                is_data_error = validated_errors['is_valid'] == 0
                df = df[~is_data_error]
                validation_complete = True
                break
        
        if not validation_complete:
            print("Manual validation required. Pipeline paused for data error review.")
            print(f"Review instructions at: {instruction_path}")
            print("After validation, save the file with '.validated.csv' suffix and restart.")
            raise ValueError("Manual data validation required - see instructions in data/error_validation/")    
    return df

def load_processed_data(file_path):
    """Load processed data"""
    print(f"Loading processed data from {file_path}")
    df_OHLCV = pd.read_csv(file_path, index_col=0, parse_dates=True)
    if 'Date' in df_OHLCV.columns:
        df_OHLCV['Date'] = pd.to_datetime(df_OHLCV['Date'])
        df_OHLCV.set_index('Date',inplace=True)
    return df_OHLCV

def load_feature_data(file_path):
    """Load feature-engineered data"""
    print(f"Loading feature data from {file_path}")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df

def prepare_time_series_data(df, time_config):
    """Prepare time series data while maintaining full historical context"""
    print("Preparing time series data with historical context")
    
    # Extract target variable (next day's closing price)
    # df['target'] = df['Close'].shift(-1) # assume target is avai in df
    
    # Drop the last row (no target)
    df = df.dropna()
    
    # Split data according to time periods
    train_start = time_config['training']['start']
    train_end = time_config['training']['end']
    val_start = time_config['validation']['start']
    val_end = time_config['validation']['end']
    test_start = time_config['testing']['start']
    test_end = time_config['testing']['end']
    
    # Create chronological splits
    train = df.loc[train_start:train_end]
    val = df.loc[val_start:val_end]
    test = df.loc[test_start:test_end]
    
    # Return all datasets plus the full dataset for retraining
    return {
        'full': df,
        'train': train,
        'val': val,
        'test': test,
        'X_train': train.drop('target', axis=1),
        'y_train': train['target'],
        'X_val': val.drop('target', axis=1),
        'y_val': val['target'],
        'X_test': test.drop('target', axis=1),
        'y_test': test['target']
    }
def get_retraining_data(full_dataset, current_date, retrain_window=""):
    """
    Safely extract retraining data up to current_date using a rolling window.
    Works even if dates are missing from the index.
    """

    if not isinstance(full_dataset.index, pd.DatetimeIndex):
        raise TypeError("full_dataset index must be a DatetimeIndex")

    print(f"Preparing retraining data up to {current_date} with window {retrain_window}")

    index = full_dataset.index.sort_values()

    # Step 1: Align current_date to nearest previous available timestamp
    if current_date < index[0]:
        return full_dataset.iloc[:0]

    if current_date > index[-1]:
        current_loc = len(index) - 1
    else:
        current_loc = index.get_indexer([current_date], method="ffill")[0]

    if current_loc < 1:
        return full_dataset.iloc[:0]

    # Step 2: Determine start index
    if retrain_window:
        window_start_date = index[current_loc] - pd.Timedelta(retrain_window)
        start_loc = index.get_indexer([window_start_date], method="ffill")[0]
        start_loc = max(start_loc, 0)
    else:
        start_loc = 0

    # Step 3: Slice safely
    retrain_data = full_dataset.iloc[start_loc:current_loc]

    # Step 4: Minimum data safeguard
    MIN_SAMPLES = 50
    if len(retrain_data) < MIN_SAMPLES:
        retrain_data = full_dataset.iloc[:current_loc]

    return retrain_data
# def get_retraining_data(full_dataset, current_date, retrain_window=""):
#     """Get appropriate retraining data including historical data"""
#     # print('retrain_window = ', retrain_window)
#     print(f"Preparing retraining data up to {current_date} with {retrain_window} window")
    
#     # Calculate start date for retraining window
#     start_date = current_date - pd.Timedelta(retrain_window)
#     print(full_dataset.index)
#     # Get all data from start_date up to (but not including) current_date
#     retrain_data = full_dataset.loc[start_date:current_date].iloc[:-1]
#     # Ensure minimum data requirement
#     if len(retrain_data) < 50:  # Minimum 50 days for reliable training
#         print(f"Insufficient data for retraining ({len(retrain_data)})")
#         # Fall back to minimum window using earliest available data
#         retrain_data = full_dataset.iloc[:current_date]
    
#     return retrain_data

def generate_desc_stats_table(df):
    columns = [
        'Open', 'High', 'Low', 'Close', 'Volume']    
    stats = pd.DataFrame({
        'count':df[columns].count(),
        'min': df[columns].min(),
        'max': df[columns].max(),
        'mean': df[columns].mean(),
        'std': df[columns].std(),
        'skewness': df[columns].skew(),
        'kurtosis': df[columns].kurt()
    })
    return stats

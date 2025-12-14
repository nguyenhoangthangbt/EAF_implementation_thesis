"""
Exploratory Data Analysis (EDA) Script for Thesis
Assumptions:
    - DataFrame `df` is already loaded in the Python environment.
    - Columns must include: Open, High, Low, Close, Volume.
    - LogReturn will be computed automatically if not present.
Outputs:
    - Figures saved under results/EDA/figures
    - Tables saved under results/EDA/tables
"""
import sys,os
from pathlib import Path
project_root = os.path.abspath(Path("../."))
sys.path.append(str(project_root))
print(project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# -----------------------------------------------------------------------------
# Folder Setup
# -----------------------------------------------------------------------------
EDA_path = Path(f'{project_root}/results/EDA')
if not os.path.exists(EDA_path): 
    EDA_path.mkdir(exist_ok=True)
    (EDA_path / 'tables').mkdir(exist_ok=True)
    (EDA_path / 'figures').mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 0. Data Preparation
# -----------------------------------------------------------------------------
def prepare_dataframe(df):
    df = df.copy()

    # Ensure datetime index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
    else:
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            try:
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
            except:
                raise ValueError("DataFrame requires a datetime index or 'Date' column.")

    # Ensure OHLCV columns exist
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Generate LogReturn if missing
    if 'LogReturn' not in df.columns:
        df['LogReturn'] = np.log(df['Close']).diff()

    df = df.dropna()
    return df

# -----------------------------------------------------------------------------
# 1. Summary Statistics
# -----------------------------------------------------------------------------
def compute_summary_stats(df):
    cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'LogReturn']
    stats_df = df[cols].describe().T
    stats_df['skew'] = df[cols].skew()
    stats_df['kurtosis'] = df[cols].kurtosis()
    stats_df = stats_df.transpose()
    stats_df.to_csv(EDA_path / "tables/summary_statistics.csv")
    print(f"Summary statistics saved -> {str(EDA_path)}")
    return stats_df

# -----------------------------------------------------------------------------
# 3. Stationarity Tests (ADF, KPSS)
# -----------------------------------------------------------------------------
def run_adf(series, regression='c'):
    stat, p, lags, nobs, crit, _ = adfuller(series, regression=regression, autolag='AIC')
    return {
        "adf_stat": stat,
        "p_value": p,
        "n_lags": lags,
        "n_obs": nobs,
        "crit_1%": crit['1%'],
        "crit_5%": crit['5%'],
        "crit_10%": crit['10%']
    }

def run_kpss(series, regression='c'):
    stat, p, lags, crit = kpss(series, regression=regression, nlags="auto")
    return {
        "kpss_stat": stat,
        "p_value": p,
        "n_lags": lags,
        "crit_10%": crit['10%'],
        "crit_5%": crit['5%'],
        "crit_1%": crit['1%']
    }
def run_adf_kpss(df):    
    stationarity_results = {
        "Close_ADF": run_adf(df["Close"], regression='ct'),
        "Close_KPSS": run_kpss(df["Close"], regression='ct'),
        "LogReturn_ADF": run_adf(df["LogReturn"], regression='c'),
        "LogReturn_KPSS": run_kpss(df["LogReturn"], regression='c'),
    }

    pd.DataFrame(stationarity_results).to_csv(EDA_path / "tables/stationarity_results.csv")
    print(f"Stationarity results saved -> {str(EDA_path)} /tables/stationarity_results.csv")
    return stationarity_results
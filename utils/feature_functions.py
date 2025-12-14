"""
Feature Functions with Academic Citations
Each function includes reference to source paper/technique.
"""

import pandas as pd
import numpy as np
import pywt
from scipy.stats import entropy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ===========================================
# PREPROCESSING
# ===========================================

def preprocess_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.ffill().dropna()
    price_cols = ['Open', 'High', 'Low', 'Close','Volume']
    df[price_cols] = df[price_cols].astype(float)
    return df

# ===========================================
# CROSS FX FEATURES — Cites: Fenn et al. (2009)
# ✅ FIXED: NO MELT — WIDE FORMAT TO PRESERVE ROW COUNT
# ===========================================

def add_cross_fx_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Pivot to wide — one column per symbol
    close_wide = df.pivot(index='Date', columns='symbol', values='Close')
    
    # Generate cross features in wide format — NO MELT
    symbols = close_wide.columns.tolist()
    new_features = {}
    
    for i, sym1 in enumerate(symbols):
        for sym2 in symbols[i+1:]:
            # Cross rate if both are USD crosses
            if sym1.endswith('USD') and sym2.endswith('USD'):
                cross_name = sym1.replace('USD','') + sym2.replace('USD','')
                new_features[f'cross_{cross_name}'] = close_wide[sym1] / close_wide[sym2]
            elif sym1.startswith('USD') and sym2.startswith('USD'):
                cross_name = sym2[3:] + sym1[3:]
                new_features[f'cross_{cross_name}'] = close_wide[sym2] / close_wide[sym1]
    
    # Triangular arbitrage
    triples = [('EURUSD', 'USDJPY', 'EURJPY'), ('GBPUSD', 'USDJPY', 'GBPJPY'), ('AUDUSD', 'USDCAD', 'AUDCAD')]
    for sym1, sym2, sym3 in triples:
        if all(s in close_wide.columns for s in [sym1, sym2, sym3]):
            implied = close_wide[sym1] * close_wide[sym2]
            residual = close_wide[sym3] - implied
            new_features[f'tri_arb_{sym3}'] = residual
            new_features[f'tri_arb_z_{sym3}'] = (residual - residual.rolling(60).mean()) / residual.rolling(60).std()
    
    # Convert new_features dict to DataFrame
    if not new_features:
        return df
    
    features_df = pd.DataFrame(new_features, index=close_wide.index).reset_index()
    
    # Merge back — one row per original row
    df = df.merge(features_df, on='Date', how='left')
    
    return df

# ===========================================
# REGIME FEATURES — Cites: Ang & Bekaert (2002)
# ===========================================

def add_fx_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['squared_log_return'] = df['log_return'] ** 2
    df['ewma_variance'] = df.groupby('symbol')['squared_log_return'].transform(
        lambda x: x.ewm(span=15, min_periods=10).mean()
    )
    df['ewma_volatility'] = np.sqrt(df['ewma_variance'])
    df['vol_quantile'] = df.groupby('symbol')['ewma_volatility'].transform(
        lambda x: pd.qcut(x, q=[0, 0.33, 0.66, 1.0], labels=['Low', 'med', 'High'], duplicates='drop')
    )
    df['price_zscore_50'] = df.groupby('symbol')['Close'].transform(
        lambda x: (x - x.rolling(50).mean()) / x.rolling(50).std()
    )
    df['momentum_regime'] = pd.cut(df['price_zscore_50'], bins=[-np.inf, -1, 1, np.inf], labels=['oversold', 'neutral', 'overbought'])
    return df

# ===========================================
# LIQUIDITY FEATURES — Cites: Parkinson (1980), Garman & Klass (1980)
# ===========================================

def add_liquidity_orderflow_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    h, l, c, v = df['High'], df['Low'], df['Close'], df['Volume']
    df['parkinson_vol'] = np.sqrt((1/(4*np.log(2))) * ((np.log(h/l)**2).rolling(5).mean()))
    df['typical_price'] = (h + l + c) / 3
    df['vwap_deviation'] = (c - df['typical_price']) / df['typical_price']
    df['volume_confirmed_move'] = (df['vwap_deviation'].abs() * v).rolling(5).mean()
    df['garman_klass'] = 0.5 * np.log(h/l)**2 - (2*np.log(2)-1) * np.log(c/df['Open'])**2
    df['liq_shock'] = (df['garman_klass'] * v).rolling(10).std() / df['garman_klass'].rolling(10).mean()
    df['close_ratio'] = (c - df['Open']) / (h - l).replace(0, np.nan)
    df['smart_money_flow'] = df['close_ratio'] * v
    return df

# ===========================================
# GOLD MACRO FEATURES — Cites: Erb & Harvey (2013), World Gold Council (2023)
# ===========================================

def add_gold_macro_features(df: pd.DataFrame, macro_data: dict = None) -> pd.DataFrame:
    if macro_data is None or len(macro_data) == 0:
        return df
    df = df.copy()
    macro_df = pd.DataFrame(macro_data)
    macro_df.index = pd.to_datetime(macro_df.index)
    
    # Deduplicate index
    if macro_df.index.duplicated().any():
        macro_df = macro_df[~macro_df.index.duplicated(keep='last')]
    
    # Join using merge to preserve row count
    original_len = len(df)
    df = df.merge(macro_df, left_on='Date', right_index=True, how='left')
    if len(df) != original_len:
        raise ValueError(f"Row explosion in gold_macro: {original_len} -> {len(df)}")
    
    target_symbols = ['XAUUSD']
    for symbol in target_symbols:
        mask = df['symbol'] == symbol
        if 'dxy' in df.columns and mask.any():
            df.loc[mask, 'DXY_1d_return'] = df['dxy'].pct_change()
            df.loc[mask, 'gold_dxy_ratio'] = df.loc[mask, 'Close'] / df.loc[mask, 'dxy'].replace(0, np.nan)
        if 'real_yield' in df.columns and mask.any():
            df.loc[mask, 'real_yield_level'] = df['real_yield']
            df.loc[mask, 'gold_yield_inverse'] = 1 / (df.loc[mask, 'real_yield'].clip(lower=0.001))
        if 'vix' in df.columns and mask.any():
            df.loc[mask, 'vix_regime'] = (df['vix'] > df['vix'].rolling(252).quantile(0.8)).astype(int)
        if 'btc' in df.columns and mask.any():
            df.loc[mask, 'btc_gold_ratio'] = df['btc'] / df['Close'].replace(0, np.nan)
    return df

# ===========================================
# GOLD CROSS ASSET — Cites: Baur & McDermott (2010), So & Wang (2014)
# ===========================================

def add_gold_cross_asset_features(df: pd.DataFrame, cross_assets: dict = None) -> pd.DataFrame:
    if cross_assets is None or len(cross_assets) == 0:
        return df

    df = df.copy()
    print(f"📊 BEFORE merge: df.shape = {df.shape}, nunique Dates = {df['Date'].nunique()}")

    # Create cross_df — ensure clean, deduplicated, Datetime index
    cross_df = pd.DataFrame(cross_assets)
    cross_df.index = pd.to_datetime(cross_df.index)
    cross_df = cross_df.sort_index()

    # DEDUPLICATE INDEX — CRITICAL
    if cross_df.index.duplicated().any():
        print(f"⚠️  Deduplicating cross_df index: {cross_df.index.duplicated().sum()} duplicates found")
        cross_df = cross_df[~cross_df.index.duplicated(keep='last')]

    print(f"📊 cross_df.shape = {cross_df.shape}, cross_df.index.is_unique = {cross_df.index.is_unique}")

    # 👇 USE MERGE WITH EXPLICIT INDEX CONTROL — NOT JOIN
    original_len = len(df)
    df = df.merge(cross_df, left_on='Date', right_index=True, how='left')
    new_len = len(df)

    print(f"ℹ️  AFTER merge: {original_len} → {new_len} rows (expected: no change)")

    if new_len != original_len:
        raise ValueError(f"Row count changed after merge! {original_len} -> {new_len}. Data alignment issue.")

    # Now compute gold-specific features for XAUUSD
    target_symbols = ['XAUUSD']
    for symbol in target_symbols:
        mask = df['symbol'] == symbol
        if not mask.any():
            continue

        if 'silver' in df.columns and mask.any():
            df.loc[mask, 'gold_silver_ratio'] = df.loc[mask, 'Close'] / df.loc[mask, 'silver'].replace(0, np.nan)
            rolling_mean = df.loc[mask, 'gold_silver_ratio'].rolling(50).mean()
            rolling_std = df.loc[mask, 'gold_silver_ratio'].rolling(50).std()
            df.loc[mask, 'gsr_zscore_50d'] = (df.loc[mask, 'gold_silver_ratio'] - rolling_mean) / rolling_std.replace(0, np.nan)

        if 'spx' in df.columns and mask.any():
            df.loc[mask, 'spx_20d_return'] = df.loc[mask, 'spx'].pct_change(20)
            df.loc[mask, 'gold_spx_divergence'] = df.loc[mask, 'Close'].pct_change(20) - df['spx_20d_return']

        if 'tlt' in df.columns and mask.any():
            df.loc[mask, 'tlt_gold_corr_30d'] = (
                df.loc[mask, 'Close'].rolling(30).corr(df.loc[mask, 'tlt']).fillna(0)
            )

    return df

# ===========================================
# ENTROPY & HURST — Cites: Pincus (1991), Peters (1994)
# ✅ FIXED: Initialize market_regime as object dtype
# ===========================================

def sliding_window_entropy(series, window=30, bins=10):
    entropies = []
    for i in range(len(series)):
        if i < window:
            entropies.append(np.nan)
        else:
            window_data = series[i-window:i]
            hist, _ = np.histogram(window_data, bins=bins, density=True)
            hist = hist[hist > 0]
            ent = entropy(hist)
            entropies.append(ent)
    return np.array(entropies)

def add_chaos_entropy_features(df: pd.DataFrame):
    df = df.copy()
    c = df['Close']
    df['price_entropy_30'] = sliding_window_entropy(c, window=30, bins=10)
    df['entropy_zscore'] = (df['price_entropy_30'] - df['price_entropy_30'].rolling(60).mean()) / df['price_entropy_30'].rolling(60).std()
    def hurst_rs(series, lag_range=[2,5,10,20]):
        lags = np.array(lag_range)
        rs_values = []
        for lag in lags:
            if len(series) <= lag: continue
            chunks = [series[i:i+lag] for i in range(0, len(series)-lag, lag)]
            if len(chunks) < 2: continue
            means = [np.mean(chunk) for chunk in chunks]
            cum_dev = [np.cumsum(chunk - mean) for chunk, mean in zip(chunks, means)]
            ranges = [np.max(cd) - np.min(cd) for cd in cum_dev]
            stds = [np.std(chunk) for chunk in chunks]
            rs = np.nanmean([r/s if s > 0 else 0 for r,s in zip(ranges,stds)])
            rs_values.append(rs)
        if len(rs_values) < 2: return np.nan
        log_rs = np.log(rs_values)
        log_lags = np.log(lags[:len(rs_values)])
        slope, _ = np.polyfit(log_lags, log_rs, 1)
        return slope
    hurst_vals = []
    for i in range(len(c)):
        if i < 120:
            hurst_vals.append(np.nan)
        else:
            h = hurst_rs(c[i-120:i])
            hurst_vals.append(h)
    df['hurst_exponent_120'] = hurst_vals
    
    # ✅ FIX: Initialize as object dtype to hold strings
    df['market_regime'] = 'random'  # Default
    df.loc[df['hurst_exponent_120'] > 0.6, 'market_regime'] = 'trending'
    df.loc[df['hurst_exponent_120'] < 0.4, 'market_regime'] = 'mean_reverting'
    
    return df

# ===========================================
# WAVELET FEATURES — Cites: Gençay et al. (2002)
# ===========================================

def add_wavelet_features(df: pd.DataFrame, wavelet='db4', level=4):
    df = df.copy()
    close = df['Close'].values
    coeffs = pywt.wavedec(close, wavelet, level=level)
    cA = coeffs[0]
    cDs = coeffs[1:]
    def pad_to_len(arr, target_len):
        pad_width = target_len - len(arr)
        return np.pad(arr, (0, pad_width), mode='edge')
    df[f'wavelet_trend_{wavelet}'] = pad_to_len(cA, len(close))
    for i, cD in enumerate(cDs, 1):
        df[f'wavelet_cycle_{i}_{wavelet}'] = pad_to_len(cD, len(close))
    for i in range(1, len(cDs)+1):
        col = f'wavelet_cycle_{i}_{wavelet}'
        df[f'{col}_vol_10'] = df[col].rolling(10).std()
    return df
# ===========================================
# TARGET ENGINEERING — For Supervised Learning
# ===========================================

def add_target_variables(df: pd.DataFrame, horizons: list = [1, 5, 10]) -> pd.DataFrame:
    """
    Add forward return and direction targets for supervised learning.
    Computed per symbol to avoid look-ahead and cross-symbol leakage.
    Args:
        df: DataFrame with 'symbol', 'Date', 'Close'
        horizons: List of forward horizons in days (e.g., [1,5,10])
    """
    df = df.copy()
    
    for h in horizons:
        df[f'target_close_{h}d'] = df.groupby('symbol')['Close'].apply(
            lambda x: x.shift(-h-1)).reset_index(level=0, drop=True)
        # Compute forward log return — grouped by symbol
        df[f'target_LogReturn_{h}d'] = df.groupby('symbol')['Close'].apply(
            lambda x: np.log(x.shift(-(h+1)) / x.shift(-1)) * 100  # in percent
        ).reset_index(level=0, drop=True)

        # Compute binary direction (1 = up, 0 = down or flat)
        df[f'target_direction_{h}d'] = (df[f'target_LogReturn_{h}d'] > 0).astype(int)
    
    return df
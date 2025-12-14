import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import List, Dict, Tuple

def load_temporal_shap(filepath: str) -> pd.DataFrame:
    """
    Load temporal SHAP values from CSV produced by the temporal SHAP computation.
    Expected format: 
        index = dates
        columns = SHAP values of features
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df = df.sort_index()
    return df


def extract_regime_samples(df: pd.DataFrame,
                           crisis_windows: List[Tuple[str, str]]
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporal SHAP values into crisis and normal regimes.
    
    Parameters:
        df : DataFrame indexed by date with SHAP values per feature
        crisis_windows : list of (start_date, end_date) tuples

    Returns:
        crisis_df : SHAP rows inside crisis windows
        normal_df : SHAP rows outside all crisis windows
    """
    crisis_mask = np.zeros(len(df), dtype=bool)

    for start, end in crisis_windows:
        crisis_mask |= (df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))

    crisis_df = df[crisis_mask]
    normal_df = df[~crisis_mask]

    return crisis_df, normal_df


def ks_test_shap_regimes(crisis_df: pd.DataFrame,
                         normal_df: pd.DataFrame,
                         top_features: List[str]
                         ) -> pd.DataFrame:
    """
    Run Kolmogorov–Smirnov test for regime dependence of SHAP values.

    Returns:
        DataFrame with KS statistic, p value, crisis mean, normal mean.
    """
    results = []

    for feat in top_features:
        crisis_vals = crisis_df[feat].dropna()
        normal_vals = normal_df[feat].dropna()

        if len(crisis_vals) < 10 or len(normal_vals) < 10:
            continue  # insufficient data for valid test

        ks_stat, pval = ks_2samp(crisis_vals, normal_vals)

        results.append({
            "feature": feat,
            "crisis_mean": crisis_vals.mean(),
            "normal_mean": normal_vals.mean(),
            "ratio_crisis_normal": crisis_vals.mean() / normal_vals.mean()
                                   if normal_vals.mean() != 0 else np.nan,
            "ks_statistic": ks_stat,
            "p_value": pval
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("ks_statistic", ascending=False)

    return results_df


def run_shap_regime_pipeline(shap_temporal_path: str,
                             top_features: List[str],
                             crisis_windows: List[Tuple[str, str]],
                             output_csv: str = None
                             ) -> pd.DataFrame:
    """
    Full RQ3 SHAP regime dependence pipeline.

    Parameters:
        shap_temporal_path : path to temporal SHAP CSV
        top_features : list of features to evaluate
        crisis_windows : windows to treat as crisis periods
        output_csv : optional path to save pipeline results

    Returns:
        DataFrame containing KS results.
    """

    df = load_temporal_shap(shap_temporal_path)

    crisis_df, normal_df = extract_regime_samples(df, crisis_windows)

    results_df = ks_test_shap_regimes(crisis_df, normal_df, top_features)

    if output_csv:
        results_df.to_csv(output_csv, index=False)

    return results_df

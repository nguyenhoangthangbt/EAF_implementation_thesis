import pathlib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt


def build_crisis_windows(drift_dates, window_days=30):
    """Build symmetric crisis windows around each drift date."""
    windows = []
    for d in drift_dates:
        start = d - pd.Timedelta(days=window_days)
        end = d + pd.Timedelta(days=window_days)
        windows.append((start, end))
    return windows


def compute_shap_values_xgb(model, X,y, approximate=False):
    """
    Compute SHAP values for an XGBoost model on a given feature matrix.

    Parameters
    ----------
    model : xgboost.Booster or xgboost.XGBRegressor
    X : pandas.DataFrame
    approximate : bool
        If True, use approximate SHAP to reduce cost.

    Returns
    -------
    shap_values : numpy.ndarray
    base_values : float
    """
    if isinstance(model, xgb.Booster):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.TreeExplainer(model.get_booster())

    if approximate:
        shap_values = explainer.shap_values(X,y, approximate=True)
    else:
        shap_values = explainer.shap_values(X,y)

    base_values = explainer.expected_value
    return shap_values, base_values


def compute_global_shap(shap_values, feature_names, top_k=20):
    """
    Compute global feature importance based on mean absolute SHAP value.

    Returns a DataFrame sorted by importance.
    """
    shap_abs_mean = np.mean(np.abs(shap_values), axis=0)
    df_global = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": shap_abs_mean
    }).sort_values("mean_abs_shap", ascending=False)

    if top_k is not None:
        df_global = df_global.head(top_k).reset_index(drop=True)

    return df_global


def compute_temporal_shap(shap_values, index, feature_names, window=30, top_k=10):
    """
    Compute temporal SHAP profiles as rolling mean of absolute SHAP values.

    Returns a DataFrame with DateTime index and columns for top features.
    """
    shap_abs = np.abs(shap_values)
    df_shap = pd.DataFrame(shap_abs, index=index, columns=feature_names)

    overall_importance = df_shap.mean(axis=0).sort_values(ascending=False)
    top_features = overall_importance.head(top_k).index

    df_top = df_shap[top_features]
    df_rolling = df_top.rolling(window=window, min_periods=1).mean()

    return df_rolling, list(top_features)


def summarize_crisis_vs_normal(df_temporal, crisis_windows):
    """
    Summarize mean absolute SHAP in crisis versus non crisis periods.

    Returns a DataFrame with
    feature, crisis_mean, normal_mean, ratio_crisis_normal.
    """
    index = df_temporal.index
    crisis_mask = pd.Series(False, index=index)

    for start, end in crisis_windows:
        crisis_mask |= (index >= start) & (index <= end)

    normal_mask = ~crisis_mask

    crisis_mean = df_temporal[crisis_mask].mean()
    normal_mean = df_temporal[normal_mask].mean()

    summary = pd.DataFrame({
        "feature": df_temporal.columns,
        "crisis_mean": crisis_mean.values,
        "normal_mean": normal_mean.values
    })

    summary["ratio_crisis_normal"] = summary["crisis_mean"] / summary["normal_mean"].replace(0, np.nan)

    return summary.sort_values("ratio_crisis_normal", ascending=False)


def save_global_shap(df_global, output_dir):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path_csv = output_dir / "global_shap_importance.csv"
    df_global.to_csv(path_csv, index=False)
    return path_csv


def save_temporal_shap(df_temporal, output_dir):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path_csv = output_dir / "temporal_shap_rolling.csv"
    df_temporal.to_csv(path_csv)
    return path_csv


def save_crisis_summary(df_crisis, output_dir):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path_csv = output_dir / "crisis_vs_normal_shap.csv"
    df_crisis.to_csv(path_csv, index=False)
    return path_csv


def plot_global_shap_bar(df_global, output_dir, top_k=10):
    """
    Simple horizontal bar chart for global SHAP.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_plot = df_global.head(top_k)[::-1]

    plt.figure(figsize=(8, 5))
    plt.barh(df_plot["feature"], df_plot["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title("Global SHAP feature importance")
    plt.tight_layout()

    path_png = output_dir / "global_shap_bar.png"
    plt.savefig(path_png, dpi=300)
    plt.close()
    return path_png


def plot_temporal_shap(df_temporal, output_dir, top_k=5):
    """
    Line plot of rolling SHAP values for the top K features.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_plot = df_temporal.iloc[:, :top_k]

    plt.figure(figsize=(10, 6))
    for col in df_plot.columns:
        plt.plot(df_plot.index, df_plot[col], label=col)

    plt.xlabel("Date")
    plt.ylabel("Rolling mean absolute SHAP")
    plt.title("Temporal SHAP dynamics for top features")
    plt.legend(loc="upper right")
    plt.tight_layout()

    path_png = output_dir / "temporal_shap_top_features.png"
    plt.savefig(path_png, dpi=300)
    plt.close()
    return path_png

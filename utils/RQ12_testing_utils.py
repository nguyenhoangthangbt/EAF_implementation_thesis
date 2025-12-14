from pathlib import Path
import os,sys

project_root = os.path.abspath(Path("../."))
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from scipy import stats




COL_MODEL = "baseline_modelName"
COL_SEED = "GLOBAL_SEED"
COL_TARGET = "target_isClose"
COL_PERIOD = "period"


def load_data(path, period):
    df = pd.read_csv(path)
    df = df[df[COL_PERIOD] == period].copy()
    df[COL_TARGET] = df[COL_TARGET].astype(bool)
    df["target_label"] = df[COL_TARGET].map({True: "Close", False: "LogReturn"})
    return df

def paired_frame(df, model, metric):
    df_m = df[df[COL_MODEL] == model]
    p = df_m.pivot_table(index=COL_SEED, columns="target_label", values=metric)
    p = p.dropna()
    return p if len(p) >= 2 else None

def paired_tests(df, model, metric):
    x = df["Close"].values
    y = df["LogReturn"].values

    t_stat, t_p = stats.ttest_rel(x, y)

    try:
        w_stat, w_p = stats.wilcoxon(x, y)
    except:
        w_stat, w_p = np.nan, np.nan

    return dict(
        model=model,
        metric=metric,
        n=len(x),
        mean_close=float(np.mean(x)),
        mean_logreturn=float(np.mean(y)),
        t_stat=t_stat,
        t_p=t_p,
        wilcoxon_stat=w_stat,
        wilcoxon_p=w_p
    )

def run_RQ1_tests(raw_output,test_result_outdir):
    PERIOD = "validating"
    METRICS = ["r2", "sharpe_ratio", "total_return", "win_rate","max_drawdown"]
    df = load_data(raw_output, PERIOD)
    models = sorted(df[COL_MODEL].unique())
    ## Financial perf testing
    rows = []
    for model in models:
        for metric in METRICS:
            paired = paired_frame(df, model, metric)
            if paired is None:
                continue
            rows.append(paired_tests(paired, model, metric))

    out = pd.DataFrame(rows)
    print(out)
    out.to_csv(test_result_outdir+"/RQ1_tests.csv", index=False)

def run_RQ2_tests(raw_output, test_result_outdir):
    PERIOD = "testing"
    TARGET = False  # Log Return
    METRICS = ["rmse","sharpe_ratio", "total_return", "win_rate", "max_drawdown"]

    df = load_data(raw_output, PERIOD)
    df = df[df[COL_TARGET] == TARGET].copy()

    # Separate EAF from competitors
    df_eaf = df[df[COL_MODEL] == "baseline_EAF"]
    competitors = sorted(df[COL_MODEL].unique())
    competitors = [m for m in competitors if m != "baseline_EAF"]

    rows = []

    for model in competitors:
        df_m = df[df[COL_MODEL] == model]

        # Align by seed
        merged = pd.merge(
            df_eaf,
            df_m,
            on=COL_SEED,
            suffixes=("_EAF", "_OTHER")
        )

        if len(merged) < 2:
            continue

        for metric in METRICS:
            x = merged[f"{metric}_EAF"].values
            y = merged[f"{metric}_OTHER"].values

            t_stat, t_p = stats.ttest_rel(x, y)

            try:
                w_stat, w_p = stats.wilcoxon(x, y)
            except:
                w_stat, w_p = np.nan, np.nan

            rows.append(dict(
                comparison=f"EAF_vs_{model}",
                metric=metric,
                n=len(x),
                mean_EAF=float(np.mean(x)),
                mean_other=float(np.mean(y)),
                t_stat=t_stat,
                t_p=t_p,
                wilcoxon_stat=w_stat,
                wilcoxon_p=w_p
            ))

    out = pd.DataFrame(rows)
    print(out)
    out.to_csv(test_result_outdir + "/RQ2_tests.csv", index=False)


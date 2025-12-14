import ast
import pandas as pd
pd.set_option('display.max_columns', None)

def convert_experiment_results_txt_to_df(path="experiment_list_baseline.txt"):
    with open(path, 'r') as f:
        raw= f.read()
    raw = str(raw).replace('inf','0').replace('nan','0').replace('fi0cial_metrics','financial_metrics')
    raw = eval(raw)
    rows = []
    for exp in raw:
        seed = exp["GLOBAL_SEED"]
        target_isClose = exp["target_isClose"]

        # iterate over each model key
        for model_key in ['baseline1_XGBM',"baseline2_ARIMA", "baseline3_LSTM", "baseline4_LGBM", "baseline5_BuyAndHold",'EAF_metrics']:
            results = exp.get(model_key,0)  
            if not results:
                continue
            for block in results:
                period = block["period"]

                # statistical metrics
                for k, v in block["statistical_metrics"].items():
                    rows.append({
                        "model": model_key,
                        "seed": seed,
                        "target_isClose": target_isClose,
                        "period": period,
                        "metric": k,
                        "value": v
                    })

                # financial metrics
                
                for k, v in block["financial_metrics"].items():
                    rows.append({
                        "model": model_key,
                        "seed": seed,
                        "target_isClose": target_isClose,
                        "period": period,
                        "metric": k,
                        "value": v
                    })
    df = pd.DataFrame(rows)
    df_comparative = df.pivot_table(
                                index=['seed','model','target_isClose','period'],
                                columns=['metric'],
                                values='value',
                                aggfunc='first').reset_index().sort_values(['model', 'seed'])
    return df_comparative


    
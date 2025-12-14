## Experimental Evaluation of an Explainable Adaptive Framework for Gold Price Prediction with XGBoost, ADWIN, and SHAP

**Author:** Nguyen Hoang Thang  
**Institution:** Liverpool John Moores University  
**Year:** 2025  

---

## Overview

This repository contains the complete Python implementation of the **Explainable Adaptive Framework (EAF)** developed for the thesis *“Experimental Evaluation of an Explainable Adaptive Framework for Gold Price Prediction with XGBoost, ADWIN, and SHAP.”*

The framework integrates:
- **XGBoost** for nonlinear financial forecasting,
- **ADWIN** for online concept drift detection and adaptive retraining,
- **TreeSHAP** for global, local, and temporal model explainability.

The implementation supports walk-forward evaluation, regime-aware adaptation, and interpretable analysis under nonstationary market conditions.

---

## Relation to the Thesis

This repository provides the computational implementation corresponding to the methodology and experiments described in the thesis:

- **Feature engineering:** Section 4.7  
- **Concept drift detection and adaptive retraining:** Section 4.9.2  
- **Explainability with TreeSHAP:** Section 4.9.3  
- **Experimental evaluation and results:** Chapter 5  
- **Research question testing (RQ1–RQ3):** Chapters 5 and 6  

All results reported in the thesis can be reproduced using the notebooks provided here.

---

## Repository Structure

```text
EAF_implementation_thesis/
│
├── 99notebooks/
│   ├── 1EAF_data_collect_preprocess_featureEng.ipynb
│   ├── 2EAF_EDA.ipynb
│   ├── 3EAF_baselines_run.ipynb
│   ├── 4EAF_adwin_shap_RQ3test.ipynb
│   ├── 5EAF_results_display.ipynb
│   └── 6EAF_RQ12_test.ipynb
│
├── config/               # Configuration files
├── data/
│   └── raw/              # Raw gold price data (pre-downloaded)
├── logs/                 # Execution logs
├── models/               # Saved models
├── results/              # Output metrics and figures
├── scripts/              # Supporting scripts
├── utils/                # Utility functions
├── utils_strategies/     # Trading strategy utilities
│
├── requirements_core.txt
├── requirements_full.txt
└── README.md
```

## Data

* **Asset:** Gold futures (GC=F)
* **Source:** Yahoo Finance
* **Period:** January 1, 2005 – December 31, 2024

Raw data are **pre-downloaded** and stored locally in:

```text
data/raw/
```

No automatic data downloading is performed by the code.

---

## Environment and Dependencies

* **Python version:** 3.10+
* Core dependencies include:

  * xgboost
  * lightgbm
  * river
  * shap
  * optuna
  * pandas
  * numpy
  * scikit-learn

To install core dependencies:

```bash
pip install -r requirements_core.txt
```

To install the full experimental environment:

```bash
pip install -r requirements_full.txt
```

---

## Running the Experiments

The experimental pipeline is notebook-based and should be executed **sequentially** in the following order:

1. **Preprocessing and feature engineering**
   `1EAF_preprocess_featureEng.ipynb`

2. **Exploratory data analysis**
   `2EAF_EDA.ipynb`

3. **Baseline model training and evaluation**
   `3EAF_baselines_run.ipynb`

4. **Explainable Adaptive Framework (ADWIN + SHAP) and RQ3 analysis**
   `4EAF_adwin_shap_RQ3test.ipynb`

5. **Results aggregation and visualization**
   `5EAF_results_display.ipynb`

6. **Research question testing for RQ1 and RQ2**
   `6EAF_RQ12_test.ipynb`

---

## Reproducibility

* All experiments are executed **five times** using fixed random seeds:

  ```text
  10, 200, 3000, 40000, 500000
  ```
* Reported results correspond to aggregated outputs across these runs.
* Temporal data ordering is strictly preserved to avoid look-ahead bias.

---

## License

This repository is provided for **academic use only**.

---

## Citation

Nguyen Hoang Thang (2025).
*Experimental Evaluation of an Explainable Adaptive Framework for Gold Price Prediction with XGBoost, ADWIN, and SHAP.*
Liverpool John Moores University.

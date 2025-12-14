import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional,List,Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
# Add these imports at top
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.ensemble import RandomForestRegressor
import optuna

import warnings
warnings.filterwarnings('ignore')
import sys,os, pandas as pd
from pathlib import Path
project_root = os.path.abspath(Path("../."))
sys.path.append(str(project_root))
from utils_strategies.backtest_utils import backtest_strategy
from utils import config_utils
config = config_utils.load_config()

# Check if PyCaret is available
try:
    from pycaret.regression import setup, compare_models, create_model, tune_model, \
        finalize_model, predict_model, plot_model,interpret_model, blend_models, ensemble_model, models

    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    print("Warning: PyCaret not available. Install with 'pip install pycaret'")

class PyCaretModelManager:
    def __init__(
        self, 
        features_df: pd.DataFrame, 
        target_series: pd.Series,
        time_periods: Dict = {
                'training': {'start': '2005-01-01', 'end': '2018-12-31'},
                'validation': {'start': '2019-01-01', 'end': '2021-12-31'},
                'testing': {'start': '2022-01-01', 'end': '2025-01-01'}
            }
    ):
        
        if not PYCARET_AVAILABLE:
            raise ImportError("PyCaret is not installed. Install with 'pip install pycaret'")
        
        # Validate inputs
        if not isinstance(features_df, pd.DataFrame):
            raise TypeError("features_df must be a pandas DataFrame")
        if not isinstance(target_series, pd.Series):
            raise TypeError("target_series must be a pandas Series")
        if not isinstance(time_periods, dict):
            raise TypeError("time_periods must be a dictionary")
        
        # Parse time periods
        if config.get('time_periods',0):
            self.time_periods = config['time_periods']
        else:
            self.time_periods = time_periods

        for period in ['training', 'validation', 'testing']:
            if period not in time_periods:
                raise ValueError(f"Missing '{period}' period in time_periods")
            if 'start' not in time_periods[period] or 'end' not in time_periods[period]:
                raise ValueError(f"Missing start/end dates for '{period}' period")
            
            self.time_periods[period] = {
                'start': pd.Timestamp(time_periods[period]['start']),
                'end': pd.Timestamp(time_periods[period]['end'])
            }
        
        # Sort time periods chronologically
        sorted_periods = sorted(
            self.time_periods.items(),
            key=lambda x: (x[1]['start'], x[1]['end'])
        )
        
        # Validate chronological order
        for i in range(1, len(sorted_periods)):
            prev_end = sorted_periods[i-1][1]['end']
            curr_start = sorted_periods[i][1]['start']
            if curr_start <= prev_end:
                raise ValueError(
                    f"Time periods must be non-overlapping and chronological. "
                    f"Period '{sorted_periods[i][0]}' starts before '{sorted_periods[i-1][0]}' ends."
                )
        
        # Align features and target
        self.features = features_df.copy()
        self.target = target_series.loc[features_df.index].copy()
        
        # Check for NaN values
        if self.features.isna().any().any():
            print("Warning: Features contain NaN values. Consider imputing or dropping.")
        if self.target.isna().any():
            print("Warning: Target contains NaN values. Consider imputing or dropping.")
        
        # Split data according to time periods
        self._split_data()
        
        # Store results
        self.setup_params = {}
        self.comparison_results = None
        self.best_model = None
        self.model_metrics = {}
        self.feature_importance = None
        self.model_explanations = {}
    
    def _split_data(self):
        """Split data according to EAF time periods"""
        # Convert index to datetime if needed
        if not isinstance(self.features.index, pd.DatetimeIndex):
            self.features.index = pd.to_datetime(self.features.index)
            self.target.index = pd.to_datetime(self.target.index)
        
        # Sort by date
        self.features = self.features.sort_index()
        self.target = self.target.sort_index()
        
        # Split data
        self.X_train = self.features[
            (self.features.index >= self.time_periods['training']['start']) & 
            (self.features.index <= self.time_periods['training']['end'])
        ]
        self.y_train = self.target.loc[self.X_train.index]
        
        self.X_val = self.features[
            (self.features.index >= self.time_periods['validation']['start']) & 
            (self.features.index <= self.time_periods['validation']['end'])
        ]
        self.y_val = self.target.loc[self.X_val.index]
        
        # Testing period might include future dates not in our data
        self.X_test = self.features[
            (self.features.index >= self.time_periods['testing']['start']) & 
            (self.features.index <= self.time_periods['testing']['end'])
        ]
        self.y_test = self.target.loc[self.X_test.index] if not self.X_test.empty else None
        
        # Print split summary
        print("\nData Split Summary (EAF Time Periods):")
        print(f"  Training: {len(self.X_train)} samples ({self.time_periods['training']['start'].date()} to {self.time_periods['training']['end'].date()})")
        print(f"  Validation: {len(self.X_val)} samples ({self.time_periods['validation']['start'].date()} to {self.time_periods['validation']['end'].date()})")
        print(f"  Testing: {len(self.X_test)} samples ({self.time_periods['testing']['start'].date()} to {min(self.time_periods['testing']['end'], self.features.index.max()).date()})")
        
        # Verify no overlap
        train_end = self.X_train.index.max()
        val_start = self.X_val.index.min()
        if val_start <= train_end:
            raise ValueError("Validation period starts before training period ends")
        
        val_end = self.X_val.index.max()
        test_start = self.X_test.index.min() if not self.X_test.empty else None
        if test_start is not None and test_start <= val_end:
            raise ValueError("Testing period starts before validation period ends")
    
    def setup_experiment(
        self,
        fold: int = 3,  # Reduced from 5 to 3 for training period
        session_id: int = 42,
        **kwargs
    ) -> pd.DataFrame:
        
        # Create a combined DataFrame for PyCaret with proper time alignment
        data = self.X_train.copy()        
        data['target'] = self.y_train
        
        # Set up PyCaret environment with time-series considerations
        self.setup_params = {
            'data': data,
            'target': 'target',
            'data_split_stratify':False,
            'data_split_shuffle': False,
            'fold_shuffle': False,
            'fold_strategy': 'timeseries',
            'fold': fold,
            'session_id': session_id,
            'numeric_features': list(self.X_train.columns),
            'ignore_features': None,
            'profile': False,
            **kwargs
        }
        
        # Run setup
        print("\nSetting up PyCaret regression experiment with EAF time periods...")
        print(f"  Training features: {self.X_train.shape[1]}")
        print(f"  Training samples: {self.X_train.shape[0]}")
        print(f"  Validation samples: {self.X_val.shape[0]}")
        print(f"  Testing samples: {len(self.X_test)}")
        print(f"  Cross-validation: {fold}-fold timeseries (training period only)")
        
        # Execute setup
        transformed_data = setup(**self.setup_params)
        self.models_dict = models().to_dict()
        return transformed_data
    
    def get_modelID_from_modelName(self, val):
        models_dict= self.models_dict['Reference']
        for k,v in models_dict.items():
            if val in v:
                return k    
        
    def compare_models(
        self,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        sort: str = 'RMSE',
        n_select: int = 1,
        turbo: bool = True,
        verbose:bool = True
    ) -> Union[pd.DataFrame, List]:
        
        if not hasattr(self, 'setup_params') or not self.setup_params:
            raise ValueError("Must call setup_experiment() before comparing models")
        
        print(f"\nComparing regression models on validation data (sorting by {sort})...")
        
        # Run model comparison on training data
        self.comparison_results = compare_models(
            include=include,
            exclude=exclude,
            sort=sort,
            n_select=n_select,
            turbo=turbo,
            verbose=verbose
        )
        
        # Handle single vs multiple model return
        if n_select == 1:
            self.best_model = self.comparison_results
            print(f"  Best model: {self.best_model.__class__.__name__}")
            return self.comparison_results
        else:
            self.best_model = self.comparison_results[0]
            print(f"  Top {n_select} models selected")
            return self.comparison_results
    
    def _tune_model(self,dic_paras):
        print(f"  Fine-tuning Best model: {self.best_model.__class__.__name__}")
        tuned_model = tune_model(self.comparison_results,**dic_paras)
        return tuned_model
   
    def finalize_and_evaluate(
        self,
        model,
        plot_results: bool = True
    ) -> Dict[str, Dict[str, float]]:
     
        print("\nFinalizing and evaluating model across all time periods...")
        
        # Finalize model (train on full training dataset)
        final_model = finalize_model(model)
        
        # Calculate metrics for all periods
        metrics = {}
        
        # Training metrics
        train_pred = predict_model(final_model, data=self.X_train)
        metrics['training'] = self._calculate_metrics(self.y_train, train_pred['prediction_label'])
        
        # Validation metrics
        val_pred = predict_model(final_model, data=self.X_val)
        metrics['validation'] = self._calculate_metrics(self.y_val, val_pred['prediction_label'])
        
        # Testing metrics (if available)
        if not self.X_test.empty:
            test_pred = predict_model(final_model, data=self.X_test)
            metrics['testing'] = self._calculate_metrics(self.y_test, test_pred['prediction_label'])
        
        # Store metrics
        self.model_metrics = metrics
        
        # Print metrics
        print("\nModel Performance Metrics by Time Period:")
        for period, period_metrics in metrics.items():
            print(f"\n{period.capitalize()} Period:")
            print(f"  RMSE: {period_metrics['RMSE']:.6f}")
            print(f"  MAE: {period_metrics['MAE']:.6f}")
            print(f"  R2: {period_metrics['R2']:.4f}")
            print(f"  MAPE: {period_metrics['MAPE']:.2f}%")
        
        # Generate diagnostic plots
        if plot_results:
            self._generate_diagnostic_plots(final_model)
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate performance metrics"""
        return {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def _generate_diagnostic_plots(self, model):
        """Generate diagnostic plots for model evaluation across time periods"""
        print("\nGenerating diagnostic plots across time periods...")
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Plot for each time period
        periods = ['training', 'validation', 'testing']
        period_data = [
            (self.X_train, self.y_train),
            (self.X_val, self.y_val),
            (self.X_test, self.y_test)
        ]
        
        for i, (period, (X, y)) in enumerate(zip(periods, period_data)):
            if X.empty or y is None:
                continue
                
            # Generate predictions
            predictions = predict_model(model, data=X)
            y_pred = predictions['prediction_label']
            
            # Actual vs Predicted
            axes[i, 0].scatter(y, y_pred, alpha=0.5)
            min_val = min(min(y), min(y_pred))
            max_val = max(max(y), max(y_pred))
            axes[i, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
            axes[i, 0].set_title(f'{period.capitalize()} Period: Actual vs Predicted', fontsize=12)
            axes[i, 0].set_xlabel('Actual Values')
            axes[i, 0].set_ylabel('Predicted Values')
            
            # Residuals
            residuals = y - y_pred
            axes[i, 1].scatter(y_pred, residuals, alpha=0.5)
            axes[i, 1].axhline(y=0, color='r', linestyle='--')
            axes[i, 1].set_title(f'{period.capitalize()} Period: Residuals', fontsize=12)
            axes[i, 1].set_xlabel('Predicted Values')
            axes[i, 1].set_ylabel('Residuals')
        
        plt.tight_layout()
        plt.show()
        
        # Feature importance (across all periods)
        try:
            # Create a DataFrame for feature importance
            feature_imp = pd.DataFrame({
                'Feature': self.X_train.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Plot
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_imp.head(15))
            plt.title('Top 15 Feature Importances', fontsize=14)
            plt.tight_layout()
            plt.show()
            
            # Store feature importance
            self.feature_importance = feature_imp
        except:
            print("  Could not generate feature importance plot")
    
    def interpret_model(self, model, plot_type: str = 'summary'):
        print(f"\nGenerating {plot_type} interpretation...")
        
        try:
            if plot_type == 'summary':
                interpret_model(model, plot='summary')
            elif plot_type == 'correlation':
                interpret_model(model, plot='correlation')
            elif plot_type == 'reason':
                # Use a sample from validation data for explanation
                if not self.X_val.empty:
                    interpret_model(model, plot='reason', observation=self.X_val.iloc[0])
            else:
                print(f"  Unknown plot type: {plot_type}")
        except Exception as e:
            print(f"  Could not generate {plot_type} plot: {str(e)}")
    
    def generate_trading_signals(
        self,
        model,
        threshold: float = 0.001,
        future_features: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.Series]:
        
        print("\nGenerating trading signals from model predictions...")
        
        # Generate signals for all periods
        signals = {}
        
        # Training period signals
        if not self.X_train.empty:
            train_pred = predict_model(model, data=self.X_train)
            train_returns = (train_pred['prediction_label'] / self.y_train.shift(1)) - 1
            signals['training'] = self._generate_signals_from_returns(train_returns, threshold)
        
        # Validation period signals
        if not self.X_val.empty:
            val_pred = predict_model(model, data=self.X_val)
            val_returns = (val_pred['prediction_label'] / self.y_val.shift(1)) - 1
            signals['validation'] = self._generate_signals_from_returns(val_returns, threshold)
        
        # Testing period signals
        if not self.X_test.empty and self.y_test is not None:
            test_pred = predict_model(model, data=self.X_test)
            test_returns = (test_pred['prediction_label'] / self.y_test.shift(1)) - 1
            signals['testing'] = self._generate_signals_from_returns(test_returns, threshold)
        
        # Future signals (if provided)
        if future_features is not None:
            future_pred = predict_model(model, data=future_features)
            # Estimate current close from most recent training data
            current_close = self.y_train.iloc[-1]
            future_returns = (future_pred['prediction_label'] / current_close) - 1
            signals['future'] = self._generate_signals_from_returns(future_returns, threshold)
        
        return signals
    
    def _generate_signals_from_returns(self, returns, threshold):
        """Helper: Generate trading signals from predicted returns"""
        signals = pd.Series(0, index=returns.index)
        signals[returns > threshold] = 1
        signals[returns < -threshold] = -1
        return signals
    
    def backtest_trading_signals_strategy(
        self,
        model,
        execution_threshold: float = 0.001,
        strategy_type: str = "intraday",
        write_OHLCV_output: bool = False,
        **backtest_params
    ):
        
        print("\nBacktesting trading strategy across all time periods...")
        
        # Generate signals for all periods
        all_signals = self.generate_trading_signals(model, execution_threshold)
        
        # Backtest each period
        backtest_results = {}
        
        # Function to run backtest for a period
        def run_period_backtest(period, signals, df_OHLCV):
            if signals.empty:
                return None
                
            # Run backtest using your existing backtester
            results = backtest_strategy.run_backtest2_SL_TP(
                df_OHLCV=df_OHLCV,
                signals=signals,
                strategy_name="model_prediction",
                strategy_type=strategy_type,
                **backtest_params
            )
            return results
        
        # Backtest training period
        if 'training' in all_signals and not all_signals['training'].empty:
            # Create OHLCV data for training period
            train_df = pd.DataFrame({
                'Date': self.X_train.index,
                'Open': [np.nan] * len(self.X_train),  # Would need actual OHLCV data
                'High': [np.nan] * len(self.X_train),
                'Low': [np.nan] * len(self.X_train),
                'Close': self.y_train.values,
                'Volume': [np.nan] * len(self.X_train)
            })
            train_df.set_index('Date', inplace=True)
            
            backtest_results['training'] = run_period_backtest(
                'training', all_signals['training'], train_df
            )
        
        # Backtest validation period
        if 'validation' in all_signals and not all_signals['validation'].empty:
            # Create OHLCV data for validation period
            val_df = pd.DataFrame({
                'Date': self.X_val.index,
                'Open': [np.nan] * len(self.X_val),
                'High': [np.nan] * len(self.X_val),
                'Low': [np.nan] * len(self.X_val),
                'Close': self.y_val.values,
                'Volume': [np.nan] * len(self.X_val)
            })
            val_df.set_index('Date', inplace=True)
            
            backtest_results['validation'] = run_period_backtest(
                'validation', all_signals['validation'], val_df
            )
        
        # Backtest testing period
        if 'testing' in all_signals and not all_signals['testing'].empty and self.y_test is not None:
            # Create OHLCV data for testing period
            test_df = pd.DataFrame({
                'Date': self.X_test.index,
                'Open': [np.nan] * len(self.X_test),
                'High': [np.nan] * len(self.X_test),
                'Low': [np.nan] * len(self.X_test),
                'Close': self.y_test.values,
                'Volume': [np.nan] * len(self.X_test)
            })
            test_df.set_index('Date', inplace=True)
            
            backtest_results['testing'] = run_period_backtest(
                'testing', all_signals['testing'], test_df
            )
        
        # Print summary of backtest results
        print("\nBacktest Performance Summary:")
        for period, results in backtest_results.items():
            if results and 'metrics' in results:
                metrics = results['metrics']
                print(f"\n{period.capitalize()} Period:")
                print(f"  Total Return: {metrics['total_return']:.2%}")
                print(f"  Annualized Return: {metrics['annualized_return']:.2%}")
                print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        
        return backtest_results

    def backtest_trading_strategy(
        self,
        model,
        Long_allowed: bool = True,
        Short_allowed: bool = False,
        execution_threshold: float = 0.001,
        strategy_type: str = "moc",
        target_isClose:bool=False,
        write_OHLCV_output: bool = False,
        **backtest_params
    ):
        
        print("\nBacktesting trading strategy based on model predictions...")
        list_result_dict=[]
        for period in ['training','validating','testing']:
            if period=='training':
                df_OHLCV=self.X_train
            elif period=='validating':
                df_OHLCV=self.X_val
            elif period=='testing':
                df_OHLCV=self.X_test

            # predict from best model
            predictions = predict_model(model, data=df_OHLCV)
            # predictions.to_csv("predictions.csv")
            ##### regression############################
            # convert from Log return to pct return
            if not target_isClose:
                # df['target'] = np.log(df['Close'].shift(-2) / df['Close'].shift(-1))*100
                predicted_pct_return = np.exp(predictions['prediction_label']/100)-1 
                #genrate signals
                signals = pd.Series(0, index=df_OHLCV.index)
                if Long_allowed:
                    signals[predicted_pct_return > execution_threshold]= 1
                if Short_allowed:
                    signals[predicted_pct_return < execution_threshold]= -1
                
                df_OHLCV['prediction_label'] = predictions['prediction_label']
                df_OHLCV['predicted_pct_return'] = predicted_pct_return
                predicted_close=None
            else:
                predicted_close = predictions['prediction_label']
                signals=None
            
            # Run backtest using your existing backtester
            backtester1=backtest_strategy(pct_transaction_fee=0.0)
            result_dict = backtester1.run_backtest2_SL_TP(
                Long_allowed=Long_allowed,
                Short_allowed=Short_allowed,
                df_OHLCV=df_OHLCV,
                signals=signals,
                predicted_close=predicted_close,
                strategy_name="pycaret",
                strategy_type=strategy_type,
                execution_threshold=execution_threshold,
                write_OHLCV_output=write_OHLCV_output,
                **backtest_params
            )
            list_result_dict.append({period:result_dict['metrics']})
        return list_result_dict
        
    def get_feature_importance(self, top_n: int = 10) -> pd.Series:
        """Get top feature importance scores"""
        if self.feature_importance is None:
            print("Feature importance not available. Run finalize_and_evaluate() first.")
            return pd.Series()
        
        return self.feature_importance.head(top_n)    
 
import sys, pathlib, pandas as pd, matplotlib.pyplot as plt, numpy as np
project_root = pathlib.Path('.')
sys.path.append(str(project_root))
from utils.config_utils import load_config
config=load_config('ta_config.yaml')
from math import sqrt
from typing import Dict, Union, Optional
from utils_strategies.signals_utils import RawSignalGenerator
from utils_strategies.signals_TAlib_utils import TALibSignalGenerator

class backtest_strategy:
    def __init__(self, pct_transaction_fee: float = 0.1/100):
        self.backtest_result={}
        self.pct_transaction_fee = pct_transaction_fee # 0.1%       

    def run_backtest2_SL_TP(
            self,
            df_OHLCV: pd.DataFrame,
            horizon_days: int = 1,
            predicted_close: Optional[pd.Series] = None,
            Long_allowed: bool = True,
            Short_allowed: bool = False,
            signals: Optional[pd.Series] = None,
            strategy_name: str = "ma_crossover",
            strategy_type: str = "moc",
            revert_position: bool = False,
            risk_free_rate: float = 0.02,
            include_metrics: bool = True,
            # EXECUTION PARAMETERS
            execution_threshold: float = 0.005,  # 0.5% minimum edge
            max_gap: float = 0.02,               # 2.0% max gap before skipping trade
            # STOP LOSS AND TAKE PROFIT PARAMETERS
            stop_loss_pct: Optional[float] = None,  # e.g., 0.02 for 2% stop loss
            take_profit_pct: Optional[float] = None,  # e.g., 0.04 for 4% take profit,
            write_OHLCV_output: bool = False
        ) -> Dict[str, Union[pd.Series, Dict]]:
        
        # ----------------------------
        # STEP 0: PREPARE DATA & VALIDATE INPUTS
        # ----------------------------
        # Prepare DataFrame
        df = df_OHLCV.copy()
        df.columns = df.columns.str.capitalize()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        # Validate input data
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain '{col}' column")
        
        # Set up price series
        close = df['Close']
        
        # For intraday: Enter at next day's open, exit at next day's close
        # For MOC: Enter at next day's close, exit at day after next day's close
        target_open = df['Open'].shift(-horizon_days)
        target_close = df['Close'].shift(-horizon_days)
        
        # Additional shift needed for MOC strategy exit
        moc_exit_close = df['Close'].shift(-horizon_days - 1)
        
        # Validate strategy type
        if strategy_type not in ["moc", "intraday"]:
            raise ValueError("strategy_type must be 'moc' or 'intraday'")
        
        # Validate signal inputs (must have one or the other, but not both)
        if predicted_close is not None and signals is not None:
            raise ValueError("Provide either predicted_close OR signals, not both")
        
        # ----------------------------
        # STEP 1: GENERATE EXECUTION DECISIONS
        # ----------------------------
        # Initialize position series and execution reasons
        position = pd.Series(0, index=close.index)
        execution_reasons = pd.Series("NO_SIGNAL", index=close.index)

        # CASE 1: Direct signals provided (bypass "second look" logic)
        if signals is not None:
            # Validate signals
            if not isinstance(signals, pd.Series):
                raise TypeError("signals must be a pandas Series")
            
            if not close.index.equals(signals.index):
                raise ValueError("close and signals must have identical datetime index")
            
            # Validate signal values
            if not set(signals.unique()).issubset({-1, 0, 1}):
                raise ValueError("signals must be -1, 0, or 1 for direct signal input")
            
            position = signals.shift(1)
            execution_reasons = pd.Series("DIRECT_SIGNAL", index=close.index).shift(1)
        
        # CASE 2: Predicted close prices provided and not directional_trade
        elif predicted_close is not None :
            # Validate prediction alignment
            if not close.index.equals(predicted_close.index):
                raise ValueError("close and predicted_close must have identical datetime index")
            
            # Calculate gap from previous close (for gap filter - intraday only)
            gap_from_prev = pd.Series(np.nan, index=close.index)
            valid_gap = pd.Series(True, index=close.index)
            
            if strategy_type == "intraday":
                gap_from_prev = target_open / close - 1
                valid_gap = abs(gap_from_prev) <= max_gap
            
            # Calculate predicted return (strategy-specific)
            predicted_return = pd.Series(np.nan, index=close.index)
            
            if strategy_type == "moc":
                # For MOC: Predict day t+1 close relative to day t close
                predicted_return = predicted_close / close - 1
            else:  # intraday
                # For intraday: Predict day t+1 close relative to day t+1 open
                predicted_return = predicted_close / target_open - 1
            
            # Apply gap filter - skip trades with excessive gaps (intraday only)
            if strategy_type == "intraday":
                # LONG positions (only where gap is valid and predicted return > threshold)
                if Long_allowed: 
                    long_mask = valid_gap & (predicted_return > execution_threshold)
                    position[long_mask] = 1
                    execution_reasons[long_mask] = "LONG: FULL POSITION"
                
                # SHORT positions (only where gap is valid and predicted return < -threshold)
                if Short_allowed:
                    short_mask = valid_gap & (predicted_return < -execution_threshold)
                    position[short_mask] = -1
                    execution_reasons[short_mask] = "SHORT: FULL POSITION"
                    
                # Mark skipped trades due to gap
                gap_skip = ~valid_gap & (predicted_close.notna())
                execution_reasons[gap_skip] = "SKIPPED: Excessive gap"
                
                # Mark skipped trades due to insufficient edge
                edge_skip = valid_gap & (predicted_close.notna()) & (abs(predicted_return) <= execution_threshold)
                execution_reasons[edge_skip] = "SKIPPED: Insufficient edge"
            else:  # MOC strategy
                # For MOC, we need to shift positions forward by 1 day
                # Signal generated on day t → Position entered on day t+1
                if Long_allowed: position[predicted_return > execution_threshold] = 1
                if Short_allowed: position[predicted_return < -execution_threshold] = -1
                # Shift positions forward by 1 day for MOC strategy
                position = position.shift(1)
                
                # Update execution reasons (shifted as well)
                execution_reasons[predicted_return > execution_threshold] = "LONG: FULL POSITION"
                # execution_reasons[predicted_return < -execution_threshold] = "SHORT: FULL POSITION"
                execution_reasons = execution_reasons.shift(1)
        
        # Apply position reversal if requested
        if revert_position:
            position *= -1
        
        # Fill NaN values with 0 (no position)
        position = position.fillna(0)

        if strategy_type == "moc":

            enter_price = target_close  # Tomorrow's close (day t+1)
            exit_price = moc_exit_close  # Day after tomorrow's close (day t+2)

            if stop_loss_pct is not None or take_profit_pct is not None:
                print("Stop loss and take profit are less meaningful for MOC strategy. "
                            "Consider using intraday strategy for stop loss implementation.")
        
        elif strategy_type == "intraday":

            enter_price = target_open  # Next day's open
            exit_price = target_close  # Next day's close (default if no stop/take profit hit)
            
            # Initialize stop/take profit tracking
            stop_loss_hit = pd.Series(False, index=close.index)
            take_profit_hit = pd.Series(False, index=close.index)
            exit_price_with_sl_tp = target_close.copy()
            
            # Apply stop loss and take profit if specified
            if stop_loss_pct is not None or take_profit_pct is not None:
                for i in range(len(position)):
                    pos = position.iloc[i]
                    if pos == 0:  # No position, skip
                        continue
                    
                    entry = enter_price.iloc[i]
                    if pd.isna(entry):
                        continue
                    
                    # Determine stop loss and take profit prices
                    if pos > 0:  # Long position
                        if stop_loss_pct is not None:
                            stop_loss_price = entry * (1 - stop_loss_pct)
                        if take_profit_pct is not None:
                            take_profit_price = entry * (1 + take_profit_pct)
                    else:  # Short position
                        if stop_loss_pct is not None:
                            stop_loss_price = entry * (1 + stop_loss_pct)
                        if take_profit_pct is not None:
                            take_profit_price = entry * (1 - take_profit_pct)
                    
                    # Get next day's price data
                    high = df['High'].iloc[i]
                    low = df['Low'].iloc[i]
                    
                    # Check if stop loss was hit
                    stop_hit = False
                    if stop_loss_pct is not None:
                        if (pos > 0 and low <= stop_loss_price) or (pos < 0 and high >= stop_loss_price):
                            stop_hit = True
                            stop_loss_hit.iloc[i] = True
                    
                    # Check if take profit was hit
                    tp_hit = False
                    if take_profit_pct is not None:
                        if (pos > 0 and high >= take_profit_price) or (pos < 0 and low <= take_profit_price):
                            # Only count as take profit hit if stop loss wasn't hit first
                            if not stop_hit:
                                tp_hit = True
                                take_profit_hit.iloc[i] = True
                    
                    # Determine actual exit price
                    if stop_hit:
                        # Conservative assumption: exit at stop loss price
                        exit_price_with_sl_tp.iloc[i] = stop_loss_price
                    elif tp_hit:
                        # Conservative assumption: exit at take profit price
                        exit_price_with_sl_tp.iloc[i] = take_profit_price
                    else:
                        # No stop/take profit hit, exit at close
                        exit_price_with_sl_tp.iloc[i] = df['Close'].iloc[i]
                
                # Update exit price with stop/take profit logic
                exit_price = exit_price_with_sl_tp
        
        # Calculate returns
        if strategy_type == "moc":# Shift  to align with moc position
            exit_price = exit_price.shift(1)  
            enter_price = enter_price.shift(1)
            if stop_loss_pct is not None or take_profit_pct is not None:
                exit_price_with_sl_tp = exit_price_with_sl_tp.shift(1)

        gross_change = exit_price / enter_price - 1
        gross_return = position * gross_change
        gross_return = gross_return.fillna(0)
        
        # Apply transaction fees (only on executed positions)
        transaction_costs = abs(position) * self.pct_transaction_fee
        strategy_return = gross_return - transaction_costs
        strategy_return = strategy_return.fillna(0)
        
        # ----------------------------
        # STEP 3: CALCULATE CUMULATIVE RETURNS
        # ----------------------------
        cumsum_gross_return = gross_return.cumsum()# (1 + gross_return).cumprod() - 1 

        compound_gross_return =  (1 + gross_return).cumprod() - 1
        cumulative_equity = (1 + strategy_return).cumprod()
        
        # ----------------------------
        # STEP 4: CALCULATE PERFORMANCE METRICS
        # ----------------------------
        metrics = {}
        if include_metrics:
            # Total Return
            total_return = cumulative_equity.iloc[-1] - 1
            
            # Annualized Return (252 trading days)
            n_days = len(cumulative_equity)
            annualized_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
            
            # CAGR (Compound Annual Growth Rate)
            start_date = close.index[0]
            end_date = close.index[-1]
            n_years = (end_date - start_date).days / 365.25
            cagr = (cumulative_equity.iloc[-1]) ** (1/n_years) - 1 if n_years > 0 else 0
            
            # Sharpe Ratio (annualized)
            daily_rf = (1 + risk_free_rate) ** (1/252) - 1
            excess_daily_return = strategy_return - daily_rf
            
            if excess_daily_return.std() == 0:
                sharpe_ratio = np.inf if excess_daily_return.mean() > 0 else -np.inf
            else:
                sharpe_ratio = (excess_daily_return.mean() / excess_daily_return.std()) * np.sqrt(252)
            
            # Sortino Ratio (downside deviation only)
            negative_returns = strategy_return[strategy_return < 0]
            if len(negative_returns) > 0:
                downside_dev = np.sqrt(np.mean(negative_returns**2))
                sortino_ratio = (excess_daily_return.mean() / downside_dev) * np.sqrt(252) if downside_dev > 0 else np.inf
            else:
                sortino_ratio = np.inf
            
            # Max Drawdown
            rolling_max = cumulative_equity.cummax()
            drawdown = (rolling_max - cumulative_equity) / rolling_max
            max_drawdown = drawdown.max()
            
            # Calmar Ratio
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else np.inf
            
            # Trade statistics
            trading_days = sum((position != 0).astype(int))
            trade_rate = trading_days / n_days
            
            # Win days (only count executed trades)
            win_days = sum((strategy_return > 0).astype(int))
            win_rate = win_days / trading_days if trading_days > 0 else 0
            
            # Skipped trades (only for predicted_close strategies)
            gap_skips = 0
            edge_skips = 0
            skipped_trades = 0
            
            if predicted_close is not None and strategy_type == "intraday":
                gap_skips = sum(execution_reasons.str.startswith("SKIPPED: Excessive gap"))
                edge_skips = sum(execution_reasons.str.startswith("SKIPPED: Insufficient edge"))
                skipped_trades = gap_skips + edge_skips
            
            # Profit Factor
            gross_profits = strategy_return[strategy_return > 0].sum()
            gross_losses = abs(strategy_return[strategy_return < 0].sum())
            profit_factor = gross_profits / gross_losses if gross_losses != 0 else np.inf
            
            # Volatility (annualized)
            volatility = strategy_return.std() * np.sqrt(252)
            
            # Omega Ratio (threshold = risk-free rate)
            threshold = daily_rf
            above_threshold = strategy_return[strategy_return > threshold]
            below_threshold = strategy_return[strategy_return <= threshold]
            
            if len(below_threshold) == 0:
                omega_ratio = np.inf
            else:
                omega_ratio = above_threshold.sum() / abs(below_threshold.sum())
            
            # Strategy statistics
            metrics = {
                'strategy_name': strategy_name,
                'strategy_type': strategy_type,
                'n_days': n_days,
                'trading_days': trading_days,
                'skipped_trades': skipped_trades,
                'gap_skips': gap_skips,
                'edge_skips': edge_skips,
                'trade_rate': trade_rate,
                'win_days': win_days,
                'win_rate': win_rate,
                'compound_gross_return': compound_gross_return.iloc[-1],
                'total_return': cumsum_gross_return.iloc[-1],
                'annualized_return': annualized_return,
                'cagr': cagr,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'omega_ratio': omega_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'profit_factor': profit_factor,
                'volatility': volatility,
                'risk_free_rate': risk_free_rate,
                'execution_threshold': execution_threshold,
                'max_gap': max_gap
            }
            
            # Add stop loss/take profit metrics if applicable
            if strategy_type == "intraday" and (stop_loss_pct is not None or take_profit_pct is not None):
                # Calculate stop loss/take profit hit rates
                total_positions = (position != 0).sum()
                sl_hits = stop_loss_hit.sum()
                tp_hits = take_profit_hit.sum()
                neither_hits = total_positions - sl_hits - tp_hits
                
                metrics.update({
                    'stop_loss_pct': stop_loss_pct,
                    'take_profit_pct': take_profit_pct,
                    'stop_loss_hits': sl_hits,
                    'take_profit_hits': tp_hits,
                    'neither_hits': neither_hits,
                    'stop_loss_hit_rate': sl_hits / total_positions if total_positions > 0 else 0,
                    'take_profit_hit_rate': tp_hits / total_positions if total_positions > 0 else 0,
                    'exit_by_stop_loss': sl_hits / total_positions if total_positions > 0 else 0,
                    'exit_by_take_profit': tp_hits / total_positions if total_positions > 0 else 0,
                    'exit_by_close': neither_hits / total_positions if total_positions > 0 else 0
                })
        
        # ----------------------------
        # STEP 5: STORE RESULTS
        # ----------------------------
        df['target_open'] = target_open
        df['target_close'] = target_close
        df['moc_exit_close'] = moc_exit_close
        df['predicted_close'] = predicted_close if predicted_close is not None else pd.Series(np.nan, index=df.index)
        df['signals_input'] = signals if signals is not None else pd.Series(np.nan, index=df.index)
        df['position'] = position
        df['execution_reason'] = execution_reasons
        df['enter_price'] = enter_price
        df['exit_price'] = exit_price
        df['gross_return'] = gross_return
        df['total_return'] = cumsum_gross_return
        df['compound_gross_return']= compound_gross_return
        # Stop loss/take profit tracking
        if strategy_type == "intraday" and (stop_loss_pct is not None or take_profit_pct is not None):
            df['stop_loss_hit'] = stop_loss_hit
            df['take_profit_hit'] = take_profit_hit
            df['exit_price_with_sl_tp'] = exit_price_with_sl_tp
        else:
            df['stop_loss_hit'] = pd.Series(False, index=df.index)
            df['take_profit_hit'] = pd.Series(False, index=df.index)
            df['exit_price_with_sl_tp'] = df['exit_price']

        # ----------------------------
        # STEP 6: RETURN RESULTS
        # ----------------------------
        result = {
            'total_return': cumsum_gross_return,
            'position': position,
            'execution_reasons': execution_reasons
        }
        
        if include_metrics:
            result['metrics'] = metrics
        
        self.backtest_result = result
        datime_str = str(pd.Timestamp.now()).replace("-","").replace(":","").replace(" ","").replace(".","_")
        if write_OHLCV_output :
            df.to_csv(f"df_OHLCV_{strategy_name}_{strategy_type}_{datime_str}.csv")
        return result

    def plot_backtest_results(self,backtest_results, benchmark_returns: Optional[pd.Series] = None, 
                            title: str = "Strategy Performance"):
        
        plt.figure(figsize=(8, 5))
        
        # Plot strategy cumulative return
        plt.plot(backtest_results['cumulative_return'], label="Strategy", linewidth=2.5)
        
        # Plot benchmark if provided
        if benchmark_returns is not None:
            plt.plot(benchmark_returns, label="Benchmark", linestyle="--", alpha=0.7)
        
        # Add metrics as text
        if 'metrics' in backtest_results:
            metrics = backtest_results['metrics']
            text_str = (
                f"Total Return: {metrics['total_return']:.2%}\n"
                f"Annualized: {metrics['annualized_return']:.2%}\n"
                f"Sharpe: {metrics['sharpe_ratio']:.2f}\n"
                f"Max Drawdown: {metrics['max_drawdown']:.2%}"
            )
            plt.annotate(text_str, xy=(0.05, 0.05), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.title(title, fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Cumulative Return", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def print_metrics(self, backtest_result):
        """Print performance metrics in a readable format"""
        metrics = backtest_result['metrics']

        print("\n===== STRATEGY PERFORMANCE METRICS =====")

        print(f"Strategy:              {metrics['strategy_name']}")
        print(f"Strategy Type:         {metrics['strategy_type']}")
        print(f"n_days:                {metrics['n_days']:>.0f}")
        print(f"trading_days:          {metrics['trading_days']:>.0f}")
        print(f"skipped_trades:        {metrics['skipped_trades']:>.0f}")
        print(f"  - Gap skips:         {metrics['gap_skips']:>.0f}")
        print(f"  - Edge skips:        {metrics['edge_skips']:>.0f}")
        print(f"trade_rate:            {metrics['trade_rate']:>10.2%}")
        print(f"win_days:              {metrics['win_days']:>.0f}")
        print(f"win_rate:              {metrics['win_rate']:>10.2%}")
        print(f"Total Gross Return:    {metrics['total_gross_return']:>10.2%}")
        print(f"Total Return:          {metrics['total_return']:>10.2%}")
        print(f"Annualized Return:     {metrics['annualized_return']:>10.2%}")
        print(f"CAGR:                  {metrics['cagr']:>10.2%}")
        print(f"Sharpe Ratio:          {metrics['sharpe_ratio']:>10.2f}")
        print(f"Sortino Ratio:         {metrics['sortino_ratio']:>10.2f}")
        print(f"Max Drawdown:          {metrics['max_drawdown']:>10.2%}")
        print(f"Calmar Ratio:          {metrics['calmar_ratio']:>10.2f}")
        print(f"Profit Factor:         {metrics['profit_factor']:>10.2f}")
        print(f"Volatility (ann.):     {metrics['volatility']:>10.2%}")
        print(f"Risk-Free Rate:        {metrics['risk_free_rate']:>10.2%}")
        print(f"Execution Threshold:   {metrics['execution_threshold']:>10.2%}")
        print(f"Max Gap:               {metrics['max_gap']:>10.2%}")
        print("========================================\n")

    def compare_strategies_metrics(self, strategies: Dict[str, Dict] = {'strategy_name1': {'metrics':{}},
                                                                        'strategy_name2': {'metrics':{}}
                                                                        }):

        # Extract metrics from each strategy
        comparison = []
        for name, results in strategies.items():
            if 'metrics' in results:
                metrics = results['metrics']
                comparison.append({
                    'Strategy Type': metrics['strategy_type'],
                    'Strategy Name': metrics['strategy_name'],
                    'Strategy': name,
                    'Total Return': metrics['total_return'],
                    'Annualized': metrics['annualized_return'],
                    'Sharpe': metrics['sharpe_ratio'],
                    'Max Drawdown': metrics['max_drawdown'],
                    'Win Rate': metrics['win_rate'],
                    'Profit Factor': metrics['profit_factor']
                })
        
        # Convert to DataFrame and format
        df = pd.DataFrame(comparison)
        df.sort_values(by=['Sharpe'],ascending=False,inplace=True)
        df.reset_index(drop=True,inplace=True)
        df['Total Return'] = df['Total Return'].map('{:.2%}'.format)
        df['Annualized'] = df['Annualized'].map('{:.2%}'.format)
        df['Max Drawdown'] = df['Max Drawdown'].map('{:.2%}'.format)
        df['Win Rate'] = df['Win Rate'].map('{:.1f}%'.format)
        df['Sharpe'] = df['Sharpe'].map('{:.2f}'.format)
        df['Profit Factor'] = df['Profit Factor'].map('{:.2f}'.format)
        
        print("\n===== STRATEGY COMPARISON =====")
        print(df.to_string())
        print("================================\n")

    def run_raw_signals_backtest(self,df_OHLCV: pd.DataFrame, horizon_days: int = 1,strategy_type = 'moc',revert_position=False):

        print("="*60)
        print("COMPLETE STRATEGY DEVELOPMENT WORKFLOW")
        print("="*60)
        
        # ----------------------------
        # STEP 1: SIGNAL GENERATION
        # ----------------------------
        print("\n[1/4] Generating trading signals...")
        
        # Initialize signal generator
        signal_gen = RawSignalGenerator(df_OHLCV.copy())
        
        # Generate multiple strategies
        ma_crossover_signals = signal_gen.generate_ma_crossover_signals(
            short_window=50, 
            long_window=200,
            signal_name="ma_crossover"
        )
        # ----------------------------
        # STEP 2: BACKTESTING
        # ----------------------------
        print("\n[2/4] Running backtests...")
        
        # Initialize backtester
        backtester = backtest_strategy(
            df_OHLCV=df_OHLCV.copy(),
            pct_transaction_fee=self.pct_transaction_fee,  # 0.1% transaction fee
            revert_position=revert_position
        )

        ma_results = backtester.run_backtest(
            horizon_days=horizon_days,
            signals=ma_crossover_signals,
            strategy_type=strategy_type,
            risk_free_rate=0.02
        )

        # rsi_results = backtester.run_backtest(
        #     horizon_days=horizon_days,
        #     signals=rsi_signals,
        #     strategy_type=strategy_type,
        #     risk_free_rate=0.02
        # )
        
        # bb_results = backtester.run_backtest(
        #     horizon_days=horizon_days,
        #     signals=bollinger_bands_signals,
        #     strategy_type=strategy_type,
        #     risk_free_rate=0.02
        # )
        
        # combined_results = backtester.run_backtest(
        #     horizon_days=horizon_days,
        #     signals=combined_signals,
        #     strategy_type=strategy_type,
        #     risk_free_rate=0.02
        # )
        
        # ----------------------------
        # STEP 3: PERFORMANCE ANALYSIS
        # ----------------------------
        print("\n[3/4] Analyzing performance...")
        
        # Print metrics for all strategies
        print("\nMA CROSSOVER STRATEGY:")
        backtester.print_metrics(ma_results)
        
        # print("\nRSI STRATEGY:")
        # backtester.print_metrics(rsi_results)
        
        # print("\nBOLLINGER BANDS STRATEGY:")
        # backtester.print_metrics(bb_results)
        
        # print("\nCOMBINED STRATEGY:")
        # backtester.print_metrics(combined_results)
        
        # # Compare strategies
        # backtester.compare_strategies_metrics({
        #     "MA Crossover": ma_results,
        #     "RSI": rsi_results,
        #     "Bollinger Bands": bb_results,
        #     "Combined": combined_results
        # })
        
        # ----------------------------
        # STEP 4: VISUALIZATION
        # ----------------------------
        print("\n[4/4] Generating performance charts...")
        
        # Plot all strategies
        plt.figure(figsize=(8, 5))
        plt.plot(ma_results['cumulative_return'], label="MA Crossover", linewidth=2)
        # plt.plot(rsi_results['cumulative_return'], label="RSI Strategy", linewidth=2)
        # plt.plot(bb_results['cumulative_return'], label="Bollinger Bands", linewidth=2)
        # plt.plot(combined_results['cumulative_return'], label="Combined Strategy", linewidth=2)
        
        plt.title(f"Strategy = {strategy_type} | revert_position = {revert_position}", fontsize=14)
        plt.xlabel("Date", fontsize=10)
        plt.ylabel("Cumulative Return", fontsize=10)
        plt.legend(fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=8, rotation=45)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.show()
        
        # Plot signal effectiveness
        plt.figure(figsize=(8, 5))
        
        # For MA Crossover
        plt.subplot(2, 2, 1)
        plt.scatter(ma_crossover_signals, backtester.df_OHLCV['daily_return'], alpha=0.5)
        plt.axvline(0, color='k', linestyle='-', alpha=0.3)
        plt.axhline(0, color='k', linestyle='-', alpha=0.3)
        plt.title("MA Crossover Effectiveness", fontsize=10)
        plt.xlabel("Signal")
        plt.ylabel("Daily Return")
        
        # For RSI
        # plt.subplot(2, 2, 2)
        # plt.scatter(rsi_signals, backtester.df_OHLCV['daily_return'], alpha=0.5)
        # plt.axvline(0, color='k', linestyle='-', alpha=0.3)
        # plt.axhline(0, color='k', linestyle='-', alpha=0.3)
        # plt.title("RSI Effectiveness", fontsize=10)
        # plt.xlabel("Signal")
        
        # For Bollinger Bands
        # plt.subplot(2, 2, 3)
        # plt.scatter(bollinger_bands_signals, backtester.df_OHLCV['daily_return'], alpha=0.5)
        # plt.axvline(0, color='k', linestyle='-', alpha=0.3)
        # plt.axhline(0, color='k', linestyle='-', alpha=0.3)
        # plt.title("Bollinger Bands Effectiveness", fontsize=10)
        # plt.xlabel("Signal")
        # plt.ylabel("Daily Return")
        
        # For Combined
        # plt.subplot(2, 2, 4)
        # plt.scatter(combined_signals, backtester.df_OHLCV['daily_return'], alpha=0.5)
        # plt.axvline(0, color='k', linestyle='-', alpha=0.3)
        # plt.axhline(0, color='k', linestyle='-', alpha=0.3)
        # plt.title("Combined Effectiveness", fontsize=10)
        # plt.xlabel("Signal")
        
        # plt.tight_layout()
        # plt.suptitle("Signal vs. Next Day Return", fontsize=12)
        # plt.subplots_adjust(top=0.88)
        # plt.show()
        
        print("\nWorkflow complete! Review the results to understand strategy performance.")
        return {
            'signal_generator': signal_gen,
            'backtester': backtester,
            'ma_results': ma_results,
            # 'rsi_results': rsi_results,
            # 'bb_results': bb_results,
            # 'combined_results': combined_results
        }
  
    def run_TAlib_Signals_backtest(self, df_OHLCV: pd.DataFrame,ta_paras_dict:dict = None, horizon_days: int = 1,strategy_type = 'moc',revert_position=False):

        print("="*70)
        print("COMPREHENSIVE BACKTESTING WITH TA-LIB SIGNALS")
        print("="*70)
        
        # ----------------------------
        # STEP 1: SIGNAL GENERATION
        # ----------------------------
        print("\n[1/4] Generating TA-Lib signals...")
        
        # Initialize signal generator
        signal_gen = TALibSignalGenerator(df_OHLCV)
        ta_default_paras = {'momentum_params':config['short_term']['momentum'],
                            'volume_params':config['short_term']['volume'],
                            'volatility_params':config['short_term']['volatility'],
                            'overlap_params':config['short_term']['overlap'],
                            'cycle_params':config['short_term']['cycle'],
                            'price_params':config['short_term']['price'],
                            'statistic_params':config['short_term']['statistic']}
        if ta_paras_dict:
            ta_default_paras.update(ta_paras_dict)
        signal_gen.generate_all_signals(
            **ta_default_paras
        )
        
        # ----------------------------
        # STEP 2: VALIDATION & ANALYSIS
        # ----------------------------
        print("\n[2/4] Validating and analyzing signals...")

        validation = signal_gen.validate_signals()

        # Analyze signal performance
        analysis = signal_gen.analyze_signal_performance()
        
        # Sort signals by Sharpe ratio
        sorted_signals = sorted(
            analysis.items(), 
            key=lambda x: x[1]['sharpe_ratio'], 
            reverse=True
        )
        
        # ----------------------------
        # STEP 3: BACKTESTING
        # ----------------------------
        top_n_signals = 30
        print(f"\n[3/4] Running backtests for {top_n_signals} signals...")
        
        top_signal_names = [name for name, _ in sorted_signals[:top_n_signals]]
        backtest_results = {}
        
        for signal_name in top_signal_names:
            signal = signal_gen.signals[signal_name]
            result = self.run_backtest2(
                df_OHLCV=df_OHLCV,
                horizon_days=horizon_days,
                signals=signal,
                strategy_type=strategy_type,
                strategy_name=signal_name,
                revert_position=revert_position,
                risk_free_rate=0.02
            )
            backtest_results[signal_name] = result
        
        # ----------------------------
        # STEP 4: PERFORMANCE ANALYSIS
        # ----------------------------
        # Compare strategies
        print("\nSTRATEGY COMPARISON:")
        self.compare_strategies_metrics({
            name: backtest_results[name] 
            for name in top_signal_names
        })

        print("\nComprehensive backtest complete! Review the results to identify the best strategy.")
    
    def run_TALib_valid_signals_backtest(self, df_OHLCV: pd.DataFrame,ta_paras_dict:dict = None,revert_position:bool=False
    ) -> Dict[str, Dict]:

        # Generate signalss
                # Initialize signal generator
        signal_gen = TALibSignalGenerator(df_OHLCV)
        ta_default_paras = {'momentum_params':config['short_term']['momentum'],
                            'volume_params':config['short_term']['volume'],
                            'volatility_params':config['short_term']['volatility'],
                            'overlap_params':config['short_term']['overlap'],
                            'cycle_params':config['short_term']['cycle'],
                            'price_params':config['short_term']['price'],
                            'statistic_params':config['short_term']['statistic']}
        if ta_paras_dict:
            ta_default_paras.update(ta_paras_dict)
        signals = signal_gen.generate_all_signals(
            **ta_default_paras
        )
        
        # STAGE 1: Signal Validation (eliminate fundamentally flawed signals)
        validation_results = signal_gen.validate_signals()
        valid_signals = {
            name: signal for name, signal in signals.items() 
            # if validation_results[name]['is_valid']
        }
        
        print(f"Stage 1: Filtered {len(signals)} signals down to {len(valid_signals)} valid signals")
        
        # STAGE 2: Performance Analysis (select signals with meaningful performance)
        performance_results = signal_gen.analyze_signal_performance()
        qualified_signals = {
            name: signal for name, signal in valid_signals.items()
            if performance_results[name]['sharpe_ratio'] > 0.3 and 
            performance_results[name]['active_signals'] > 30
        }
        
        ## take all signals
        qualified_signals = valid_signals
        print(f"Stage 2: Filtered {len(valid_signals)} valid signals down to {len(qualified_signals)} qualified signals")
        
        # STAGE 3: Backtesting (run comprehensive backtests on qualified signals)
        backtest_results = {}
        for signal_name, signal in qualified_signals.items():
            
            # Intraday strategy
            intraday_results = self.run_backtest2(
                df_OHLCV=df_OHLCV,
                signals=signal,
                revert_position=revert_position,
                strategy_name=signal_name,
                strategy_type="intraday",
            )
           
            # MOC strategy
            moc_results = self.run_backtest2(
                df_OHLCV=df_OHLCV,
                signals=signal,
                revert_position=revert_position,
                strategy_name=signal_name,
                strategy_type="moc",
            )
            
            backtest_results[signal_name] = {
                'intraday': intraday_results['metrics'],
                'moc': moc_results['metrics']
            }
        
        # ----------------------------
        # STEP 3: CREATE COMPARISON TABLE
        # ----------------------------
        print("\n" + "="*70)
        print("STEP 3: CREATING STRATEGY COMPARISON TABLE")
        print("="*70)
        
        # Prepare comparison data
        comparison_data = []
        
        for signal_name, results in backtest_results.items():
            # Intraday strategy metrics
            intraday = results['intraday']
            comparison_data.append({
                'Signal Name': signal_name,
                'Strategy Type': 'Intraday',
                'Total Return': intraday['total_return'],
                'Annualized Return': intraday['annualized_return'],
                'CAGR': intraday['cagr'],
                'Sharpe Ratio': intraday['sharpe_ratio'],
                'Sortino Ratio': intraday['sortino_ratio'],
                'Win Rate': intraday['win_rate'],
                'Max Drawdown': intraday['max_drawdown'],
                'Calmar Ratio': intraday['calmar_ratio'],
                'Profit Factor': intraday['profit_factor'],
                'Volatility': intraday['volatility'],
                'Trade Rate': intraday['trade_rate'],
                'Skipped Trades': intraday['skipped_trades'],
                'Total Days': intraday['n_days'],
                'Trading Days': intraday['trading_days']
            })
            
            # MOC strategy metrics
            moc = results['moc']
            comparison_data.append({
                'Signal Name': signal_name,
                'Strategy Type': 'MOC',
                'Total Return': moc['total_return'],
                'Annualized Return': moc['annualized_return'],
                'CAGR': moc['cagr'],
                'Sharpe Ratio': moc['sharpe_ratio'],
                'Sortino Ratio': moc['sortino_ratio'],
                'Win Rate': moc['win_rate'],
                'Max Drawdown': moc['max_drawdown'],
                'Calmar Ratio': moc['calmar_ratio'],
                'Profit Factor': moc['profit_factor'],
                'Volatility': moc['volatility'],
                'Trade Rate': moc['trade_rate'],
                'Skipped Trades': 0,  # MOC doesn't skip trades based on gap
                'Total Days': moc['n_days'],
                'Trading Days': moc['trading_days']
            })
        
        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate risk-adjusted metrics
        comparison_df['Return/Drawdown'] = comparison_df['Annualized Return'] / comparison_df['Max Drawdown']
        comparison_df['Return/Volatility'] = comparison_df['Annualized Return'] / comparison_df['Volatility']
        
        # Format for display
        display_df = comparison_df.copy()
        
        # Format numeric columns
        numeric_cols = [
            'Total Return', 'Annualized Return', 'CAGR', 'Sharpe Ratio', 'Sortino Ratio',
            'Win Rate', 'Max Drawdown', 'Calmar Ratio', 'Profit Factor', 'Volatility',
            'Trade Rate', 'Return/Drawdown', 'Return/Volatility'
        ]
        
        for col in numeric_cols:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if x < 10 else f"{x:.2f}")
        
        # Sort by Sharpe Ratio (primary metric for risk-adjusted performance)
        display_df = display_df.sort_values('Sharpe Ratio', ascending=False)
        
        # Add ranking
        display_df['Rank'] = range(1, len(display_df) + 1)
        
        # Reorder columns for better readability
        column_order = [
            'Rank', 'Signal Name', 'Strategy Type', 
            'Total Return', 'Annualized Return', 'CAGR',
            'Sharpe Ratio', 'Sortino Ratio',
            'Win Rate', 'Profit Factor',
            'Max Drawdown', 'Calmar Ratio', 'Return/Drawdown',
            'Volatility', 'Return/Volatility',
            'Trade Rate', 'Trading Days', 'Total Days'
        ]
        
        # Ensure all columns exist before reordering
        column_order = [col for col in column_order if col in display_df.columns]
        
        display_df = display_df[column_order]
        
        # ----------------------------
        # STEP 4: IDENTIFY TOP STRATEGIES
        # ----------------------------
        print(f"\nTotal valid strategies analyzed: {len(display_df)}")
        
        # Get top strategies
        top_strategies = display_df.head(5)
        
        print("\nTop 5 Strategies by Sharpe Ratio:")
        print(top_strategies.to_string(index=False))
        
        # Identify strategy with best balance of metrics
        # Weighted score: 40% Sharpe, 20% Calmar, 20% Win Rate, 10% Profit Factor, 10% CAGR
        comparison_df['Weighted Score'] = (
            0.4 * comparison_df['Sharpe Ratio'] +
            0.2 * comparison_df['Calmar Ratio'] +
            0.2 * comparison_df['Win Rate'] +
            0.1 * comparison_df['Profit Factor'] +
            0.1 * comparison_df['CAGR']
        )
        
        best_balanced = comparison_df.loc[comparison_df['Weighted Score'].idxmax()]
        print(f"\nBest Balanced Strategy: {best_balanced['Signal Name']} ({best_balanced['Strategy Type']})")
        print(f"Weighted Score: {best_balanced['Weighted Score']:.4f}")
        
        # ----------------------------
        # STEP 5: GENERATE DETAILED REPORT
        # ----------------------------
        print("\n" + "="*70)
        print("STEP 4: DETAILED STRATEGY ANALYSIS")
        print("="*70)
        
        # Find the signal with the highest Sharpe for each strategy type
        intraday_top = display_df[display_df['Strategy Type'] == 'Intraday'].iloc[0]
        moc_top = display_df[display_df['Strategy Type'] == 'MOC'].iloc[0]
        
        print("\nIntraday Strategy Leader:")
        print(f"Signal: {intraday_top['Signal Name']}")
        print(f"Sharpe Ratio: {intraday_top['Sharpe Ratio']}")
        print(f"Annualized Return: {intraday_top['Annualized Return']}")
        print(f"Max Drawdown: {intraday_top['Max Drawdown']}")
        print(f"Win Rate: {intraday_top['Win Rate']}")
        
        print("\nMOC Strategy Leader:")
        print(f"Signal: {moc_top['Signal Name']}")
        print(f"Sharpe Ratio: {moc_top['Sharpe Ratio']}")
        print(f"Annualized Return: {moc_top['Annualized Return']}")
        print(f"Max Drawdown: {moc_top['Max Drawdown']}")
        print(f"Win Rate: {moc_top['Win Rate']}")
        
        # ----------------------------
        # STEP 6: RETURN COMPARISON TABLE
        # ----------------------------
        comparison_df=comparison_df.sort_values(by=['Sharpe Ratio'],ascending=False).reset_index(drop=True)
        return comparison_df
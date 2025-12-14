import numpy as np
import pandas as pd
from typing import Dict, Callable, Union, Optional, List, Tuple
import warnings

class RawSignalGenerator:
    
    def __init__(self, df_OHLCV: pd.DataFrame):
        # Validate input data
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df_OHLCV.columns:
                raise ValueError(f"DataFrame must contain '{col}' column")
        
        # Store and prepare data
        self.df = df_OHLCV.copy()
        self.signals = {}
        self.feature_sets = {}
    
    def generate_ma_crossover_signals(
        self,
        short_window: int = 50,
        long_window: int = 200,
        signal_name: str = "ma_crossover"
    ) -> pd.Series:
        
        # Validate parameters
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
        
        # Calculate moving averages (shift by 1 to prevent look-ahead)
        df = self.df.copy()
        df['ma_short'] = df['Close'].rolling(short_window).mean()#.shift(1)
        df['ma_long'] = df['Close'].rolling(long_window).mean()#.shift(1)
        
        # LONG signal: short MA crosses above long MA
        signals = (df['ma_short'] > df['ma_long']).astype(int)
        
        # Store signals
        self.signals[signal_name] = signals
        return signals
    
    def generate_rsi_signals(
        self,
        rsi_window: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        signal_name: str = "rsi"
    ) -> pd.Series:
        
        # Validate parameters
        if oversold >= overbought:
            raise ValueError("oversold must be less than overbought")
        
        # Calculate RSI (shift by 1 to prevent look-ahead)
        df = self.df.copy()
        delta = df['Close'].diff()
        
        # Calculate gain and loss
        gain = (delta.where(delta > 0, 0)).rolling(rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_window).mean()
        
        # Calculate RS and RSI
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
                
        # Generate signals        
        # LONG signal: RSI crosses below oversold
        signals = (df['rsi'] < oversold).astype(int)
        
        # Store signals
        self.signals[signal_name] = signals
        return signals
    
    def generate_bollinger_bands_signals(
        self,
        window: int = 20,
        num_std: float = 2.0,
        signal_name: str = "bollinger_bands"
    ) -> pd.Series:
        
        # Calculate Bollinger Bands (shift by 1 to prevent look-ahead)
        df = self.df.copy()
        df['sma'] = df['Close'].rolling(window).mean()
        df['std'] = df['Close'].rolling(window).std()
        df['upper_band'] = df['sma'] + (num_std * df['std'])
        df['lower_band'] = df['sma'] - (num_std * df['std'])
        
        # Generate signals
        signals = (df['Close'] < df['lower_band'])

        # Store signals
        self.signals[signal_name] = signals
        return signals
    
    def generate_momentum_signals(
        self,
        window: int = 10,
        threshold: float = 0.02,
        signal_name: str = "momentum"
    ) -> pd.Series:
        
        # Calculate momentum (shift by 1 to prevent look-ahead)
        df = self.df.copy()
        df['momentum'] = df['Close'].pct_change(window)
        
        # Generate signals
        signals = (df['momentum'] > threshold).astype(int)
        # Store signals
        self.signals[signal_name] = signals
        return signals
    
    def generate_mean_reversion_signals(
        self,
        window: int = 20,
        z_threshold: float = 1.5,
        signal_name: str = "mean_reversion"
    ) -> pd.Series:
        
        # Calculate z-score (shift by 1 to prevent look-ahead)
        df = self.df.copy()
        df['sma'] = df['Close'].rolling(window).mean()
        df['std'] = df['Close'].rolling(window).std()
        
        # Calculate z-score
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df['z_score'] = (df['Close'] - df['sma']) / df['std']
        
        # Generate signals
        signals = (df['z_score'] < -z_threshold).astype(int)

        # Store signals
        self.signals[signal_name] = signals
        return signals
    
    def generate_breakout_signals(
        self,
        window: int = 20,
        signal_name: str = "breakout"
    ) -> pd.Series:
        
        # Calculate recent high/low (shift by 1 to prevent look-ahead)
        df = self.df.copy()
        df['recent_high'] = df['High'].rolling(window).max().shift(1)
        df['recent_low'] = df['Low'].rolling(window).min().shift(1)
        
        # Generate signals
        signals = (df['Close'] >= df['recent_high'])
        
        # Store signals
        self.signals[signal_name] = signals
        return signals
    
    def generate_combined_signals(
        self,
        strategy_weights: Dict[str, float],
        signal_name: str = "combined"
    ) -> pd.Series:
        
        # Validate strategy weights
        valid_strategies = [
            'ma_crossover', 'rsi', 'bollinger_bands',
            'momentum', 'mean_reversion', 'breakout'
        ]
        
        # Check for invalid strategies
        for strategy in strategy_weights.keys():
            if strategy not in valid_strategies:
                raise ValueError(f"Invalid strategy: {strategy}. Must be one of {valid_strategies}")
        
        # Check if requested strategies have been generated
        for strategy in strategy_weights.keys():
            if strategy not in self.signals:
                raise ValueError(f"Strategy '{strategy}' has not been generated. Call its method first.")
        
        # Normalize weights to sum to 1
        total_weight = sum(strategy_weights.values())
        if abs(total_weight - 1.0) > 1e-5:
            strategy_weights = {k: v/total_weight for k, v in strategy_weights.items()}
        
        # Generate combined signal
        combined_signal = pd.Series(0, index=self.df.index)
        for strategy, weight in strategy_weights.items():
            combined_signal += self.signals[strategy] * weight
        
        # Generate final signals
        signals = (combined_signal > 0.5).astype(int)
        # Store signals
        self.signals[signal_name] = signals
        return signals
    
    def validate_signals(self, signal_name: str) -> Dict:
        
        if signal_name not in self.signals:
            raise ValueError(f"Signal '{signal_name}' not found. Generate signals first.")
        
        signals = self.signals[signal_name]
        
        # Validation checks
        validation = {
            'has_no_nan': not signals.isna().any(),
            'signal_range': (signals.min(), signals.max()),
            'non_zero_ratio': (signals != 0).mean(),
            'long_ratio': (signals > 0).mean(),
            'short_ratio': (signals < 0).mean(),
            'zero_crossings': (signals != 0) & (signals.shift(1) == 0).sum()
        }
        
        # Flag issues
        issues = []
        if not validation['has_no_nan']:
            issues.append("NaN values in signals (look-ahead bias possible)")
        if validation['non_zero_ratio'] < 0.05:
            issues.append(f"Low signal frequency ({validation['non_zero_ratio']:.1%} < 5%)")
        
        validation['issues'] = issues
        validation['is_valid'] = len(issues) == 0
        
        return validation
     
    def get_all_signals(self) -> Dict[str, pd.Series]:
        """Get all generated signals"""
        return self.signals
    

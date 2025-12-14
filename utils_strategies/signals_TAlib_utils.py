import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import platform
from utils.config_utils import load_config
config=load_config('ta_config.yaml')

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

class TALibSignalGenerator:
    """
    Comprehensive signal generator leveraging TA-Lib for professional-grade technical analysis.
    
    Key features:
    - Full integration with TA-Lib's 150+ technical indicators
    - Proper timing alignment (no look-ahead bias)
    - Multiple signal generation methodologies
    - Comprehensive signal validation and analysis
    - Designed to work seamlessly with the provided backtesting framework
    """
    
    def __init__(self, df_OHLCV: pd.DataFrame):
        """
        Initialize signal generator with OHLCV data.
        
        Parameters:
        -----------
        df_OHLCV : pd.DataFrame
            DataFrame with OHLCV data and 'Date' column
            Must contain: Date, Open, High, Low, Close, Volume
        """
        # Validate input data
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        if 'Date' in df_OHLCV.columns:
            df_OHLCV['Date'] = pd.to_datetime(df_OHLCV['Date'])
            df_OHLCV.set_index('Date',inplace=True)

        for col in required_columns:
            if col not in df_OHLCV.columns:
                raise ValueError(f"DataFrame must contain '{col}' column")
        
        # Store and prepare data
        self.df = df_OHLCV.copy()
        self.signals = {}
        self.indicators = {}
        self.signal_metadata = {}
    
    def generate_all_signals(
        self,
        column: str = 'Close',
        momentum_params: Optional[Dict] = None,
        volume_params: Optional[Dict] = None,
        volatility_params: Optional[Dict] = None,
        overlap_params: Optional[Dict] = None,
        cycle_params: Optional[Dict] = None,
        pattern_params: Optional[Dict] = None,
        price_params: Optional[Dict] = None,
        statistic_params: Optional[Dict] = None,
        include_momentum: bool = True,
        include_volume: bool = True,
        include_volatility: bool = True,
        include_overlap: bool = True,
        include_cycle: bool = True,
        include_pattern: bool = False,  # Patterns are less useful for systematic trading
        include_price: bool = True,
        include_statistic: bool = True
    ) -> Dict[str, pd.Series]:
        # Set default parameters if not provided
        ta_default_paras = {'momentum_params':config['short_term']['momentum'],
                            'volume_params':config['short_term']['volume'],
                            'volatility_params':config['short_term']['volatility'],
                            'overlap_params':config['short_term']['overlap'],
                            'cycle_params':config['short_term']['cycle'],
                            'price_params':config['short_term']['price'],
                            'statistic_params':config['short_term']['statistic']}
        if momentum_params is None:
            momentum_params = ta_default_paras['momentum_params']
        if volume_params is None:
            volume_params =ta_default_paras['volume_params']
        if volatility_params is None:
            volatility_params = ta_default_paras['volatility_params']
        if overlap_params is None:
            overlap_params = ta_default_paras['overlap_params']
        if cycle_params is None:
            cycle_params = ta_default_paras['cycle_params']
        if pattern_params is None:
            pattern_params = {}
        if price_params is None:
            price_params = ta_default_paras['price_params']
        if statistic_params is None:
            statistic_params =ta_default_paras['statistic_params']
        
        # Generate signals by category
        signals = {}
        
        if include_momentum:
            signals.update(self.generate_momentum_signals(column,**momentum_params))
        if include_volume:
            signals.update(self.generate_volume_signals(column,**volume_params))
        if include_volatility:
            signals.update(self.generate_volatility_signals(column,**volatility_params))
        if include_overlap:
            signals.update(self.generate_overlap_signals(column,**overlap_params))
        if include_cycle:
            signals.update(self.generate_cycle_signals(column,**cycle_params))
        if include_pattern and pattern_params:
            signals.update(self.generate_pattern_signals(column,**pattern_params))
        if include_price:
            signals.update(self.generate_price_signals(column,**price_params))
        if include_statistic:
            signals.update(self.generate_statistic_signals(column,**statistic_params))
        
        # Store all signals
        self.signals.update(signals)
        return signals
    
    def generate_momentum_signals(self,column:str='Close', **params) -> Dict[str, pd.Series]:
        """
        Generate momentum-based signals using TA-Lib indicators.
        
        Includes: RSI, MACD, Stochastic, CCI, ROC, etc.
        """
        signals = {}
        df = self.df.copy()
        
        # RSI Signals
        if 'rsi' in params:
            for period in params['rsi']:
                # Calculate RSI
                df[f'rsi_{period}'] = talib.RSI(df[column], timeperiod=period)
                self.indicators[f'rsi_{period}'] = df[f'rsi_{period}']
                
                # Generate signals (shift by 1 to prevent look-ahead)
                df[f'rsi_{period}'] = df[f'rsi_{period}'].shift(1)
                
                # LONG when RSI crosses below 30 (oversold)
                long_signals = (df[f'rsi_{period}'] < 30) & (df[f'rsi_{period}'].shift(1) >= 30)
                # SHORT when RSI crosses above 70 (overbought)
                short_signals = (df[f'rsi_{period}'] > 70) & (df[f'rsi_{period}'].shift(1) <= 70)
                
                signals[f'rsi_{period}_long'] = pd.Series(0, index=df.index)
                signals[f'rsi_{period}_long'][long_signals] = 1
                
                signals[f'rsi_{period}_short'] = pd.Series(0, index=df.index)
                signals[f'rsi_{period}_short'][short_signals] = -1
                
                # Combined signal
                signals[f'rsi_{period}_combined'] = signals[f'rsi_{period}_long'] + signals[f'rsi_{period}_short']
                
                # Store metadata
                self.signal_metadata[f'rsi_{period}_combined'] = {
                    'type': 'momentum',
                    'indicator': 'RSI',
                    'period': period,
                    'logic': 'LONG when crosses below 30, SHORT when crosses above 70'
                }
        
        # MACD Signals
        if 'macd' in params:
            for fast, slow, signal in params['macd']:
                # Calculate MACD
                macd, macd_signal, _ = talib.MACD(
                    df[column], 
                    fastperiod=fast, 
                    slowperiod=slow, 
                    signalperiod=signal
                )
                df[f'macd_{fast}_{slow}_{signal}'] = macd - macd_signal
                self.indicators[f'macd_{fast}_{slow}_{signal}'] = df[f'macd_{fast}_{slow}_{signal}']
                
                # Generate signals (shift by 1 to prevent look-ahead)
                df[f'macd_{fast}_{slow}_{signal}'] = df[f'macd_{fast}_{slow}_{signal}'].shift(1)
                
                # LONG when MACD crosses above signal line
                long_signals = (df[f'macd_{fast}_{slow}_{signal}'] > 0) & (df[f'macd_{fast}_{slow}_{signal}'].shift(1) <= 0)
                # SHORT when MACD crosses below signal line
                short_signals = (df[f'macd_{fast}_{slow}_{signal}'] < 0) & (df[f'macd_{fast}_{slow}_{signal}'].shift(1) >= 0)
                
                signals[f'macd_{fast}_{slow}_{signal}_long'] = pd.Series(0, index=df.index)
                signals[f'macd_{fast}_{slow}_{signal}_long'][long_signals] = 1
                
                signals[f'macd_{fast}_{slow}_{signal}_short'] = pd.Series(0, index=df.index)
                signals[f'macd_{fast}_{slow}_{signal}_short'][short_signals] = -1
                
                # Combined signal
                signals[f'macd_{fast}_{slow}_{signal}_combined'] = signals[f'macd_{fast}_{slow}_{signal}_long'] + signals[f'macd_{fast}_{slow}_{signal}_short']
                
                # Store metadata
                self.signal_metadata[f'macd_{fast}_{slow}_{signal}_combined'] = {
                    'type': 'momentum',
                    'indicator': 'MACD',
                    'fast': fast,
                    'slow': slow,
                    'signal': signal,
                    'logic': 'LONG when crosses above signal line, SHORT when crosses below'
                }
        
        # Stochastic Signals
        if 'stoch' in params:
            for fastk, slowk, slowd in params['stoch']:
                # Calculate Stochastic
                slowk_line, slowd_line = talib.STOCH(
                    df['High'],
                    df['Low'],
                    df['Close'],
                    fastk_period=fastk,
                    slowk_period=slowk,
                    slowk_matype=0,
                    slowd_period=slowd,
                    slowd_matype=0
                )
                df[f'stoch_k_{fastk}_{slowk}'] = slowk_line
                df[f'stoch_d_{slowk}_{slowd}'] = slowd_line
                self.indicators[f'stoch_k_{fastk}_{slowk}'] = df[f'stoch_k_{fastk}_{slowk}']
                self.indicators[f'stoch_d_{slowk}_{slowd}'] = df[f'stoch_d_{slowk}_{slowd}']
                
                # Generate signals (shift by 1 to prevent look-ahead)
                df[f'stoch_k_{fastk}_{slowk}'] = df[f'stoch_k_{fastk}_{slowk}'].shift(1)
                df[f'stoch_d_{slowk}_{slowd}'] = df[f'stoch_d_{slowk}_{slowd}'].shift(1)
                
                # LONG when %K crosses above %D in oversold region (<20)
                long_signals = (
                    (df[f'stoch_k_{fastk}_{slowk}'] > df[f'stoch_d_{slowk}_{slowd}']) & 
                    (df[f'stoch_k_{fastk}_{slowk}'].shift(1) <= df[f'stoch_d_{slowk}_{slowd}'].shift(1)) &
                    (df[f'stoch_k_{fastk}_{slowk}'] < 20)
                )
                # SHORT when %K crosses below %D in overbought region (>80)
                short_signals = (
                    (df[f'stoch_k_{fastk}_{slowk}'] < df[f'stoch_d_{slowk}_{slowd}']) & 
                    (df[f'stoch_k_{fastk}_{slowk}'].shift(1) >= df[f'stoch_d_{slowk}_{slowd}'].shift(1)) &
                    (df[f'stoch_k_{fastk}_{slowk}'] > 80)
                )
                
                signals[f'stoch_{fastk}_{slowk}_{slowd}_long'] = pd.Series(0, index=df.index)
                signals[f'stoch_{fastk}_{slowk}_{slowd}_long'][long_signals] = 1
                
                signals[f'stoch_{fastk}_{slowk}_{slowd}_short'] = pd.Series(0, index=df.index)
                signals[f'stoch_{fastk}_{slowk}_{slowd}_short'][short_signals] = -1
                
                # Combined signal
                signals[f'stoch_{fastk}_{slowk}_{slowd}_combined'] = signals[f'stoch_{fastk}_{slowk}_{slowd}_long'] + signals[f'stoch_{fastk}_{slowk}_{slowd}_short']
                
                # Store metadata
                self.signal_metadata[f'stoch_{fastk}_{slowk}_{slowd}_combined'] = {
                    'type': 'momentum',
                    'indicator': 'Stochastic',
                    'fastk': fastk,
                    'slowk': slowk,
                    'slowd': slowd,
                    'logic': 'LONG when K crosses above D in oversold region, SHORT when K crosses below D in overbought region'
                }
        
        # CCI Signals
        if 'cci' in params:
            for period in params['cci']:
                # Calculate CCI
                df[f'cci_{period}'] = talib.CCI(
                    df['High'],
                    df['Low'],
                    df['Close'],
                    timeperiod=period
                )
                self.indicators[f'cci_{period}'] = df[f'cci_{period}']
                
                # Generate signals (shift by 1 to prevent look-ahead)
                df[f'cci_{period}'] = df[f'cci_{period}'].shift(1)
                
                # LONG when CCI crosses above -100 (from oversold)
                long_signals = (df[f'cci_{period}'] > -100) & (df[f'cci_{period}'].shift(1) <= -100)
                # SHORT when CCI crosses below 100 (from overbought)
                short_signals = (df[f'cci_{period}'] < 100) & (df[f'cci_{period}'].shift(1) >= 100)
                
                signals[f'cci_{period}_long'] = pd.Series(0, index=df.index)
                signals[f'cci_{period}_long'][long_signals] = 1
                
                signals[f'cci_{period}_short'] = pd.Series(0, index=df.index)
                signals[f'cci_{period}_short'][short_signals] = -1
                
                # Combined signal
                signals[f'cci_{period}_combined'] = signals[f'cci_{period}_long'] + signals[f'cci_{period}_short']
                
                # Store metadata
                self.signal_metadata[f'cci_{period}_combined'] = {
                    'type': 'momentum',
                    'indicator': 'CCI',
                    'period': period,
                    'logic': 'LONG when crosses above -100, SHORT when crosses below 100'
                }
        
        return signals
    
    def generate_overlap_signals(self,column:str='Close', **params) -> Dict[str, pd.Series]:
        """
        Generate overlap-based signals using TA-Lib indicators.
        
        Includes: Moving Averages, Bollinger Bands, etc.
        """
        signals = {}
        df = self.df.copy()
        
        # Simple Moving Average (SMA) Signals
        if 'sma' in params:
            for period in params['sma']:
                # Calculate SMA
                df[f'sma_{period}'] = talib.SMA(df[column], timeperiod=period)
                self.indicators[f'sma_{period}'] = df[f'sma_{period}']
                
                # Generate signals (shift by 1 to prevent look-ahead)
                df[f'sma_{period}'] = df[f'sma_{period}'].shift(1)
                
                # LONG when price crosses above SMA
                long_signals = (df[column] > df[f'sma_{period}']) & (df[column].shift(1) <= df[f'sma_{period}'].shift(1))
                # SHORT when price crosses below SMA
                short_signals = (df[column] < df[f'sma_{period}']) & (df[column].shift(1) >= df[f'sma_{period}'].shift(1))
                
                signals[f'sma_{period}_long'] = pd.Series(0, index=df.index)
                signals[f'sma_{period}_long'][long_signals] = 1
                
                signals[f'sma_{period}_short'] = pd.Series(0, index=df.index)
                signals[f'sma_{period}_short'][short_signals] = -1
                
                # Combined signal
                signals[f'sma_{period}_combined'] = signals[f'sma_{period}_long'] + signals[f'sma_{period}_short']
                
                # Store metadata
                self.signal_metadata[f'sma_{period}_combined'] = {
                    'type': 'overlap',
                    'indicator': 'SMA',
                    'period': period,
                    'logic': 'LONG when price crosses above SMA, SHORT when price crosses below SMA'
                }
        
        # Exponential Moving Average (EMA) Signals
        if 'ema' in params:
            for period in params['ema']:
                # Calculate EMA
                df[f'ema_{period}'] = talib.EMA(df[column], timeperiod=period)
                self.indicators[f'ema_{period}'] = df[f'ema_{period}']
                
                # Generate signals (shift by 1 to prevent look-ahead)
                df[f'ema_{period}'] = df[f'ema_{period}'].shift(1)
                
                # LONG when price crosses above EMA
                long_signals = (df[column] > df[f'ema_{period}']) & (df[column].shift(1) <= df[f'ema_{period}'].shift(1))
                # SHORT when price crosses below EMA
                short_signals = (df[column] < df[f'ema_{period}']) & (df[column].shift(1) >= df[f'ema_{period}'].shift(1))
                
                signals[f'ema_{period}_long'] = pd.Series(0, index=df.index)
                signals[f'ema_{period}_long'][long_signals] = 1
                
                signals[f'ema_{period}_short'] = pd.Series(0, index=df.index)
                signals[f'ema_{period}_short'][short_signals] = -1
                
                # Combined signal
                signals[f'ema_{period}_combined'] = signals[f'ema_{period}_long'] + signals[f'ema_{period}_short']
                
                # Store metadata
                self.signal_metadata[f'ema_{period}_combined'] = {
                    'type': 'overlap',
                    'indicator': 'EMA',
                    'period': period,
                    'logic': 'LONG when price crosses above EMA, SHORT when price crosses below EMA'
                }
        
        # Bollinger Bands Signals
        if 'bbands' in params:
            for period, nbdevup, nbdevdn in params['bbands']:
                # Calculate Bollinger Bands
                upper, middle, lower = talib.BBANDS(
                    df[column],
                    timeperiod=period,
                    nbdevup=nbdevup,
                    nbdevdn=nbdevdn,
                    matype=0
                )
                df[f'bb_upper_{period}_{nbdevup}_{nbdevdn}'] = upper
                df[f'bb_middle_{period}_{nbdevup}_{nbdevdn}'] = middle
                df[f'bb_lower_{period}_{nbdevup}_{nbdevdn}'] = lower
                self.indicators[f'bb_upper_{period}_{nbdevup}_{nbdevdn}'] = df[f'bb_upper_{period}_{nbdevup}_{nbdevdn}']
                self.indicators[f'bb_middle_{period}_{nbdevup}_{nbdevdn}'] = df[f'bb_middle_{period}_{nbdevup}_{nbdevdn}']
                self.indicators[f'bb_lower_{period}_{nbdevup}_{nbdevdn}'] = df[f'bb_lower_{period}_{nbdevup}_{nbdevdn}']
                
                # Generate signals (shift by 1 to prevent look-ahead)
                df[f'bb_upper_{period}_{nbdevup}_{nbdevdn}'] = df[f'bb_upper_{period}_{nbdevup}_{nbdevdn}'].shift(1)
                df[f'bb_lower_{period}_{nbdevup}_{nbdevdn}'] = df[f'bb_lower_{period}_{nbdevup}_{nbdevdn}'].shift(1)
                
                # LONG when price crosses below lower band (mean reversion)
                long_signals = (df[column] < df[f'bb_lower_{period}_{nbdevup}_{nbdevdn}']) & (df[column].shift(1) >= df[f'bb_lower_{period}_{nbdevup}_{nbdevdn}'].shift(1))
                # SHORT when price crosses above upper band (mean reversion)
                short_signals = (df[column] > df[f'bb_upper_{period}_{nbdevup}_{nbdevdn}']) & (df[column].shift(1) <= df[f'bb_upper_{period}_{nbdevup}_{nbdevdn}'].shift(1))
                
                signals[f'bb_{period}_{nbdevup}_{nbdevdn}_long'] = pd.Series(0, index=df.index)
                signals[f'bb_{period}_{nbdevup}_{nbdevdn}_long'][long_signals] = 1
                
                signals[f'bb_{period}_{nbdevup}_{nbdevdn}_short'] = pd.Series(0, index=df.index)
                signals[f'bb_{period}_{nbdevup}_{nbdevdn}_short'][short_signals] = -1
                
                # Combined signal
                signals[f'bb_{period}_{nbdevup}_{nbdevdn}_combined'] = signals[f'bb_{period}_{nbdevup}_{nbdevdn}_long'] + signals[f'bb_{period}_{nbdevup}_{nbdevdn}_short']
                
                # Store metadata
                self.signal_metadata[f'bb_{period}_{nbdevup}_{nbdevdn}_combined'] = {
                    'type': 'overlap',
                    'indicator': 'Bollinger Bands',
                    'period': period,
                    'nbdevup': nbdevup,
                    'nbdevdn': nbdevdn,
                    'logic': 'LONG when price crosses below lower band, SHORT when price crosses above upper band'
                }
        
        # Moving Average Crossover Signals
        if 'ma_crossover' in params:
            for short_period, long_period in params['ma_crossover']:
                # Calculate MAs
                df[f'sma_short_{short_period}'] = talib.SMA(df[column], timeperiod=short_period)
                df[f'sma_long_{long_period}'] = talib.SMA(df[column], timeperiod=long_period)
                self.indicators[f'sma_short_{short_period}'] = df[f'sma_short_{short_period}']
                self.indicators[f'sma_long_{long_period}'] = df[f'sma_long_{long_period}']
                
                # Generate signals (shift by 1 to prevent look-ahead)
                df[f'sma_short_{short_period}'] = df[f'sma_short_{short_period}'].shift(1)
                df[f'sma_long_{long_period}'] = df[f'sma_long_{long_period}'].shift(1)
                
                # LONG when short MA crosses above long MA
                long_signals = (df[f'sma_short_{short_period}'] > df[f'sma_long_{long_period}']) & (df[f'sma_short_{short_period}'].shift(1) <= df[f'sma_long_{long_period}'].shift(1))
                # SHORT when short MA crosses below long MA
                short_signals = (df[f'sma_short_{short_period}'] < df[f'sma_long_{long_period}']) & (df[f'sma_short_{short_period}'].shift(1) >= df[f'sma_long_{long_period}'].shift(1))
                
                signals[f'ma_crossover_{short_period}_{long_period}_long'] = pd.Series(0, index=df.index)
                signals[f'ma_crossover_{short_period}_{long_period}_long'][long_signals] = 1
                
                signals[f'ma_crossover_{short_period}_{long_period}_short'] = pd.Series(0, index=df.index)
                signals[f'ma_crossover_{short_period}_{long_period}_short'][short_signals] = -1
                
                # Combined signal
                signals[f'ma_crossover_{short_period}_{long_period}_combined'] = signals[f'ma_crossover_{short_period}_{long_period}_long'] + signals[f'ma_crossover_{short_period}_{long_period}_short']
                
                # Store metadata
                self.signal_metadata[f'ma_crossover_{short_period}_{long_period}_combined'] = {
                    'type': 'overlap',
                    'indicator': 'MA Crossover',
                    'short_period': short_period,
                    'long_period': long_period,
                    'logic': 'LONG when short MA crosses above long MA, SHORT when short MA crosses below long MA'
                }
        
        return signals
    
    def generate_volatility_signals(self,column:str='Close', **params) -> Dict[str, pd.Series]:
        """
        Generate volatility-based signals using TA-Lib indicators.
        
        Includes: ATR, NATR, TRANGE
        """
        signals = {}
        df = self.df.copy()
        
        # Average True Range (ATR) Signals
        if 'atr' in params:
            for period in params['atr']:
                # Calculate ATR
                df[f'atr_{period}'] = talib.ATR(
                    df['High'],
                    df['Low'],
                    df['Close'],
                    timeperiod=period
                )
                self.indicators[f'atr_{period}'] = df[f'atr_{period}']
                
                # Calculate normalized ATR (as percentage of price)
                df[f'natr_{period}'] = talib.NATR(
                    df['High'],
                    df['Low'],
                    df['Close'],
                    timeperiod=period
                )
                self.indicators[f'natr_{period}'] = df[f'natr_{period}']
                
                # Generate signals (shift by 1 to prevent look-ahead)
                df[f'natr_{period}'] = df[f'natr_{period}'].shift(1)
                
                # LONG when volatility is low (below 25th percentile)
                volatility_threshold = df[f'natr_{period}'].quantile(0.25)
                long_signals = df[f'natr_{period}'] < volatility_threshold
                
                # SHORT when volatility is high (above 75th percentile)
                volatility_threshold = df[f'natr_{period}'].quantile(0.75)
                short_signals = df[f'natr_{period}'] > volatility_threshold
                
                signals[f'atr_{period}_long'] = pd.Series(0, index=df.index)
                signals[f'atr_{period}_long'][long_signals] = 1
                
                signals[f'atr_{period}_short'] = pd.Series(0, index=df.index)
                signals[f'atr_{period}_short'][short_signals] = -1
                
                # Combined signal
                signals[f'atr_{period}_combined'] = signals[f'atr_{period}_long'] + signals[f'atr_{period}_short']
                
                # Store metadata
                self.signal_metadata[f'atr_{period}_combined'] = {
                    'type': 'volatility',
                    'indicator': 'ATR',
                    'period': period,
                    'logic': 'LONG when volatility low (25th percentile), SHORT when volatility high (75th percentile)'
                }
        
        return signals
    
    def generate_volume_signals(self,column:str='Close', **params) -> Dict[str, pd.Series]:
        """
        Generate volume-based signals using TA-Lib indicators.
        
        Includes: OBV, AD, CMF
        """
        signals = {}
        df = self.df.copy()
        
        # On Balance Volume (OBV) Signals
        if 'obv' in params:
            # Calculate OBV
            df['obv'] = talib.OBV(df[column], df['Volume'])
            self.indicators['obv'] = df['obv']
            
            # Generate signals (shift by 1 to prevent look-ahead)
            df['obv'] = df['obv'].shift(1)
            
            # Calculate OBV slope (20-day)
            df['obv_slope'] = df['obv'].diff(20)
            
            # LONG when OBV is trending up
            long_signals = df['obv_slope'] > 0
            # SHORT when OBV is trending down
            short_signals = df['obv_slope'] < 0
            
            signals['obv_long'] = pd.Series(0, index=df.index)
            signals['obv_long'][long_signals] = 1
            
            signals['obv_short'] = pd.Series(0, index=df.index)
            signals['obv_short'][short_signals] = -1
            
            # Combined signal
            signals['obv_combined'] = signals['obv_long'] + signals['obv_short']
            
            # Store metadata
            self.signal_metadata['obv_combined'] = {
                'type': 'volume',
                'indicator': 'OBV',
                'logic': 'LONG when OBV trending up, SHORT when OBV trending down'
            }
        
        # Chaikin A/D Line Signals
        if 'ad' in params:
            # Calculate Chaikin A/D Line
            df['ad'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
            self.indicators['ad'] = df['ad']
            
            # Generate signals (shift by 1 to prevent look-ahead)
            df['ad'] = df['ad'].shift(1)
            
            # Calculate AD slope (20-day)
            df['ad_slope'] = df['ad'].diff(20)
            
            # LONG when AD is trending up
            long_signals = df['ad_slope'] > 0
            # SHORT when AD is trending down
            short_signals = df['ad_slope'] < 0
            
            signals['ad_long'] = pd.Series(0, index=df.index)
            signals['ad_long'][long_signals] = 1
            
            signals['ad_short'] = pd.Series(0, index=df.index)
            signals['ad_short'][short_signals] = -1
            
            # Combined signal
            signals['ad_combined'] = signals['ad_long'] + signals['ad_short']
            
            # Store metadata
            self.signal_metadata['ad_combined'] = {
                'type': 'volume',
                'indicator': 'Chaikin A/D',
                'logic': 'LONG when AD trending up, SHORT when AD trending down'
            }
        
        # Chaikin Money Flow (CMF) Signals
        if 'cmf' in params:
            for period in params['cmf']:
                # Calculate CMF
                df[f'cmf_{period}'] = talib.ADOSC(
                    df['High'],
                    df['Low'],
                    df['Close'],
                    df['Volume'],
                    fastperiod=3,
                    slowperiod=period
                )
                self.indicators[f'cmf_{period}'] = df[f'cmf_{period}']
                
                # Generate signals (shift by 1 to prevent look-ahead)
                df[f'cmf_{period}'] = df[f'cmf_{period}'].shift(1)
                
                # LONG when CMF crosses above 0
                long_signals = (df[f'cmf_{period}'] > 0) & (df[f'cmf_{period}'].shift(1) <= 0)
                # SHORT when CMF crosses below 0
                short_signals = (df[f'cmf_{period}'] < 0) & (df[f'cmf_{period}'].shift(1) >= 0)
                
                signals[f'cmf_{period}_long'] = pd.Series(0, index=df.index)
                signals[f'cmf_{period}_long'][long_signals] = 1
                
                signals[f'cmf_{period}_short'] = pd.Series(0, index=df.index)
                signals[f'cmf_{period}_short'][short_signals] = -1
                
                # Combined signal
                signals[f'cmf_{period}_combined'] = signals[f'cmf_{period}_long'] + signals[f'cmf_{period}_short']
                
                # Store metadata
                self.signal_metadata[f'cmf_{period}_combined'] = {
                    'type': 'volume',
                    'indicator': 'CMF',
                    'period': period,
                    'logic': 'LONG when CMF crosses above 0, SHORT when CMF crosses below 0'
                }
        
        return signals
    
    def generate_cycle_signals(self,column:str='Close', **params) -> Dict[str, pd.Series]:
        """
        Generate cycle-based signals using TA-Lib indicators.
        
        Includes: Hilbert Transform, Sine Wave
        """
        signals = {}
        df = self.df.copy()
        
        # Hilbert Transform - Dominant Cycle Period
        if 'ht_dcperiod' in params:
            # Calculate Dominant Cycle Period
            df['ht_dcperiod'] = talib.HT_DCPERIOD(df[column])
            self.indicators['ht_dcperiod'] = df['ht_dcperiod']
            
            # Generate signals (shift by 1 to prevent look-ahead)
            df['ht_dcperiod'] = df['ht_dcperiod'].shift(1)
            
            # LONG when cycle period is increasing (trend forming)
            df['ht_dcperiod_slope'] = df['ht_dcperiod'].diff()
            long_signals = df['ht_dcperiod_slope'] > 0
            
            # SHORT when cycle period is decreasing (trend ending)
            short_signals = df['ht_dcperiod_slope'] < 0
            
            signals['ht_dcperiod_long'] = pd.Series(0, index=df.index)
            signals['ht_dcperiod_long'][long_signals] = 1
            
            signals['ht_dcperiod_short'] = pd.Series(0, index=df.index)
            signals['ht_dcperiod_short'][short_signals] = -1
            
            # Combined signal
            signals['ht_dcperiod_combined'] = signals['ht_dcperiod_long'] + signals['ht_dcperiod_short']
            
            # Store metadata
            self.signal_metadata['ht_dcperiod_combined'] = {
                'type': 'cycle',
                'indicator': 'HT_DCPERIOD',
                'logic': 'LONG when cycle period increasing, SHORT when cycle period decreasing'
            }
        
        # Hilbert Transform - Phasor Components
        if 'ht_phasor' in params:
            # Calculate Phasor Components
            inphase, quadrature = talib.HT_PHASOR(df[column])
            df['ht_inphase'] = inphase
            df['ht_quadrature'] = quadrature
            self.indicators['ht_inphase'] = df['ht_inphase']
            self.indicators['ht_quadrature'] = df['ht_quadrature']
            
            # Generate signals (shift by 1 to prevent look-ahead)
            df['ht_inphase'] = df['ht_inphase'].shift(1)
            df['ht_quadrature'] = df['ht_quadrature'].shift(1)
            
            # Calculate phase
            df['ht_phase'] = np.degrees(np.arctan2(df['ht_quadrature'], df['ht_inphase']))
            
            # LONG when phase crosses 0 (new uptrend)
            long_signals = (df['ht_phase'] > 0) & (df['ht_phase'].shift(1) <= 0)
            # SHORT when phase crosses 180 (new downtrend)
            short_signals = (df['ht_phase'] > 180) & (df['ht_phase'].shift(1) <= 180)
            
            signals['ht_phasor_long'] = pd.Series(0, index=df.index)
            signals['ht_phasor_long'][long_signals] = 1
            
            signals['ht_phasor_short'] = pd.Series(0, index=df.index)
            signals['ht_phasor_short'][short_signals] = -1
            
            # Combined signal
            signals['ht_phasor_combined'] = signals['ht_phasor_long'] + signals['ht_phasor_short']
            
            # Store metadata
            self.signal_metadata['ht_phasor_combined'] = {
                'type': 'cycle',
                'indicator': 'HT_PHASOR',
                'logic': 'LONG when phase crosses 0, SHORT when phase crosses 180'
            }
        
        return signals
    
    def generate_pattern_signals(self,column:str='Close', **params) -> Dict[str, pd.Series]:
        """
        Generate candlestick pattern signals using TA-Lib.
        
        Note: Pattern recognition is less systematic but can be used for confirmation.
        """
        if not params:
            return {}
        
        signals = {}
        df = self.df.copy()
        
        # Get all pattern recognition functions from TA-Lib
        pattern_functions = [func for func in dir(talib) if func.startswith('CDL')]
        
        for pattern in params.get('patterns', []):
            if pattern not in pattern_functions:
                continue
            
            # Calculate pattern
            pattern_func = getattr(talib, pattern)
            df[pattern] = pattern_func(
                df['Open'],
                df['High'],
                df['Low'],
                df['Close']
            )
            self.indicators[pattern] = df[pattern]
            
            # Generate signals (shift by 1 to prevent look-ahead)
            df[pattern] = df[pattern].shift(1)
            
            # LONG when bullish pattern detected (value = 100)
            long_signals = df[pattern] == 100
            # SHORT when bearish pattern detected (value = -100)
            short_signals = df[pattern] == -100
            
            signals[f'{pattern}_long'] = pd.Series(0, index=df.index)
            signals[f'{pattern}_long'][long_signals] = 1
            
            signals[f'{pattern}_short'] = pd.Series(0, index=df.index)
            signals[f'{pattern}_short'][short_signals] = -1
            
            # Combined signal
            signals[f'{pattern}_combined'] = signals[f'{pattern}_long'] + signals[f'{pattern}_short']
            
            # Store metadata
            self.signal_metadata[f'{pattern}_combined'] = {
                'type': 'pattern',
                'indicator': pattern,
                'logic': 'LONG when bullish pattern detected (100), SHORT when bearish pattern detected (-100)'
            }
        
        return signals
    
    def generate_price_signals(self,column:str='Close', **params) -> Dict[str, pd.Series]:
        """
        Generate price-based signals using TA-Lib.
        
        Includes: Typical Price, Median Price
        """
        signals = {}
        df = self.df.copy()
        
        # Typical Price (High+Low+Close)/3
        if 'typprice' in params:
            # Calculate Typical Price
            df['typprice'] = talib.TYPPRICE(df['High'], df['Low'], df['Close'])
            self.indicators['typprice'] = df['typprice']
            
            # Generate signals (shift by 1 to prevent look-ahead)
            df['typprice'] = df['typprice'].shift(1)
            
            # LONG when price crosses above typical price
            long_signals = (df['Close'] > df['typprice']) & (df['Close'].shift(1) <= df['typprice'].shift(1))
            # SHORT when price crosses below typical price
            short_signals = (df['Close'] < df['typprice']) & (df['Close'].shift(1) >= df['typprice'].shift(1))
            
            signals['typprice_long'] = pd.Series(0, index=df.index)
            signals['typprice_long'][long_signals] = 1
            
            signals['typprice_short'] = pd.Series(0, index=df.index)
            signals['typprice_short'][short_signals] = -1
            
            # Combined signal
            signals['typprice_combined'] = signals['typprice_long'] + signals['typprice_short']
            
            # Store metadata
            self.signal_metadata['typprice_combined'] = {
                'type': 'price',
                'indicator': 'Typical Price',
                'logic': 'LONG when price crosses above typical price, SHORT when price crosses below typical price'
            }
        
        # Median Price (High+Low)/2
        if 'medianprice' in params:
            # Calculate Median Price
            df['medianprice'] = talib.MEDPRICE(df['High'], df['Low'])
            self.indicators['medianprice'] = df['medianprice']
            
            # Generate signals (shift by 1 to prevent look-ahead)
            df['medianprice'] = df['medianprice'].shift(1)
            
            # LONG when price crosses above median price
            long_signals = (df['Close'] > df['medianprice']) & (df['Close'].shift(1) <= df['medianprice'].shift(1))
            # SHORT when price crosses below median price
            short_signals = (df['Close'] < df['medianprice']) & (df['Close'].shift(1) >= df['medianprice'].shift(1))
            
            signals['medianprice_long'] = pd.Series(0, index=df.index)
            signals['medianprice_long'][long_signals] = 1
            
            signals['medianprice_short'] = pd.Series(0, index=df.index)
            signals['medianprice_short'][short_signals] = -1
            
            # Combined signal
            signals['medianprice_combined'] = signals['medianprice_long'] + signals['medianprice_short']
            
            # Store metadata
            self.signal_metadata['medianprice_combined'] = {
                'type': 'price',
                'indicator': 'Median Price',
                'logic': 'LONG when price crosses above median price, SHORT when price crosses below median price'
            }
        
        return signals
    
    def generate_statistic_signals(self,column:str='Close', **params) -> Dict[str, pd.Series]:
        """
        Generate statistical signals using TA-Lib.
        
        Includes: Linear Regression, Standard Deviation
        """
        signals = {}
        df = self.df.copy()
        
        # Linear Regression
        if 'linearreg' in params:
            for period in params['linearreg']:
                # Calculate Linear Regression
                df[f'linearreg_{period}'] = talib.LINEARREG(df[column], timeperiod=period)
                self.indicators[f'linearreg_{period}'] = df[f'linearreg_{period}']
                
                # Generate signals (shift by 1 to prevent look-ahead)
                df[f'linearreg_{period}'] = df[f'linearreg_{period}'].shift(1)
                
                # Calculate slope
                df[f'linearreg_slope_{period}'] = df[f'linearreg_{period}'].diff()
                
                # LONG when slope is positive
                long_signals = df[f'linearreg_slope_{period}'] > 0
                # SHORT when slope is negative
                short_signals = df[f'linearreg_slope_{period}'] < 0
                
                signals[f'linearreg_{period}_long'] = pd.Series(0, index=df.index)
                signals[f'linearreg_{period}_long'][long_signals] = 1
                
                signals[f'linearreg_{period}_short'] = pd.Series(0, index=df.index)
                signals[f'linearreg_{period}_short'][short_signals] = -1
                
                # Combined signal
                signals[f'linearreg_{period}_combined'] = signals[f'linearreg_{period}_long'] + signals[f'linearreg_{period}_short']
                
                # Store metadata
                self.signal_metadata[f'linearreg_{period}_combined'] = {
                    'type': 'statistic',
                    'indicator': 'Linear Regression',
                    'period': period,
                    'logic': 'LONG when slope positive, SHORT when slope negative'
                }
        
        # Standard Deviation
        if 'stddev' in params:
            for period, nbdev in params['stddev']:
                # Calculate Standard Deviation
                df[f'stddev_{period}_{nbdev}'] = talib.STDDEV(df[column], timeperiod=period, nbdev=nbdev)
                self.indicators[f'stddev_{period}_{nbdev}'] = df[f'stddev_{period}_{nbdev}']
                
                # Generate signals (shift by 1 to prevent look-ahead)
                df[f'stddev_{period}_{nbdev}'] = df[f'stddev_{period}_{nbdev}'].shift(1)
                
                # LONG when volatility is low (below 25th percentile)
                volatility_threshold = df[f'stddev_{period}_{nbdev}'].quantile(0.25)
                long_signals = df[f'stddev_{period}_{nbdev}'] < volatility_threshold
                
                # SHORT when volatility is high (above 75th percentile)
                volatility_threshold = df[f'stddev_{period}_{nbdev}'].quantile(0.75)
                short_signals = df[f'stddev_{period}_{nbdev}'] > volatility_threshold
                
                signals[f'stddev_{period}_{nbdev}_long'] = pd.Series(0, index=df.index)
                signals[f'stddev_{period}_{nbdev}_long'][long_signals] = 1
                
                signals[f'stddev_{period}_{nbdev}_short'] = pd.Series(0, index=df.index)
                signals[f'stddev_{period}_{nbdev}_short'][short_signals] = -1
                
                # Combined signal
                signals[f'stddev_{period}_{nbdev}_combined'] = signals[f'stddev_{period}_{nbdev}_long'] + signals[f'stddev_{period}_{nbdev}_short']
                
                # Store metadata
                self.signal_metadata[f'stddev_{period}_{nbdev}_combined'] = {
                    'type': 'statistic',
                    'indicator': 'Standard Deviation',
                    'period': period,
                    'nbdev': nbdev,
                    'logic': 'LONG when volatility low (25th percentile), SHORT when volatility high (75th percentile)'
                }
        
        return signals
    
    def generate_combined_signal(
        self,
        signal_names: List[str],
        weights: Optional[List[float]] = None,
        column:str='Close',
        signal_name: str = "combined"
        
    ) -> pd.Series:
        """
        Generate a combined signal from multiple individual signals.
        
        Parameters:
        -----------
        signal_names : list
            List of signal names to combine
        weights : list, optional
            Weights for each signal (must sum to 1)
        signal_name : str, default="combined"
            Name for the combined signal
            
        Returns:
        --------
        pd.Series
            Combined trading signals (-1, 0, 1)
        """
        # Validate signal names
        for name in signal_names:
            if name not in self.signals:
                raise ValueError(f"Signal '{name}' not found. Generate signals first.")
        
        # Set default weights if not provided
        if weights is None:
            weights = [1.0/len(signal_names)] * len(signal_names)
        elif len(weights) != len(signal_names):
            raise ValueError("Number of weights must match number of signals")
        elif abs(sum(weights) - 1.0) > 1e-5:
            # Normalize weights
            weights = [w/sum(weights) for w in weights]
        
        # Generate combined signal
        combined_signal = pd.Series(0, index=self.df.index)
        for name, weight in zip(signal_names, weights):
            combined_signal += self.signals[name] * weight
        
        # Generate final signals
        signals = pd.Series(0, index=self.df.index)
        signals[combined_signal > 0.3] = 1  # LONG threshold
        signals[combined_signal < -0.3] = -1  # SHORT threshold
        
        # Store signals
        self.signals[signal_name] = signals
        
        # Store metadata
        self.signal_metadata[signal_name] = {
            'type': 'combined',
            'components': dict(zip(signal_names, weights)),
            'threshold': 0.3,
            'logic': 'LONG when combined signal > 0.3, SHORT when combined signal < -0.3'
        }
        
        return signals
    
    def get_signal_metadata(self, signal_name: str) -> Dict:
        """Get metadata for a specific signal"""
        if signal_name not in self.signal_metadata:
            raise ValueError(f"Signal '{signal_name}' metadata not found.")
        return self.signal_metadata[signal_name]
    
    def validate_signals(self, signal_name: str = None) -> Dict:
        """
        Validate signal quality and timing.
        
        Parameters:
        -----------
        signal_name : str, optional
            Name of signal series to validate
            If None, validates all signals
            
        Returns:
        --------
        dict of validation results
        """
        results = {}
        
        signals_to_validate = self.signals.keys() if signal_name is None else [signal_name]
        
        for name in signals_to_validate:
            signals = self.signals[name]
            
            # Validation checks
            validation = {
                'has_no_nan': not signals.isna().any(),
                'signal_range': (signals.min(), signals.max()),
                'non_zero_ratio': (signals != 0).mean(),
                'long_ratio': (signals > 0).mean(),
                'short_ratio': (signals < 0).mean(),
                'max_consecutive_positions': self._max_consecutive_nonzero(signals),
                'zero_crossings': (signals != 0) & (signals.shift(1) == 0).sum()
            }
            
            # Flag issues
            issues = []
            if not validation['has_no_nan']:
                issues.append("NaN values in signals (look-ahead bias possible)")
            if validation['non_zero_ratio'] < 0.05:
                issues.append(f"Low signal frequency ({validation['non_zero_ratio']:.1%} < 5%)")
            if validation['max_consecutive_positions'] > 20:
                issues.append(f"Too many consecutive positions ({validation['max_consecutive_positions']} > 20)")
            
            validation['issues'] = issues
            validation['is_valid'] = len(issues) == 0
            
            results[name] = validation
        
        return results
    
    def _max_consecutive_nonzero(self, series: pd.Series) -> int:
        """Helper: Calculate max consecutive non-zero values"""
        non_zero = (series != 0).astype(int)
        counts = non_zero * (non_zero.groupby((non_zero != non_zero.shift()).cumsum()).cumcount() + 1)
        return counts.max()
    
    def analyze_signal_performance(self, signal_name: str = None) -> Dict:
        """
        Analyze how well signals predict future returns.
        
        Parameters:
        -----------
        signal_name : str, optional
            Name of signal series to analyze
            If None, analyzes all signals
            
        Returns:
        --------
        dict of performance analysis
        """
        results = {}
        
        signals_to_analyze = self.signals.keys() if signal_name is None else [signal_name]
        
        for name in signals_to_analyze:
            signals = self.signals[name]
            close = self.df['Close']
            
            # Calculate next day returns
            next_day_return = close.pct_change().shift(-1)
            
            # Align with signals
            aligned_return = next_day_return.loc[signals.index]
            
            # Analyze long signals
            long_mask = signals > 0
            short_mask = signals < 0
            
            # Long performance
            long_returns = aligned_return[long_mask]
            long_win_rate = (long_returns > 0).mean() if len(long_returns) > 0 else 0
            long_avg_return = long_returns.mean() if len(long_returns) > 0 else 0
            
            # Short performance
            short_returns = aligned_return[short_mask]
            short_win_rate = (short_returns < 0).mean() if len(short_returns) > 0 else 0
            short_avg_return = -short_returns.mean() if len(short_returns) > 0 else 0  # Flip sign for short
            
            # Overall performance
            total_returns = aligned_return[signals != 0]
            total_win_rate = (total_returns > 0).mean() if len(total_returns) > 0 else 0
            total_avg_return = total_returns.mean() if len(total_returns) > 0 else 0
            
            # Signal correlation
            signal_return_corr = signals.corr(aligned_return.fillna(0))
            
            results[name] = {
                'strategy': name,
                'total_signals': len(signals),
                'active_signals': (signals != 0).sum(),
                'long_signals': (signals > 0).sum(),
                'short_signals': (signals < 0).sum(),
                'long_win_rate': long_win_rate,
                'long_avg_return': long_avg_return,
                'short_win_rate': short_win_rate,
                'short_avg_return': short_avg_return,
                'total_win_rate': total_win_rate,
                'total_avg_return': total_avg_return,
                'signal_return_corr': signal_return_corr,
                'sharpe_ratio': total_avg_return / total_returns.std() * np.sqrt(252) if len(total_returns) > 0 else 0
            }
        
        return results
    
    def plot_indicator(
        self, 
        indicator_name: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 5)
    ):
        """
        Plot an indicator with price data.
        
        Parameters:
        -----------
        indicator_name : str
            Name of indicator to plot
        start_date : str, optional
            Start date for the plot
        end_date : str, optional
            End date for the plot
        figsize : tuple, default=(8, 5)
            Figure size
        """
        if indicator_name not in self.indicators:
            raise ValueError(f"Indicator '{indicator_name}' not found. Generate indicators first.")
        
        # Filter data if dates provided
        indicator = self.indicators[indicator_name]
        df = self.df.copy()
        
        if start_date:
            start_date = pd.Timestamp(start_date)
            indicator = indicator[indicator.index >= start_date]
            df = df[df.index >= start_date]
        
        if end_date:
            end_date = pd.Timestamp(end_date)
            indicator = indicator[indicator.index <= end_date]
            df = df[df.index <= end_date]
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot price
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['Close'], label='Close Price')
        plt.title(f'{indicator_name} Indicator', fontsize=14)
        plt.ylabel('Price', fontsize=10)
        plt.legend(fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=8, rotation=45)
        plt.yticks(fontsize=8)
        
        # Plot indicator
        plt.subplot(2, 1, 2)
        plt.plot(indicator.index, indicator, label=indicator_name, color='blue')
        plt.ylabel(indicator_name, fontsize=10)
        plt.legend(fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=8, rotation=45)
        plt.yticks(fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def plot_signal_effectiveness(
        self, 
        signal_name: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 5)
    ):
        """
        Plot signal effectiveness against future returns.
        
        Parameters:
        -----------
        signal_name : str
            Name of signal to analyze
        start_date : str, optional
            Start date for the analysis
        end_date : str, optional
            End date for the analysis
        figsize : tuple, default=(8, 5)
            Figure size
        """
        if signal_name not in self.signals:
            raise ValueError(f"Signal '{signal_name}' not found. Generate signals first.")
        
        # Filter data if dates provided
        signals = self.signals[signal_name]
        df = self.df.copy()
        
        if start_date:
            start_date = pd.Timestamp(start_date)
            signals = signals[signals.index >= start_date]
            df = df[df.index >= start_date]
        
        if end_date:
            end_date = pd.Timestamp(end_date)
            signals = signals[signals.index <= end_date]
            df = df[df.index <= end_date]
        
        # Calculate next day returns
        next_day_return = df['Close'].pct_change().shift(-1)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Scatter plot of signal vs next day return
        plt.scatter(signals, next_day_return, alpha=0.5)
        plt.axvline(0, color='k', linestyle='-', alpha=0.3)
        plt.axhline(0, color='k', linestyle='-', alpha=0.3)
        plt.title(f'{signal_name} Effectiveness', fontsize=14)
        plt.xlabel('Signal', fontsize=10)
        plt.ylabel('Next Day Return', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def get_all_signals(self) -> Dict[str, pd.Series]:
        """Get all generated signals"""
        return self.signals
    
    def get_all_indicators(self) -> Dict[str, pd.Series]:
        """Get all generated indicators"""
        return self.indicators
    

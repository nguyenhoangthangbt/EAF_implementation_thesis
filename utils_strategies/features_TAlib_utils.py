import numpy as np
import pandas as pd
import warnings, sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from utils.config_utils import load_config
config=load_config('ta_config.yaml')

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib not available. Some features will not be generated.")

class TALibFeatureGenerator:

    def __init__(self, df_OHLCV: pd.DataFrame):

        # Validate input data
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if 'Date' in df_OHLCV.columns:
            df_OHLCV['Date'] = pd.to_datetime(df_OHLCV['Date'])
            df_OHLCV.set_index('Date', inplace=True)
        
        for col in required_columns:
            if col not in df_OHLCV.columns:
                raise ValueError(f"DataFrame must contain '{col}' column")
        
        # Store and prepare data
        self.df = df_OHLCV.copy()
        self.features = {}
        self.feature_metadata = {}
    
    def generate_all_features(
        self,
        momentum_params: Optional[Dict] = None,
        volume_params: Optional[Dict] = None,
        volatility_params: Optional[Dict] = None,
        overlap_params: Optional[Dict] = None,
        cycle_params: Optional[Dict] = None,
        price_params: Optional[Dict] = None,
        statistic_params: Optional[Dict] = None,
        include_momentum: bool = True,
        include_volume: bool = True,
        include_volatility: bool = True,
        include_overlap: bool = True,
        include_cycle: bool = True,
        include_price: bool = True,
        include_statistic: bool = True,
        normalize_features: bool = False
    ) -> pd.DataFrame:

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
        if price_params is None:
            price_params = ta_default_paras['price_params']
        if statistic_params is None:
            statistic_params =ta_default_paras['statistic_params']
        
        # Generate features by category
        features = {}
        if include_momentum:
            features.update(self.generate_momentum_features(**momentum_params))
        if include_volume:
            features.update(self.generate_Volume_features(**volume_params))
        if include_volatility:
            features.update(self.generate_volatility_features(**volatility_params))
        if include_overlap:
            features.update(self.generate_overlap_features(**overlap_params))
        if include_cycle:
            features.update(self.generate_cycle_features(**cycle_params))
        if include_price:
            features.update(self.generate_price_features(**price_params))
        if include_statistic:
            features.update(self.generate_statistic_features(**statistic_params))
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(features, index=self.df.index)
        
        # Normalize if requested
        if normalize_features:
            feature_df = (feature_df - feature_df.mean()) / feature_df.std()
        
        # Store features
        self.features = feature_df
        return feature_df
    
    def generate_momentum_features(self, **params) -> Dict[str, pd.Series]:
        """
        Generate momentum-based features using TA-Lib indicators.
        Includes: RSI, MACD, Stochastic, CCI, etc.
        
        Key differences from signal generation:
        - Returns raw indicator values (not thresholded signals)
        - Creates ratio-based features where appropriate
        - Shifts all features by 1 to prevent look-ahead bias
        - Adds normalized versions of indicators
        """
        features = {}
        df = self.df.copy()
        Close = df['Close']
        
        # RSI Features
        if 'rsi' in params:
            for period in params['rsi']:
                # Calculate RSI (shifted by 1 to prevent look-ahead)
                rsi = talib.RSI(Close, timeperiod=period).shift(1)
                self.features[f'rsi_{period}'] = rsi
                
                # Normalized RSI (0-1 range)
                rsi_norm = rsi / 100.0
                features[f'rsi_{period}_norm'] = rsi_norm
                
                # RSI momentum (change in RSI)
                rsi_momentum = rsi.diff()
                features[f'rsi_{period}_momentum'] = rsi_momentum
                
                # RSI vs. moving average of RSI
                rsi_ma = rsi.rolling(5).mean()
                rsi_vs_ma = rsi - rsi_ma
                features[f'rsi_{period}_vs_ma'] = rsi_vs_ma
                
                # Store metadata
                self.feature_metadata[f'rsi_{period}_norm'] = {
                    'category': 'momentum',
                    'indicator': 'RSI',
                    'period': period,
                    'description': 'Normalized RSI (0-1 range)',
                    'prediction_logic': 'Higher values indicate stronger momentum'
                }
                self.feature_metadata[f'rsi_{period}_momentum'] = {
                    'category': 'momentum',
                    'indicator': 'RSI Momentum',
                    'period': period,
                    'description': 'Change in RSI value',
                    'prediction_logic': 'Positive momentum suggests continuing trend'
                }
                self.feature_metadata[f'rsi_{period}_vs_ma'] = {
                    'category': 'momentum',
                    'indicator': 'RSI vs MA',
                    'period': period,
                    'description': 'RSI value relative to its moving average',
                    'prediction_logic': 'Values above zero suggest bullish momentum'
                }
        
        # MACD Features
        if 'macd' in params and TALIB_AVAILABLE:
            for fast, slow, signal in params['macd']:
                # Calculate MACD (shifted by 1 to prevent look-ahead)
                macd, macd_signal, _ = talib.MACD(
                    Close, 
                    fastperiod=fast, 
                    slowperiod=slow, 
                    signalperiod=signal
                )
                macd = macd.shift(1)
                macd_signal = macd_signal.shift(1)
                
                # MACD line
                features[f'macd_{fast}_{slow}_{signal}'] = macd
                
                # MACD signal line
                features[f'macd_signal_{fast}_{slow}_{signal}'] = macd_signal
                
                # MACD histogram (difference between MACD and signal)
                macd_hist = macd - macd_signal
                features[f'macd_hist_{fast}_{slow}_{signal}'] = macd_hist
                
                # Normalized MACD histogram (relative to ATR)
                if 'atr_14' in self.features:
                    macd_hist_norm = macd_hist / self.features['atr_14']
                    features[f'macd_hist_norm_{fast}_{slow}_{signal}'] = macd_hist_norm
                    self.feature_metadata[f'macd_hist_norm_{fast}_{slow}_{signal}'] = {
                        'category': 'momentum',
                        'indicator': 'Normalized MACD Histogram',
                        'fast': fast,
                        'slow': slow,
                        'signal': signal,
                        'description': 'MACD histogram normalized by ATR',
                        'prediction_logic': 'Measures momentum strength relative to volatility'
                    }
                
                # Store metadata
                self.feature_metadata[f'macd_{fast}_{slow}_{signal}'] = {
                    'category': 'momentum',
                    'indicator': 'MACD Line',
                    'fast': fast,
                    'slow': slow,
                    'signal': signal,
                    'description': 'MACD line value',
                    'prediction_logic': 'Higher values indicate stronger bullish momentum'
                }
                self.feature_metadata[f'macd_signal_{fast}_{slow}_{signal}'] = {
                    'category': 'momentum',
                    'indicator': 'MACD Signal Line',
                    'fast': fast,
                    'slow': slow,
                    'signal': signal,
                    'description': 'MACD signal line value',
                    'prediction_logic': 'Used to identify momentum crossovers'
                }
                self.feature_metadata[f'macd_hist_{fast}_{slow}_{signal}'] = {
                    'category': 'momentum',
                    'indicator': 'MACD Histogram',
                    'fast': fast,
                    'slow': slow,
                    'signal': signal,
                    'description': 'Difference between MACD line and signal line',
                    'prediction_logic': 'Increasing histogram suggests strengthening momentum'
                }
        
        # Stochastic Features
        if 'stoch' in params and TALIB_AVAILABLE:
            for fastk, slowk, slowd in params['stoch']:
                # Calculate Stochastic (shifted by 1 to prevent look-ahead)
                slowk_line, slowd_line = talib.STOCH(
                    df['High'],
                    df['Low'],
                    Close,
                    fastk_period=fastk,
                    slowk_period=slowk,
                    slowk_matype=0,
                    slowd_period=slowd,
                    slowd_matype=0
                )
                slowk_line = slowk_line.shift(1)
                slowd_line = slowd_line.shift(1)
                
                # %K line
                features[f'stoch_k_{fastk}_{slowk}'] = slowk_line
                
                # %D line
                features[f'stoch_d_{slowk}_{slowd}'] = slowd_line
                
                # %K - %D difference
                stoch_diff = slowk_line - slowd_line
                features[f'stoch_diff_{fastk}_{slowk}_{slowd}'] = stoch_diff
                
                # Normalized stochastic position (0-1 range)
                stoch_position = slowk_line / 100.0
                features[f'stoch_position_{fastk}_{slowk}'] = stoch_position
                
                # Store metadata
                self.feature_metadata[f'stoch_k_{fastk}_{slowk}'] = {
                    'category': 'momentum',
                    'indicator': '%K Line',
                    'fastk': fastk,
                    'slowk': slowk,
                    'description': 'Stochastic %K line value',
                    'prediction_logic': 'Measures relative position within recent price range'
                }
                self.feature_metadata[f'stoch_d_{slowk}_{slowd}'] = {
                    'category': 'momentum',
                    'indicator': '%D Line',
                    'slowk': slowk,
                    'slowd': slowd,
                    'description': 'Stochastic %D line value (signal line)',
                    'prediction_logic': 'Smoothed version of %K line'
                }
                self.feature_metadata[f'stoch_diff_{fastk}_{slowk}_{slowd}'] = {
                    'category': 'momentum',
                    'indicator': 'Stochastic Difference',
                    'fastk': fastk,
                    'slowk': slowk,
                    'slowd': slowd,
                    'description': 'Difference between %K and %D lines',
                    'prediction_logic': 'Positive values suggest bullish momentum'
                }
                self.feature_metadata[f'stoch_position_{fastk}_{slowk}'] = {
                    'category': 'momentum',
                    'indicator': 'Stochastic Position',
                    'fastk': fastk,
                    'slowk': slowk,
                    'description': 'Normalized position within stochastic range',
                    'prediction_logic': 'Values Closer to 1 suggest overbought conditions'
                }
        
        # CCI Features
        if 'cci' in params and TALIB_AVAILABLE:
            for period in params['cci']:
                # Calculate CCI (shifted by 1 to prevent look-ahead)
                cci = talib.CCI(
                    df['High'],
                    df['Low'],
                    Close,
                    timeperiod=period
                ).shift(1)
                
                # Raw CCI
                features[f'cci_{period}'] = cci
                
                # Normalized CCI (0-1 range based on typical bounds)
                cci_norm = (cci + 100) / 200.0  # Assuming CCI typically ranges from -100 to +100
                features[f'cci_{period}_norm'] = cci_norm
                
                # CCI momentum
                cci_momentum = cci.diff()
                features[f'cci_{period}_momentum'] = cci_momentum
                
                # Store metadata
                self.feature_metadata[f'cci_{period}'] = {
                    'category': 'momentum',
                    'indicator': 'CCI',
                    'period': period,
                    'description': 'Commodity Channel Index value',
                    'prediction_logic': 'Measures deviation from statistical mean'
                }
                self.feature_metadata[f'cci_{period}_norm'] = {
                    'category': 'momentum',
                    'indicator': 'Normalized CCI',
                    'period': period,
                    'description': 'CCI normalized to 0-1 range',
                    'prediction_logic': 'Easier for models to interpret magnitude'
                }
                self.feature_metadata[f'cci_{period}_momentum'] = {
                    'category': 'momentum',
                    'indicator': 'CCI Momentum',
                    'period': period,
                    'description': 'Change in CCI value',
                    'prediction_logic': 'Measures acceleration of price movement'
                }
        
        return features
    
    def generate_overlap_features(self, **params) -> Dict[str, pd.Series]:
        features = {}
        df = self.df.copy()
        Close = df['Close']
        
        # Simple Moving Average Features
        if 'sma' in params and TALIB_AVAILABLE:
            for period in params['sma']:
                # Calculate SMA (shifted by 1 to prevent look-ahead)
                sma = talib.SMA(Close, timeperiod=period).shift(1)
                self.features[f'sma_{period}'] = sma
                
                # Price to SMA ratio (key feature for prediction)
                price_to_sma = Close / sma
                features[f'price_sma_{period}_ratio'] = price_to_sma
                
                # SMA slope (trend strength)
                sma_slope = sma.diff()
                features[f'sma_{period}_slope'] = sma_slope
                
                # SMA acceleration (change in slope)
                sma_accel = sma_slope.diff()
                features[f'sma_{period}_accel'] = sma_accel
                
                # Price to SMA distance (in standard deviations)
                if 'stddev_20_1' in self.features:
                    price_sma_distance = (Close - sma) / self.features['stddev_20_1']
                    features[f'price_sma_{period}_distance'] = price_sma_distance
                
                # Store metadata
                self.feature_metadata[f'price_sma_{period}_ratio'] = {
                    'category': 'overlap',
                    'indicator': 'Price/SMA Ratio',
                    'period': period,
                    'description': 'Ratio of price to simple moving average',
                    'prediction_logic': 'Values > 1 indicate price above trend'
                }
                self.feature_metadata[f'sma_{period}_slope'] = {
                    'category': 'overlap',
                    'indicator': 'SMA Slope',
                    'period': period,
                    'description': 'Rate of change of the moving average',
                    'prediction_logic': 'Positive values indicate uptrend'
                }
                self.feature_metadata[f'sma_{period}_accel'] = {
                    'category': 'overlap',
                    'indicator': 'SMA Acceleration',
                    'period': period,
                    'description': 'Change in SMA slope',
                    'prediction_logic': 'Measures trend strength acceleration'
                }
                if 'price_sma_{period}_distance' in features:
                    self.feature_metadata[f'price_sma_{period}_distance'] = {
                        'category': 'overlap',
                        'indicator': 'Price/SMA Distance',
                        'period': period,
                        'description': 'Price distance from SMA in standard deviations',
                        'prediction_logic': 'Measures how far price is from trend'
                    }
        
        # Exponential Moving Average Features
        if 'ema' in params and TALIB_AVAILABLE:
            for period in params['ema']:
                # Calculate EMA (shifted by 1 to prevent look-ahead)
                ema = talib.EMA(Close, timeperiod=period).shift(1)
                self.features[f'ema_{period}'] = ema
                
                # Price to EMA ratio
                price_to_ema = Close / ema
                features[f'price_ema_{period}_ratio'] = price_to_ema
                
                # Short vs. long EMA ratio (if multiple periods)
                if period == 12 and 'ema_26' in self.features:
                    ema_26 = self.features['ema_26']
                    short_long_ema_ratio = ema / ema_26
                    features['short_long_ema_ratio'] = short_long_ema_ratio
                    self.feature_metadata['short_long_ema_ratio'] = {
                        'category': 'overlap',
                        'indicator': 'Short/Long EMA Ratio',
                        'description': 'Ratio of short EMA to long EMA',
                        'prediction_logic': 'Values > 1 indicate short-term trend above long-term trend'
                    }
                
                # Store metadata
                self.feature_metadata[f'price_ema_{period}_ratio'] = {
                    'category': 'overlap',
                    'indicator': 'Price/EMA Ratio',
                    'period': period,
                    'description': 'Ratio of price to exponential moving average',
                    'prediction_logic': 'Measures relative position to trend'
                }
        
        # Bollinger Bands Features
        if 'bbands' in params and TALIB_AVAILABLE:
            for period, nbdevup, nbdevdn in params['bbands']:
                # Calculate Bollinger Bands (shifted by 1 to prevent look-ahead)
                upper, middle, Lower = talib.BBANDS(
                    Close,
                    timeperiod=period,
                    nbdevup=nbdevup,
                    nbdevdn=nbdevdn,
                    matype=0
                )
                upper = upper.shift(1)
                middle = middle.shift(1)
                Lower = Lower.shift(1)
                
                # %B (position within Bollinger Bands)
                percent_b = (Close - Lower) / (upper - Lower)
                features[f'bb_{period}_{nbdevup}_{nbdevdn}_percent_b'] = percent_b
                
                # Bandwidth (normalized volatility measure)
                bandwidth = (upper - Lower) / middle
                features[f'bb_{period}_{nbdevup}_{nbdevdn}_bandwidth'] = bandwidth
                
                # Z-score (standard deviations from middle band)
                z_score = (Close - middle) / ((upper - Lower) / (2 * nbdevup))
                features[f'bb_{period}_{nbdevup}_{nbdevdn}_zscore'] = z_score
                
                # Store metadata
                self.feature_metadata[f'bb_{period}_{nbdevup}_{nbdevdn}_percent_b'] = {
                    'category': 'overlap',
                    'indicator': 'Bollinger %B',
                    'period': period,
                    'nbdevup': nbdevup,
                    'nbdevdn': nbdevdn,
                    'description': 'Position within Bollinger Bands (0-1 range)',
                    'prediction_logic': 'Values > 0.8 suggest overbought, < 0.2 suggest oversold'
                }
                self.feature_metadata[f'bb_{period}_{nbdevup}_{nbdevdn}_bandwidth'] = {
                    'category': 'overlap',
                    'indicator': 'Bollinger Bandwidth',
                    'period': period,
                    'nbdevup': nbdevup,
                    'nbdevdn': nbdevdn,
                    'description': 'Normalized measure of volatility',
                    'prediction_logic': 'Higher values indicate Higher volatility'
                }
                self.feature_metadata[f'bb_{period}_{nbdevup}_{nbdevdn}_zscore'] = {
                    'category': 'overlap',
                    'indicator': 'Bollinger Z-Score',
                    'period': period,
                    'nbdevup': nbdevup,
                    'nbdevdn': nbdevdn,
                    'description': 'Standard deviations from middle band',
                    'prediction_logic': 'Measures how far price is from mean in volatility terms'
                }
        
        return features
    
    def generate_volatility_features(self, **params) -> Dict[str, pd.Series]:
    
        features = {}
        df = self.df.copy()
        Close = df['Close']
        
        # Average True Range Features
        if 'atr' in params and TALIB_AVAILABLE:
            for period in params['atr']:
                # Calculate ATR (shifted by 1 to prevent look-ahead)
                atr = talib.ATR(
                    df['High'],
                    df['Low'],
                    Close,
                    timeperiod=period
                ).shift(1)
                self.features[f'atr_{period}'] = atr
                
                # Normalized ATR (as percentage of price)
                natr = talib.NATR(
                    df['High'],
                    df['Low'],
                    Close,
                    timeperiod=period
                ).shift(1)
                features[f'natr_{period}'] = natr
                
                # ATR ratio (current vs. historical)
                atr_ma = atr.rolling(20).mean()
                atr_ratio = atr / atr_ma
                features[f'atr_{period}_ratio'] = atr_ratio
                
                # FIXED: Volatility regime (Low/medium/High) using proper rolling quantile calculation
                # Calculate rolling 25th and 75th percentiles separately
                rolling_Lower = atr.rolling(100).quantile(0.25)
                rolling_upper = atr.rolling(100).quantile(0.75)
                
                # Determine volatility regime based on current ATR vs rolling quantiles
                volatility_regime = pd.Series(1, index=atr.index)  # Default to medium volatility (1)
                volatility_regime[atr < rolling_Lower] = 0  # Low volatility (0)
                volatility_regime[atr > rolling_upper] = 2  # High volatility (2)
                
                # Handle NaN values (first 99 days with 100-day window)
                volatility_regime = volatility_regime.fillna(1)  # Fill with medium volatility
                
                features[f'vol_regime_{period}'] = volatility_regime
                
                # Store metadata
                self.feature_metadata[f'natr_{period}'] = {
                    'category': 'volatility',
                    'indicator': 'Normalized ATR',
                    'period': period,
                    'description': 'ATR as percentage of price',
                    'prediction_logic': 'Measures volatility relative to price level'
                }
                self.feature_metadata[f'atr_{period}_ratio'] = {
                    'category': 'volatility',
                    'indicator': 'ATR Ratio',
                    'period': period,
                    'description': 'Current ATR relative to its moving average',
                    'prediction_logic': 'Values > 1 indicate increasing volatility'
                }
                self.feature_metadata[f'vol_regime_{period}'] = {
                    'category': 'volatility',
                    'indicator': 'Volatility Regime',
                    'period': period,
                    'description': 'Categorical volatility regime (0=Low, 1=medium, 2=High)',
                    'prediction_logic': 'Helps models adapt to different volatility environments'
                }
    
        return features

    def generate_Volume_features(self, **params) -> Dict[str, pd.Series]:
        """
        Generate Volume-based features using TA-Lib indicators.
        Includes: OBV, AD, CMF, etc.
        
        Key differences from signal generation:
        - Returns normalized Volume measures
        - Creates Volume-price correlation features
        - Adds Volume trend features
        - All features shifted by 1 to prevent look-ahead bias
        """
        features = {}
        df = self.df.copy()
        Close = df['Close']
        Volume = df['Volume']
        
        # On Balance Volume Features
        if 'obv' in params and TALIB_AVAILABLE:
            # Calculate OBV (shifted by 1 to prevent look-ahead)
            obv = talib.OBV(Close, Volume).shift(1)
            self.features['obv'] = obv
            
            # OBV slope (trend strength)
            obv_slope = obv.diff()
            features['obv_slope'] = obv_slope
            
            # OBV momentum (acceleration)
            obv_momentum = obv_slope.diff()
            features['obv_momentum'] = obv_momentum
            
            # Volume-weighted price change
            price_change = Close.pct_change()
            vw_price_change = price_change * Volume
            features['vw_price_change'] = vw_price_change
            
            # Store metadata
            self.feature_metadata['obv_slope'] = {
                'category': 'Volume',
                'indicator': 'OBV Slope',
                'description': 'Rate of change of On-Balance Volume',
                'prediction_logic': 'Positive values suggest buying pressure'
            }
            self.feature_metadata['obv_momentum'] = {
                'category': 'Volume',
                'indicator': 'OBV Momentum',
                'description': 'Change in OBV slope',
                'prediction_logic': 'Measures acceleration of Volume trend'
            }
            self.feature_metadata['vw_price_change'] = {
                'category': 'Volume',
                'indicator': 'Volume-Weighted Price Change',
                'description': 'Price change weighted by Volume',
                'prediction_logic': 'Measures price movement significance based on Volume'
            }
        
        # Chaikin Money FLow Features
        if 'cmf' in params and TALIB_AVAILABLE:
            for period in params['cmf']:
                # Calculate CMF (shifted by 1 to prevent look-ahead)
                cmf = talib.ADOSC(
                    df['High'],
                    df['Low'],
                    Close,
                    Volume,
                    fastperiod=3,
                    slowperiod=period
                ).shift(1)
                
                # Raw CMF
                features[f'cmf_{period}'] = cmf
                
                # CMF momentum
                cmf_momentum = cmf.diff()
                features[f'cmf_{period}_momentum'] = cmf_momentum
                
                # CMF normalized by volatility
                if 'natr_14' in self.features:
                    cmf_norm = cmf / self.features['natr_14']
                    features[f'cmf_{period}_norm'] = cmf_norm
                    self.feature_metadata[f'cmf_{period}_norm'] = {
                        'category': 'Volume',
                        'indicator': 'Normalized CMF',
                        'period': period,
                        'description': 'CMF normalized by volatility',
                        'prediction_logic': 'Measures money fLow relative to volatility'
                    }
                
                # Store metadata
                self.feature_metadata[f'cmf_{period}'] = {
                    'category': 'Volume',
                    'indicator': 'Chaikin Money FLow',
                    'period': period,
                    'description': 'Money fLow indicator measuring accumulation/distribution',
                    'prediction_logic': 'Positive values suggest accumulation'
                }
                self.feature_metadata[f'cmf_{period}_momentum'] = {
                    'category': 'Volume',
                    'indicator': 'CMF Momentum',
                    'period': period,
                    'description': 'Change in CMF value',
                    'prediction_logic': 'Measures acceleration of money fLow'
                }
        
        return features
    
    def generate_cycle_features(self, **params) -> Dict[str, pd.Series]:
        """
        Generate cycle-based features using TA-Lib indicators.
        Includes: Hilbert Transform, etc.
        
        Key differences from signal generation:
        - Returns continuous phase values instead of discrete signals
        - Creates cycle period features
        - Adds cycle strength metrics
        - All features shifted by 1 to prevent look-ahead bias
        """
        features = {}
        df = self.df.copy()
        Close = df['Close']
        
        # Hilbert Transform Features
        if 'ht_dcperiod' in params and TALIB_AVAILABLE:
            # Calculate Dominant Cycle Period (shifted by 1 to prevent look-ahead)
            dcperiod = talib.HT_DCPERIOD(Close).shift(1)
            self.features['ht_dcperiod'] = dcperiod
            
            # Normalized cycle period (relative to historical)
            dcperiod_ma = dcperiod.rolling(20).mean()
            dcperiod_ratio = dcperiod / dcperiod_ma
            features['ht_dcperiod_ratio'] = dcperiod_ratio
            
            # Cycle period acceleration
            dcperiod_slope = dcperiod.diff()
            dcperiod_accel = dcperiod_slope.diff()
            features['ht_dcperiod_accel'] = dcperiod_accel
            
            # Store metadata
            self.feature_metadata['ht_dcperiod_ratio'] = {
                'category': 'cycle',
                'indicator': 'Cycle Period Ratio',
                'description': 'Current cycle period relative to its moving average',
                'prediction_logic': 'Values > 1 indicate lengthening cycles'
            }
            self.feature_metadata['ht_dcperiod_accel'] = {
                'category': 'cycle',
                'indicator': 'Cycle Period Acceleration',
                'description': 'Change in cycle period slope',
                'prediction_logic': 'Measures rate of change in cycle length'
            }
        
        # Hilbert Transform Phasor Features
        if 'ht_phasor' in params and TALIB_AVAILABLE:
            # Calculate Phasor Components (shifted by 1 to prevent look-ahead)
            inphase, quadrature = talib.HT_PHASOR(Close)
            inphase = inphase.shift(1)
            quadrature = quadrature.shift(1)
            
            # Hilbert phase (0-360 degrees)
            phase = np.degrees(np.arctan2(quadrature, inphase)) % 360
            features['ht_phase'] = phase
            
            # Hilbert phase velocity (rate of change)
            phase_velocity = phase.diff()
            features['ht_phase_velocity'] = phase_velocity
            
            # Hilbert phase acceleration
            phase_accel = phase_velocity.diff()
            features['ht_phase_accel'] = phase_accel
            
            # Store metadata
            self.feature_metadata['ht_phase'] = {
                'category': 'cycle',
                'indicator': 'Hilbert Phase',
                'description': 'Current phase in the market cycle (0-360 degrees)',
                'prediction_logic': 'Phase values predict turning points in price'
            }
            self.feature_metadata['ht_phase_velocity'] = {
                'category': 'cycle',
                'indicator': 'Phase Velocity',
                'description': 'Rate of change of the market cycle phase',
                'prediction_logic': 'Measures speed of cycle progression'
            }
            self.feature_metadata['ht_phase_accel'] = {
                'category': 'cycle',
                'indicator': 'Phase Acceleration',
                'description': 'Change in phase velocity',
                'prediction_logic': 'Measures acceleration of cycle progression'
            }
        
        return features
    
    def generate_price_features(self, **params) -> Dict[str, pd.Series]:
        """
        Generate price-based features.
        Includes: Typical Price, Median Price, etc.
        
        Key differences from signal generation:
        - Returns price ratios instead of binary signals
        - Creates price momentum features
        - Adds price volatility metrics
        - All features shifted by 1 to prevent look-ahead bias
        """
        features = {}
        df = self.df.copy()
        High = df['High']
        Low = df['Low']
        Close = df['Close']
        Open_price = df['Open']
        
        # Price Range Features
        # True Range (shifted by 1 to prevent look-ahead)
        true_range = talib.TRANGE(High, Low, Close).shift(1)
        self.features['true_range'] = true_range
        
        # Normalized true range (ATR as percentage)
        natr = talib.NATR(High, Low, Close, timeperiod=14).shift(1)
        features['natr_14'] = natr
        
        # Price momentum features
        price_momentum_1 = Close.pct_change(1)
        features['price_momentum_1'] = price_momentum_1
        
        price_momentum_5 = Close.pct_change(5)
        features['price_momentum_5'] = price_momentum_5
        
        price_momentum_20 = Close.pct_change(20)
        features['price_momentum_20'] = price_momentum_20
        
        # Price acceleration features
        price_accel_1 = price_momentum_1.diff()
        features['price_accel_1'] = price_accel_1
        
        # Volatility-adjusted momentum
        if 'natr_14' in features:
            vol_adj_momentum = price_momentum_1 / features['natr_14']
            features['vol_adj_momentum'] = vol_adj_momentum
        
        # Price pattern features
        # Candle body to range ratio
        candle_body = abs(Close - Open_price)
        candle_range = High - Low
        body_to_range = candle_body / candle_range.replace(0, np.nan)
        features['body_to_range'] = body_to_range
        
        # Upper shadow ratio
        upper_shadow = High - np.maximum(Open_price, Close)
        upper_shadow_ratio = upper_shadow / candle_range.replace(0, np.nan)
        features['upper_shadow_ratio'] = upper_shadow_ratio
        
        # Lower shadow ratio
        Lower_shadow = np.minimum(Open_price, Close) - Low
        Lower_shadow_ratio = Lower_shadow / candle_range.replace(0, np.nan)
        features['Lower_shadow_ratio'] = Lower_shadow_ratio
        
        # Store metadata
        self.feature_metadata['price_momentum_1'] = {
            'category': 'price',
            'indicator': '1-Day Price Momentum',
            'description': '1-day percentage price change',
            'prediction_logic': 'Short-term price trend'
        }
        self.feature_metadata['price_momentum_5'] = {
            'category': 'price',
            'indicator': '5-Day Price Momentum',
            'description': '5-day percentage price change',
            'prediction_logic': 'Medium-term price trend'
        }
        self.feature_metadata['price_momentum_20'] = {
            'category': 'price',
            'indicator': '20-Day Price Momentum',
            'description': '20-day percentage price change',
            'prediction_logic': 'Long-term price trend'
        }
        self.feature_metadata['body_to_range'] = {
            'category': 'price',
            'indicator': 'Candle Body to Range Ratio',
            'description': 'Ratio of candle body to total price range',
            'prediction_logic': 'Measures price conviction (Higher values = stronger trend)'
        }
        self.feature_metadata['upper_shadow_ratio'] = {
            'category': 'price',
            'indicator': 'Upper Shadow Ratio',
            'description': 'Ratio of upper shadow to total price range',
            'prediction_logic': 'Measures rejection of Higher prices'
        }
        self.feature_metadata['Lower_shadow_ratio'] = {
            'category': 'price',
            'indicator': 'Lower Shadow Ratio',
            'description': 'Ratio of Lower shadow to total price range',
            'prediction_logic': 'Measures rejection of Lower prices'
        }
        
        return features
    
    def generate_statistic_features(self, **params) -> Dict[str, pd.Series]:
        """
        Generate statistical features.
        Includes: Linear Regression, Standard Deviation, etc.
        
        CORRECTED IMPLEMENTATION:
        - Properly handles parameter structure for stddev
        - Works with both single configuration and multiple configurations
        - Maintains correct timing alignment
        """
        features = {}
        df = self.df.copy()
        Close = df['Close']
        
        # Linear Regression Features
        if 'linearreg' in params and TALIB_AVAILABLE:
            for period in params['linearreg']:
                # Calculate Linear Regression (shifted by 1 to prevent look-ahead)
                reg_line = talib.LINEARREG(Close, timeperiod=period).shift(1)
                self.features[f'linearreg_{period}'] = reg_line
                
                # Regression slope (normalized by price)
                reg_slope = talib.LINEARREG_SLOPE(Close, timeperiod=period).shift(1)
                reg_slope_norm = reg_slope / Close.shift(1)
                features[f'linearreg_slope_norm_{period}'] = reg_slope_norm
                
                # Calculate R-squared manually (since LINEARREG_RSQ is not available)
                if period > 1:  # Need at least 2 points for correlation
                    # Calculate correlation between price and regression line
                    correlation = Close.rolling(period).corr(reg_line)
                    # R-squared is the square of correlation
                    r_squared = correlation ** 2
                    # Handle NaN and inf values
                    r_squared = r_squared.replace([np.inf, -np.inf], np.nan).fillna(0)
                    features[f'linearreg_rsq_{period}'] = r_squared
                    self.feature_metadata[f'linearreg_rsq_{period}'] = {
                        'category': 'statistic',
                        'indicator': 'Regression R-squared',
                        'period': period,
                        'description': 'Coefficient of determination for linear regression',
                        'prediction_logic': 'Higher values indicate stronger linear trend'
                    }
                
                # Price to regression line distance
                price_to_reg = (Close - reg_line) / reg_line
                features[f'price_reg_dist_{period}'] = price_to_reg
                
                # Store metadata
                self.feature_metadata[f'linearreg_slope_norm_{period}'] = {
                    'category': 'statistic',
                    'indicator': 'Normalized Regression Slope',
                    'period': period,
                    'description': 'Regression slope normalized by price',
                    'prediction_logic': 'Measures trend strength as percentage'
                }
                self.feature_metadata[f'price_reg_dist_{period}'] = {
                    'category': 'statistic',
                    'indicator': 'Price to Regression Distance',
                    'period': period,
                    'description': 'Distance from price to regression line',
                    'prediction_logic': 'Measures deviation from trend'
                }
        
        # Standard Deviation Features - FIXED PARAMETER HANDLING
        if 'stddev' in params and TALIB_AVAILABLE:
            # Handle different parameter structures:
            # Option 1: params['stddev'] = [(5, 1), (10, 2)] - list of tuples
            # Option 2: params['stddev'] = [5, 1] - single configuration as list
            # Option 3: params['stddev'] = (5, 1) - single configuration as tuple
            
            stddev_configs = []
            
            # Case 1: List of tuples - already in correct format
            if all(isinstance(x, tuple) for x in params['stddev']):
                stddev_configs = params['stddev']
            # Case 2: Single configuration as list or tuple
            elif not any(isinstance(x, (list, tuple)) for x in params['stddev']):
                # Convert [5, 1] or (5, 1) to [(5, 1)]
                stddev_configs = [tuple(params['stddev'])]
            # Case 3: Single configuration as two separate values
            elif isinstance(params['stddev'], (list, tuple)) and len(params['stddev']) == 2:
                stddev_configs = [tuple(params['stddev'])]
            
            # Process all configurations
            for config in stddev_configs:
                period, nbdev = config
                
                # Calculate Standard Deviation (shifted by 1 to prevent look-ahead)
                stddev = talib.STDDEV(Close, timeperiod=period, nbdev=nbdev).shift(1)
                self.features[f'stddev_{period}_{nbdev}'] = stddev
                
                # Price to standard deviation ratio
                price_stddev_ratio = stddev / Close.shift(1)
                features[f'price_stddev_ratio_{period}_{nbdev}'] = price_stddev_ratio
                
                # Volatility trend
                stddev_ma = stddev.rolling(10).mean()
                volatility_trend = stddev / stddev_ma
                features[f'volatility_trend_{period}_{nbdev}'] = volatility_trend
                
                # Store metadata
                self.feature_metadata[f'price_stddev_ratio_{period}_{nbdev}'] = {
                    'category': 'statistic',
                    'indicator': 'Price to Standard Deviation Ratio',
                    'period': period,
                    'nbdev': nbdev,
                    'description': 'Volatility relative to price level',
                    'prediction_logic': 'Measures normalized volatility'
                }
                self.feature_metadata[f'volatility_trend_{period}_{nbdev}'] = {
                    'category': 'statistic',
                    'indicator': 'Volatility Trend',
                    'period': period,
                    'nbdev': nbdev,
                    'description': 'Current volatility relative to its moving average',
                    'prediction_logic': 'Values > 1 indicate increasing volatility'
                }
        
        return features
    
    def generate_all_TALib_features(self,
        include_categories: Optional[List[str]] = None,
        custom_periods: Optional[Dict[str, Union[List[int], List[Tuple]]]] = None
    ) -> pd.DataFrame:
        
        # Define default categories if none specified
        if include_categories is None:
            include_categories = [
                'momentum', 'Volume', 'volatility', 'overlap', 
                'cycle', 'price', 'statistic',
                #  'pattern',
            ]
        
        default_periods = {
            # Momentum indicators - heavily focused on very short-term (3-10 days)
            'adx': [5, 8],  # Shorter than standard 14
            'adxr': [5, 8],
            'apo': [(5, 10), (8, 15)],  # Shorter fast/slow periods
            'cci': [6, 10],  # Shorter than standard 14
            'cmo': [5, 8],   # Shorter than standard 9-14
            'dx': [5, 8],
            'macd': (5, 10, 5),  # Shorter periods: fast=5, slow=10, signal=5
            'mom': [2, 3, 5],   # Very short-term momentum (added more granular options)
            'plus_di': [5, 8],
            'minus_di': [5, 8],
            'plus_dm': [5, 8],
            'minus_dm': [5, 8],
            'ppo': (5, 10),  # Shorter periods
            'roc': [1, 2, 3, 5],   # Ultra-short to short-term rate of change
            'rsi': [5, 7, 10],  # Shorter than standard 14 (added more granular options)
            'stoch': (3, 2, 2),  # Shorter periods
            'stochf': (3, 2),    # Shorter periods
            'stochrsi': (5, 3, 2),  # Shorter periods
            'trix': [3, 5, 8],  # Shorter than standard 15-20
            'ultosc': (3, 5, 8),  # Shorter periods
            'willr': [5, 7, 10],  # Shorter than standard 14
            
            # Volume indicators - focus on short-term
            'adosc': [(2, 5), (3, 8)],  # Shorter periods
            'mfi': [5, 7, 10],  # Shorter than standard 14
            
            # Volatility indicators - focus on short-term
            'atr': [5, 7, 10],  # Shorter than standard 14
            'natr': [5, 7, 10],
            
            # Overlap studies - heavily focused on short-term
            'bbands': [
                (3, 1.5, 1.5), (3, 2.0, 2.0),  # Very tight bands for immediate reaction
                (5, 1.5, 1.5), (5, 2.0, 2.0),  # Standard short-term bands
                (8, 2.0, 2.0)  # Slightly longer reference
            ],
            'dema': [3, 5, 8, 13],  # Shorter periods with Fibonacci sequence
            'ema': [3, 5, 8, 13, 21],  # Fibonacci-inspired short periods
            'kama': [3, 5, 8],
            'sma': [2, 3, 5, 8, 13, 21],  # Added very short 2-period SMA
            't3': [3, 5],
            'tema': [3, 5, 8],
            'trima': [3, 5, 8],
            'wma': [3, 5, 8],
            
            # Statistic functions - focus on very short-term
            'beta': [3, 5],
            'correl': [2, 3, 5],  # Added ultra-short correlation
            'linearreg': [3, 5, 8],
            'stddev': [
                (2, 1), (2, 1.5), (2, 2),  # Ultra-short volatility
                (3, 1), (3, 1.5), (3, 2),  # Very short volatility
                (5, 1), (5, 1.5), (5, 2)   # Short volatility
            ],
            'var': [2, 3, 5],
            'max_min': [2, 3, 5],  # Added very short lookback
            'sum': [2, 3, 5]
        }
        # Merge custom periods with defaults
        if custom_periods:
            for key, value in custom_periods.items():
                if key in default_periods:
                    default_periods[key] = value
        
        # Verify required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in self.df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.df.columns]
            raise ValueError(f"DataFrame missing required columns: {missing}. Must contain {required_cols}")
        
        # Convert OHLCV to float64 (critical for TA-Lib compatibility)
        Open_prices = self.df['Open'].astype(np.float64).values
        High_prices = self.df['High'].astype(np.float64).values
        Low_prices = self.df['Low'].astype(np.float64).values
        Close_prices = self.df['Close'].astype(np.float64).values
        Volume = self.df['Volume'].astype(np.float64).values
        
        # Create a dictionary to collect features (more efficient than adding columns one by one)
        features_dict = {}
        
        # Helper function for safe TA computation with error handling
        def safe_compute(func, *args, **kwargs):
            try:
                result = func(*args, **kwargs)
                # Handle cases where function returns multiple arrays
                if isinstance(result, tuple):
                    return [np.array(r, dtype=np.float64) for r in result]
                return np.array(result, dtype=np.float64)
            except Exception as e:
                warnings.warn(f"TA-Lib function {func.__name__} failed: {str(e)}")
                return np.full(len(Close_prices), np.nan, dtype=np.float64)
        
        # 1. Momentum Indicators
        if 'momentum' in include_categories:
            # ADX - Average Directional Movement Index
            for period in default_periods['adx']:
                features_dict[f'adx_{period}'] = safe_compute(talib.ADX, High_prices, Low_prices, Close_prices, timeperiod=period)
            
            # ADXR - Average Directional Movement Index Rating
            for period in default_periods['adxr']:
                features_dict[f'adxr_{period}'] = safe_compute(talib.ADXR, High_prices, Low_prices, Close_prices, timeperiod=period)
            
            # APO - Absolute Price Oscillator
            for (fast, slow) in default_periods['apo']:
                features_dict[f'apo_{fast}_{slow}'] = safe_compute(talib.APO, Close_prices, fastperiod=fast, slowperiod=slow)
            
            # CCI - Commodity Channel Index
            for period in default_periods['cci']:
                features_dict[f'cci_{period}'] = safe_compute(talib.CCI, High_prices, Low_prices, Close_prices, timeperiod=period)
            
            # CMO - Chande Momentum Oscillator
            for period in default_periods['cmo']:
                features_dict[f'cmo_{period}'] = safe_compute(talib.CMO, Close_prices, timeperiod=period)
            
            # MACD - Moving Average Convergence/Divergence
            fast, slow, signal = default_periods['macd']
            macd, signal_line, hist = safe_compute(
                talib.MACD, Close_prices, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            features_dict['macd'] = macd
            features_dict['macd_signal'] = signal_line
            features_dict['macd_hist'] = hist
            
            # MOM - Momentum
            for period in default_periods['mom']:
                features_dict[f'mom_{period}'] = safe_compute(talib.MOM, Close_prices, timeperiod=period)
            
            # Directional Movement Indicators
            for period in default_periods['plus_di']:
                features_dict[f'plus_di_{period}'] = safe_compute(talib.PLUS_DI, High_prices, Low_prices, Close_prices, timeperiod=period)
            
            for period in default_periods['minus_di']:
                features_dict[f'minus_di_{period}'] = safe_compute(talib.MINUS_DI, High_prices, Low_prices, Close_prices, timeperiod=period)
            
            for period in default_periods['plus_dm']:
                features_dict[f'plus_dm_{period}'] = safe_compute(talib.PLUS_DM, High_prices, Low_prices, timeperiod=period)
            
            for period in default_periods['minus_dm']:
                features_dict[f'minus_dm_{period}'] = safe_compute(talib.MINUS_DM, High_prices, Low_prices, timeperiod=period)
            
            # Rate of Change indicators
            for period in default_periods['roc']:
                features_dict[f'roc_{period}'] = safe_compute(talib.ROC, Close_prices, timeperiod=period)
                features_dict[f'rocp_{period}'] = safe_compute(talib.ROCP, Close_prices, timeperiod=period)
                features_dict[f'rocr_{period}'] = safe_compute(talib.ROCR, Close_prices, timeperiod=period)
                features_dict[f'rocr100_{period}'] = safe_compute(talib.ROCR100, Close_prices, timeperiod=period)
            
            # RSI - Relative Strength Index
            for period in default_periods['rsi']:
                features_dict[f'rsi_{period}'] = safe_compute(talib.RSI, Close_prices, timeperiod=period)
            
            # Stochastic Oscillators
            k_period, slowk_period, d_period = default_periods['stoch']
            slowk, slowd = safe_compute(
                talib.STOCH, High_prices, Low_prices, Close_prices, 
                fastk_period=k_period, slowk_period=slowk_period, 
                slowd_period=d_period
            )
            features_dict['stoch_slowk'] = slowk
            features_dict['stoch_slowd'] = slowd
            
            k_period, d_period = default_periods['stochf']
            fastk, fastd = safe_compute(
                talib.STOCHF, High_prices, Low_prices, Close_prices,
                fastk_period=k_period, fastd_period=d_period
            )
            features_dict['stochf_fastk'] = fastk
            features_dict['stochf_fastd'] = fastd
            
            rsi_period, k_period, d_period = default_periods['stochrsi']
            stochrsi_k, stochrsi_d = safe_compute(
                talib.STOCHRSI, Close_prices, timeperiod=rsi_period,
                fastk_period=k_period, fastd_period=d_period
            )
            features_dict['stochrsi_k'] = stochrsi_k
            features_dict['stochrsi_d'] = stochrsi_d
            
            # TRIX - 1-day Rate-Of-Change of Triple EMA
            for period in default_periods['trix']:
                features_dict[f'trix_{period}'] = safe_compute(talib.TRIX, Close_prices, timeperiod=period)
            
            # ULTOSC - Ultimate Oscillator
            t1, t2, t3 = default_periods['ultosc']
            features_dict['ultosc'] = safe_compute(
                talib.ULTOSC, High_prices, Low_prices, Close_prices,
                timeperiod1=t1, timeperiod2=t2, timeperiod3=t3
            )
            
            # WILLR - Williams' %R
            for period in default_periods['willr']:
                features_dict[f'willr_{period}'] = safe_compute(talib.WILLR, High_prices, Low_prices, Close_prices, timeperiod=period)
        
        # 2. Volume Indicators
        if 'Volume' in include_categories:
            # AD - Chaikin A/D Line
            features_dict['ad'] = safe_compute(talib.AD, High_prices, Low_prices, Close_prices, Volume)
            
            # ADOSC - Chaikin A/D Oscillator
            for (fast, slow) in default_periods['adosc']:
                features_dict[f'adosc_{fast}_{slow}'] = safe_compute(
                    talib.ADOSC, High_prices, Low_prices, Close_prices, Volume,
                    fastperiod=fast, slowperiod=slow
                )
            
            # OBV - On Balance Volume
            features_dict['obv'] = safe_compute(talib.OBV, Close_prices, Volume)
            
            # MFI - Money FLow Index
            for period in default_periods['mfi']:
                features_dict[f'mfi_{period}'] = safe_compute(
                    talib.MFI, High_prices, Low_prices, Close_prices, Volume, timeperiod=period
                )
        
        # 3. Volatility Indicators
        if 'volatility' in include_categories:
            # ATR - Average True Range
            for period in default_periods['atr']:
                features_dict[f'atr_{period}'] = safe_compute(talib.ATR, High_prices, Low_prices, Close_prices, timeperiod=period)
            
            # NATR - Normalized Average True Range
            for period in default_periods['natr']:
                features_dict[f'natr_{period}'] = safe_compute(talib.NATR, High_prices, Low_prices, Close_prices, timeperiod=period)
            
            # TRANGE - True Range
            features_dict['trange'] = safe_compute(talib.TRANGE, High_prices, Low_prices, Close_prices)
        
        # 4. Overlap Studies
        if 'overlap' in include_categories:
            # BBANDS - Bollinger Bands
            for (period, nbdevup, nbdevdn) in default_periods['bbands']:
                upper, middle, Lower = safe_compute(
                    talib.BBANDS, Close_prices, timeperiod=period, 
                    nbdevup=nbdevup, nbdevdn=nbdevdn
                )
                features_dict[f'bb_upper_{period}_{nbdevup}_{nbdevdn}'] = upper
                features_dict[f'bb_middle_{period}_{nbdevup}_{nbdevdn}'] = middle
                features_dict[f'bb_Lower_{period}_{nbdevup}_{nbdevdn}'] = Lower
            
            # Various moving averages
            for period in default_periods['dema']:
                features_dict[f'dema_{period}'] = safe_compute(talib.DEMA, Close_prices, timeperiod=period)
            
            for period in default_periods['ema']:
                features_dict[f'ema_{period}'] = safe_compute(talib.EMA, Close_prices, timeperiod=period)
            
            features_dict['ht_trendline'] = safe_compute(talib.HT_TRENDLINE, Close_prices)
            
            for period in default_periods['kama']:
                features_dict[f'kama_{period}'] = safe_compute(talib.KAMA, Close_prices, timeperiod=period)
            
            for period in default_periods['sma']:
                features_dict[f'sma_{period}'] = safe_compute(talib.SMA, Close_prices, timeperiod=period)
            
            for period in default_periods['t3']:
                features_dict[f't3_{period}'] = safe_compute(talib.T3, Close_prices, timeperiod=period)
            
            for period in default_periods['tema']:
                features_dict[f'tema_{period}'] = safe_compute(talib.TEMA, Close_prices, timeperiod=period)
            
            for period in default_periods['trima']:
                features_dict[f'trima_{period}'] = safe_compute(talib.TRIMA, Close_prices, timeperiod=period)
            
            for period in default_periods['wma']:
                features_dict[f'wma_{period}'] = safe_compute(talib.WMA, Close_prices, timeperiod=period)
        
        # 5. Pattern Recognition
        if 'pattern' in include_categories:
            pattern_functions = [
                'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3STARSINSOUTH',
                'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY',
                'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER',
                'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR',
                'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN',
                'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON',
                'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH',
                'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW',
                'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING',
                'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR',
                'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI',
                'CDLTASUKIGAP', 'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS',
                'CDLXSIDEGAP3METHODS'
            ]
            
            for pattern in pattern_functions:
                func = getattr(talib, pattern)
                features_dict[pattern.lower()] = safe_compute(
                    func, Open_prices, High_prices, Low_prices, Close_prices
                )
        
        # 6. Cycle Indicators
        if 'cycle' in include_categories:
            features_dict['ht_dcperiod'] = safe_compute(talib.HT_DCPERIOD, Close_prices)
            features_dict['ht_dcphase'] = safe_compute(talib.HT_DCPHASE, Close_prices)
            
            inphase, quadrature = safe_compute(talib.HT_PHASOR, Close_prices)
            features_dict['ht_phasor_inphase'] = inphase
            features_dict['ht_phasor_quadrature'] = quadrature
            
            sine, leadsine = safe_compute(talib.HT_SINE, Close_prices)
            features_dict['ht_sine'] = sine
            features_dict['ht_leadsine'] = leadsine
            
            features_dict['ht_trendmode'] = safe_compute(talib.HT_TRENDMODE, Close_prices)
        
        # 7. Price Transform
        if 'price' in include_categories:
            features_dict['avgprice'] = safe_compute(talib.AVGPRICE, Open_prices, High_prices, Low_prices, Close_prices)
            features_dict['medprice'] = safe_compute(talib.MEDPRICE, High_prices, Low_prices)
            features_dict['typprice'] = safe_compute(talib.TYPPRICE, High_prices, Low_prices, Close_prices)
            features_dict['wclprice'] = safe_compute(talib.WCLPRICE, High_prices, Low_prices, Close_prices)
        
        # 8. Statistic Functions
        if 'statistic' in include_categories:
            # BETA - Beta
            for period in default_periods['beta']:
                features_dict[f'beta_{period}'] = safe_compute(talib.BETA, High_prices, Low_prices, timeperiod=period)
            
            # CORREL - Pearson's Correlation Coefficient
            for period in default_periods['correl']:
                features_dict[f'correl_{period}'] = safe_compute(talib.CORREL, High_prices, Low_prices, timeperiod=period)
            
            # Linear Regression
            for period in default_periods['linearreg']:
                features_dict[f'linreg_{period}'] = safe_compute(talib.LINEARREG, Close_prices, timeperiod=period)
                features_dict[f'linreg_angle_{period}'] = safe_compute(talib.LINEARREG_ANGLE, Close_prices, timeperiod=period)
                features_dict[f'linreg_intercept_{period}'] = safe_compute(talib.LINEARREG_INTERCEPT, Close_prices, timeperiod=period)
                features_dict[f'linreg_slope_{period}'] = safe_compute(talib.LINEARREG_SLOPE, Close_prices, timeperiod=period)
            
            # Standard Deviation and Variance
            for (period, nbdev) in default_periods['stddev']:
                features_dict[f'stddev_{period}_{nbdev}'] = safe_compute(
                    talib.STDDEV, Close_prices, timeperiod=period, nbdev=nbdev
                )
            
            for period in default_periods['var']:
                features_dict[f'var_{period}'] = safe_compute(talib.VAR, Close_prices, timeperiod=period)
            
            # Min/Max functions
            for period in default_periods['max_min']:
                features_dict[f'max_{period}'] = safe_compute(talib.MAX, Close_prices, timeperiod=period)
                features_dict[f'min_{period}'] = safe_compute(talib.MIN, Close_prices, timeperiod=period)
                features_dict[f'max_idx_{period}'] = safe_compute(talib.MAXINDEX, Close_prices, timeperiod=period)
                features_dict[f'min_idx_{period}'] = safe_compute(talib.MININDEX, Close_prices, timeperiod=period)
            
            # Summation
            for period in default_periods['sum']:
                features_dict[f'sum_{period}'] = safe_compute(talib.SUM, Close_prices, timeperiod=period)
        
        # Convert dictionary to DataFrame all at once (avoids fragmentation warning)
        features = pd.DataFrame(features_dict, index=self.df.index)
        self.features = features

        return features

    def get_feature_metadata(self, feature_name: Optional[str] = None) -> Dict:
        """Get metadata for features"""
        if feature_name is None:
            return self.feature_metadata
        if feature_name not in self.feature_metadata:
            raise ValueError(f"Feature '{feature_name}' metadata not found.")
        return self.feature_metadata[feature_name]
    
    def validate_features(self, feature_name: Optional[str] = None) -> Dict:
        """
        Validate feature quality and timing.
        
        Parameters:
        -----------
        feature_name : str, optional
            Name of feature series to validate
            If None, validates all features
            
        Returns:
        --------
        dict of validation results
        """
        results = {}
        features_to_validate = self.features.columns if feature_name is None else [feature_name]
        
        for name in features_to_validate:
            if name not in self.features:
                continue
                
            feature = self.features[name]
            
            # Validation checks
            validation = {
                'has_no_nan': not feature.isna().any(),
                'has_infinite': not np.isinf(feature).any(),
                'value_range': (feature.min(), feature.max()),
                'missing_ratio': feature.isna().mean(),
                'infinite_ratio': np.isinf(feature).mean(),
                'zero_ratio': (feature == 0).mean(),
                'constant_ratio': (feature == feature.iloc[0]).mean()
            }
            
            # Flag issues
            issues = []
            if validation['missing_ratio'] > 0.05:
                issues.append(f"High missing ratio ({validation['missing_ratio']:.1%} > 5%)")
            if validation['infinite_ratio'] > 0:
                issues.append(f"Contains infinite values ({validation['infinite_ratio']:.1%})")
            if validation['constant_ratio'] > 0.95:
                issues.append(f"Nearly constant values ({validation['constant_ratio']:.1%} > 95%)")
            
            validation['issues'] = issues
            validation['is_valid'] = len(issues) == 0
            results[name] = validation
        
        return results
    
    def analyze_feature_importance(
        self, 
        target: pd.Series,
        feature_name: Optional[str] = None
    ) -> Dict:
        """
        Analyze feature importance for predicting target (next day's Close).
        
        Parameters:
        -----------
        target : pd.Series
            Target variable (next day's Close)
        feature_name : str, optional
            Name of feature to analyze
            If None, analyzes all features
            
        Returns:
        --------
        dict of analysis results
        """
        results = {}
        features_to_analyze = self.features.columns if feature_name is None else [feature_name]
        
        # Align target with features (shifted by 1)
        aligned_target = target.shift(-1).loc[self.features.index]
        
        for name in features_to_analyze:
            if name not in self.features:
                continue
                
            feature = self.features[name]
            
            # Calculate correlation
            correlation = feature.corr(aligned_target)
            
            # Calculate information coefficient (IC)
            ic = correlation
            
            # Calculate predictive power (R-squared from simple linear regression)
            if len(feature.dropna()) > 2:
                X = feature.dropna()
                y = aligned_target.loc[X.index]
                if len(X) > 2:
                    # Simple linear regression
                    slope, intercept = np.polyfit(X, y, 1)
                    y_pred = slope * X + intercept
                    ss_total = ((y - y.mean()) ** 2).sum()
                    ss_residual = ((y - y_pred) ** 2).sum()
                    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
                else:
                    r_squared = 0
            else:
                r_squared = 0
            
            results[name] = {
                'feature': name,
                'correlation': correlation,
                'information_coefficient': ic,
                'r_squared': r_squared,
                'mean': feature.mean(),
                'std': feature.std(),
                'min': feature.min(),
                'max': feature.max()
            }
        
        return results
    
    def plot_feature(
        self, 
        feature_name: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot a feature against price data.
        
        Parameters:
        -----------
        feature_name : str
            Name of feature to plot
        start_date : str, optional
            Start date for the plot
        end_date : str, optional
            End date for the plot
        figsize : tuple, default=(10, 6)
            Figure size
        """
        if feature_name not in self.features:
            raise ValueError(f"Feature '{feature_name}' not found. Generate features first.")
        
        # Filter data if dates provided
        feature = self.features[feature_name]
        df = self.df.copy()
        if start_date:
            start_date = pd.Timestamp(start_date)
            feature = feature[feature.index >= start_date]
            df = df[df.index >= start_date]
        if end_date:
            end_date = pd.Timestamp(end_date)
            feature = feature[feature.index <= end_date]
            df = df[df.index <= end_date]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Plot price
        ax1.plot(df.index, df['Close'], label='Close Price', color='blue')
        ax1.set_title(f'Price and {feature_name} Feature', fontsize=14)
        ax1.set_ylabel('Price', fontsize=10)
        ax1.legend(fontsize=8)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot feature
        ax2.plot(feature.index, feature, label=feature_name, color='red')
        ax2.set_ylabel(feature_name, fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add target (next day's Close) if available
        if 'target_next_Close' in df.columns:
            ax1.scatter(
                df.index, 
                df['target_next_Close'], 
                color='green', 
                alpha=0.3, 
                s=10,
                label='Next Day Close'
            )
            ax1.legend(fontsize=8)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def get_all_features(self) -> pd.DataFrame:
        """Get all generated features"""
        return self.features
    
    def prepare_training_data(
        self,
        target_column: str = 'Close',
        horizon: int = 1,
        include_target: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare data for ML training.
        
        Parameters:
        -----------
        target_column : str, default='Close'
            Column to predict
        horizon : int, default=1
            How many periods ahead to predict
        include_target : bool, default=True
            Whether to include the target series
            
        Returns:
        --------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series, optional
            Target series (if include_target=True)
        """
        # Generate features
        X = self.get_all_features()
        
        # Prepare target (next day's Close)
        y = None
        if include_target:
            y = self.df[target_column].shift(-horizon)
            # Align with features
            y = y.loc[X.index]
        
        return X, y
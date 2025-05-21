"""
Risk-Averse Optimized Elliptical Arc Trading Bot with MT5 Integration and Backtesting Capabilities

This bot identifies trading opportunities by fitting elliptical arcs to price data,
combining this with advanced risk management, dynamic position sizing, and
technical indicator confirmations. It supports multi-symbol trading and
visualizes key market data and bot activity.

This version integrates a full backtesting engine, allowing you to switch between
live trading and historical simulation using a configuration flag.
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import time
from datetime import datetime, timedelta
import statistics
from sklearn.cluster import DBSCAN
from collections import deque
import os # For file path operations

# Import TA-Lib based indicators
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


class Config:
    """
    Enhanced Configuration Parameters for the trading bot.
    These parameters control the bot's behavior, from trading strategy
    to risk management and performance optimization.
    """
    # Basic settings
    symbols = ["EURUSD.m", "GBPUSD.m", "USDJPY.m", "XAUUSD.m"]  # List of symbols to trade
    timeframe = mt5.TIMEFRAME_H1  # Timeframe for market data (e.g., H1 for 1-hour candles)
    arc_lookback = 200  # Number of historical bars to consider for arc detection

    # Risk management
    max_risk_percent = 0.5  # Maximum percentage of account balance to risk per trade
    max_portfolio_risk = 2.0  # Maximum total risk exposure across all open positions
    profit_ratio = 2.0  # Desired Take Profit to Stop Loss ratio
    trailing_stop = True  # Enable trailing stops to lock in profits
    trailing_stop_trigger = 0.5  # Trigger trailing stop when profit reaches this ratio of TP

    # Position sizing
    min_lots = 0.01  # Minimum lot size for any trade
    max_lots = 0.5  # Maximum lot size to cap exposure
    position_scaling = True  # Scale position sizes based on signal conviction

    # Arc detection & quality
    max_arcs = 10  # Maximum number of top quality arcs to track for visualization/signals
    min_r2 = 0.75  # Minimum R-squared value for a valid arc fit (goodness of fit)
    min_curvature = 0.08  # Minimum curvature for a pronounced arc pattern
    quality_threshold = 0.65  # Minimum overall quality score for a signal to be considered

    # --- OPTIMIZATION PARAMETERS FOR ARC DETECTION (Used mainly in backtest mode) ---
    # To make analysis very quick, we reduce the number of window sizes
    # and increase the step size for sliding window arc detection.
    # This reduces the number of computationally expensive curve_fit calls.
    # Trade-off: Might miss some valid arcs if the sampling is too sparse.
    arc_detection_sizes = [60, 80] # Reduced for speed in backtest
    arc_detection_step_multiplier = 0.4 # Increased for larger steps in backtest

    # Advanced features
    anti_correlation_min = -0.6  # Not currently used in the provided code, but good for future expansion
    max_open_trades = 3  # Limit total number of concurrent open trades
    max_open_sells = 1  # Limit to only one active SELL position at a time (risk control)

    # Performance optimization
    momentum_lookback = 20  # Periods for momentum indicator calculation
    volatility_lookback = 14  # Periods for volatility (ATR) calculation

    # System settings
    visual_update_seconds = 15  # How often to refresh the visualization in seconds (Live Mode)
    log_level = "ERROR"  # Options: DEBUG, INFO, WARNING, ERROR. Set to "ERROR" for max speed in backtest.
    backtest_mode = True  # Set to True for backtesting, False for live trading


class ArcTradingBot:
    def __init__(self):
        """
        Initializes the trading bot, setting up configurations,
        internal state variables, and logging.
        This includes variables for both live trading and backtesting.
        """
        self.config = Config()
        self.running = False  # Flag to control the main bot loop

        # --- Live Trading Specific Attributes ---
        self.open_positions = {}  # Dictionary to track all open positions by ticket (Live)
        self.open_sells = 0  # Counter for active sell positions (Live)

        # --- Backtesting Specific Attributes ---
        self.initial_balance = 10000.0 # Starting balance for backtest
        self.balance = self.initial_balance # Current balance in backtest
        self.equity = self.initial_balance # Equity in backtest (balance + floating P/L)
        self.commission_per_lot = 7.0 # Simulated commission cost per standard lot
        self.slippage_pips = 1 # Simulated slippage in pips for market orders
        self.simulated_open_positions = {} # Tracks simulated open positions (Backtest)
        self.simulated_trade_history = [] # Stores simulated closed trades (Backtest)
        self.simulated_account_history = [] # Tracks balance/equity over time (Backtest)
        self.current_bar_index = 0 # To track progress through historical data (Backtest)
        self.current_data = None # Stores the current bar's data for processing (Backtest)

        # --- Common Attributes ---
        self.current_arcs = []  # Stores detected arcs for visualization (Live & Backtest)
        self.trade_history = deque(maxlen=100)  # Stores recent closed trade outcomes (Live)
        self.last_signal_time = {}  # Prevents rapid re-entry for the same signal type (Live & Backtest)
        self.market_state = {"trend": "neutral", "volatility": "medium"}  # Overall market state assessment
        self.volatility_history = {symbol: deque(maxlen=100) for symbol in self.config.symbols}  # Per-symbol volatility history
        self.risk_adjustment = 1.0  # Dynamically adjusts risk based on bot performance
        self.market_regimes = {}  # Stores market regime analysis per symbol

        # Visualization
        self.fig = None
        self.ax = None
        self.ax2 = None
        self.ax3 = None

        # Logger setup
        self.log_method = getattr(self, f"log_{self.config.log_level.lower()}")

    def _log(self, level: str, message: str) -> None:
        """Internal helper for logging messages, respecting the configured log_level."""
        log_levels_map = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        if log_levels_map.get(level, 0) >= log_levels_map.get(self.config.log_level, 0):
            print(f"[{level.upper()}] {datetime.now().strftime('%H:%M:%S')}: {message}")

    def log_debug(self, message: str) -> None:
        """Logs a debug message if the configured log level allows."""
        self._log("DEBUG", message)

    def log_info(self, message: str) -> None:
        """Logs an info message if the configured log level allows."""
        self._log("INFO", message)

    def log_warning(self, message: str) -> None:
        """Logs a warning message if the configured log level allows."""
        self._log("WARNING", message)

    def log_error(self, message: str) -> None:
        """Logs an error message (always printed)."""
        self._log("ERROR", message)

    # --- Historical Data Loading (for Backtest Mode) ---
    def load_historical_data(self, file_path: str, symbol: str) -> pd.DataFrame:
        """
        Loads historical data from a CSV file.
        The CSV should have columns: 'time', 'open', 'high', 'low', 'close', 'volume'.
        'time' column should be parseable as datetime.
        Generates synthetic data if the file is not found or is malformed.
        """
        if not os.path.exists(file_path):
            self.log_warning(f"Historical data file not found at {file_path}. Generating synthetic data.")
            return self._generate_synthetic_data(symbol, 10000) # Generate a longer period for backtesting

        try:
            df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                self.log_error(f"CSV file {file_path} is missing required columns: {required_cols}. Generating synthetic data.")
                return self._generate_synthetic_data(symbol, 10000)

            # Add basic indicators needed for the strategy (pre-calculated)
            df['ema20'] = EMAIndicator(close=df['close'], window=20, fillna=True).ema_indicator()
            df['ema50'] = EMAIndicator(close=df['close'], window=50, fillna=True).ema_indicator()
            df['ema200'] = EMAIndicator(close=df['close'], window=200, fillna=True).ema_indicator()
            df['rsi'] = RSIIndicator(close=df['close'], window=14, fillna=True).rsi()
            atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14, fillna=True)
            df['atr'] = atr_indicator.average_true_range()
            macd_indicator = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
            df['macd'] = macd_indicator.macd()
            df['macd_signal'] = macd_indicator.macd_signal()
            df['macd_hist'] = macd_indicator.macd_diff()

            df = df.dropna()
            self.log_info(f"Loaded {len(df)} bars for {symbol} from {file_path}.")
            return df
        except Exception as e:
            self.log_error(f"Error loading historical data from {file_path}: {e}. Generating synthetic data.")
            return self._generate_synthetic_data(symbol, 10000)

    def _generate_synthetic_data(self, symbol: str, count: int) -> pd.DataFrame:
        """Generates synthetic market data for testing purposes (hourly data)."""
        self.log_warning(f"Generating {count} bars of synthetic data for {symbol}.")
        now = datetime.now()
        dates = pd.date_range(end=now, periods=count, freq='H')
        np.random.seed(42) # for reproducibility
        prices = 1.1 + np.cumsum(np.random.normal(0, 0.0005, count)) + np.sin(np.arange(count) * 0.1) * 0.01
        df = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.0002, count))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.0002, count))),
            'close': prices,
            'volume': np.random.randint(100, 1000, count)
        }, index=dates)

        # Recalculate indicators for synthetic data to ensure consistency
        df['ema20'] = EMAIndicator(close=df['close'], window=20, fillna=True).ema_indicator()
        df['ema50'] = EMAIndicator(close=df['close'], window=50, fillna=True).ema_indicator()
        df['ema200'] = EMAIndicator(close=df['close'], window=200, fillna=True).ema_indicator()
        df['rsi'] = RSIIndicator(close=df['close'], window=14, fillna=True).rsi()
        atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14, fillna=True)
        df['atr'] = atr_indicator.average_true_range()
        macd_indicator = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_hist'] = macd_indicator.macd_diff()

        return df.dropna()

    # --- MT5 Live Data Fetch (for Live Mode) ---
    def get_market_data(self, symbol: str, count: int = None, additional_indicators: bool = True) -> pd.DataFrame:
        """
        Fetches market data for a specified symbol from MT5 and calculates technical indicators.
        Used only in live trading mode.
        """
        if self.config.backtest_mode:
            self.log_error("get_market_data called in backtest mode. This should not happen.")
            return pd.DataFrame() # Should not be called in backtest mode

        if count is None:
            count = self.config.arc_lookback * 2

        if not mt5.symbol_select(symbol, True):
            self.log_error(f"Failed to select symbol {symbol} for data retrieval.")
            return pd.DataFrame()

        rates = mt5.copy_rates_from_pos(symbol, self.config.timeframe, 0, count)

        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            if additional_indicators:
                df['ema20'] = EMAIndicator(close=df['close'], window=20, fillna=True).ema_indicator()
                df['ema50'] = EMAIndicator(close=df['close'], window=50, fillna=True).ema_indicator()
                df['ema200'] = EMAIndicator(close=df['close'], window=200, fillna=True).ema_indicator()
                df['rsi'] = RSIIndicator(close=df['close'], window=14, fillna=True).rsi()
                atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14, fillna=True)
                df['atr'] = atr_indicator.average_true_range()
                macd_indicator = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
                df['macd'] = macd_indicator.macd()
                df['macd_signal'] = macd_indicator.macd_signal()
                df['macd_hist'] = macd_indicator.macd_diff()

                if not df['atr'].empty and not df['close'].empty:
                    current_volatility = df['atr'].iloc[-1] / df['close'].iloc[-1] * 100
                    self.volatility_history[symbol].append(current_volatility)
                    df['volatility'] = current_volatility
                else:
                    df['volatility'] = 0.0

                df['trend'] = np.where(df['ema20'] > df['ema50'], 1,
                                    np.where(df['ema20'] < df['ema50'], -1, 0))
                df['swing_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
                df['swing_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))

            return df.dropna()
        else:
            self.log_warning(f"Market data unavailable for {symbol} from MT5. Returning empty DataFrame.")
            return pd.DataFrame()


    def fit_arc(self, data: pd.DataFrame, start: int, end: int) -> dict | None:
        """
        Fits an elliptical arc to a segment of price data using `scipy.optimize.curve_fit`.
        Calculates R-squared, curvature, and a quality score for the arc.
        """
        x = np.arange(end - start)
        y = data['close'].iloc[start:end].values

        if len(x) < 5:
            self.log_debug("Not enough data points for arc fitting.")
            return None

        x_norm = x / len(x)
        y_mean, y_std = y.mean(), y.std()
        y_norm = (y - y_mean) / (y_std if y_std != 0 else 1)

        def elliptical(x_val, a, b, c, d, e):
            return a * x_val ** 2 + b * x_val + c + d * np.sin(e * x_val)

        try:
            popt, pcov = curve_fit(
                elliptical, x_norm, y_norm,
                p0=[0, 0, 0, 0.1, 5],
                bounds=([-5, -5, -5, -5, 0.1], [5, 5, 5, 5, 20]),
                maxfev=10000
            )

            fit = elliptical(x_norm, *popt) * y_std + y_mean
            ss_res = ((y - fit) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            curvature = abs(popt[3] * popt[4])
            arc_direction = 1 if popt[0] < 0 else -1
            end_direction = np.sign(2 * popt[0] * x_norm[-1] + popt[1] + popt[3] * popt[4] * np.cos(popt[4] * x_norm[-1]))

            perr = np.sqrt(np.diag(pcov))
            param_stability = np.mean(np.abs(perr / (np.abs(popt) + 1e-9)))
            param_stability = min(1.0, param_stability)

            x_extended = np.linspace(0, 1.5, 100)
            y_extended = elliptical(x_extended, *popt) * y_std + y_mean

            if arc_direction > 0:
                extremum_idx = np.argmin(y_extended[x_extended > x_norm[-1]]) if np.any(x_extended > x_norm[-1]) else -1
            else:
                extremum_idx = np.argmax(y_extended[x_extended > x_norm[-1]]) if np.any(x_extended > x_norm[-1]) else -1

            if extremum_idx != -1:
                extremum_x_norm = x_extended[x_extended > x_norm[-1]][extremum_idx]
                remaining_potential = max(0, extremum_x_norm - x_norm[-1])
                remaining_potential = remaining_potential / max(0.01, (x_norm[-1] - x_norm[0]))
            else:
                remaining_potential = 0.0

            quality = r2 * curvature * (1 - param_stability) * (1 + remaining_potential)
            quality = max(0.0, quality)

            return {
                'start_idx': start,
                'end_idx': end,
                'start_time': data.index[start],
                'end_time': data.index[end - 1],
                'fitted': fit,
                'r2': r2,
                'curvature': curvature,
                'params': popt,
                'volatility': y_std,
                'param_stability': param_stability,
                'arc_direction': arc_direction,
                'end_direction': end_direction,
                'remaining_potential': remaining_potential,
                'quality': quality
            }

        except RuntimeError as re:
            self.log_debug(f"Arc fitting RuntimeError: {str(re)}")
            return None
        except Exception as e:
            self.log_debug(f"Arc fitting failed for segment [{start}:{end}]: {str(e)}")
            return None

    def detect_arcs(self, data: pd.DataFrame) -> list[dict]:
        """
        Detects high-quality elliptical arcs within the price data.
        It iterates through various window sizes, fits arcs, filters them
        by R-squared and curvature, and then uses DBSCAN to cluster similar arcs,
        selecting the best one from each cluster.
        Uses `arc_detection_sizes` and `arc_detection_step_multiplier` from Config.
        """
        arcs = []
        # Iterate through different window sizes for arc detection
        for size in self.config.arc_detection_sizes: # Uses configurable sizes for speed
            step = max(5, int(size * self.config.arc_detection_step_multiplier)) # Uses configurable step
            if len(data) < size:
                continue
            for i in range(0, len(data) - size + 1, step):
                arc = self.fit_arc(data, i, i + size)
                if arc:
                    if arc['r2'] > self.config.min_r2 and arc['curvature'] > self.config.min_curvature:
                        if arc['volatility'] > 0.00001:
                            arcs.append(arc)

        if not arcs:
            self.log_debug("No arcs detected meeting initial quality criteria.")
            return []

        arc_midpoints = np.array([[a['start_idx'] + (a['end_idx'] - a['start_idx']) / 2] for a in arcs])

        clustering = DBSCAN(eps=20, min_samples=1).fit(arc_midpoints)
        labels = clustering.labels_
        best_arcs = []

        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            cluster_arcs = [arc for i, arc in enumerate(arcs) if labels[i] == cluster_id]
            if cluster_arcs:
                best_arc = max(cluster_arcs, key=lambda a: a['quality'])
                best_arcs.append(best_arc)

        self.log_debug(f"Detected {len(best_arcs)} high-quality, distinct arcs after clustering.")
        return sorted(best_arcs, key=lambda a: a['quality'], reverse=True)

    def analyze_market_regime(self, symbol: str, data: pd.DataFrame) -> dict:
        """
        Analyzes current market conditions (trend and volatility) for a given symbol.
        This helps the bot adapt its trading strategy to different market environments.
        """
        if data.empty:
            self.log_warning(f"[{symbol}] Cannot analyze market regime: no data.")
            return {"trend": "neutral", "volatility": "medium"}

        if data['ema20'].iloc[-1] > data['ema50'].iloc[-1]:
            trend = "bullish"
        elif data['ema20'].iloc[-1] < data['ema50'].iloc[-1]:
            trend = "bearish"
        else:
            trend = "ranging"

        current_atr = data['atr'].iloc[-1]
        if self.volatility_history[symbol]:
            avg_vol = np.mean(list(self.volatility_history[symbol]))
        else:
            avg_vol = current_atr

        if current_atr > avg_vol * 1.5:
            vol_regime = "high"
        elif current_atr < avg_vol * 0.7:
            vol_regime = "low"
        else:
            vol_regime = "medium"

        self.market_regimes[symbol] = {"trend": trend, "volatility": vol_regime}
        self.log_debug(f"[{symbol}] Market Regime: Trend={trend}, Volatility={vol_regime}")
        return self.market_regimes[symbol]

    def find_signals(self, arcs: list[dict], data: pd.DataFrame) -> list[dict]:
        """
        Generates potential trade signals based on detected arcs and market conditions.
        Applies various filters including market regime, risk limits, and indicator confirmations.
        Adjusts behavior based on `backtest_mode`.
        """
        signals = []
        current_price = data['close'].iloc[-1]
        symbol = data.index.name if data.index.name else self.config.symbols[0] # Get symbol from data's index name

        market_regime = self.analyze_market_regime(symbol, data)

        # Determine which set of open positions to check
        current_open_positions = self.simulated_open_positions if self.config.backtest_mode else self.open_positions
        current_open_sells = sum(1 for pos in current_open_positions.values() if pos['type'] == 'sell') if self.config.backtest_mode else self.open_sells


        # Do not generate new signals in high volatility if positions are already open
        if market_regime["volatility"] == "high" and current_open_positions:
            self.log_debug(f"[{symbol}] High volatility detected and positions open - not generating new signals.")
            return []

        # Check portfolio risk before generating new signals
        current_risk = self.calculate_portfolio_risk()
        if current_risk >= self.config.max_portfolio_risk:
            self.log_debug(f"[{symbol}] Maximum portfolio risk reached ({current_risk:.2f}%) - not generating new signals.")
            return []

        # Check maximum number of open trades
        if len(current_open_positions) >= self.config.max_open_trades:
            self.log_debug(f"[{symbol}] Maximum open trades ({self.config.max_open_trades}) reached - not generating new signals.")
            return []

        for arc in arcs:
            if arc['quality'] < self.config.quality_threshold:
                self.log_debug(f"[{symbol}] Arc quality {arc['quality']:.2f} below threshold {self.config.quality_threshold}.")
                continue

            a, b, c, d, e = arc['params']
            def deriv_at_x(x_val):
                return 2 * a * x_val + b + d * e * math.cos(e * x_val)

            current_deriv = deriv_at_x(1)
            prev_deriv = deriv_at_x(0.9)
            angle_diff = math.degrees(math.atan(current_deriv) - math.atan(prev_deriv))

            signal_type = None

            # Use current bar's timestamp for last_signal_time check in backtest
            current_time_for_signal = data.index[-1] if self.config.backtest_mode else datetime.now()

            # Bullish signal criteria
            if (angle_diff > 15 and arc['end_direction'] > 0 and
                (market_regime["trend"] != "bearish" or arc['quality'] > 0.8)):
                if 'buy' in self.last_signal_time and \
                   (current_time_for_signal - self.last_signal_time['buy']).total_seconds() < 3600:
                    self.log_debug(f"[{symbol}] Buy signal too soon after last one. Skipping.")
                    continue
                signal_type = 'buy'

            # Bearish signal criteria
            elif (angle_diff < -15 and arc['end_direction'] < 0 and
                  (market_regime["trend"] != "bullish" or arc['quality'] > 0.8) and
                  current_open_sells < self.config.max_open_sells):
                if 'sell' in self.last_signal_time and \
                   (current_time_for_signal - self.last_signal_time['sell']).total_seconds() < 3600:
                    self.log_debug(f"[{symbol}] Sell signal too soon after last one. Skipping.")
                    continue
                signal_type = 'sell'

            if signal_type:
                conviction = min(1.0, arc['quality'] * (1 + abs(angle_diff) / 90))
                indicator_confirmation = self.validate_with_indicators(data, signal_type)

                if indicator_confirmation >= 0.6:
                    signals.append({
                        'type': signal_type,
                        'price': current_price,
                        'optimal_entry': self.calculate_optimal_entry(data, signal_type),
                        'volatility': arc['volatility'],
                        'quality': arc['quality'],
                        'conviction': conviction * indicator_confirmation,
                        'arc_id': id(arc),
                        'symbol': symbol
                    })
                    self.last_signal_time[signal_type] = current_time_for_signal

        if signals:
            signals = sorted(signals, key=lambda s: s['conviction'], reverse=True)
            top_signal = signals[0]
            self.log_debug(f"[{symbol}] Generated {len(signals)} signals. Top: {top_signal['type'].upper()} with {top_signal['conviction']:.2f} conviction.")
        else:
            self.log_debug(f"[{symbol}] No valid trade signals generated.")

        return signals

    def validate_with_indicators(self, data: pd.DataFrame, signal_type: str) -> float:
        """
        Validates a potential trade signal using a set of technical indicators.
        Returns a confirmation score (0.0 to 1.0) based on how many indicators
        support the signal.
        """
        confirmations = 0
        total_indicators = 5

        if data.empty or len(data) < max(self.config.momentum_lookback, self.config.volatility_lookback, 200):
            self.log_warning("Not enough data to validate with indicators.")
            return 0.0

        if signal_type == 'buy':
            if data['ema20'].iloc[-1] > data['ema50'].iloc[-1]:
                confirmations += 1
        else:
            if data['ema20'].iloc[-1] < data['ema50'].iloc[-1]:
                confirmations += 1

        rsi = data['rsi'].iloc[-1]
        if signal_type == 'buy' and rsi < 70:
            confirmations += 1
        elif signal_type == 'sell' and rsi > 30:
            confirmations += 1

        if signal_type == 'buy' and data['macd_hist'].iloc[-1] > 0:
            confirmations += 1
        elif signal_type == 'sell' and data['macd_hist'].iloc[-1] < 0:
            confirmations += 1

        symbol = data.index.name if data.index.name else self.config.symbols[0]
        current_atr = data['atr'].iloc[-1]
        if self.volatility_history[symbol]:
            avg_vol = np.mean(list(self.volatility_history[symbol]))
        else:
            avg_vol = current_atr

        if current_atr < avg_vol * 1.5:
            confirmations += 1

        if signal_type == 'buy' and data['close'].iloc[-1] > data['ema200'].iloc[-1]:
            confirmations += 1
        elif signal_type == 'sell' and data['close'].iloc[-1] < data['ema200'].iloc[-1]:
            confirmations += 1

        return confirmations / total_indicators

    def calculate_optimal_entry(self, data: pd.DataFrame, signal_type: str) -> dict[str, float]:
        """
        Calculates optimal entry levels (immediate, preferred, limit) based on
        current price and Average True Range (ATR) to suggest potential pullbacks.
        """
        current_price = data['close'].iloc[-1]
        atr = data['atr'].iloc[-1]

        if atr == 0:
            atr = 0.0001

        if signal_type == 'buy':
            return {
                'immediate': current_price,
                'preferred': max(current_price - 0.3 * atr, data['low'].iloc[-1]),
                'limit': max(current_price - 0.5 * atr, data['low'].iloc[-5:].min())
            }
        else:
            return {
                'immediate': current_price,
                'preferred': min(current_price + 0.3 * atr, data['high'].iloc[-1]),
                'limit': min(current_price + 0.5 * atr, data['high'].iloc[-5:].max())
            }

    def calculate_portfolio_risk(self) -> float:
        """
        Calculates the current total risk exposure as a percentage of the account balance.
        Adapts for live or simulated positions.
        """
        if self.config.backtest_mode:
            current_positions = self.simulated_open_positions
            current_balance = self.balance
        else:
            current_positions = self.open_positions
            account_info = mt5.account_info()
            if not account_info:
                self.log_error("Failed to get account info for live portfolio risk calculation.")
                return 0.0
            current_balance = account_info.balance

        if not current_positions:
            return 0.0

        total_risk_percent = 0.0
        for ticket, pos in current_positions.items():
            if pos['sl'] != 0:
                # Get symbol info (simulated for backtest, real for live)
                if self.config.backtest_mode:
                    symbol_info = self._get_simulated_symbol_info(pos['symbol'])
                    # For backtest, pip_value_per_lot is simplified
                    if "JPY" in pos['symbol']:
                        pip_value_per_lot = 1000
                    else:
                        pip_value_per_lot = 10
                else:
                    symbol_info = mt5.symbol_info(pos['symbol'])
                    if not symbol_info:
                        self.log_warning(f"Could not get symbol info for {pos['symbol']} (ticket {ticket}). Skipping risk calculation for this position.")
                        continue
                    pip_value_per_lot = symbol_info.trade_tick_value * 10 # Assuming 1 pip = 10 ticks

                price_diff = abs(pos['price'] - pos['sl'])
                if "JPY" in pos['symbol']:
                    sl_pips = price_diff / 0.01
                else:
                    sl_pips = price_diff / 0.0001

                risk_amount_in_currency = sl_pips * pip_value_per_lot * pos['volume']

                risk_percent = (risk_amount_in_currency / current_balance) * 100
                total_risk_percent += risk_percent

        self.log_debug(f"Current portfolio risk: {total_risk_percent:.2f}%")
        return total_risk_percent

    def calculate_position_size(self, sl_pips: float, conviction: float = 1.0, symbol_name: str = None) -> float:
        """
        Calculates the appropriate position size (in lots) based on risk per trade,
        stop loss in pips, and signal conviction. Adapts for live or simulated balance.
        """
        if self.config.backtest_mode:
            current_balance = self.balance
        else:
            account_info = mt5.account_info()
            if not account_info:
                self.log_error("Failed to get account info for live position sizing.")
                return self.config.min_lots
            current_balance = account_info.balance

        if current_balance <= 0:
            self.log_warning("Account balance is zero or negative. Cannot calculate position size.")
            return self.config.min_lots

        if sl_pips <= 0:
            self.log_warning("Stop loss pips must be positive for position sizing. Using min_lots.")
            return self.config.min_lots

        # Get symbol info (simulated for backtest, real for live)
        if self.config.backtest_mode:
            symbol_info = self._get_simulated_symbol_info(symbol_name)
            if "JPY" in symbol_name:
                pip_value_per_lot = 1000
            else:
                pip_value_per_lot = 10
        else:
            symbol_info = mt5.symbol_info(symbol_name)
            if not symbol_info:
                self.log_error(f"Failed to get symbol info for {symbol_name} for live position sizing.")
                return self.config.min_lots
            pip_value_per_lot = symbol_info.trade_tick_value * 10

        if pip_value_per_lot == 0:
            self.log_error(f"Calculated pip_value_per_lot for {symbol_name} is zero. Using min_lots.")
            return self.config.min_lots

        adjusted_risk_percent = self.config.max_risk_percent * self.risk_adjustment * conviction
        risk_amount_in_currency = current_balance * (adjusted_risk_percent / 100)

        position_size_lots = risk_amount_in_currency / (sl_pips * pip_value_per_lot)

        lots = max(self.config.min_lots, min(round(position_size_lots, 2), self.config.max_lots))

        self.log_debug(f"Position size calculation for {symbol_name}: "
                      f"Risk={adjusted_risk_percent:.2f}%, SL={sl_pips:.1f} pips, "
                      f"Pip Value/Lot={pip_value_per_lot:.2f}, Size={lots:.2f} lots")
        return lots

    def update_risk_adjustment(self) -> None:
        """
        Dynamically adjusts the bot's risk exposure based on recent trade performance.
        Uses either live or simulated trade history.
        """
        trade_history_source = self.simulated_trade_history if self.config.backtest_mode else self.trade_history

        if not trade_history_source:
            self.risk_adjustment = 1.0
            self.log_debug("No trade history to update risk adjustment.")
            return

        win_count = sum(1 for trade in trade_history_source if trade['profit'] > 0)
        total_trades = len(trade_history_source)
        loss_count = total_trades - win_count

        win_rate = win_count / total_trades if total_trades > 0 else 0

        loss_sum = sum(abs(t['profit']) for t in trade_history_source if t['profit'] < 0)
        profit_sum = sum(t['profit'] for t in trade_history_source if t['profit'] > 0)

        profit_factor = profit_sum / max(0.1, loss_sum)

        if win_rate >= 0.6 and profit_factor >= 1.5:
            self.risk_adjustment = min(1.2, self.risk_adjustment + 0.02)
        elif win_rate <= 0.4 or profit_factor < 1.0:
            self.risk_adjustment = max(0.5, self.risk_adjustment - 0.05)
        else:
            if self.risk_adjustment < 1.0:
                self.risk_adjustment = min(1.0, self.risk_adjustment + 0.01)
            elif self.risk_adjustment > 1.0:
                self.risk_adjustment = max(1.0, self.risk_adjustment - 0.01)

        self.log_debug(f"Risk adjustment updated: {self.risk_adjustment:.2f} "
                      f"(Win Rate: {win_rate:.1%}, Profit Factor: {profit_factor:.2f})")

    # --- Simulated MT5 Functions (for Backtest Mode) ---
    def _get_simulated_symbol_info(self, symbol_name: str):
        """
        Provides simulated symbol information for backtesting.
        """
        if "JPY" in symbol_name:
            return {'point': 0.01, 'trade_tick_value': 0.001, 'trade_stops_level': 20}
        else:
            return {'point': 0.0001, 'trade_tick_value': 0.00001, 'trade_stops_level': 20}

    def simulate_order_execution(self, signal: dict, current_bar: pd.Series) -> None:
        """
        Simulates placing a trade order based on a signal.
        Updates simulated account and open positions.
        """
        if not self.config.backtest_mode:
            self.log_error("simulate_order_execution called in live mode. This should not happen.")
            return

        order_type = signal['type']
        symbol_name = signal['symbol']
        entry_price = signal['optimal_entry']['immediate']

        # Simulate slippage
        if "JPY" in symbol_name:
            slippage_unit = 0.001
        else:
            slippage_unit = 0.00001

        if order_type == 'buy':
            simulated_entry_price = entry_price + (self.slippage_pips * slippage_unit)
        else:
            simulated_entry_price = entry_price - (self.slippage_pips * slippage_unit)

        sim_symbol_info = self._get_simulated_symbol_info(symbol_name)
        
        atr = current_bar['atr']
        if atr == 0: atr = 0.0001

        if order_type == 'buy':
            sl_level = (current_bar['low'] - 0.5 * atr)
        else:
            sl_level = (current_bar['high'] + 0.5 * atr)
        
        min_dist_price = sim_symbol_info['trade_stops_level'] * sim_symbol_info['point']

        if order_type == 'buy':
            sl_level = min(sl_level, simulated_entry_price - min_dist_price)
        else:
            sl_level = max(sl_level, simulated_entry_price + min_dist_price)

        if "JPY" in symbol_name:
            pip_divisor = 0.01
        else:
            pip_divisor = 0.0001
        sl_pips = abs(simulated_entry_price - sl_level) / pip_divisor

        if sl_pips < 5:
            sl_pips = 5
            if order_type == 'buy':
                sl_level = simulated_entry_price - (sl_pips * pip_divisor)
            else:
                sl_level = simulated_entry_price + (sl_pips * pip_divisor)

        tp_pips = sl_pips * self.config.profit_ratio
        if order_type == 'buy':
            tp_level = simulated_entry_price + (tp_pips * pip_divisor)
        else:
            tp_level = simulated_entry_price - (tp_pips * pip_divisor)

        lots = self.calculate_position_size(sl_pips, signal['conviction'], symbol_name=symbol_name)
        
        if lots < self.config.min_lots:
            self.log_debug(f"[{symbol_name}] Simulated lots ({lots:.2f}) too small. Skipping order.")
            return

        commission_cost = (lots / 1.0) * self.commission_per_lot
        self.balance -= commission_cost
        self.equity -= commission_cost

        ticket = len(self.simulated_open_positions) + 1

        self.simulated_open_positions[ticket] = {
            'ticket': ticket,
            'symbol': symbol_name,
            'type': order_type,
            'price': simulated_entry_price,
            'volume': lots,
            'sl': sl_level,
            'tp': tp_level,
            'profit': 0.0,
            'open_time': current_bar.name
        }
        self.log_info(f"Simulated {order_type.upper()} order placed for {symbol_name} @ {simulated_entry_price:.5f} "
                      f"Lots: {lots:.2f}, SL: {sl_level:.5f}, TP: {tp_level:.5f}. Commission: {commission_cost:.2f}")


    def simulate_position_management(self, current_bar: pd.Series) -> None:
        """
        Simulates managing open positions (e.g., trailing stops, SL/TP hits).
        This is called for each bar in the backtest.
        """
        if not self.config.backtest_mode:
            self.log_error("simulate_position_management called in live mode. This should not happen.")
            return

        closed_tickets_in_this_bar = []
        current_price = current_bar['close']
        high_price = current_bar['high']
        low_price = current_bar['low']
        bar_time = current_bar.name
        symbol_name = current_bar.name # Assuming index name is symbol for dataframes in main loop

        for ticket, pos in list(self.simulated_open_positions.items()):
            if pos['symbol'] != symbol_name:
                continue

            if "JPY" in pos['symbol']:
                contract_size = 100000
            else:
                contract_size = 100000

            if pos['type'] == 'buy':
                current_profit = (current_price - pos['price']) * pos['volume'] * contract_size
            else:
                current_profit = (pos['price'] - current_price) * pos['volume'] * contract_size

            sl_hit = False
            tp_hit = False

            if pos['type'] == 'buy':
                if pos['sl'] != 0 and low_price <= pos['sl']:
                    sl_hit = True
                    close_price = pos['sl']
                elif pos['tp'] != 0 and high_price >= pos['tp']:
                    tp_hit = True
                    close_price = pos['tp']
            else:
                if pos['sl'] != 0 and high_price >= pos['sl']:
                    sl_hit = True
                    close_price = pos['sl']
                elif pos['tp'] != 0 and low_price <= pos['tp']:
                    tp_hit = True
                    close_price = pos['tp']

            if sl_hit or tp_hit:
                if pos['type'] == 'buy':
                    final_profit = (close_price - pos['price']) * pos['volume'] * contract_size
                else:
                    final_profit = (pos['price'] - close_price) * pos['volume'] * contract_size

                self.balance += final_profit
                self.equity = self.balance

                closed_tickets_in_this_bar.append(ticket)
                self.simulated_trade_history.append({
                    'ticket': pos['ticket'],
                    'symbol': pos['symbol'],
                    'type': pos['type'],
                    'open_price': pos['price'],
                    'close_price': close_price,
                    'volume': pos['volume'],
                    'profit': final_profit,
                    'open_time': pos['open_time'],
                    'close_time': bar_time,
                    'status': 'SL Hit' if sl_hit else 'TP Hit'
                })
                self.log_info(f"Simulated position {ticket} for {pos['symbol']} closed at {close_price:.5f} ({'SL' if sl_hit else 'TP'}) with profit: {final_profit:.2f}")

            if self.config.trailing_stop and not (sl_hit or tp_hit):
                sim_symbol_info = self._get_simulated_symbol_info(symbol_name)
                pip_size = sim_symbol_info['point']

                if pos['type'] == 'buy':
                    profit_pips = (current_price - pos['price']) / pip_size
                else:
                    profit_pips = (pos['price'] - current_price) / pip_size

                if profit_pips > 0:
                    if pos['tp'] != 0:
                        target_pips = abs(pos['tp'] - pos['price']) / pip_size
                    else:
                        target_pips = abs(pos['sl'] - pos['price']) / pip_size * self.config.profit_ratio
                        if target_pips == 0:
                            continue

                    profit_ratio = profit_pips / max(0.01, target_pips)

                    if profit_ratio >= self.config.trailing_stop_trigger:
                        trailing_pips_from_current_price = profit_pips * 0.5
                        if pos['type'] == 'buy':
                            new_sl = current_price - (trailing_pips_from_current_price * pip_size)
                            if new_sl > pos['sl'] and new_sl > pos['price']:
                                pos['sl'] = new_sl
                                self.log_debug(f"Simulated position {ticket} trailing SL to {new_sl:.5f}")
                        else:
                            new_sl = current_price + (trailing_pips_from_current_price * pip_size)
                            if new_sl < pos['sl'] and new_sl < pos['price']:
                                pos['sl'] = new_sl
                                self.log_debug(f"Simulated position {ticket} trailing SL to {new_sl:.5f}")

        for ticket in closed_tickets_in_this_bar:
            del self.simulated_open_positions[ticket]


    def check_open_positions(self) -> None:
        """
        Retrieves and updates the status of all open positions from MT5.
        Identifies newly opened or closed positions and updates `self.open_positions`
        and `self.trade_history`. Only used in live trading mode.
        """
        if self.config.backtest_mode:
            self.log_error("check_open_positions called in backtest mode. This should not happen.")
            return

        if not mt5.is_connected():
            self.log_error("MT5 is not connected. Cannot check open positions.")
            return

        positions = mt5.positions_get()

        current_tickets_on_mt5 = set()
        self.open_sells = 0

        if positions:
            for position in positions:
                ticket = position.ticket
                current_tickets_on_mt5.add(ticket)

                if position.type == mt5.POSITION_TYPE_SELL:
                    self.open_sells += 1

                if ticket not in self.open_positions:
                    self.open_positions[ticket] = {
                        'ticket': ticket,
                        'symbol': position.symbol,
                        'type': 'sell' if position.type == mt5.POSITION_TYPE_SELL else 'buy',
                        'price': position.price_open,
                        'volume': position.volume,
                        'sl': position.sl,
                        'tp': position.tp,
                        'profit': position.profit,
                        'open_time': datetime.fromtimestamp(position.time)
                    }
                    self.log_info(f"New position detected: {position.symbol} {position.type} #{ticket}")
                else:
                    self.open_positions[ticket]['profit'] = position.profit
                    self.open_positions[ticket]['sl'] = position.sl
                    self.open_positions[ticket]['tp'] = position.tp
        else:
            self.log_info("No open positions found on MT5.")

        closed_tickets = set(self.open_positions.keys()) - current_tickets_on_mt5
        for ticket in closed_tickets:
            position = self.open_positions.pop(ticket)
            position['close_time'] = datetime.now()
            position['duration'] = (position['close_time'] - position['open_time']).total_seconds() / 3600
            self.trade_history.append(position)
            self.log_info(f"Position {ticket} ({position['symbol']}) closed with profit: {position['profit']:.2f}")

        self.update_risk_adjustment()


    def manage_open_positions(self, data: pd.DataFrame) -> None:
        """
        Manages open positions by implementing trailing stops.
        Used only in live trading mode.
        """
        if self.config.backtest_mode:
            self.log_error("manage_open_positions called in backtest mode. This should not happen.")
            return

        if not self.open_positions:
            return

        current_price = data['close'].iloc[-1]
        symbol_name = data.index.name
        if symbol_name is None:
            symbol_name = self.config.symbols[0]

        symbol_info = mt5.symbol_info(symbol_name)
        if not symbol_info:
            self.log_error(f"Failed to get symbol info for {symbol_name} in manage_open_positions.")
            return

        pip_size = symbol_info.point

        for ticket, position in list(self.open_positions.items()):
            if position['symbol'] != symbol_name:
                continue

            if position['sl'] == 0:
                self.log_debug(f"Position {ticket} has no SL set. Skipping trailing stop.")
                continue

            if position['type'] == 'buy':
                profit_pips = (current_price - position['price']) / pip_size
            else:
                profit_pips = (position['price'] - current_price) / pip_size

            if self.config.trailing_stop and profit_pips > 0:
                if position['tp'] != 0:
                    target_pips = abs(position['tp'] - position['price']) / pip_size
                else:
                    target_pips = abs(position['sl'] - position['price']) / pip_size * self.config.profit_ratio
                    if target_pips == 0:
                        self.log_warning(f"Position {ticket} has no TP and SL leads to 0 target pips. Cannot trail.")
                        continue

                profit_ratio = profit_pips / max(0.01, target_pips)

                if profit_ratio >= self.config.trailing_stop_trigger:
                    trailing_pips_from_current_price = profit_pips * 0.5

                    if position['type'] == 'buy':
                        new_sl = current_price - (trailing_pips_from_current_price * pip_size)
                        if new_sl > position['sl'] and new_sl > position['price']:
                            self.modify_position(ticket, new_sl, position['tp'])
                    else:
                        new_sl = current_price + (trailing_pips_from_current_price * pip_size)
                        if new_sl < position['sl'] and new_sl < position['price']:
                            self.modify_position(ticket, new_sl, position['tp'])

    def modify_position(self, ticket: int, sl: float, tp: float) -> None:
        """
        Sends a request to MetaTrader 5 to modify an existing position's
        stop loss (SL) and take profit (TP) levels. Only used in live trading mode.
        """
        if self.config.backtest_mode:
            self.log_error("modify_position called in backtest mode. This should not happen.")
            return

        if not mt5.is_connected():
            self.log_error("MT5 is not connected. Cannot modify position.")
            return

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": 123456,
            "comment": "Arc Bot - Modify",
            "type_time": mt5.ORDER_TIME_GTC,
        }
        result = mt5.order_send(request)

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.log_info(f"Modified position {ticket}: SL={sl:.5f}, TP={tp:.5f}")
            if ticket in self.open_positions:
                self.open_positions[ticket]['sl'] = sl
                self.open_positions[ticket]['tp'] = tp
        else:
            self.log_error(f"Failed to modify position {ticket}: {result.retcode} - {result.comment}")
            self.log_error(f"Modification request: {request}")
            self.log_error(f"Modification result: {result}")


    def place_order(self, signal: dict) -> bool:
        """
        Places a new trade order based on a generated signal, incorporating
        stop loss, take profit, and position sizing calculations.
        Executes live order or simulates based on `backtest_mode`.
        """
        if self.config.backtest_mode:
            self.simulate_order_execution(signal, self.current_data)
            return True # Assume simulated order is "placed"
        else:
            # Live trading logic
            if not mt5.is_connected():
                self.log_error("MT5 is not connected. Cannot place live order.")
                return False

            order_type = signal['type']
            symbol_name = signal['symbol']
            entry_price = signal['optimal_entry']['immediate']

            data = self.get_market_data(symbol=symbol_name) # Fetch fresh data for live order
            if data.empty:
                self.log_error(f"[{symbol_name}] Failed to get market data for live order placement.")
                return False

            atr = data['atr'].iloc[-1]
            if atr == 0:
                self.log_warning(f"[{symbol_name}] ATR is zero. Using a default small value for SL/TP calculation.")
                atr = 0.0001

            symbol_info = mt5.symbol_info(symbol_name)
            if not symbol_info:
                self.log_error(f"Failed to get symbol info for {symbol_name}.")
                return False

            if order_type == 'buy':
                sl_level = (data['low'].iloc[-5:].min() - 0.5 * atr)
            else:
                sl_level = (data['high'].iloc[-5:].max() + 0.5 * atr)

            min_dist_points = symbol_info.trade_stops_level
            min_dist_price = min_dist_points * symbol_info.point

            if order_type == 'buy':
                sl_level = min(sl_level, entry_price - min_dist_price)
            else:
                sl_level = max(sl_level, entry_price + min_dist_price)

            if "JPY" in symbol_name:
                pip_divisor = 0.01
            else:
                pip_divisor = 0.0001
            sl_pips = abs(entry_price - sl_level) / pip_divisor

            if sl_pips < 5:
                self.log_warning(f"[{symbol_name}] Calculated SL pips ({sl_pips:.1f}) too small. Adjusting to 5 pips.")
                sl_pips = 5
                if order_type == 'buy':
                    sl_level = entry_price - (sl_pips * pip_divisor)
                else:
                    sl_level = entry_price + (sl_pips * pip_divisor)

            tp_pips = sl_pips * self.config.profit_ratio
            if order_type == 'buy':
                tp_level = entry_price + (tp_pips * pip_divisor)
            else:
                tp_level = entry_price - (tp_pips * pip_divisor)

            lots = self.calculate_position_size(sl_pips, signal['conviction'], symbol_name=symbol_name)

            if lots < self.config.min_lots:
                self.log_warning(f"[{symbol_name}] Calculated lots ({lots:.2f}) too small. Skipping live order.")
                return False

            if len(self.open_positions) >= self.config.max_open_trades:
                self.log_warning(f"[{symbol_name}] Maximum open trades ({self.config.max_open_trades}) reached - skipping live order.")
                return False

            if order_type == 'sell' and self.open_sells >= self.config.max_open_sells:
                self.log_warning(f"[{symbol_name}] Maximum open sell positions ({self.config.max_open_sells}) reached - skipping live sell order.")
                return False

            self.log_info(f"[{symbol_name}] Placing live order: {order_type.upper()} @ {entry_price:.5f}, "
                          f"SL={sl_level:.5f}, TP={tp_level:.5f}, Size={lots:.2f}")

            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': symbol_name,
                'volume': lots,
                'type': mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL,
                'price': entry_price,
                'sl': sl_level,
                'tp': tp_level,
                'deviation': 20,
                'magic': 123456,
                'comment': f'Arc Bot - Q:{signal["quality"]:.2f} C:{signal["conviction"]:.2f}',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_FOK
            }

            result = mt5.order_send(request)

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.log_info(f"[{symbol_name}] Live order successfully placed: ticket #{result.order}")
                if order_type == 'sell':
                    self.open_sells += 1
                self.open_positions[result.order] = {
                    'ticket': result.order,
                    'symbol': symbol_name,
                    'type': order_type,
                    'price': result.price,
                    'volume': lots,
                    'sl': sl_level,
                    'tp': tp_level,
                    'profit': 0,
                    'open_time': datetime.now()
                }
                return True
            else:
                self.log_error(f"[{symbol_name}] Live order failed: {result.retcode} - {result.comment}")
                self.log_error(f"Order request: {request}")
                self.log_error(f"Order result: {result}")
                return False

    def visualize(self, data: pd.DataFrame) -> None:
        """
        Creates a real-time visualization of market data, detected arcs,
        and open positions using Matplotlib.
        Only used in live trading mode.
        """
        if self.config.backtest_mode:
            # In backtest mode, visualization is handled by generate_report at the end.
            return

        if data.empty:
            self.log_warning("No data to visualize.")
            return

        if not self.fig:
            plt.ion()
            self.fig, (self.ax, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(14, 12),
                                                                   gridspec_kw={'height_ratios': [3, 1, 1]})
            plt.subplots_adjust(hspace=0.3)
            self.fig.suptitle(f"Arc Trading Bot - {self.config.symbols[0]} Live Data", fontsize=16)

        self.ax.clear()
        self.ax2.clear()
        self.ax3.clear()

        self.ax.plot(data.index, data['close'], label='Price', linewidth=1.5, color='blue')
        self.ax.plot(data.index, data['ema20'], label='EMA20', linewidth=1, alpha=0.7, color='orange')
        self.ax.plot(data.index, data['ema50'], label='EMA50', linewidth=1, alpha=0.7, color='purple')
        self.ax.plot(data.index, data['ema200'], label='EMA200', linewidth=1, alpha=0.7, color='gray', linestyle='--')

        if self.current_arcs:
            sorted_arcs = sorted(self.current_arcs, key=lambda a: a['quality'], reverse=True)
            for i, arc in enumerate(sorted_arcs[:self.config.max_arcs]):
                start_idx, end_idx = arc['start_idx'], arc['end_idx']
                if start_idx >= len(data) or end_idx > len(data):
                    continue
                color = 'green' if arc['arc_direction'] > 0 else 'red'
                best_quality = max(0.001, sorted_arcs[0]['quality'])
                raw_alpha = 0.4 + 0.6 * (arc['quality'] / best_quality)
                alpha = max(0.0, min(1.0, raw_alpha))
                self.ax.plot(data.index[start_idx:end_idx], arc['fitted'],
                             linestyle='--', color=color, alpha=alpha,
                             linewidth=2.5 if i == 0 else (1.5 if i < 3 else 0.8),
                             label=f"Arc Q:{arc['quality']:.2f}" if i == 0 else None)

        for ticket, pos in self.open_positions.items():
            if pos['symbol'] != self.config.symbols[0]:
                continue

            marker = '^' if pos['type'] == 'buy' else 'v'
            color = 'green' if pos['type'] == 'buy' else 'red'
            idx_loc = data.index.get_indexer([pos['open_time']], method='nearest')[0]
            if 0 <= idx_loc < len(data.index):
                self.ax.plot(data.index[idx_loc], pos['price'], marker=marker, markersize=12, color=color,
                             label=f"{pos['type'].upper()} Entry" if ticket == list(self.open_positions.keys())[0] else None)

                if pos['sl'] != 0:
                    self.ax.axhline(y=pos['sl'], linestyle=':', color=color, alpha=0.6,
                                    label=f"{pos['type'].upper()} SL" if ticket == list(self.open_positions.keys())[0] else None)
                if pos['tp'] != 0:
                    self.ax.axhline(y=pos['tp'], linestyle=':', color='blue' if pos['type'] == 'buy' else 'orange', alpha=0.6,
                                    label=f"{pos['type'].upper()} TP" if ticket == list(self.open_positions.keys())[0] else None)

        market_regime_text = self.market_regimes.get(self.config.symbols[0], {"trend": "N/A", "volatility": "N/A"})
        self.ax.set_title(f"{self.config.symbols[0]} Price Chart - {market_regime_text['trend'].capitalize()} Trend, "
                          f"{market_regime_text['volatility'].capitalize()} Volatility", fontsize=12)
        self.ax.legend(loc='upper left', fontsize=9)
        self.ax.grid(True, alpha=0.3)
        self.ax.tick_params(axis='x', rotation=45)

        self.ax2.plot(data.index, data['rsi'], color='purple', linewidth=1.5, label='RSI')
        self.ax2.axhline(y=70, linestyle='--', color='red', alpha=0.5, label='Overbought (70)')
        self.ax2.axhline(y=30, linestyle='--', color='green', alpha=0.5, label='Oversold (30)')
        self.ax2.axhline(y=50, linestyle='--', color='gray', alpha=0.3)
        self.ax2.set_ylim(0, 100)
        self.ax2.set_title('RSI(14)', fontsize=10)
        self.ax2.legend(loc='upper left', fontsize=8)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.tick_params(axis='x', rotation=45)

        self.ax3.plot(data.index, data['macd'], color='blue', linewidth=1, label='MACD Line')
        self.ax3.plot(data.index, data['macd_signal'], color='red', linewidth=1, label='Signal Line')
        self.ax3.bar(data.index, data['macd_hist'], color=['green' if x > 0 else 'red' for x in data['macd_hist']],
                     alpha=0.5, label='Histogram')
        self.ax3.axhline(y=0, linestyle='-', color='gray', alpha=0.3)
        self.ax3.set_title('MACD (12, 26, 9)', fontsize=10)
        self.ax3.legend(loc='upper left', fontsize=8)
        self.ax3.grid(True, alpha=0.3)
        self.ax3.tick_params(axis='x', rotation=45)

        stats_text = f"Open Trades: {len(self.open_positions)}\n"
        if self.trade_history:
            wins = sum(1 for t in self.trade_history if t['profit'] > 0)
            total_trades = len(self.trade_history)
            win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
            stats_text += f"Closed Trades: {total_trades}\n"
            stats_text += f"Win Rate: {win_rate:.1f}%\n"
            stats_text += f"Risk Adjustment: {self.risk_adjustment:.2f}"
        else:
            stats_text += "Closed Trades: 0\nWin Rate: N/A\nRisk Adjustment: 1.00"

        self.ax.annotate(stats_text, xy=(0.01, 0.99), xycoords='axes fraction',
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8, edgecolor='lightgray'))

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.draw()
        plt.pause(0.01)

    def generate_report(self) -> None:
        """Generates and prints a performance report for the backtest."""
        if not self.config.backtest_mode:
            self.log_error("generate_report called in live mode. This should not happen.")
            return

        self.log_info("\n--- Backtest Report ---")
        self.log_info(f"Initial Balance: ${self.initial_balance:.2f}")
        self.log_info(f"Final Balance:   ${self.balance:.2f}")
        total_profit = self.balance - self.initial_balance
        self.log_info(f"Total Profit/Loss: ${total_profit:.2f}")
        self.log_info(f"Return on Investment (ROI): {(total_profit / self.initial_balance * 100):.2f}%")

        if not self.simulated_trade_history:
            self.log_info("No trades executed during backtest.")
            return

        total_trades = len(self.simulated_trade_history)
        winning_trades = [t for t in self.simulated_trade_history if t['profit'] > 0]
        losing_trades = [t for t in self.simulated_trade_history if t['profit'] <= 0]

        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        self.log_info(f"Total Trades:    {total_trades}")
        self.log_info(f"Winning Trades:  {len(winning_trades)} ({win_rate:.2f}%)")
        self.log_info(f"Losing Trades:   {len(losing_trades)}")

        gross_profit = sum(t['profit'] for t in winning_trades)
        gross_loss = sum(abs(t['profit']) for t in losing_trades)
        self.log_info(f"Gross Profit:    ${gross_profit:.2f}")
        self.log_info(f"Gross Loss:      ${gross_loss:.2f}")

        profit_factor = gross_profit / max(0.01, gross_loss)
        self.log_info(f"Profit Factor:   {profit_factor:.2f}")

        equity_curve = pd.DataFrame(self.simulated_account_history)
        if not equity_curve.empty:
            equity_curve['peak'] = equity_curve['equity'].cummax()
            equity_curve['drawdown'] = equity_curve['peak'] - equity_curve['equity']
            max_drawdown = equity_curve['drawdown'].max()
            max_drawdown_percent = (max_drawdown / equity_curve['peak'].max()) * 100 if equity_curve['peak'].max() > 0 else 0
            self.log_info(f"Max Drawdown:    ${max_drawdown:.2f} ({max_drawdown_percent:.2f}%)")

            plt.figure(figsize=(12, 6))
            plt.plot(equity_curve['time'], equity_curve['equity'], label='Equity Curve', color='blue')
            plt.plot(equity_curve['time'], equity_curve['balance'], label='Balance Curve', color='green', linestyle='--')
            plt.title(f"{self.config.symbols[0]} Strategy Equity Curve")
            plt.xlabel("Date")
            plt.ylabel("Account Value ($)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        else:
            self.log_warning("No equity curve data to plot.")

        self.log_info("--- End of Report ---")


    def run(self) -> None:
        """
        Main execution loop of the trading bot.
        Switches between live trading and backtesting based on `self.config.backtest_mode`.
        """
        self.running = True

        if self.config.backtest_mode:
            self.log_info("Starting backtest simulation...")
            historical_data_for_backtest = {}
            long_period_bars = 10000 # Default for synthetic data generation

            for symbol in self.config.symbols:
                csv_file_path = f"{symbol}_H1_data.csv"
                # Load historical data or generate synthetic if not found
                historical_data_for_backtest[symbol] = self.load_historical_data(csv_file_path, symbol)

            # Ensure all dataframes have enough history for initial lookback
            min_data_length = self.config.arc_lookback * 2
            for symbol, df in historical_data_for_backtest.items():
                if len(df) < min_data_length:
                    self.log_error(f"Not enough historical data for {symbol}. Need at least {min_data_length} bars for backtest.")
                    self.running = False
                    return

            max_bars = min(len(df) for df in historical_data_for_backtest.values())
            self.log_info(f"Backtesting over {max_bars - min_data_length} bars for {len(historical_data_for_backtest)} symbols.")

            try:
                for i in range(min_data_length, max_bars):
                    self.current_bar_index = i
                    current_timestamp = historical_data_for_backtest[list(historical_data_for_backtest.keys())[0]].index[i]
                    self.log_debug(f"Processing bar: {current_timestamp}")

                    best_signal_overall = None

                    for symbol in self.config.symbols:
                        if symbol not in historical_data_for_backtest:
                            continue

                        full_data = historical_data_for_backtest[symbol]
                        data_slice = full_data.iloc[i - self.config.arc_lookback : i + 1]

                        if data_slice.empty or len(data_slice) < self.config.arc_lookback:
                            self.log_debug(f"[{symbol}] Not enough data in slice for current bar. Skipping signal generation.")
                            continue

                        self.current_data = data_slice.iloc[-1]
                        
                        if not data_slice['atr'].empty and not data_slice['close'].empty:
                            current_vol = data_slice['atr'].iloc[-1] / data_slice['close'].iloc[-1] * 100
                            self.volatility_history[symbol].append(current_vol)

                        self.analyze_market_regime(symbol, data_slice)
                        arcs = self.detect_arcs(data_slice)
                        signals = self.find_signals(arcs, data_slice)

                        if signals:
                            current_symbol_top_signal = signals[0]
                            if current_symbol_top_signal['conviction'] > 0.7:
                                if (not best_signal_overall) or \
                                   (current_symbol_top_signal['conviction'] > best_signal_overall['conviction']):
                                    best_signal_overall = current_symbol_top_signal

                    # Simulate position management (SL/TP hits, trailing stops) for all open positions
                    # This needs to be done for each open position, using the latest price for its symbol.
                    closed_tickets_in_this_bar = []
                    for ticket, pos in list(self.simulated_open_positions.items()):
                        symbol_of_pos = pos['symbol']
                        if symbol_of_pos in historical_data_for_backtest:
                            current_bar_for_pos = historical_data_for_backtest[symbol_of_pos].iloc[i]
                            self.simulate_position_management(current_bar_for_pos) # This method will update self.balance and self.simulated_trade_history

                    # Calculate global equity after all position management for this bar
                    floating_pl_total = 0.0
                    for ticket, pos in self.simulated_open_positions.items():
                        symbol_of_pos = pos['symbol']
                        if symbol_of_pos in historical_data_for_backtest:
                            current_price_for_equity = historical_data_for_backtest[symbol_of_pos].iloc[i]['close']
                            if "JPY" in symbol_of_pos:
                                contract_size = 100000
                            else:
                                contract_size = 100000

                            if pos['type'] == 'buy':
                                floating_pl_total += (current_price_for_equity - pos['price']) * pos['volume'] * contract_size
                            else:
                                floating_pl_total += (pos['price'] - current_price_for_equity) * pos['volume'] * contract_size
                    self.equity = self.balance + floating_pl_total
                    self.simulated_account_history.append({
                        'time': current_timestamp,
                        'balance': self.balance,
                        'equity': self.equity
                    })

                    if best_signal_overall:
                        self.place_order(best_signal_overall) # This will call simulate_order_execution

                    self.update_risk_adjustment()

            except KeyboardInterrupt:
                self.log_info("Backtest stopped by user.")
            except Exception as e:
                self.log_error(f"Critical error in backtest loop: {str(e)}")
                import traceback
                self.log_error(traceback.format_exc())
            finally:
                self.running = False
                self.generate_report() # Generate report at the end of backtest

        else:
            # --- Original Live Trading Loop ---
            if not self.initialize_mt5():
                self.log_error("Failed to initialize MT5. Exiting bot.")
                return

            self.log_info("Starting multi-symbol live trading loop...")

            try:
                while self.running:
                    start_time = datetime.now()
                    symbols_to_process = self.config.symbols if hasattr(self.config, 'symbols') and self.config.symbols else []
                    if not symbols_to_process:
                        self.log_error("No symbols configured to trade. Please check Config.symbols.")
                        break

                    best_signal = None
                    self.check_open_positions() # Check live positions once per loop

                    for symbol in symbols_to_process:
                        self.log_info(f"Processing live symbol: {symbol}")
                        data = self.get_market_data(symbol=symbol)
                        if data.empty:
                            self.log_warning(f"[{symbol}] Skipping processing: No valid live market data.")
                            continue

                        self.analyze_market_regime(symbol, data)
                        arcs = self.detect_arcs(data)
                        if symbol == symbols_to_process[0]:
                            self.current_arcs = arcs

                        self.manage_open_positions(data) # Manage live positions for this symbol
                        signals = self.find_signals(arcs, data)

                        if signals:
                            current_symbol_top_signal = signals[0]
                            if current_symbol_top_signal['conviction'] > 0.7:
                                if (not best_signal) or (current_symbol_top_signal['conviction'] > best_signal['conviction']):
                                    best_signal = current_symbol_top_signal

                        if symbol == symbols_to_process[0]:
                            self.visualize(data)

                    if best_signal:
                        self.log_info(f"Attempting to place best live signal: {best_signal['type'].upper()} for {best_signal['symbol']} "
                                      f"with conviction {best_signal['conviction']:.2f}")
                        self.place_order(best_signal)
                    else:
                        self.log_info("No strong signal found across all symbols in this iteration.")

                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    sleep_duration = max(0, self.config.visual_update_seconds - elapsed_time)
                    time.sleep(sleep_duration)

            except KeyboardInterrupt:
                self.log_info("Bot stopped by user (KeyboardInterrupt).")
            except Exception as e:
                self.log_error(f"Critical error in main live loop: {str(e)}")
                import traceback
                self.log_error(traceback.format_exc())
            finally:
                self.running = False
                self.log_info("Shutting down MT5 connection and closing plots...")
                mt5.shutdown()
                plt.close('all')

if __name__ == '__main__':
    bot = ArcTradingBot()
    bot.run()

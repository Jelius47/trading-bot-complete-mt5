"""
Risk-Averse Optimized Elliptical Arc Trading Bot with MT5 Integration

This bot identifies trading opportunities by fitting elliptical arcs to price data,
combining this with advanced risk management, dynamic position sizing, and
technical indicator confirmations. It supports multi-symbol trading and
visualizes key market data and bot activity.
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

    # Advanced features
    anti_correlation_min = -0.6  # Not currently used in the provided code, but good for future expansion
    max_open_trades = 3  # Limit total number of concurrent open trades
    max_open_sells = 1  # Limit to only one active SELL position at a time (risk control)

    # Performance optimization
    momentum_lookback = 20  # Periods for momentum indicator calculation
    volatility_lookback = 14  # Periods for volatility (ATR) calculation

    # System settings
    visual_update_seconds = 15  # How often to refresh the visualization in seconds
    log_level = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
    backtest_mode = False  # Set to True for backtesting (disables live trading)


class ArcTradingBot:
    def __init__(self):
        """
        Initializes the trading bot, setting up configurations,
        internal state variables, and logging.
        """
        self.config = Config()
        self.running = False  # Flag to control the main bot loop
        self.current_arcs = []  # Stores detected arcs for visualization
        self.open_positions = {}  # Dictionary to track all open positions by ticket
        self.open_sells = 0  # Counter for active sell positions
        self.trade_history = deque(maxlen=100)  # Stores recent closed trade outcomes
        self.last_signal_time = {}  # Prevents rapid re-entry for the same signal type
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
        # Dynamically set the logging method based on config.log_level
        self.log_method = getattr(self, f"log_{self.config.log_level.lower()}")

    def _log(self, level: str, message: str) -> None:
        """Internal helper for logging messages."""
        print(f"[{level.upper()}] {datetime.now().strftime('%H:%M:%S')}: {message}")

    def log_debug(self, message: str) -> None:
        """Logs a debug message if the configured log level allows."""
        if self.config.log_level in ["DEBUG"]:
            self._log("DEBUG", message)

    def log_info(self, message: str) -> None:
        """Logs an info message if the configured log level allows."""
        if self.config.log_level in ["DEBUG", "INFO"]:
            self._log("INFO", message)

    def log_warning(self, message: str) -> None:
        """Logs a warning message if the configured log level allows."""
        if self.config.log_level in ["DEBUG", "INFO", "WARNING"]:
            self._log("WARNING", message)

    def log_error(self, message: str) -> None:
        """Logs an error message (always printed)."""
        self._log("ERROR", message)

    def initialize_mt5(self) -> bool:
        """
        Initializes connection to the MetaTrader 5 platform.
        Returns True on successful connection, False otherwise.
        """
        self.log_info("Initializing MT5...")
        if not mt5.initialize():
            self.log_error(f"MT5 initialization failed: {mt5.last_error()}")
            return False

        # # Check if connected
        # if not mt5.is_connected():
        #     self.log_error("MT5 is not connected after initialization.")
        #     return False

        info = mt5.terminal_info()
        account = mt5.account_info()

        if not info or not account:
            self.log_error("Failed to retrieve MT5 terminal or account info.")
            mt5.shutdown()
            return False

        self.log_info("✅ MT5 connected successfully:")
        self.log_info(f"  Path:          {info.path}")
        self.log_info(f"  Trade Allowed: {'Yes' if info.trade_allowed else 'No'}")
        self.log_info(f"  Version:       {mt5.version()[0]}.{mt5.version()[1]}")
        self.log_info(f"  Account:       {account.login} ({account.server})")
        self.log_info(f"  Balance:       {account.balance} {account.currency}")

        # Ensure all configured symbols are selected
        for symbol in self.config.symbols:
            if not mt5.symbol_select(symbol, True):
                self.log_error(f"Symbol {symbol} not available on MT5. Please check symbol name or broker.")
                # It's better to continue if some symbols are not available rather than stopping the bot
                # but log a warning or remove the symbol from the config if it's critical.
                # For now, we'll just log and continue.
        return True

    def get_market_data(self, symbol: str, count: int = None, additional_indicators: bool = True) -> pd.DataFrame:
        """
        Fetches market data for a specified symbol and calculates technical indicators.
        Generates synthetic data if real data is unavailable (for testing/development).
        """
        if count is None:
            count = self.config.arc_lookback * 2 # Fetch enough data for lookback and indicators

        # Ensure the symbol is selected before copying rates
        if not mt5.symbol_select(symbol, True):
            self.log_error(f"Failed to select symbol {symbol} for data retrieval.")
            return pd.DataFrame() # Return empty DataFrame on failure

        rates = mt5.copy_rates_from_pos(symbol, self.config.timeframe, 0, count)

        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            if additional_indicators:
                # Calculate Exponential Moving Averages (EMA)
                df['ema20'] = EMAIndicator(close=df['close'], window=20, fillna=True).ema_indicator()
                df['ema50'] = EMAIndicator(close=df['close'], window=50, fillna=True).ema_indicator()
                df['ema200'] = EMAIndicator(close=df['close'], window=200, fillna=True).ema_indicator()

                # Calculate Relative Strength Index (RSI)
                df['rsi'] = RSIIndicator(close=df['close'], window=14, fillna=True).rsi()

                # Calculate Average True Range (ATR) for volatility
                atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14, fillna=True)
                df['atr'] = atr_indicator.average_true_range()

                # Calculate Moving Average Convergence Divergence (MACD)
                macd_indicator = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
                df['macd'] = macd_indicator.macd()
                df['macd_signal'] = macd_indicator.macd_signal()
                df['macd_hist'] = macd_indicator.macd_diff()

                # Store current volatility for historical tracking
                # Ensure there's enough data for ATR calculation before accessing last element
                if not df['atr'].empty and not df['close'].empty:
                    current_volatility = df['atr'].iloc[-1] / df['close'].iloc[-1] * 100
                    # Append to the symbol-specific deque
                    self.volatility_history[symbol].append(current_volatility)
                    df['volatility'] = current_volatility
                else:
                    df['volatility'] = 0.0 # Default value if ATR or close is not available

                # Determine trend based on EMAs
                df['trend'] = np.where(df['ema20'] > df['ema50'], 1,
                                    np.where(df['ema20'] < df['ema50'], -1, 0))
                # Identify swing highs and lows (simple approach)
                df['swing_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
                df['swing_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))

            return df.dropna() # Drop rows with NaN values resulting from indicator calculations

        self.log_warning(f"Market data unavailable for {symbol} from MT5. Generating synthetic data for testing...")
        # Generate synthetic data for development/testing if MT5 data is not available
        now = datetime.now()
        dates = pd.date_range(end=now, periods=count, freq='H')
        # Create more realistic synthetic price movements
        np.random.seed(42) # for reproducibility
        prices = 1.1 + np.cumsum(np.random.normal(0, 0.0005, count)) + np.sin(np.arange(count) * 0.1) * 0.01
        df = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.0002, count))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.0002, count))),
            'close': prices,
            'tick_volume': np.random.randint(100, 1000, count),
            'spread': np.random.randint(1, 5, count),
            'real_volume': np.random.randint(10, 100, count)
        }, index=dates)

        # Recalculate indicators for synthetic data to ensure consistency
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

    def fit_arc(self, data: pd.DataFrame, start: int, end: int) -> dict | None:
        """
        Fits an elliptical arc to a segment of price data using `scipy.optimize.curve_fit`.
        Calculates R-squared, curvature, and a quality score for the arc.
        """
        x = np.arange(end - start)
        y = data['close'].iloc[start:end].values

        if len(x) < 5: # Need at least 5 points for curve_fit with 5 parameters
            self.log_debug("Not enough data points for arc fitting.")
            return None

        # Normalize x and y for better curve_fit performance and stability
        x_norm = x / len(x)
        y_mean, y_std = y.mean(), y.std()
        # Avoid division by zero if y_std is 0 (flat line)
        y_norm = (y - y_mean) / (y_std if y_std != 0 else 1)

        # Elliptical function definition
        def elliptical(x_val, a, b, c, d, e):
            """
            Mathematical function for fitting an elliptical arc.
            a, b, c: quadratic components
            d, e: sinusoidal components for oscillation/curvature
            """
            return a * x_val ** 2 + b * x_val + c + d * np.sin(e * x_val)

        try:
            # p0: initial guess for parameters
            # bounds: min/max values for parameters to constrain the fit
            popt, pcov = curve_fit(
                elliptical, x_norm, y_norm,
                p0=[0, 0, 0, 0.1, 5], # Reasonable initial guesses
                bounds=([-5, -5, -5, -5, 0.1], [5, 5, 5, 5, 20]), # Parameter constraints
                maxfev=10000 # Maximum number of function evaluations
            )

            # Reconstruct the fitted curve in original scale
            fit = elliptical(x_norm, *popt) * y_std + y_mean

            # Calculate R-squared (goodness of fit)
            ss_res = ((y - fit) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0 # Handle division by zero

            # Calculate curvature (absolute value of second derivative at midpoint, adjusted)
            # The original curvature calculation was simplified, let's make it more robust
            # Curvature of y = f(x) is |f''(x)| / (1 + f'(x)^2)^(3/2)
            # For elliptical(x, a, b, c, d, e):
            # f'(x) = 2*a*x + b + d*e*cos(e*x)
            # f''(x) = 2*a - d*e^2*sin(e*x)
            # We can approximate curvature based on the sinusoidal component's amplitude and frequency
            curvature = abs(popt[3] * popt[4]) # Amplitude * Frequency of sin component

            # Determine arc direction based on the quadratic coefficient 'a'
            arc_direction = 1 if popt[0] < 0 else -1 # -1 for upward opening (bearish), 1 for downward opening (bullish)

            # End direction: slope at the end of the arc
            # f'(x_norm_end) = 2*a*x_norm[-1] + b + d*e*cos(e*x_norm[-1])
            end_direction = np.sign(2 * popt[0] * x_norm[-1] + popt[1] + popt[3] * popt[4] * np.cos(popt[4] * x_norm[-1]))

            # Parameter stability (inverse of how much parameters vary relative to their values)
            perr = np.sqrt(np.diag(pcov)) # Standard deviations of the parameters
            # Avoid division by zero for popt values close to zero
            param_stability = np.mean(np.abs(perr / (np.abs(popt) + 1e-9)))
            # Cap param_stability to prevent extreme values from dominating quality score
            param_stability = min(1.0, param_stability)


            # Remaining potential (how much of the arc's 'movement' is left)
            # Extend the fitted curve to see where it might go
            x_extended = np.linspace(0, 1.5, 100) # Extend 50% beyond the fitted range
            y_extended = elliptical(x_extended, *popt) * y_std + y_mean

            # Find the extremum (peak/trough) in the extended part
            if arc_direction > 0: # Upward arc (expecting a trough)
                extremum_idx = np.argmin(y_extended[x_extended > x_norm[-1]]) if np.any(x_extended > x_norm[-1]) else -1
            else: # Downward arc (expecting a peak)
                extremum_idx = np.argmax(y_extended[x_extended > x_norm[-1]]) if np.any(x_extended > x_norm[-1]) else -1

            if extremum_idx != -1:
                # Adjust extremum_idx to be relative to the start of x_extended
                extremum_x_norm = x_extended[x_extended > x_norm[-1]][extremum_idx]
                # Remaining potential is the distance from current end to the extremum
                remaining_potential = max(0, extremum_x_norm - x_norm[-1])
                # Normalize remaining potential by the length of the fitted arc
                remaining_potential = remaining_potential / max(0.01, (x_norm[-1] - x_norm[0]))
            else:
                remaining_potential = 0.0 # No clear extremum found in extended range

            # Calculate overall quality score
            # Quality combines R-squared, curvature, parameter stability, and remaining potential
            # (1 - param_stability) means higher stability gives higher quality
            # (1 + remaining_potential) means more potential gives higher quality
            quality = r2 * curvature * (1 - param_stability) * (1 + remaining_potential)
            # Ensure quality is not negative due to (1 - param_stability) if param_stability is > 1
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
                'volatility': y_std, # Use y_std for volatility of the segment
                'param_stability': param_stability,
                'arc_direction': arc_direction, # 1 for bullish (downward opening), -1 for bearish (upward opening)
                'end_direction': end_direction, # Slope at the end of the arc
                'remaining_potential': remaining_potential,
                'quality': quality
            }

        except RuntimeError as re:
            # curve_fit can raise RuntimeError if it fails to converge
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
        """
        arcs = []
        # Iterate through different window sizes for arc detection
        for size in [30, 40, 50, 60, 80, 100]: # Added more sizes for better detection
            step = max(5, size // 5) # Step size for sliding window
            if len(data) < size:
                continue # Skip if data is too short for the current window size

            for i in range(0, len(data) - size + 1, step): # Iterate through data with sliding window
                arc = self.fit_arc(data, i, i + size)
                if arc:
                    # Filter arcs based on configured quality thresholds
                    if arc['r2'] > self.config.min_r2 and arc['curvature'] > self.config.min_curvature:
                        # Ensure there's enough volatility in the segment
                        if arc['volatility'] > 0.00001: # A very small threshold to avoid flat lines
                            arcs.append(arc)

        if not arcs:
            self.log_info("No arcs detected meeting initial quality criteria.")
            return []

        # Cluster arcs to avoid redundant signals from overlapping patterns
        # Use the midpoint of the arc as the feature for clustering
        arc_midpoints = np.array([[a['start_idx'] + (a['end_idx'] - a['start_idx']) / 2] for a in arcs])

        # DBSCAN clustering: eps is the maximum distance between two samples for one to be considered as in the neighborhood of the other.
        # min_samples is the number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        # Here, we want to group arcs that are close to each other.
        # Adjust eps based on the typical distance between arc midpoints for your timeframe.
        # A value of 15-20 seems reasonable for H1 data.
        clustering = DBSCAN(eps=20, min_samples=1).fit(arc_midpoints)
        labels = clustering.labels_
        best_arcs = []

        # Select the highest quality arc from each cluster
        for cluster_id in set(labels):
            if cluster_id == -1: # -1 indicates noise points (outliers)
                continue
            cluster_arcs = [arc for i, arc in enumerate(arcs) if labels[i] == cluster_id]
            if cluster_arcs:
                # Select the arc with the highest 'quality' score within the cluster
                best_arc = max(cluster_arcs, key=lambda a: a['quality'])
                best_arcs.append(best_arc)

        self.log_info(f"Detected {len(best_arcs)} high-quality, distinct arcs after clustering.")
        # Sort arcs by quality in descending order
        return sorted(best_arcs, key=lambda a: a['quality'], reverse=True)

    def analyze_market_regime(self, symbol: str, data: pd.DataFrame) -> dict:
        """
        Analyzes current market conditions (trend and volatility) for a given symbol.
        This helps the bot adapt its trading strategy to different market environments.
        """
        if data.empty:
            self.log_warning(f"[{symbol}] Cannot analyze market regime: no data.")
            return {"trend": "neutral", "volatility": "medium"}

        close_prices = data['close'].values
        # Trend analysis using EMA20 and EMA50 crossover
        if data['ema20'].iloc[-1] > data['ema50'].iloc[-1]:
            trend = "bullish"
        elif data['ema20'].iloc[-1] < data['ema50'].iloc[-1]:
            trend = "bearish"
        else:
            trend = "ranging" # Or neutral

        # Volatility analysis using ATR
        current_atr = data['atr'].iloc[-1]
        # Ensure volatility history is not empty for the symbol
        if self.volatility_history[symbol]:
            avg_vol = np.mean(list(self.volatility_history[symbol])) # Convert deque to list for mean
        else:
            avg_vol = current_atr # If no history, use current ATR

        if current_atr > avg_vol * 1.5: # Current ATR significantly higher than average
            vol_regime = "high"
        elif current_atr < avg_vol * 0.7: # Current ATR significantly lower than average
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
        """
        signals = []
        current_price = data['close'].iloc[-1]
        symbol = self.config.symbols[0] # Assuming this is called per symbol in the main loop

        market_regime = self.analyze_market_regime(symbol, data)

        # Do not generate new signals in high volatility if positions are already open
        if market_regime["volatility"] == "high" and self.open_positions:
            self.log_warning(f"[{symbol}] High volatility detected and positions open - not generating new signals.")
            return []

        # Check portfolio risk before generating new signals
        current_risk = self.calculate_portfolio_risk()
        if current_risk >= self.config.max_portfolio_risk:
            self.log_warning(f"[{symbol}] Maximum portfolio risk reached ({current_risk:.2f}%) - not generating new signals.")
            return []

        # Check maximum number of open trades
        if len(self.open_positions) >= self.config.max_open_trades:
            self.log_info(f"[{symbol}] Maximum open trades ({self.config.max_open_trades}) reached - not generating new signals.")
            return []

        for arc in arcs:
            # Filter out arcs below the quality threshold
            if arc['quality'] < self.config.quality_threshold:
                self.log_debug(f"[{symbol}] Arc quality {arc['quality']:.2f} below threshold {self.config.quality_threshold}.")
                continue

            # Calculate the derivative (slope) at the end of the arc
            # This indicates the immediate direction and strength of the arc's curve
            a, b, c, d, e = arc['params'] # c is not used in derivative
            def deriv_at_x(x_val):
                return 2 * a * x_val + b + d * e * math.cos(e * x_val)

            # Evaluate derivative at the end of the fitted segment (normalized x=1)
            current_deriv = deriv_at_x(1)
            # Evaluate derivative slightly before the end to gauge change in slope
            prev_deriv = deriv_at_x(0.9) # Small step back
            # Angle difference to quantify the "turn" of the arc
            angle_diff = math.degrees(math.atan(current_deriv) - math.atan(prev_deriv))

            signal_type = None

            # Bullish signal criteria
            # Positive angle_diff (turning upwards), positive end_direction (upward slope at end)
            # and either not bearish trend or very high arc quality
            if (angle_diff > 15 and arc['end_direction'] > 0 and
                (market_regime["trend"] != "bearish" or arc['quality'] > 0.8)):
                # Prevent rapid re-entry for buy signals (e.g., within 1 hour)
                if 'buy' in self.last_signal_time and \
                   (datetime.now() - self.last_signal_time['buy']).total_seconds() < 3600:
                    self.log_debug(f"[{symbol}] Buy signal too soon after last one. Skipping.")
                    continue
                signal_type = 'buy'

            # Bearish signal criteria
            # Negative angle_diff (turning downwards), negative end_direction (downward slope at end)
            # and either not bullish trend or very high arc quality
            # Also check max_open_sells limit
            elif (angle_diff < -15 and arc['end_direction'] < 0 and
                  (market_regime["trend"] != "bullish" or arc['quality'] > 0.8) and
                  self.open_sells < self.config.max_open_sells):
                # Prevent rapid re-entry for sell signals
                if 'sell' in self.last_signal_time and \
                   (datetime.now() - self.last_signal_time['sell']).total_seconds() < 3600:
                    self.log_debug(f"[{symbol}] Sell signal too soon after last one. Skipping.")
                    continue
                signal_type = 'sell'

            if signal_type:
                # Conviction combines arc quality and the strength of the angle change
                conviction = min(1.0, arc['quality'] * (1 + abs(angle_diff) / 90)) # Max conviction 1.0

                # Validate signal with other technical indicators
                indicator_confirmation = self.validate_with_indicators(data, signal_type)

                # Only add signal if indicator confirmation is strong enough
                if indicator_confirmation >= 0.6: # Minimum 60% confirmation from indicators
                    signals.append({
                        'type': signal_type,
                        'price': current_price,
                        'optimal_entry': self.calculate_optimal_entry(data, signal_type),
                        'volatility': arc['volatility'],
                        'quality': arc['quality'],
                        'conviction': conviction * indicator_confirmation, # Final conviction score
                        'arc_id': id(arc), # Unique ID for the arc
                        'symbol': symbol # Add symbol to the signal
                    })
                    self.last_signal_time[signal_type] = datetime.now() # Update last signal time

        if signals:
            # Sort signals by conviction in descending order and log the top one
            signals = sorted(signals, key=lambda s: s['conviction'], reverse=True)
            top_signal = signals[0]
            self.log_info(f"[{symbol}] Generated {len(signals)} signals. Top: {top_signal['type'].upper()} with {top_signal['conviction']:.2f} conviction.")
        else:
            self.log_info(f"[{symbol}] No valid trade signals generated.")

        return signals

    def validate_with_indicators(self, data: pd.DataFrame, signal_type: str) -> float:
        """
        Validates a potential trade signal using a set of technical indicators.
        Returns a confirmation score (0.0 to 1.0) based on how many indicators
        support the signal.
        """
        confirmations = 0
        total_indicators = 5 # Number of indicators used for validation

        if data.empty or len(data) < max(self.config.momentum_lookback, self.config.volatility_lookback, 200):
            self.log_warning("Not enough data to validate with indicators.")
            return 0.0

        # Indicator 1: EMA Crossover (Trend confirmation)
        if signal_type == 'buy':
            if data['ema20'].iloc[-1] > data['ema50'].iloc[-1]:
                confirmations += 1
        else: # sell signal
            if data['ema20'].iloc[-1] < data['ema50'].iloc[-1]:
                confirmations += 1

        # Indicator 2: RSI (Overbought/Oversold confirmation)
        rsi = data['rsi'].iloc[-1]
        if signal_type == 'buy' and rsi < 70: # Not overbought for a buy
            confirmations += 1
        elif signal_type == 'sell' and rsi > 30: # Not oversold for a sell
            confirmations += 1

        # Indicator 3: MACD Histogram (Momentum confirmation)
        if signal_type == 'buy' and data['macd_hist'].iloc[-1] > 0: # Bullish momentum
            confirmations += 1
        elif signal_type == 'sell' and data['macd_hist'].iloc[-1] < 0: # Bearish momentum
            confirmations += 1

        # Indicator 4: Volatility Check (Avoid high volatility entries unless specified)
        symbol = self.config.symbols[0] # Assuming this is called per symbol in the main loop
        current_atr = data['atr'].iloc[-1]
        if self.volatility_history[symbol]:
            avg_vol = np.mean(list(self.volatility_history[symbol]))
        else:
            avg_vol = current_atr # Fallback if history is empty

        # Confirm if current volatility is not excessively high
        if current_atr < avg_vol * 1.5: # Current volatility is within 1.5x of average
            confirmations += 1

        # Indicator 5: Price vs. EMA200 (Long-term trend alignment)
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

        if atr == 0: # Prevent division by zero or nonsensical ATR
            atr = 0.0001 # A small default if ATR is zero

        if signal_type == 'buy':
            return {
                'immediate': current_price,
                # Preferred entry: slightly below current price, but not lower than recent low
                'preferred': max(current_price - 0.3 * atr, data['low'].iloc[-1]),
                # Limit entry: deeper pullback, but not lower than recent significant low
                'limit': max(current_price - 0.5 * atr, data['low'].iloc[-5:].min())
            }
        else: # signal_type == 'sell'
            return {
                'immediate': current_price,
                # Preferred entry: slightly above current price, but not higher than recent high
                'preferred': min(current_price + 0.3 * atr, data['high'].iloc[-1]),
                # Limit entry: deeper pullback, but not higher than recent significant high
                'limit': min(current_price + 0.5 * atr, data['high'].iloc[-5:].max())
            }

    def calculate_portfolio_risk(self) -> float:
        """
        Calculates the current total risk exposure of all open positions
        as a percentage of the account balance.
        """
        if not self.open_positions:
            return 0.0

        account_info = mt5.account_info()
        if not account_info:
            self.log_error("Failed to get account info for portfolio risk calculation.")
            return 0.0

        total_risk_percent = 0.0
        for ticket, pos in self.open_positions.items():
            if pos['sl'] != 0: # Only consider positions with a stop loss
                symbol_info = mt5.symbol_info(pos['symbol']) # Get symbol info for the specific position
                if not symbol_info:
                    self.log_warning(f"Could not get symbol info for {pos['symbol']} (ticket {ticket}). Skipping risk calculation for this position.")
                    continue

                # Calculate risk amount in deposit currency
                # Risk = (Entry Price - SL Price) * Volume * Contract Size * Tick Value / Point
                # For forex, often (Price difference in pips) * Pip Value * Lots
                # Let's use the actual tick value and point for accuracy
                price_diff = abs(pos['price'] - pos['sl'])
                # Convert price difference to deposit currency using tick value and point
                # For 1 lot, 1 pip value is symbol_info.trade_tick_value / symbol_info.point
                # Total risk in deposit currency = price_diff / symbol_info.point * symbol_info.trade_tick_value * pos['volume']
                risk_amount_in_currency = price_diff / symbol_info.point * symbol_info.trade_tick_value * pos['volume']

                risk_percent = (risk_amount_in_currency / account_info.balance) * 100
                total_risk_percent += risk_percent

        self.log_debug(f"Current portfolio risk: {total_risk_percent:.2f}%")
        return total_risk_percent

    def calculate_position_size(self, sl_pips: float, conviction: float = 1.0, symbol_name: str = None) -> float:
        """
        Calculates the appropriate position size (in lots) based on risk per trade,
        stop loss in pips, and signal conviction.
        """
        account_info = mt5.account_info()
        if not account_info:
            self.log_error("Failed to get account info for position sizing.")
            return self.config.min_lots

        if symbol_name is None:
            symbol_name = self.config.symbols[0] # Default to first symbol if not specified

        symbol_info = mt5.symbol_info(symbol_name)
        if not symbol_info:
            self.log_error(f"Failed to get symbol info for {symbol_name} for position sizing.")
            return self.config.min_lots

        if sl_pips <= 0:
            self.log_warning("Stop loss pips must be positive for position sizing. Using min_lots.")
            return self.config.min_lots

        # Calculate pip value in deposit currency for 1 lot
        # pip_value_per_lot = symbol_info.trade_tick_value / symbol_info.point
        # This is more direct: value of 1 pip for 1 lot
        # For EURUSD, 1 pip = 0.0001, if tick_size = 0.00001, then point = 0.00001
        # trade_tick_value is the value of a tick in deposit currency
        # For a standard lot (100,000 units), 1 pip movement is usually $10 for USD pairs
        # Let's use the actual contract size and tick value for precision
        # Value of 1 pip for 1 standard lot = symbol_info.trade_tick_value / symbol_info.point * symbol_info.trade_contract_size
        # This is often simplified for forex to $10 per pip per standard lot.
        # Let's calculate the value of one pip for a single unit of the base currency
        # and then scale by contract size.
        # A more robust way:
        # Get the value of one point for one lot.
        # This is `symbol_info.trade_tick_value` if `symbol_info.point` is the tick size
        # and `symbol_info.trade_contract_size` is the contract size.
        # For simplicity and common forex pairs, 1 pip for 1 lot is often 10 USD.
        # Let's use `symbol_info.trade_tick_value` which is the value of a tick in the deposit currency.
        # And `symbol_info.point` is the size of a tick.
        # So, value of 1 pip = (1 pip / symbol_info.point) * symbol_info.trade_tick_value
        # For example, if point is 0.00001 and tick_value is 0.1, then 1 pip (0.0001) is 10 ticks.
        # Value of 1 pip = (0.0001 / 0.00001) * 0.1 = 10 * 0.1 = 1 USD per lot.
        # This is for a standard lot.
        # So, for 1 lot, the value of 1 pip in deposit currency is `symbol_info.trade_tick_value / symbol_info.point * 10` (assuming 1 pip = 10 points).
        # A safer way is to calculate risk per lot directly:
        # Risk per lot = sl_pips * (symbol_info.trade_tick_value / symbol_info.point * 10) # Assuming 1 pip = 10 points
        # Or, more accurately:
        # Risk per lot = sl_pips * symbol_info.point * symbol_info.trade_contract_size * (conversion_rate_to_deposit_currency)
        # For simplicity, if base currency is USD, and deposit currency is USD, then:
        # Risk per lot (in deposit currency) = sl_pips * symbol_info.point * symbol_info.trade_contract_size
        # This is still tricky. Let's use the direct calculation of risk amount per lot for the given SL.
        # Assume 1 pip = 10 points for calculation purposes if not explicitly defined by symbol_info.
        # A more robust way is to calculate the value of one point for one lot:
        # point_value_per_lot = symbol_info.trade_tick_value / symbol_info.point
        # Risk per lot = sl_pips * 10 * point_value_per_lot # Assuming 1 pip = 10 points
        # This is the most common interpretation for forex.

        # Calculate the value of 1 pip for 1 standard lot (100,000 units)
        # This is often $10 for major USD pairs.
        # Let's use the actual symbol info to derive it:
        # The value of a change of `symbol_info.point` (tick size) for 1 unit of the base currency is `symbol_info.trade_tick_value`.
        # So, for `symbol_info.trade_contract_size` units (1 lot), the value of 1 tick is `symbol_info.trade_tick_value * symbol_info.trade_contract_size`.
        # The value of 1 pip (which is typically 10 ticks for 5-digit brokers) for 1 lot:
        # pip_value_per_lot = (10 * symbol_info.trade_tick_value) * symbol_info.trade_contract_size
        # This is still not quite right. A simpler way is to use `mt5.order_calc_profit` or `mt5.order_check`
        # to estimate risk, but that's for a specific order.

        # Let's stick to the common formula:
        # Risk per trade = Account Balance * (Max Risk Percent / 100)
        # Lots = Risk per trade / (SL Pips * Pip Value per Lot)
        # Pip Value per Lot (for 1 standard lot) is often 10 USD for USD quoted pairs.
        # For non-USD quoted pairs, it involves conversion.
        # A simpler way to estimate pip value is `symbol_info.trade_tick_value / symbol_info.point`
        # This gives the value of 1 point for 1 unit of base currency.
        # To get for 1 lot: `(symbol_info.trade_tick_value / symbol_info.point) * symbol_info.trade_contract_size`
        # This is the value of 1 pip for 1 lot.

        # Let's use a more direct approach for pip value if possible, or assume a standard value.
        # For now, let's assume `symbol_info.trade_tick_value` is the value of a tick for a standard lot.
        # And `symbol_info.point` is the size of a tick.
        # So, `(1 / symbol_info.point) * symbol_info.trade_tick_value` is the value of 1 unit of price change.
        # To get pip value: `(1 / symbol_info.point) * symbol_info.trade_tick_value * 10` (for 1 pip = 10 points)
        # This is value of 1 pip for 1 unit of volume.
        # So, for `lots` volume: `lots * (1 / symbol_info.point) * symbol_info.trade_tick_value * 10`

        # Re-evaluating pip value calculation:
        # `symbol_info.tick_value` is the value of a tick in the deposit currency.
        # `symbol_info.tick_size` is the minimum price change.
        # `symbol_info.point` is usually the same as `tick_size`.
        # `symbol_info.trade_tick_value` is the value of a tick for 1 lot.
        # So, the value of 1 pip for 1 lot is `symbol_info.trade_tick_value * 10` (assuming 1 pip = 10 ticks).
        # This seems to be the most practical approach for MT5.

        # Ensure `trade_tick_value` is not zero to prevent division by zero
        if symbol_info.trade_tick_value == 0:
            self.log_error(f"Symbol {symbol_name} has zero trade_tick_value. Cannot calculate position size.")
            return self.config.min_lots

        # Calculate the value of 1 pip for 1 lot
        pip_value_per_lot = symbol_info.trade_tick_value * 10 # Assuming 1 pip = 10 ticks

        # Adjusted risk based on dynamic risk adjustment and conviction
        adjusted_risk_percent = self.config.max_risk_percent * self.risk_adjustment * conviction
        risk_amount_in_currency = account_info.balance * (adjusted_risk_percent / 100)

        # Calculate required lots
        # Prevent division by zero for `pip_value_per_lot`
        if pip_value_per_lot == 0:
            self.log_error(f"Calculated pip_value_per_lot for {symbol_name} is zero. Using min_lots.")
            return self.config.min_lots

        # Ensure sl_pips is not zero or negative
        if sl_pips <= 0:
            self.log_error(f"Stop loss pips for {symbol_name} is zero or negative. Using min_lots.")
            return self.config.min_lots

        position_size_lots = risk_amount_in_currency / (sl_pips * pip_value_per_lot)

        # Round to appropriate decimal places for lots (e.g., 0.01 for micro lots)
        lots = max(self.config.min_lots, min(round(position_size_lots, 2), self.config.max_lots))

        self.log_info(f"Position size calculation for {symbol_name}: "
                      f"Risk={adjusted_risk_percent:.2f}%, SL={sl_pips:.1f} pips, "
                      f"Pip Value/Lot={pip_value_per_lot:.2f}, Size={lots:.2f} lots")
        return lots


    def update_risk_adjustment(self) -> None:
        """
        Dynamically adjusts the bot's risk exposure based on recent trade performance.
        Increases risk after a winning streak, decreases after losses.
        """
        if not self.trade_history:
            self.risk_adjustment = 1.0 # Reset or keep neutral if no trades
            self.log_info("No trade history to update risk adjustment.")
            return

        win_count = sum(1 for trade in self.trade_history if trade['profit'] > 0)
        loss_count = len(self.trade_history) - win_count

        win_rate = win_count / len(self.trade_history)

        loss_sum = sum(abs(t['profit']) for t in self.trade_history if t['profit'] < 0)
        profit_sum = sum(t['profit'] for t in self.trade_history if t['profit'] > 0)

        # Avoid division by zero if no losses
        profit_factor = profit_sum / max(0.1, loss_sum)

        # Adjust risk_adjustment based on performance
        if win_rate >= 0.6 and profit_factor >= 1.5:
            # Increase risk if performance is strong
            self.risk_adjustment = min(1.2, self.risk_adjustment + 0.02) # Smaller increment
        elif win_rate <= 0.4 or profit_factor < 1.0:
            # Decrease risk if performance is weak
            self.risk_adjustment = max(0.5, self.risk_adjustment - 0.05) # Larger decrement for losses
        else:
            # Gradually revert to neutral if performance is average
            if self.risk_adjustment < 1.0:
                self.risk_adjustment = min(1.0, self.risk_adjustment + 0.01)
            elif self.risk_adjustment > 1.0:
                self.risk_adjustment = max(1.0, self.risk_adjustment - 0.01)


        self.log_info(f"Risk adjustment updated: {self.risk_adjustment:.2f} "
                      f"(Win Rate: {win_rate:.1%}, Profit Factor: {profit_factor:.2f})")

    def check_open_positions(self) -> None:
        """
        Retrieves and updates the status of all open positions from MT5.
        Identifies newly opened or closed positions and updates `self.open_positions`
        and `self.trade_history`.
        """
        # if not mt5.is_connected():
        #     self.log_error("MT5 is not connected. Cannot check open positions.")
        #     return

        # Get all open positions
        positions = mt5.positions_get()

        current_tickets_on_mt5 = set()
        self.open_sells = 0 # Reset counter for sell positions

        if positions:
            for position in positions:
                ticket = position.ticket
                current_tickets_on_mt5.add(ticket)

                if position.type == mt5.POSITION_TYPE_SELL:
                    self.open_sells += 1

                # If position is new or updated, store/update its details
                if ticket not in self.open_positions:
                    self.open_positions[ticket] = {
                        'ticket': ticket,
                        'symbol': position.symbol, # Store symbol for the position
                        'type': 'sell' if position.type == mt5.POSITION_TYPE_SELL else 'buy',
                        'price': position.price_open,
                        'volume': position.volume,
                        'sl': position.sl,
                        'tp': position.tp,
                        'profit': position.profit,
                        'open_time': datetime.fromtimestamp(position.time) # Use position open time
                    }
                    self.log_info(f"New position detected: {position.symbol} {position.type} #{ticket}")
                else:
                    # Update existing position details
                    self.open_positions[ticket]['profit'] = position.profit
                    self.open_positions[ticket]['sl'] = position.sl
                    self.open_positions[ticket]['tp'] = position.tp
        else:
            self.log_info("No open positions found on MT5.")

        # Identify positions that were previously tracked but are now closed on MT5
        closed_tickets = set(self.open_positions.keys()) - current_tickets_on_mt5
        for ticket in closed_tickets:
            position = self.open_positions.pop(ticket) # Remove from open positions
            position['close_time'] = datetime.now()
            position['duration'] = (position['close_time'] - position['open_time']).total_seconds() / 3600 # in hours
            self.trade_history.append(position) # Add to trade history
            self.log_info(f"Position {ticket} ({position['symbol']}) closed with profit: {position['profit']:.2f}")

        # After checking all positions, update the dynamic risk adjustment
        self.update_risk_adjustment()


    def manage_open_positions(self, data: pd.DataFrame) -> None:
        """
        Manages open positions by implementing trailing stops.
        Can be extended for partial closes or other management strategies.
        """
        if not self.open_positions:
            return

        current_price = data['close'].iloc[-1]
        symbol_name = data.index.name # Get symbol from data's index name if available, otherwise default
        if symbol_name is None:
            symbol_name = self.config.symbols[0] # Fallback

        symbol_info = mt5.symbol_info(symbol_name)
        if not symbol_info:
            self.log_error(f"Failed to get symbol info for {symbol_name} in manage_open_positions.")
            return

        pip_size = symbol_info.point # The value of one point

        # Iterate over a copy of open_positions to allow modification during iteration
        for ticket, position in list(self.open_positions.items()):
            # Ensure the position's symbol matches the current data's symbol
            if position['symbol'] != symbol_name:
                continue

            # Skip if no stop loss is set for the position
            if position['sl'] == 0:
                self.log_debug(f"Position {ticket} has no SL set. Skipping trailing stop.")
                continue

            # Calculate current profit in pips
            if position['type'] == 'buy':
                profit_pips = (current_price - position['price']) / pip_size
            else: # sell position
                profit_pips = (position['price'] - current_price) / pip_size

            # Implement trailing stop
            if self.config.trailing_stop and profit_pips > 0: # Only trail if in profit
                # Calculate the target profit in pips based on initial TP and entry
                if position['tp'] != 0:
                    target_pips = abs(position['tp'] - position['price']) / pip_size
                else:
                    # If no TP set, use a default large target or calculate from SL * profit_ratio
                    target_pips = abs(position['sl'] - position['price']) / pip_size * self.config.profit_ratio
                    if target_pips == 0:
                        self.log_warning(f"Position {ticket} has no TP and SL leads to 0 target pips. Cannot trail.")
                        continue

                # Calculate profit ratio relative to target profit
                profit_ratio = profit_pips / max(0.01, target_pips) # Avoid division by zero

                if profit_ratio >= self.config.trailing_stop_trigger:
                    # Trailing stop level: move SL to lock in a percentage of current profit
                    # For example, if trigger is 0.5 and profit_pips is 100, trailing_level is 50 pips.
                    # This means SL will be moved to 50 pips from entry.
                    # A more aggressive trail would be to move SL to a certain percentage of current profit.
                    # Let's trail by a fixed percentage of current profit.
                    trailing_pips_from_current_price = profit_pips * 0.5 # Trail by 50% of current profit

                    if position['type'] == 'buy':
                        new_sl = current_price - (trailing_pips_from_current_price * pip_size)
                        # Ensure new SL is not worse than current SL and is above entry price
                        if new_sl > position['sl'] and new_sl > position['price']:
                            self.modify_position(ticket, new_sl, position['tp'])
                    else: # sell position
                        new_sl = current_price + (trailing_pips_from_current_price * pip_size)
                        # Ensure new SL is not worse than current SL and is below entry price
                        if new_sl < position['sl'] and new_sl < position['price']:
                            self.modify_position(ticket, new_sl, position['tp'])

    def modify_position(self, ticket: int, sl: float, tp: float) -> None:
        """
        Sends a request to MetaTrader 5 to modify an existing position's
        stop loss (SL) and take profit (TP) levels.
        """
        # if not mt5.is_connected():
        #     self.log_error("MT5 is not connected. Cannot modify position.")
        #     return

        request = {
            "action": mt5.TRADE_ACTION_SLTP, # Action type for modifying SL/TP
            "position": ticket, # Ticket number of the position to modify
            "sl": sl, # New stop loss level
            "tp": tp, # New take profit level
            "deviation": 10, # Allowed deviation from the requested price
            "magic": 123456, # Magic number to identify orders from this bot
            "comment": "Arc Bot - Modify", # Comment for the trade
            "type_time": mt5.ORDER_TIME_GTC, # Good Till Cancelled
        }
        result = mt5.order_send(request)

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.log_info(f"Modified position {ticket}: SL={sl:.5f}, TP={tp:.5f}")
            # Update the local tracking of the position
            if ticket in self.open_positions:
                self.open_positions[ticket]['sl'] = sl
                self.open_positions[ticket]['tp'] = tp
        else:
            self.log_error(f"Failed to modify position {ticket}: {result.retcode} - {result.comment}")
            # Log all relevant result information for debugging
            self.log_error(f"Modification request: {request}")
            self.log_error(f"Modification result: {result}")


    def place_order(self, signal: dict) -> bool:
        """
        Places a new trade order based on a generated signal, incorporating
        stop loss, take profit, and position sizing calculations.
        """
        if self.config.backtest_mode:
            self.log_info("Bot is in backtest mode. Not placing live orders.")
            return False

        # if not mt5.is_connected():
        #     self.log_error("MT5 is not connected. Cannot place order.")
        #     return False

        order_type = signal['type']
        symbol_name = signal['symbol'] # Use the symbol from the signal
        entry_price = signal['optimal_entry']['immediate'] # Use immediate entry price

        # Fetch fresh data for the specific symbol to ensure up-to-date ATR
        # and recent high/lows for SL/TP calculation
        data = self.get_market_data(symbol=symbol_name, count=self.config.arc_lookback)
        if data.empty:
            self.log_error(f"[{symbol_name}] Failed to get market data for order placement.")
            return False

        atr = data['atr'].iloc[-1]
        if atr == 0:
            self.log_warning(f"[{symbol_name}] ATR is zero. Using a default small value for SL/TP calculation.")
            atr = 0.0001

        symbol_info = mt5.symbol_info(symbol_name)
        if not symbol_info:
            self.log_error(f"Failed to get symbol info for {symbol_name}.")
            return False

        # Calculate Stop Loss (SL) level
        # SL should be placed considering recent price action and ATR
        if order_type == 'buy':
            # For buy, SL below recent low, cushioned by ATR
            sl_level = (data['low'].iloc[-5:].min() - 0.5 * atr)
        else: # sell order
            # For sell, SL above recent high, cushioned by ATR
            sl_level = (data['high'].iloc[-5:].max() + 0.5 * atr)

        # Ensure SL is outside the `trade_stops_level` (minimum distance from current price)
        min_dist_points = symbol_info.trade_stops_level # This is in points
        min_dist_price = min_dist_points * symbol_info.point # Convert to price units

        if order_type == 'buy':
            # SL must be at least min_dist_price below entry_price
            sl_level = min(sl_level, entry_price - min_dist_price)
        else: # sell order
            # SL must be at least min_dist_price above entry_price
            sl_level = max(sl_level, entry_price + min_dist_price)

        # Calculate SL in pips for position sizing
        # sl_pips = abs(entry_price - sl_level) / symbol_info.point / 10 # Assuming 1 pip = 10 points
        # More robust: use the actual pip size (often 0.0001 for 4-digit, 0.00001 for 5-digit)
        # Assuming 1 pip is 10 * symbol_info.point for 5-digit brokers, or 1 * symbol_info.point for 4-digit.
        # Let's use a standard definition of pip for calculation.
        # For a 5-digit broker, 1 pip = 0.0001. If symbol.point = 0.00001, then 1 pip = 10 * symbol.point
        # For a 4-digit broker, 1 pip = 0.0001. If symbol.point = 0.0001, then 1 pip = 1 * symbol.point
        # Let's just calculate the price difference and divide by a standard pip size for consistency.
        # A common way to define pips for forex is 0.0001 for most pairs.
        # For JPY pairs, it's 0.01.
        # A robust pip calculation:
        if "JPY" in symbol_name:
            pip_divisor = 0.01
        else:
            pip_divisor = 0.0001
        sl_pips = abs(entry_price - sl_level) / pip_divisor

        # Ensure SL pips is not too small
        if sl_pips < 5: # Minimum 5 pips SL to avoid being too tight
            self.log_warning(f"[{symbol_name}] Calculated SL pips ({sl_pips:.1f}) too small. Adjusting to 5 pips.")
            sl_pips = 5
            # Recalculate sl_level based on new sl_pips
            if order_type == 'buy':
                sl_level = entry_price - (sl_pips * pip_divisor)
            else:
                sl_level = entry_price + (sl_pips * pip_divisor)


        # Calculate Take Profit (TP) level based on profit_ratio
        tp_pips = sl_pips * self.config.profit_ratio
        if order_type == 'buy':
            tp_level = entry_price + (tp_pips * pip_divisor)
        else: # sell order
            tp_level = entry_price - (tp_pips * pip_divisor)

        # Calculate position size in lots
        lots = self.calculate_position_size(sl_pips, signal['conviction'], symbol_name=symbol_name)

        # Final checks before placing order
        if lots < self.config.min_lots:
            self.log_warning(f"[{symbol_name}] Calculated lots ({lots:.2f}) too small. Skipping order.")
            return False

        if len(self.open_positions) >= self.config.max_open_trades:
            self.log_warning(f"[{symbol_name}] Maximum open trades ({self.config.max_open_trades}) reached - skipping order.")
            return False

        if order_type == 'sell' and self.open_sells >= self.config.max_open_sells:
            self.log_warning(f"[{symbol_name}] Maximum open sell positions ({self.config.max_open_sells}) reached - skipping sell order.")
            return False

        self.log_info(f"[{symbol_name}] Placing order: {order_type.upper()} @ {entry_price:.5f}, "
                      f"SL={sl_level:.5f}, TP={tp_level:.5f}, Size={lots:.2f}")

        # Prepare the MT5 order request
        request = {
            'action': mt5.TRADE_ACTION_DEAL, # Market execution order
            'symbol': symbol_name,
            'volume': lots,
            'type': mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL,
            'price': entry_price, # Requested entry price
            'sl': sl_level,
            'tp': tp_level,
            'deviation': 20, # Allowed deviation in points from the requested price
            'magic': 123456, # Unique ID for bot's trades
            'comment': f'Arc Bot - Q:{signal["quality"]:.2f} C:{signal["conviction"]:.2f}',
            'type_time': mt5.ORDER_TIME_GTC, # Good Till Cancelled
            'type_filling': mt5.ORDER_FILLING_FOK # Fill Or Kill: either fully filled or cancelled
        }

        result = mt5.order_send(request)

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.log_info(f"[{symbol_name}] Order successfully placed: ticket #{result.order}")
            if order_type == 'sell':
                self.open_sells += 1
            # Add the new position to local tracking
            self.open_positions[result.order] = {
                'ticket': result.order,
                'symbol': symbol_name,
                'type': order_type,
                'price': result.price, # Use actual execution price from result
                'volume': lots,
                'sl': sl_level,
                'tp': tp_level,
                'profit': 0, # Initial profit is 0
                'open_time': datetime.now()
            }
            return True
        else:
            self.log_error(f"[{symbol_name}] Order failed: {result.retcode} - {result.comment}")
            # Log all relevant result information for debugging
            self.log_error(f"Order request: {request}")
            self.log_error(f"Order result: {result}")
            return False

    def visualize(self, data: pd.DataFrame) -> None:
        """
        Creates a real-time visualization of market data, detected arcs,
        and open positions using Matplotlib.
        """
        if data.empty:
            self.log_warning("No data to visualize.")
            return

        # Initialize figure and axes if not already created
        if not self.fig:
            plt.ion() # Turn on interactive mode
            self.fig, (self.ax, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(14, 12),
                                                                   gridspec_kw={'height_ratios': [3, 1, 1]})
            plt.subplots_adjust(hspace=0.3) # Adjust space between subplots
            self.fig.suptitle(f"Arc Trading Bot - {self.config.symbols[0]} Live Data", fontsize=16)

        # Clear previous plots
        self.ax.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Plot Price and EMAs on the main chart (ax)
        self.ax.plot(data.index, data['close'], label='Price', linewidth=1.5, color='blue')
        self.ax.plot(data.index, data['ema20'], label='EMA20', linewidth=1, alpha=0.7, color='orange')
        self.ax.plot(data.index, data['ema50'], label='EMA50', linewidth=1, alpha=0.7, color='purple')
        self.ax.plot(data.index, data['ema200'], label='EMA200', linewidth=1, alpha=0.7, color='gray', linestyle='--')


        # Plot detected arcs
        if self.current_arcs:
            # Sort arcs by quality for better visualization (highest quality on top)
            sorted_arcs = sorted(self.current_arcs, key=lambda a: a['quality'], reverse=True)
            for i, arc in enumerate(sorted_arcs[:self.config.max_arcs]):
                start_idx, end_idx = arc['start_idx'], arc['end_idx']
                # Ensure indices are within data bounds
                if start_idx >= len(data) or end_idx > len(data):
                    continue
                color = 'green' if arc['arc_direction'] > 0 else 'red' # Green for bullish, Red for bearish
                # Alpha based on quality, making higher quality arcs more visible
                best_quality = max(0.001, sorted_arcs[0]['quality']) # Avoid division by zero
                raw_alpha = 0.4 + 0.6 * (arc['quality'] / best_quality)
                alpha = max(0.0, min(1.0, raw_alpha)) # Clamp alpha between 0 and 1
                self.ax.plot(data.index[start_idx:end_idx], arc['fitted'],
                             linestyle='--', color=color, alpha=alpha,
                             linewidth=2.5 if i == 0 else (1.5 if i < 3 else 0.8),
                             label=f"Arc Q:{arc['quality']:.2f}" if i == 0 else None) # Only label the best arc

        # Plot open positions (entry, SL, TP)
        for ticket, pos in self.open_positions.items():
            # Only plot positions for the currently visualized symbol
            if pos['symbol'] != self.config.symbols[0]:
                continue

            marker = '^' if pos['type'] == 'buy' else 'v' # Up arrow for buy, down for sell
            color = 'green' if pos['type'] == 'buy' else 'red'
            # Find the closest index in data.index for the open_time
            # This handles cases where open_time might not exactly match a bar's timestamp
            idx_loc = data.index.get_indexer([pos['open_time']], method='nearest')[0]
            if 0 <= idx_loc < len(data.index):
                self.ax.plot(data.index[idx_loc], pos['price'], marker=marker, markersize=12, color=color,
                             label=f"{pos['type'].upper()} Entry" if ticket == list(self.open_positions.keys())[0] else None) # Label only first position

                # Plot SL and TP lines
                if pos['sl'] != 0:
                    self.ax.axhline(y=pos['sl'], linestyle=':', color=color, alpha=0.6,
                                    label=f"{pos['type'].upper()} SL" if ticket == list(self.open_positions.keys())[0] else None)
                if pos['tp'] != 0:
                    self.ax.axhline(y=pos['tp'], linestyle=':', color='blue' if pos['type'] == 'buy' else 'orange', alpha=0.6,
                                    label=f"{pos['type'].upper()} TP" if ticket == list(self.open_positions.keys())[0] else None)

        # Main chart settings
        market_regime_text = self.market_regimes.get(self.config.symbols[0], {"trend": "N/A", "volatility": "N/A"})
        self.ax.set_title(f"{self.config.symbols[0]} Price Chart - {market_regime_text['trend'].capitalize()} Trend, "
                          f"{market_regime_text['volatility'].capitalize()} Volatility", fontsize=12)
        self.ax.legend(loc='upper left', fontsize=9)
        self.ax.grid(True, alpha=0.3)
        self.ax.tick_params(axis='x', rotation=45) # Rotate x-axis labels for readability

        # Plot RSI on ax2
        self.ax2.plot(data.index, data['rsi'], color='purple', linewidth=1.5, label='RSI')
        self.ax2.axhline(y=70, linestyle='--', color='red', alpha=0.5, label='Overbought (70)')
        self.ax2.axhline(y=30, linestyle='--', color='green', alpha=0.5, label='Oversold (30)')
        self.ax2.axhline(y=50, linestyle='--', color='gray', alpha=0.3)
        self.ax2.set_ylim(0, 100)
        self.ax2.set_title('RSI(14)', fontsize=10)
        self.ax2.legend(loc='upper left', fontsize=8)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.tick_params(axis='x', rotation=45)

        # Plot MACD on ax3
        self.ax3.plot(data.index, data['macd'], color='blue', linewidth=1, label='MACD Line')
        self.ax3.plot(data.index, data['macd_signal'], color='red', linewidth=1, label='Signal Line')
        # Use a bar chart for MACD Histogram
        self.ax3.bar(data.index, data['macd_hist'], color=['green' if x > 0 else 'red' for x in data['macd_hist']],
                     alpha=0.5, label='Histogram')
        self.ax3.axhline(y=0, linestyle='-', color='gray', alpha=0.3)
        self.ax3.set_title('MACD (12, 26, 9)', fontsize=10)
        self.ax3.legend(loc='upper left', fontsize=8)
        self.ax3.grid(True, alpha=0.3)
        self.ax3.tick_params(axis='x', rotation=45)

        # Add bot statistics as annotation
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

        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to make room for suptitle
        plt.draw() # Redraw the canvas
        plt.pause(0.01) # Short pause to allow plot to update


    def run(self) -> None:
        """
        Main execution loop of the trading bot.
        Initializes MT5, continuously fetches data, detects arcs,
        manages positions, finds and places signals, and updates visualization.
        """
        # 1. Initialize MT5 connection
        if not self.initialize_mt5():
            self.log_error("Failed to initialize MT5. Exiting bot.")
            return

        self.running = True
        self.log_info("Starting multi-symbol trading loop...")

        try:
            while self.running:
                start_time = datetime.now()
                # Ensure symbols list is valid
                symbols_to_process = self.config.symbols if hasattr(self.config, 'symbols') and self.config.symbols else []
                if not symbols_to_process:
                    self.log_error("No symbols configured to trade. Please check Config.symbols.")
                    break # Exit loop if no symbols

                best_signal = None # To store the highest conviction signal across all symbols

                # 2. Check and update status of all open positions across all symbols
                # This needs to be done once per loop iteration to get a global view
                self.check_open_positions()

                for symbol in symbols_to_process:
                    self.log_info(f"Processing symbol: {symbol}")

                    # 3. Get fresh market data for the current symbol
                    data = self.get_market_data(symbol=symbol)
                    if data.empty:
                        self.log_warning(f"[{symbol}] Skipping processing: No valid market data.")
                        continue

                    # 4. Analyze market regime (trend & volatility) for the current symbol
                    self.analyze_market_regime(symbol, data)

                    # 5. Detect arcs (elliptical patterns) for the current symbol
                    arcs = self.detect_arcs(data)
                    # Store current arcs for visualization (only for the first symbol)
                    if symbol == symbols_to_process[0]:
                        self.current_arcs = arcs

                    # 6. Manage open positions for this specific symbol (trailing stops etc.)
                    self.manage_open_positions(data)

                    # 7. Generate trade signals for the current symbol
                    signals = self.find_signals(arcs, data)

                    # 8. Select the top signal across all symbols based on conviction
                    if signals:
                        # Signals are already sorted by conviction in find_signals
                        current_symbol_top_signal = signals[0]
                        if current_symbol_top_signal['conviction'] >= 0.6: # Only consider high conviction signals
                            # If no best_signal yet, or current symbol's top signal is better
                            if (not best_signal) or \
                               (current_symbol_top_signal['conviction'] > best_signal['conviction']):
                                best_signal = current_symbol_top_signal
                                # The symbol is already part of the signal dictionary, no need to add it again
                                # best_signal['symbol'] = symbol # This is already there from find_signals

                    # 9. Visualize only the first symbol to save resources and avoid multiple plots
                    # if symbol == symbols_to_process[0]:
                        # self.visualize(data)

                # 10. Place the best signal found across all symbols (if any)
                if best_signal:
                    self.log_info(f"Attempting to place best signal: {best_signal['type'].upper()} for {best_signal['symbol']} "
                                  f"with conviction {best_signal['conviction']:.2f}")
                    self.place_order(best_signal)
                else:
                    self.log_info("No strong signal found across all symbols in this iteration.")

                # 11. Sleep until the next iteration to control update frequency
                elapsed_time = (datetime.now() - start_time).total_seconds()
                sleep_duration = max(0, self.config.visual_update_seconds - elapsed_time)
                time.sleep(sleep_duration)

        except KeyboardInterrupt:
            self.log_info("Bot stopped by user (KeyboardInterrupt).")
        except Exception as e:
            self.log_error(f"Critical error in main loop: {str(e)}")
            import traceback
            self.log_error(traceback.format_exc()) # Print full traceback for debugging
        finally:
            self.running = False
            self.log_info("Shutting down MT5 connection and closing plots...")
            mt5.shutdown() # Disconnect from MT5
            plt.close('all') # Close all matplotlib figures

if __name__ == '__main__':
    bot = ArcTradingBot()
    bot.run()

"""
Strategy Tester for the Risk-Averse Optimized Elliptical Arc Trading Bot.

This module provides a framework to backtest the ArcTradingBot's strategy
using historical data, simulating trade execution and position management
without requiring a live MetaTrader 5 connection.
"""

import MetaTrader5 as mt5 # Still imported for symbol info, but not for live trading operations
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

# Import the Config class from the main bot file
# Assuming Config class is available in the same directory or path
# If ArcTradingBot and Config are in the same file, you might need to adjust this.
# For this example, we'll assume Config is directly accessible or copied.
# In a real setup, you'd typically import it:
# from your_main_bot_file import Config, ArcTradingBot

# For the purpose of this standalone tester, let's copy the Config class
# and the core logic functions that don't depend on live MT5.

class Config:
    """Enhanced Configuration Parameters for the trading bot."""
    # Basic settings
    symbols = ["EURUSD.m", "GBPUSD.m", "USDJPY.m", "XAUUSD.m"]
    timeframe = mt5.TIMEFRAME_H1 # Still used for data aggregation, but not live MT5 copy
    arc_lookback = 200

    # Risk management
    max_risk_percent = 0.5
    max_portfolio_risk = 2.0
    profit_ratio = 2.0
    trailing_stop = True
    trailing_stop_trigger = 0.5

    # Position sizing
    min_lots = 0.01
    max_lots = 0.5
    position_scaling = True

    # Arc detection & quality
    max_arcs = 10
    min_r2 = 0.75
    min_curvature = 0.08
    quality_threshold = 0.65

    # Advanced features
    anti_correlation_min = -0.6
    max_open_trades = 3
    max_open_sells = 1

    # Performance optimization
    momentum_lookback = 20
    volatility_lookback = 14

    # System settings
    visual_update_seconds = 15 # Not directly used in backtester, but kept for consistency
    log_level = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
    backtest_mode = True # Set to True for backtesting

class StrategyTester:
    def __init__(self, initial_balance: float = 10000.0, commission_per_lot: float = 7.0, slippage_pips: int = 1):
        """
        Initializes the Strategy Tester with a simulated trading environment.

        Args:
            initial_balance (float): Starting account balance for the backtest.
            commission_per_lot (float): Simulated commission cost per standard lot.
            slippage_pips (int): Simulated slippage in pips for market orders.
        """
        self.config = Config()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance # Equity will fluctuate with open positions
        self.commission_per_lot = commission_per_lot
        self.slippage_pips = slippage_pips

        self.simulated_open_positions = {} # Tracks simulated open positions
        self.simulated_trade_history = [] # Stores simulated closed trades
        self.simulated_account_history = [] # Tracks balance/equity over time

        self.last_signal_time = {} # Prevents rapid re-entry, same as bot
        self.volatility_history = {symbol: deque(maxlen=100) for symbol in self.config.symbols} # Per-symbol volatility history
        self.risk_adjustment = 1.0 # Dynamic risk adjustment, same as bot
        self.market_regimes = {} # Stores market regime analysis per symbol

        self.current_bar_index = 0 # To track progress through historical data
        self.current_data = None # Stores the current bar's data for processing

        # Logger setup - using the same logging methods as the bot
        self.log_method = getattr(self, f"log_{self.config.log_level.lower()}")

        self.log_info(f"Strategy Tester initialized with balance: {self.initial_balance}")

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

    def load_historical_data(self, file_path: str, symbol: str) -> pd.DataFrame:
        """
        Loads historical data from a CSV file.
        The CSV should have columns: 'time', 'open', 'high', 'low', 'close', 'volume'.
        'time' column should be parseable as datetime.
        """
        if not os.path.exists(file_path):
            self.log_warning(f"Historical data file not found at {file_path}. Generating synthetic data.")
            return self._generate_synthetic_data(symbol, self.config.arc_lookback * 5) # Generate more data for backtesting

        try:
            df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
            # Ensure required columns are present
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                self.log_error(f"CSV file {file_path} is missing required columns: {required_cols}. Generating synthetic data.")
                return self._generate_synthetic_data(symbol, self.config.arc_lookback * 5)

            # Add basic indicators needed for the strategy
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

            # Fill initial NaN values from indicators, if any
            df = df.dropna()
            self.log_info(f"Loaded {len(df)} bars for {symbol} from {file_path}.")
            return df
        except Exception as e:
            self.log_error(f"Error loading historical data from {file_path}: {e}. Generating synthetic data.")
            return self._generate_synthetic_data(symbol, self.config.arc_lookback * 5)

    def _generate_synthetic_data(self, symbol: str, count: int) -> pd.DataFrame:
        """Generates synthetic market data for testing purposes."""
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

        # Add indicators to synthetic data
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

    # --- Core ArcTradingBot logic adapted for simulation ---

    def fit_arc(self, data: pd.DataFrame, start: int, end: int) -> dict | None:
        """
        (Copied from ArcTradingBot)
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
        (Copied from ArcTradingBot)
        Detects high-quality elliptical arcs within the price data.
        """
        arcs = []
        for size in [30, 40, 50, 60, 80, 100]:
            step = max(5, size // 5)
            if len(data) < size:
                continue
            for i in range(0, len(data) - size + 1, step):
                arc = self.fit_arc(data, i, i + size)
                if arc and arc['r2'] > self.config.min_r2 and arc['curvature'] > self.config.min_curvature:
                    if arc['volatility'] > 0.00001:
                        arcs.append(arc)

        if not arcs:
            self.log_info("No arcs detected meeting initial quality criteria.")
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

        return sorted(best_arcs, key=lambda a: a['quality'], reverse=True)

    def analyze_market_regime(self, symbol: str, data: pd.DataFrame) -> dict:
        """
        (Copied from ArcTradingBot)
        Analyzes current market conditions (trend and volatility) for a given symbol.
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

    def validate_with_indicators(self, data: pd.DataFrame, signal_type: str) -> float:
        """
        (Copied from ArcTradingBot)
        Validates a potential trade signal using a set of technical indicators.
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

        symbol = data.index.name if data.index.name else self.config.symbols[0] # Use symbol from data if available
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
        (Copied from ArcTradingBot)
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
        (Adapted from ArcTradingBot)
        Calculates the current total risk exposure of all simulated open positions
        as a percentage of the simulated account balance.
        """
        if not self.simulated_open_positions:
            return 0.0

        total_risk_percent = 0.0
        for ticket, pos in self.simulated_open_positions.items():
            if pos['sl'] != 0:
                # For simulation, we need a way to get symbol info.
                # Since we don't have live MT5, we'll use a simplified pip value.
                # This is a simplification; a real backtester would need accurate pip values.
                # Assuming 1 pip = 0.0001 for most pairs, 0.01 for JPY pairs.
                if "JPY" in pos['symbol']:
                    pip_value_per_lot = 1000 # 1 pip = 0.01, 1 lot = 100,000 units. 100,000 * 0.01 = 1000
                else:
                    pip_value_per_lot = 10 # 1 pip = 0.0001, 1 lot = 100,000 units. 100,000 * 0.0001 = 10

                price_diff = abs(pos['price'] - pos['sl'])
                # Convert price difference to pips
                if "JPY" in pos['symbol']:
                    sl_pips = price_diff / 0.01
                else:
                    sl_pips = price_diff / 0.0001

                risk_amount_in_currency = sl_pips * pip_value_per_lot * pos['volume']

                risk_percent = (risk_amount_in_currency / self.balance) * 100
                total_risk_percent += risk_percent

        return total_risk_percent

    def calculate_position_size(self, sl_pips: float, conviction: float = 1.0, symbol_name: str = None) -> float:
        """
        (Adapted from ArcTradingBot)
        Calculates the appropriate position size (in lots) based on risk management
        for the simulated environment.
        """
        if self.balance <= 0:
            self.log_warning("Account balance is zero or negative. Cannot calculate position size.")
            return self.config.min_lots

        if sl_pips <= 0:
            self.log_warning("Stop loss pips must be positive for position sizing. Using min_lots.")
            return self.config.min_lots

        # Simplified pip value for simulation
        if symbol_name and "JPY" in symbol_name:
            pip_value_per_lot = 1000 # Value of 1 pip for 1 lot (e.g., 100,000 units * 0.01 per pip)
        else:
            pip_value_per_lot = 10 # Value of 1 pip for 1 lot (e.g., 100,000 units * 0.0001 per pip)

        # Ensure pip_value_per_lot is not zero
        if pip_value_per_lot == 0:
            self.log_error(f"Calculated pip_value_per_lot for {symbol_name} is zero. Cannot calculate position size.")
            return self.config.min_lots

        adjusted_risk_percent = self.config.max_risk_percent * self.risk_adjustment * conviction
        risk_amount_in_currency = self.balance * (adjusted_risk_percent / 100)

        position_size_lots = risk_amount_in_currency / (sl_pips * pip_value_per_lot)

        lots = max(self.config.min_lots, min(round(position_size_lots, 2), self.config.max_lots))

        self.log_info(f"Simulated position size calculation for {symbol_name}: "
                      f"Risk={adjusted_risk_percent:.2f}%, SL={sl_pips:.1f} pips, "
                      f"Pip Value/Lot={pip_value_per_lot:.2f}, Size={lots:.2f} lots")
        return lots

    def update_risk_adjustment(self) -> None:
        """
        (Copied from ArcTradingBot)
        Dynamically adjusts the bot's risk exposure based on recent trade performance.
        """
        if not self.simulated_trade_history:
            self.risk_adjustment = 1.0
            self.log_info("No simulated trade history to update risk adjustment.")
            return

        win_count = sum(1 for trade in self.simulated_trade_history if trade['profit'] > 0)
        total_trades = len(self.simulated_trade_history)
        loss_count = total_trades - win_count

        win_rate = win_count / total_trades if total_trades > 0 else 0

        loss_sum = sum(abs(t['profit']) for t in self.simulated_trade_history if t['profit'] < 0)
        profit_sum = sum(t['profit'] for t in self.simulated_trade_history if t['profit'] > 0)

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

        self.log_info(f"Risk adjustment updated: {self.risk_adjustment:.2f} "
                      f"(Win Rate: {win_rate:.1%}, Profit Factor: {profit_factor:.2f})")

    def _get_simulated_symbol_info(self, symbol_name: str):
        """
        Provides simulated symbol information for backtesting.
        In a real backtester, this would be more detailed.
        """
        # A very basic simulation for common forex pairs
        if "JPY" in symbol_name:
            return {'point': 0.01, 'trade_tick_value': 0.001, 'trade_stops_level': 20} # Example values
        else:
            return {'point': 0.0001, 'trade_tick_value': 0.00001, 'trade_stops_level': 20} # Example values

    def simulate_order_execution(self, signal: dict, current_bar: pd.Series) -> None:
        """
        Simulates placing a trade order based on a signal.
        Updates simulated account and open positions.
        """
        order_type = signal['type']
        symbol_name = signal['symbol']
        entry_price = signal['optimal_entry']['immediate']

        # Simulate slippage
        if order_type == 'buy':
            simulated_entry_price = entry_price + (self.slippage_pips * 0.00001) # Assuming 5-digit broker
        else:
            simulated_entry_price = entry_price - (self.slippage_pips * 0.00001)

        # Get simulated symbol info for SL/TP calculation
        sim_symbol_info = self._get_simulated_symbol_info(symbol_name)
        
        # Recalculate SL/TP based on current bar's data and simulated entry
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
            self.log_warning(f"[{symbol_name}] Simulated lots ({lots:.2f}) too small. Skipping order.")
            return

        # Simulate commission cost
        commission_cost = (lots / 1.0) * self.commission_per_lot # Assuming commission is per standard lot (1.0)
        self.balance -= commission_cost
        self.equity -= commission_cost

        # Generate a unique ticket for the simulated trade
        ticket = len(self.simulated_open_positions) + 1

        self.simulated_open_positions[ticket] = {
            'ticket': ticket,
            'symbol': symbol_name,
            'type': order_type,
            'price': simulated_entry_price,
            'volume': lots,
            'sl': sl_level,
            'tp': tp_level,
            'profit': 0.0, # Initial profit
            'open_time': current_bar.name # Timestamp of the bar when opened
        }
        self.log_info(f"Simulated {order_type.upper()} order placed for {symbol_name} @ {simulated_entry_price:.5f} "
                      f"Lots: {lots:.2f}, SL: {sl_level:.5f}, TP: {tp_level:.5f}. Commission: {commission_cost:.2f}")

    def simulate_position_management(self, current_bar: pd.Series) -> None:
        """
        Simulates managing open positions (e.g., trailing stops, SL/TP hits).
        This is called for each bar in the backtest.
        """
        closed_tickets_in_this_bar = []
        current_price = current_bar['close']
        high_price = current_bar['high']
        low_price = current_bar['low']
        bar_time = current_bar.name
        symbol_name = current_bar.name # Assuming index name is symbol for dataframes in main loop

        for ticket, pos in list(self.simulated_open_positions.items()): # Iterate over a copy
            if pos['symbol'] != symbol_name:
                continue # Only manage positions for the current symbol's bar

            # Calculate current floating profit/loss
            if pos['type'] == 'buy':
                current_profit = (current_price - pos['price']) * pos['volume'] * 100000 # Simplified contract size
            else: # sell
                current_profit = (pos['price'] - current_price) * pos['volume'] * 100000

            # Check for SL/TP hits
            sl_hit = False
            tp_hit = False

            if pos['type'] == 'buy':
                # Check if low of current bar hit SL
                if pos['sl'] != 0 and low_price <= pos['sl']:
                    sl_hit = True
                    close_price = pos['sl']
                # Check if high of current bar hit TP
                elif pos['tp'] != 0 and high_price >= pos['tp']:
                    tp_hit = True
                    close_price = pos['tp']
            else: # sell
                # Check if high of current bar hit SL
                if pos['sl'] != 0 and high_price >= pos['sl']:
                    sl_hit = True
                    close_price = pos['sl']
                # Check if low of current bar hit TP
                elif pos['tp'] != 0 and low_price <= pos['tp']:
                    tp_hit = True
                    close_price = pos['tp']

            if sl_hit or tp_hit:
                # Calculate final profit/loss for the closed trade
                if pos['type'] == 'buy':
                    final_profit = (close_price - pos['price']) * pos['volume'] * 100000
                else:
                    final_profit = (pos['price'] - close_price) * pos['volume'] * 100000

                self.balance += final_profit
                self.equity = self.balance # Equity becomes balance upon closure

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

            # Trailing Stop Logic (only if not already closed)
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
                                pos['sl'] = new_sl # Update simulated SL
                                self.log_info(f"Simulated position {ticket} trailing SL to {new_sl:.5f}")
                        else:
                            new_sl = current_price + (trailing_pips_from_current_price * pip_size)
                            if new_sl < pos['sl'] and new_sl < pos['price']:
                                pos['sl'] = new_sl # Update simulated SL
                                self.log_info(f"Simulated position {ticket} trailing SL to {new_sl:.5f}")

        # Remove closed positions
        for ticket in closed_tickets_in_this_bar:
            del self.simulated_open_positions[ticket]

        # Update equity based on current floating P/L of remaining open positions
        floating_pl = 0.0
        for ticket, pos in self.simulated_open_positions.items():
            if pos['symbol'] != symbol_name: # Only consider positions for the current symbol for this bar's equity update
                continue
            if pos['type'] == 'buy':
                floating_pl += (current_price - pos['price']) * pos['volume'] * 100000
            else:
                floating_pl += (pos['price'] - current_price) * pos['volume'] * 100000
        self.equity = self.balance + floating_pl

    def find_signals(self, arcs: list[dict], data_slice: pd.DataFrame, symbol: str) -> list[dict]:
        """
        (Adapted from ArcTradingBot)
        Generates potential trade signals based on detected arcs and market conditions.
        Adjusted to use simulated account state.
        """
        signals = []
        current_price = data_slice['close'].iloc[-1]

        market_regime = self.analyze_market_regime(symbol, data_slice)

        if market_regime["volatility"] == "high" and self.simulated_open_positions:
            self.log_debug(f"[{symbol}] High volatility detected and positions open - not generating new signals.")
            return []

        current_risk = self.calculate_portfolio_risk()
        if current_risk >= self.config.max_portfolio_risk:
            self.log_debug(f"[{symbol}] Maximum portfolio risk reached ({current_risk:.2f}%) - not generating new signals.")
            return []

        if len(self.simulated_open_positions) >= self.config.max_open_trades:
            self.log_debug(f"[{symbol}] Maximum open trades ({self.config.max_open_trades}) reached - not generating new signals.")
            return []

        # Count simulated open sells
        simulated_open_sells = sum(1 for pos in self.simulated_open_positions.values() if pos['type'] == 'sell')

        for arc in arcs:
            if arc['quality'] < self.config.quality_threshold:
                continue

            a, b, c, d, e = arc['params']
            def deriv_at_x(x_val):
                return 2 * a * x_val + b + d * e * math.cos(e * x_val)

            current_deriv = deriv_at_x(1)
            prev_deriv = deriv_at_x(0.9)
            angle_diff = math.degrees(math.atan(current_deriv) - math.atan(prev_deriv))

            signal_type = None

            if (angle_diff > 15 and arc['end_direction'] > 0 and
                (market_regime["trend"] != "bearish" or arc['quality'] > 0.8)):
                if 'buy' in self.last_signal_time and \
                   (self.current_data.name - self.last_signal_time['buy']).total_seconds() < 3600: # Use current bar time
                    continue
                signal_type = 'buy'

            elif (angle_diff < -15 and arc['end_direction'] < 0 and
                  (market_regime["trend"] != "bullish" or arc['quality'] > 0.8) and
                  simulated_open_sells < self.config.max_open_sells): # Use simulated count
                if 'sell' in self.last_signal_time and \
                   (self.current_data.name - self.last_signal_time['sell']).total_seconds() < 3600: # Use current bar time
                    continue
                signal_type = 'sell'

            if signal_type:
                conviction = min(1.0, arc['quality'] * (1 + abs(angle_diff) / 90))
                indicator_confirmation = self.validate_with_indicators(data_slice, signal_type)

                if indicator_confirmation >= 0.6:
                    signals.append({
                        'type': signal_type,
                        'price': current_price,
                        'optimal_entry': self.calculate_optimal_entry(data_slice, signal_type),
                        'volatility': arc['volatility'],
                        'quality': arc['quality'],
                        'conviction': conviction * indicator_confirmation,
                        'arc_id': id(arc),
                        'symbol': symbol # Add symbol to the signal
                    })
                    self.last_signal_time[signal_type] = self.current_data.name # Update last signal time with current bar's timestamp

        if signals:
            signals = sorted(signals, key=lambda s: s['conviction'], reverse=True)
            top_signal = signals[0]
            self.log_info(f"[{symbol}] Generated {len(signals)} signals. Top: {top_signal['type'].upper()} with {top_signal['conviction']:.2f} conviction.")
        else:
            self.log_debug(f"[{symbol}] No valid trade signals generated.")

        return signals

    def run_backtest(self, historical_data_map: dict[str, pd.DataFrame]) -> None:
        """
        Runs the backtest simulation over the provided historical data.

        Args:
            historical_data_map (dict[str, pd.DataFrame]): A dictionary where keys are
                symbol names and values are DataFrames of historical data for that symbol.
        """
        self.log_info("Starting backtest simulation...")

        # Ensure all dataframes have enough history for initial lookback
        min_data_length = self.config.arc_lookback * 2 # Minimum data needed for indicators and arcs
        for symbol, df in historical_data_map.items():
            if len(df) < min_data_length:
                self.log_error(f"Not enough historical data for {symbol}. Need at least {min_data_length} bars.")
                return

        # Determine the maximum number of bars to iterate through
        # This assumes all symbols have roughly the same length or we backtest for the shortest period common to all.
        # For simplicity, let's take the shortest length for now.
        max_bars = min(len(df) for df in historical_data_map.values())
        self.log_info(f"Backtesting over {max_bars} bars for {len(historical_data_map)} symbols.")

        # Iterate bar by bar
        for i in range(min_data_length, max_bars): # Start after enough data for initial calculations
            self.current_bar_index = i
            # Store current account state
            self.simulated_account_history.append({
                'time': historical_data_map[list(historical_data_map.keys())[0]].index[i], # Use time from first symbol
                'balance': self.balance,
                'equity': self.equity
            })

            best_signal_overall = None

            for symbol in self.config.symbols:
                if symbol not in historical_data_map:
                    self.log_warning(f"Skipping {symbol}: No historical data provided.")
                    continue

                full_data = historical_data_map[symbol]
                # Get a slice of data up to the current bar for calculations
                data_slice = full_data.iloc[i - self.config.arc_lookback : i + 1]

                if data_slice.empty or len(data_slice) < self.config.arc_lookback:
                    self.log_debug(f"[{symbol}] Not enough data in slice for current bar. Skipping signal generation.")
                    continue

                self.current_data = data_slice.iloc[-1] # Set current_data to the latest bar's data for consistency

                # Update volatility history for the current symbol
                if not data_slice['atr'].empty and not data_slice['close'].empty:
                    current_vol = data_slice['atr'].iloc[-1] / data_slice['close'].iloc[-1] * 100
                    self.volatility_history[symbol].append(current_vol)

                # Analyze market regime
                self.analyze_market_regime(symbol, data_slice)

                # Detect arcs
                arcs = self.detect_arcs(data_slice)

                # Find signals
                signals = self.find_signals(arcs, data_slice, symbol)

                # Select the best signal across all symbols
                if signals:
                    current_symbol_top_signal = signals[0]
                    if current_symbol_top_signal['conviction'] > 0.7:
                        if (not best_signal_overall) or \
                           (current_symbol_top_signal['conviction'] > best_signal_overall['conviction']):
                            best_signal_overall = current_symbol_top_signal

            # Simulate position management (SL/TP hits, trailing stops) for all open positions
            # This needs to be done for each symbol's current bar to check for hits.
            for symbol in self.config.symbols:
                if symbol in historical_data_map:
                    self.simulate_position_management(historical_data_map[symbol].iloc[i])


            # Place the best signal if found for this bar
            if best_signal_overall:
                self.simulate_order_execution(best_signal_overall, self.current_data)

            # Update risk adjustment after all potential trades for the bar are processed
            self.update_risk_adjustment()

        self.log_info("Backtest simulation finished.")
        self.generate_report()

    def generate_report(self) -> None:
        """Generates and prints a performance report for the backtest."""
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

        # Calculate Max Drawdown
        equity_curve = pd.DataFrame(self.simulated_account_history)
        if not equity_curve.empty:
            equity_curve['peak'] = equity_curve['equity'].cummax()
            equity_curve['drawdown'] = equity_curve['peak'] - equity_curve['equity']
            max_drawdown = equity_curve['drawdown'].max()
            max_drawdown_percent = (max_drawdown / equity_curve['peak'].max()) * 100 if equity_curve['peak'].max() > 0 else 0
            self.log_info(f"Max Drawdown:    ${max_drawdown:.2f} ({max_drawdown_percent:.2f}%)")

            # Plot Equity Curve
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

# Example Usage:
if __name__ == '__main__':
    tester = StrategyTester(initial_balance=10000)

    # Prepare historical data for multiple symbols
    # You would replace these with paths to your actual historical data CSVs
    # For demonstration, we'll generate synthetic data for all symbols
    historical_data_for_backtest = {}
    for symbol in tester.config.symbols:
        # Create dummy CSV files for demonstration if they don't exist
        csv_file_path = f"{symbol}_H1_data.csv"
        if not os.path.exists(csv_file_path):
            dummy_df = tester._generate_synthetic_data(symbol, 1000) # Generate 1000 bars of synthetic data
            dummy_df.to_csv(csv_file_path)
            tester.log_info(f"Created dummy CSV: {csv_file_path}")

        historical_data_for_backtest[symbol] = tester.load_historical_data(csv_file_path, symbol)

    # Run the backtest
    tester.run_backtest(historical_data_for_backtest)

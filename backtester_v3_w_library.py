"""
Arc Trading Bot Strategy implemented with Backtrader for Efficient Backtesting.

This script demonstrates how to integrate the core logic of the
Risk-Averse Optimized Elliptical Arc Trading Bot into the backtrader framework.
It allows for rapid backtesting of the strategy on historical data.
"""

import backtrader as bt
import numpy as np
import pandas as pd
import math
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN
from collections import deque
import os
from datetime import datetime, timedelta

# Import TA-Lib based indicators (Backtrader has its own, but we'll keep
# the ta.trend and ta.momentum for consistency with original logic where needed)
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

class ArcStrategy(bt.Strategy):
    """
    Backtrader Strategy implementation for the Arc Trading Bot.
    This class contains the core trading logic, adapted to backtrader's event-driven model.
    """

    params = (
        ('arc_lookback', 200),
        ('max_risk_percent', 0.5),
        ('profit_ratio', 2.0),
        ('trailing_stop', True),
        ('trailing_stop_trigger', 0.5),
        ('min_r2', 0.75),
        ('min_curvature', 0.08),
        ('quality_threshold', 0.65),
        ('arc_detection_sizes', [60, 80]), # Optimized for speed
        ('arc_detection_step_multiplier', 0.4), # Optimized for speed
        ('max_open_trades', 3),
        ('max_open_sells', 1),
        ('momentum_lookback', 20),
        ('volatility_lookback', 14),
        ('commission_per_lot', 7.0),
        ('slippage_pips', 1),
        ('min_lots', 0.01),
        ('max_lots', 0.5),
    )

    def __init__(self):
        """
        Initializes the strategy.
        Defines indicators, tracks open positions, and sets up logging.
        """
        self.log_level = "ERROR" # Set to "INFO" or "DEBUG" for more verbose output

        # Keep track of orders and positions
        self.order = None
        self.position_tracker = {} # {order_ref: {'symbol':, 'type':, 'volume':, 'price':, 'sl':, 'tp':, 'open_time':}}
        self.open_sells_count = 0

        # Backtrader indicators (automatically managed by the framework)
        # Corrected: Reverted ATR initialization to standard backtrader usage
        self.ema20 = bt.indicators.EMA(self.data.lines.close, period=20)
        self.ema50 = bt.indicators.EMA(self.data.lines.close, period=50)
        self.ema200 = bt.indicators.EMA(self.data.lines.close, period=200)
        self.rsi = bt.indicators.RSI(self.data.lines.close, period=14)
        self.atr = bt.indicators.ATR(self.data, period=14) # <--- MODIFIED
        self.macd = bt.indicators.MACD(self.data.lines.close, period_fast=12, period_slow=26, period_signal=9)
        self.macd_hist = self.macd.macd - self.macd.signal

        # Custom variables for arc logic and risk management
        self.last_signal_time = {}
        self.volatility_history = deque(maxlen=100) # For current symbol
        self.risk_adjustment = 1.0
        self.market_regime = {"trend": "neutral", "volatility": "medium"}

        # Store strategy performance for reporting
        self.trade_history = deque(maxlen=100) # For risk adjustment

    def _log_message(self, level: str, message: str) -> None:
        """Internal helper for logging messages."""
        log_levels_map = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        if log_levels_map.get(level, 0) >= log_levels_map.get(self.log_level, 0):
            print(f"[{level.upper()}] {self.data.datetime.datetime(0).strftime('%Y-%m-%d %H:%M:%S')}: {message}")

    def notify_order(self, order):
        """Receives notification of an order."""
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted - Nothing to do
            return

        if order.status in [order.Completed]:
            if order.isbuy:
                self._log_message("INFO", f"BUY EXECUTED, Price: {order.executed.price:.5f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}")
                self.position_tracker[order.ref] = {
                    'symbol': self.data._name,
                    'type': 'buy',
                    'volume': order.executed.size,
                    'price': order.executed.price,
                    'sl': order.created.sl, # Store SL/TP from the order creation
                    'tp': order.created.tp,
                    'open_time': self.data.datetime.datetime(0)
                }
            elif order.issell:
                self._log_message("INFO", f"SELL EXECUTED, Price: {order.executed.price:.5f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}")
                self.position_tracker[order.ref] = {
                    'symbol': self.data._name,
                    'type': 'sell',
                    'volume': order.executed.size,
                    'price': order.executed.price,
                    'sl': order.created.sl,
                    'tp': order.created.tp,
                    'open_time': self.data.datetime.datetime(0)
                }
            self.order = None # No pending order
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self._log_message("ERROR", f"Order Canceled/Margin/Rejected: {order.getstatusname(order.status)}")
            self.order = None # No pending order

    def notify_trade(self, trade):
        """Receives notification of a trade (position closed)."""
        if not trade.isclosed:
            return

        self._log_message("INFO", f"TRADE PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}")
        # Update trade history for risk adjustment
        self.trade_history.append({'profit': trade.pnlcomm})
        self.update_risk_adjustment()

        # Remove from position tracker
        # Note: Backtrader doesn't directly link order.ref to trade.ref in this way for easy cleanup.
        # A more robust solution for multi-data/multi-position tracking would involve a custom
        # position management system within the strategy. For simplicity, we'll clear all
        # positions for the current symbol when a trade is closed, as backtrader manages them.
        # In a real multi-symbol scenario, you'd iterate self.broker.getpositions()
        # and update self.position_tracker more precisely.
        self.position_tracker = {} # Clear for current symbol (simplification)
        self.open_sells_count = 0 # Reset for current symbol

    def next(self):
        """
        This method is called for each bar of data for each symbol.
        It contains the main strategy logic.
        """
        # Ensure enough data for indicators and arc detection
        if len(self.data) < self.p.arc_lookback:
            return

        # Prepare data slice for arc detection and other calculations
        data_slice = pd.DataFrame({
            'open': self.data.open.get(size=self.p.arc_lookback),
            'high': self.data.high.get(size=self.p.arc_lookback),
            'low': self.data.low.get(size=self.p.arc_lookback),
            'close': self.data.close.get(size=self.p.arc_lookback),
            'volume': self.data.volume.get(size=self.p.arc_lookback)
        }, index=pd.to_datetime([self.data.datetime.datetime(-i) for i in range(self.p.arc_lookback -1, -1, -1)]))
        data_slice.index.name = self.data._name # Set symbol name for logging

        # Calculate indicators for the current slice (using ta-lib for consistency with original logic)
        # Backtrader's indicators are already calculated, but we need these for the validation logic
        # that expects a DataFrame with these columns.
        data_slice['ema20'] = EMAIndicator(close=data_slice['close'], window=20, fillna=True).ema_indicator()
        data_slice['ema50'] = EMAIndicator(close=data_slice['close'], window=50, fillna=True).ema_indicator()
        data_slice['ema200'] = EMAIndicator(close=data_slice['close'], window=200, fillna=True).ema_indicator()
        data_slice['rsi'] = RSIIndicator(close=data_slice['close'], window=14, fillna=True).rsi()
        atr_indicator = AverageTrueRange(high=data_slice['high'], low=data_slice['low'], close=data_slice['close'], window=14, fillna=True)
        data_slice['atr'] = atr_indicator.average_true_range()
        macd_indicator = MACD(close=data_slice['close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
        data_slice['macd'] = macd_indicator.macd()
        data_slice['macd_signal'] = macd_indicator.macd_signal()
        data_slice['macd_hist'] = macd_indicator.macd_diff()
        data_slice = data_slice.dropna() # Drop NaNs introduced by indicators

        if data_slice.empty:
            self._log_message("DEBUG", "Data slice is empty after indicator calculation. Skipping.")
            return

        # Update volatility history for the current symbol
        if not data_slice['atr'].empty and not data_slice['close'].empty:
            current_vol = data_slice['atr'].iloc[-1] / data_slice['close'].iloc[-1] * 100
            self.volatility_history.append(current_vol)

        # Analyze market regime
        self.market_regime = self._analyze_market_regime(data_slice)

        # Detect arcs
        arcs = self._detect_arcs(data_slice)

        # Find signals
        signals = self._find_signals(arcs, data_slice)

        # Check for existing positions for this data feed
        has_open_position = bool(self.broker.getposition(self.data).size)

        # If an order is pending, don't send another one
        if self.order:
            return

        # Place the best signal if found
        if signals and not has_open_position:
            best_signal = signals[0]
            if best_signal['conviction'] > 0.7:
                self._place_order(best_signal, data_slice.iloc[-1]) # Pass current bar for SL/TP calculation
        else:
            self._log_message("DEBUG", "No strong signal or position already open.")

        # Manage open positions (trailing stop)
        self._manage_open_positions(data_slice.iloc[-1])


    def _fit_arc(self, data: pd.DataFrame, start: int, end: int) -> dict | None:
        """
        Fits an elliptical arc to a segment of price data.
        (Copied directly from your original bot logic)
        """
        x = np.arange(end - start)
        y = data['close'].iloc[start:end].values

        if len(x) < 5:
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
            self._log_message("DEBUG", f"Arc fitting RuntimeError: {str(re)}")
            return None
        except Exception as e:
            self._log_message("DEBUG", f"Arc fitting failed for segment [{start}:{end}]: {str(e)}")
            return None

    def _detect_arcs(self, data: pd.DataFrame) -> list[dict]:
        """
        Detects high-quality elliptical arcs within the price data.
        (Copied directly from your original bot logic, using params from Config)
        """
        arcs = []
        for size in self.p.arc_detection_sizes:
            step = max(5, int(size * self.p.arc_detection_step_multiplier))
            if len(data) < size:
                continue
            for i in range(0, len(data) - size + 1, step):
                arc = self._fit_arc(data, i, i + size)
                if arc and arc['r2'] > self.p.min_r2 and arc['curvature'] > self.p.min_curvature:
                    if arc['volatility'] > 0.00001:
                        arcs.append(arc)

        if not arcs:
            self._log_message("DEBUG", "No arcs detected meeting initial quality criteria.")
            return []

        # Corrected: Use a['end_idx'] - a['start_idx'] for midpoint calculation
        arc_midpoints = np.array([[a['start_idx'] + (a['end_idx'] - a['start_idx']) / 2] for a in arcs]) # <--- MODIFIED
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

        self._log_message("DEBUG", f"Detected {len(best_arcs)} high-quality, distinct arcs after clustering.")
        return sorted(best_arcs, key=lambda a: a['quality'], reverse=True)

    def _analyze_market_regime(self, data: pd.DataFrame) -> dict:
        """
        Analyzes current market conditions (trend and volatility).
        (Copied directly from your original bot logic)
        """
        if data.empty:
            self._log_message("WARNING", "Cannot analyze market regime: no data.")
            return {"trend": "neutral", "volatility": "medium"}

        # Use the ta-lib calculated EMAs from the data_slice
        if data['ema20'].iloc[-1] > data['ema50'].iloc[-1]:
            trend = "bullish"
        elif data['ema20'].iloc[-1] < data['ema50'].iloc[-1]:
            trend = "bearish"
        else:
            trend = "ranging"

        current_atr = data['atr'].iloc[-1]
        if self.volatility_history:
            avg_vol = np.mean(list(self.volatility_history))
        else:
            avg_vol = current_atr

        if current_atr > avg_vol * 1.5:
            vol_regime = "high"
        elif current_atr < avg_vol * 0.7:
            vol_regime = "low"
        else:
            vol_regime = "medium"

        self._log_message("DEBUG", f"Market Regime: Trend={trend}, Volatility={vol_regime}")
        return {"trend": trend, "volatility": vol_regime}

    def _validate_with_indicators(self, data: pd.DataFrame, signal_type: str) -> float:
        """
        Validates a potential trade signal using a set of technical indicators.
        (Copied directly from your original bot logic)
        """
        confirmations = 0
        total_indicators = 5

        if data.empty or len(data) < max(self.p.momentum_lookback, self.p.volatility_lookback, 200):
            self._log_message("WARNING", "Not enough data to validate with indicators.")
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

        current_atr = data['atr'].iloc[-1]
        if self.volatility_history:
            avg_vol = np.mean(list(self.volatility_history))
        else:
            avg_vol = current_atr

        if current_atr < avg_vol * 1.5:
            confirmations += 1

        if signal_type == 'buy' and data['close'].iloc[-1] > data['ema200'].iloc[-1]:
            confirmations += 1
        elif signal_type == 'sell' and data['close'].iloc[-1] < data['ema200'].iloc[-1]:
            confirmations += 1

        return confirmations / total_indicators

    def _calculate_optimal_entry(self, data: pd.DataFrame, signal_type: str) -> dict[str, float]:
        """
        Calculates optimal entry levels.
        (Copied directly from your original bot logic)
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

    def _calculate_portfolio_risk(self) -> float:
        """
        Calculates the current total risk exposure as a percentage of the account balance.
        Adapted for backtrader's broker and positions.
        """
        if not self.position_tracker:
            return 0.0

        total_risk_percent = 0.0
        current_balance = self.broker.getcash() + self.broker.getvalue() # Cash + value of open positions

        for order_ref, pos_info in self.position_tracker.items():
            if pos_info['sl'] != 0:
                # Simplified pip value for backtrader simulation
                if "JPY" in pos_info['symbol']:
                    pip_value_per_lot = 1000 # 1 pip = 0.01, 1 lot = 100,000 units
                else:
                    pip_value_per_lot = 10 # 1 pip = 0.0001, 1 lot = 100,000 units

                price_diff = abs(pos_info['price'] - pos_info['sl'])
                if "JPY" in pos_info['symbol']:
                    sl_pips = price_diff / 0.01
                else:
                    sl_pips = price_diff / 0.0001

                risk_amount_in_currency = sl_pips * pip_value_per_lot * pos_info['volume']

                risk_percent = (risk_amount_in_currency / current_balance) * 100
                total_risk_percent += risk_percent

        self._log_message("DEBUG", f"Current portfolio risk: {total_risk_percent:.2f}%")
        return total_risk_percent

    def _calculate_position_size(self, sl_pips: float, conviction: float = 1.0, symbol_name: str = None) -> float:
        """
        Calculates the appropriate position size (in lots) based on risk management.
        Adapted for backtrader's broker.
        """
        current_balance = self.broker.getcash() + self.broker.getvalue()

        if current_balance <= 0:
            self._log_message("WARNING", "Account balance is zero or negative. Cannot calculate position size.")
            return self.p.min_lots

        if sl_pips <= 0:
            self._log_message("WARNING", "Stop loss pips must be positive for position sizing. Using min_lots.")
            return self.p.min_lots

        # Simplified pip value for backtrader simulation
        if symbol_name and "JPY" in symbol_name:
            pip_value_per_lot = 1000
        else:
            pip_value_per_lot = 10

        if pip_value_per_lot == 0:
            self._log_message("ERROR", f"Calculated pip_value_per_lot for {symbol_name} is zero. Using min_lots.")
            return self.p.min_lots

        adjusted_risk_percent = self.p.max_risk_percent * self.risk_adjustment * conviction
        risk_amount_in_currency = current_balance * (adjusted_risk_percent / 100)

        position_size_lots = risk_amount_in_currency / (sl_pips * pip_value_per_lot)

        lots = max(self.p.min_lots, min(round(position_size_lots, 2), self.p.max_lots))

        self._log_message("DEBUG", f"Position size calculation for {symbol_name}: "
                                  f"Risk={adjusted_risk_percent:.2f}%, SL={sl_pips:.1f} pips, "
                                  f"Pip Value/Lot={pip_value_per_lot:.2f}, Size={lots:.2f} lots")
        return lots

    def update_risk_adjustment(self) -> None:
        """
        Dynamically adjusts the bot's risk exposure based on recent trade performance.
        Uses the internal trade_history.
        """
        if not self.trade_history:
            self.risk_adjustment = 1.0
            self._log_message("DEBUG", "No trade history to update risk adjustment.")
            return

        win_count = sum(1 for trade in self.trade_history if trade['profit'] > 0)
        total_trades = len(self.trade_history)
        loss_count = total_trades - win_count

        win_rate = win_count / total_trades if total_trades > 0 else 0

        loss_sum = sum(abs(t['profit']) for t in self.trade_history if t['profit'] < 0)
        profit_sum = sum(t['profit'] for t in self.trade_history if t['profit'] > 0)

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

        self._log_message("DEBUG", f"Risk adjustment updated: {self.risk_adjustment:.2f} "
                                  f"(Win Rate: {win_rate:.1%}, Profit Factor: {profit_factor:.2f})")

    def _find_signals(self, arcs: list[dict], data_slice: pd.DataFrame) -> list[dict]:
        """
        Generates potential trade signals based on detected arcs and market conditions.
        (Copied directly from your original bot logic)
        """
        signals = []
        current_price = data_slice['close'].iloc[-1]
        symbol = data_slice.index.name

        # Determine which set of open positions to check
        current_open_positions = self.position_tracker
        current_open_sells = sum(1 for pos in current_open_positions.values() if pos['type'] == 'sell')

        if self.market_regime["volatility"] == "high" and current_open_positions:
            self._log_message("DEBUG", f"[{symbol}] High volatility detected and positions open - not generating new signals.")
            return []

        current_risk = self._calculate_portfolio_risk()
        if current_risk >= self.p.max_portfolio_risk:
            self._log_message("DEBUG", f"[{symbol}] Maximum portfolio risk reached ({current_risk:.2f}%) - not generating new signals.")
            return []

        if len(current_open_positions) >= self.p.max_open_trades:
            self._log_message("DEBUG", f"[{symbol}] Maximum open trades ({self.p.max_open_trades}) reached - not generating new signals.")
            return []

        for arc in arcs:
            if arc['quality'] < self.p.quality_threshold:
                self._log_message("DEBUG", f"[{symbol}] Arc quality {arc['quality']:.2f} below threshold {self.p.quality_threshold}.")
                continue

            a, b, c, d, e = arc['params']
            def deriv_at_x(x_val):
                return 2 * a * x_val + b + d * e * math.cos(e * x_val)

            current_deriv = deriv_at_x(1)
            prev_deriv = deriv_at_x(0.9)
            angle_diff = math.degrees(math.atan(current_deriv) - math.atan(prev_deriv))

            signal_type = None
            current_time_for_signal = self.data.datetime.datetime(0)

            if (angle_diff > 15 and arc['end_direction'] > 0 and
                (self.market_regime["trend"] != "bearish" or arc['quality'] > 0.8)):
                if 'buy' in self.last_signal_time and \
                   (current_time_for_signal - self.last_signal_time['buy']).total_seconds() < 3600:
                    self._log_message("DEBUG", f"[{symbol}] Buy signal too soon after last one. Skipping.")
                    continue
                signal_type = 'buy'

            elif (angle_diff < -15 and arc['end_direction'] < 0 and
                  (self.market_regime["trend"] != "bullish" or arc['quality'] > 0.8) and
                  current_open_sells < self.p.max_open_sells):
                if 'sell' in self.last_signal_time and \
                   (current_time_for_signal - self.last_signal_time['sell']).total_seconds() < 3600:
                    self._log_message("DEBUG", f"[{symbol}] Sell signal too soon after last one. Skipping.")
                    continue
                signal_type = 'sell'

            if signal_type:
                conviction = min(1.0, arc['quality'] * (1 + abs(angle_diff) / 90))
                indicator_confirmation = self._validate_with_indicators(data_slice, signal_type)

                if indicator_confirmation >= 0.6:
                    signals.append({
                        'type': signal_type,
                        'price': current_price,
                        'optimal_entry': self._calculate_optimal_entry(data_slice, signal_type),
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
            self._log_message("DEBUG", f"[{symbol}] Generated {len(signals)} signals. Top: {top_signal['type'].upper()} with {top_signal['conviction']:.2f} conviction.")
        else:
            self._log_message("DEBUG", f"[{symbol}] No valid trade signals generated.")

        return signals

    def _place_order(self, signal: dict, current_bar: pd.Series) -> None:
        """
        Places a new trade order using backtrader's API.
        """
        order_type = signal['type']
        symbol_name = signal['symbol']
        entry_price = signal['optimal_entry']['immediate']

        # Simplified pip value for backtrader simulation
        if "JPY" in symbol_name:
            pip_divisor = 0.01
        else:
            pip_divisor = 0.0001
        
        atr = current_bar['atr']
        if atr == 0: atr = 0.0001

        if order_type == 'buy':
            sl_level = (current_bar['low'] - 0.5 * atr)
        else:
            sl_level = (current_bar['high'] + 0.5 * atr)

        # Ensure SL is at least 5 pips away (simplified minimum distance)
        sl_pips = abs(entry_price - sl_level) / pip_divisor
        if sl_pips < 5:
            sl_pips = 5
            if order_type == 'buy':
                sl_level = entry_price - (sl_pips * pip_divisor)
            else:
                sl_level = entry_price + (sl_pips * pip_divisor)

        tp_pips = sl_pips * self.p.profit_ratio
        if order_type == 'buy':
            tp_level = entry_price + (tp_pips * pip_divisor)
        else:
            tp_level = entry_price - (tp_pips * pip_divisor)
        
        lots = self._calculate_position_size(sl_pips, signal['conviction'], symbol_name=symbol_name)

        if lots < self.p.min_lots:
            self._log_message("WARNING", f"[{symbol_name}] Calculated lots ({lots:.2f}) too small. Skipping order.")
            return

        if order_type == 'buy':
            self.order = self.buy(size=lots, exectype=bt.Order.Market, price=entry_price,
                                  transmit=True, valid=self.data.datetime.datetime(0) + timedelta(days=365)) # GTC
            # Store SL/TP with the order for later retrieval in notify_order
            self.order.created.sl = sl_level
            self.order.created.tp = tp_level
            self._log_message("INFO", f"BUY Order created: {symbol_name} Lots: {lots:.2f}, SL: {sl_level:.5f}, TP: {tp_level:.5f}")
        else:
            self.order = self.sell(size=lots, exectype=bt.Order.Market, price=entry_price,
                                   transmit=True, valid=self.data.datetime.datetime(0) + timedelta(days=365)) # GTC
            self.order.created.sl = sl_level
            self.order.created.tp = tp_level
            self.open_sells_count += 1 # Increment simulated sell count
            self._log_message("INFO", f"SELL Order created: {symbol_name} Lots: {lots:.2f}, SL: {sl_level:.5f}, TP: {tp_level:.5f}")

    def _manage_open_positions(self, current_bar: pd.Series) -> None:
        """
        Manages open positions, specifically implementing trailing stops.
        Backtrader automatically handles SL/TP hits. We only manage trailing stops here.
        """
        if not self.position_tracker:
            return

        current_price = current_bar['close']
        symbol_name = self.data._name
        pip_size = 0.0001 # Default for most pairs
        if "JPY" in symbol_name:
            pip_size = 0.01

        for order_ref, pos_info in list(self.position_tracker.items()):
            if pos_info['symbol'] != symbol_name:
                continue

            if pos_info['sl'] == 0: # If SL is not set, we can't trail
                continue

            if pos_info['type'] == 'buy':
                profit_pips = (current_price - pos_info['price']) / pip_size
            else: # sell
                profit_pips = (pos_info['price'] - current_price) / pip_size

            if self.p.trailing_stop and profit_pips > 0:
                if pos_info['tp'] != 0:
                    target_pips = abs(pos_info['tp'] - pos_info['price']) / pip_size
                else:
                    # If no TP, use a default target based on SL * profit_ratio
                    target_pips = abs(pos_info['sl'] - pos_info['price']) / pip_size * self.p.profit_ratio
                    if target_pips == 0:
                        continue

                profit_ratio = profit_pips / max(0.01, target_pips)

                if profit_ratio >= self.p.trailing_stop_trigger:
                    trailing_pips_from_current_price = profit_pips * 0.5 # Trail by 50% of current profit

                    if pos_info['type'] == 'buy':
                        new_sl = current_price - (trailing_pips_from_current_price * pip_size)
                        if new_sl > pos_info['sl'] and new_sl > pos_info['price']:
                            # Modify the existing order's SL
                            self.broker.getposition(self.data).setstoploss(new_sl)
                            pos_info['sl'] = new_sl # Update our tracker
                            self._log_message("INFO", f"Trailing SL for BUY position to {new_sl:.5f}")
                    else: # sell
                        new_sl = current_price + (trailing_pips_from_current_price * pip_size)
                        if new_sl < pos_info['sl'] and new_sl < pos_info['price']:
                            # Modify the existing order's SL
                            self.broker.getposition(self.data).setstoploss(new_sl)
                            pos_info['sl'] = new_sl # Update our tracker
                            self._log_message("INFO", f"Trailing SL for SELL position to {new_sl:.5f}")


def run_backtest_with_backtrader(symbols: list[str], start_date: datetime, end_date: datetime):
    """
    Sets up and runs a backtest using the Backtrader framework.
    """
    cerebro = bt.Cerebro()

    # Add the strategy
    cerebro.addstrategy(ArcStrategy)

    # Add data feeds for each symbol
    for symbol in symbols:
        # Generate synthetic data if CSV not found or use existing CSV
        csv_file_path = f"{symbol}_H1_data.csv"
        if not os.path.exists(csv_file_path):
            print(f"Generating synthetic data for {symbol}...")
            # Generate 10000 hourly bars (approx 1.1 years)
            dates = pd.date_range(start=start_date, end=end_date, freq='H')
            count = len(dates)
            np.random.seed(42 + hash(symbol) % (2**32 - 1)) # Different seed per symbol
            prices = 1.1 + np.cumsum(np.random.normal(0, 0.0005, count)) + np.sin(np.arange(count) * 0.1) * 0.01
            df = pd.DataFrame({
                'open': prices,
                'high': prices * (1 + np.abs(np.random.normal(0, 0.0002, count))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.0002, count))),
                'close': prices,
                'volume': np.random.randint(100, 1000, count)
            }, index=dates)
            # Ensure the datetime format written to CSV matches the one expected by backtrader
            df.index = df.index.strftime('%Y-%m-%d %H:%M:%S.%f') # Include fractional seconds
            df.to_csv(csv_file_path)
            print(f"Created dummy CSV: {csv_file_path}")

        data = bt.feeds.GenericCSVData(
            dataname=csv_file_path,
            dtformat='%Y-%m-%d %H:%M:%S.%f', # <--- MODIFIED: Added .%f for fractional seconds
            datetime=0,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5,
            openinterest=-1,
            fromdate=start_date,
            todate=end_date
        )
        cerebro.adddata(data, name=symbol)

    # Set starting cash
    cerebro.broker.setcash(10000.0)

    # Set commission and slippage (simplified)
    cerebro.broker.setcommission(commission=ArcStrategy.params.commission_per_lot / 100000) # Per unit, not per lot
    # Backtrader has slippage models, but for simplicity, we'll rely on the simulated
    # slippage in our _place_order logic if needed, or let Backtrader's default handle it.

    # Add analyzers for performance metrics
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # Run the backtest
    results = cerebro.run()

    # Print results
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # Access and print analyzer results
    strat = results[0] # Get the first strategy instance
    print("\n--- Backtest Report (Backtrader) ---")
    print(f"Sharpe Ratio: {strat.analyzers.sharpe.get_analysis().sharperatio:.2f}")
    print(f"Max Drawdown: {strat.analyzers.drawdown.get_analysis().max.drawdown:.2f}%")
    print(f"SQN: {strat.analyzers.sqn.get_analysis().sqn:.2f}")

    trades_analysis = strat.analyzers.trades.get_analysis()
    if trades_analysis.total.closed > 0:
        print(f"Total Trades: {trades_analysis.total.closed}")
        print(f"Winning Trades: {trades_analysis.won.total} ({trades_analysis.won.pnl.average:.2f} avg profit)")
        print(f"Losing Trades: {trades_analysis.lost.total} ({trades_analysis.lost.pnl.average:.2f} avg loss)")
        print(f"Win Rate: {(trades_analysis.won.total / trades_analysis.total.closed * 100):.2f}%")
        print(f"Gross Profit: {trades_analysis.won.pnl.total:.2f}")
        print(f"Gross Loss: {trades_analysis.lost.pnl.total:.2f}")
        profit_factor = trades_analysis.won.pnl.total / max(0.01, abs(trades_analysis.lost.pnl.total))
        print(f"Profit Factor: {profit_factor:.2f}")
    else:
        print("No trades executed during backtest.")

    # Plotting (optional, uncomment to enable)
    # cerebro.plot(style='candlestick', iplot=False) # iplot=False to prevent interactive plot in some envs


if __name__ == '__main__':
    # Define the date range for your backtest
    # For a longer period, adjust these dates
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 1, 1) # 1 year of hourly data

    # Symbols to backtest
    symbols_to_test = ["EURUSD.m", "GBPUSD.m"] # You can add more symbols from your config

    run_backtest_with_backtrader(symbols_to_test, start_date, end_date)

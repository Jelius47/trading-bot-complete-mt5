"""
Risk-Averse Optimized Elliptical Arc Trading Bot with MT5 Integration
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
# import talib
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


# Enhanced Configuration Parameters
class Config:
    # Basic settings
    symbol = "EURUSD.m"
    timeframe = mt5.TIMEFRAME_H1
    arc_lookback = 200  # Increased lookback for better pattern recognition
    
    # Risk management 
    max_risk_percent = 0.5  # Reduced from 1% to 0.5% of account balance per trade
    max_portfolio_risk = 2.0  # Maximum total risk exposure as percentage of account
    profit_ratio = 2.0  # Conservative profit target (was 3)
    trailing_stop = True  # Enable trailing stops to lock in profits
    trailing_stop_trigger = 0.5  # Trigger trailing stop at 50% of target profit
    
    # Position sizing
    min_lots = 0.01
    max_lots = 0.5  # Cap maximum position size for risk control
    position_scaling = True  # Scale position sizes based on conviction
    
    # Arc detection & quality
    max_arcs = 10  # Track more arcs for better analysis
    min_r2 = 0.75  # Increased from 0.7 for better fit quality
    min_curvature = 0.08  # Increased for more pronounced arc patterns
    quality_threshold = 0.65  # Minimum quality score for trading signals
    
    # Advanced features
    anti_correlation_min = -0.6  # Minimum anti-correlation for risk diversification
    max_open_trades = 3  # Limit total number of concurrent trades 
    max_open_sells = 1  # Limit to only one active SELL position at a time
    
    # Performance optimization
    momentum_lookback = 20  # Periods for momentum confirmation
    volatility_lookback = 14  # Periods for volatility calculation
    
    # System settings
    visual_update_seconds = 15
    log_level = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
    backtest_mode = False  # Set to True for backtesting

class ArcTradingBot:
    def __init__(self):
        self.config = Config()
        self.running = False
        self.current_arcs = []
        self.open_positions = {}  # Track all open positions
        self.open_sells = 0  # Counter for open sell positions
        self.trade_history = deque(maxlen=100)  # Store recent trade outcomes
        self.last_signal_time = {}  # Prevent signal duplication
        self.market_state = "neutral"  # Overall market state assessment
        self.volatility_history = deque(maxlen=50)  # Store historical volatility
        self.risk_adjustment = 1.0  # Dynamically adjust risk based on performance
        
        # Visualization
        self.fig = None
        self.ax = None
        self.ax2 = None
        self.ax3 = None
        
        # Logger setup
        self.log_level = getattr(self, "log_" + self.config.log_level.lower())
        
    def log_debug(self, message):
        if self.config.log_level in ["DEBUG"]:
            print(f"[DEBUG] {datetime.now().strftime('%H:%M:%S')}: {message}")
            
    def log_info(self, message):
        if self.config.log_level in ["DEBUG", "INFO"]:
            print(f"[INFO] {datetime.now().strftime('%H:%M:%S')}: {message}")
            
    def log_warning(self, message):
        if self.config.log_level in ["DEBUG", "INFO", "WARNING"]:
            print(f"[WARNING] {datetime.now().strftime('%H:%M:%S')}: {message}")
            
    def log_error(self, message):
        print(f"[ERROR] {datetime.now().strftime('%H:%M:%S')}: {message}")

    def initialize_mt5(self):
        self.log_info("Initializing MT5...")
        if not mt5.initialize():
            self.log_error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        info = mt5.terminal_info()
        account = mt5.account_info()
        self.log_info("✅ MT5 connected successfully:")
        self.log_info(f"  Path:          {info.path}")
        self.log_info(f"  Trade Allowed: {'Yes' if info.trade_allowed else 'No'}")
        self.log_info(f"  Version:       {mt5.version()[0]}")
        self.log_info(f"  Account:       {account.login} ({account.server})")
        self.log_info(f"  Balance:       {account.balance} {account.currency}")
        
        # Check if symbol is available and initialize
        if not mt5.symbol_select(self.config.symbol, True):
            self.log_error(f"Symbol {self.config.symbol} not available")
            return False
            
        return True

    def get_market_data(self, count=None, additional_indicators=True):
        if count is None:
            count = self.config.arc_lookback * 2
        
        mt5.symbol_select(self.config.symbol, True)
        rates = mt5.copy_rates_from_pos(self.config.symbol, self.config.timeframe, 0, count)
        
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
           

            # Add technical indicators if requested
            if additional_indicators:
                # Exponential Moving Averages
                df['ema20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
                df['ema50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
                df['ema200'] = EMAIndicator(close=df['close'], window=200).ema_indicator()
                
                # RSI
                df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
                
                # ATR
                atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
                df['atr'] = atr.average_true_range()
                
                # MACD
                macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()
                df['macd_hist'] = macd.macd_diff()
                
                # Calculate volatility
                current_volatility = df['atr'].iloc[-1] / df['close'].iloc[-1] * 100
                self.volatility_history.append(current_volatility)
                df['volatility'] = current_volatility
                
                # Market regime detection
                df['trend'] = np.where(df['ema20'] > df['ema50'], 1,
                                    np.where(df['ema20'] < df['ema50'], -1, 0))
                
                # Identifying swings
                df['swing_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
                df['swing_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))

            return df

            
        self.log_warning("Market data unavailable, generating synthetic data...")
        now = datetime.now()
        dates = pd.date_range(end=now, periods=count, freq='H')
        prices = 1.1 + np.cumsum(np.random.normal(0, 0.001, count))
        return pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.0005, count))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.0005, count))),
            'close': prices,
            'volume': np.random.randint(100, 1000, count)
        }, index=dates)

    def fit_arc(self, data, start, end):
        x = np.arange(end - start)
        y = data['close'].iloc[start:end].values
        x_norm = x / len(x)
        y_mean, y_std = y.mean(), y.std() or 1
        y_norm = (y - y_mean) / y_std
        
        def elliptical(x, a, b, c, d, e):
            return a*x**2 + b*x + c + d*np.sin(e*x)
            
        try:
            # More robust optimization with better initial guesses and bounds
            popt, pcov = curve_fit(
                elliptical, x_norm, y_norm, 
                p0=[0, 0, 0, 0.1, 5], 
                bounds=([-5, -5, -5, -5, 0.1], [5, 5, 5, 5, 20]),
                maxfev=10000
            )
            
            # Convert fitted values back to price scale
            fit = elliptical(x_norm, *popt) * y_std + y_mean
            
            # Calculate fit metrics
            r2 = 1 - ((y - fit)**2).sum() / ((y - y.mean())**2).sum()
            curvature = abs(popt[3]*popt[4])/(1+abs(popt[0]))
            
            # Calculate direction of arc curvature
            arc_direction = 1 if popt[0] < 0 else -1
            
            # Calculate momentum at end of arc (for signal direction)
            end_direction = np.sign(2*popt[0] + popt[1] + popt[3]*popt[4]*np.cos(popt[4]))
            
            # Add error metrics for uncertainty estimation
            perr = np.sqrt(np.diag(pcov))
            param_stability = np.mean(np.abs(perr / (np.abs(popt) + 1e-10)))
            
            # Calculate remaining potential of the arc
            # Higher values mean we're earlier in the arc formation
            x_extended = np.linspace(0, 2, 100)  # Extend beyond current data
            y_extended = elliptical(x_extended, *popt) * y_std + y_mean
            max_idx = np.argmax(y_extended) if arc_direction > 0 else np.argmin(y_extended)
            remaining_potential = abs(max_idx/100 - x_norm[-1]) / max(0.01, x_norm[-1])
            
            return {
                'start_idx': start,
                'end_idx': end,
                'start_time': data.index[start],
                'end_time': data.index[end-1],
                'fitted': fit,
                'r2': r2,
                'curvature': curvature,
                'params': popt,
                'volatility': y.std(),
                'param_stability': param_stability,
                'arc_direction': arc_direction,  # Positive is U-shaped, negative is inverted-U
                'end_direction': end_direction,  # Direction at current price
                'remaining_potential': remaining_potential,
                'quality': r2 * curvature * (1 - param_stability) * (1 + remaining_potential)
            }
        except Exception as e:
            self.log_debug(f"Arc fitting failed: {str(e)}")
            return None

    def detect_arcs(self, data):
        arcs = []
        # Use variable window sizes to detect arcs at different scales
        for size in [30, 40, 50, 60, 80]:
            step = max(5, size//5)  # Smaller step for more detection points
            
            # Ensure we have enough data
            if len(data) < size:
                continue
                
            # Scan for potential arcs
            for i in range(0, len(data) - size, step):
                arc = self.fit_arc(data, i, i + size)
                if arc and arc['r2'] > self.config.min_r2 and arc['curvature'] > self.config.min_curvature:
                    # Additional validation: check volatility relevance
                    if arc['volatility'] > 0.0001:  # Minimum meaningful volatility
                        arcs.append(arc)
        
        # Filter overlapping arcs using clustering
        if len(arcs) > 0:
            # Extract arc centers for clustering
            arc_centers = np.array([[a['start_idx'] + (a['end_idx'] - a['start_idx'])/2] for a in arcs])
            
            # Use DBSCAN to cluster nearby arcs
            clustering = DBSCAN(eps=15, min_samples=1).fit(arc_centers)
            labels = clustering.labels_
            
            # Select best arc from each cluster
            best_arcs = []
            for cluster_id in range(max(labels) + 1):
                cluster_arcs = [arc for i, arc in enumerate(arcs) if labels[i] == cluster_id]
                if cluster_arcs:
                    best_arc = max(cluster_arcs, key=lambda a: a['quality'])
                    best_arcs.append(best_arc)
            
            arcs = best_arcs
        
        self.log_info(f"Detected {len(arcs)} high-quality arcs")
        return sorted(arcs, key=lambda a: a['quality'], reverse=True)

    def analyze_market_regime(self, data):
        """Analyze market conditions to adjust strategy"""
        # Determine market trend
        close = data['close'].values
        last_50_change = (close[-1] / close[-50]) - 1
        
        if last_50_change > 0.01:
            trend = "bullish"
        elif last_50_change < -0.01:
            trend = "bearish"
        else:
            trend = "ranging"
            
        # Determine market volatility regime
        current_vol = data['atr'].iloc[-1] / data['close'].iloc[-1]
        avg_vol = np.mean(self.volatility_history) if self.volatility_history else current_vol
        
        if current_vol > avg_vol * 1.5:
            vol_regime = "high"
        elif current_vol < avg_vol * 0.7:
            vol_regime = "low"
        else:
            vol_regime = "normal"
            
        # Determine market momentum
        momentum = "positive" if data['macd_hist'].iloc[-1] > 0 else "negative"
        
        self.market_state = {
            "trend": trend,
            "volatility": vol_regime,
            "momentum": momentum
        }
        
        self.log_info(f"Market regime: {trend} trend, {vol_regime} volatility, {momentum} momentum")
        return self.market_state

    def find_signals(self, arcs, data):
        """Generate trade signals with advanced filtering"""
        signals = []
        current_price = data['close'].iloc[-1]
        market_regime = self.analyze_market_regime(data)
        
        # Skip signal generation in extreme volatility
        if market_regime["volatility"] == "high" and len(self.open_positions) > 0:
            self.log_warning("High volatility detected - not generating new signals")
            return []
            
        # Calculate total portfolio risk before adding new positions
        current_risk = self.calculate_portfolio_risk()
        if current_risk >= self.config.max_portfolio_risk:
            self.log_warning(f"Maximum portfolio risk reached ({current_risk:.2f}%) - not generating new signals")
            return []
            
        # Check open positions count
        if len(self.open_positions) >= self.config.max_open_trades:
            self.log_info(f"Maximum open trades ({self.config.max_open_trades}) reached - not generating new signals")
            return []
            
        # Check open sell positions count
        if self.open_sells >= self.config.max_open_sells:
            self.log_info(f"Maximum open sell positions ({self.config.max_open_sells}) reached - will only generate BUY signals")
            
        # Process each arc for potential signals
        for arc in arcs:
            if arc['quality'] < self.config.quality_threshold:
                continue
                
            # Extract parameters for signal generation
            a, b, _, d, e = arc['params']
            
            # Calculate derivative to find direction and momentum
            def deriv(xv): return 2*a*xv + b + d*e*math.cos(e*xv)
            current_deriv = deriv(1)
            prev_deriv = deriv(0.9)
            angle_diff = math.degrees(current_deriv - prev_deriv)
            
            # Additional signal filters based on market conditions
            signal_type = None
            
            # Generate BUY signals
            if (angle_diff > 30 and arc['end_direction'] > 0 and 
                (market_regime["trend"] != "bearish" or arc['quality'] > 0.8)):
                
                # Don't generate duplicate signals
                if 'buy' in self.last_signal_time and (datetime.now() - self.last_signal_time['buy']).seconds < 3600:
                    continue
                    
                signal_type = 'buy'
            
            # Generate SELL signals
            elif (angle_diff < -30 and arc['end_direction'] < 0 and 
                  (market_regime["trend"] != "bullish" or arc['quality'] > 0.8) and
                  self.open_sells < self.config.max_open_sells):  # Ensure no more than max_open_sells
                
                # Don't generate duplicate signals
                if 'sell' in self.last_signal_time and (datetime.now() - self.last_signal_time['sell']).seconds < 3600:
                    continue
                    
                signal_type = 'sell'
                
            if signal_type:
                # Calculate conviction level for position sizing
                conviction = min(1.0, arc['quality'] * (1 + abs(angle_diff)/90))
                
                # Calculate optimal entry based on recent price action
                optimal_entry_levels = self.calculate_optimal_entry(data, signal_type)
                
                # Validate signal with technical indicators
                indicator_confirmation = self.validate_with_indicators(data, signal_type)
                
                if indicator_confirmation >= 0.6:  # Require minimum confirmation level
                    signals.append({
                        'type': signal_type, 
                        'price': current_price,
                        'optimal_entry': optimal_entry_levels,
                        'volatility': arc['volatility'],
                        'quality': arc['quality'],
                        'conviction': conviction * indicator_confirmation,  # Scale by indicator confirmation
                        'arc_id': id(arc)  # Reference to originating arc
                    })
                    
                    # Update last signal time
                    self.last_signal_time[signal_type] = datetime.now()
        
        # Log signal generation
        if signals:
            top_signal = max(signals, key=lambda s: s['conviction'])
            self.log_info(f"Generated {len(signals)} signals, top signal: {top_signal['type']} with {top_signal['conviction']:.2f} conviction")
        
        return sorted(signals, key=lambda s: s['conviction'], reverse=True)

    def validate_with_indicators(self, data, signal_type):
        """Validate signals with technical indicators"""
        confirmations = 0
        total_indicators = 5
        
        # 1. Moving Average alignment
        if signal_type == 'buy':
            if data['ema20'].iloc[-1] > data['ema50'].iloc[-1]:
                confirmations += 1
        else:  # sell
            if data['ema20'].iloc[-1] < data['ema50'].iloc[-1]:
                confirmations += 1
                
        # 2. RSI confirmation
        rsi = data['rsi'].iloc[-1]
        if signal_type == 'buy' and rsi < 70:  # Not overbought
            confirmations += 1
        elif signal_type == 'sell' and rsi > 30:  # Not oversold
            confirmations += 1
            
        # 3. MACD confirmation
        if signal_type == 'buy' and data['macd_hist'].iloc[-1] > 0:
            confirmations += 1
        elif signal_type == 'sell' and data['macd_hist'].iloc[-1] < 0:
            confirmations += 1
            
        # 4. Volatility check - avoid trading in extremely high volatility
        avg_vol = np.mean(self.volatility_history) if len(self.volatility_history) else data['atr'].iloc[-1]
        current_vol = data['atr'].iloc[-1]
        if current_vol < avg_vol * 1.5:  # Not extremely volatile
            confirmations += 1
            
        # 5. Price position relative to long-term average
        if signal_type == 'buy' and data['close'].iloc[-1] > data['ema200'].iloc[-1]:
            confirmations += 1
        elif signal_type == 'sell' and data['close'].iloc[-1] < data['ema200'].iloc[-1]:
            confirmations += 1
            
        # Calculate confirmation level (0.0 to 1.0)
        return confirmations / total_indicators

    def calculate_optimal_entry(self, data, signal_type):
        """Calculate optimal entry levels for order execution"""
        current_price = data['close'].iloc[-1]
        atr = data['atr'].iloc[-1]
        
        if signal_type == 'buy':
            # For buys, optimal entry is slightly below current price
            levels = {
                'immediate': current_price,
                'preferred': max(current_price - 0.3 * atr, data['low'].iloc[-1]),
                'limit': max(current_price - 0.5 * atr, data['low'].iloc[-3:].min())
            }
        else:  # sell
            # For sells, optimal entry is slightly above current price
            levels = {
                'immediate': current_price,
                'preferred': min(current_price + 0.3 * atr, data['high'].iloc[-1]),
                'limit': min(current_price + 0.5 * atr, data['high'].iloc[-3:].max())
            }
            
        return levels

    def calculate_portfolio_risk(self):
        """Calculate current risk exposure from all open positions"""
        if not self.open_positions:
            return 0.0
            
        account = mt5.account_info()
        if not account:
            return 0.0
            
        total_risk = 0
        
        # Calculate risk for each position
        for ticket, pos in self.open_positions.items():
            if pos['sl'] != 0:  # Position has a stop loss
                risk_amount = abs(pos['price'] - pos['sl']) * pos['volume'] * 100000
                risk_percent = (risk_amount / account.balance) * 100
                total_risk += risk_percent
                
        return total_risk

    def calculate_position_size(self, sl_pips, conviction=1.0):
        """Calculate appropriate position size based on risk management"""
        account = mt5.account_info()
        symbol = mt5.symbol_info(self.config.symbol)
        if not account or not symbol:
            return self.config.min_lots
            
        # Calculate pip value
        pip_value = symbol.trade_tick_value / symbol.point
        
        # Apply dynamic risk adjustment based on recent performance
        adjusted_risk = self.config.max_risk_percent * self.risk_adjustment * conviction
        
        # Calculate risk amount from account balance
        risk_amount = account.balance * (adjusted_risk / 100)
        
        # Calculate position size
        position_size = risk_amount / (sl_pips * pip_value)
        
        # Apply limits
        lots = max(min(round(position_size, 2), self.config.max_lots), self.config.min_lots)
        
        self.log_info(f"Position size calculation: risk={adjusted_risk:.2f}%, SL={sl_pips:.1f} pips, size={lots:.2f} lots")
        return lots

    def update_risk_adjustment(self):
        """Dynamically adjust risk based on recent performance"""
        if not self.trade_history:
            return
            
        # Calculate win rate from recent trades
        win_count = sum(1 for trade in self.trade_history if trade['profit'] > 0)
        win_rate = win_count / len(self.trade_history)
        
        # Calculate average profit factor
        if sum(abs(t['profit']) for t in self.trade_history if t['profit'] < 0) == 0:
            profit_factor = 3.0  # Max value if no losses
        else:
            profit_factor = sum(t['profit'] for t in self.trade_history if t['profit'] > 0) / max(0.1, sum(abs(t['profit']) for t in self.trade_history if t['profit'] < 0))
            
        # Adjust risk based on performance metrics
        if win_rate >= 0.6 and profit_factor >= 1.5:
            self.risk_adjustment = min(1.2, self.risk_adjustment + 0.05)
        elif win_rate <= 0.4 or profit_factor < 1.0:
            self.risk_adjustment = max(0.5, self.risk_adjustment - 0.1)
            
        self.log_info(f"Risk adjustment: {self.risk_adjustment:.2f} (win rate: {win_rate:.2f}, profit factor: {profit_factor:.2f})")

    def check_open_positions(self):
        """Update status of all open positions"""
        if not mt5.initialize():
            return
            
        positions = mt5.positions_get(symbol=self.config.symbol)
        if positions:
            # Update our position tracking
            current_tickets = set()
            self.open_sells = 0
            
            for position in positions:
                ticket = position.ticket
                current_tickets.add(ticket)
                
                # Count open sells
                if position.type == mt5.POSITION_TYPE_SELL:
                    self.open_sells += 1
                
                # Track position if not already tracked
                if ticket not in self.open_positions:
                    self.open_positions[ticket] = {
                        'ticket': ticket,
                        'type': 'sell' if position.type == mt5.POSITION_TYPE_SELL else 'buy',
                        'price': position.price_open,
                        'volume': position.volume,
                        'sl': position.sl,
                        'tp': position.tp,
                        'profit': position.profit,
                        'open_time': datetime.now()
                    }
                else:
                    # Update position details
                    self.open_positions[ticket]['profit'] = position.profit
                    self.open_positions[ticket]['sl'] = position.sl
                    self.open_positions[ticket]['tp'] = position.tp
                    
            # Remove closed positions and add to history
            closed_tickets = set(self.open_positions.keys()) - current_tickets
            for ticket in closed_tickets:
                position = self.open_positions.pop(ticket)
                position['close_time'] = datetime.now()
                position['duration'] = (position['close_time'] - position['open_time']).total_seconds() / 3600  # hours
                self.trade_history.append(position)
                self.log_info(f"Position {ticket} closed with profit: {position['profit']}")
                
            # Update risk adjustment based on recent performance
            self.update_risk_adjustment()
            
        else:
            # No open positions
            if self.open_positions:
                # All positions were closed
                for ticket, position in self.open_positions.items():
                    position['close_time'] = datetime.now()
                    position['duration'] = (position['close_time'] - position['open_time']).total_seconds() / 3600
                    self.trade_history.append(position)
                
                self.open_positions = {}
                self.open_sells = 0
                self.update_risk_adjustment()

    def manage_open_positions(self, data):
        """Manage open positions: trailing stops, partial close, etc."""
        if not self.open_positions:
            return
            
        for ticket, position in list(self.open_positions.items()):
            # Skip positions without proper stop loss
            if position['sl'] == 0:
                continue
                
            current_price = data['close'].iloc[-1]
            
            # Calculate current profit in pips
            pip_size = mt5.symbol_info(self.config.symbol).point * 10
            if position['type'] == 'buy':
                profit_pips = (current_price - position['price']) / pip_size
            else:  # sell
                profit_pips = (position['price'] - current_price) / pip_size
                
            # Update trailing stop if enabled
            if self.config.trailing_stop and profit_pips > 0:
                target_pips = abs(position['tp'] - position['price']) / pip_size
                profit_ratio = profit_pips / target_pips
                
                # If profit reaches trailing trigger threshold
                if profit_ratio >= self.config.trailing_stop_trigger:
                    # Calculate new stop loss level
                    trailing_level = profit_pips * 0.7  # Trail at 70% of current profit
                    
                    if position['type'] == 'buy':
                        new_sl = position['price'] + (trailing_level * pip_size)
                        # Only move stop loss upward
                        if new_sl > position['sl']:
                            self.modify_position(ticket, new_sl, position['tp'])
                    else:  # sell
                        new_sl = position['price'] - (trailing_level * pip_size)
                        # Only move stop loss downward
                        if position['sl'] == 0 or new_sl < position['sl']:
                            self.modify_position(ticket, new_sl, position['tp'])

    def modify_position(self, ticket, sl, tp):
        """Modify existing position stop loss and take profit"""
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
            # Update our tracking
            if ticket in self.open_positions:
                self.open_positions[ticket]['sl'] = sl
                self.open_positions[ticket]['tp'] = tp
        else:
            self.log_error(f"Failed to modify position {ticket}: {result.retcode} - {result.comment}")

    def place_order(self, signal):
        """Place a new trade based on signal with advanced risk management"""
        order_type = signal['type']
        entry_price = signal['optimal_entry']['immediate']  # Use immediate entry price
        
        # Get fresh market data for stop loss/take profit calculation
        data = self.get_market_data(30)
        atr = data['atr'].iloc[-1]
        
        # Calculate stop loss based on recent volatility and support/resistance
        if order_type == 'buy':
            # For buys, place stop below recent lows, scaled by volatility
            sl_level = min(data['low'].iloc[-5:]) - 0.5 * atr
            sl_pips = (entry_price - sl_level) / (mt5.symbol_info(self.config.symbol).point * 10)
        else:  # sell
            # For sells, place stop above recent highs, scaled by volatility
            sl_level = max(data['high'].iloc[-5:]) + 0.5 * atr
            sl_pips = (sl_level - entry_price) / (mt5.symbol_info(self.config.symbol).point * 10)
        
        # Enforce minimum stop loss distance
        symbol = mt5.symbol_info(self.config.symbol)
        if not symbol:
            self.log_error("Failed to get symbol info")
            return False
            
        # Ensure stop loss is at least minimum required distance
        min_dist = symbol.trade_stops_level * symbol.point
        if order_type == 'buy':
            sl_level = min(sl_level, entry_price - min_dist)
        else:
            sl_level = max(sl_level, entry_price + min_dist)
            
        # Re-calculate stop loss in pips after adjustments
        pip_size = symbol.point * 10
        sl_pips = abs(entry_price - sl_level) / pip_size
        
        # Calculate take profit with asymmetric risk:reward
        tp_pips = sl_pips * self.config.profit_ratio
        if order_type == 'buy':
            tp_level = entry_price + tp_pips * pip_size
        else:
            tp_level = entry_price - tp_pips * pip_size
            
        # Calculate position size based on risk parameters and signal conviction
        lots = self.calculate_position_size(sl_pips, signal['conviction'])
        
        # Check for maximum open trades and open sells limit
        if len(self.open_positions) >= self.config.max_open_trades:
            self.log_warning(f"Maximum open trades ({self.config.max_open_trades}) reached - skipping order")
            return False
            
        if order_type == 'sell' and self.open_sells >= self.config.max_open_sells:
            self.log_warning(f"Maximum open sell positions ({self.config.max_open_sells}) reached - skipping sell order")
            return False
            
        # Log detailed trade plan
        self.log_info(f"Placing order: {order_type.upper()} @ {entry_price:.5f}, SL={sl_level:.5f}, TP={tp_level:.5f}, Size={lots}")
        
        # Prepare order request
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': self.config.symbol,
            'volume': lots,
            'type': mt5.ORDER_TYPE_BUY if order_type=='buy' else mt5.ORDER_TYPE_SELL,
            'price': entry_price,
            'sl': sl_level,
            'tp': tp_level,
            'deviation': 20,
            'magic': 123456,
            'comment': f'Arc Bot - Q:{signal["quality"]:.2f}',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK
        }
        
        # Send order
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.log_info(f"Order successfully placed: ticket #{result.order}")
            
            # Track new position
            if order_type == 'sell':
                self.open_sells += 1
                
            # Add to open positions tracking
            self.open_positions[result.order] = {
                'ticket': result.order,
                'type': order_type,
                'price': entry_price,
                'volume': lots,
                'sl': sl_level,
                'tp': tp_level,
                'profit': 0,
                'open_time': datetime.now()
            }
            
            return True
        else:
            self.log_error(f"Order failed: {result.retcode} - {result.comment}")
            return False

    def visualize(self, data):
        """Create visualization of market data, detected arcs, and positions"""
        if not self.fig:
            plt.ion()
            self.fig, (self.ax, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(14, 12), 
                                                                   gridspec_kw={'height_ratios': [3, 1, 1]})
            plt.subplots_adjust(hspace=0.3)
        
        # Clear previous plots
        self.ax.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Plot price chart with moving averages
        self.ax.plot(data.index, data['close'], label='Price', linewidth=1.5)
        self.ax.plot(data.index, data['ema20'], label='EMA20', linewidth=1, alpha=0.7)
        self.ax.plot(data.index, data['ema50'], label='EMA50', linewidth=1, alpha=0.7)
        
        # Plot detected arcs
        for i, arc in enumerate(self.current_arcs[:self.config.max_arcs]):
            start_idx, end_idx = arc['start_idx'], arc['end_idx']
            
            # Skip arcs that are out of range
            if start_idx >= len(data) or end_idx > len(data):
                continue
                
            # Use color coding based on arc direction (red for bearish, green for bullish)
            color = 'green' if arc['arc_direction'] > 0 else 'red'
            # alpha = min(1.0, 0.4 + 0.6 * (arc['quality'] / max(0.001, self.current_arcs[0]['quality'])))
            best_quality = max(0.001, abs(self.current_arcs[0]['quality']))  # Ensure positive divisor
            raw_alpha = 0.4 + 0.6 * (arc['quality'] / best_quality)
            alpha = max(0.0, min(1.0, raw_alpha))

            
            # Plot the arc
            self.ax.plot(data.index[start_idx:end_idx], arc['fitted'], 
                        linestyle='--', color=color, alpha=alpha, 
                        linewidth=2.0 if i < 3 else 1.0)
        
        # Mark open positions on the chart
        current_price = data['close'].iloc[-1]
        for ticket, pos in self.open_positions.items():
            marker = '^' if pos['type'] == 'buy' else 'v'
            color = 'green' if pos['type'] == 'buy' else 'red'
            
            # Find index closest to open time
            idx = data.index.get_indexer([pos['open_time']], method='nearest')[0]
            if 0 <= idx < len(data):
                self.ax.plot(data.index[idx], pos['price'], marker=marker, 
                            markersize=10, color=color)
                
                # Draw stop loss and take profit levels
                if pos['sl'] != 0:
                    self.ax.axhline(y=pos['sl'], linestyle=':', color=color, alpha=0.5)
                if pos['tp'] != 0:
                    self.ax.axhline(y=pos['tp'], linestyle=':', color=color, alpha=0.5)
        
        # Add chart title and legend
        self.ax.set_title(f"{self.config.symbol} - {self.market_state['trend'].capitalize()} Trend, " +
                        f"{self.market_state['volatility'].capitalize()} Volatility", fontsize=12)
        self.ax.legend(loc='upper left')
        self.ax.grid(True, alpha=0.3)
        
        # Plot technical indicators
        # RSI subplot
        self.ax2.plot(data.index, data['rsi'], color='purple', linewidth=1.5)
        self.ax2.axhline(y=70, linestyle='--', color='red', alpha=0.5)
        self.ax2.axhline(y=30, linestyle='--', color='green', alpha=0.5)
        self.ax2.axhline(y=50, linestyle='--', color='gray', alpha=0.3)
        self.ax2.set_ylim(0, 100)
        self.ax2.set_title('RSI(14)', fontsize=10)
        self.ax2.grid(True, alpha=0.3)
        
        # MACD subplot
        self.ax3.plot(data.index, data['macd'], color='blue', linewidth=1, label='MACD')
        self.ax3.plot(data.index, data['macd_signal'], color='red', linewidth=1, label='Signal')
        self.ax3.bar(data.index, data['macd_hist'], color=['green' if x > 0 else 'red' for x in data['macd_hist']], 
                    alpha=0.5)
        self.ax3.axhline(y=0, linestyle='-', color='gray', alpha=0.3)
        self.ax3.set_title('MACD', fontsize=10)
        self.ax3.legend(loc='upper left', fontsize=8)
        self.ax3.grid(True, alpha=0.3)
        
        # Display trading statistics in the corner
        stats_text = f"Trades: {len(self.open_positions)} open, {len(self.trade_history)} closed\n"
        if self.trade_history:
            wins = sum(1 for t in self.trade_history if t['profit'] > 0)
            win_rate = (wins / len(self.trade_history)) * 100
            stats_text += f"Win Rate: {win_rate:.1f}%, Risk Adj: {self.risk_adjustment:.2f}"
        
        self.ax.annotate(stats_text, xy=(0.01, 0.01), xycoords='axes fraction', 
                        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        
        # Update display
        plt.tight_layout()
        plt.pause(0.01)

    def run(self):
        """Main bot loop"""
        if not self.initialize_mt5():
            return
            
        self.running = True
        self.log_info("Starting main trading loop...")
        
        try:
            while self.running:
                start_time = datetime.now()
                
                # Update portfolio status
                self.check_open_positions()
                
                # Get market data with indicators
                data = self.get_market_data()
                
                # Detect elliptical arcs
                self.current_arcs = self.detect_arcs(data)
                
                # Manage existing positions (trailing stops, etc.)
                self.manage_open_positions(data)
                
                # Generate trading signals
                signals = self.find_signals(self.current_arcs, data)
                
                # Execute top signals if they meet criteria
                if signals:
                    top_signal = signals[0]
                    # Only trade if conviction is high enough
                    if top_signal['conviction'] > 0.7:
                        self.place_order(top_signal)
                
                # Update visualization
                self.visualize(data)
                
                # Calculate time until next update
                elapsed = (datetime.now() - start_time).total_seconds()
                sleep_time = max(0, self.config.visual_update_seconds - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.log_info("Bot stopped by user")
        except Exception as e:
            self.log_error(f"Error in main loop: {str(e)}")
        finally:
            # Clean shutdown
            self.running = False
            mt5.shutdown()
            plt.close('all')

if __name__ == '__main__':
    bot = ArcTradingBot()
    bot.run()
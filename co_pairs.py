import blpapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import threading
import queue
import time
import warnings
warnings.filterwarnings('ignore')

class HFTLeadLagStrategy:
    def __init__(self):
        self.session = None
        self.ref_data_service = None
        self.mktdata_service = None
        self.tick_data = {}
        self.real_time_data = queue.Queue()
        self.running = False
        
        # High-frequency correlated pairs with expected lead-lag relationships
        # Format: (leader_ticker, follower_ticker, expected_lag_ms, description)
        self.hft_pairs = [
            # Index Futures vs ETFs (most reliable HFT relationships)
            ('ES1 Index', 'SPY US Equity', 50, 'E-mini S&P vs SPY ETF'),
            ('NQ1 Index', 'QQQ US Equity', 50, 'E-mini NASDAQ vs QQQ ETF'),
            ('YM1 Index', 'DIA US Equity', 100, 'E-mini Dow vs DIA ETF'),
            ('RTY1 Index', 'IWM US Equity', 100, 'E-mini Russell vs IWM ETF'),
            
            # Currency Futures vs Spot (different liquidity/venues)
            ('EC1 Curncy', 'EURUSD Curncy', 25, 'EUR Futures vs EUR Spot'),
            ('BP1 Curncy', 'GBPUSD Curncy', 25, 'GBP Futures vs GBP Spot'),
            
            # Commodity Futures vs ETFs
            ('CL1 Comdty', 'USO US Equity', 200, 'Crude Oil vs Oil ETF'),
            ('GC1 Comdty', 'GLD US Equity', 150, 'Gold Futures vs Gold ETF'),
            
            # Cross-asset relationships (news/macro driven)
            ('VIX Index', 'ES1 Index', 30, 'VIX vs S&P Futures'),
            ('TNX Index', 'TLT US Equity', 100, '10Y Yield vs Bond ETF'),
            
            # Individual stocks vs sector ETFs during earnings
            ('AAPL US Equity', 'XLK US Equity', 75, 'Apple vs Tech ETF'),
            ('JPM US Equity', 'XLF US Equity', 75, 'JPMorgan vs Financials ETF'),
            
            # Different exchange same underlying
            ('SPX Index', 'SPY US Equity', 25, 'S&P Index vs ETF'),
            
            # International lead-lag (time zone arbitrage opportunities)
            ('NKY Index', 'EWJ US Equity', 500, 'Nikkei vs Japan ETF'),
        ]
    
    def connect_bloomberg(self):
        """Connect to Bloomberg Terminal with real-time capabilities"""
        try:
            # Create session options
            options = blpapi.SessionOptions()
            options.setServerHost('localhost')
            options.setServerPort(8194)
            
            # Create and start session
            self.session = blpapi.Session(options)
            if not self.session.start():
                print("Failed to start Bloomberg session")
                return False
            
            # Open reference data service
            if not self.session.openService("//blp/refdata"):
                print("Failed to open //blp/refdata service")
                return False
            self.ref_data_service = self.session.getService("//blp/refdata")
            
            # Open market data service for real-time subscriptions
            if not self.session.openService("//blp/mktdata"):
                print("Failed to open //blp/mktdata service")
                return False
            self.mktdata_service = self.session.getService("//blp/mktdata")
            
            print("Successfully connected to Bloomberg Terminal with real-time data")
            return True
            
        except Exception as e:
            print(f"Error connecting to Bloomberg: {e}")
            return False
    
    def get_tick_data(self, ticker, start_datetime, end_datetime):
        """Get historical tick-by-tick data from Bloomberg"""
        try:
            # Create intraday tick request
            request = self.ref_data_service.createRequest("IntradayTickRequest")
            request.set("security", ticker)
            request.set("startDateTime", start_datetime)
            request.set("endDateTime", end_datetime)
            request.set("eventType", "TRADE")  # Only actual trades, not quotes
            request.set("includeConditionCodes", True)
            
            print(f"Fetching tick data for {ticker} from {start_datetime} to {end_datetime}")
            
            # Send request
            cid = self.session.sendRequest(request)
            tick_data = []
            
            # Process response
            while True:
                event = self.session.nextEvent(10000)  # 10 second timeout
                
                if event.eventType() == blpapi.Event.RESPONSE or event.eventType() == blpapi.Event.PARTIAL_RESPONSE:
                    for msg in event:
                        tick_data_array = msg.getElement("tickData").getElement("tickData")
                        
                        for tick in tick_data_array.values():
                            time_str = tick.getElement("time").getValueAsString()
                            price = tick.getElement("value").getValueAsFloat()
                            size = tick.getElement("size").getValueAsInteger() if tick.hasElement("size") else 0
                            
                            # Convert Bloomberg time to pandas timestamp with microseconds
                            tick_time = pd.to_datetime(time_str)
                            
                            tick_data.append({
                                'timestamp': tick_time,
                                'price': price,
                                'size': size,
                                'microseconds': int(tick_time.microsecond)
                            })
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
            
            return pd.DataFrame(tick_data).set_index('timestamp')
            
        except Exception as e:
            print(f"Error fetching tick data for {ticker}: {e}")
            return pd.DataFrame()
    
    def calculate_microsecond_returns(self, tick_data, window_ms=100):
        """Calculate returns over microsecond windows"""
        if tick_data.empty:
            return pd.Series()
        
        # Resample tick data to fixed time intervals (e.g., 100ms)
        resampled = tick_data['price'].resample(f'{window_ms}ms').last().dropna()
        returns = resampled.pct_change().dropna()
        
        return returns
    
    def find_hft_lead_lag(self, leader_ticks, follower_ticks, max_lag_ms=1000, min_lag_ms=10):
        """Find optimal lead-lag relationship in milliseconds"""
        
        # Align data to common time grid (10ms intervals)
        time_grid = pd.date_range(
            start=max(leader_ticks.index.min(), follower_ticks.index.min()),
            end=min(leader_ticks.index.max(), follower_ticks.index.max()),
            freq='10ms'
        )
        
        # Forward fill prices to time grid
        leader_aligned = leader_ticks['price'].reindex(time_grid, method='ffill').dropna()
        follower_aligned = follower_ticks['price'].reindex(time_grid, method='ffill').dropna()
        
        # Calculate returns
        leader_returns = leader_aligned.pct_change().dropna()
        follower_returns = follower_aligned.pct_change().dropna()
        
        best_lag_ms = 0
        best_corr = 0
        best_pval = 1
        correlations = {}
        
        # Test different lag periods in milliseconds
        lag_range = range(min_lag_ms, max_lag_ms + 1, 10)  # 10ms increments
        
        for lag_ms in lag_range:
            lag_periods = lag_ms // 10  # Convert to periods (10ms each)
            
            if lag_periods >= len(leader_returns) or lag_periods >= len(follower_returns):
                continue
            
            # Align with lag
            if lag_periods == 0:
                leader_lagged = leader_returns
                follower_current = follower_returns
            else:
                leader_lagged = leader_returns[:-lag_periods]
                follower_current = follower_returns[lag_periods:]
            
            # Find common timestamps
            common_idx = leader_lagged.index.intersection(follower_current.index)
            if len(common_idx) < 100:  # Minimum observations
                continue
            
            leader_common = leader_lagged.loc[common_idx]
            follower_common = follower_current.loc[common_idx]
            
            # Filter out extreme outliers (likely data errors)
            leader_filtered = leader_common[abs(leader_common) < leader_common.std() * 5]
            follower_filtered = follower_common[abs(follower_common) < follower_common.std() * 5]
            
            if len(leader_filtered) < 50:
                continue
            
            # Calculate correlation
            try:
                corr, pval = stats.pearsonr(leader_filtered, follower_filtered)
                correlations[lag_ms] = {
                    'correlation': corr, 
                    'p_value': pval,
                    'observations': len(leader_filtered)
                }
                
                # Update best if stronger and significant
                if abs(corr) > abs(best_corr) and pval < 0.01:  # Stricter p-value for HFT
                    best_lag_ms = lag_ms
                    best_corr = corr
                    best_pval = pval
            
            except Exception as e:
                continue
        
        return best_lag_ms, best_corr, best_pval, correlations
    
    def generate_hft_signals(self, leader_returns, follower_returns, optimal_lag_ms, 
                            correlation, signal_threshold=3.0):
        """Generate high-frequency trading signals"""
        
        if optimal_lag_ms == 0:
            return pd.Series(0, index=follower_returns.index)
        
        signals = pd.Series(0, index=follower_returns.index)
        lag_periods = optimal_lag_ms // 10  # Convert to 10ms periods
        
        # Use shorter rolling window for HFT (5 seconds = 500 periods)
        rolling_window = 500
        
        # Calculate rolling z-scores
        leader_rolling_mean = leader_returns.rolling(window=rolling_window, min_periods=50).mean()
        leader_rolling_std = leader_returns.rolling(window=rolling_window, min_periods=50).std()
        
        leader_rolling_std = leader_rolling_std.replace(0, np.nan)
        leader_zscore = (leader_returns - leader_rolling_mean) / leader_rolling_std
        
        # Generate signals
        for i in range(rolling_window, len(leader_returns) - lag_periods):
            current_zscore = leader_zscore.iloc[i]
            
            if pd.isna(current_zscore):
                continue
            
            # Higher threshold for HFT due to noise
            if abs(current_zscore) > signal_threshold:
                signal_time = follower_returns.index[i + lag_periods]
                
                if correlation > 0:
                    signals.loc[signal_time] = 1 if current_zscore > 0 else -1
                else:
                    signals.loc[signal_time] = -1 if current_zscore > 0 else 1
        
        return signals
    
    def calculate_hft_performance(self, follower_returns, signals, transaction_cost=0.0005):
        """Calculate HFT strategy performance with realistic costs"""
        
        # Align signals with returns
        aligned_signals = signals.reindex(follower_returns.index, fill_value=0)
        signals_shifted = aligned_signals.shift(1).fillna(0)
        
        # Calculate gross returns
        strategy_returns = signals_shifted * follower_returns
        
        # Apply transaction costs (higher for HFT due to bid-ask spread)
        position_changes = signals_shifted.diff().abs()
        transaction_costs = position_changes * transaction_cost
        net_returns = strategy_returns - transaction_costs
        
        # Calculate performance metrics
        cumulative_returns = (1 + net_returns).cumprod()
        
        if len(cumulative_returns) == 0:
            return self._empty_performance_dict()
        
        # Annualized metrics (assuming 252 trading days, 6.5 hours, 60 minutes/hour, 6 periods/minute)
        periods_per_year = 252 * 6.5 * 60 * 6  # 10-second periods per year
        
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (cumulative_returns.iloc[-1] ** (periods_per_year / len(cumulative_returns))) - 1
        volatility = net_returns.std() * np.sqrt(periods_per_year)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        winning_trades = (net_returns > 0).sum()
        total_trades = (net_returns != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Average trade duration (for HFT, this should be very short)
        avg_trade_duration_seconds = self._calculate_avg_trade_duration(signals_shifted)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'avg_trade_duration_seconds': avg_trade_duration_seconds,
            'cumulative_returns': cumulative_returns,
            'net_returns': net_returns
        }
    
    def _calculate_avg_trade_duration(self, signals):
        """Calculate average trade duration in seconds"""
        trade_durations = []
        current_position = 0
        entry_time = None
        
        for timestamp, signal in signals.items():
            if signal != 0 and current_position == 0:
                # Opening position
                current_position = signal
                entry_time = timestamp
            elif signal == 0 and current_position != 0:
                # Closing position
                if entry_time is not None:
                    duration = (timestamp - entry_time).total_seconds()
                    trade_durations.append(duration)
                current_position = 0
                entry_time = None
        
        return np.mean(trade_durations) if trade_durations else 0
    
    def _empty_performance_dict(self):
        """Return empty performance dictionary"""
        return {
            'total_return': 0, 'annual_return': 0, 'volatility': 0,
            'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0,
            'total_trades': 0, 'avg_trade_duration_seconds': 0,
            'cumulative_returns': pd.Series(), 'net_returns': pd.Series()
        }
    
    def run_hft_backtest(self, test_date, start_hour=9, end_hour=16, signal_threshold=3.0):
        """Run HFT backtest for a specific trading day"""
        print("Starting High-Frequency Lead-Lag Strategy Backtest")
        print("=" * 70)
        
        results = []
        
        # Define time window for the test day
        start_datetime = datetime.combine(test_date, datetime.min.time().replace(hour=start_hour))
        end_datetime = datetime.combine(test_date, datetime.min.time().replace(hour=end_hour))
        
        for leader_ticker, follower_ticker, expected_lag_ms, description in self.hft_pairs:
            print(f"\nTesting: {description}")
            print(f"Leader: {leader_ticker}, Follower: {follower_ticker}")
            print(f"Expected lag: {expected_lag_ms}ms")
            
            # Fetch tick data
            leader_ticks = self.get_tick_data(leader_ticker, start_datetime, end_datetime)
            follower_ticks = self.get_tick_data(follower_ticker, start_datetime, end_datetime)
            
            if leader_ticks.empty or follower_ticks.empty:
                print(f"Failed to fetch tick data for {description}")
                continue
            
            print(f"Leader ticks: {len(leader_ticks)}, Follower ticks: {len(follower_ticks)}")
            
            # Find lead-lag relationship
            optimal_lag_ms, correlation, p_value, all_correlations = self.find_hft_lead_lag(
                leader_ticks, follower_ticks, max_lag_ms=1000
            )
            
            print(f"Optimal lag: {optimal_lag_ms}ms (expected: {expected_lag_ms}ms)")
            print(f"Correlation: {correlation:.4f}")
            print(f"P-value: {p_value:.6f}")
            
            if p_value > 0.01 or abs(correlation) < 0.05:
                print("Correlation not significant enough for HFT")
                continue
            
            # Calculate returns for signal generation
            leader_returns = self.calculate_microsecond_returns(leader_ticks, window_ms=100)
            follower_returns = self.calculate_microsecond_returns(follower_ticks, window_ms=100)
            
            if leader_returns.empty or follower_returns.empty:
                print("Failed to calculate returns")
                continue
            
            # Generate HFT signals
            signals = self.generate_hft_signals(
                leader_returns, follower_returns, optimal_lag_ms, 
                correlation, signal_threshold
            )
            
            signal_count = (signals != 0).sum()
            print(f"Generated {signal_count} trading signals")
            
            if signal_count == 0:
                print("No trading signals generated")
                continue
            
            # Calculate performance
            performance = self.calculate_hft_performance(follower_returns, signals)
            
            print(f"Total Return: {performance['total_return']:.4%}")
            print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {performance['max_drawdown']:.4%}")
            print(f"Win Rate: {performance['win_rate']:.2%}")
            print(f"Total Trades: {performance['total_trades']}")
            print(f"Avg Trade Duration: {performance['avg_trade_duration_seconds']:.1f} seconds")
            
            # Store results
            result = {
                'pair': description,
                'leader': leader_ticker,
                'follower': follower_ticker,
                'expected_lag_ms': expected_lag_ms,
                'optimal_lag_ms': optimal_lag_ms,
                'lag_accuracy': abs(optimal_lag_ms - expected_lag_ms),
                'correlation': correlation,
                'p_value': p_value,
                'leader_ticks': len(leader_ticks),
                'follower_ticks': len(follower_ticks),
                **performance
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def setup_realtime_monitoring(self, pairs_to_monitor=None):
        """Set up real-time monitoring for live trading (advanced feature)"""
        if pairs_to_monitor is None:
            pairs_to_monitor = self.hft_pairs[:3]  # Monitor top 3 pairs
        
        print("Setting up real-time monitoring...")
        print("Note: This requires additional Bloomberg permissions and careful risk management")
        
        # This would set up real-time subscriptions
        # Implementation would require additional Bloomberg real-time API usage
        
    def plot_hft_results(self, results_df):
        """Plot HFT-specific results"""
        if results_df.empty:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Lag Accuracy
        axes[0, 0].scatter(results_df['expected_lag_ms'], results_df['optimal_lag_ms'])
        axes[0, 0].plot([0, 1000], [0, 1000], 'r--', alpha=0.5)
        axes[0, 0].set_xlabel('Expected Lag (ms)')
        axes[0, 0].set_ylabel('Detected Lag (ms)')
        axes[0, 0].set_title('Lead-Lag Detection Accuracy')
        
        # 2. Return vs Trade Frequency
        axes[0, 1].scatter(results_df['total_trades'], results_df['total_return'])
        axes[0, 1].set_xlabel('Number of Trades')
        axes[0, 1].set_ylabel('Total Return')
        axes[0, 1].set_title('Return vs Trading Frequency')
        
        # 3. Sharpe vs Correlation
        axes[0, 2].scatter(results_df['correlation'], results_df['sharpe_ratio'])
        axes[0, 2].set_xlabel('Correlation Strength')
        axes[0, 2].set_ylabel('Sharpe Ratio')
        axes[0, 2].set_title('Correlation vs Performance')
        
        # 4. Trade Duration Distribution
        if 'avg_trade_duration_seconds' in results_df.columns:
            axes[1, 0].hist(results_df['avg_trade_duration_seconds'], bins=20, alpha=0.7)
            axes[1, 0].set_xlabel('Average Trade Duration (seconds)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Trade Duration Distribution')
        
        # 5. Win Rate vs Max Drawdown
        axes[1, 1].scatter(results_df['max_drawdown'], results_df['win_rate'])
        axes[1, 1].set_xlabel('Max Drawdown')
        axes[1, 1].set_ylabel('Win Rate')
        axes[1, 1].set_title('Risk vs Success Rate')
        
        # 6. Data Quality (tick counts)
        axes[1, 2].scatter(results_df['leader_ticks'], results_df['follower_ticks'])
        axes[1, 2].set_xlabel('Leader Tick Count')
        axes[1, 2].set_ylabel('Follower Tick Count')
        axes[1, 2].set_title('Data Quality Check')
        
        plt.tight_layout()
        plt.show()
    
    def disconnect(self):
        """Disconnect from Bloomberg"""
        self.running = False
        if self.session:
            self.session.stop()
            print("Disconnected from Bloomberg Terminal")

def main():
    """Main execution function"""
    strategy = HFTLeadLagStrategy()
    
    # Connect to Bloomberg
    if not strategy.connect_bloomberg():
        print("Unable to connect to Bloomberg Terminal")
        return
    
    try:
        # Test on a recent trading day (adjust as needed)
        test_date = datetime(2024, 12, 2)  # Monday - good for testing
        
        print(f"Testing HFT lead-lag relationships for {test_date.strftime('%Y-%m-%d')}")
        
        # Run backtest
        results = strategy.run_hft_backtest(
            test_date=test_date,
            start_hour=9,   # Market open
            end_hour=16,    # Market close
            signal_threshold=3.0  # Conservative threshold for HFT
        )
        
        if results.empty:
            print("No successful HFT backtests completed")
            return
        
        # Print summary
        print("\n" + "="*80)
        print("HFT LEAD-LAG BACKTEST SUMMARY")
        print("="*80)
        
        # Sort by Sharpe ratio
        results_sorted = results.sort_values('sharpe_ratio', ascending=False)
        
        print("\nTop Performing HFT Strategies:")
        print("-" * 100)
        for _, row in results_sorted.head().iterrows():
            print(f"{row['pair'][:35]:<35} | "
                  f"Lag: {row['optimal_lag_ms']:4.0f}ms | "
                  f"Corr: {row['correlation']:6.3f} | "
                  f"Sharpe: {row['sharpe_ratio']:6.2f} | "
                  f"Return: {row['total_return']:7.3%} | "
                  f"Trades: {row['total_trades']:4.0f}")
        
        # Plot results
        strategy.plot_hft_results(results_sorted)
        
        # Save results
        filename = f'hft_leadlag_results_{test_date.strftime("%Y%m%d")}.csv'
        results_sorted.to_csv(filename, index=False)
        print(f"\nDetailed results saved to {filename}")
        
    except Exception as e:
        print(f"Error during HFT backtest: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        strategy.disconnect()

if __name__ == "__main__":
    main()

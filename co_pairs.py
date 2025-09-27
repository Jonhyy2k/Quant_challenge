import blpapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class LeadLagStrategy:
    def __init__(self):
        self.session = None
        self.service = None
        
        # Define correlated asset pairs for testing
        # Format: (leader_ticker, follower_ticker, description)
        self.asset_pairs = [
            # Oil and oil-related assets
            ('CL1 Comdty', 'XLE US Equity', 'Crude Oil vs Energy ETF'),
            ('CL1 Comdty', 'USDCAD Curncy', 'Crude Oil vs USD/CAD'),
            
            # Currency correlations
            ('EURUSD Curncy', 'GBPUSD Curncy', 'EUR/USD vs GBP/USD'),
            
            # Safe haven assets
            ('GC1 Comdty', 'USDJPY Curncy', 'Gold vs USD/JPY'),
            ('VIX Index', 'GC1 Comdty', 'VIX vs Gold'),
            
            # Index and sector relationships
            ('SPX Index', 'QQQ US Equity', 'S&P 500 vs QQQ'),
            ('SPX Index', 'XLF US Equity', 'S&P 500 vs Financials ETF'),
            
            # Commodity correlations
            ('GC1 Comdty', 'SI1 Comdty', 'Gold vs Silver'),
            ('DXY Curncy', 'GC1 Comdty', 'Dollar Index vs Gold'),
        ]
    
    def connect_bloomberg(self):
        """Connect to Bloomberg Terminal"""
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
            
            self.service = self.session.getService("//blp/refdata")
            print("Successfully connected to Bloomberg Terminal")
            return True
            
        except Exception as e:
            print(f"Error connecting to Bloomberg: {e}")
            return False
    
    def get_historical_data(self, tickers, start_date, end_date, frequency='DAILY'):
        """Fetch historical price data from Bloomberg"""
        try:
            # Create request
            request = self.service.createRequest("HistoricalDataRequest")
            
            # Add securities
            for ticker in tickers:
                request.getElement("securities").appendValue(ticker)
            
            # Add fields
            request.getElement("fields").appendValue("PX_LAST")
            request.getElement("fields").appendValue("VOLUME")
            
            # Set date range
            request.set("startDate", start_date.strftime('%Y%m%d'))
            request.set("endDate", end_date.strftime('%Y%m%d'))
            request.set("periodicitySelection", frequency)
            
            # Send request
            cid = self.session.sendRequest(request)
            data = {}
            
            # Process response
            while True:
                event = self.session.nextEvent(500)  # 500ms timeout
                
                if event.eventType() == blpapi.Event.RESPONSE or event.eventType() == blpapi.Event.PARTIAL_RESPONSE:
                    for msg in event:
                        security_data = msg.getElement("securityData")
                        security = security_data.getElement("security").getValueAsString()
                        
                        field_data = security_data.getElement("fieldData")
                        dates = []
                        prices = []
                        volumes = []
                        
                        for point in field_data.values():
                            date = point.getElement("date").getValueAsString()
                            dates.append(pd.to_datetime(date))
                            
                            if point.hasElement("PX_LAST"):
                                prices.append(point.getElement("PX_LAST").getValueAsFloat())
                            else:
                                prices.append(np.nan)
                            
                            if point.hasElement("VOLUME"):
                                volumes.append(point.getElement("VOLUME").getValueAsFloat())
                            else:
                                volumes.append(np.nan)
                        
                        data[security] = pd.DataFrame({
                            'date': dates,
                            'price': prices,
                            'volume': volumes
                        }).set_index('date')
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
            
            return data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return {}
    
    def calculate_returns(self, price_data, lookback_window=1):
        """Calculate returns for given price series"""
        return price_data.pct_change(periods=lookback_window).dropna()
    
    def find_lead_lag_relationship(self, leader_returns, follower_returns, max_lag=5):
        """
        Find optimal lead-lag relationship between two return series
        Returns: (optimal_lag, correlation, p_value)
        """
        best_lag = 0
        best_corr = 0
        best_pval = 1
        
        correlations = {}
        
        # Test different lag periods
        for lag in range(0, max_lag + 1):
            if lag == 0:
                # Contemporaneous correlation
                leader_aligned = leader_returns
                follower_aligned = follower_returns
            else:
                # Leader leads by 'lag' periods
                leader_aligned = leader_returns[:-lag]
                follower_aligned = follower_returns[lag:]
            
            # Align series by common dates
            common_dates = leader_aligned.index.intersection(follower_aligned.index)
            if len(common_dates) < 30:  # Minimum observations
                continue
            
            leader_common = leader_aligned.loc[common_dates]
            follower_common = follower_aligned.loc[common_dates]
            
            # Calculate correlation
            corr, pval = stats.pearsonr(leader_common, follower_common)
            correlations[lag] = {'correlation': corr, 'p_value': pval}
            
            # Update best if this correlation is stronger and significant
            if abs(corr) > abs(best_corr) and pval < 0.05:
                best_lag = lag
                best_corr = corr
                best_pval = pval
        
        return best_lag, best_corr, best_pval, correlations
    
    def generate_signals(self, leader_returns, follower_returns, optimal_lag, correlation, 
                        signal_threshold=1.5):
        """
        Generate trading signals based on lead-lag relationship
        """
        if optimal_lag == 0:
            return pd.Series(0, index=follower_returns.index)
        
        signals = pd.Series(0, index=follower_returns.index)
        
        # Calculate z-scores for leader returns
        leader_std = leader_returns.rolling(window=20).std()
        leader_mean = leader_returns.rolling(window=20).mean()
        leader_zscore = (leader_returns - leader_mean) / leader_std
        
        # Generate signals with lag
        for i in range(optimal_lag, len(leader_returns)):
            signal_date = leader_returns.index[i]
            trading_date_idx = i + optimal_lag
            
            if trading_date_idx < len(follower_returns):
                trading_date = follower_returns.index[trading_date_idx]
                
                # Signal logic
                if abs(leader_zscore.iloc[i]) > signal_threshold:
                    if correlation > 0:
                        # Positive correlation: follow the leader
                        signals.loc[trading_date] = 1 if leader_zscore.iloc[i] > 0 else -1
                    else:
                        # Negative correlation: opposite direction
                        signals.loc[trading_date] = -1 if leader_zscore.iloc[i] > 0 else 1
        
        return signals
    
    def calculate_strategy_performance(self, follower_returns, signals, transaction_cost=0.001):
        """Calculate strategy returns and performance metrics"""
        # Shift signals to avoid look-ahead bias
        signals_shifted = signals.shift(1).fillna(0)
        
        # Calculate gross returns
        strategy_returns = signals_shifted * follower_returns
        
        # Apply transaction costs
        position_changes = signals_shifted.diff().abs()
        transaction_costs = position_changes * transaction_cost
        net_returns = strategy_returns - transaction_costs
        
        # Calculate cumulative performance
        cumulative_returns = (1 + net_returns).cumprod()
        
        # Performance metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (cumulative_returns.iloc[-1] ** (252/len(cumulative_returns))) - 1
        volatility = net_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        # Win rate
        winning_trades = (net_returns > 0).sum()
        total_trades = (net_returns != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'cumulative_returns': cumulative_returns,
            'strategy_returns': net_returns
        }
    
    def run_backtest(self, start_date, end_date, signal_threshold=1.5, max_lag=5):
        """Run complete backtest for all asset pairs"""
        print("Starting Lead-Lag Strategy Backtest")
        print("=" * 50)
        
        results = []
        
        for leader_ticker, follower_ticker, description in self.asset_pairs:
            print(f"\nTesting: {description}")
            print(f"Leader: {leader_ticker}, Follower: {follower_ticker}")
            
            # Fetch data
            tickers = [leader_ticker, follower_ticker]
            data = self.get_historical_data(tickers, start_date, end_date)
            
            if len(data) != 2:
                print(f"Failed to fetch data for {description}")
                continue
            
            # Get price series
            leader_prices = data[leader_ticker]['price'].dropna()
            follower_prices = data[follower_ticker]['price'].dropna()
            
            # Calculate returns
            leader_returns = self.calculate_returns(leader_prices)
            follower_returns = self.calculate_returns(follower_prices)
            
            # Align data
            common_dates = leader_returns.index.intersection(follower_returns.index)
            leader_returns = leader_returns.loc[common_dates]
            follower_returns = follower_returns.loc[common_dates]
            
            if len(common_dates) < 100:
                print(f"Insufficient data for {description}")
                continue
            
            # Find lead-lag relationship
            optimal_lag, correlation, p_value, all_correlations = self.find_lead_lag_relationship(
                leader_returns, follower_returns, max_lag
            )
            
            print(f"Optimal lag: {optimal_lag} days")
            print(f"Correlation: {correlation:.4f}")
            print(f"P-value: {p_value:.4f}")
            
            if p_value > 0.05:
                print("Correlation not statistically significant")
                continue
            
            # Generate signals
            signals = self.generate_signals(
                leader_returns, follower_returns, optimal_lag, 
                correlation, signal_threshold
            )
            
            # Calculate performance
            performance = self.calculate_strategy_performance(
                follower_returns, signals
            )
            
            print(f"Total Return: {performance['total_return']:.2%}")
            print(f"Annual Return: {performance['annual_return']:.2%}")
            print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
            print(f"Win Rate: {performance['win_rate']:.2%}")
            print(f"Total Trades: {performance['total_trades']}")
            
            # Store results
            result = {
                'pair': description,
                'leader': leader_ticker,
                'follower': follower_ticker,
                'optimal_lag': optimal_lag,
                'correlation': correlation,
                'p_value': p_value,
                **performance
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def plot_results(self, results_df):
        """Plot backtest results"""
        if results_df.empty:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Sharpe Ratio by pair
        axes[0, 0].barh(results_df['pair'], results_df['sharpe_ratio'])
        axes[0, 0].set_xlabel('Sharpe Ratio')
        axes[0, 0].set_title('Sharpe Ratio by Asset Pair')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # 2. Total Return vs Max Drawdown
        scatter = axes[0, 1].scatter(results_df['max_drawdown'], results_df['total_return'], 
                                   c=results_df['sharpe_ratio'], cmap='RdYlGn', s=100)
        axes[0, 1].set_xlabel('Max Drawdown')
        axes[0, 1].set_ylabel('Total Return')
        axes[0, 1].set_title('Return vs Risk')
        plt.colorbar(scatter, ax=axes[0, 1], label='Sharpe Ratio')
        
        # 3. Win Rate vs Total Trades
        axes[1, 0].scatter(results_df['total_trades'], results_df['win_rate'])
        axes[1, 0].set_xlabel('Total Trades')
        axes[1, 0].set_ylabel('Win Rate')
        axes[1, 0].set_title('Trading Frequency vs Success Rate')
        
        # 4. Correlation vs Lag
        axes[1, 1].scatter(results_df['optimal_lag'], results_df['correlation'])
        axes[1, 1].set_xlabel('Optimal Lag (days)')
        axes[1, 1].set_ylabel('Correlation')
        axes[1, 1].set_title('Lead-Lag Relationships')
        
        plt.tight_layout()
        plt.show()
    
    def disconnect(self):
        """Disconnect from Bloomberg"""
        if self.session:
            self.session.stop()
            print("Disconnected from Bloomberg Terminal")

def main():
    """Main execution function"""
    strategy = LeadLagStrategy()
    
    # Connect to Bloomberg
    if not strategy.connect_bloomberg():
        print("Unable to connect to Bloomberg Terminal")
        return
    
    try:
        # Define backtest period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # 2 years of data
        
        # Run backtest
        results = strategy.run_backtest(
            start_date=start_date,
            end_date=end_date,
            signal_threshold=1.5,  # Z-score threshold for signals
            max_lag=5  # Maximum lag to test (days)
        )
        
        # Print summary
        print("\n" + "="*50)
        print("BACKTEST SUMMARY")
        print("="*50)
        print(results.to_string(index=False))
        
        # Plot results
        strategy.plot_results(results)
        
        # Save results
        results.to_csv(f'leadlag_backtest_results_{datetime.now().strftime("%Y%m%d")}.csv', index=False)
        print(f"\nResults saved to leadlag_backtest_results_{datetime.now().strftime('%Y%m%d')}.csv")
        
    except Exception as e:
        print(f"Error during backtest: {e}")
    
    finally:
        strategy.disconnect()

if __name__ == "__main__":
    main()

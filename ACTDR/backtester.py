#!/usr/bin/env python3
"""
Pairs Trading Backtester - ACTDR
Monetize lag relationships through pairs trading strategy

USAGE:
======
python pairs_backtest.py --pair1 AG --pair2 PAAS --start-date 2025-01-01 --end-date 2025-01-31

Features:
- Detects lag using cross-correlation
- Generates trading signals based on lead-lag relationship
- Simulates trades with realistic transaction costs
- Calculates returns, Sharpe ratio, max drawdown
- Provides detailed trade log
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_fetcher_bloomberg import BloombergDataFetcher, get_bloomberg_ticker
from lag_detection import LagDetector


class PairsBacktester:
    """
    Backtest pairs trading strategy based on detected lag relationships
    """

    def __init__(self, asset1, asset2, transaction_cost=0.001, position_size=10000):
        """
        Initialize backtester

        Parameters:
        -----------
        asset1, asset2 : str
            Asset tickers
        transaction_cost : float
            Transaction cost as % of trade value (default: 0.1% = 0.001)
        position_size : float
            Dollar value of each position (default: $10,000)
        """
        self.asset1 = asset1
        self.asset2 = asset2
        self.transaction_cost = transaction_cost
        self.position_size = position_size

        self.trades = []
        self.equity_curve = []
        self.positions = {'asset1': 0, 'asset2': 0}
        self.cash = 100000  # Starting capital

    def detect_lag(self, data1, data2, max_lag=500):
        """
        Detect lag relationship between assets

        Parameters:
        -----------
        data1, data2 : pd.Series
            Price series for both assets
        max_lag : int
            Maximum lag to test

        Returns:
        --------
        dict : Lag detection results
        """
        detector = LagDetector(
            data1, data2,
            self.asset1, self.asset2,
            frequency='1000ms'
        )

        results = detector.comprehensive_analysis(max_lag=max_lag)

        return {
            'lag_periods': results['cross_correlation']['lag_periods'],
            'leader': results['cross_correlation']['leader'],
            'follower': results['cross_correlation']['follower'],
            'correlation': results['pearson']['correlation'],
        }

    def generate_signals(self, data1, data2, lag_info, lookback=20):
        """
        Generate trading signals based on lag relationship

        Parameters:
        -----------
        data1, data2 : pd.DataFrame
            Price data with 'close' column
        lag_info : dict
            Lag detection results
        lookback : int
            Lookback period for mean reversion calculation

        Returns:
        --------
        pd.DataFrame : Signals dataframe
        """
        df = pd.DataFrame({
            'price1': data1['close'],
            'price2': data2['close'],
        })

        # Calculate returns
        df['ret1'] = df['price1'].pct_change()
        df['ret2'] = df['price2'].pct_change()

        # Calculate spread (ratio between assets)
        df['spread'] = df['price1'] / df['price2']

        # Calculate z-score of spread (mean reversion signal)
        df['spread_mean'] = df['spread'].rolling(window=lookback).mean()
        df['spread_std'] = df['spread'].rolling(window=lookback).std()
        df['z_score'] = (df['spread'] - df['spread_mean']) / df['spread_std']

        # Generate signals based on z-score
        # When z-score > 2: spread is high → short spread (sell asset1, buy asset2)
        # When z-score < -2: spread is low → long spread (buy asset1, sell asset2)
        # When z-score near 0: close positions

        df['signal'] = 0
        df.loc[df['z_score'] > 2, 'signal'] = -1  # Short spread
        df.loc[df['z_score'] < -2, 'signal'] = 1   # Long spread
        df.loc[np.abs(df['z_score']) < 0.5, 'signal'] = 0  # Close

        return df

    def backtest(self, data1, data2, lag_info):
        """
        Run backtest simulation

        Parameters:
        -----------
        data1, data2 : pd.DataFrame
            Price data
        lag_info : dict
            Lag detection results

        Returns:
        --------
        dict : Backtest results
        """
        # Generate signals
        signals = self.generate_signals(data1, data2, lag_info)

        portfolio_value = self.cash
        equity_curve = [portfolio_value]
        trades = []

        current_position = 0  # 0 = flat, 1 = long spread, -1 = short spread

        for i in range(len(signals)):
            if i < 20:  # Skip initial period for moving averages
                equity_curve.append(portfolio_value)
                continue

            signal = signals.iloc[i]['signal']
            price1 = signals.iloc[i]['price1']
            price2 = signals.iloc[i]['price2']

            # Check for position changes
            if signal != current_position:
                # Close existing position if any
                if current_position != 0:
                    # Calculate P&L
                    if current_position == 1:  # Was long spread
                        pnl1 = self.positions['asset1'] * (price1 - self.entry_price1)
                        pnl2 = self.positions['asset2'] * (price2 - self.entry_price2)
                    else:  # Was short spread
                        pnl1 = self.positions['asset1'] * (self.entry_price1 - price1)
                        pnl2 = self.positions['asset2'] * (self.entry_price2 - price2)

                    total_pnl = pnl1 + pnl2

                    # Subtract transaction costs
                    total_pnl -= self.transaction_cost * (abs(self.positions['asset1'] * price1) +
                                                          abs(self.positions['asset2'] * price2))

                    portfolio_value += total_pnl

                    trades.append({
                        'timestamp': signals.index[i],
                        'action': 'CLOSE',
                        'position_type': 'LONG_SPREAD' if current_position == 1 else 'SHORT_SPREAD',
                        'pnl': total_pnl,
                        'portfolio_value': portfolio_value
                    })

                    self.positions['asset1'] = 0
                    self.positions['asset2'] = 0

                # Open new position if signal is not 0
                if signal != 0:
                    shares1 = self.position_size / price1
                    shares2 = self.position_size / price2

                    if signal == 1:  # Long spread: buy asset1, sell asset2
                        self.positions['asset1'] = shares1
                        self.positions['asset2'] = -shares2
                    else:  # Short spread: sell asset1, buy asset2
                        self.positions['asset1'] = -shares1
                        self.positions['asset2'] = shares2

                    self.entry_price1 = price1
                    self.entry_price2 = price2

                    trades.append({
                        'timestamp': signals.index[i],
                        'action': 'OPEN',
                        'position_type': 'LONG_SPREAD' if signal == 1 else 'SHORT_SPREAD',
                        'price1': price1,
                        'price2': price2,
                        'shares1': self.positions['asset1'],
                        'shares2': self.positions['asset2'],
                    })

                current_position = signal

            # Calculate current portfolio value (including unrealized P&L)
            if current_position != 0:
                unrealized_pnl1 = self.positions['asset1'] * (price1 - self.entry_price1)
                unrealized_pnl2 = self.positions['asset2'] * (price2 - self.entry_price2)
                current_portfolio_value = portfolio_value + unrealized_pnl1 + unrealized_pnl2
            else:
                current_portfolio_value = portfolio_value

            equity_curve.append(current_portfolio_value)

        # Calculate performance metrics
        equity_series = pd.Series(equity_curve, index=signals.index)
        returns = equity_series.pct_change().dropna()

        total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 6.5 * 60)  # Annualized
        max_drawdown = self.calculate_max_drawdown(equity_series)

        # Count winning/losing trades
        closed_trades = [t for t in trades if t['action'] == 'CLOSE']
        winning_trades = len([t for t in closed_trades if t['pnl'] > 0])
        losing_trades = len([t for t in closed_trades if t['pnl'] <= 0])

        return {
            'trades': trades,
            'equity_curve': equity_series,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(closed_trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / len(closed_trades) * 100 if closed_trades else 0,
            'final_portfolio_value': equity_series.iloc[-1],
        }

    def calculate_max_drawdown(self, equity_curve):
        """
        Calculate maximum drawdown

        Parameters:
        -----------
        equity_curve : pd.Series
            Portfolio value over time

        Returns:
        --------
        float : Maximum drawdown as percentage
        """
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax * 100
        return drawdown.min()


def run_backtest(ticker1, ticker2, start_date, end_date,
                 interval_minutes=1, transaction_cost=0.001, position_size=10000):
    """
    Run complete backtest for a pair

    Parameters:
    -----------
    ticker1, ticker2 : str
        Asset tickers
    start_date, end_date : str
        Date range (YYYY-MM-DD)
    interval_minutes : int
        Data frequency in minutes
    transaction_cost : float
        Transaction cost as % (default: 0.1%)
    position_size : float
        Dollar value per position

    Returns:
    --------
    dict : Backtest results
    """
    print("\n" + "="*80)
    print(f"PAIRS TRADING BACKTEST: {ticker1} <--> {ticker2}")
    print("="*80)
    print(f"Period: {start_date} to {end_date}")
    print(f"Frequency: {interval_minutes} minute bars")
    print(f"Transaction cost: {transaction_cost*100:.2f}%")
    print(f"Position size: ${position_size:,.0f}")
    print("="*80 + "\n")

    # Fetch data from Bloomberg
    fetcher = BloombergDataFetcher()
    if not fetcher.connect():
        raise Exception("Failed to connect to Bloomberg Terminal")

    try:
        bb_ticker1 = get_bloomberg_ticker(ticker1)
        bb_ticker2 = get_bloomberg_ticker(ticker2)

        print(f"Fetching data for {ticker1} ({bb_ticker1})...")
        data1 = fetcher.get_intraday_bars(bb_ticker1, start_date, end_date, interval_minutes)

        print(f"Fetching data for {ticker2} ({bb_ticker2})...")
        data2 = fetcher.get_intraday_bars(bb_ticker2, start_date, end_date, interval_minutes)

        if len(data1) == 0 or len(data2) == 0:
            print(f"✗ Insufficient data for {ticker1}-{ticker2}")
            return None

        print(f"\n✓ Retrieved {len(data1)} bars for {ticker1}")
        print(f"✓ Retrieved {len(data2)} bars for {ticker2}")

        # Align data
        common_index = data1.index.intersection(data2.index)
        data1 = data1.loc[common_index]
        data2 = data2.loc[common_index]

        print(f"✓ Aligned data: {len(common_index)} common timestamps\n")

        # Initialize backtester
        backtester = PairsBacktester(ticker1, ticker2, transaction_cost, position_size)

        # Detect lag
        print("Detecting lag relationship...")
        lag_info = backtester.detect_lag(data1['close'], data2['close'])

        print(f"✓ Lag detected: {lag_info['lag_periods']} periods")
        print(f"  Leader: {lag_info['leader']}")
        print(f"  Correlation: {lag_info['correlation']:.4f}\n")

        # Run backtest
        print("Running backtest simulation...")
        results = backtester.backtest(data1, data2, lag_info)

        return {
            **results,
            'lag_info': lag_info,
            'asset1': ticker1,
            'asset2': ticker2,
            'start_date': start_date,
            'end_date': end_date,
        }

    finally:
        fetcher.disconnect()


def generate_report(results, output_file='backtest_report.txt'):
    """
    Generate backtest report

    Parameters:
    -----------
    results : dict
        Backtest results
    output_file : str
        Output filename
    """
    report = []
    report.append("="*100)
    report.append("PAIRS TRADING BACKTEST REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*100)
    report.append("")

    report.append(f"Asset Pair: {results['asset1']} <--> {results['asset2']}")
    report.append(f"Period: {results['start_date']} to {results['end_date']}")
    report.append("")

    report.append("LAG ANALYSIS:")
    report.append("-"*100)
    report.append(f"  Lag: {results['lag_info']['lag_periods']} periods")
    report.append(f"  Leader: {results['lag_info']['leader']}")
    report.append(f"  Correlation: {results['lag_info']['correlation']:.4f}")
    report.append("")

    report.append("PERFORMANCE METRICS:")
    report.append("-"*100)
    report.append(f"  Total Return: {results['total_return']:.2f}%")
    report.append(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    report.append(f"  Max Drawdown: {results['max_drawdown']:.2f}%")
    report.append(f"  Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
    report.append("")

    report.append("TRADING STATISTICS:")
    report.append("-"*100)
    report.append(f"  Total Trades: {results['num_trades']}")
    report.append(f"  Winning Trades: {results['winning_trades']}")
    report.append(f"  Losing Trades: {results['losing_trades']}")
    report.append(f"  Win Rate: {results['win_rate']:.1f}%")
    report.append("")

    report.append("TRADE LOG:")
    report.append("-"*100)
    for i, trade in enumerate(results['trades'][:20], 1):  # Show first 20 trades
        if trade['action'] == 'OPEN':
            report.append(f"{i}. {trade['timestamp']} - OPEN {trade['position_type']}")
            report.append(f"   {results['asset1']}: {trade['shares1']:.2f} shares @ ${trade['price1']:.2f}")
            report.append(f"   {results['asset2']}: {trade['shares2']:.2f} shares @ ${trade['price2']:.2f}")
        else:
            report.append(f"{i}. {trade['timestamp']} - CLOSE {trade['position_type']}")
            report.append(f"   P&L: ${trade['pnl']:.2f}")
            report.append(f"   Portfolio Value: ${trade['portfolio_value']:,.2f}")
        report.append("")

    if len(results['trades']) > 20:
        report.append(f"... ({len(results['trades']) - 20} more trades)")
        report.append("")

    report.append("="*100)

    report_text = "\n".join(report)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n✓ Report saved to: {output_file}")
    return report_text


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Backtest pairs trading strategy based on lag relationships',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic backtest
  python pairs_backtest.py --pair1 AG --pair2 PAAS --start-date 2025-01-01 --end-date 2025-01-31

  # Custom transaction costs and position size
  python pairs_backtest.py --pair1 MP --pair2 REMX --start-date 2025-01-01 --end-date 2025-01-31 \\
    --transaction-cost 0.002 --position-size 50000

  # Different data frequency
  python pairs_backtest.py --pair1 BABA --pair2 KWEB --start-date 2025-01-01 --end-date 2025-01-31 \\
    --interval 5
        """
    )

    parser.add_argument('--pair1', type=str, required=True, help='First asset ticker')
    parser.add_argument('--pair2', type=str, required=True, help='Second asset ticker')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', type=int, default=1, help='Data interval in minutes (default: 1)')
    parser.add_argument('--transaction-cost', type=float, default=0.001,
                       help='Transaction cost as decimal (default: 0.001 = 0.1%%)')
    parser.add_argument('--position-size', type=float, default=10000,
                       help='Dollar value per position (default: 10000)')

    args = parser.parse_args()

    # Run backtest
    results = run_backtest(
        args.pair1, args.pair2,
        args.start_date, args.end_date,
        interval_minutes=args.interval,
        transaction_cost=args.transaction_cost,
        position_size=args.position_size
    )

    if results:
        # Generate report
        generate_report(results)

        # Print summary
        print("\n" + "="*80)
        print("BACKTEST SUMMARY")
        print("="*80)
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Total Trades: {results['num_trades']}")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()

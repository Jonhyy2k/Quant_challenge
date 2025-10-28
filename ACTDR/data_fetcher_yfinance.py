#!/usr/bin/env python3
"""
YFinance Data Fetcher
Fetches intraday data using yfinance (free, no Bloomberg required)

Use this for:
- Quick testing of lag detection methodology
- When Bloomberg is not available
- Proof-of-concept analysis

Limitations:
- Only 1-minute, 5-minute intervals (not tick data)
- Limited history (7 days for 1min, 60 days for 5min)
- Delayed data (15-20 min delay)
- Less reliable for real arbitrage detection

USAGE:
======
from data_fetcher_yfinance import fetch_pair_data_yfinance

data1, data2 = fetch_pair_data_yfinance('VALE', 'GDX', days=5, interval='1m')
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def fetch_intraday_yfinance(ticker, days=5, interval='1m'):
    """
    Fetch intraday data from yfinance

    Parameters:
    -----------
    ticker : str
        Stock ticker (e.g., 'VALE', 'GDX')
    days : int
        Number of days of history (max 7 for 1m, 60 for 5m)
    interval : str
        Data interval: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h'

    Returns:
    --------
    pd.DataFrame with OHLCV data
    """
    print(f"Fetching {ticker} - last {days} days, interval {interval}...")

    # Validate interval/period combination
    if interval == '1m' and days > 7:
        print(f"  Warning: 1m data limited to 7 days, adjusting from {days} to 7 days")
        days = 7

    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=f'{days}d', interval=interval)

        if len(data) == 0:
            print(f"  ✗ No data retrieved for {ticker}")
            return pd.DataFrame()

        # Rename columns to match Bloomberg format
        data = data.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # Keep only OHLCV
        data = data[['open', 'high', 'low', 'close', 'volume']]

        print(f"  ✓ Retrieved {len(data)} bars for {ticker}")
        return data

    except Exception as e:
        print(f"  ✗ Error fetching {ticker}: {e}")
        return pd.DataFrame()


def fetch_pair_data_yfinance(ticker1, ticker2, days=5, interval='1m'):
    """
    Fetch intraday data for a pair of assets

    Parameters:
    -----------
    ticker1 : str
        First ticker symbol
    ticker2 : str
        Second ticker symbol
    days : int
        Number of days of history
    interval : str
        Data interval ('1m', '5m', etc.)

    Returns:
    --------
    tuple: (data1, data2) - DataFrames for both assets
    """
    print("\n" + "="*70)
    print(f"Fetching data via yfinance")
    print(f"Pair: {ticker1} <--> {ticker2}")
    print(f"Period: {days} days, Interval: {interval}")
    print("="*70)

    data1 = fetch_intraday_yfinance(ticker1, days=days, interval=interval)
    data2 = fetch_intraday_yfinance(ticker2, days=days, interval=interval)

    # Align timestamps
    if len(data1) > 0 and len(data2) > 0:
        # Find common timestamps
        common_times = data1.index.intersection(data2.index)
        data1 = data1.loc[common_times]
        data2 = data2.loc[common_times]

        print(f"\n✓ Aligned data: {len(common_times)} common timestamps")
        print(f"  Date range: {data1.index[0]} to {data1.index[-1]}")
    else:
        print("\n✗ Failed to retrieve data for one or both assets")

    print("="*70 + "\n")

    return data1, data2


def fetch_multiple_pairs_yfinance(pairs, days=5, interval='1m'):
    """
    Fetch data for multiple pairs

    Parameters:
    -----------
    pairs : list of tuples
        List of (ticker1, ticker2) pairs
    days : int
        Number of days of history
    interval : str
        Data interval

    Returns:
    --------
    dict: {(ticker1, ticker2): (data1, data2)}
    """
    results = {}

    for ticker1, ticker2 in pairs:
        data1, data2 = fetch_pair_data_yfinance(ticker1, ticker2, days, interval)

        if len(data1) > 0 and len(data2) > 0:
            results[(ticker1, ticker2)] = (data1, data2)
        else:
            print(f"✗ Skipping {ticker1}-{ticker2} due to data issues\n")

    return results


def get_recent_trading_days(num_days=5):
    """
    Get list of recent trading days (excluding weekends)

    Parameters:
    -----------
    num_days : int
        Number of trading days to retrieve

    Returns:
    --------
    list of datetime objects
    """
    days = []
    current = datetime.now()

    while len(days) < num_days:
        # Skip weekends
        if current.weekday() < 5:  # Monday=0, Friday=4
            days.append(current)
        current -= timedelta(days=1)

    return list(reversed(days))


if __name__ == "__main__":
    """
    Test yfinance data fetching
    """
    print("Testing yfinance data fetching...")
    print("="*70)

    # Test single pair
    ticker1 = 'VALE'
    ticker2 = 'GDX'

    data1, data2 = fetch_pair_data_yfinance(ticker1, ticker2, days=5, interval='1m')

    if len(data1) > 0:
        print(f"\nSample data for {ticker1}:")
        print(data1.head())
        print(f"\nData shape: {data1.shape}")
        print(f"Date range: {data1.index[0]} to {data1.index[-1]}")
    else:
        print(f"\nNo data retrieved for {ticker1}")

    print("\n" + "="*70)
    print("Note: yfinance provides delayed data (~15-20 min)")
    print("For real-time lag detection, use Bloomberg API")
    print("="*70)

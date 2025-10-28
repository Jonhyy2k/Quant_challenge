#!/usr/bin/env python3
"""
Correlation Analysis for Asset Pairs
Finds strongly correlated assets across multiple asset classes
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import combinations

def fetch_data(tickers, period='1y', interval='1d'):
    """Fetch historical data for given tickers"""
    print(f"Fetching data for {len(tickers)} tickers...")
    data = yf.download(tickers, period=period, interval=interval, progress=False, auto_adjust=True)

    # Handle different return structures
    if isinstance(data.columns, pd.MultiIndex):
        # Multiple tickers - extract Close prices
        if 'Close' in data.columns.get_level_values(0):
            data = data['Close']
        else:
            # Fallback to first price column
            data = data.xs('Close', level=0, axis=1)
    else:
        # Single ticker or already simplified
        if 'Close' in data.columns:
            data = data['Close']

    return data

def calculate_correlations(data, min_correlation=0.8):
    """Calculate correlations and find strong pairs"""
    # Calculate returns instead of raw prices for better correlation
    returns = data.pct_change().dropna()

    # Calculate correlation matrix
    corr_matrix = returns.corr()

    # Find pairs with correlation > threshold
    pairs = []
    tickers = list(data.columns)

    for i, ticker1 in enumerate(tickers):
        for ticker2 in tickers[i+1:]:
            corr = corr_matrix.loc[ticker1, ticker2]
            if abs(corr) >= min_correlation and not np.isnan(corr):
                pairs.append({
                    'asset1': ticker1,
                    'asset2': ticker2,
                    'correlation': corr,
                    'abs_correlation': abs(corr)
                })

    # Sort by absolute correlation
    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df) > 0:
        pairs_df = pairs_df.sort_values('abs_correlation', ascending=False)

    return pairs_df, corr_matrix

def analyze_asset_universe():
    """Analyze a diverse universe of assets"""

    # Diverse asset universe
    assets = {
        # Emerging Market ETFs and Indices
        'Emerging Markets': ['EWZ', 'EWW', 'EEM', 'FXI', 'EWY', 'INDA', 'EWT', 'EWA', 'RSX', 'ERUS'],

        # Commodities
        'Commodities': ['GLD', 'SLV', 'USO', 'UNG', 'DBA', 'DBB', 'COPX', 'PALL'],

        # Forex (Currency ETFs)
        'Forex': ['FXE', 'FXY', 'FXB', 'FXA', 'FXC', 'CYB', 'FXF'],

        # Sector ETFs
        'Sectors': ['XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLU', 'XLP', 'XLY', 'XLB'],

        # Major Indices
        'Indices': ['SPY', 'QQQ', 'DIA', 'IWM', 'VT', 'VTI'],

        # Crypto (via proxies)
        'Crypto': ['BITO', 'GBTC'],

        # Bonds
        'Bonds': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'EMB']
    }

    # Flatten all tickers
    all_tickers = []
    for category, tickers in assets.items():
        all_tickers.extend(tickers)

    print(f"Analyzing {len(all_tickers)} assets across {len(assets)} categories...")

    # Fetch data
    data = fetch_data(all_tickers, period='2y', interval='1d')

    # Remove tickers with too much missing data
    valid_tickers = data.columns[data.isnull().sum() < len(data) * 0.1]
    data = data[valid_tickers]
    data = data.dropna(axis=1, how='all')

    print(f"Valid tickers after cleanup: {len(data.columns)}")

    # Find correlations
    pairs_df, corr_matrix = calculate_correlations(data, min_correlation=0.8)

    return pairs_df, data, assets

def categorize_pair(asset1, asset2, asset_categories):
    """Determine which categories the assets belong to"""
    cat1 = None
    cat2 = None

    for category, tickers in asset_categories.items():
        if asset1 in tickers:
            cat1 = category
        if asset2 in tickers:
            cat2 = category

    return cat1, cat2

def generate_report(pairs_df, asset_categories):
    """Generate a detailed report of findings"""

    if len(pairs_df) == 0:
        return "No pairs found with correlation > 0.8"

    report = []
    report.append("=" * 80)
    report.append("CORRELATED ASSET PAIRS ANALYSIS")
    report.append("Research Project: Emerging Market Inefficiency Detection")
    report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")
    report.append(f"Total pairs found with correlation > 0.8: {len(pairs_df)}")
    report.append("")
    report.append("=" * 80)
    report.append("TOP CORRELATED PAIRS")
    report.append("=" * 80)
    report.append("")

    for idx, row in pairs_df.head(20).iterrows():
        cat1, cat2 = categorize_pair(row['asset1'], row['asset2'], asset_categories)

        report.append(f"PAIR #{idx + 1}")
        report.append(f"  Assets: {row['asset1']} <--> {row['asset2']}")
        report.append(f"  Correlation: {row['correlation']:.4f}")
        report.append(f"  Categories: {cat1} <--> {cat2}")

        # Add descriptions
        descriptions = {
            'EWZ': 'iShares MSCI Brazil ETF',
            'EWW': 'iShares MSCI Mexico ETF',
            'EEM': 'iShares MSCI Emerging Markets ETF',
            'FXI': 'iShares China Large-Cap ETF',
            'EWY': 'iShares MSCI South Korea ETF',
            'INDA': 'iShares MSCI India ETF',
            'EWT': 'iShares MSCI Taiwan ETF',
            'GLD': 'SPDR Gold Trust',
            'SLV': 'iShares Silver Trust',
            'USO': 'United States Oil Fund',
            'UNG': 'United States Natural Gas Fund',
            'SPY': 'S&P 500 ETF',
            'QQQ': 'Nasdaq 100 ETF',
            'XLF': 'Financial Sector ETF',
            'XLE': 'Energy Sector ETF',
            'TLT': '20+ Year Treasury Bond ETF',
            'FXE': 'Euro Currency Trust',
            'FXY': 'Japanese Yen Trust',
        }

        desc1 = descriptions.get(row['asset1'], 'N/A')
        desc2 = descriptions.get(row['asset2'], 'N/A')

        report.append(f"  {row['asset1']}: {desc1}")
        report.append(f"  {row['asset2']}: {desc2}")
        report.append("")

    report.append("=" * 80)
    report.append("RESEARCH NOTES")
    report.append("=" * 80)
    report.append("")
    report.append("Strong correlations indicate assets that move together, making them")
    report.append("candidates for pairs trading strategies. For emerging markets research,")
    report.append("focus on pairs where one asset may have delayed price discovery due to:")
    report.append("  - Lower trading volume")
    report.append("  - Time zone differences")
    report.append("  - Information asymmetry")
    report.append("  - Market microstructure differences")
    report.append("")
    report.append("Next steps:")
    report.append("  1. Analyze intraday data to detect price lag between pairs")
    report.append("  2. Measure time-to-adjust after correlated shocks")
    report.append("  3. Quantify theoretical arbitrage opportunities")
    report.append("  4. Model transaction costs and slippage")
    report.append("")

    return "\n".join(report)

if __name__ == "__main__":
    print("Starting correlation analysis...")
    print()

    # Run analysis
    pairs_df, data, asset_categories = analyze_asset_universe()

    # Generate report
    report = generate_report(pairs_df, asset_categories)

    # Save to file
    output_file = "/home/joaop/PyResearch/ITAU/ACTDR/correlated_pairs_findings.txt"
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"\n✓ Analysis complete!")
    print(f"✓ Found {len(pairs_df)} pairs with correlation > 0.8")
    print(f"✓ Results saved to: {output_file}")
    print("\nTop 10 pairs:")
    print(pairs_df.head(10)[['asset1', 'asset2', 'correlation']].to_string(index=False))

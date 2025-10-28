#!/usr/bin/env python3
"""
Main Lag Detection Analysis Script
Runs comprehensive lag analysis on top emerging market pairs

USAGE:
======

Option 1: Use yfinance (quick test, free)
    python run_lag_analysis.py --source yfinance --days 5

Option 2: Use Bloomberg (high-quality data)
    python run_lag_analysis.py --source bloomberg --start 2025-01-15 --end 2025-01-20

Option 3: Analyze specific pair
    python run_lag_analysis.py --source yfinance --pair VALE,GDX --days 7
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

# Import our modules
from lag_detection import LagDetector, convert_lag_to_time
from data_fetcher_yfinance import fetch_pair_data_yfinance
from data_fetcher_bloomberg import fetch_pair_data_bloomberg, get_bloomberg_ticker
from lag_visualization import plot_comprehensive_analysis, create_summary_dashboard


# Top pairs identified from correlation analysis (in priority order)
TOP_PAIRS = [
    # Commodity-EM pairs (HIGHEST PRIORITY)
    ('COPX', 'EPU'),    # Copper vs Peru - Corr: 0.84
    ('PICK', 'VALE'),   # Metals vs Vale - Corr: 0.77W
    ('PICK', 'DEM'),    # Metals vs EM Dividend - Corr: 0.80
    ('GDX', 'EPU'),     # Gold vs Peru - Corr: 0.69
    ('COPX', 'VALE'),   # Copper vs Vale - Corr: 0.66

    # Brazilian stock pairs
    ('SID', 'VALE'),    # Steel vs Mining - Corr: 0.73
    ('ITUB', 'BBD'),    # Banks - Corr: 0.71
    ('EWZ', 'ITUB'),    # Brazil ETF vs Bank - Corr: 0.83

    # Chinese stock pairs
    ('BABA', 'JD'),     # Alibaba vs JD - Corr: 0.73
    ('BABA', 'KWEB'),   # Alibaba vs China Internet - Corr: 0.83
    ('JD', 'KWEB'),     # JD vs China Internet - Corr: 0.83

    # Broad EM pairs (control group)
    ('EEM', 'IEMG'),    # Broad EM ETFs - Corr: 0.99
    ('FXI', 'MCHI'),    # China ETFs - Corr: 0.98
]


def analyze_pair(ticker1, ticker2, data_source='yfinance', **kwargs):
    """
    Run comprehensive lag analysis on a single pair

    Parameters:
    -----------
    ticker1, ticker2 : str
        Asset tickers
    data_source : str
        'yfinance' or 'bloomberg'
    **kwargs : additional arguments for data fetching

    Returns:
    --------
    dict : Analysis results
    """
    print("\n" + "="*80)
    print(f"ANALYZING PAIR: {ticker1} <--> {ticker2}")
    print("="*80)

    # Fetch data
    if data_source == 'yfinance':
        days = kwargs.get('days', 5)
        interval = kwargs.get('interval', '1m')
        data1, data2 = fetch_pair_data_yfinance(ticker1, ticker2, days=days, interval=interval)
        frequency = interval

    elif data_source == 'bloomberg':
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        interval_minutes = kwargs.get('interval_minutes', 1)

        # Convert tickers to Bloomberg format
        bb_ticker1 = get_bloomberg_ticker(ticker1)
        bb_ticker2 = get_bloomberg_ticker(ticker2)

        data1, data2 = fetch_pair_data_bloomberg(
            bb_ticker1, bb_ticker2,
            start_date, end_date,
            interval_minutes=interval_minutes
        )
        frequency = f'{interval_minutes}min'

    else:
        raise ValueError(f"Unknown data source: {data_source}")

    # Check if we have data
    if len(data1) == 0 or len(data2) == 0:
        print(f"✗ Insufficient data for {ticker1}-{ticker2}, skipping...")
        return None

    # Create detector and run analysis
    detector = LagDetector(
        data1['close'], data2['close'],
        ticker1, ticker2,
        frequency=frequency
    )

    results = detector.comprehensive_analysis(max_lag=60)

    # Generate visualizations
    plot_dir = f'./lag_plots/{ticker1}_{ticker2}'
    os.makedirs(plot_dir, exist_ok=True)
    plot_comprehensive_analysis(detector, save_dir=plot_dir)

    return results


def analyze_multiple_pairs(pairs, data_source='yfinance', **kwargs):
    """
    Analyze multiple pairs and generate summary report

    Parameters:
    -----------
    pairs : list of tuples
        List of (ticker1, ticker2) pairs
    data_source : str
        'yfinance' or 'bloomberg'
    **kwargs : additional arguments for data fetching

    Returns:
    --------
    list : Results for all pairs
    """
    all_results = []

    for ticker1, ticker2 in pairs:
        try:
            result = analyze_pair(ticker1, ticker2, data_source=data_source, **kwargs)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n✗ Error analyzing {ticker1}-{ticker2}: {e}")
            continue

    return all_results


def generate_summary_report(results, output_file='lag_analysis_summary.txt'):
    """
    Generate a comprehensive summary report

    Parameters:
    -----------
    results : list
        List of analysis results
    output_file : str
        Path to save report
    """
    report = []
    report.append("="*100)
    report.append("LAG DETECTION ANALYSIS - SUMMARY REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*100)
    report.append("")

    report.append(f"Total pairs analyzed: {len(results)}")
    report.append("")

    # Summary statistics
    pairs_with_lag = []
    for r in results:
        if r['cross_correlation']['lag_periods'] > 0:
            pairs_with_lag.append((
                r['asset1'],
                r['asset2'],
                r['cross_correlation']['lag_periods'],
                r['cross_correlation']['leader'],
                r['pearson']['correlation']
            ))

    report.append("="*100)
    report.append(f"PAIRS WITH DETECTED LAG: {len(pairs_with_lag)}")
    report.append("="*100)
    report.append("")

    if len(pairs_with_lag) > 0:
        for asset1, asset2, lag, leader, corr in sorted(pairs_with_lag, key=lambda x: x[2], reverse=True):
            report.append(f"{asset1} <--> {asset2}")
            report.append(f"  Correlation: {corr:.4f}")
            report.append(f"  Lag: {lag} periods")
            report.append(f"  Leader: {leader}")
            report.append("")

    # Detailed results for each pair
    report.append("="*100)
    report.append("DETAILED RESULTS BY PAIR")
    report.append("="*100)
    report.append("")

    for i, r in enumerate(results, 1):
        report.append(f"PAIR #{i}: {r['asset1']} <--> {r['asset2']}")
        report.append("-"*100)
        report.append(f"Data Points: {r['data_points']}")
        report.append(f"Frequency: {r['frequency']}")
        report.append("")

        # Correlation
        report.append("Pearson Correlation:")
        report.append(f"  Coefficient: {r['pearson']['correlation']:.4f}")
        report.append(f"  P-value: {r['pearson']['p_value']:.6f}")
        report.append(f"  Significant: {r['pearson']['significant']}")
        report.append("")

        # Cross-correlation
        ccf = r['cross_correlation']
        report.append("Cross-Correlation Analysis:")
        report.append(f"  Leader: {ccf['leader']}")
        report.append(f"  Follower: {ccf['follower']}")
        report.append(f"  Lag: {ccf['lag_periods']} periods ({ccf['frequency']})")
        report.append(f"  Max Correlation: {ccf['max_correlation']:.4f}")

        # Convert to time
        time_lag = convert_lag_to_time(ccf['lag_periods'], ccf['frequency'])
        report.append(f"  Time Lag: {time_lag}")
        report.append("")

        # Granger causality
        granger = r['granger']
        report.append("Granger Causality Test:")
        report.append(f"  Leader: {granger.get('leader', 'N/A')}")
        if 'lag' in granger:
            report.append(f"  Lag: {granger['lag']} periods")
        report.append("")

        # Response time
        if 'response_time' in r and 'error' not in r['response_time']:
            rt = r['response_time']
            report.append("Response Time Analysis:")
            report.append(f"  Mean response: {rt['mean_response_periods']:.2f} periods")
            report.append(f"  Median response: {rt['median_response_periods']:.2f} periods")
            report.append(f"  Events analyzed: {rt['num_events']}")
            report.append("")

        report.append("")

    # Recommendations
    report.append("="*100)
    report.append("RESEARCH RECOMMENDATIONS")
    report.append("="*100)
    report.append("")

    if len(pairs_with_lag) > 0:
        report.append("✓ LAG DETECTED in some pairs!")
        report.append("")
        report.append("Next steps:")
        report.append("  1. Focus on pairs with consistent, measurable lag")
        report.append("  2. Use Bloomberg tick data for higher precision")
        report.append("  3. Analyze lag patterns across different market conditions")
        report.append("  4. Quantify theoretical arbitrage opportunity")
        report.append("  5. Model transaction costs and slippage")
    else:
        report.append("✗ No significant lag detected at this frequency/timeframe")
        report.append("")
        report.append("Considerations:")
        report.append("  1. Try higher frequency data (tick-level)")
        report.append("  2. Analyze during volatile market periods")
        report.append("  3. Consider that markets may be more efficient than hypothesized")
        report.append("  4. Test with different time windows")

    report.append("")
    report.append("="*100)

    # Save report
    report_text = "\n".join(report)
    with open(output_file, 'w') as f:
        f.write(report_text)

    print(f"\n✓ Summary report saved to: {output_file}")

    return report_text


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Run lag detection analysis on asset pairs')

    parser.add_argument('--source', type=str, choices=['yfinance', 'bloomberg'], default='yfinance',
                        help='Data source to use')
    parser.add_argument('--pair', type=str, help='Single pair to analyze (e.g., VALE,GDX)')
    parser.add_argument('--top', type=int, help='Analyze top N pairs from default list')

    # yfinance options
    parser.add_argument('--days', type=int, default=5, help='Days of history (yfinance)')
    parser.add_argument('--interval', type=str, default='1m', help='Data interval (yfinance)')

    # Bloomberg options
    parser.add_argument('--start', type=str, help='Start date YYYY-MM-DD (bloomberg)')
    parser.add_argument('--end', type=str, help='End date YYYY-MM-DD (bloomberg)')
    parser.add_argument('--interval-min', type=int, default=1, help='Interval in minutes (bloomberg)')

    args = parser.parse_args()

    print("\n" + "="*100)
    print("ACTDR - LAG DETECTION ANALYSIS")
    print("Emerging Markets Arbitrage Research Project")
    print("="*100)
    print(f"\nData Source: {args.source.upper()}")

    # Determine which pairs to analyze
    if args.pair:
        # Single pair specified
        ticker1, ticker2 = args.pair.split(',')
        pairs = [(ticker1.strip(), ticker2.strip())]
        print(f"Analyzing single pair: {pairs[0][0]} <--> {pairs[0][1]}")
    elif args.top:
        # Top N pairs
        pairs = TOP_PAIRS[:args.top]
        print(f"Analyzing top {args.top} pairs")
    else:
        # All default pairs
        pairs = TOP_PAIRS
        print(f"Analyzing all {len(TOP_PAIRS)} default pairs")

    print("="*100 + "\n")

    # Prepare kwargs for data fetching
    if args.source == 'yfinance':
        kwargs = {
            'days': args.days,
            'interval': args.interval
        }
    else:  # bloomberg
        if not args.start or not args.end:
            print("✗ Bloomberg source requires --start and --end dates")
            return

        kwargs = {
            'start_date': args.start,
            'end_date': args.end,
            'interval_minutes': args.interval_min
        }

    # Run analysis
    results = analyze_multiple_pairs(pairs, data_source=args.source, **kwargs)

    if len(results) == 0:
        print("\n✗ No successful analyses. Check your data source and tickers.")
        return

    # Generate reports
    print("\n" + "="*100)
    print("GENERATING REPORTS")
    print("="*100)

    # Text summary
    generate_summary_report(results, output_file='lag_analysis_summary.txt')

    # Visual dashboard
    if len(results) > 0:
        print("\nCreating visual dashboard...")
        create_summary_dashboard(results, save_path='lag_analysis_dashboard.png')

    # Save raw results as JSON
    json_results = []
    for r in results:
        # Convert to JSON-serializable format
        json_r = {
            'asset1': r['asset1'],
            'asset2': r['asset2'],
            'correlation': r['pearson']['correlation'],
            'lag_periods': r['cross_correlation']['lag_periods'],
            'leader': r['cross_correlation']['leader'],
            'frequency': r['frequency']
        }
        json_results.append(json_r)

    with open('lag_analysis_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    print("✓ Results saved to lag_analysis_results.json")

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print("\nGenerated files:")
    print("  - lag_analysis_summary.txt (detailed text report)")
    print("  - lag_analysis_dashboard.png (visual summary)")
    print("  - lag_analysis_results.json (raw data)")
    print("  - lag_plots/ (individual pair visualizations)")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()

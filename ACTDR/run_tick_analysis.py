#!/usr/bin/env python3
"""
Tick-Level Lag Detection Analysis
High-frequency analysis using Bloomberg tick data for millisecond-precision lag detection

USAGE:
======

Single trading day analysis:
    python run_tick_analysis.py --pair JD,KWEB --date 2025-01-15

Specific time window (intraday):
    python run_tick_analysis.py --pair JD,KWEB --date 2025-01-15 --start-time "09:30:00" --end-time "10:30:00"

Multiple trading days:
    python run_tick_analysis.py --pair COPX,EPU --start-date 2025-01-13 --end-date 2025-01-17

Batch analysis (top pairs):
    python run_tick_analysis.py --top 5 --date 2025-01-15

Sample data (for testing):
    python run_tick_analysis.py --pair JD,KWEB --date 2025-01-15 --start-time "09:30:00" --end-time "09:45:00" --sample 1000

IMPORTANT:
- Bloomberg Terminal must be running
- Tick data is LARGE - start with small time windows
- Recommended: 1-hour windows for initial testing
- Full trading day (6.5 hours) = millions of ticks
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from lag_detection import LagDetector, convert_lag_to_time
from data_fetcher_bloomberg import BloombergDataFetcher, get_bloomberg_ticker
from lag_visualization import plot_comprehensive_analysis, create_summary_dashboard


# Top pairs for tick analysis (same as bar analysis)
TOP_PAIRS = [
    ('JD', 'KWEB'),      # Showed 15-period Granger lag
    ('COPX', 'EPU'),     # Copper vs Peru
    ('BABA', 'KWEB'),    # High correlation
    ('GDX', 'EPU'),      # Gold vs Peru
    ('PICK', 'VALE'),    # Metals vs Mining
]


def resample_tick_data(tick_data, interval_ms=1000):
    """
    Resample tick data to regular intervals

    Parameters:
    -----------
    tick_data : pd.DataFrame
        Tick data with 'value' column
    interval_ms : int
        Resampling interval in milliseconds (default 1000 = 1 second)

    Returns:
    --------
    pd.DataFrame : Resampled data
    """
    # Resample to regular intervals using last price in each interval
    rule = f'{interval_ms}ms'
    resampled = tick_data['value'].resample(rule).last().ffill()

    return pd.DataFrame({'close': resampled})


def analyze_tick_pair(ticker1, ticker2, start_datetime, end_datetime,
                     resample_ms=None, sample_size=None):
    """
    Run tick-level lag analysis on a pair

    Parameters:
    -----------
    ticker1, ticker2 : str
        Asset tickers
    start_datetime : str or datetime
        Start datetime (YYYY-MM-DD HH:MM:SS)
    end_datetime : str or datetime
        End datetime
    resample_ms : int, optional
        Resample ticks to this interval (milliseconds). If None, uses raw ticks
    sample_size : int, optional
        Limit to first N ticks (for testing)

    Returns:
    --------
    dict : Analysis results
    """
    print("\n" + "="*80)
    print(f"TICK ANALYSIS: {ticker1} <--> {ticker2}")
    print("="*80)
    print(f"Period: {start_datetime} to {end_datetime}")
    if resample_ms:
        print(f"Resampling: {resample_ms}ms intervals")
    if sample_size:
        print(f"Sample size: {sample_size} ticks")
    print("="*80 + "\n")

    # Connect to Bloomberg
    fetcher = BloombergDataFetcher()
    if not fetcher.connect():
        raise Exception("Failed to connect to Bloomberg Terminal")

    try:
        # Convert tickers to Bloomberg format
        bb_ticker1 = get_bloomberg_ticker(ticker1)
        bb_ticker2 = get_bloomberg_ticker(ticker2)

        # Fetch tick data
        print(f"Fetching tick data for {ticker1}...")
        tick_data1 = fetcher.get_tick_data(bb_ticker1, start_datetime, end_datetime)

        print(f"Fetching tick data for {ticker2}...")
        tick_data2 = fetcher.get_tick_data(bb_ticker2, start_datetime, end_datetime)

        if len(tick_data1) == 0 or len(tick_data2) == 0:
            print(f"✗ Insufficient tick data for {ticker1}-{ticker2}")
            return None

        # Sample if requested
        if sample_size:
            tick_data1 = tick_data1.head(sample_size)
            tick_data2 = tick_data2.head(sample_size)
            print(f"Limited to {sample_size} ticks per asset")

        print(f"\n✓ Retrieved {len(tick_data1)} ticks for {ticker1}")
        print(f"✓ Retrieved {len(tick_data2)} ticks for {ticker2}")

        # Resample if requested (recommended for large datasets)
        if resample_ms:
            print(f"\nResampling to {resample_ms}ms intervals...")
            data1 = resample_tick_data(tick_data1, interval_ms=resample_ms)
            data2 = resample_tick_data(tick_data2, interval_ms=resample_ms)
            frequency = f'{resample_ms}ms'
            print(f"✓ Resampled to {len(data1)} data points")
        else:
            # Use raw tick data
            data1 = pd.DataFrame({'close': tick_data1['value']})
            data2 = pd.DataFrame({'close': tick_data2['value']})
            frequency = 'tick'

        # Align timestamps
        aligned_index = data1.index.intersection(data2.index)
        if len(aligned_index) == 0:
            # No exact overlaps - resample both to common grid
            print("\n⚠ No exact timestamp matches, resampling to common grid...")
            combined_index = data1.index.union(data2.index)
            data1 = data1.reindex(combined_index).ffill().bfill()
            data2 = data2.reindex(combined_index).ffill().bfill()
            aligned_index = data1.index
        else:
            data1 = data1.loc[aligned_index]
            data2 = data2.loc[aligned_index]

        print(f"✓ Aligned data: {len(aligned_index)} common timestamps")
        print(f"  Time range: {aligned_index[0]} to {aligned_index[-1]}")

        # Create detector and run analysis
        detector = LagDetector(
            data1['close'], data2['close'],
            ticker1, ticker2,
            frequency=frequency
        )

        # For tick data, use more lags
        max_lag = min(500, len(aligned_index) // 10)

        results = detector.comprehensive_analysis(max_lag=max_lag)

        # Generate visualizations
        plot_dir = f'./tick_plots/{ticker1}_{ticker2}'
        os.makedirs(plot_dir, exist_ok=True)
        plot_comprehensive_analysis(detector, save_dir=plot_dir)

        return results

    finally:
        fetcher.disconnect()


def analyze_multiple_days(ticker1, ticker2, start_date, end_date,
                         trading_start="09:30:00", trading_end="16:00:00",
                         resample_ms=1000):
    """
    Analyze tick data across multiple trading days

    Parameters:
    -----------
    ticker1, ticker2 : str
        Asset tickers
    start_date, end_date : str
        Date range (YYYY-MM-DD)
    trading_start, trading_end : str
        Trading hours (HH:MM:SS)
    resample_ms : int
        Resampling interval in milliseconds

    Returns:
    --------
    list : Results for each day
    """
    all_results = []

    # Parse dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    current = start
    while current <= end:
        # Skip weekends
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        date_str = current.strftime('%Y-%m-%d')
        start_dt = f"{date_str} {trading_start}"
        end_dt = f"{date_str} {trading_end}"

        print(f"\n{'='*80}")
        print(f"ANALYZING: {date_str}")
        print(f"{'='*80}")

        try:
            result = analyze_tick_pair(
                ticker1, ticker2,
                start_dt, end_dt,
                resample_ms=resample_ms
            )

            if result:
                result['date'] = date_str
                all_results.append(result)

        except Exception as e:
            print(f"✗ Error analyzing {date_str}: {e}")
            continue

        current += timedelta(days=1)

    return all_results


def generate_tick_report(results, output_file='tick_analysis_summary.txt'):
    """
    Generate comprehensive summary report for tick analysis

    Parameters:
    -----------
    results : list
        List of analysis results
    output_file : str
        Path to save report
    """
    report = []
    report.append("="*100)
    report.append("TICK-LEVEL LAG DETECTION ANALYSIS - SUMMARY REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*100)
    report.append("")

    report.append(f"Total pairs analyzed: {len(results)}")
    report.append("")

    # Find pairs with detected lag
    pairs_with_lag = []
    for r in results:
        if r['cross_correlation']['lag_periods'] > 0:
            pairs_with_lag.append((
                r['asset1'],
                r['asset2'],
                r['cross_correlation']['lag_periods'],
                r['cross_correlation']['leader'],
                r['pearson']['correlation'],
                r['frequency']
            ))

    report.append("="*100)
    report.append(f"PAIRS WITH DETECTED LAG: {len(pairs_with_lag)}")
    report.append("="*100)
    report.append("")

    if len(pairs_with_lag) > 0:
        for asset1, asset2, lag, leader, corr, freq in sorted(pairs_with_lag, key=lambda x: x[2], reverse=True):
            report.append(f"{asset1} <--> {asset2}")
            report.append(f"  Correlation: {corr:.4f}")
            report.append(f"  Lag: {lag} periods ({freq})")

            # Convert to time
            if 'ms' in freq:
                ms = int(freq.replace('ms', ''))
                lag_ms = lag * ms
                if lag_ms < 1000:
                    time_lag = f"{lag_ms} milliseconds"
                else:
                    time_lag = f"{lag_ms/1000:.2f} seconds"
            else:
                time_lag = f"{lag} ticks"

            report.append(f"  Time lag: {time_lag}")
            report.append(f"  Leader: {leader}")
            report.append("")

    # Detailed results
    report.append("="*100)
    report.append("DETAILED RESULTS BY PAIR")
    report.append("="*100)
    report.append("")

    for i, r in enumerate(results, 1):
        report.append(f"PAIR #{i}: {r['asset1']} <--> {r['asset2']}")
        report.append("-"*100)
        report.append(f"Data Points: {r['data_points']}")
        report.append(f"Frequency: {r['frequency']}")

        if 'date' in r:
            report.append(f"Date: {r['date']}")

        report.append("")

        # Correlation
        report.append("Pearson Correlation:")
        report.append(f"  Coefficient: {r['pearson']['correlation']:.4f}")
        report.append(f"  P-value: {r['pearson']['p_value']:.6f}")
        report.append("")

        # Cross-correlation
        ccf = r['cross_correlation']
        report.append("Cross-Correlation Analysis:")
        report.append(f"  Leader: {ccf['leader']}")
        report.append(f"  Follower: {ccf['follower']}")
        report.append(f"  Lag: {ccf['lag_periods']} periods")

        # Convert to time
        if 'ms' in r['frequency']:
            ms = int(r['frequency'].replace('ms', ''))
            lag_ms = ccf['lag_periods'] * ms
            if lag_ms < 1000:
                report.append(f"  Time Lag: {lag_ms} milliseconds")
            else:
                report.append(f"  Time Lag: {lag_ms/1000:.2f} seconds")

        report.append("")

        # Granger
        granger = r['granger']
        report.append("Granger Causality Test:")
        report.append(f"  Leader: {granger.get('leader', 'N/A')}")
        if 'lag' in granger:
            report.append(f"  Lag: {granger['lag']} periods")
        report.append("")

        report.append("")

    # Recommendations
    report.append("="*100)
    report.append("RESEARCH RECOMMENDATIONS")
    report.append("="*100)
    report.append("")

    if len(pairs_with_lag) > 0:
        report.append("✓ LAG DETECTED at tick/millisecond level!")
        report.append("")
        report.append("Next steps:")
        report.append("  1. Verify lag consistency across multiple days")
        report.append("  2. Calculate theoretical arbitrage profit")
        report.append("  3. Model transaction costs (spread, fees, latency)")
        report.append("  4. Test during different market conditions")
        report.append("  5. Consider building predictive model")
    else:
        report.append("✗ No significant lag detected even at tick level")
        report.append("")
        report.append("Conclusions:")
        report.append("  1. Markets are highly efficient at all frequencies")
        report.append("  2. Any lag is sub-tick (microseconds)")
        report.append("  3. Requires HFT infrastructure to exploit")
        report.append("  4. Document as negative finding in research")

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
    parser = argparse.ArgumentParser(
        description='Run tick-level lag detection analysis using Bloomberg data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single day, full trading hours
  python run_tick_analysis.py --pair JD,KWEB --date 2025-01-15

  # Single day, first hour only (recommended for testing)
  python run_tick_analysis.py --pair JD,KWEB --date 2025-01-15 --start-time "09:30:00" --end-time "10:30:00"

  # Multiple days
  python run_tick_analysis.py --pair COPX,EPU --start-date 2025-01-13 --end-date 2025-01-17

  # Sample data (first 1000 ticks only)
  python run_tick_analysis.py --pair JD,KWEB --date 2025-01-15 --sample 1000

  # Top 3 pairs (be careful - lots of data!)
  python run_tick_analysis.py --top 3 --date 2025-01-15 --start-time "09:30:00" --end-time "10:00:00"
        """
    )

    parser.add_argument('--pair', type=str, help='Single pair to analyze (e.g., JD,KWEB)')
    parser.add_argument('--top', type=int, help='Analyze top N pairs from default list')

    # Date options
    parser.add_argument('--date', type=str, help='Single date to analyze (YYYY-MM-DD)')
    parser.add_argument('--start-date', type=str, help='Start date for multi-day analysis (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for multi-day analysis (YYYY-MM-DD)')

    # Time options
    parser.add_argument('--start-time', type=str, default='09:30:00',
                       help='Start time (HH:MM:SS, default: 09:30:00)')
    parser.add_argument('--end-time', type=str, default='16:00:00',
                       help='End time (HH:MM:SS, default: 16:00:00)')

    # Processing options
    parser.add_argument('--resample-ms', type=int, default=1000,
                       help='Resample to N milliseconds (default: 1000 = 1 second)')
    parser.add_argument('--sample', type=int,
                       help='Limit to first N ticks (for testing)')
    parser.add_argument('--no-resample', action='store_true',
                       help='Use raw tick data (not recommended - very large)')

    args = parser.parse_args()

    print("\n" + "="*100)
    print("ACTDR - TICK-LEVEL LAG DETECTION ANALYSIS")
    print("Bloomberg High-Frequency Data Analysis")
    print("="*100)

    # Determine pairs to analyze
    if args.pair:
        ticker1, ticker2 = args.pair.split(',')
        pairs = [(ticker1.strip(), ticker2.strip())]
        print(f"\nAnalyzing single pair: {pairs[0][0]} <--> {pairs[0][1]}")
    elif args.top:
        pairs = TOP_PAIRS[:args.top]
        print(f"\nAnalyzing top {args.top} pairs")
    else:
        print("\n✗ Error: Must specify --pair or --top")
        parser.print_help()
        return

    # Determine date range
    if args.date:
        # Single day
        single_day = True
        date_str = args.date
    elif args.start_date and args.end_date:
        # Multiple days
        single_day = False
    else:
        print("\n✗ Error: Must specify --date OR (--start-date and --end-date)")
        parser.print_help()
        return

    # Resampling
    resample_ms = None if args.no_resample else args.resample_ms

    if args.no_resample:
        print("\n⚠ WARNING: Using raw tick data (no resampling)")
        print("  This will be VERY large and slow!")
        print("  Recommended: use --resample-ms 1000 (1 second intervals)")
    else:
        print(f"\nResampling: {resample_ms}ms intervals")

    if args.sample:
        print(f"Sample mode: Limited to {args.sample} ticks per asset")

    print("="*100 + "\n")

    # Run analysis
    all_results = []

    for ticker1, ticker2 in pairs:
        try:
            if single_day:
                # Single day analysis
                start_dt = f"{date_str} {args.start_time}"
                end_dt = f"{date_str} {args.end_time}"

                result = analyze_tick_pair(
                    ticker1, ticker2,
                    start_dt, end_dt,
                    resample_ms=resample_ms,
                    sample_size=args.sample
                )

                if result:
                    all_results.append(result)
            else:
                # Multi-day analysis
                day_results = analyze_multiple_days(
                    ticker1, ticker2,
                    args.start_date, args.end_date,
                    args.start_time, args.end_time,
                    resample_ms=resample_ms or 1000
                )
                all_results.extend(day_results)

        except Exception as e:
            print(f"\n✗ Error analyzing {ticker1}-{ticker2}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_results) == 0:
        print("\n✗ No successful analyses. Check Bloomberg connection and data availability.")
        return

    # Generate reports
    print("\n" + "="*100)
    print("GENERATING REPORTS")
    print("="*100)

    generate_tick_report(all_results, output_file='tick_analysis_summary.txt')

    # Visual dashboard
    if len(all_results) > 0:
        print("\nCreating visual dashboard...")
        create_summary_dashboard(all_results, save_path='tick_analysis_dashboard.png')

    # Save JSON
    json_results = []
    for r in all_results:
        json_r = {
            'asset1': r['asset1'],
            'asset2': r['asset2'],
            'correlation': r['pearson']['correlation'],
            'lag_periods': r['cross_correlation']['lag_periods'],
            'leader': r['cross_correlation']['leader'],
            'frequency': r['frequency']
        }
        if 'date' in r:
            json_r['date'] = r['date']
        json_results.append(json_r)

    with open('tick_analysis_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    print("✓ Results saved to tick_analysis_results.json")

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print("\nGenerated files:")
    print("  - tick_analysis_summary.txt (detailed text report)")
    print("  - tick_analysis_dashboard.png (visual summary)")
    print("  - tick_analysis_results.json (raw data)")
    print("  - tick_plots/ (individual pair visualizations)")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()

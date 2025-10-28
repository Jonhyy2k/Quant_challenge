#!/usr/bin/env python3
"""

Specific time window (intraday):
    python run_tick_analysis.py --pair JD,KWEB --date 2025-01-15 --start-time "09:30:00" --end-time "10:30:00"
corre primeiro:
emerging_markets_correlation.py

vai dar o csv

Depois corre esta

python run_tick_analysis.py --use-csv --min-correlation 0.70 \
    --start-date 2024-10-01 --end-date 2024-12-31
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import os
import json
import warnings

warnings.filterwarnings("ignore")

from lag_detection import LagDetector, convert_lag_to_time
from data_fetcher_bloomberg import BloombergDataFetcher, get_bloomberg_ticker
from lag_visualization import plot_comprehensive_analysis, create_summary_dashboard

TOP_PAIRS = [
    ("JD", "KWEB"),
    ("COPX", "EPU"),
    ("BABA", "KWEB"),
    ("GDX", "EPU"),
    ("PICK", "VALE"),
]


def load_pairs_from_csv(csv_path="em_pairs_detailed.csv", min_correlation=0.65, max_pairs=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    df_filtered = df[df["abs_correlation"] >= min_correlation].copy()
    df_filtered = df_filtered.sort_values("abs_correlation", ascending=False)

    if max_pairs:
        df_filtered = df_filtered.head(max_pairs)

    pairs = [(row["asset1"], row["asset2"]) for _, row in df_filtered.iterrows()]

    print(f"\nLoaded {len(pairs)} pairs from {csv_path}")
    print(f"Correlation threshold: >= {min_correlation}")
    if max_pairs:
        print(f"Limited to top {max_pairs} pairs")

    return pairs


def resample_tick_data(tick_data, interval_ms=1):
    rule = f"{interval_ms}ms"
    resampled = tick_data["value"].resample(rule).last().ffill()
    return pd.DataFrame({"close": resampled})


def analyze_tick_pair(
    ticker1, ticker2, start_datetime, end_datetime, resample_ms=None, sample_size=None
):
    print("\n" + "=" * 80)
    print(f"TICK ANALYSIS: {ticker1} <--> {ticker2}")
    print("=" * 80)
    print(f"Period: {start_datetime} to {end_datetime}")
    if resample_ms:
        print(f"Resampling: {resample_ms}ms intervals")
    if sample_size:
        print(f"Sample size: {sample_size} ticks")
    print("=" * 80 + "\n")

    fetcher = BloombergDataFetcher()
    if not fetcher.connect():
        raise Exception("Failed to connect to Bloomberg Terminal")

    try:
        bb_ticker1 = get_bloomberg_ticker(ticker1)
        bb_ticker2 = get_bloomberg_ticker(ticker2)

        print(f"Fetching tick data for {ticker1}...")
        tick_data1 = fetcher.get_tick_data(bb_ticker1, start_datetime, end_datetime)

        print(f"Fetching tick data for {ticker2}...")
        tick_data2 = fetcher.get_tick_data(bb_ticker2, start_datetime, end_datetime)

        if len(tick_data1) == 0 or len(tick_data2) == 0:
            print(f"✗ Insufficient tick data for {ticker1}-{ticker2}")
            return None

        if sample_size:
            tick_data1 = tick_data1.head(sample_size)
            tick_data2 = tick_data2.head(sample_size)
            print(f"Limited to {sample_size} ticks per asset")

        print(f"\n✓ Retrieved {len(tick_data1)} ticks for {ticker1}")
        print(f"✓ Retrieved {len(tick_data2)} ticks for {ticker2}")

        if resample_ms:
            print(f"\nResampling to {resample_ms}ms intervals...")
            data1 = resample_tick_data(tick_data1, interval_ms=resample_ms)
            data2 = resample_tick_data(tick_data2, interval_ms=resample_ms)
            frequency = f"{resample_ms}ms"
            print(f"✓ Resampled to {len(data1)} data points")
        else:
            data1 = pd.DataFrame({"close": tick_data1["value"]})
            data2 = pd.DataFrame({"close": tick_data2["value"]})
            frequency = "tick"

        aligned_index = data1.index.intersection(data2.index)
        if len(aligned_index) == 0:
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

        detector = LagDetector(
            data1["close"], data2["close"], ticker1, ticker2, frequency=frequency
        )

        max_lag = min(500, len(aligned_index) // 10)
        results = detector.comprehensive_analysis(max_lag=max_lag)

        plot_dir = f"./tick_plots/{ticker1}_{ticker2}"
        os.makedirs(plot_dir, exist_ok=True)
        plot_comprehensive_analysis(detector, save_dir=plot_dir)

        return results

    finally:
        fetcher.disconnect()


def analyze_multiple_days(
    ticker1,
    ticker2,
    start_date,
    end_date,
    trading_start="09:30:00",
    trading_end="16:00:00",
    resample_ms=1,
):
    all_results = []
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    current = start
    while current <= end:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        date_str = current.strftime("%Y-%m-%d")
        start_dt = f"{date_str} {trading_start}"
        end_dt = f"{date_str} {trading_end}"

        print(f"\n{'=' * 80}")
        print(f"ANALYZING: {date_str}")
        print(f"{'=' * 80}")

        try:
            result = analyze_tick_pair(
                ticker1, ticker2, start_dt, end_dt, resample_ms=resample_ms
            )

            if result:
                result["date"] = date_str
                all_results.append(result)

        except Exception as e:
            print(f"✗ Error analyzing {date_str}: {e}")
            continue

        current += timedelta(days=1)

    return all_results


def generate_tick_report(results, output_file="tick_analysis_summary.txt"):
    report = []
    report.append("=" * 100)
    report.append("TICK-LEVEL LAG DETECTION ANALYSIS - SUMMARY REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 100)
    report.append("")

    report.append(f"Total pairs analyzed: {len(results)}")
    report.append("")

    pairs_with_lag = []
    for r in results:
        if r["cross_correlation"]["lag_periods"] > 0:
            pairs_with_lag.append(
                (
                    r["asset1"],
                    r["asset2"],
                    r["cross_correlation"]["lag_periods"],
                    r["cross_correlation"]["leader"],
                    r["pearson"]["correlation"],
                    r["frequency"],
                )
            )

    report.append("=" * 100)
    report.append(f"PAIRS WITH DETECTED LAG: {len(pairs_with_lag)}")
    report.append("=" * 100)
    report.append("")

    if len(pairs_with_lag) > 0:
        for asset1, asset2, lag, leader, corr, freq in sorted(
            pairs_with_lag, key=lambda x: x[2], reverse=True
        ):
            report.append(f"{asset1} <--> {asset2}")
            report.append(f"  Correlation: {corr:.4f}")
            report.append(f"  Lag: {lag} periods ({freq})")

            if "ms" in freq:
                ms = int(freq.replace("ms", ""))
                lag_ms = lag * ms
                if lag_ms < 1000:
                    time_lag = f"{lag_ms} milliseconds"
                else:
                    time_lag = f"{lag_ms / 1000:.2f} seconds"
            else:
                time_lag = f"{lag} ticks"

            report.append(f"  Time lag: {time_lag}")
            report.append(f"  Leader: {leader}")
            report.append("")

    report.append("=" * 100)
    report.append("DETAILED RESULTS BY PAIR")
    report.append("=" * 100)
    report.append("")

    for i, r in enumerate(results, 1):
        report.append(f"PAIR #{i}: {r['asset1']} <--> {r['asset2']}")
        report.append("-" * 100)
        report.append(f"Data Points: {r['data_points']}")
        report.append(f"Frequency: {r['frequency']}")

        if "date" in r:
            report.append(f"Date: {r['date']}")

        report.append("")

        report.append("Pearson Correlation:")
        report.append(f"  Coefficient: {r['pearson']['correlation']:.4f}")
        report.append(f"  P-value: {r['pearson']['p_value']:.6f}")
        report.append("")

        ccf = r["cross_correlation"]
        report.append("Cross-Correlation Analysis:")
        report.append(f"  Leader: {ccf['leader']}")
        report.append(f"  Follower: {ccf['follower']}")
        report.append(f"  Lag: {ccf['lag_periods']} periods")

        if "ms" in r["frequency"]:
            ms = int(r["frequency"].replace("ms", ""))
            lag_ms = ccf["lag_periods"] * ms
            if lag_ms < 1000:
                report.append(f"  Time Lag: {lag_ms} milliseconds")
            else:
                report.append(f"  Time Lag: {lag_ms / 1000:.2f} seconds")

        report.append("")

        granger = r["granger"]
        report.append("Granger Causality Test:")
        report.append(f"  Leader: {granger.get('leader', 'N/A')}")
        if "lag" in granger:
            report.append(f"  Lag: {granger['lag']} periods")
        report.append("")
        report.append("")

    report.append("=" * 100)
    report.append("RESEARCH RECOMMENDATIONS")
    report.append("=" * 100)
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
    report.append("=" * 100)

    report_text = "\n".join(report)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n✓ Summary report saved to: {output_file}")

    return report_text


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Run tick-level lag detection analysis using Bloomberg data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog,
    )

    parser.add_argument(
        "--pair", type=str, help="Single pair to analyze (e.g., JD,KWEB)"
    )
    parser.add_argument("--top", type=int, help="Analyze top N pairs from default list")
    parser.add_argument(
        "--all-pairs", action="store_true", help="Analyze all pairs from default list"
    )
    parser.add_argument(
        "--use-csv",
        action="store_true",
        help="Load pairs from em_pairs_detailed.csv instead of default list",
    )
    parser.add_argument(
        "--min-correlation",
        type=float,
        default=0.65,
        help="Minimum correlation threshold when using --use-csv (default: 0.65)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        help="Maximum number of pairs to analyze from CSV (takes top N by correlation)",
    )

    # Date options
    parser.add_argument("--date", type=str, help="Single date to analyze (YYYY-MM-DD)")
    parser.add_argument(
        "--start-date", type=str, help="Start date for multi-day analysis (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, help="End date for multi-day analysis (YYYY-MM-DD)"
    )

    # Time options
    parser.add_argument(
        "--start-time",
        type=str,
        default="09:30:00",
        help="Start time (HH:MM:SS, default: 09:30:00)",
    )
    parser.add_argument(
        "--end-time",
        type=str,
        default="16:00:00",
        help="End time (HH:MM:SS, default: 16:00:00)",
    )

    # Processing options
    parser.add_argument(
        "--resample-ms",
        type=int,
        default=1000,
        help="Resample to N milliseconds (default: 1000 = 1 second)",
    )
    parser.add_argument(
        "--sample", type=int, help="Limit to first N ticks (for testing)"
    )
    parser.add_argument(
        "--no-resample",
        action="store_true",
        help="Use raw tick data (not recommended - very large)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 100)
    print("ACTDR - TICK-LEVEL LAG DETECTION ANALYSIS")
    print("Bloomberg High-Frequency Data Analysis")
    print("=" * 100)

    if args.pair:
        ticker1, ticker2 = args.pair.split(",")
        pairs = [(ticker1.strip(), ticker2.strip())]
        print(f"\nAnalyzing single pair: {pairs[0][0]} <--> {pairs[0][1]}")
    elif args.use_csv:
        pairs = load_pairs_from_csv(
            csv_path="em_pairs_detailed.csv",
            min_correlation=args.min_correlation,
            max_pairs=args.max_pairs,
        )
    elif args.all_pairs:
        pairs = TOP_PAIRS
        print(f"\nAnalyzing all {len(pairs)} pairs from default list")
    elif args.top:
        pairs = TOP_PAIRS[: args.top]
        print(f"\nAnalyzing top {args.top} pairs")
    else:
        print("\n✗ Error: Must specify --pair, --top, --all-pairs, or --use-csv")
        parser.print_help()
        return

    if args.date:
        single_day = True
        date_str = args.date
    elif args.start_date and args.end_date:
        single_day = False
    else:
        print("\n✗ Error: Must specify --date OR (--start-date and --end-date)")
        parser.print_help()
        return

    resample_ms = None if args.no_resample else args.resample_ms

    if args.no_resample:
        print("\n⚠ WARNING: Using raw tick data (no resampling)")
        print("  This will be VERY large and slow!")
        print("  Recommended: use --resample-ms 1000 (1 second intervals)")
    else:
        print(f"\nResampling: {resample_ms}ms intervals")

    if args.sample:
        print(f"Sample mode: Limited to {args.sample} ticks per asset")

    print("=" * 100 + "\n")

    all_results = []

    for ticker1, ticker2 in pairs:
        try:
            if single_day:
                start_dt = f"{date_str} {args.start_time}"
                end_dt = f"{date_str} {args.end_time}"

                result = analyze_tick_pair(
                    ticker1,
                    ticker2,
                    start_dt,
                    end_dt,
                    resample_ms=resample_ms,
                    sample_size=args.sample,
                )

                if result:
                    all_results.append(result)
            else:
                day_results = analyze_multiple_days(
                    ticker1,
                    ticker2,
                    args.start_date,
                    args.end_date,
                    args.start_time,
                    args.end_time,
                    resample_ms=resample_ms or 1000,
                )
                all_results.extend(day_results)

        except Exception as e:
            print(f"\n✗ Error analyzing {ticker1}-{ticker2}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if len(all_results) == 0:
        print(
            "\n✗ No successful analyses. Check Bloomberg connection and data availability."
        )
        return

    print("\n" + "=" * 100)
    print("GENERATING REPORTS")
    print("=" * 100)

    generate_tick_report(all_results, output_file="tick_analysis_summary.txt")

    if len(all_results) > 0:
        print("\nCreating visual dashboard...")
        create_summary_dashboard(all_results, save_path="tick_analysis_dashboard.png")

    json_results = []
    for r in all_results:
        json_r = {
            "asset1": r["asset1"],
            "asset2": r["asset2"],
            "correlation": float(r["pearson"]["correlation"]),
            "lag_periods": int(r["cross_correlation"]["lag_periods"]),
            "leader": r["cross_correlation"]["leader"],
            "frequency": r["frequency"],
        }
        if "date" in r:
            json_r["date"] = r["date"]
        json_results.append(json_r)

    with open("tick_analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)

    print("✓ Results saved to tick_analysis_results.json")

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    print("\nGenerated files:")
    print("  - tick_analysis_summary.txt (detailed text report)")
    print("  - tick_analysis_dashboard.png (visual summary)")
    print("  - tick_analysis_results.json (raw data)")
    print("  - tick_plots/ (individual pair visualizations)")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()

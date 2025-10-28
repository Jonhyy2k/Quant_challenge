# ACTDR - Setup Guide

Complete setup instructions for running the tick-level lag analysis.

## Prerequisites

1. **Python 3.8 or higher**
   ```bash
   python --version
   ```

2. **Bloomberg Terminal** (must be running)
   - Open Bloomberg Terminal application
   - Ensure you're logged in
   - Keep it running in the background while analysis executes

## Installation Steps

### Step 1: Install Python Dependencies

Navigate to the ACTDR directory and install required packages:

```bash
cd /path/to/ACTDR
pip install -r requirements.txt
```

**Note:** If `blpapi` installation fails (common on some systems), follow these alternative steps:

#### Bloomberg API Installation (if needed):

**Windows:**
```bash
python -m pip install --index-url=https://bcms.bloomberg.com/pip/simple blpapi
```

**macOS/Linux:**
```bash
pip install --index-url=https://bcms.bloomberg.com/pip/simple blpapi
```

If issues persist, download and install manually from:
https://www.bloomberg.com/professional/support/api-library/

### Step 2: Verify Installation

Run the test script to ensure everything is set up correctly:

```bash
python test_installation.py
```

This will check:
- All Python packages are installed
- Bloomberg Terminal connection works
- Sample data can be fetched

### Step 3: Run Your First Analysis

Once installation is verified, you can run the tick analysis:

#### Quick Test (Single Pair, 1 Hour):
```bash
python run_tick_analysis.py --pair JD,KWEB --date 2025-01-15 \
  --start-time "09:30:00" --end-time "10:30:00"
```

#### Full Analysis (All Pairs, Full Week):
```bash
python run_tick_analysis.py --all-pairs \
  --start-date 2025-01-13 --end-date 2025-01-20
```

#### Extended Analysis (3 Months):
```bash
python run_tick_analysis.py --all-pairs \
  --start-date 2024-10-01 --end-date 2024-12-31
```

## Important Files

### Files You Need:
- `run_tick_analysis.py` - Main analysis script
- `lag_detection.py` - Core lag detection algorithms
- `data_fetcher_bloomberg.py` - Bloomberg data connector
- `lag_visualization.py` - Plotting and visualization
- `requirements.txt` - Python dependencies

### Files Generated After Running:
- `tick_analysis_summary.txt` - Detailed text report
- `tick_analysis_results.json` - Machine-readable results
- `tick_analysis_dashboard.png` - Visual summary
- `tick_plots/` - Individual pair visualizations

## Troubleshooting

### "Failed to connect to Bloomberg Terminal"
- Ensure Bloomberg Terminal is running and you're logged in
- Try restarting Bloomberg Terminal
- Check Bloomberg Terminal has API access enabled

### "No data returned"
- Verify the date range includes trading days (not weekends)
- Check the ticker symbols are correct
- Ensure you have Bloomberg data subscription for those assets

### "Out of Memory" errors
- Reduce time window (use `--start-time` and `--end-time`)
- Use larger resampling interval (`--resample-ms 1000` or higher)
- Analyze fewer pairs at once (use `--pair` instead of `--all-pairs`)

### Import errors
- Run `pip install -r requirements.txt` again
- Check Python version is 3.8+
- Try creating a new virtual environment

## What to Expect

### Data Volume:
- **1 hour of tick data**: ~500 MB per pair
- **1 full trading day**: ~3-5 GB per pair
- **1 week (all 5 pairs)**: ~100+ GB total

### Processing Time:
- **Single pair, 1 hour**: ~5-10 minutes
- **Single pair, 1 day**: ~30-60 minutes
- **All pairs, 1 week**: ~4-6 hours
- **All pairs, 3 months**: ~2-3 days

### Recommendations:
1. Start with single pair, 1-hour window for testing
2. Once confirmed working, expand to full days
3. For multi-month analysis, run overnight or over weekend
4. Monitor disk space (results can be large)

## Next Steps After Setup

Once analysis completes, you'll have:
1. Summary report with detected lags
2. Correlation coefficients
3. Leader/follower relationships
4. Visualizations for each pair

Review the `tick_analysis_summary.txt` file for key findings.

## Support

For issues:
1. Check TROUBLESHOOTING section above
2. Review Bloomberg API documentation
3. Verify all prerequisites are met
4. Check Python package versions match requirements.txt

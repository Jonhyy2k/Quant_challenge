# Quick Start Guide - ACTDR Lag Detection

Get started in 5 minutes with lag detection analysis.

---

## Step 1: Install Dependencies (2 minutes)

```bash
cd /home/joaop/PyResearch/ITAU/ACTDR

# Install required packages
pip install -r requirements.txt
```

**Note:** Skip `blpapi` if you don't have Bloomberg Terminal - yfinance will work fine for testing.

---

## Step 2: Run Your First Analysis (3 minutes)

### Option A: Quick Test (Single Pair)

```bash
# Analyze VALE vs GDX with free data
python run_lag_analysis.py --source yfinance --pair VALE,GDX --days 5
```

**What this does:**
- Fetches 5 days of 1-minute data for VALE and GDX
- Runs correlation and lag detection
- Generates 4 visualization charts
- Creates summary report

**Output files:**
- `lag_analysis_summary.txt` - Detailed results
- `lag_plots/VALE_GDX/` - Charts
- `lag_analysis_dashboard.png` - Visual summary

### Option B: Top 5 Pairs

```bash
# Analyze top 5 most promising pairs
python run_lag_analysis.py --source yfinance --top 5 --days 7
```

Analyzes:
1. COPX <--> EPU (Copper vs Peru)
2. PICK <--> VALE (Metals vs Mining)
3. PICK <--> DEM (Metals vs EM)
4. GDX <--> EPU (Gold vs Peru)
5. COPX <--> VALE (Copper vs Vale)

**Runtime:** ~5-10 minutes

---

## Step 3: Review Results

### Check the Summary Report

```bash
cat lag_analysis_summary.txt
```

Look for:
- **Correlation**: Should be >0.6 for our pairs
- **Leader**: Which asset moves first
- **Lag**: How many periods of delay
- **Time Lag**: Converted to seconds/minutes

### View the Charts

```bash
# On Linux/Mac
open lag_plots/VALE_GDX/*.png

# Or navigate to folder and view manually
```

Charts show:
1. **Price comparison** - Do they move together?
2. **Returns scatter** - Correlation visualization
3. **Cross-correlation** - Lag detection (key chart!)
4. **Rolling correlation** - Stability over time

---

## Understanding Your Results

### ✓ Good Signs (Lag Detected)

```
Cross-Correlation Analysis:
  Leader: COPX
  Follower: EPU
  Lag: 15 periods (1min)
  Time Lag: 15 minutes
```

**This means:** COPX moves first, EPU follows 15 minutes later.

**Interpretation:** Potential arbitrage window!

### ✗ No Lag Detected

```
Cross-Correlation Analysis:
  Leader: No clear leader
  Lag: 0 periods
```

**This means:** Assets move simultaneously (efficient market).

**Next steps:**
- Try higher frequency data (Bloomberg)
- Test different time periods
- Try different pairs

---

## Next Steps Based on Results

### If You Found Lag:

1. **Verify with Bloomberg** (if available)
   ```bash
   python run_lag_analysis.py \
       --source bloomberg \
       --pair COPX,EPU \
       --start 2025-01-15 \
       --end 2025-01-20 \
       --interval-min 1
   ```

2. **Analyze more pairs**
   ```bash
   python run_lag_analysis.py --source yfinance --top 10 --days 7
   ```

3. **Document findings** in your research

### If No Lag Found:

1. **Try different data frequency**
   ```bash
   # 5-minute bars (can get more history)
   python run_lag_analysis.py --pair VALE,GDX --interval 5m --days 30
   ```

2. **Test different pairs**
   - Check `em_pairs_detailed.csv` for all 728 pairs
   - Try pairs with lower correlation (0.6-0.7 range)

3. **Use Bloomberg tick data** (if available)

---

## Common Commands Cheat Sheet

```bash
# Single pair, 7 days, 1-minute data
python run_lag_analysis.py --pair VALE,GDX --days 7

# Top 3 pairs, 5-minute data (longer history)
python run_lag_analysis.py --top 3 --interval 5m --days 30

# Specific commodity-EM pair
python run_lag_analysis.py --pair COPX,EPU --days 7

# Brazilian bank pair
python run_lag_analysis.py --pair ITUB,BBD --days 7

# Chinese tech pair
python run_lag_analysis.py --pair BABA,JD --days 7

# Bloomberg (1-week analysis)
python run_lag_analysis.py \
    --source bloomberg \
    --start 2025-01-13 \
    --end 2025-01-20 \
    --top 5
```

---

## Troubleshooting

### "No data retrieved"

**Problem:** Ticker might be invalid or no recent trading

```bash
# Test if ticker works
python -c "import yfinance as yf; print(yf.Ticker('VALE').history(period='5d'))"
```

**Solution:** Try different ticker or check spelling

### "Insufficient data for analysis"

**Problem:** Not enough overlap between two assets

**Solutions:**
- Increase days: `--days 7` instead of `--days 5`
- Use 5-minute interval: `--interval 5m` (allows more history)
- Check if both tickers are actively traded

### "Module not found"

**Problem:** Dependencies not installed

```bash
pip install -r requirements.txt
```

---

## What to Look For in Results

### Key Metrics

1. **Pearson Correlation** (0.6-1.0)
   - How strongly assets move together
   - We pre-filtered for >0.6, so should be good

2. **Cross-Correlation Lag** (0-60 periods)
   - Most important metric!
   - Non-zero lag = potential arbitrage
   - Look at cross-correlation plot

3. **Granger Causality** (Leader identified)
   - Confirms which asset predicts the other
   - Should align with cross-correlation

4. **Response Time** (periods)
   - Average time for asset 2 to react
   - Smaller = harder to capture
   - Larger = better arbitrage window

### Interpreting Time Lags

```
1-minute data:
  - 5 periods = 5 minutes
  - 15 periods = 15 minutes
  - 60 periods = 1 hour

5-minute data:
  - 3 periods = 15 minutes
  - 12 periods = 1 hour
```

---

## Example Workflow

Here's a complete research workflow:

```bash
# 1. Quick test to validate methodology
python run_lag_analysis.py --pair VALE,GDX --days 5

# 2. If promising, analyze all commodity-EM pairs
python run_lag_analysis.py --top 5 --days 7

# 3. Get more history with 5-min data
python run_lag_analysis.py --top 5 --interval 5m --days 30

# 4. If you have Bloomberg, get high-quality data
python run_lag_analysis.py \
    --source bloomberg \
    --top 10 \
    --start 2025-01-13 \
    --end 2025-01-20

# 5. Review all results
cat lag_analysis_summary.txt
```

---

## Where to Go From Here

### For Research Paper:

1. ✓ Run analysis on top 10 pairs
2. ✓ Document which pairs show lag
3. ✓ Include visualizations in paper
4. ✓ Discuss why some pairs have lag and others don't
5. ✓ Calculate theoretical arbitrage profit (even if negative after costs)

### For Deeper Analysis:

1. Read `STRATEGY_INTENTIONS.txt` for full research plan
2. Review `em_correlation_report.txt` for all 320 EM pairs
3. Explore `em_pairs_detailed.csv` for custom pair selection
4. Modify `lag_detection.py` to add your own statistical tests

### For Bloomberg Users:

1. Test Bloomberg connection:
   ```bash
   python data_fetcher_bloomberg.py
   ```

2. Run tick-level analysis (if available):
   - Modify `data_fetcher_bloomberg.py` to use `get_tick_data()`
   - Provides millisecond-level precision

---

## Getting Help

- **README.md** - Full documentation
- **STRATEGY_INTENTIONS.txt** - Research objectives
- **Code comments** - Detailed explanations in each module

---

## Success Criteria

Remember: This is a RESEARCH project, not a trading system.

✓ **Success** = Documented findings about lag (even if negative)
✓ **Success** = Understanding EM market efficiency
✓ **Success** = Quantifying arbitrage opportunity (even if unprofitable)

✗ **Not Required** = Profitable trading strategy
✗ **Not Required** = Deployable system

The goal is to TEST THE HYPOTHESIS and learn!

---

**Ready?** Run your first analysis now:

```bash
python run_lag_analysis.py --pair VALE,GDX --days 5
```

Good luck with your research! 🚀

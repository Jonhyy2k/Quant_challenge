# Tick-Level Analysis Guide

## Quick Start - Millisecond Precision Lag Detection

---

## Prerequisites

1. ✅ Bloomberg Terminal **RUNNING** (not just installed)
2. ✅ Logged into Bloomberg Terminal
3. ✅ Python packages installed (`blpapi`, etc.)

---

## Step 1: Test Connection (30 seconds)

```bash
cd /home/joaop/PyResearch/ITAU/ACTDR

# Test Bloomberg connection
python3 data_fetcher_bloomberg.py
```

**Expected:** `✓ Bloomberg connection successful!`

**If fails:**
- Check Terminal is running
- Wait 2-3 minutes for Terminal to fully load
- Try again

---

## Step 2: Your First Tick Analysis (5 minutes)

### **Recommended: Start Small (First 15 minutes of trading)**

```bash
# Analyze JD vs KWEB - first 15 minutes of Jan 15, 2025
python3 run_tick_analysis.py \
    --pair JD,KWEB \
    --date 2025-01-15 \
    --start-time "09:30:00" \
    --end-time "09:45:00" \
    --resample-ms 1000
```

**What this does:**
- Fetches tick data for JD and KWEB
- Only first 15 minutes of trading (manageable data size)
- Resamples to 1-second intervals (1000ms)
- Runs lag detection
- Creates charts

**Runtime:** ~2-5 minutes

---

## Step 3: Full Trading Day (If Step 2 Works)

```bash
# Full trading day (6.5 hours)
python3 run_tick_analysis.py \
    --pair JD,KWEB \
    --date 2025-01-15 \
    --resample-ms 1000
```

**Default trading hours:** 09:30:00 to 16:00:00 (US market)

**Runtime:** ~10-20 minutes (lots of data!)

---

## Command Reference

### **Basic Usage**

```bash
python3 run_tick_analysis.py --pair TICKER1,TICKER2 --date YYYY-MM-DD
```

### **All Options**

```bash
--pair JD,KWEB              # Single pair to analyze
--top 5                     # Analyze top N pairs

--date 2025-01-15           # Single date
--start-date 2025-01-13     # Multi-day analysis start
--end-date 2025-01-17       # Multi-day analysis end

--start-time "09:30:00"     # Trading start (default: 09:30:00)
--end-time "16:00:00"       # Trading end (default: 16:00:00)

--resample-ms 1000          # Resample to N milliseconds (default: 1000)
--no-resample               # Use raw ticks (HUGE data, not recommended)
--sample 1000               # Limit to first N ticks (testing only)
```

---

## Example Workflows

### **Example 1: Quick Test (First Hour)**

```bash
# Test with 1 hour of data
python3 run_tick_analysis.py \
    --pair JD,KWEB \
    --date 2025-01-15 \
    --start-time "09:30:00" \
    --end-time "10:30:00"
```

### **Example 2: Market Open (High Volatility)**

```bash
# First 30 minutes (most volatile period)
python3 run_tick_analysis.py \
    --pair COPX,EPU \
    --date 2025-01-15 \
    --start-time "09:30:00" \
    --end-time "10:00:00" \
    --resample-ms 500
```

### **Example 3: Multiple Days**

```bash
# Analyze full week
python3 run_tick_analysis.py \
    --pair JD,KWEB \
    --start-date 2025-01-13 \
    --end-date 2025-01-17 \
    --start-time "09:30:00" \
    --end-time "10:00:00"
```

### **Example 4: Higher Precision (500ms intervals)**

```bash
# Half-second intervals
python3 run_tick_analysis.py \
    --pair BABA,KWEB \
    --date 2025-01-15 \
    --start-time "14:00:00" \
    --end-time "15:00:00" \
    --resample-ms 500
```

### **Example 5: Top 3 Pairs (Small Window)**

```bash
# Batch analysis - first 30 min only
python3 run_tick_analysis.py \
    --top 3 \
    --date 2025-01-15 \
    --start-time "09:30:00" \
    --end-time "10:00:00"
```

---

## Understanding Resample Intervals

| Resample | Precision | Data Size | Use Case |
|----------|-----------|-----------|----------|
| 5000ms (5s) | Low | Small | Quick testing |
| 1000ms (1s) | Medium | Medium | **Recommended starting point** |
| 500ms | High | Large | Detailed analysis |
| 100ms | Very High | Very Large | HFT research |
| No resample | Raw ticks | MASSIVE | Expert only |

**Rule of thumb:** Start with 1000ms (1 second), then go higher precision if needed.

---

## Output Files

After running, you'll get:

```
ACTDR/
├── tick_analysis_summary.txt        # Detailed statistical report
├── tick_analysis_dashboard.png      # Visual summary
├── tick_analysis_results.json       # Machine-readable data
└── tick_plots/
    └── JD_KWEB/
        ├── JD_KWEB_prices_*.png     # Price comparison
        ├── JD_KWEB_scatter_*.png    # Returns correlation
        ├── JD_KWEB_ccf_*.png        # Cross-correlation ⭐
        └── JD_KWEB_rolling_*.png    # Rolling correlation
```

---

## What to Look For

### ✓ Positive Results (Lag Detected)

```
Cross-Correlation Analysis:
  Leader: JD
  Follower: KWEB
  Lag: 50 periods (1000ms)
  Time Lag: 50 seconds
```

**This means:** JD moves first, KWEB follows 50 seconds later!

### ✗ No Lag

```
Cross-Correlation Analysis:
  Leader: No clear leader
  Lag: 0 periods
```

**This means:** Even at millisecond level, no lag detected (very efficient market).

---

## Top Pairs to Analyze

Pre-configured pairs (use with `--top N`):

1. **JD <--> KWEB** ⭐ (showed 15-period lag in 1-min data)
2. **COPX <--> EPU** (Copper vs Peru, commodity-dependent)
3. **BABA <--> KWEB** (High correlation: 0.88)
4. **GDX <--> EPU** (Gold vs Peru)
5. **PICK <--> VALE** (Metals vs Mining)

---

## Troubleshooting

### "Failed to connect to Bloomberg"

**Solution:**
1. Open Bloomberg Terminal
2. Wait for full login (2-3 minutes)
3. Try command again

### "No data retrieved"

**Possible causes:**
- Date is weekend (markets closed)
- Date is too old (check Bloomberg subscription)
- Time range outside trading hours
- Asset not traded that day (earnings halt, etc.)

**Solution:**
- Use recent weekday (Mon-Fri)
- Check: Jan 15, 2025 was a Wednesday (should work)

### "Out of memory" or very slow

**Cause:** Too much tick data

**Solution:**
- Reduce time window: `--end-time "10:00:00"` instead of full day
- Increase resample: `--resample-ms 5000` (5 seconds)
- Use sample mode: `--sample 10000`

### Bloomberg Terminal not responding

**Solution:**
1. Close Terminal completely
2. Restart Terminal
3. Wait for full login
4. Try again

---

## Recommended Analysis Strategy

### Phase 1: Quick Validation (10 minutes)

```bash
# Test connection
python3 data_fetcher_bloomberg.py

# Quick 15-min analysis
python3 run_tick_analysis.py \
    --pair JD,KWEB \
    --date 2025-01-15 \
    --start-time "09:30:00" \
    --end-time "09:45:00"
```

### Phase 2: Full Day Analysis (30 minutes)

```bash
# Full trading day on best pair
python3 run_tick_analysis.py \
    --pair JD,KWEB \
    --date 2025-01-15
```

### Phase 3: Multi-Day Validation (1 hour)

```bash
# Analyze full week
python3 run_tick_analysis.py \
    --pair JD,KWEB \
    --start-date 2025-01-13 \
    --end-date 2025-01-17 \
    --start-time "09:30:00" \
    --end-time "11:00:00"
```

### Phase 4: High Precision (If lag found)

```bash
# 100ms precision on specific time window
python3 run_tick_analysis.py \
    --pair JD,KWEB \
    --date 2025-01-15 \
    --start-time "10:00:00" \
    --end-time "11:00:00" \
    --resample-ms 100
```

---

## Data Size Estimates

| Time Window | Resample | Approx Data Points | RAM Usage | Runtime |
|-------------|----------|-------------------|-----------|---------|
| 15 min | 1000ms | ~900 | <100MB | 2-3 min |
| 1 hour | 1000ms | ~3,600 | ~200MB | 5-10 min |
| Full day | 1000ms | ~23,400 | ~500MB | 10-20 min |
| 1 hour | 100ms | ~36,000 | ~1GB | 15-30 min |
| Full day | Raw ticks | ~1-5 million | 5-10GB | 1+ hour |

---

## Expected Results

### If Lag is Detected ✓

**You'll see:**
- Non-zero lag in cross-correlation
- Clear leader/follower relationship
- Granger causality significance
- Consistent lag across time periods

**What it means:**
- ✓ Arbitrage opportunity exists (theoretically)
- ✓ Measurable inefficiency
- ✓ Good research finding!

**Next steps:**
- Calculate profit per trade
- Model transaction costs
- Test across multiple days
- Document for research paper

### If No Lag ✗

**You'll see:**
- Zero lag in cross-correlation
- "No clear leader"
- Simultaneous movement

**What it means:**
- ✗ Market is efficient even at millisecond level
- ✗ No exploitable arbitrage
- ✓ Still valid research (negative result)

**Next steps:**
- Document findings
- Try different market conditions (volatile days)
- Test other pair types
- Conclude: markets efficient at tested frequencies

---

## Tips for Success

✅ **Do:**
- Start with small time windows (15-30 min)
- Use 1000ms resampling initially
- Test connection first
- Check dates are weekdays
- Save results frequently

✗ **Don't:**
- Use `--no-resample` on full days (too large)
- Analyze weekends/holidays
- Run multiple full-day analyses simultaneously
- Forget to check Terminal is running

---

## Quick Command Cheat Sheet

```bash
# Quick test
python3 run_tick_analysis.py --pair JD,KWEB --date 2025-01-15 --start-time "09:30:00" --end-time "09:45:00"

# Full day
python3 run_tick_analysis.py --pair JD,KWEB --date 2025-01-15

# High precision
python3 run_tick_analysis.py --pair JD,KWEB --date 2025-01-15 --resample-ms 500

# Multiple days
python3 run_tick_analysis.py --pair JD,KWEB --start-date 2025-01-13 --end-date 2025-01-17

# Top 3 pairs
python3 run_tick_analysis.py --top 3 --date 2025-01-15 --start-time "09:30:00" --end-time "10:00:00"
```

---

## Need Help?

1. Check Bloomberg Terminal is running: `python3 data_fetcher_bloomberg.py`
2. Verify date is valid trading day (Mon-Fri, not holiday)
3. Start with small time window
4. Check error messages carefully

---

**Ready to run?**

```bash
python3 run_tick_analysis.py --pair JD,KWEB --date 2025-01-15 --start-time "09:30:00" --end-time "09:45:00"
```

Good luck! 🚀

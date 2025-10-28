# ACTDR - Arbitrage on Correlated Trading with Delayed Response

Research project analyzing lag relationships in emerging market assets to identify potential arbitrage opportunities due to market inefficiencies.

## Project Overview

**Hypothesis:** Emerging market assets exhibit temporary price dislocations in correlated pairs due to:
- Low trading volume
- Information asymmetry
- Time zone differences
- Market microstructure differences

**Goal:** Detect and quantify lead-lag relationships between correlated assets using statistical analysis.

---

## Project Structure

```
ACTDR/
├── README.md                           # This file
├── STRATEGY_INTENTIONS.txt             # Detailed research strategy and objectives
├── requirements.txt                    # Python dependencies
│
├── find_correlated_pairs.py            # Initial correlation analysis (all asset classes)
├── emerging_markets_correlation.py     # EM-focused correlation analysis (142 assets)
│
├── lag_detection.py                    # Core lag detection algorithms
├── data_fetcher_bloomberg.py           # Bloomberg data fetcher
├── data_fetcher_yfinance.py            # YFinance data fetcher (free alternative)
├── lag_visualization.py                # Plotting and visualization
├── run_lag_analysis.py                 # Main analysis script
│
├── correlated_pairs_findings.txt       # Initial analysis results (45 pairs)
├── em_correlation_report.txt           # EM analysis results (728 pairs)
├── em_pairs_detailed.csv               # All EM pairs with metadata
│
└── lag_plots/                          # Generated visualizations (created on run)
```

---

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Bloomberg API (Optional)

If using Bloomberg Terminal:

```bash
pip install blpapi
```

**Note:** Bloomberg Terminal must be running on your machine.

---

## Quick Start

### Option 1: Test with YFinance (Recommended First Step)

Free data, no Bloomberg required. Good for proof-of-concept.

```bash
# Analyze top 5 pairs with 5 days of 1-minute data
python run_lag_analysis.py --source yfinance --top 5 --days 5 --interval 1m

# Analyze specific pair
python run_lag_analysis.py --source yfinance --pair VALE,GDX --days 7
```

**Limitations:**
- Only 1-minute minimum interval (no tick data)
- Max 7 days for 1-min data
- 15-20 minute delay

### Option 2: Use Bloomberg (Production Quality)

High-quality tick/bar data for research paper.

```bash
# Analyze with 1-minute Bloomberg data
python run_lag_analysis.py --source bloomberg --start 2025-01-15 --end 2025-01-20 --interval-min 1

# Analyze top 3 pairs
python run_lag_analysis.py --source bloomberg --top 3 --start 2025-01-15 --end 2025-01-20
```

**Bloomberg Ticker Format:**
- VALE → `VALE US Equity`
- GDX → `GDX US Equity`
- (handled automatically by script)

---

## Understanding the Output

### Generated Files

1. **lag_analysis_summary.txt**
   - Detailed text report with all statistics
   - Correlation coefficients
   - Lag detection results (which asset leads)
   - Granger causality tests
   - Response time analysis

2. **lag_analysis_dashboard.png**
   - Visual summary of all analyzed pairs
   - Cross-correlation functions
   - Quick comparison chart

3. **lag_analysis_results.json**
   - Raw data in JSON format
   - Easy to import into other tools

4. **lag_plots/[PAIR]/**
   - Individual visualizations for each pair
   - Price comparison charts
   - Returns scatter plots
   - Cross-correlation functions
   - Rolling correlation over time

### Key Metrics Explained

**Pearson Correlation**
- Measures how strongly two assets move together
- Range: -1 to +1 (we focus on >0.6)

**Cross-Correlation Lag**
- How many periods one asset leads the other
- Positive lag: Asset 1 leads Asset 2
- Negative lag: Asset 2 leads Asset 1

**Granger Causality**
- Statistical test: does one series predict the other?
- "Asset A causes Asset B" = A's past values predict B's future

**Response Time**
- How quickly Asset 2 reacts to moves in Asset 1
- Measured in periods (convert based on frequency)

---

## Top Priority Pairs

From our correlation analysis (>0.6 correlation):

### Commodity-EM Pairs (Highest Priority)
```
COPX (Copper) <--> EPU (Peru)        Corr: 0.84
PICK (Metals)  <--> VALE (Mining)     Corr: 0.77
PICK (Metals)  <--> DEM (EM Broad)    Corr: 0.80
GDX (Gold)     <--> EPU (Peru)        Corr: 0.69
```

### Brazilian Stock Pairs
```
SID (Steel)    <--> VALE (Mining)     Corr: 0.73
ITUB (Bank)    <--> BBD (Bank)        Corr: 0.71
EWZ (Brazil)   <--> ITUB (Bank)       Corr: 0.83
```

### Chinese Stock Pairs
```
BABA  <--> JD                         Corr: 0.73
BABA  <--> KWEB (Internet ETF)        Corr: 0.83
JD    <--> KWEB                       Corr: 0.83
```

---

## Usage Examples

### Example 1: Quick Test (5 minutes)

```bash
# Test methodology with free data
python run_lag_analysis.py --source yfinance --pair VALE,GDX --days 5
```

**Expected output:**
- Correlation analysis
- Lag detection (if any)
- 4 visualization charts
- Summary report

### Example 2: Full EM Analysis (30 minutes)

```bash
# Analyze all top pairs
python run_lag_analysis.py --source yfinance --days 7 --interval 1m
```

**Analyzes 13 pairs:**
- Commodity-EM relationships
- Brazilian stocks
- Chinese stocks
- Broad EM indices

### Example 3: Bloomberg Production Run

```bash
# High-quality data for research paper
python run_lag_analysis.py \
    --source bloomberg \
    --start 2025-01-15 \
    --end 2025-01-20 \
    --interval-min 1 \
    --top 10
```

---

## Advanced Usage

### Programmatic Analysis

```python
from lag_detection import LagDetector
from data_fetcher_yfinance import fetch_pair_data_yfinance

# Fetch data
data1, data2 = fetch_pair_data_yfinance('VALE', 'GDX', days=7, interval='1m')

# Create detector
detector = LagDetector(
    data1['close'],
    data2['close'],
    'VALE',
    'GDX',
    frequency='1min'
)

# Run analysis
results = detector.comprehensive_analysis(max_lag=60)

# Access results
print(f"Correlation: {results['pearson']['correlation']:.4f}")
print(f"Leader: {results['cross_correlation']['leader']}")
print(f"Lag: {results['cross_correlation']['lag_periods']} periods")
```

### Custom Pair Analysis

```python
from run_lag_analysis import analyze_pair

# Analyze any pair
results = analyze_pair(
    'BABA',
    'JD',
    data_source='yfinance',
    days=7,
    interval='1m'
)
```

---

## Interpreting Results

### Positive Results (Lag Detected)

✓ **Cross-correlation shows clear lag**
  - One asset consistently leads the other
  - Lag is measurable and consistent

✓ **Granger causality is significant**
  - Past values of one asset predict the other
  - Statistical confidence (p < 0.05)

✓ **Response time is quantifiable**
  - Mean response time is consistent
  - Multiple events analyzed

**Next Steps:**
1. Verify with Bloomberg tick data
2. Measure lag during different market conditions
3. Calculate theoretical arbitrage profit
4. Model transaction costs

### Negative Results (No Lag)

✗ **No clear lag detected**
  - Assets move simultaneously
  - Market may be more efficient than expected

**Considerations:**
- Try higher frequency data (tick-level)
- Test during volatile periods
- Analyze different time windows
- Market may already arbitrage these pairs

---

## Research Workflow

### Phase 1: Discovery ✓ COMPLETE
- [x] Find correlated pairs
- [x] Identify 728 pairs >0.6 correlation
- [x] Prioritize by research value

### Phase 2: Proof-of-Concept ← YOU ARE HERE
- [ ] Test lag detection with yfinance
- [ ] Validate methodology
- [ ] Identify promising pairs

### Phase 3: Detailed Analysis
- [ ] Use Bloomberg tick data
- [ ] Measure precise lag timing
- [ ] Analyze across market conditions

### Phase 4: Documentation
- [ ] Quantify theoretical arbitrage
- [ ] Model transaction costs
- [ ] Write research paper

---

## Troubleshooting

### YFinance Issues

**Problem:** "No data retrieved"
```bash
# Check if ticker is valid
python -c "import yfinance as yf; print(yf.Ticker('VALE').info)"

# Try different interval
python run_lag_analysis.py --pair VALE,GDX --interval 5m
```

**Problem:** "Limited to 7 days"
- 1-minute data has 7-day limit in yfinance
- Use 5-minute interval for longer history
- Or switch to Bloomberg

### Bloomberg Issues

**Problem:** "Failed to connect"
```bash
# 1. Ensure Bloomberg Terminal is running
# 2. Test connection
python data_fetcher_bloomberg.py

# 3. Check blpapi installation
pip install --upgrade blpapi
```

**Problem:** "No data for ticker"
- Verify Bloomberg ticker format: `VALE US Equity`
- Check ticker is valid in Terminal
- Ensure date range has trading data

---

## Files Explanation

### Analysis Scripts

**find_correlated_pairs.py**
- Initial broad analysis
- 142 assets across all classes
- Found 45 pairs >0.8 correlation

**emerging_markets_correlation.py**
- EM-focused analysis
- 142 EM assets, commodities, currencies
- Found 728 pairs >0.6 correlation
- Generates: em_correlation_report.txt, em_pairs_detailed.csv

**run_lag_analysis.py**
- Main execution script
- Runs lag detection on selected pairs
- Generates reports and visualizations

### Core Modules

**lag_detection.py**
- LagDetector class
- Cross-correlation analysis
- Granger causality tests
- Response time measurement
- All statistical methods

**data_fetcher_bloomberg.py**
- Bloomberg API integration
- Intraday bar data
- Tick data support
- Auto ticker conversion

**data_fetcher_yfinance.py**
- Free data alternative
- 1-minute to hourly bars
- Quick testing capability

**lag_visualization.py**
- Plot generation
- Price comparisons
- Cross-correlation charts
- Rolling correlation
- Summary dashboards

---

## Data Sources Comparison

| Feature | YFinance | Bloomberg |
|---------|----------|-----------|
| Cost | Free | Requires Terminal ($$$) |
| Min Interval | 1 minute | 1 second or tick |
| History (1m) | 7 days | Unlimited |
| Delay | 15-20 min | Real-time |
| Quality | Good | Excellent |
| Use Case | Testing | Production |

---

## Contact & Support

For questions about this research project, refer to:
- `STRATEGY_INTENTIONS.txt` - Detailed strategy document
- `em_correlation_report.txt` - Full correlation analysis results
- Code comments in each module

---

## Citation

If using this code for research:

```
ACTDR - Arbitrage on Correlated Trading with Delayed Response
Emerging Markets Lag Detection Framework
2025
```

---

## License

Research project - for educational and research purposes.

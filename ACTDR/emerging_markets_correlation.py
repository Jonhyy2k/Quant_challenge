#!/usr/bin/env python3
"""
Comprehensive Emerging Markets Correlation Analysis
Focus: Finding correlated pairs in emerging markets for arbitrage research
Correlation Threshold: > 0.6
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def fetch_data(tickers, period='2y', interval='1d'):
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

def calculate_correlations(data, min_correlation=0.6):
    """Calculate correlations and find pairs above threshold"""
    # Calculate returns instead of raw prices
    returns = data.pct_change(fill_method=None).dropna()

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

def build_em_universe():
    """Build comprehensive emerging markets asset universe"""

    assets = {
        # ===== EMERGING MARKET COUNTRY ETFs =====
        'EM_Country_ETFs': [
            'EWZ',   # Brazil
            'EWW',   # Mexico
            'FXI',   # China Large Cap
            'MCHI',  # China
            'ASHR',  # China A-Shares
            'KWEB',  # China Internet
            'EWY',   # South Korea
            'EWT',   # Taiwan
            'INDA',  # India
            'INDY',  # India
            'EPI',   # India
            'EWH',   # Hong Kong
            'EPHE',  # Philippines
            'EIDO',  # Indonesia
            'THD',   # Thailand
            'EZA',   # South Africa
            'TUR',   # Turkey
            'ECH',   # Chile
            'EPU',   # Peru
            'ARGT',  # Argentina
            'EWS',   # Singapore
            'EWM',   # Malaysia
            'VNM',   # Vietnam
            'EGPT',  # Egypt
            'AFK',   # Africa
            'GAF',   # Africa
        ],

        # ===== BROAD EMERGING MARKETS ETFs =====
        'EM_Broad': [
            'EEM',   # MSCI Emerging Markets
            'VWO',   # Vanguard EM
            'IEMG',  # iShares Core EM
            'DEM',   # WisdomTree EM
            'SCHE',  # Schwab EM
            'SPEM',  # SPDR EM
            'EEMV',  # EM Min Volatility
            'EDIV',  # EM Dividend
        ],

        # ===== EMERGING MARKET SECTORS =====
        'EM_Sector': [
            'FM',    # Frontier Markets
            'EMQQ',  # EM Internet & Ecommerce
            'KEMX',  # EM Consumer
            'EMFM',  # EM Financials
            'EELV',  # EM Value
        ],

        # ===== BRAZILIAN ADRs & STOCKS =====
        'Brazil_Stocks': [
            'VALE',  # Vale - Mining
            'PBR',   # Petrobras - Oil & Gas
            'ITUB',  # Itau Unibanco - Banking
            'BBD',   # Bradesco - Banking
            'ABEV',  # Ambev - Beverages
            'SID',   # CSN - Steel
            'UGP',   # Ultrapar - Energy
            'CBD',   # CBD - Retail
            'TIMB',  # TIM Brasil - Telecom
            'CIG',   # Energia - Utilities
            'ASAI',  # Assai - Retail
            'NU',    # Nubank - Fintech
        ],

        # ===== CHINESE ADRs & STOCKS =====
        'China_Stocks': [
            'BABA',  # Alibaba
            'BIDU',  # Baidu
            'JD',    # JD.com
            'PDD',   # Pinduoduo
            'NIO',   # Nio - EV
            'XPEV',  # XPeng - EV
            'LI',    # Li Auto - EV
            'BILI',  # Bilibili
            'TME',   # Tencent Music
            'NTES',  # NetEase
            'WB',    # Weibo
            'YMM',   # Full Truck Alliance
            'VIPS',  # Vipshop
            'HTHT',  # Huazhu Hotels
            'EDU',   # New Oriental Education
            'TAL',   # TAL Education
        ],

        # ===== INDIAN ADRs & STOCKS =====
        'India_Stocks': [
            'INFY',  # Infosys
            'WIT',   # Wipro
            'HDB',   # HDFC Bank
            'IBN',   # ICICI Bank
            'TTM',   # Tata Motors
            'VEDL',  # Vedanta
            'RDY',   # Dr. Reddy's
            'SIFY',  # Sify Technologies
        ],

        # ===== TAIWAN STOCKS =====
        'Taiwan_Stocks': [
            'TSM',   # TSMC
            'UMC',   # United Microelectronics
            'ASX',   # Advanced Semiconductor
        ],

        # ===== OTHER EM STOCKS =====
        'Other_EM_Stocks': [
            'AMX',   # America Movil - Mexico Telecom
            'KB',    # KB Financial - Korea
            'SHG',   # Shinhan - Korea
            'LPL',   # LG Display - Korea
            'SSL',   # Sasol - South Africa
            'GOLD',  # Harmony Gold - South Africa
            'TEO',   # Telecom Argentina
        ],

        # ===== COMMODITIES (EM economies are commodity-heavy) =====
        'Commodities_Metals': [
            'GLD',   # Gold
            'SLV',   # Silver
            'GDX',   # Gold Miners
            'GDXJ',  # Junior Gold Miners
            'COPX',  # Copper Miners
            'DBB',   # Base Metals
            'PICK',  # Global Metals & Mining
            'PALL',  # Palladium
            'PPLT',  # Platinum
            'GLTR',  # Precious Metals
        ],

        'Commodities_Energy': [
            'USO',   # Oil
            'UNG',   # Natural Gas
            'DBO',   # Oil ETF
            'BNO',   # Brent Oil
            'XLE',   # Energy Sector
            'XOP',   # Oil & Gas Exploration
            'OIH',   # Oil Services
        ],

        'Commodities_Agriculture': [
            'DBA',   # Agriculture
            'CORN',  # Corn
            'WEAT',  # Wheat
            'SOYB',  # Soybeans
            'CANE',  # Sugar
            'JO',    # Coffee
            'NIB',   # Cocoa
        ],

        # ===== CURRENCY ETFs =====
        'Currencies': [
            'FXE',   # Euro
            'FXY',   # Japanese Yen
            'FXB',   # British Pound
            'FXA',   # Australian Dollar
            'FXC',   # Canadian Dollar
            'CYB',   # Chinese Yuan
            'BZF',   # Brazilian Real
            'CEW',   # EM Currencies
            'UUP',   # US Dollar Index
        ],

        # ===== DEVELOPED MARKETS (for comparison/correlation) =====
        'Developed_Indices': [
            'SPY',   # S&P 500
            'QQQ',   # Nasdaq
            'DIA',   # Dow Jones
            'IWM',   # Russell 2000
            'VTI',   # Total US Market
            'VT',    # Total World
            'EFA',   # Developed Markets ex-US
            'VEA',   # Developed Markets
        ],

        'Developed_Sectors': [
            'XLF',   # Financials
            'XLE',   # Energy
            'XLK',   # Technology
            'XLV',   # Healthcare
            'XLI',   # Industrials
            'XLU',   # Utilities
            'XLP',   # Consumer Staples
            'XLY',   # Consumer Discretionary
            'XLB',   # Materials
            'XLRE',  # Real Estate
        ],

        # ===== BONDS (for macro correlation) =====
        'Bonds': [
            'TLT',   # 20+ Year Treasury
            'IEF',   # 7-10 Year Treasury
            'SHY',   # 1-3 Year Treasury
            'LQD',   # Investment Grade Corp
            'HYG',   # High Yield Corp
            'EMB',   # Emerging Market Bonds
            'PCY',   # EM Sovereign Debt
        ],
    }

    return assets

def categorize_asset(ticker, asset_categories):
    """Find which category an asset belongs to"""
    for category, tickers in asset_categories.items():
        if ticker in tickers:
            return category
    return 'Unknown'

def analyze_em_pairs():
    """Main analysis function"""

    # Build asset universe
    asset_categories = build_em_universe()

    # Flatten all tickers
    all_tickers = []
    for category, tickers in asset_categories.items():
        all_tickers.extend(tickers)

    # Remove duplicates (XLE appears in both commodities and sectors)
    all_tickers = list(set(all_tickers))

    print(f"\n{'='*80}")
    print(f"EMERGING MARKETS CORRELATION ANALYSIS")
    print(f"{'='*80}")
    print(f"Total tickers to analyze: {len(all_tickers)}")
    print(f"Categories: {len(asset_categories)}")
    print(f"Correlation threshold: > 0.6")
    print(f"{'='*80}\n")

    # Fetch data
    data = fetch_data(all_tickers, period='2y', interval='1d')

    # Remove tickers with too much missing data (>20% missing)
    threshold = len(data) * 0.2
    valid_tickers = data.columns[data.isnull().sum() < threshold]
    data = data[valid_tickers]
    data = data.dropna(axis=1, how='all')

    print(f"Valid tickers after removing incomplete data: {len(data.columns)}")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print()

    # Calculate correlations
    pairs_df, corr_matrix = calculate_correlations(data, min_correlation=0.6)

    return pairs_df, data, asset_categories, corr_matrix

def generate_em_report(pairs_df, asset_categories):
    """Generate detailed report focusing on EM opportunities"""

    if len(pairs_df) == 0:
        return "No pairs found with correlation > 0.6"

    # Add categories to pairs
    pairs_df['category1'] = pairs_df['asset1'].apply(lambda x: categorize_asset(x, asset_categories))
    pairs_df['category2'] = pairs_df['asset2'].apply(lambda x: categorize_asset(x, asset_categories))

    # Create pair type classification
    def classify_pair_type(row):
        c1, c2 = row['category1'], row['category2']

        # Check if both are EM
        em_categories = ['EM_Country_ETFs', 'EM_Broad', 'EM_Sector',
                        'Brazil_Stocks', 'China_Stocks', 'India_Stocks',
                        'Taiwan_Stocks', 'Other_EM_Stocks']

        c1_is_em = c1 in em_categories
        c2_is_em = c2 in em_categories

        if c1_is_em and c2_is_em:
            return 'EM-EM (HIGH PRIORITY)'
        elif (c1_is_em or c2_is_em) and ('Commodities' in c1 or 'Commodities' in c2):
            return 'EM-Commodity (MEDIUM PRIORITY)'
        elif (c1_is_em or c2_is_em) and ('Developed' in c1 or 'Developed' in c2):
            return 'EM-Developed (MEDIUM PRIORITY)'
        elif c1_is_em or c2_is_em:
            return 'EM-Other (MEDIUM PRIORITY)'
        else:
            return 'Non-EM (LOW PRIORITY)'

    pairs_df['pair_type'] = pairs_df.apply(classify_pair_type, axis=1)

    # Separate into priority groups
    high_priority = pairs_df[pairs_df['pair_type'].str.contains('HIGH')]
    medium_priority = pairs_df[pairs_df['pair_type'].str.contains('MEDIUM')]

    report = []
    report.append("=" * 100)
    report.append("EMERGING MARKETS CORRELATION ANALYSIS - COMPREHENSIVE REPORT")
    report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 100)
    report.append("")
    report.append(f"TOTAL PAIRS FOUND (correlation > 0.6): {len(pairs_df)}")
    report.append(f"  - HIGH PRIORITY (EM-EM pairs): {len(high_priority)}")
    report.append(f"  - MEDIUM PRIORITY (EM-Commodity, EM-Developed): {len(medium_priority)}")
    report.append(f"  - LOW PRIORITY (Non-EM): {len(pairs_df) - len(high_priority) - len(medium_priority)}")
    report.append("")
    report.append("=" * 100)
    report.append("HIGH PRIORITY PAIRS: EMERGING MARKET <--> EMERGING MARKET")
    report.append("=" * 100)
    report.append("These pairs are most likely to exhibit lag due to:")
    report.append("  - Different time zones (Asia vs Latin America)")
    report.append("  - Varying liquidity levels")
    report.append("  - Information asymmetry between markets")
    report.append("  - Market microstructure differences")
    report.append("")

    if len(high_priority) > 0:
        for idx, (_, row) in enumerate(high_priority.iterrows(), 1):
            report.append(f"PAIR #{idx}")
            report.append(f"  {row['asset1']} <--> {row['asset2']}")
            report.append(f"  Correlation: {row['correlation']:.4f}")
            report.append(f"  Categories: {row['category1']} <--> {row['category2']}")
            report.append("")
    else:
        report.append("  [No pure EM-EM pairs found above 0.6 correlation]")
        report.append("")

    report.append("=" * 100)
    report.append("MEDIUM PRIORITY PAIRS: EM <--> COMMODITIES / DEVELOPED MARKETS")
    report.append("=" * 100)
    report.append("Cross-asset and cross-market pairs with potential lag opportunities:")
    report.append("")

    # Show top 50 medium priority pairs
    for idx, (_, row) in enumerate(medium_priority.head(50).iterrows(), 1):
        report.append(f"PAIR #{idx}")
        report.append(f"  {row['asset1']} <--> {row['asset2']}")
        report.append(f"  Correlation: {row['correlation']:.4f}")
        report.append(f"  Categories: {row['category1']} <--> {row['category2']}")
        report.append(f"  Type: {row['pair_type']}")
        report.append("")

    # Summary statistics by category combination
    report.append("=" * 100)
    report.append("SUMMARY BY PAIR TYPE")
    report.append("=" * 100)
    summary = pairs_df.groupby('pair_type').agg({
        'correlation': ['count', 'mean', 'min', 'max']
    }).round(4)
    report.append(summary.to_string())
    report.append("")

    # Top correlations by EM region
    report.append("=" * 100)
    report.append("RESEARCH RECOMMENDATIONS")
    report.append("=" * 100)
    report.append("")
    report.append("PHASE 1: Test with high-priority EM-EM pairs")
    report.append("  → Focus on cross-regional pairs (e.g., China vs Brazil)")
    report.append("  → Time zone differences create natural lag opportunities")
    report.append("")
    report.append("PHASE 2: Commodity-dependent EM economies")
    report.append("  → Oil-dependent: Brazil (PBR), Mexico (EWW) vs USO/XLE")
    report.append("  → Mining-dependent: Chile (ECH), Peru (EPU), South Africa (EZA) vs metals")
    report.append("  → Natural lag: commodity moves first, EM equity responds")
    report.append("")
    report.append("PHASE 3: Intraday analysis with Bloomberg data")
    report.append("  → Measure millisecond-level price discovery lags")
    report.append("  → Quantify lead-lag relationships")
    report.append("  → Calculate theoretical arbitrage window")
    report.append("")
    report.append("PHASE 4: Machine learning enhancement")
    report.append("  → Use neural networks to predict lag duration")
    report.append("  → Feature engineering: volume, volatility, time-of-day")
    report.append("  → Real-time lag detection system")
    report.append("")

    return "\n".join(report)

def save_detailed_csv(pairs_df, asset_categories):
    """Save detailed CSV for further analysis"""

    # Add categories
    pairs_df['category1'] = pairs_df['asset1'].apply(lambda x: categorize_asset(x, asset_categories))
    pairs_df['category2'] = pairs_df['asset2'].apply(lambda x: categorize_asset(x, asset_categories))

    # Save
    csv_path = "/home/joaop/PyResearch/ITAU/ACTDR/em_pairs_detailed.csv"
    pairs_df.to_csv(csv_path, index=False)
    print(f"✓ Detailed CSV saved to: {csv_path}")

    return csv_path

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE EMERGING MARKETS ANALYSIS")
    print("="*80 + "\n")

    # Run analysis
    pairs_df, data, asset_categories, corr_matrix = analyze_em_pairs()

    # Generate report
    report = generate_em_report(pairs_df, asset_categories)

    # Save report
    report_file = "/home/joaop/PyResearch/ITAU/ACTDR/em_correlation_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)

    # Save detailed CSV
    csv_file = save_detailed_csv(pairs_df, asset_categories)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Total pairs found: {len(pairs_df)}")
    print(f"✓ Report saved to: {report_file}")
    print(f"✓ CSV data saved to: {csv_file}")
    print(f"{'='*80}\n")

#!/usr/bin/env python3
"""
Core Lag Detection Module
Statistical analysis for detecting lead-lag relationships between correlated assets
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import correlate
from statsmodels.tsa.stattools import grangercausalitytests, ccf
import warnings
warnings.filterwarnings('ignore')


class LagDetector:
    """
    Detects and quantifies lag relationships between two time series
    """

    def __init__(self, asset1_data, asset2_data, asset1_name, asset2_name, frequency='1min'):
        """
        Initialize with price data for two assets

        Parameters:
        -----------
        asset1_data : pd.Series
            Price series for asset 1 (with datetime index)
        asset2_data : pd.Series
            Price series for asset 2 (with datetime index)
        asset1_name : str
            Name of asset 1
        asset2_name : str
            Name of asset 2
        frequency : str
            Data frequency ('1min', '1s', 'tick')
        """
        self.asset1_name = asset1_name
        self.asset2_name = asset2_name
        self.frequency = frequency

        # Align the data
        df = pd.DataFrame({
            asset1_name: asset1_data,
            asset2_name: asset2_data
        }).dropna()

        self.data = df
        self.returns1 = df[asset1_name].pct_change().dropna()
        self.returns2 = df[asset2_name].pct_change().dropna()

    def pearson_correlation(self):
        """Calculate Pearson correlation coefficient"""
        corr, pvalue = stats.pearsonr(self.returns1, self.returns2)
        return {
            'correlation': corr,
            'p_value': pvalue,
            'significant': pvalue < 0.05
        }

    def cross_correlation_analysis(self, max_lag=60):
        """
        Cross-correlation analysis to detect lead-lag relationship

        Parameters:
        -----------
        max_lag : int
            Maximum lag to test (in periods)

        Returns:
        --------
        dict with lag detection results
        """
        # Normalize returns
        r1_norm = (self.returns1 - self.returns1.mean()) / self.returns1.std()
        r2_norm = (self.returns2 - self.returns2.mean()) / self.returns2.std()

        # Calculate cross-correlation
        correlation = correlate(r1_norm, r2_norm, mode='full')
        lags = np.arange(-len(r1_norm) + 1, len(r1_norm))

        # Limit to max_lag
        center = len(correlation) // 2
        lag_range = range(center - max_lag, center + max_lag + 1)
        limited_corr = correlation[lag_range]
        limited_lags = lags[lag_range]

        # Find maximum correlation
        max_corr_idx = np.argmax(np.abs(limited_corr))
        optimal_lag = limited_lags[max_corr_idx]
        max_corr = limited_corr[max_corr_idx]

        # Interpretation
        if optimal_lag < 0:
            leader = self.asset2_name
            follower = self.asset1_name
            lag_periods = abs(optimal_lag)
        elif optimal_lag > 0:
            leader = self.asset1_name
            follower = self.asset2_name
            lag_periods = optimal_lag
        else:
            leader = "No clear leader"
            follower = "Simultaneous movement"
            lag_periods = 0

        return {
            'optimal_lag': optimal_lag,
            'lag_periods': lag_periods,
            'max_correlation': max_corr,
            'leader': leader,
            'follower': follower,
            'all_lags': limited_lags,
            'all_correlations': limited_corr,
            'frequency': self.frequency
        }

    def granger_causality_test(self, max_lag=20):
        """
        Granger causality test to determine if one series predicts the other

        Returns:
        --------
        dict with causality test results
        """
        # Prepare data
        df = pd.DataFrame({
            'asset1': self.returns1,
            'asset2': self.returns2
        }).dropna()

        if len(df) < max_lag + 10:
            return {
                'error': 'Insufficient data for Granger test',
                'asset1_causes_asset2': False,
                'asset2_causes_asset1': False
            }

        results = {}

        # Test if asset1 Granger-causes asset2
        try:
            test1 = grangercausalitytests(df[['asset2', 'asset1']], max_lag, verbose=False)
            # Get p-values for each lag
            pvalues1 = [test1[i+1][0]['ssr_ftest'][1] for i in range(max_lag)]
            min_pvalue1 = min(pvalues1)
            best_lag1 = pvalues1.index(min_pvalue1) + 1

            results['asset1_causes_asset2'] = min_pvalue1 < 0.05
            results['asset1_to_asset2_pvalue'] = min_pvalue1
            results['asset1_to_asset2_lag'] = best_lag1
        except Exception as e:
            results['asset1_causes_asset2'] = False
            results['asset1_to_asset2_error'] = str(e)

        # Test if asset2 Granger-causes asset1
        try:
            test2 = grangercausalitytests(df[['asset1', 'asset2']], max_lag, verbose=False)
            pvalues2 = [test2[i+1][0]['ssr_ftest'][1] for i in range(max_lag)]
            min_pvalue2 = min(pvalues2)
            best_lag2 = pvalues2.index(min_pvalue2) + 1

            results['asset2_causes_asset1'] = min_pvalue2 < 0.05
            results['asset2_to_asset1_pvalue'] = min_pvalue2
            results['asset2_to_asset1_lag'] = best_lag2
        except Exception as e:
            results['asset2_causes_asset1'] = False
            results['asset2_to_asset1_error'] = str(e)

        # Interpretation
        if results.get('asset1_causes_asset2') and not results.get('asset2_causes_asset1'):
            results['leader'] = self.asset1_name
            results['follower'] = self.asset2_name
            results['lag'] = results.get('asset1_to_asset2_lag', 0)
        elif results.get('asset2_causes_asset1') and not results.get('asset1_causes_asset2'):
            results['leader'] = self.asset2_name
            results['follower'] = self.asset1_name
            results['lag'] = results.get('asset2_to_asset1_lag', 0)
        elif results.get('asset1_causes_asset2') and results.get('asset2_causes_asset1'):
            results['leader'] = 'Bidirectional causality'
            results['follower'] = 'Both assets lead and follow'
        else:
            results['leader'] = 'No significant causality'
            results['follower'] = 'No clear lag relationship'

        return results

    def rolling_correlation(self, window=60):
        """
        Calculate rolling correlation to see if relationship changes over time

        Parameters:
        -----------
        window : int
            Rolling window size in periods
        """
        rolling_corr = self.returns1.rolling(window).corr(self.returns2)

        return {
            'rolling_correlation': rolling_corr,
            'mean_correlation': rolling_corr.mean(),
            'std_correlation': rolling_corr.std(),
            'min_correlation': rolling_corr.min(),
            'max_correlation': rolling_corr.max()
        }

    def time_to_adjust(self, threshold=0.001):
        """
        Measure how long it takes for asset2 to respond to significant moves in asset1

        Parameters:
        -----------
        threshold : float
            Minimum return to consider a "significant move" (e.g., 0.001 = 0.1%)

        Returns:
        --------
        Statistics on response time
        """
        # Find significant moves in asset1
        significant_moves = self.returns1.abs() > threshold
        move_indices = self.returns1[significant_moves].index

        response_times = []

        for move_time in move_indices:
            # Look at asset2 returns in the next 10 periods
            future_window = self.returns2.loc[move_time:].iloc[:11]  # Current + next 10

            if len(future_window) < 2:
                continue

            # Find when asset2 makes a similar magnitude move in same direction
            move_direction = np.sign(self.returns1.loc[move_time])

            for i, (timestamp, ret) in enumerate(future_window.items()):
                if i == 0:  # Skip the current period
                    continue
                if np.sign(ret) == move_direction and abs(ret) > threshold * 0.5:
                    response_times.append(i)
                    break

        if len(response_times) == 0:
            return {
                'error': 'No significant moves detected or no responses found',
                'threshold': threshold
            }

        return {
            'mean_response_periods': np.mean(response_times),
            'median_response_periods': np.median(response_times),
            'min_response_periods': np.min(response_times),
            'max_response_periods': np.max(response_times),
            'num_events': len(response_times),
            'threshold_used': threshold,
            'frequency': self.frequency
        }

    def comprehensive_analysis(self, max_lag=60):
        """
        Run all lag detection methods and return comprehensive report
        """
        print(f"\nAnalyzing {self.asset1_name} <--> {self.asset2_name}")
        print(f"Data points: {len(self.data)}, Frequency: {self.frequency}")
        print("="*70)

        results = {
            'asset1': self.asset1_name,
            'asset2': self.asset2_name,
            'data_points': len(self.data),
            'frequency': self.frequency
        }

        # 1. Basic correlation
        print("\n1. Pearson Correlation...")
        corr_results = self.pearson_correlation()
        results['pearson'] = corr_results
        print(f"   Correlation: {corr_results['correlation']:.4f} (p={corr_results['p_value']:.4f})")

        # 2. Cross-correlation
        print("\n2. Cross-Correlation Analysis...")
        ccf_results = self.cross_correlation_analysis(max_lag=max_lag)
        results['cross_correlation'] = ccf_results
        print(f"   Leader: {ccf_results['leader']}")
        print(f"   Follower: {ccf_results['follower']}")
        print(f"   Lag: {ccf_results['lag_periods']} periods ({self.frequency})")
        print(f"   Max correlation at lag: {ccf_results['max_correlation']:.4f}")

        # 3. Granger causality
        print("\n3. Granger Causality Test...")
        granger_results = self.granger_causality_test(max_lag=min(max_lag, 20))
        results['granger'] = granger_results
        print(f"   Leader: {granger_results.get('leader', 'N/A')}")
        if 'lag' in granger_results:
            print(f"   Lag: {granger_results['lag']} periods")

        # 4. Rolling correlation
        print("\n4. Rolling Correlation Analysis...")
        rolling_results = self.rolling_correlation(window=min(60, len(self.data)//10))
        results['rolling'] = {
            'mean': rolling_results['mean_correlation'],
            'std': rolling_results['std_correlation'],
            'min': rolling_results['min_correlation'],
            'max': rolling_results['max_correlation']
        }
        print(f"   Mean: {rolling_results['mean_correlation']:.4f}")
        print(f"   Stability (std): {rolling_results['std_correlation']:.4f}")

        # 5. Response time
        print("\n5. Response Time Analysis...")
        response_results = self.time_to_adjust(threshold=0.001)
        results['response_time'] = response_results
        if 'error' not in response_results:
            print(f"   Mean response: {response_results['mean_response_periods']:.2f} periods")
            print(f"   Events analyzed: {response_results['num_events']}")
        else:
            print(f"   {response_results['error']}")

        print("="*70)

        return results


def convert_lag_to_time(lag_periods, frequency):
    """
    Convert lag in periods to human-readable time

    Parameters:
    -----------
    lag_periods : int
        Number of periods of lag
    frequency : str
        Data frequency ('1min', '1s', 'tick')
    """
    if frequency == '1min':
        seconds = lag_periods * 60
        if seconds < 60:
            return f"{seconds:.0f} seconds"
        else:
            return f"{seconds/60:.1f} minutes"
    elif frequency == '1s':
        if lag_periods < 60:
            return f"{lag_periods} seconds"
        else:
            return f"{lag_periods/60:.1f} minutes"
    elif frequency == 'tick':
        return f"{lag_periods} ticks (milliseconds)"
    else:
        return f"{lag_periods} periods"

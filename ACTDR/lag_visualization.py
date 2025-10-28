#!/usr/bin/env python3
"""
Visualization Module for Lag Detection Analysis
Creates charts and plots to visualize lead-lag relationships
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def plot_price_comparison(data1, data2, asset1_name, asset2_name, save_path=None):
    """
    Plot normalized prices of two assets to visualize co-movement

    Parameters:
    -----------
    data1, data2 : pd.Series or pd.DataFrame
        Price data (Series) or DataFrame with 'close' column
    asset1_name, asset2_name : str
        Asset names for labeling
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Handle both Series and DataFrame input
    if isinstance(data1, pd.DataFrame):
        prices1 = data1['close'] if 'close' in data1.columns else data1.iloc[:, 0]
    else:
        prices1 = data1

    if isinstance(data2, pd.DataFrame):
        prices2 = data2['close'] if 'close' in data2.columns else data2.iloc[:, 0]
    else:
        prices2 = data2

    # Normalize to 100
    norm1 = (prices1 / prices1.iloc[0]) * 100
    norm2 = (prices2 / prices2.iloc[0]) * 100

    ax.plot(norm1.index, norm1, label=asset1_name, linewidth=2, alpha=0.8)
    ax.plot(norm2.index, norm2, label=asset2_name, linewidth=2, alpha=0.8)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Normalized Price (Base=100)', fontsize=12)
    ax.set_title(f'Price Comparison: {asset1_name} vs {asset2_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved price comparison plot to {save_path}")

    return fig


def plot_returns_scatter(returns1, returns2, asset1_name, asset2_name, save_path=None):
    """
    Scatter plot of returns to visualize correlation

    Parameters:
    -----------
    returns1, returns2 : pd.Series
        Return series
    asset1_name, asset2_name : str
        Asset names
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create scatter plot
    ax.scatter(returns1, returns2, alpha=0.5, s=20)

    # Add regression line
    z = np.polyfit(returns1, returns2, 1)
    p = np.poly1d(z)
    x_line = np.linspace(returns1.min(), returns1.max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')

    # Calculate correlation
    corr = returns1.corr(returns2)

    ax.set_xlabel(f'{asset1_name} Returns', fontsize=12)
    ax.set_ylabel(f'{asset2_name} Returns', fontsize=12)
    ax.set_title(f'Return Correlation: {asset1_name} vs {asset2_name}\nCorrelation: {corr:.4f}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved returns scatter plot to {save_path}")

    return fig


def plot_cross_correlation(lags, correlations, asset1_name, asset2_name, save_path=None):
    """
    Plot cross-correlation function showing lag relationship

    Parameters:
    -----------
    lags : array
        Lag values
    correlations : array
        Correlation values at each lag
    asset1_name, asset2_name : str
        Asset names
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot cross-correlation
    ax.plot(lags, correlations, linewidth=2, color='blue')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Zero Lag')

    # Highlight maximum
    max_idx = np.argmax(np.abs(correlations))
    max_lag = lags[max_idx]
    max_corr = correlations[max_idx]

    ax.plot(max_lag, max_corr, 'ro', markersize=10, label=f'Max: lag={max_lag}, corr={max_corr:.4f}')

    ax.set_xlabel('Lag (periods)', fontsize=12)
    ax.set_ylabel('Cross-Correlation', fontsize=12)
    ax.set_title(f'Cross-Correlation Analysis: {asset1_name} vs {asset2_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add interpretation text
    if max_lag < 0:
        interpretation = f"{asset2_name} leads {asset1_name} by {abs(max_lag)} periods"
    elif max_lag > 0:
        interpretation = f"{asset1_name} leads {asset2_name} by {max_lag} periods"
    else:
        interpretation = "No significant lag detected"

    ax.text(0.02, 0.98, interpretation, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved cross-correlation plot to {save_path}")

    return fig


def plot_rolling_correlation(rolling_corr, asset1_name, asset2_name, window, save_path=None):
    """
    Plot rolling correlation to show stability over time

    Parameters:
    -----------
    rolling_corr : pd.Series
        Rolling correlation values
    asset1_name, asset2_name : str
        Asset names
    window : int
        Window size used for rolling correlation
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(rolling_corr.index, rolling_corr, linewidth=2, color='green', alpha=0.8)
    ax.axhline(y=rolling_corr.mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {rolling_corr.mean():.4f}')

    # Add confidence bands
    std = rolling_corr.std()
    ax.fill_between(rolling_corr.index,
                     rolling_corr.mean() - std,
                     rolling_corr.mean() + std,
                     alpha=0.2, color='red', label=f'±1 Std Dev ({std:.4f})')

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Rolling Correlation', fontsize=12)
    ax.set_title(f'Rolling Correlation ({window} period window): {asset1_name} vs {asset2_name}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved rolling correlation plot to {save_path}")

    return fig


def plot_comprehensive_analysis(detector, save_dir='./plots'):
    """
    Create comprehensive visualization suite for a LagDetector instance

    Parameters:
    -----------
    detector : LagDetector
        Instance of LagDetector with analyzed data
    save_dir : str
        Directory to save plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    asset1 = detector.asset1_name
    asset2 = detector.asset2_name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"\nGenerating visualizations for {asset1} <--> {asset2}...")
    print("="*70)

    # 1. Price comparison
    print("1. Creating price comparison chart...")
    plot_price_comparison(
        detector.data[asset1], detector.data[asset2],
        asset1, asset2,
        save_path=f"{save_dir}/{asset1}_{asset2}_prices_{timestamp}.png"
    )
    plt.close()

    # 2. Returns scatter
    print("2. Creating returns scatter plot...")
    plot_returns_scatter(
        detector.returns1, detector.returns2,
        asset1, asset2,
        save_path=f"{save_dir}/{asset1}_{asset2}_scatter_{timestamp}.png"
    )
    plt.close()

    # 3. Cross-correlation
    print("3. Creating cross-correlation plot...")
    ccf_results = detector.cross_correlation_analysis()
    plot_cross_correlation(
        ccf_results['all_lags'], ccf_results['all_correlations'],
        asset1, asset2,
        save_path=f"{save_dir}/{asset1}_{asset2}_ccf_{timestamp}.png"
    )
    plt.close()

    # 4. Rolling correlation
    print("4. Creating rolling correlation plot...")
    rolling_results = detector.rolling_correlation(window=min(60, len(detector.data)//10))
    plot_rolling_correlation(
        rolling_results['rolling_correlation'],
        asset1, asset2,
        window=min(60, len(detector.data)//10),
        save_path=f"{save_dir}/{asset1}_{asset2}_rolling_{timestamp}.png"
    )
    plt.close()

    print("="*70)
    print(f"✓ All plots saved to {save_dir}/")
    print("="*70)


def create_summary_dashboard(results_list, save_path='lag_analysis_dashboard.png'):
    """
    Create a summary dashboard showing multiple pairs

    Parameters:
    -----------
    results_list : list of dict
        List of analysis results from LagDetector.comprehensive_analysis()
    save_path : str
        Path to save dashboard
    """
    n_pairs = len(results_list)

    fig, axes = plt.subplots(n_pairs, 2, figsize=(16, 4*n_pairs))

    if n_pairs == 1:
        axes = axes.reshape(1, -1)

    for i, result in enumerate(results_list):
        asset1 = result['asset1']
        asset2 = result['asset2']

        # Left: Correlation info
        ax_left = axes[i, 0]
        ax_left.axis('off')

        info_text = f"""
        Pair: {asset1} <--> {asset2}

        Correlation: {result['pearson']['correlation']:.4f}

        Cross-Correlation:
          Leader: {result['cross_correlation']['leader']}
          Lag: {result['cross_correlation']['lag_periods']} periods

        Granger Causality:
          Leader: {result['granger'].get('leader', 'N/A')}
        """

        ax_left.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
                     family='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # Right: Cross-correlation plot
        ax_right = axes[i, 1]
        ccf_data = result['cross_correlation']
        ax_right.plot(ccf_data['all_lags'], ccf_data['all_correlations'], linewidth=2)
        ax_right.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax_right.set_xlabel('Lag (periods)')
        ax_right.set_ylabel('Correlation')
        ax_right.set_title(f'CCF: {asset1} vs {asset2}')
        ax_right.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved dashboard to {save_path}")

    return fig


if __name__ == "__main__":
    print("Lag Visualization Module")
    print("="*70)
    print("This module provides visualization functions for lag analysis.")
    print("\nMain functions:")
    print("  - plot_price_comparison()")
    print("  - plot_returns_scatter()")
    print("  - plot_cross_correlation()")
    print("  - plot_rolling_correlation()")
    print("  - plot_comprehensive_analysis()")
    print("  - create_summary_dashboard()")
    print("="*70)

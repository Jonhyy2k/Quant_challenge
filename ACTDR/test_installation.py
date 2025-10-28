#!/usr/bin/env python3
"""
Installation Test Script
Verifies that all dependencies are installed and modules can be imported
"""

import sys

def test_imports():
    """Test if all required modules can be imported"""

    print("="*70)
    print("TESTING INSTALLATION")
    print("="*70)
    print()

    errors = []
    warnings = []

    # Test core dependencies
    modules = [
        ('pandas', 'Data manipulation'),
        ('numpy', 'Numerical computing'),
        ('scipy', 'Scientific computing'),
        ('statsmodels', 'Statistical models'),
        ('matplotlib', 'Plotting'),
        ('seaborn', 'Statistical visualization'),
        ('yfinance', 'Yahoo Finance data'),
    ]

    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name:15} - {description}")
        except ImportError as e:
            errors.append(f"✗ {module_name:15} - MISSING ({description})")
            print(errors[-1])

    print()

    # Test Bloomberg (optional)
    print("Optional Dependencies:")
    print("-" * 70)
    try:
        import blpapi
        print(f"✓ blpapi          - Bloomberg API (installed)")
    except ImportError:
        warnings.append("! blpapi          - Not installed (optional, only needed for Bloomberg)")
        print(warnings[-1])

    print()

    # Test our modules
    print("Project Modules:")
    print("-" * 70)

    project_modules = [
        ('lag_detection', 'Core lag detection algorithms'),
        ('data_fetcher_yfinance', 'YFinance data fetcher'),
        ('data_fetcher_bloomberg', 'Bloomberg data fetcher'),
        ('lag_visualization', 'Visualization functions'),
    ]

    for module_name, description in project_modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name:25} - {description}")
        except ImportError as e:
            errors.append(f"✗ {module_name:25} - FAILED: {e}")
            print(errors[-1])

    print()
    print("="*70)

    # Summary
    if len(errors) == 0:
        print("✓ ALL CHECKS PASSED")
        print()
        print("Installation is complete and working!")
        print()
        print("Next steps:")
        print("  1. Read QUICK_START.md for getting started")
        print("  2. Run: python run_lag_analysis.py --pair VALE,GDX --days 5")
        print()

        if len(warnings) > 0:
            print("Note: Bloomberg API not installed (only needed if using Bloomberg Terminal)")

        return True
    else:
        print("✗ INSTALLATION INCOMPLETE")
        print()
        print("Errors found:")
        for error in errors:
            print(f"  {error}")
        print()
        print("To fix:")
        print("  pip install -r requirements.txt")
        print()
        return False


def test_data_fetch():
    """Quick test of data fetching"""

    print("="*70)
    print("TESTING DATA FETCHING")
    print("="*70)
    print()

    try:
        from data_fetcher_yfinance import fetch_intraday_yfinance

        print("Attempting to fetch 1 day of VALE data...")
        data = fetch_intraday_yfinance('VALE', days=1, interval='1m')

        if len(data) > 0:
            print(f"✓ Successfully fetched {len(data)} data points")
            print(f"  Date range: {data.index[0]} to {data.index[-1]}")
            print()
            print("Sample data:")
            print(data.head(3))
            print()
            return True
        else:
            print("✗ No data retrieved (market might be closed)")
            print("  This is normal if outside trading hours")
            print()
            return True

    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        print()
        return False


def main():
    """Run all tests"""

    print("\n" + "="*70)
    print("ACTDR - INSTALLATION VERIFICATION")
    print("="*70)
    print()

    # Test imports
    imports_ok = test_imports()

    if imports_ok:
        print()
        # Test data fetching
        data_ok = test_data_fetch()

        print("="*70)
        if data_ok:
            print("✓ READY TO USE")
            print()
            print("Run your first analysis:")
            print("  python run_lag_analysis.py --pair VALE,GDX --days 5")
        else:
            print("⚠ Installation OK but data fetch failed")
            print("  (This is normal if markets are closed)")
        print("="*70)
        print()
    else:
        print("="*70)
        print("Please install missing dependencies and try again")
        print("="*70)
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Bloomberg Data Fetcher
Fetches intraday data using Bloomberg API (blpapi)

SETUP INSTRUCTIONS:
===================
1. Install Bloomberg API:
   pip install blpapi

2. Ensure Bloomberg Terminal is running on your machine

3. Test connection:
   python -c "import blpapi; print('Bloomberg API installed successfully')"

USAGE:
======
from data_fetcher_bloomberg import BloombergDataFetcher

fetcher = BloombergDataFetcher()
data = fetcher.get_intraday_data('VALE US Equity', start_date='2025-01-15', end_date='2025-01-20', interval=1)
"""

import blpapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class BloombergDataFetcher:
    """
    Fetch intraday tick/bar data from Bloomberg Terminal
    """

    def __init__(self, host='localhost', port=8194):
        """
        Initialize Bloomberg connection

        Parameters:
        -----------
        host : str
            Bloomberg API host (default: localhost)
        port : int
            Bloomberg API port (default: 8194)
        """
        self.host = host
        self.port = port
        self.session = None

    def connect(self):
        """Establish connection to Bloomberg"""
        try:
            sessionOptions = blpapi.SessionOptions()
            sessionOptions.setServerHost(self.host)
            sessionOptions.setServerPort(self.port)

            self.session = blpapi.Session(sessionOptions)

            if not self.session.start():
                raise Exception("Failed to start Bloomberg session")

            if not self.session.openService("//blp/refdata"):
                raise Exception("Failed to open //blp/refdata service")

            print("✓ Connected to Bloomberg Terminal")
            return True

        except Exception as e:
            print(f"✗ Bloomberg connection failed: {e}")
            print("\nTroubleshooting:")
            print("  1. Is Bloomberg Terminal running?")
            print("  2. Is blpapi installed? (pip install blpapi)")
            print("  3. Try restarting Bloomberg Terminal")
            return False

    def disconnect(self):
        """Close Bloomberg connection"""
        if self.session:
            self.session.stop()
            print("✓ Disconnected from Bloomberg")

    def get_intraday_bars(self, ticker, start_datetime, end_datetime, interval_minutes=1):
        """
        Fetch intraday bar data from Bloomberg

        Parameters:
        -----------
        ticker : str
            Bloomberg ticker (e.g., 'VALE US Equity', 'COPX US Equity')
        start_datetime : str or datetime
            Start date/time (format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS')
        end_datetime : str or datetime
            End date/time
        interval_minutes : int
            Bar interval in minutes (1, 5, 15, 30, 60)

        Returns:
        --------
        pd.DataFrame with columns: [time, open, high, low, close, volume]
        """
        if not self.session:
            if not self.connect():
                raise Exception("Cannot connect to Bloomberg")

        # Convert to datetime if string
        if isinstance(start_datetime, str):
            start_datetime = pd.to_datetime(start_datetime)
        if isinstance(end_datetime, str):
            end_datetime = pd.to_datetime(end_datetime)

        refDataService = self.session.getService("//blp/refdata")
        request = refDataService.createRequest("IntradayBarRequest")

        request.set("security", ticker)
        request.set("eventType", "TRADE")
        request.set("interval", interval_minutes)  # Minutes

        # Set date range
        request.set("startDateTime", start_datetime)
        request.set("endDateTime", end_datetime)

        print(f"Fetching {ticker} from {start_datetime} to {end_datetime} ({interval_minutes}min bars)...")

        self.session.sendRequest(request)

        # Process response
        data = []
        while True:
            event = self.session.nextEvent(500)

            if event.eventType() == blpapi.Event.RESPONSE or \
               event.eventType() == blpapi.Event.PARTIAL_RESPONSE:

                for msg in event:
                    if msg.hasElement("barData"):
                        barData = msg.getElement("barData")
                        barTickData = barData.getElement("barTickData")

                        for bar in barTickData.values():
                            data.append({
                                'time': bar.getElementAsDatetime("time"),
                                'open': bar.getElementAsFloat("open"),
                                'high': bar.getElementAsFloat("high"),
                                'low': bar.getElementAsFloat("low"),
                                'close': bar.getElementAsFloat("close"),
                                'volume': bar.getElementAsInteger("volume")
                            })

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        df = pd.DataFrame(data)
        if len(df) > 0:
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)

        print(f"✓ Retrieved {len(df)} bars")
        return df

    def get_tick_data(self, ticker, start_datetime, end_datetime):
        """
        Fetch tick-by-tick data from Bloomberg

        Parameters:
        -----------
        ticker : str
            Bloomberg ticker
        start_datetime : str or datetime
            Start date/time
        end_datetime : str or datetime
            End date/time

        Returns:
        --------
        pd.DataFrame with tick data
        """
        if not self.session:
            if not self.connect():
                raise Exception("Cannot connect to Bloomberg")

        # Convert to datetime if string
        if isinstance(start_datetime, str):
            start_datetime = pd.to_datetime(start_datetime)
        if isinstance(end_datetime, str):
            end_datetime = pd.to_datetime(end_datetime)

        refDataService = self.session.getService("//blp/refdata")
        request = refDataService.createRequest("IntradayTickRequest")

        request.set("security", ticker)
        request.getElement("eventTypes").appendValue("TRADE")
        request.set("startDateTime", start_datetime)
        request.set("endDateTime", end_datetime)
        request.set("includeConditionCodes", False)

        print(f"Fetching tick data for {ticker}...")

        self.session.sendRequest(request)

        # Process response
        data = []
        while True:
            event = self.session.nextEvent(500)

            if event.eventType() == blpapi.Event.RESPONSE or \
               event.eventType() == blpapi.Event.PARTIAL_RESPONSE:

                for msg in event:
                    if msg.hasElement("tickData"):
                        tickData = msg.getElement("tickData")
                        tickDataArray = tickData.getElement("tickData")

                        for tick in tickDataArray.values():
                            data.append({
                                'time': tick.getElementAsDatetime("time"),
                                'value': tick.getElementAsFloat("value"),
                                'size': tick.getElementAsInteger("size") if tick.hasElement("size") else 0
                            })

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        df = pd.DataFrame(data)
        if len(df) > 0:
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)

        print(f"✓ Retrieved {len(df)} ticks")
        return df


def fetch_pair_data_bloomberg(ticker1, ticker2, start_date, end_date, interval_minutes=1):
    """
    Convenience function to fetch data for a pair of assets

    Parameters:
    -----------
    ticker1 : str
        First Bloomberg ticker (e.g., 'VALE US Equity')
    ticker2 : str
        Second Bloomberg ticker (e.g., 'GDX US Equity')
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    interval_minutes : int
        Bar interval in minutes

    Returns:
    --------
    tuple: (data1, data2) - DataFrames for both assets
    """
    fetcher = BloombergDataFetcher()

    try:
        fetcher.connect()

        # Fetch data for both tickers
        data1 = fetcher.get_intraday_bars(ticker1, start_date, end_date, interval_minutes)
        data2 = fetcher.get_intraday_bars(ticker2, start_date, end_date, interval_minutes)

        return data1, data2

    finally:
        fetcher.disconnect()


# Bloomberg ticker mapping for common assets
BLOOMBERG_TICKERS = {
    # Emerging Market ETFs
    'EWZ': 'EWZ US Equity',
    'EWW': 'EWW US Equity',
    'FXI': 'FXI US Equity',
    'MCHI': 'MCHI US Equity',
    'INDA': 'INDA US Equity',
    'EWT': 'EWT US Equity',
    'EPU': 'EPU US Equity',
    'ECH': 'ECH US Equity',
    'EEM': 'EEM US Equity',
    'VWO': 'VWO US Equity',

    # Brazilian Stocks
    'VALE': 'VALE US Equity',
    'PBR': 'PBR US Equity',
    'ITUB': 'ITUB US Equity',
    'BBD': 'BBD US Equity',
    'SID': 'SID US Equity',
    'NU': 'NU US Equity',

    # Chinese Stocks
    'BABA': 'BABA US Equity',
    'JD': 'JD US Equity',
    'BIDU': 'BIDU US Equity',
    'NIO': 'NIO US Equity',
    'XPEV': 'XPEV US Equity',

    # Commodities
    'GLD': 'GLD US Equity',
    'SLV': 'SLV US Equity',
    'GDX': 'GDX US Equity',
    'GDXJ': 'GDXJ US Equity',
    'COPX': 'COPX US Equity',
    'PICK': 'PICK US Equity',
    'USO': 'USO US Equity',
    'XLE': 'XLE US Equity',

    # Add more as needed
}


def get_bloomberg_ticker(symbol):
    """
    Convert simple symbol to Bloomberg ticker format

    Parameters:
    -----------
    symbol : str
        Simple ticker symbol (e.g., 'VALE')

    Returns:
    --------
    str : Bloomberg ticker (e.g., 'VALE US Equity')
    """
    return BLOOMBERG_TICKERS.get(symbol, f"{symbol} US Equity")


if __name__ == "__main__":
    """
    Test Bloomberg connection
    """
    print("Testing Bloomberg connection...")
    print("="*70)

    fetcher = BloombergDataFetcher()

    if fetcher.connect():
        print("\n✓ Bloomberg connection successful!")
        print("\nTo fetch data, use:")
        print("  data = fetcher.get_intraday_bars('VALE US Equity', '2025-01-15', '2025-01-20', interval_minutes=1)")
        fetcher.disconnect()
    else:
        print("\n✗ Bloomberg connection failed")
        print("\nMake sure:")
        print("  1. Bloomberg Terminal is running")
        print("  2. You have installed: pip install blpapi")

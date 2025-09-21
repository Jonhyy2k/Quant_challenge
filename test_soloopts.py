# ======================= Bloomberg Historical Option Data Tester =======================
# This script is a simplified test to verify the functionality of a
# Bloomberg API connection for fetching historical options data.
# It isolates the core request logic to confirm that the issue is not with
# the API syntax, but likely with the backtesting strategy's filtering.

import blpapi
import pandas as pd
import numpy as np

# -------------------- Bloomberg session helpers (reused from your code) --------------------
def _start_bbg_session(host="localhost", port=8194):
    """Starts and opens a Bloomberg session."""
    so = blpapi.SessionOptions()
    so.setServerHost(host)
    so.setServerPort(port)
    s = blpapi.Session(so)
    if not s.start():
        raise RuntimeError("Failed to start Bloomberg session. Check if the Bloomberg Terminal is running.")
    if not s.openService("//blp/refdata"):
        raise RuntimeError("Failed to open //blp/refdata service.")
    return s, s.getService("//blp/refdata")

def _send_req(session, req, timeout_ms=120_000):
    """Sends a request and collects all messages from the response event."""
    session.sendRequest(req)
    msgs = []
    while True:
        ev = session.nextEvent(timeout_ms)
        for m in ev:
            msgs.append(m)
        if ev.eventType() == blpapi.Event.RESPONSE:
            break
    return msgs

# -------------------- Safe extractor for blpapi.Element (reused from your code) --------------------
def _blp_get(parent_el: blpapi.Element, field_name: str):
    """Safely extracts a value from a blpapi Element, returning NaN if not found."""
    if not parent_el.hasElement(field_name):
        return np.nan
    sub = parent_el.getElement(field_name)
    for meth in ("getValueAsFloat64", "getValueAsInt64", "getValueAsInteger"):
        try:
            return getattr(sub, meth)()
        except Exception:
            pass
    for meth in ("getValueAsBool", "getValueAsDatetime", "getValueAsString"):
        try:
            val = getattr(sub, meth)()
            if isinstance(val, blpapi.Datetime):
                return pd.to_datetime(val.strftime("%Y-%m-%d"))
            return val
        except Exception:
            pass
    try:
        return sub.getValue()
    except Exception:
        return np.nan

# -------------------- Function to fetch historical option data --------------------
def get_historical_option_data(option_ticker, start_date, end_date, fields=("PX_LAST",)):
    """
    Fetches historical data for a specific, pre-constructed option ticker.
    This function isolates the core API request logic.
    """
    s, ref = _start_bbg_session()
    try:
        r = ref.createRequest("HistoricalDataRequest")
        r.getElement("securities").appendValue(option_ticker)
        fe = r.getElement("fields")
        for f in fields:
            fe.appendValue(f)
        r.set("startDate", start_date)
        r.set("endDate", end_date)
        r.set("periodicitySelection", "DAILY")
        
        print(f"Requesting data for: {option_ticker} from {start_date} to {end_date}")
        msgs = _send_req(s, r)

        rows = []
        for msg in msgs:
            if not msg.hasElement("securityData"):
                continue
            sd = msg.getElement("securityData")
            if sd.hasElement("securityError"):
                print(f"Bloomberg security error: {sd.getElementAsString('securityError')}")
                continue
            
            if not sd.hasElement("fieldData"):
                print(f"No field data for {option_ticker}. It may not have traded on these dates.")
                continue

            fd = sd.getElement("fieldData")
            for i in range(fd.numValues()):
                row_el = fd.getValueAsElement(i)
                d = row_el.getElementAsDatetime("date")
                out = {"date": pd.to_datetime(d.strftime("%Y-%m-%d"))}
                for f in fields:
                    val = _blp_get(row_el, f)
                    out[f] = val if val is not None else np.nan
                rows.append(out)

        df = pd.DataFrame(rows)
        if df.empty:
            print("No historical data found for this option. It might be illiquid or the ticker is incorrect.")
            return pd.DataFrame()
            
        df = df.sort_values("date").set_index("date")
        for f in fields:
            try:
                df[f] = pd.to_numeric(df[f])
            except Exception:
                pass
        return df
    except RuntimeError as e:
        print(f"API Connection Error: {e}")
        return pd.DataFrame()
    finally:
        s.stop()

if __name__ == "__main__":

    test_ticker = "AAPL US 10/18/24 C200 Equity"
    
    test_start = "20240901"
    test_end = "20240930"
    
    print(f"=== Running test for a single, specific option ticker ===")
    
    historical_data = get_historical_option_data(test_ticker, test_start, test_end)
    
    print("\n--- TEST RESULTS ---")
    if not historical_data.empty:
        print("Success! Data was fetched.")
        print(historical_data.head())
        print(f"\nFetched {len(historical_data)} historical data points.")
    else:
        print("Failed to fetch data. Check your Bloomberg Terminal connection and the ticker syntax.")

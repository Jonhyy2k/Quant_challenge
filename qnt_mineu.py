# ======================= FIXED Bloomberg Parity Backtest =======================
# Critical fixes for option chain retrieval and data handling

import blpapi, pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta
from pathlib import Path

# -------------------- Bloomberg session helpers --------------------
def _start_bbg_session(host="localhost", port=8194):
    so = blpapi.SessionOptions()
    so.setServerHost(host); so.setServerPort(port)
    s = blpapi.Session(so)
    if not s.start(): raise RuntimeError("Failed to start Bloomberg session.")
    if not s.openService("//blp/refdata"): raise RuntimeError("Failed to open //blp/refdata.")
    return s, s.getService("//blp/refdata")

def _send_req(session, req, timeout_ms=120_000):
    session.sendRequest(req)
    msgs = []
    while True:
        ev = session.nextEvent(timeout_ms)
        for m in ev: msgs.append(m)
        if ev.eventType() == blpapi.Event.RESPONSE:
            break
    return msgs

# -------------------- Safe extractor for blpapi.Element --------------------
def _blp_get(parent_el: blpapi.Element, field_name: str):
    if not parent_el.hasElement(field_name):
        return np.nan
    sub = parent_el.getElement(field_name)
    for meth in ("getValueAsFloat64","getValueAsInt64","getValueAsInteger"):
        try: return getattr(sub, meth)()
        except Exception: pass
    for meth in ("getValueAsBool","getValueAsDatetime","getValueAsString"):
        try:
            val = getattr(sub, meth)()
            if isinstance(val, blpapi.Datetime):
                return pd.to_datetime(val.strftime("%Y-%m-%d"))
            return val
        except Exception: pass
    try: return sub.getValue()
    except Exception: return np.nan

# -------------------- BDH: historical for any security --------------------
def get_bdh_equity(symbol_bbg, start_yyyymmdd, end_yyyymmdd, fields=("PX_LAST",), periodicity="DAILY"):
    s, ref = _start_bbg_session()
    try:
        r = ref.createRequest("HistoricalDataRequest")
        r.getElement("securities").appendValue(symbol_bbg)
        fe = r.getElement("fields")
        for f in fields: fe.appendValue(f)
        r.set("startDate", start_yyyymmdd)
        r.set("endDate", end_yyyymmdd)
        r.set("periodicitySelection", periodicity)
        msgs = _send_req(s, r)

        rows = []
        for msg in msgs:
            if not msg.hasElement("securityData"): continue
            sd = msg.getElement("securityData")
            if not sd.hasElement("fieldData"): continue
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
            return pd.DataFrame(columns=list(fields)).astype(float)
        df = df.sort_values("date").set_index("date")
        for f in fields:
            try: df[f] = pd.to_numeric(df[f])
            except Exception: pass
        return df
    finally:
        s.stop()

def get_riskfree_series(start_yyyymmdd, end_yyyymmdd, ticker="USGG3M Index"):
    rf = get_bdh_equity(ticker, start_yyyymmdd, end_yyyymmdd, fields=("PX_LAST",))
    rf = rf.rename(columns={"PX_LAST":"RF_YLD_PCT"})
    rf["rf_daily"] = (rf["RF_YLD_PCT"]/100.0)/365.0
    return rf[["rf_daily"]]

# -------------------- FIXED Option chain retrieval --------------------
def get_chain_for_day(
    underlying_bbg,
    spot_px,
    trade_date,
    target_min_dte=25,
    target_max_dte=45,
    max_points=None,
    verbose=True
):
    """
    FIXED version with proper Bloomberg option field handling
    """
    import pandas as pd, numpy as np, blpapi
    from datetime import timedelta

    def _bbg_base(under):
        parts = under.split()
        if len(parts) < 3:
            raise ValueError(f"Unexpected underlying format: {under}")
        return f"{parts[0]} {parts[1]}"

    def _fridays_between(start_dt, min_dte, max_dte):
        fr = []
        for d in range(min_dte, max_dte + 1):
            dt = start_dt + timedelta(days=d)
            if dt.weekday() == 4:  # Friday
                fr.append(dt)
        return fr

    def _third_fridays(start_dt, count=6):
        exps, dt = [], start_dt
        while len(exps) < count:
            y, m = dt.year + (dt.month // 12), (dt.month % 12) + 1
            first_next = pd.Timestamp(year=y, month=m, day=1)
            wd = first_next.weekday()
            first_friday = first_next + timedelta(days=(4 - wd) % 7)
            third_friday = first_friday + timedelta(days=14)
            exps.append(third_friday)
            dt = third_friday
        return exps

    def _strike_increment(S):
        if S < 25:   return 0.5
        if S < 200:  return 1.0
        if S < 500:  return 2.5
        if S < 1000: return 5.0
        return 10.0

    def _atm_ladder(S, steps=3):  # Reduced steps to limit query size
        inc = _strike_increment(S)
        k0  = round(S / inc) * inc
        ks  = [k0 + i*inc for i in range(-steps, steps+1)]
        return sorted(set([round(k, 2) for k in ks]))

    def _format_bbg_opt(base, exp_dt, cp, strike):
        # Try standard US equity option format
        mmddyy = pd.Timestamp(exp_dt).strftime("%m/%d/%y")
        if abs(strike - round(strike)) < 1e-6:
            strike_txt = str(int(round(strike)))
        else:
            strike_txt = f"{strike:.2f}".rstrip("0").rstrip(".")
        return f"{base} {mmddyy} {cp}{strike_txt} Equity"

    # Main logic
    base = _bbg_base(underlying_bbg)
    trade_dt = pd.Timestamp(trade_date)

    # Get expiries - limit to reduce query size
    expiry_fridays = _fridays_between(trade_dt, target_min_dte, target_max_dte)
    expiry_monthlies = _third_fridays(trade_dt, count=3)  # Reduced count
    expiries = sorted(set(expiry_fridays + expiry_monthlies))[:8]  # Further limit

    if verbose:
        print(f"[chain] expiries: {len(expiries)} (DTE {target_min_dte}-{target_max_dte})")

    if not expiries:
        return pd.DataFrame()

    strikes = _atm_ladder(spot_px, steps=3)  # Reduced strikes
    if verbose:
        print(f"[chain] strikes around {spot_px:.2f}: {strikes}")

    # Build security list
    sec_list, meta = [], []
    for exp in expiries:
        for cp in ("C", "P"):
            for K in strikes:
                sec = _format_bbg_opt(base, exp, cp, K)
                sec_list.append(sec)
                meta.append({"sec": sec, "cp": cp, "K": float(K), "expiry": pd.Timestamp(exp)})

    if verbose:
        print(f"[chain] querying {len(sec_list)} options")

    # FIXED: Use consistent field names and better error handling
    s, ref = _start_bbg_session()
    try:
        # Standard Bloomberg option fields
        fields = [
            "PX_BID", "PX_ASK", "PX_LAST",  # Consistent pricing fields
            "OPT_STRIKE_PX", "OPT_EXPIRE_DT", "OPT_PUT_CALL",
            "IVOL_MID"  # Simplified IV field
        ]
        
        bdp = ref.createRequest("ReferenceDataRequest")
        se = bdp.getElement("securities")
        for sec in sec_list: se.appendValue(sec)
        fe = bdp.getElement("fields")
        for f in fields: fe.appendValue(f)
        
        msgs = _send_req(s, bdp)

        rows = []
        meta_map = {m["sec"]: m for m in meta}
        
        valid_count = 0
        price_count = 0

        for msg in msgs:
            if not msg.hasElement("securityData"): continue
            sdata = msg.getElement("securityData")
            
            for i in range(sdata.numValues()):
                e = sdata.getValueAsElement(i)
                sec = e.getElementAsString("security")
                
                # Check for field errors
                if e.hasElement("fieldExceptions"):
                    if verbose:
                        print(f"[chain] Field errors for {sec}")
                    continue
                    
                if not e.hasElement("fieldData"): continue
                fd = e.getElement("fieldData")
                valid_count += 1

                # Extract data with better error handling
                bid = _blp_get(fd, "PX_BID")
                ask = _blp_get(fd, "PX_ASK")
                last = _blp_get(fd, "PX_LAST")
                
                # Pricing logic: prefer bid/ask, fallback to last
                mid = np.nan
                if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
                    mid = (float(bid) + float(ask)) / 2.0
                    price_count += 1
                elif pd.notna(last) and last > 0:
                    mid = float(last)
                    price_count += 1

                if pd.isna(mid) or mid <= 0:
                    continue  # Skip if no valid price

                # Get option details
                K_val = _blp_get(fd, "OPT_STRIKE_PX")
                E_val = _blp_get(fd, "OPT_EXPIRE_DT")
                PC_val = _blp_get(fd, "OPT_PUT_CALL")
                iv_val = _blp_get(fd, "IVOL_MID")

                # Use metadata as fallback
                meta_row = meta_map.get(sec, {})
                K = float(K_val) if pd.notna(K_val) else meta_row.get("K", np.nan)
                E = (E_val if isinstance(E_val, pd.Timestamp) else 
                     (pd.to_datetime(E_val) if pd.notna(E_val) else meta_row.get("expiry", np.nan)))
                cp = (str(PC_val).upper() if pd.notna(PC_val) else meta_row.get("cp", ""))

                rows.append({
                    "security": sec,
                    "OPT_STRIKE_PX": K,
                    "OPT_EXPIRE_DT": E,
                    "OPT_PUT_CALL": cp,
                    "PX_BID": float(bid) if pd.notna(bid) else np.nan,
                    "PX_ASK": float(ask) if pd.notna(ask) else np.nan,
                    "PX_LAST": float(last) if pd.notna(last) else np.nan,
                    "MID": mid,
                    "IMPLIED_VOLATILITY": float(iv_val) if pd.notna(iv_val) else np.nan
                })

        if verbose:
            print(f"[chain] valid responses: {valid_count}, with prices: {price_count}")

        df = pd.DataFrame(rows)
        if df.empty:
            if verbose: print("[chain] No valid options with prices")
            return pd.DataFrame()

        # Clean and filter
        df = df.dropna(subset=["OPT_STRIKE_PX", "OPT_EXPIRE_DT", "OPT_PUT_CALL", "MID"]).copy()
        df["OPT_EXPIRE_DT"] = pd.to_datetime(df["OPT_EXPIRE_DT"])
        df["DTE"] = (df["OPT_EXPIRE_DT"] - pd.Timestamp(trade_date)).dt.days
        df = df[(df["DTE"] >= target_min_dte) & (df["DTE"] <= target_max_dte)]
        
        if df.empty:
            if verbose: print("[chain] No options in DTE range")
            return pd.DataFrame()

        # Find ATM pairs
        df["atm_dist"] = (df["OPT_STRIKE_PX"] - spot_px).abs()
        df["OPT_PUT_CALL"] = df["OPT_PUT_CALL"].str.upper()
        
        calls = df[df["OPT_PUT_CALL"] == "C"].sort_values(["atm_dist", "DTE"])
        puts = df[df["OPT_PUT_CALL"] == "P"].sort_values(["atm_dist", "DTE"])
        
        if calls.empty or puts.empty:
            if verbose: print("[chain] Missing calls or puts")
            return pd.DataFrame()

        # Get best ATM pair
        best_call = calls.iloc[0]
        best_put = puts.iloc[0]

        if verbose:
            print(f"[chain] Selected: {best_call['OPT_EXPIRE_DT'].date()}, "
                  f"K={best_call['OPT_STRIKE_PX']:.2f}, "
                  f"C_mid={best_call['MID']:.3f}, P_mid={best_put['MID']:.3f}")

        return pd.DataFrame([{
            "expiry": best_call["OPT_EXPIRE_DT"],
            "DTE": int(best_call["DTE"]),
            "call_tkr": best_call["security"],
            "put_tkr": best_put["security"],
            "K": float(best_call["OPT_STRIKE_PX"]),
            "C_mid": float(best_call["MID"]),
            "P_mid": float(best_put["MID"]),
            "IV_call": float(best_call["IMPLIED_VOLATILITY"]) if pd.notna(best_call["IMPLIED_VOLATILITY"]) else np.nan,
            "IV_put": float(best_put["IMPLIED_VOLATILITY"]) if pd.notna(best_put["IMPLIED_VOLATILITY"]) else np.nan
        }])

    finally:
        s.stop()

# -------------------- Quant helpers --------------------
def black_scholes_price(S, K, T, r, sigma, cp):
    d1 = (np.log(S/K) + (r + 0.5*sigma*sigma)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return (S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)) if cp.upper()=="C" else (K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1))

def realized_vol_from_close(close):
    ret = np.log(close/close.shift(1)).dropna()
    return ret.rolling(21).std()*np.sqrt(252)

# -------------------- IMPROVED Backtest --------------------
def backtest_parity(
    underlying="AAPL US Equity",
    start="20240101",
    end="20250918",
    min_dte=25, max_dte=45,
    gap_threshold=0.10,
    max_hold_days=60,
    riskfree_ticker="USGG3M Index",
    step_days=5,  # Reduced frequency for testing
    progress_every=1
):
    print(f"Starting backtest: {underlying} from {start} to {end}")
    
    # Get underlying and risk-free data
    px = get_bdh_equity(underlying, start, end, fields=("PX_LAST",))
    rf = get_riskfree_series(start, end, ticker=riskfree_ticker)
    df = px.join(rf, how="left").ffill()
    df["rv_21"] = realized_vol_from_close(df["PX_LAST"])

    print(f"Loaded {len(df)} days of underlying data")

    trades, equity_curve = [], []
    open_pos = None
    dates = df.index.to_list()

    for idx, dt in enumerate(dates):
        if step_days > 1 and idx % step_days != 0:
            equity_curve.append({"date": dt, "equity": 0.0 if not open_pos else open_pos["equity_mark"]})
            continue

        S = float(df.loc[dt, "PX_LAST"])
        rf_day = float(df.loc[dt, "rf_daily"]) if "rf_daily" in df.columns and pd.notna(df.loc[dt, "rf_daily"]) else 0.0

        # Update open position
        if open_pos:
            open_pos["days_held"] += step_days
            open_pos["cash"] *= (1.0 + rf_day) ** step_days
            equity_curve.append({"date": dt, "equity": open_pos["equity_mark"]})
        else:
            equity_curve.append({"date": dt, "equity": 0.0})

        # Check exit conditions
        if open_pos:
            if dt >= open_pos["expiry"] or open_pos["days_held"] >= max_hold_days:
                pnl = open_pos["gap_signed"]
                trades.append({**open_pos, "exit_date": dt, "pnl": pnl})
                print(f"[{dt.date()}] EXIT: PnL={pnl:.3f}, held={open_pos['days_held']} days")
                open_pos = None
            continue

        # Look for new opportunities
        if progress_every and idx % progress_every == 0:
            print(f"\n[{dt.date()}] Checking options... S=${S:.2f}")

        try:
            chain = get_chain_for_day(underlying, S, dt, min_dte, max_dte, verbose=(idx % 10 == 0))
        except Exception as e:
            if progress_every and idx % progress_every == 0:
                print(f"[{dt.date()}] Chain error: {e}")
            continue

        if chain.empty:
            continue

        # Calculate parity gap
        K = chain["K"].iloc[0]
        Cmid = chain["C_mid"].iloc[0]
        Pmid = chain["P_mid"].iloc[0]
        expiry = chain["expiry"].iloc[0]
        dte = int(chain["DTE"].iloc[0])
        T = dte/365.0

        r_ann = rf_day * 365.0
        dfac = np.exp(-r_ann * T)
        gap = (Cmid - Pmid) - (S - K * dfac)

        if abs(gap) < gap_threshold:
            if progress_every and idx % progress_every == 0:
                print(f"[{dt.date()}] Gap {gap:.3f} < threshold {gap_threshold}")
            continue

        # Enter position
        direction = "short_call_long_put" if gap > 0 else "long_call_short_put"
        open_pos = {
            "entry_date": dt, "expiry": expiry, "days_held": 0,
            "underlying": underlying,
            "S0": S, "K": K, "C_mid0": Cmid, "P_mid0": Pmid,
            "r_ann": r_ann, "T": T, "dte": dte,
            "gap": float(gap), "gap_signed": float(np.sign(gap) * abs(gap)),
            "direction": direction,
            "cash": K * dfac if gap > 0 else -K * dfac,
            "equity_mark": float(np.sign(gap) * abs(gap))
        }

        print(f"[{dt.date()}] ENTER {direction}: DTE={dte}, K=${K:.2f}, gap=${gap:.3f}")

    # Compile results
    eq = pd.DataFrame(equity_curve).set_index("date")
    tr = pd.DataFrame(trades)

    if not tr.empty:
        total_pnl = tr["pnl"].sum()
        hitrate = (tr["pnl"] > 0).mean()
        avg_hold = tr["days_held"].mean()
    else:
        total_pnl = 0.0; hitrate = np.nan; avg_hold = np.nan

    return {
        "underlying_history": df,
        "equity_curve": eq,
        "trades": tr,
        "summary": {
            "total_pnl_units": float(total_pnl),
            "num_trades": int(len(tr)),
            "hit_ratio": float(hitrate) if pd.notna(hitrate) else None,
            "avg_hold_days": float(avg_hold) if pd.notna(avg_hold) else None
        }
    }

# -------------------- Plot helper --------------------
def plot_backtest(res, title="Parity Arbitrage Backtest", outfile=None):
    if not isinstance(res, dict) or "summary" not in res or res["summary"].get("num_trades", 0) == 0:
        print("No trades — skipping plot.")
        return

    uh = res["underlying_history"].copy()
    tr = res["trades"].copy().set_index("exit_date").sort_index() if not res["trades"].empty else pd.DataFrame()
    daily = pd.Series(0.0, index=uh.index)
    if not tr.empty:
        for dt, row in tr.iterrows():
            if dt in daily.index:
                daily.loc[dt] += float(row["pnl"])
    cum_pnl = daily.cumsum()
    
    if outfile is None:
        outfile = Path(f"./backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    plt.figure(figsize=(12, 6))
    plt.plot(cum_pnl.index, cum_pnl.values, linewidth=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL")
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved chart: {outfile}")

# -------------------- MAIN EXECUTION --------------------
if __name__ == "__main__":
    # Test with more recent data and smaller scope
    underlying = "AAPL US Equity"
    start = "20240901"  # More recent data
    end = "20240930"    # Smaller range for testing
    min_dte, max_dte = 15, 45
    gap_threshold = 0.05

    print("=== Starting Bloomberg Parity Backtest ===")
    
    res = backtest_parity(
        underlying=underlying,
        start=start,
        end=end,
        min_dte=min_dte,
        max_dte=max_dte,
        gap_threshold=gap_threshold,
        step_days=3,  # Test every 3 days
        progress_every=1
    )
    
    print("\n=== RESULTS ===")
    print("Summary:", res["summary"])
    
    if res["summary"]["num_trades"] > 0:
        print("\nTrade Details:")
        print(res["trades"][["entry_date", "direction", "dte", "gap", "pnl", "days_held"]])
    
    outfile = f"{underlying.replace(' ','_')}_parity_test.png"
    plot_backtest(res, title=f"{underlying} Parity Test", outfile=outfile)
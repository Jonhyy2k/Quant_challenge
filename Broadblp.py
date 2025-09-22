import blpapi, pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta
from pathlib import Path

# -------------------- Bloomberg session helpers --------------------
def _start_bbg_session(host="localhost", port=8194):
    so = blpapi.SessionOptions()
    so.setServerHost(host); so.setServerPort(port)
    s = blpapi.Session(so)
    if not s.start(): raise RuntimeError("Failed to start Bloomberg session. Is the Bloomberg Terminal running?")
    if not s.openService("//blp/refdata"): raise RuntimeError("Failed to open //blp/refdata. Check your permissions.")
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
        except Exception: return np.nan
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
            print(f"Warning: No historical data returned for {symbol_bbg}")
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
    if rf.empty:
        print(f"Warning: No risk-free data for {ticker}. Assuming 0.")
        return pd.DataFrame(columns=["rf_daily"], index=rf.index, data=0.0)
    rf = rf.rename(columns={"PX_LAST":"RF_YLD_PCT"})
    rf["rf_daily"] = (rf["RF_YLD_PCT"]/100.0)/365.0
    return rf[["rf_daily"]]

# Findd all pairs instead of a specific option
def find_all_pairs_for_day(
    underlying_bbg,
    spot_px,
    trade_date,
    target_min_dte=20,
    target_max_dte=60,
    verbose=True
):
    
    def _bbg_base(under):
        return f"{under.split()[0]} {under.split()[1]}"
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
    def _atm_ladder(S, steps=20):
        inc = _strike_increment(S)
        k0 = round(S / inc) * inc
        ks = [k0 + i*inc for i in range(-steps, steps + 1)]
        return sorted(set([round(k, 2) for k in ks]))
    def _format_bbg_opt(base, exp_dt, cp, strike):
        mmddyy = pd.Timestamp(exp_dt).strftime("%m/%d/%y")
        if abs(strike - round(strike)) < 1e-6:
            strike_txt = str(int(round(strike)))
        else:
            strike_txt = f"{strike:.2f}".rstrip("0").rstrip(".")
        return f"{base} {mmddyy} {cp}{strike_txt} Equity"

    base = _bbg_base(underlying_bbg)
    trade_dt = pd.Timestamp(trade_date)
    expiries = _third_fridays(trade_dt, count=6)
    if verbose: print(f"[chain] expiries for {trade_date}: {len(expiries)}")
    
    strikes = _atm_ladder(spot_px, steps=30) 
    if verbose: print(f"[chain] searching for {len(strikes)} strikes around {spot_px:.2f}")

    sec_list, meta = [], []
    for exp in expiries:
        for cp in ("C", "P"):
            for K in strikes:
                sec = _format_bbg_opt(base, exp, cp, K)
                sec_list.append(sec)
                meta.append({"sec": sec, "cp": cp, "K": float(K), "expiry": pd.Timestamp(exp)})
    if verbose: print(f"[chain] querying {len(sec_list)} options")
    
    s, ref = _start_bbg_session()
    try:
        fields = ["PX_BID", "PX_ASK", "OPT_STRIKE_PX", "OPT_EXPIRE_DT", "OPT_PUT_CALL"]
        bdp = ref.createRequest("ReferenceDataRequest")
        se = bdp.getElement("securities")
        for sec in sec_list: se.appendValue(sec)
        fe = bdp.getElement("fields")
        for f in fields: fe.appendValue(f)
        msgs = _send_req(s, bdp)
        
        rows = []
        for msg in msgs:
            if not msg.hasElement("securityData"): continue
            sdata = msg.getElement("securityData")
            for i in range(sdata.numValues()):
                e = sdata.getValueAsElement(i)
                if e.hasElement("fieldExceptions") or not e.hasElement("fieldData"): continue
                fd = e.getElement("fieldData")
                bid, ask = _blp_get(fd, "PX_BID"), _blp_get(fd, "PX_ASK")
                if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
                    mid = (float(bid) + float(ask)) / 2.0
                    rows.append({
                        "security": e.getElementAsString("security"),
                        "K": _blp_get(fd, "OPT_STRIKE_PX"),
                        "expiry": _blp_get(fd, "OPT_EXPIRE_DT"),
                        "cp": _blp_get(fd, "OPT_PUT_CALL").upper(),
                        "mid": mid
                    })
        df_options = pd.DataFrame(rows)
        if df_options.empty: return pd.DataFrame()

        df_options["expiry"] = pd.to_datetime(df_options["expiry"])
        df_options["dte"] = (df_options["expiry"] - pd.Timestamp(trade_date)).dt.days
        df_options = df_options[(df_options["dte"] >= target_min_dte) & (df_options["dte"] <= target_max_dte)]
        
        pairs = []
        unique_expiries = df_options["expiry"].unique()
        for exp in unique_expiries:
            exp_df = df_options[df_options["expiry"] == exp]
            calls = exp_df[exp_df["cp"] == "C"].set_index("K")
            puts = exp_df[exp_df["cp"] == "P"].set_index("K")
            
            common_strikes = calls.index.intersection(puts.index)
            
            for K in common_strikes:
                pair = {
                    "K": K,
                    "expiry": exp,
                    "dte": calls.loc[K, "dte"],
                    "C_mid": calls.loc[K, "mid"],
                    "P_mid": puts.loc[K, "mid"],
                    "call_tkr": calls.loc[K, "security"],
                    "put_tkr": puts.loc[K, "security"]
                }
                pairs.append(pair)
        
        if verbose: print(f"[chain] Found {len(pairs)} suitable pairs.")
        return pd.DataFrame(pairs)
    finally:
        s.stop()

def backtest_parity(
    underlying="PETR4 BZ Equity",
    start="20220101",
    end="20240901",
    min_dte=20,
    max_dte=60,
    gap_threshold=0.05,
    catalyst_threshold=0.02,
    riskfree_ticker="USGG3M Index",
    progress_every=10
):
    print(f"Starting catalyst-driven backtest: {underlying} from {start} to {end}")
    
    px = get_bdh_equity(underlying, start, end, fields=("PX_LAST",))
    rf = get_riskfree_series(start, end, ticker=riskfree_ticker)
    df = px.join(rf, how="left").ffill()
    if df.empty:
        print("Underlying data is empty. Cannot perform backtest.")
        return { "summary": {"num_trades": 0} }
    
    df["daily_ret"] = df["PX_LAST"].pct_change()
    print(f"Loaded {len(df)} days of underlying data")
    
    trades = []
    dates = df.index.to_list()
    
    for idx, dt in enumerate(dates):
        if idx == 0: continue
        
        daily_return = df.loc[dt, "daily_ret"]
        if abs(daily_return) < catalyst_threshold:
            continue

        if progress_every and idx % progress_every == 0:
            print(f"\n[{dt.date()}] Catalyst detected! Daily return: {daily_return:.2%}. Checking options...")

        S = float(df.loc[dt, "PX_LAST"])
        rf_ann = float(df.loc[dt, "rf_daily"]) * 365.0 if "rf_daily" in df.columns and pd.notna(df.loc[dt, "rf_daily"]) else 0.0

        try:
            available_pairs = find_all_pairs_for_day(underlying, S, dt, min_dte, max_dte, verbose=(idx % 10 == 0))
        except Exception as e:
            if verbose:
                print(f"[{dt.date()}] Chain error: {e}")
            continue

        if available_pairs.empty:
            print(f"[{dt.date()}] No suitable option pairs found after catalyst. Moving on.")
            continue
        
        best_gap = 0
        best_pair = None
        for i, row in available_pairs.iterrows():
            K, Cmid, Pmid, dte = row["K"], row["C_mid"], row["P_mid"], row["dte"]
            T = dte/365.0
            dfac = np.exp(-rf_ann * T)
            gap = (Cmid - Pmid) - (S - K * dfac)
            
            if abs(gap) > abs(best_gap):
                best_gap = gap
                best_pair = row

        if best_pair is None or abs(best_gap) < gap_threshold:
            print(f"[{dt.date()}] Best gap {best_gap:.3f} < threshold {gap_threshold}. No trade.")
            continue

        direction = "short_call_long_put" if best_gap > 0 else "long_call_short_put"
        trades.append({
            "entry_date": dt, "expiry": best_pair["expiry"], "K": best_pair["K"],
            "C_mid": best_pair["C_mid"], "P_mid": best_pair["P_mid"], "gap": float(best_gap),
            "direction": direction, "pnl": float(np.sign(best_gap) * abs(best_gap))
        })
        print(f"[{dt.date()}] ENTER {direction}: DTE={best_pair['dte']}, K=${best_pair['K']:.2f}, gap=${best_gap:.3f}")

    tr = pd.DataFrame(trades)
    if not tr.empty:
        total_pnl = tr["pnl"].sum()
        hitrate = (tr["pnl"] > 0).mean()
        avg_hold = np.nan
    else:
        total_pnl = 0.0; hitrate = np.nan; avg_hold = np.nan
    return {
        "underlying_history": df,
        "trades": tr,
        "summary": { "total_pnl_units": float(total_pnl), "num_trades": int(len(tr)),
                     "hit_ratio": float(hitrate) if pd.notna(hitrate) else None,
                     "avg_hold_days": float(avg_hold) if pd.notna(avg_hold) else None }
    }

def plot_backtest(res, title="Parity Arbitrage Backtest", outfile=None):
    if not isinstance(res, dict) or "summary" not in res or res["summary"].get("num_trades", 0) == 0:
        print("No trades — skipping plot.")
        return
    tr = res["trades"].copy() if not res["trades"].empty else pd.DataFrame()
    if tr.empty:
      print("No trades — skipping plot.")
      return

    tr = tr.set_index("entry_date").sort_index()
    cum_pnl = tr["pnl"].cumsum().fillna(method='ffill')
    
    if outfile is None:
        outfile = Path(f"./backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cumulative PnL", color=color)
    ax1.plot(cum_pnl.index, cum_pnl.values, color=color, linewidth=2, label="Cumulative PnL")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title(title, pad=20)
    ax1.legend(loc="upper left")
    
    for idx, row in tr.iterrows():
        ax1.axvline(x=row.name, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
    
    fig.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved chart: {outfile}")


if __name__ == "__main__":
    underlying = "PETR4 BZ Equity"
    start = "20220101"
    end = "20240901"
    min_dte, max_dte = 20, 60
    gap_threshold = 0.05
    catalyst_threshold = 0.02
    
    print("=== Starting Bloomberg Parity Backtest (Synthetic Pairs) ===")
    
    res = backtest_parity(
        underlying=underlying,
        start=start,
        end=end,
        min_dte=min_dte,
        max_dte=max_dte,
        gap_threshold=gap_threshold,
        catalyst_threshold=catalyst_threshold
    )
    
    print("\n=== RESULTS ===")
    print("Summary:", res["summary"])
    
    if res["summary"]["num_trades"] > 0:
        print("\nTrade Details:")
        print(res["trades"][["entry_date", "direction", "dte", "gap", "pnl"]])
    
    outfile = f"{underlying.replace(' ','_')}_parity_synthetic_test.png"
    plot_backtest(res, title=f"{underlying} Synthetic Pair Test", outfile=outfile)

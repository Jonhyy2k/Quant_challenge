# ======================= ALL-IN-ONE (Bloomberg parity backtest; robust BDP chain scan) =======================
# Works on a Bloomberg Terminal PC (Desktop API; localhost:8194)
# Public functions:
#   get_bdh_equity, get_chain_for_day, backtest_parity, plot_backtest

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

# -------------------- BDH: historical for any security (e.g., equity, rates) --------------------
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

# -------------------- Option chain (heavy-duty BDP scan; no CHAIN_TICKERS needed) --------------------
def get_chain_for_day(
    underlying_bbg,
    spot_px,
    trade_date,
    target_min_dte=25,
    target_max_dte=45,
    max_points=None,         # accepted for compatibility; not used
    verbose=True
):
    """
    Robust BDP scanner (no CHAIN_TICKERS):
      • Expiries: all Fridays in [min_dte, max_dte] + next 6 third-Fridays (monthlies)
      • Strikes: price-aware ATM ladder (0.5/1/2.5/5/10 increments), a few steps each side
      • Pricing: prefer BID/ASK -> MID; fallback to LAST_PRICE when quotes are unavailable

    Returns a 1-row DataFrame with the chosen ATM call/put, or empty DF if none.
    """
    import pandas as pd, numpy as np, blpapi
    from datetime import timedelta

    # ---- helpers ----
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

    def _atm_ladder(S, steps=5):
        inc = _strike_increment(S)
        k0  = round(S / inc) * inc
        ks  = [k0 + i*inc for i in range(-steps, steps+1)]
        coarse = [k0 + i*(5*inc) for i in range(-2, 3)]
        ks = sorted(set([round(k, 2) for k in ks + coarse]))
        return ks

    def _format_bbg_opt(base, exp_dt, cp, strike):
        mmddyy = pd.Timestamp(exp_dt).strftime("%m/%d/%y")
        if abs(strike - round(strike)) < 1e-6:
            strike_txt = str(int(round(strike)))
        else:
            strike_txt = f"{strike:.2f}".rstrip("0").rstrip(".")
        return f"{base} {mmddyy} {cp}{strike_txt} Equity"

    # ---- assemble candidates ----
    base = _bbg_base(underlying_bbg)
    trade_dt = pd.Timestamp(trade_date)

    expiry_fridays   = _fridays_between(trade_dt, target_min_dte, target_max_dte)
    expiry_monthlies = _third_fridays(trade_dt, count=6)
    expiries = sorted(set(expiry_fridays + expiry_monthlies))[:24]  # cap

    if verbose:
        print(f"[chain] expiries candidates: {len(expiries)} (min_dte={target_min_dte}, max_dte={target_max_dte})")

    if not expiries:
        return pd.DataFrame()

    strikes = _atm_ladder(spot_px, steps=5)
    if verbose:
        print(f"[chain] strike candidates around spot {spot_px:.2f}: {len(strikes)}")

    sec_list, meta = [], []
    for exp in expiries:
        for cp in ("C", "P"):
            for K in strikes:
                sec = _format_bbg_opt(base, exp, cp, K)
                sec_list.append(sec)
                meta.append({"sec": sec, "cp": cp, "K": float(K), "expiry": pd.Timestamp(exp)})

    if verbose:
        print(f"[chain] requesting {len(sec_list)} option securities via BDP")

    # ---- BDP request (use BID/ASK + LAST_PRICE) ----
    s, ref = _start_bbg_session()
    try:
        fields = [
            "BID","ASK","LAST_PRICE",
            "OPT_STRIKE_PX","OPT_EXPIRE_DT","OPT_PUT_CALL",
            "IVOL_MID","IMPLIED_VOLATILITY"
        ]
        bdp = ref.createRequest("ReferenceDataRequest")
        se = bdp.getElement("securities"); [se.appendValue(sec) for sec in sec_list]
        fe = bdp.getElement("fields");     [fe.appendValue(f)   for f   in fields]
        msgs = _send_req(s, bdp)

        rows = []
        meta_map = {m["sec"]: m for m in meta}

        def _get(fd, name):
            if not fd.hasElement(name): return np.nan
            sub = fd.getElement(name)
            for meth in ("getValueAsFloat64","getValueAsInt64","getValueAsInteger"):
                try: return getattr(sub, meth)()
                except Exception: pass
            for meth in ("getValueAsDatetime","getValueAsString"):
                try:
                    val = getattr(sub, meth)()
                    if isinstance(val, blpapi.Datetime):
                        return pd.to_datetime(val.strftime("%Y-%m-%d"))
                    return val
                except Exception: pass
            try: return sub.getValue()
            except Exception: return np.nan

        returned = 0
        with_quotes = 0
        with_last   = 0

        for msg in msgs:
            if not msg.hasElement("securityData"): continue
            sdata = msg.getElement("securityData")
            for i in range(sdata.numValues()):
                e = sdata.getValueAsElement(i)
                sec = e.getElementAsString("security")
                if not e.hasElement("fieldData"): continue
                fd = e.getElement("fieldData")
                returned += 1

                bid = pd.to_numeric(_get(fd, "BID"), errors="coerce")
                ask = pd.to_numeric(_get(fd, "ASK"), errors="coerce")
                last= pd.to_numeric(_get(fd, "LAST_PRICE"), errors="coerce")

                mid = np.nan
                if pd.notna(bid) and pd.notna(ask):
                    with_quotes += 1
                    mid = (float(bid) + float(ask)) / 2.0
                elif pd.notna(last):
                    with_last += 1
                    mid = float(last)

                if pd.isna(mid):
                    continue  # still nothing usable

                Kfd  = _get(fd, "OPT_STRIKE_PX")
                Efd  = _get(fd, "OPT_EXPIRE_DT")
                PCfd = _get(fd, "OPT_PUT_CALL")
                iv1  = _get(fd, "IVOL_MID")
                iv2  = _get(fd, "IMPLIED_VOLATILITY")

                meta_row = meta_map.get(sec, {})
                K = float(Kfd) if pd.notna(Kfd) else meta_row.get("K", np.nan)
                E = (Efd if isinstance(Efd, pd.Timestamp) else
                     (pd.to_datetime(Efd) if pd.notna(Efd) else meta_row.get("expiry", np.nan)))
                cp = (str(PCfd).upper() if isinstance(PCfd, str) else meta_row.get("cp", None))

                rows.append({
                    "security": sec,
                    "OPT_STRIKE_PX": K,
                    "OPT_EXPIRE_DT": E,
                    "OPT_PUT_CALL": cp,
                    "BID": float(bid) if pd.notna(bid) else np.nan,
                    "ASK": float(ask) if pd.notna(ask) else np.nan,
                    "LAST_PRICE": float(last) if pd.notna(last) else np.nan,
                    "MID": mid,
                    "IMPLIED_VOLATILITY": (pd.to_numeric(iv1, errors="coerce")
                                           if pd.notna(iv1) else pd.to_numeric(iv2, errors="coerce"))
                })

        if verbose:
            print(f"[chain] returned: {returned}, with quotes: {with_quotes}, with last-only: {with_last}")

        df = pd.DataFrame(rows)
        if df.empty:
            if verbose: print("[chain] No valid options returned via BDP.")
            return pd.DataFrame()

        # Clean & filter by DTE
        df = df.dropna(subset=["OPT_STRIKE_PX","OPT_EXPIRE_DT","OPT_PUT_CALL","MID"]).copy()
        df["OPT_EXPIRE_DT"] = pd.to_datetime(df["OPT_EXPIRE_DT"], errors="coerce")
        df["DTE"] = (df["OPT_EXPIRE_DT"] - pd.Timestamp(trade_date)).dt.days
        df = df[(df["DTE"]>=target_min_dte) & (df["DTE"]<=target_max_dte)]
        if df.empty:
            if verbose: print("[chain] All candidates filtered out by DTE window.")
            return pd.DataFrame()

        # Pair ATM per expiry
        df["atm_dist"] = (df["OPT_STRIKE_PX"] - spot_px).abs()
        df["OPT_PUT_CALL"] = df["OPT_PUT_CALL"].astype(str).str.upper()
        atmC = df[df["OPT_PUT_CALL"]=="C"].sort_values(["atm_dist","DTE"]).groupby("OPT_EXPIRE_DT").head(1)
        atmP = df[df["OPT_PUT_CALL"]=="P"].sort_values(["atm_dist","DTE"]).groupby("OPT_EXPIRE_DT").head(1)
        pair = atmC.merge(atmP, on="OPT_EXPIRE_DT", suffixes=("_C","_P"))
        if pair.empty:
            if verbose: print("[chain] Could not pair ATM call/put on any expiry.")
            return pd.DataFrame()
        pair["combo_dist"] = pair["atm_dist_C"] + pair["atm_dist_P"]
        best = pair.sort_values(["combo_dist","DTE_C"]).head(1)

        if verbose:
            print(f"[chain] picked expiry {best['OPT_EXPIRE_DT'].iloc[0].date()}, K≈{best['OPT_STRIKE_PX_C'].iloc[0]}")

        return pd.DataFrame([{
            "expiry":   best["OPT_EXPIRE_DT"].iloc[0],
            "DTE":      int(best["DTE_C"].iloc[0]),
            "call_tkr": best["security_C"].iloc[0],
            "put_tkr":  best["security_P"].iloc[0],
            "K":        float(best["OPT_STRIKE_PX_C"].iloc[0]),
            "C_mid":    float(best["MID_C"].iloc[0]),
            "P_mid":    float(best["MID_P"].iloc[0]),
            "IV_call":  float(best["IMPLIED_VOLATILITY_C"].iloc[0]) if pd.notna(best["IMPLIED_VOLATILITY_C"].iloc[0]) else np.nan,
            "IV_put":   float(best["IMPLIED_VOLATILITY_P"].iloc[0]) if pd.notna(best["IMPLIED_VOLATILITY_P"].iloc[0]) else np.nan
        }])
    finally:
        s.stop()
    # ---- helpers ----
    def _bbg_base(under):
        # "AAPL US Equity" -> "AAPL US"
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
        # next N standard monthly expiries (third Friday)
        exps, dt = [], start_dt
        while len(exps) < count:
            # move to first day next month
            y, m = dt.year + (dt.month // 12), (dt.month % 12) + 1
            first_next = pd.Timestamp(year=y, month=m, day=1)
            # third Friday of that month
            wd = first_next.weekday()
            # Friday index in that month: find first Friday then add 14 days
            first_friday = first_next + timedelta(days=(4 - wd) % 7)
            third_friday = first_friday + timedelta(days=14)
            exps.append(third_friday)
            dt = third_friday
        return exps

    def _strike_increment(S):
        # heuristic for US equity option grids
        if S < 25:   return 0.5
        if S < 200:  return 1.0
        if S < 500:  return 2.5
        if S < 1000: return 5.0
        return 10.0

    def _atm_ladder(S, steps=4):
        inc = _strike_increment(S)
        k0  = round(S / inc) * inc
        ks  = [k0 + i*inc for i in range(-steps, steps+1)]
        # also add a coarser grid around spot to catch odd OCC ladders
        coarse = [k0 + i*(5*inc) for i in range(-2, 3)]
        ks = sorted(set([round(k, 2) for k in ks + coarse]))
        return ks

    def _format_bbg_opt(base, exp_dt, cp, strike):
        mmddyy = pd.Timestamp(exp_dt).strftime("%m/%d/%y")
        if abs(strike - round(strike)) < 1e-6:
            strike_txt = str(int(round(strike)))
        else:
            strike_txt = f"{strike:.2f}".rstrip("0").rstrip(".")
        return f"{base} {mmddyy} {cp}{strike_txt} Equity"

    # ---- assemble candidates ----
    base = _bbg_base(underlying_bbg)
    trade_dt = pd.Timestamp(trade_date)

    expiry_fridays = _fridays_between(trade_dt, target_min_dte, target_max_dte)
    expiry_monthlies = _third_fridays(trade_dt, count=6)
    expiries = sorted(set(expiry_fridays + expiry_monthlies))[:24]  # cap to 24 expiries

    if verbose:
        print(f"[chain] expiries candidates: {len(expiries)} (min_dte={target_min_dte}, max_dte={target_max_dte})")

    if not expiries:
        return pd.DataFrame()

    strikes = _atm_ladder(spot_px, steps=5)  # 5 steps each side + coarse grid
    if verbose:
        print(f"[chain] strike candidates around spot {spot_px:.2f}: {len(strikes)}")

    # Build bulk list
    sec_list, meta = [], []
    for exp in expiries:
        for cp in ("C", "P"):
            for K in strikes:
                sec = _format_bbg_opt(base, exp, cp, K)
                sec_list.append(sec)
                meta.append({"sec": sec, "cp": cp, "K": float(K), "expiry": pd.Timestamp(exp)})

    if verbose:
        print(f"[chain] requesting {len(sec_list)} option securities via BDP")

    # ---- BDP request ----
    s, ref = _start_bbg_session()
    try:
        fields = [
            "PX_BID","PX_ASK",
            "OPT_STRIKE_PX","OPT_EXPIRE_DT","OPT_PUT_CALL",
            "IVOL_MID","IMPLIED_VOLATILITY"
        ]
        bdp = ref.createRequest("ReferenceDataRequest")
        se = bdp.getElement("securities"); [se.appendValue(sec) for sec in sec_list]
        fe = bdp.getElement("fields");     [fe.appendValue(f)   for f   in fields]
        msgs = _send_req(s, bdp)

        rows = []
        meta_map = {m["sec"]: m for m in meta}

        def _get(fd, name):
            if not fd.hasElement(name): return np.nan
            sub = fd.getElement(name)
            for meth in ("getValueAsFloat64","getValueAsInt64","getValueAsInteger"):
                try: return getattr(sub, meth)()
                except Exception: pass
            for meth in ("getValueAsDatetime","getValueAsString"):
                try:
                    val = getattr(sub, meth)()
                    if isinstance(val, blpapi.Datetime):
                        return pd.to_datetime(val.strftime("%Y-%m-%d"))
                    return val
                except Exception: pass
            try: return sub.getValue()
            except Exception: return np.nan

        returned = 0
        with_quotes = 0
        for msg in msgs:
            if not msg.hasElement("securityData"): continue
            sdata = msg.getElement("securityData")
            for i in range(sdata.numValues()):
                e = sdata.getValueAsElement(i)
                sec = e.getElementAsString("security")
                if not e.hasElement("fieldData"): continue
                fd = e.getElement("fieldData")
                returned += 1

                bid = pd.to_numeric(_get(fd, "PX_BID"), errors="coerce")
                ask = pd.to_numeric(_get(fd, "PX_ASK"), errors="coerce")
                if pd.isna(bid) and pd.isna(ask):
                    continue  # skip dead securities
                with_quotes += 1

                Kfd  = _get(fd, "OPT_STRIKE_PX")
                Efd  = _get(fd, "OPT_EXPIRE_DT")
                PCfd = _get(fd, "OPT_PUT_CALL")
                iv1  = _get(fd, "IVOL_MID")
                iv2  = _get(fd, "IMPLIED_VOLATILITY")

                meta_row = meta_map.get(sec, {})
                K = float(Kfd) if pd.notna(Kfd) else meta_row.get("K", np.nan)
                E = (Efd if isinstance(Efd, pd.Timestamp) else
                     (pd.to_datetime(Efd) if pd.notna(Efd) else meta_row.get("expiry", np.nan)))
                cp = (str(PCfd).upper() if isinstance(PCfd, str) else meta_row.get("cp", None))

                rows.append({
                    "security": sec,
                    "OPT_STRIKE_PX": K,
                    "OPT_EXPIRE_DT": E,
                    "OPT_PUT_CALL": cp,
                    "PX_BID": float(bid) if pd.notna(bid) else np.nan,
                    "PX_ASK": float(ask) if pd.notna(ask) else np.nan,
                    "MID": np.nan if (pd.isna(bid) or pd.isna(ask)) else (float(bid)+float(ask))/2.0,
                    "IMPLIED_VOLATILITY": (pd.to_numeric(iv1, errors="coerce")
                                           if pd.notna(iv1) else pd.to_numeric(iv2, errors="coerce"))
                })

        if verbose:
            print(f"[chain] returned: {returned}, with quotes: {with_quotes}")

        df = pd.DataFrame(rows)
        if df.empty:
            if verbose: print("[chain] No valid options returned via BDP.")
            return pd.DataFrame()

        # Clean & filter by DTE
        df = df.dropna(subset=["OPT_STRIKE_PX","OPT_EXPIRE_DT","OPT_PUT_CALL","MID"]).copy()
        df["OPT_EXPIRE_DT"] = pd.to_datetime(df["OPT_EXPIRE_DT"], errors="coerce")
        trade_dt = pd.Timestamp(trade_date)
        df["DTE"] = (df["OPT_EXPIRE_DT"] - trade_dt).dt.days
        df = df[(df["DTE"]>=target_min_dte) & (df["DTE"]<=target_max_dte)]
        if df.empty:
            if verbose: print("[chain] All candidates filtered out by DTE window.")
            return pd.DataFrame()

        # Pair ATM per expiry
        df["atm_dist"] = (df["OPT_STRIKE_PX"] - spot_px).abs()
        df["OPT_PUT_CALL"] = df["OPT_PUT_CALL"].astype(str).str.upper()
        atmC = df[df["OPT_PUT_CALL"]=="C"].sort_values(["atm_dist","DTE"]).groupby("OPT_EXPIRE_DT").head(1)
        atmP = df[df["OPT_PUT_CALL"]=="P"].sort_values(["atm_dist","DTE"]).groupby("OPT_EXPIRE_DT").head(1)
        pair = atmC.merge(atmP, on="OPT_EXPIRE_DT", suffixes=("_C","_P"))
        if pair.empty:
            if verbose: print("[chain] Could not pair ATM call/put on any expiry.")
            return pd.DataFrame()
        pair["combo_dist"] = pair["atm_dist_C"] + pair["atm_dist_P"]
        best = pair.sort_values(["combo_dist","DTE_C"]).head(1)

        if verbose:
            print(f"[chain] picked expiry {best['OPT_EXPIRE_DT'].iloc[0].date()}, K≈{best['OPT_STRIKE_PX_C'].iloc[0]}")

        return pd.DataFrame([{
            "expiry":   best["OPT_EXPIRE_DT"].iloc[0],
            "DTE":      int(best["DTE_C"].iloc[0]),
            "call_tkr": best["security_C"].iloc[0],
            "put_tkr":  best["security_P"].iloc[0],
            "K":        float(best["OPT_STRIKE_PX_C"].iloc[0]),
            "C_mid":    float(best["MID_C"].iloc[0]),
            "P_mid":    float(best["MID_P"].iloc[0]),
            "IV_call":  float(best["IMPLIED_VOLATILITY_C"].iloc[0]) if pd.notna(best["IMPLIED_VOLATILITY_C"].iloc[0]) else np.nan,
            "IV_put":   float(best["IMPLIED_VOLATILITY_P"].iloc[0]) if pd.notna(best["IMPLIED_VOLATILITY_P"].iloc[0]) else np.nan
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

# -------------------- Backtest --------------------
def backtest_parity(
    underlying="AAPL US Equity",
    start="20240101",
    end  ="20250918",
    min_dte=25, max_dte=45,
    gap_threshold=0.10,
    max_hold_days=60,
    riskfree_ticker="USGG3M Index",
    step_days=1,
    progress_every=10
):
    px = get_bdh_equity(underlying, start, end, fields=("PX_LAST",))
    rf = get_riskfree_series(start, end, ticker=riskfree_ticker)
    df = px.join(rf, how="left").ffill()
    df["rv_21"] = realized_vol_from_close(df["PX_LAST"])

    trades, equity_curve = [], []
    open_pos = None
    dates = df.index.to_list()

    for idx, dt in enumerate(dates):
        if step_days > 1 and idx % step_days != 0:
            equity_curve.append({"date": dt, "equity": 0.0 if not open_pos else open_pos["equity_mark"]})
            continue

        S = float(df.loc[dt, "PX_LAST"])
        rf_day = float(df.loc[dt, "rf_daily"]) if "rf_daily" in df.columns and pd.notna(df.loc[dt, "rf_daily"]) else 0.0

        if open_pos:
            open_pos["days_held"] += step_days
            open_pos["cash"] *= (1.0 + rf_day) ** step_days
            equity_curve.append({"date": dt, "equity": open_pos["equity_mark"]})
        else:
            equity_curve.append({"date": dt, "equity": 0.0})

        if open_pos:
            if dt >= open_pos["expiry"] or open_pos["days_held"] >= max_hold_days:
                pnl = open_pos["gap_signed"]
                trades.append({**open_pos, "exit_date": dt, "pnl": pnl})
                open_pos = None
            continue

        chain = get_chain_for_day(underlying, S, dt, min_dte, max_dte, max_points=600, verbose=True)
        if chain.empty:
            if progress_every and idx % progress_every == 0:
                print(f"[{dt.date()}] No options found in DTE window ({min_dte}-{max_dte}).")
            continue

        K    = chain["K"].iloc[0]
        Cmid = chain["C_mid"].iloc[0]
        Pmid = chain["P_mid"].iloc[0]
        expiry = chain["expiry"].iloc[0]
        dte   = int(chain["DTE"].iloc[0])
        T     = dte/365.0

        r_ann = (df.loc[dt, "rf_daily"]*365.0) if "rf_daily" in df.columns and pd.notna(df.loc[dt, "rf_daily"]) else 0.0
        dfac  = np.exp(-r_ann*T)
        gap   = (Cmid - Pmid) - (S - K*dfac)

        if abs(gap) < gap_threshold:
            if progress_every and idx % progress_every == 0:
                print(f"[{dt.date()}] Gap {gap:.3f} < thr {gap_threshold}")
            continue

        direction = "short_call_long_put_shortS_longB" if gap > 0 else "long_call_short_put_longS_shortB"
        open_pos = {
            "entry_date": dt, "expiry": expiry, "days_held": 0,
            "underlying": underlying,
            "S0": S, "K": K, "C_mid0": Cmid, "P_mid0": Pmid,
            "r_ann": r_ann, "T": T, "dte": dte,
            "gap": float(gap), "gap_signed": float(np.sign(gap)*abs(gap)),
            "direction": direction,
            "cash": K*dfac if gap>0 else -K*dfac,
            "equity_mark": float(np.sign(gap)*abs(gap))
        }

        if progress_every and idx % progress_every == 0:
            print(f"[{dt.date()}] ENTER {direction} | DTE={dte} K={K:.2f} S={S:.2f} gap={gap:.3f}")

    eq = pd.DataFrame(equity_curve).set_index("date")
    tr = pd.DataFrame(trades)

    if not tr.empty:
        total_pnl = tr["pnl"].sum()
        hitrate   = (tr["pnl"]>0).mean()
        avg_hold  = tr["days_held"].mean()
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

    plt.figure(figsize=(10,5))
    plt.plot(cum_pnl.index, cum_pnl.values, linewidth=2)
    plt.title(title); plt.xlabel("Date"); plt.ylabel("Cumulative Realized PnL (units)")
    plt.grid(True, linewidth=0.5, alpha=0.6); plt.tight_layout()
    plt.savefig(outfile, dpi=200); plt.close()
    print("Saved chart to:", outfile)

# -------------------- Example run --------------------
underlying = "AAPL US Equity"        # try other US names once AAPL works
start      = "20240701"
end        = "20240815"
min_dte, max_dte = 10, 60
gap_threshold    = 0.05

res = backtest_parity(
    underlying=underlying,
    start=start,
    end=end,
    min_dte=min_dte,
    max_dte=max_dte,
    gap_threshold=gap_threshold,
    riskfree_ticker="USGG3M Index",
    step_days=2,
    progress_every=5
)
print("Summary:", res["summary"])
outfile = f"{underlying.replace(' ','_')}_parity_backtest.png"
plot_backtest(res, title=f"{underlying} Parity Arbitrage Backtest", outfile=outfile)
print(f"Chart path: {outfile}")
# ===============================================================================================================
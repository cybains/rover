# services/ingest/connectors/features_materializer.py
import os, math, time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne, ASCENDING
from pymongo.errors import BulkWriteError

# --- Config ---
P_LOWER, P_UPPER = 5, 95   # winsorization per (indicator, year)
WINDOW_SLOPE = 5           # years for rolling OLS slope
CAGR_YEARS   = 5           # years for CAGR
BATCH_UPSERT = 5000

# Polarity map: "higher", "lower", or "neutral" (neutral = don't invert)
POLARITY: Dict[str, str] = {
    # Headline / Economy
    "NY.GDP.PCAP.KD":"higher", "FP.CPI.TOTL.ZG":"lower", "NE.EXP.GNFS.ZS":"higher",
    "SL.UEM.TOTL.ZS":"lower", "SL.TLF.CACT.ZS":"higher", "SP.POP.TOTL":"neutral",
    "SP.DYN.LE00.IN":"higher", "SE.TER.ENRR":"higher", "SE.ADT.LITR.ZS":"higher",
    "IT.NET.USER.ZS":"higher", "IT.CEL.SETS.P2":"higher", "EN.ATM.CO2E.PC":"lower",
    # Momentum / trade
    "NY.GDP.MKTP.KD.ZG":"higher", "BX.GSR.CCIS.CD":"higher",
    # Energy / environment
    "EG.USE.PCAP.KG.OE":"neutral", "EN.ATM.PM25.MC.M3":"lower",
    "EG.ELC.ACCS.ZS":"higher", "EG.ELC.RNEW.ZS":"higher",
    # Education extras
    "SE.SEC.ENRR":"higher",
    # Connectivity extras
    "IT.NET.BBND.P2":"higher",
    # Health extras
    "SH.IMM.MEAS":"higher", "SH.XPD.CHEX.PC.CD":"higher",
    # Business / finance / innovation / trade
    "FS.AST.PRVT.GD.ZS":"higher", "IC.BUS.NDNS.ZS":"higher",
    "TX.VAL.TECH.MF.ZS":"higher", "IP.JRN.ARTC.SC":"higher",
    # Affordability / safety / governance
    "PA.NUS.PPPC.RF":"lower", "VC.IHR.PSRC.P5":"lower", "RL.EST":"higher",
    # Urbanization
    "SP.URB.TOTL.IN.ZS":"higher",
}

# Which indicators are rates where "absolute" YoY should be reported as Δ percentage points
ABS_IS_PP = {
    "FP.CPI.TOTL.ZG", "SL.UEM.TOTL.ZS", "NE.EXP.GNFS.ZS", "SL.TLF.CACT.ZS",
    "SE.SEC.ENRR", "SE.TER.ENRR", "SE.ADT.LITR.ZS", "IT.NET.USER.ZS",
    "EG.ELC.ACCS.ZS", "EG.ELC.RNEW.ZS", "IT.NET.BBND.P2", "SP.URB.TOTL.IN.ZS"
}

# Bounded 0..100 (eligible for logit later if you want)
BOUNDED_0_100 = ABS_IS_PP | {"SH.IMM.MEAS"}

# --- Mongo ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise SystemExit("MONGO_URI missing in .env")
client = MongoClient(MONGO_URI)
db = client["worldbank_core"]
col_obs = db["observations"]
col_feat = db["features"]

def upsert_many(col, ops: List[UpdateOne]):
    if not ops: return 0
    try:
        res = col.bulk_write(ops, ordered=False, bypass_document_validation=True)
        return (res.upserted_count or 0) + (res.modified_count or 0)
    except BulkWriteError as e:
        # ignore dup key races; rethrow other issues
        non_dupes = [we for we in e.details.get("writeErrors", []) if we.get("code") not in (11000,11001)]
        if non_dupes: raise
        return 0

# --- Helpers ---
def percentile_rank(values_sorted: np.ndarray, x: float) -> float:
    # percent of values <= x (0..100)
    if values_sorted.size == 0: return float("nan")
    idx = np.searchsorted(values_sorted, x, side="right")
    return 100.0 * idx / values_sorted.size

def robust_z(values: np.ndarray, x: float) -> float:
    if values.size == 0: return float("nan")
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    scale = 1.4826 * mad
    if scale == 0: return 0.0
    return float((x - med) / scale)

def ols_slope(points: List[Tuple[int, float]]) -> Optional[float]:
    xs, ys = [], []
    for t, v in points:
        if v is None or math.isnan(v): continue
        xs.append(t); ys.append(float(v))
    if len(xs) < 2: return None
    x = np.array(xs, dtype=float); y = np.array(ys, dtype=float)
    xmean = x.mean(); ymean = y.mean()
    denom = np.sum((x - xmean)**2)
    if denom == 0: return None
    return float(np.sum((x - xmean)*(y - ymean)) / denom)

def cagr(v_t: float, v_0: float, years: int) -> Optional[float]:
    if v_t is None or v_0 is None: return None
    if v_t <= 0 or v_0 <= 0 or years <= 0: return None
    return float((v_t / v_0) ** (1/years) - 1)

# --- Load all observations for core indicators ---
def load_all() -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    Returns: series[indicator][country_iso3][year] = value (float or np.nan)
    """
    series: Dict[str, Dict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    cur = col_obs.find(
        {"indicator.id": {"$in": list(POLARITY.keys())}, "value": {"$ne": None}},
        {"indicator.id": 1, "country.id": 1, "date": 1, "value": 1}
    )
    n = 0
    for d in cur:
        ind = d["indicator"]["id"]
        c = d["country"]["id"]
        try:
            y = int(d["date"])
        except:
            continue
        v = d.get("value")
        if v is None: continue
        series[ind][c][y] = float(v)
        n += 1
    print(f"Loaded {n:,} observations across {len(series)} indicators.")
    return series

def materialize():
    series = load_all()

    # Precompute winsor thresholds per (indicator, year)
    thresholds: Dict[Tuple[str,int], Tuple[float,float]] = {}
    for ind, by_country in series.items():
        # collect values by year
        vals_by_year: Dict[int, List[float]] = defaultdict(list)
        for c, ys in by_country.items():
            for y, v in ys.items():
                if v is None or math.isnan(v): continue
                vals_by_year[y].append(float(v))
        for y, vals in vals_by_year.items():
            arr = np.array(vals, dtype=float)
            lo = float(np.nanpercentile(arr, P_LOWER))
            hi = float(np.nanpercentile(arr, P_UPPER))
            if lo > hi: lo, hi = hi, lo
            thresholds[(ind, y)] = (lo, hi)

    ops: List[UpdateOne] = []
    total = 0
    t0 = time.time()

    for ind, by_country in series.items():
        pol = POLARITY.get(ind, "neutral")

        # Build per-year sorted arrays for percentiles (winsorized)
        year_values_sorted: Dict[int, np.ndarray] = {}
        for y in {yy for ys in by_country.values() for yy in ys.keys()}:
            lo, hi = thresholds.get((ind, y), (float("nan"), float("nan")))
            vals = []
            for c, ys in by_country.items():
                v = ys.get(y)
                if v is None or math.isnan(v): continue
                vv = min(max(v, lo), hi) if not math.isnan(lo) and not math.isnan(hi) else v
                vals.append(vv)
            year_values_sorted[y] = np.array(sorted(vals), dtype=float)

        # For each country, compute time-series features
        for c, ys in by_country.items():
            years_sorted = sorted(ys.keys())
            # rolling window for slope
            win = deque(maxlen=WINDOW_SLOPE)

            for i, y in enumerate(years_sorted):
                v = ys[y]
                lo, hi = thresholds.get((ind, y), (float("nan"), float("nan")))
                v_clip = min(max(v, lo), hi) if not math.isnan(lo) and not math.isnan(hi) else v
                clipped = int(v_clip != v)

                # YoY % and absolute (use clipped prev)
                yoy_pct = None
yoy_abs = None
if i > 0:
    yprev = years_sorted[i-1]
    vprev = ys.get(yprev)
    if vprev is not None:
        lo_prev, hi_prev = thresholds.get((ind, yprev), (float("nan"), float("nan")))
        # clip previous and current to their respective year thresholds
        prev_clip = min(max(vprev, lo_prev), hi_prev) if not (math.isnan(lo_prev) or math.isnan(hi_prev)) else vprev
        curr_clip = v_clip
        yoy_abs = float(curr_clip - prev_clip)
        if prev_clip > 0 and curr_clip > 0:
            yoy_pct = float((curr_clip / prev_clip - 1.0) * 100.0)


                # 5y slope (OLS) over clipped values
                win.append((y, v_clip))
                slope_5y = None
                if len(win) >= 2:
                    slope_5y = ols_slope(list(win))

                # 5y CAGR
                cagr_5y = None
                y0 = y - CAGR_YEARS
                if y0 in ys:
                    v0 = ys[y0]
                    # use raw positive values for CAGR
                    cagr_5y = cagr(v_clip, v0, CAGR_YEARS)

                # Percentiles & robust z (per year, winsorized pool)
                arr_sorted = year_values_sorted.get(y, np.array([], dtype=float))
                pctl_raw = percentile_rank(arr_sorted, v_clip)
                z_rob = robust_z(arr_sorted, v_clip)

                # Polarity-adjusted percentile (higher is better)
                pctl = pctl_raw
                if pol == "lower" and not math.isnan(pctl_raw):
                    pctl = 100.0 - pctl_raw

                doc = {
                    "_id": f"{ind}|{c}|{y}",
                    "indicator": ind,
                    "country": c,
                    "year": y,
                    "value": v,                 # raw
                    "value_clip": v_clip,       # winsorized per (ind,year)
                    "clipped": clipped,         # 0/1
                    "yoy_pct": yoy_pct,         # %
                    "yoy_abs": yoy_abs,         # absolute (pp for rate-like)
                    "cagr_5y": cagr_5y,         # fraction (e.g., 0.03 = 3%)
                    "slope_5y": slope_5y,       # units/year
                    "pctl_world_raw": pctl_raw, # 0..100
                    "pctl_world": pctl,         # polarity-adjusted
                    "z_robust": z_rob,
                    "polarity": pol,
                    "bounded_0_100": ind in BOUNDED_0_100,
                    "meta": {
                        "winsor": [P_LOWER, P_UPPER],
                        "window_slope": WINDOW_SLOPE,
                        "cagr_years": CAGR_YEARS,
                    }
                }
                ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True))
                if len(ops) >= BATCH_UPSERT:
                    total += upsert_many(col_feat, ops); ops = []

        # progress per indicator
        print(f"[features] done {ind}")

    if ops:
        total += upsert_many(col_feat, ops)

    # indexes (idempotent)
    col_feat.create_index([("indicator", ASCENDING), ("country", ASCENDING), ("year", ASCENDING)], name="ind_country_year")
    col_feat.create_index([("country", ASCENDING), ("year", ASCENDING)], name="country_year")

    elapsed = time.time() - t0
    print(f"Upserted/modified ~{total:,} feature docs in {elapsed:,.1f}s")

if __name__ == "__main__":
    materialize()

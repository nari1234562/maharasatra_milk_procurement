"""
COW v2 (NEW): COW is forecast by its own method, "Cow Ensemble v2" (cow_model.py, must sit in the same folder).  BUFFALO and every shared
function are unchanged (cow_extra=False is the original behaviour); the replaced cow lines are kept as comments marked "(original)".
step4_horizon_models.py - FILE 4 of 4: the 8 HORIZON MODELS  (TASK 3, built on TASK 1 weather band + TASK 2 intervention flags)
=================================================================================================================================
Project  : CDPL Maharashtra milk procurement rate, Rs per kg of Total Solids, DISTRICT level, COW and BUFFALO separately.

THE 8 MODELS (standing at the end of month t = the "origin", N = t)
-------------------------------------------------------------------
    N+1  N+2  N+3  N+4  N+5  N+6   six MONTHLY models.  Model N+h predicts ONE month only: the rate of month t+h.
                                   Each is trained on its own rows (district x origin) with target = Rate(t+h) - Rate(t).
    Q1                             QUARTERLY model 1 = the AVERAGE rate of months t+1, t+2, t+3
    Q2                             QUARTERLY model 2 = the AVERAGE rate of months t+4, t+5, t+6
    (origin Dec-25  ->  N+1 = Jan-26 ... N+6 = Jun-26,  Q1 = Jan-Mar 2026,  Q2 = Apr-Jun 2026)

HOW EACH MODEL IS BUILT (same recipe at every origin; nothing after the origin is ever used)
--------------------------------------------------------------------------------------------
 1. Training rows come from step 1 (build_training_table).  One row = (district, origin t, horizon h).
    Inputs = price history up to t, weather / fat / SNF FOR THE TARGET MONTH (actual up to t, same-month earlier-year average after t),
    market inputs projected to the target month, calendar of the target month, and the INTERVENTION FLAGS run_up_share / crash_share.
 2. MONTHLY model N+h : one Ridge + one XGBoost fitted on the rows with horizon h only (the "Direct" method)  -> Direct Ridge / XGB / Blend.
    The same horizon is also produced by the RecursiveX method (one-step model fed back its own forecasts, step 2)  -> RecursiveX Ridge / XGB / Blend.
 3. QUARTERLY model Qk : its own training table.  For every (district, origin) the 3 rows of the quarter's horizons (1-3 or 4-6) are
    collapsed into ONE row: inputs = average of the 3 rows' inputs, target = mean(Rate over the 3 months) - Rate(t).
    A row is used for training only when all 3 months are known at the origin (assert_no_leakage on the quarter's LAST month).
    Ridge + XGBoost + Blend are fitted on these rows  -> "Q-model Ridge / XGB / Blend".
    A quarter can also be forecast as the plain average of its 3 monthly forecasts -> "Avg of monthly <method>".
 4. METHOD CHOICE PER MODEL (not one method for everything).  Candidates: 3 naive baselines, 6 learned methods, 3 ensembles (mean / median of the
    6 learned, and a 50%-shrunk-towards-last-rate version); quarters also: Q-model x3, Avg of monthly x5, Avg of the SELECTED monthly forecasts.
    At every origin the method with the lowest MAE over the errors ALREADY KNOWN at that origin (target month <= origin) wins; it replaces the
    default only if it is at least SWITCH_MARGIN (5%) better (stops chasing noise) and has >= MIN_SELECT_N scored rows.
    Because the pick uses only errors known at the origin, the walk-forward score of the picks (NESTED / adaptive) is an honest out-of-sample number.
 5. BANDS for every one of the 8 models (TASK 1): the fitted models are run with LOW / EXPECTED / HIGH weather (weather band) and the 80% model-error
    band (10th-90th percentile of the selected method's past errors known at the origin) is added -> TOTAL band.
 6. INTERVENTION (TASK 2): flags are inputs of every model.  They are 0 for every forecast made after the swing (forecast = "no intervention").
    Override the dates with --run-up / --crash once the business confirms them.
 7. VALIDATION: MAE per model against the targets (N+1 <= 2.0 Rs/kg TS, Q1 <= 5.0), at DISTRICT level and at the volume-weighted ALL-DISTRICT
    (aggregate) level, with bias, direction accuracy and skill vs "rate stays flat".  Scopes: A = origins Dec-24..Nov-25, B = held-out origin
    Dec-25 (Jan-Jun 2026), C = all walk-forward origins.

    python step4_horizon_models.py                       (origin = last actual month, e.g. Jun-26 -> Jul-Dec 2026)
    python step4_horizon_models.py --origin 2025-12-01   (held-out check: forecast Jan-Jun 2026 and score it against the actuals)

MULTI-HOLDOUT (new): scope D of the validation = HOLDOUT_N (6) forecast origins, 2 months apart, over one year (step 2, PART D, same origins).  One origin gives only
ONE test month per horizon; with 6 batches every N+1 .. N+6 and Q1 / Q2 model has 6 test months per district.  Sheet Holdout_6_Batches shows MAE, weighted MAE,
aggregate MAE, the MAE of each batch and the number of batches inside the target, next to "last month repeated", rolling 3-month average and ARIMA.
Run steps 1 and 2 first only if you want their reports; step 4 builds everything it needs from the data workbook.
"""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import step1_data_features as s1
import step2_backtest as s2
import cow_model                                   # NEW: separate COW method (cow_model.py); buffalo never uses it

warnings.filterwarnings("ignore")
log = logging.getLogger("milk")

# ------------------------------------------------------------------ settings
FIRST_ORIGIN = "2024-12-01"          # first backtest origin (same as step 2)
HORIZONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
QUARTERS: Dict[str, Tuple[int, ...]] = {"Q1": (1, 2, 3), "Q2": (4, 5, 6), "Q3": (7, 8, 9), "Q4": (10, 11, 12)}
MODELS: List[str] = [f"N+{h}" for h in HORIZONS] + list(QUARTERS)
SCN_LOW, SCN_HIGH = "WEATHER LOW", "WEATHER HIGH"
SCNS = (SCN_LOW, SCN_HIGH)

DEFAULT_MONTHLY = "RecursiveX Blend"         # used until enough errors are known (best h1-6 method of your selection set)
DEFAULT_QUARTER = "Avg of selected monthly"   # the quarter follows the monthly picks until it has its own track record
MIN_SELECT_N = 12                    # scored rows needed before a method can replace the default
SWITCH_MARGIN = 0.05                 # a challenger must be >= 5% better than the default to replace it
SHRINK = 0.5                         # "Shrunk ens" = last rate + SHRINK x (ensemble mean change)
BAND_MIN_N = 5                       # errors needed for a model-error band
CBE_TARGET, Q1_TARGET = 2.0, 5.0     # Rs/kg TS: N+1 and Q1 targets (Q1 as agreed in step 2)

HOLDOUT_SET: List[pd.Timestamp] = []  # origins of the multi-holdout batches (set in main from step 2: s2.holdout_origins)

BASELINES = list(s2.BASELINES)       # Naive, Seasonal-trend naive, Drift (6m avg)
LEARNED = list(s2.LEARNED)           # Direct/RecursiveX x Ridge/XGB/Blend
ENSEMBLES = ["Ens mean", "Ens median", "Shrunk ens"]
MONTHLY_CANDS = BASELINES + LEARNED + ENSEMBLES
Q_MODEL = [f"Q-model {k}" for k in s2.LEARNERS]
AVG_OF = ["Direct Blend", "RecursiveX Blend", "Ens mean", "Ens median", "Shrunk ens"]
QUARTER_CANDS = BASELINES + Q_MODEL + [f"Avg of monthly {m}" for m in AVG_OF] + ["Avg of selected monthly"]
# ---- NEW (COW only).  BUFFALO keeps the lists above, unchanged.  Cow uses its own method (cow_model.py) and chooses only among the 3 naive baselines
# ---- and the new cow members, so the old Ridge / XGBoost / RecursiveX methods can no longer be picked for cow.  (Old cow behaviour: use MONTHLY_CANDS / QUARTER_CANDS.)
AVG_OF_COW = [cow_model.METHOD]                                        # quarters of cow can average the monthly cow-model forecasts
COW_DEFAULT_MONTHLY = cow_model.METHOD                                  # cow default monthly method (buffalo: DEFAULT_MONTHLY)
COW_MONTHLY_CANDS = BASELINES + cow_model.COW_COLUMNS
COW_QUARTER_CANDS = BASELINES + [f"Avg of monthly {m}" for m in AVG_OF_COW] + ["Avg of selected monthly"]


# ===================================================================== helpers: columns and ensembles
def sfx(scn: str | None) -> str:
    return f" | {scn}" if scn else ""


def get_col(df: pd.DataFrame, method: str, scn: str | None = None) -> pd.Series:
    """Forecast column of ``method`` under weather scenario ``scn`` (None = expected).  Methods that do not use weather
    (the naive baselines) have no scenario column: their value is the same in all three scenarios."""
    if scn and f"{method}{sfx(scn)}" in df.columns:
        return df[f"{method}{sfx(scn)}"]
    return df[method]


def add_ensembles(d: pd.DataFrame) -> pd.DataFrame:
    """Ens mean / Ens median / Shrunk ens of the 6 learned methods, for the expected weather and for each weather scenario present."""
    for s in ("",) + tuple(sfx(x) for x in SCNS):
        comp = [f"{m}{s}" for m in LEARNED]
        if all(c in d.columns for c in comp):
            arr = d[comp].to_numpy(dtype=float)
            d[f"Ens mean{s}"] = arr.mean(axis=1)
            d[f"Ens median{s}"] = np.median(arr, axis=1)
            d[f"Shrunk ens{s}"] = d["rate_t"].to_numpy() + SHRINK * (d[f"Ens mean{s}"].to_numpy() - d["rate_t"].to_numpy())
    return d


# ===================================================================== quarterly training table
def quarter_table(table: pd.DataFrame, feats: List[str], hs: Tuple[int, ...]) -> pd.DataFrame:
    """Collapse the rows of horizons ``hs`` into ONE row per (district, origin).

    inputs  = average over the quarter's months of the inputs of that month's row (so weather / market / flags / calendar are quarter averages)
    target  = mean(Rate(t+h) for h in hs) - Rate(t)   (NaN unless ALL months are known)
    target_month = LAST month of the quarter (used by assert_no_leakage: the whole quarter must be known at the origin)."""
    keep = list(dict.fromkeys(list(feats) + ["ly_change", "diff_mean6"]))
    key = [s1.DISTRICT, "origin"]
    t = table[table["horizon"].isin(hs)]
    t = t[t.groupby(key)["horizon"].transform("size") == len(hs)]
    q = t.groupby(key)[keep].mean()
    q["rate_t"] = t.groupby(key)["rate_t"].first()
    ra = t.set_index(key + ["horizon"])["rate_actual"].unstack("horizon")
    q["rate_actual"] = ra.mean(axis=1, skipna=False).reindex(q.index)
    q["target"] = q["rate_actual"] - q["rate_t"]
    q = q.reset_index()
    q["target_month"] = q["origin"] + pd.DateOffset(months=max(hs))
    q["first_month"] = q["origin"] + pd.DateOffset(months=min(hs))
    q["horizon"] = max(hs)
    return q


def quarter_baselines(rows: pd.DataFrame, hs: Tuple[int, ...]) -> Dict[str, np.ndarray]:
    return {"Naive": rows["rate_t"].to_numpy(),
            "Seasonal-trend naive": rows["rate_t"].to_numpy() + rows["ly_change"].fillna(0.0).to_numpy(),
            "Drift (6m avg)": rows["rate_t"].to_numpy() + rows["diff_mean6"].fillna(0.0).to_numpy() * float(np.mean(hs))}


def avg_of_months(d: pd.DataFrame, hs: Tuple[int, ...], col: str) -> pd.Series:
    """Average of a monthly forecast column over the quarter's months, per district (NaN if a month is missing)."""
    p = d[d["horizon"].isin(hs)].pivot(index=s1.DISTRICT, columns="horizon", values=col)
    if set(hs) - set(p.columns):
        return pd.Series(np.nan, index=d[s1.DISTRICT].unique())
    return p[list(hs)].mean(axis=1, skipna=False)


def quarter_forecast(panel: pd.DataFrame, qtabs: Dict[str, pd.DataFrame], t: pd.Timestamp, feats: List[str], d: pd.DataFrame,
                     qscn: Dict[str, Dict[str, pd.DataFrame]] | None = None, keep_fits: bool = False):
    """Both quarterly models at origin t.  ``d`` = the monthly forecasts of the SAME origin (from step 2's forecast_origin, with the ensembles).

    qscn = {scenario name: {quarter id: quarter table of the scenario rows}} -> same fitted models, other forward-weather inputs."""
    parts, fits = [], {}
    districts = list(d[s1.DISTRICT].unique())
    for qid, hs in QUARTERS.items():
        qt = qtabs[qid]
        train = qt[(qt["target_month"] <= t) & qt["target"].notna()]
        s1.assert_no_leakage(train, t)
        test = qt[(qt["origin"] == t) & qt[s1.DISTRICT].isin(districts)]
        if test.empty or len(train) < s2.MIN_TRAIN_ROWS:
            continue
        res = test[[s1.DISTRICT, "origin", "first_month", "target_month", "rate_t", "rate_actual"]].copy()
        res["model"] = qid
        res["n_train"] = len(train)
        for m, v in quarter_baselines(test, hs).items():
            res[m] = v
        r, x = s2.fit_pair(train, feats)
        pr, px = r.predict(test[feats]), x.predict(test[feats])
        for k, c in (("Ridge", pr), ("XGB", px), ("Blend", (pr + px) / 2)):
            res[f"Q-model {k}"] = test["rate_t"].to_numpy() + c
        # for k in AVG_OF:                                                                  # (original)
        for k in AVG_OF + [c for c in AVG_OF_COW if c in d.columns]:                         # NEW: + the cow method (cow rows only)
            res[f"Avg of monthly {k}"] = avg_of_months(d, hs, k).reindex(res[s1.DISTRICT]).to_numpy()
        for nm, tabs in (qscn or {}).items():
            ts = tabs[qid]
            ts = ts[ts["origin"] == t].set_index(s1.DISTRICT).reindex(res[s1.DISTRICT])
            ps_r, ps_x = r.predict(ts[feats]), x.predict(ts[feats])
            for k, c in (("Ridge", ps_r), ("XGB", ps_x), ("Blend", (ps_r + ps_x) / 2)):
                res[f"Q-model {k}{sfx(nm)}"] = res["rate_t"].to_numpy() + c
            # for k in AVG_OF:                                                              # (original)
            for k in AVG_OF + [c for c in AVG_OF_COW if c in d.columns]:                     # NEW
                col = f"{k}{sfx(nm)}"
                res[f"Avg of monthly {k}{sfx(nm)}"] = (avg_of_months(d, hs, col) if col in d.columns else avg_of_months(d, hs, k)).reindex(res[s1.DISTRICT]).to_numpy()
        months = [res["origin"] + pd.DateOffset(months=h) for h in hs]
        wv = [panel["ts_kg"].reindex(pd.MultiIndex.from_arrays([res[s1.DISTRICT], mth])).to_numpy() for mth in months]
        res["weight_ts_kg"] = np.mean(wv, axis=0)                       # NaN if a month is missing -> weight 0 in the aggregate metric
        parts.append(res)
        if keep_fits:
            fits[qid] = (r, x, train)
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), fits


# ===================================================================== backtest of all 8 models
def monthly_frame(d: pd.DataFrame, mt: str) -> pd.DataFrame:
    d = add_ensembles(d.copy())
    d["milk_type"] = mt
    d["model"] = "N+" + d["horizon"].astype(int).astype(str)
    return d


def run_backtest(mt: str, panel: pd.DataFrame, market: pd.DataFrame, table: pd.DataFrame, recx: Dict, cfg: s1.Config,
                 origins: pd.DatetimeIndex) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward: at every origin retrain all 8 models on rows known at that origin and forecast the next 6 months / 2 quarters."""
    feats = recx["spec"]["all"]
    qtabs = {qid: quarter_table(table, feats, hs) for qid, hs in QUARTERS.items()}
    mrows, qrows = [], []
    for t in origins:
        try:
            # d, _ = s2.forecast_origin(panel, table, t, cfg.max_horizon, cfg, recx)                       # (original)
            d, _ = s2.forecast_origin(panel, table, t, cfg.max_horizon, cfg, recx, cow_extra=(mt == "COW"))  # NEW: cow also gets cow_model columns
        except ValueError:                                   # too little history at this origin
            continue
        d = monthly_frame(d, mt)
        q, _ = quarter_forecast(panel, qtabs, t, feats, d)
        if len(q):
            q["milk_type"] = mt
            qrows.append(q)
        mrows.append(d)
        log.info("  %s origin %s: 6 monthly + 2 quarterly models done", mt, f"{t:%b %Y}")
    return pd.concat(mrows, ignore_index=True), (pd.concat(qrows, ignore_index=True) if qrows else pd.DataFrame())


# ===================================================================== method choice (known-at-origin only)
def pick_methods(det: pd.DataFrame, t: pd.Timestamp, cands: List[str], default: str, min_n: int, margin: float) -> Dict[Tuple[str, str], Dict]:
    """For every (milk type, model): the method with the lowest MAE over rows whose target month is <= t.
    The default is kept unless a challenger is >= ``margin`` better.  Returns {(milk, model): {method, n, mae, default_mae}}."""
    d = det[det["target_month"] <= t]
    out: Dict[Tuple[str, str], Dict] = {}
    for key, g in d.groupby(["milk_type", "model"]):
        # cc = [c for c in cands if c in g.columns]                                  # (original)
        cc = [c for c in cands if c in g.columns and g[c].notna().any()]             # NEW: a method that exists for the other milk type only is skipped
        gg = g.dropna(subset=cc + ["rate_actual"])
        pick = {"method": default, "n": len(gg), "mae": np.nan, "default_mae": np.nan}
        if len(gg) >= min_n and default in cc:
            maes = {c: float((gg[c] - gg["rate_actual"]).abs().mean()) for c in cc}
            best = min(maes, key=maes.get)
            pick.update(mae=maes[default], default_mae=maes[default])
            if best != default and maes[best] <= (1 - margin) * maes[default]:
                pick.update(method=best, mae=maes[best])
        out[key] = pick
    return out


def apply_nested(det: pd.DataFrame, cands: List[str], default: str, min_n: int, margin: float) -> pd.DataFrame:
    """Add ``sel_method`` / ``sel_fc`` to every row: the method that was picked using ONLY the errors known at that row's origin."""
    det = det.copy()
    det["sel_method"], det["sel_fc"] = default, np.nan
    for t in sorted(det["origin"].unique()):
        picks = pick_methods(det, pd.Timestamp(t), cands, default, min_n, margin)
        at = det["origin"] == t
        for (mt, model), p in picks.items():
            m = at & (det["milk_type"] == mt) & (det["model"] == model)
            det.loc[m, "sel_method"] = p["method"]
            det.loc[m, "sel_fc"] = det.loc[m, p["method"]].to_numpy()
        m = at & det["sel_fc"].isna()                        # groups with no track record yet: default method
        det.loc[m, "sel_fc"] = det.loc[m, default].to_numpy()
    return det


def avg_selected_monthly(mad: pd.DataFrame, qdet: pd.DataFrame) -> pd.Series:
    """'Avg of selected monthly': for each quarter row, the mean of the (nested-)selected monthly forecasts of its 3 months."""
    key = ["milk_type", s1.DISTRICT, "origin"]
    w = mad.set_index(key + ["horizon"])["sel_fc"].unstack("horizon")
    out = pd.Series(np.nan, index=qdet.index)
    for qid, hs in QUARTERS.items():
        cols = [h for h in hs if h in w.columns]
        if len(cols) < len(hs):
            continue
        v = w[list(hs)].mean(axis=1, skipna=False)
        m = qdet["model"] == qid
        idx = pd.MultiIndex.from_frame(qdet.loc[m, key])
        out[m] = v.reindex(idx).to_numpy()
    return out


# ===================================================================== error band + final forecast
def error_band(det: pd.DataFrame, mt: str, model: str, method: str, t: pd.Timestamp, fallback: str) -> Tuple[float, float, int]:
    """80% band (p10, p90) of the errors (actual - forecast) of ``method`` known at t.  Too few errors -> the fallback method's errors."""
    g = det[(det["milk_type"] == mt) & (det["model"] == model) & (det["target_month"] <= t)]
    for m in (method, fallback):
        if m in g.columns:
            e = (g["rate_actual"] - g[m]).dropna().to_numpy()
            if len(e) >= BAND_MIN_N:
                return float(np.percentile(e, 10)), float(np.percentile(e, 90)), len(e)
    return np.nan, np.nan, 0


def forecast_rows(mt: str, dl: pd.DataFrame, ql: pd.DataFrame, picks_m: Dict, picks_q: Dict, mdet: pd.DataFrame, qdet: pd.DataFrame,
                  t: pd.Timestamp, scored: bool) -> pd.DataFrame:
    """One table with a row per (district, model): forecast under expected / LOW / HIGH weather, weather band, model-error band, total band."""
    rows = []
    # for kind, df, picks, det, fb in (("monthly", dl, picks_m, mdet, DEFAULT_MONTHLY), ("quarter", ql, picks_q, qdet, DEFAULT_QUARTER)):     # (original)
    for kind, df, picks, det, fb in (("monthly", dl, picks_m, mdet, COW_DEFAULT_MONTHLY if mt == "COW" else DEFAULT_MONTHLY),               # NEW: cow default = cow_model.METHOD
                                     ("quarter", ql, picks_q, qdet, DEFAULT_QUARTER)):
        if df is None or df.empty:
            continue
        for model, g in df.groupby("model"):
            method = picks.get((mt, model), {}).get("method", fb)
            f0 = get_col(g, method).to_numpy()
            f_lo, f_hi = get_col(g, method, SCN_LOW).to_numpy(), get_col(g, method, SCN_HIGH).to_numpy()
            st = np.vstack([f0, f_lo, f_hi])
            lo_e, hi_e, n_e = error_band(det, mt, model, method, t, fb)
            lo, hi = min(lo_e, 0.0) if lo_e == lo_e else 0.0, max(hi_e, 0.0) if hi_e == hi_e else 0.0
            wlo, whi = st.min(axis=0), st.max(axis=0)
            if kind == "monthly":
                label = g["target_month"].dt.strftime("%b %Y")
            else:
                label = g["first_month"].dt.strftime("%b") + "-" + g["target_month"].dt.strftime("%b %Y")
            out = pd.DataFrame({
                "Milk Type": mt.title(), "Model": model, "Type": "Monthly (1 month)" if kind == "monthly" else "Quarterly (average of 3 months)",
                "District": g[s1.DISTRICT].to_numpy(), "Forecast period": label.to_numpy(), "Last Actual (Rs/kg TS)": g["rate_t"].to_numpy(),
                "Forecast (Rs/kg TS)": f0, "Method used": method,
                "Forecast, LOW weather": f_lo, "Forecast, HIGH weather": f_hi,
                "Weather band low": wlo, "Weather band high": whi, "Weather band width": whi - wlo,
                "Lower 80% (model error)": f0 + lo, "Upper 80% (model error)": f0 + hi, "Model-error band n": n_e,
                "TOTAL band low (weather + model error)": wlo + lo, "TOTAL band high (weather + model error)": whi + hi,
                "Status": "Short-term (CBE)" if model in ("N+1", "N+2", "N+3", "N+4", "Q1") else "INDICATIVE - beyond 4 months"})
            if scored:
                out["Actual (Rs/kg TS)"] = g["rate_actual"].to_numpy()
                out["Abs error (Rs/kg TS)"] = (out["Forecast (Rs/kg TS)"] - out["Actual (Rs/kg TS)"]).abs()
            # for m in (BASELINES + ([c for c in (LEARNED + ENSEMBLES) if c in g.columns] if kind == "monthly" else Q_MODEL + [f"Avg of monthly {k}" for k in AVG_OF])):   # (original)
            for m in (BASELINES + ([c for c in (LEARNED + ENSEMBLES + cow_model.COW_COLUMNS) if c in g.columns and g[c].notna().any()] if kind == "monthly"
                                   else Q_MODEL + [f"Avg of monthly {k}" for k in AVG_OF + [c for c in AVG_OF_COW if f"Avg of monthly {c}" in g.columns and g[f"Avg of monthly {c}"].notna().any()]])):   # NEW: + cow-only columns
                out[m] = g[m].to_numpy()
            rows.append(out)
    r = pd.concat(rows, ignore_index=True)
    r["_o"] = r["Model"].map({m: i for i, m in enumerate(MODELS)})
    return r.sort_values(["Milk Type", "_o", "District"]).drop(columns="_o").reset_index(drop=True)


# ===================================================================== validation
def _agg_mae(g: pd.DataFrame, col: str) -> float:
    w = g["weight_ts_kg"].fillna(0)
    s = g.assign(_w=w, _a=g["rate_actual"] * w, _f=g[col] * w).groupby("origin")[["_w", "_a", "_f"]].sum()
    s = s[s["_w"] > 0]
    return float(((s["_f"] / s["_w"] - s["_a"] / s["_w"]).abs().mean())) if len(s) else np.nan


def score(g: pd.DataFrame, col: str) -> Dict[str, float]:
    e = g[col] - g["rate_actual"]
    nv = (g["Naive"] - g["rate_actual"]).abs().mean()
    mae = e.abs().mean()
    dacc = float((np.sign(g[col] - g["rate_t"]) == np.sign(g["rate_actual"] - g["rate_t"])).mean() * 100)
    w = g["weight_ts_kg"].fillna(0.0)
    mask = g["rate_actual"].notna() & g[col].notna() & (w > 0)
    vw = np.nan if not mask.any() else float(((g.loc[mask, col] - g.loc[mask, "rate_actual"]).abs() * w.loc[mask]).sum() / w.loc[mask].sum())
    return {"n": len(g), "districts": g[s1.DISTRICT].nunique(), "origins": g["origin"].nunique(), "MAE": mae, "VW-MAE": vw,
            "Aggregate MAE": _agg_mae(g, col), "Bias (fcst-actual)": e.mean(), "Direction acc %": dacc,
            "MAE if rate stays flat": nv, "Skill vs flat": 1 - mae / nv if nv > 0 else np.nan}


def scopes(origin_held: pd.Timestamp) -> Dict[str, callable]:
    return {"A: origins Dec-24..Nov-25": lambda o: (o >= pd.Timestamp(FIRST_ORIGIN)) & (o <= pd.Timestamp("2025-11-01")),
            f"B: held-out origin {origin_held:%b-%y}": lambda o: o == origin_held,
            "C: all walk-forward origins": lambda o: o >= pd.Timestamp(FIRST_ORIGIN),
            f"D: {len(HOLDOUT_SET)} holdout batches": lambda o: o.isin(HOLDOUT_SET)}


def quarter_target(model: str) -> float:
    targets = {"Q1": 5.0, "Q2": 8.0, "Q3": 13.0, "Q4": 17.0}
    return targets.get(model, np.nan)


def validation_table(mad: pd.DataFrame, qad: pd.DataFrame, origin_held: pd.Timestamp) -> pd.DataFrame:
    """MAE of the NESTED (adaptive) forecast of each of the 12 monthly and 4 quarterly models against the targets, per scope."""
    rec = []
    for scope, fn in scopes(origin_held).items():
        for df in (mad, qad):
            d = df[fn(df["origin"]) & df["rate_actual"].notna() & df["sel_fc"].notna()].assign(fc=lambda x: x["sel_fc"])
            for (mt, model), g in d.groupby(["milk_type", "model"]):
                s = score(g, "fc")
                tgt = CBE_TARGET if model == "N+1" else quarter_target(model) if model.startswith("Q") else np.nan
                rec.append({"Scope": scope, "Milk Type": mt.title(), "Model": model, **s, "Target": tgt,
                            "Check (district level)": "n/a" if tgt != tgt else "MEETS" if s["MAE"] <= tgt else "OVER",
                            "Check (aggregate level)": "n/a" if tgt != tgt else "MEETS" if s["Aggregate MAE"] <= tgt else "OVER"})
        # Q1 / Q2 / Q3 / Q4 as the AVERAGE MONTHLY error inside each quarter (the definition used in step 2)
        for model, hs in QUARTERS.items():
            for mt, g in mad[fn(mad["origin"]) & mad["rate_actual"].notna() & mad["horizon"].isin(hs)].groupby("milk_type"):
                s = score(g, "sel_fc")
                tgt = quarter_target(model)
                rec.append({"Scope": scope, "Milk Type": mt.title(), "Model": f"{model} as avg monthly error (h{hs[0]}-{hs[-1]})", **s, "Target": tgt,
                            "Check (district level)": "n/a" if tgt != tgt else "MEETS" if s["MAE"] <= tgt else "OVER",
                            "Check (aggregate level)": "n/a" if tgt != tgt else "MEETS" if s["Aggregate MAE"] <= tgt else "OVER"})
    v = pd.DataFrame(rec)
    order = {m: i for i, m in enumerate(MODELS)}
    v["_o"] = v["Model"].map(lambda m: order.get(m, 10 + (m.startswith("Q2"))))
    return v.sort_values(["Scope", "Milk Type", "_o"]).drop(columns="_o").reset_index(drop=True)


def holdout_table(mad: pd.DataFrame, qad: pd.DataFrame, origins: List[pd.Timestamp]) -> pd.DataFrame:
    """Multi-holdout view of the NESTED forecasts: for each milk type and model N+1..N+6, Q1, Q2 -> MAE / weighted MAE / aggregate MAE over all
    batches, the MAE of every single batch, how many batches are inside the target, and the simple baselines on the same rows."""
    rec = []
    for det in (mad, qad):
        d = det[det["origin"].isin(origins) & det["rate_actual"].notna() & det["sel_fc"].notna()].assign(fc=lambda x: x["sel_fc"])
        for (mt, model), g in d.groupby(["milk_type", "model"]):
            if model not in MODELS or model in ("Q3", "Q4") or (model.startswith("N+") and int(model[2:]) > 6):
                continue
            tgt = CBE_TARGET if model == "N+1" else quarter_target(model) if model.startswith("Q") else np.nan
            sc = score(g, "fc")
            row = {"Milk Type": mt.title(), "Model": model, "Batches": g["origin"].nunique(), "Test points (district x batch)": sc["n"], "Districts": sc["districts"],
                   "MAE": sc["MAE"], "Weighted MAE": sc["VW-MAE"], "Aggregate MAE": sc["Aggregate MAE"], "Bias (fcst-actual)": sc["Bias (fcst-actual)"], "Target": tgt}
            per = []
            for k, o in enumerate(origins, 1):
                go = g[g["origin"] == o]
                v = float((go["fc"] - go["rate_actual"]).abs().mean()) if len(go) else np.nan
                row[f"Batch {k} MAE ({o:%b-%y})"] = v
                per.append(v)
            per = np.array(per, dtype=float)
            row["Batches within target"] = int(np.nansum(per <= tgt)) if tgt == tgt else np.nan
            row["Check (MAE)"] = "n/a" if tgt != tgt else "MEETS" if sc["MAE"] <= tgt else "OVER"
            row["Check (weighted MAE)"] = "n/a" if tgt != tgt else "MEETS" if sc["VW-MAE"] <= tgt else "OVER"
            for c, label in (("Naive", "Last month repeated"), ("Rolling 3M avg", "Rolling 3M avg"), ("ARIMA(1,1,0)", "ARIMA(1,1,0)")):
                if c in g.columns and g[c].notna().any():
                    gg = g[g[c].notna()]
                    row[f"{label} MAE"] = float((gg[c] - gg["rate_actual"]).abs().mean())
            rec.append(row)
    r = pd.DataFrame(rec)
    r["_o"] = r["Model"].map({m: i for i, m in enumerate(MODELS)})
    return r.sort_values(["Milk Type", "_o"]).drop(columns="_o").reset_index(drop=True)


def method_scores(mad: pd.DataFrame, qad: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """MAE of every candidate method per model over the errors known at t (this is what the method choice looks at)."""
    rec = []
    for det, cands in ((mad, MONTHLY_CANDS), (qad, QUARTER_CANDS)):
        d = det[(det["target_month"] <= t) & det["rate_actual"].notna()]
        for (mt, model), g in d.groupby(["milk_type", "model"]):
            # cc = [c for c in cands if c in g.columns]                                 # (original)
            cc = [c for c in (cands + (cow_model.COW_COLUMNS if mt == "COW" and cands is MONTHLY_CANDS else [f"Avg of monthly {c}" for c in AVG_OF_COW] if mt == "COW" else []))
                  if c in g.columns and g[c].notna().any()]                              # NEW: cow-only columns are scored for cow, skipped for buffalo
            gg = g.dropna(subset=cc)
            row = {"Milk Type": mt.title(), "Model": model, "n rows": len(gg)}
            row.update({c: float((gg[c] - gg["rate_actual"]).abs().mean()) if len(gg) else np.nan for c in cc})
            rec.append(row)
    r = pd.DataFrame(rec)
    r["_o"] = r["Model"].map({m: i for i, m in enumerate(MODELS)})
    return r.sort_values(["Milk Type", "_o"]).drop(columns="_o").reset_index(drop=True)


def band_coverage(fc: pd.DataFrame) -> pd.DataFrame:
    g = fc.dropna(subset=["Actual (Rs/kg TS)"])
    if g.empty:
        return pd.DataFrame()
    a = g["Actual (Rs/kg TS)"]
    cov = pd.DataFrame({"Milk Type": g["Milk Type"], "Model": g["Model"],
                        "in weather band": (a >= g["Weather band low"]) & (a <= g["Weather band high"]),
                        "in model-error 80% band": (a >= g["Lower 80% (model error)"]) & (a <= g["Upper 80% (model error)"]),
                        "in TOTAL band": (a >= g["TOTAL band low (weather + model error)"]) & (a <= g["TOTAL band high (weather + model error)"]),
                        "weather band width": g["Weather band width"],
                        "total band width": g["TOTAL band high (weather + model error)"] - g["TOTAL band low (weather + model error)"]})
    out = cov.groupby(["Milk Type", "Model"]).agg(n=("in TOTAL band", "size"), **{c: (c, "mean") for c in cov.columns[2:]}).reset_index()
    for c in ("in weather band", "in model-error 80% band", "in TOTAL band"):
        out[c] = out[c] * 100
    out["_o"] = out["Model"].map({m: i for i, m in enumerate(MODELS)})
    return out.sort_values(["Milk Type", "_o"]).drop(columns="_o")


# ===================================================================== importance / equations / saved models
def importance_tables(mt: str, mfits: Dict, qfits: Dict, feats: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Importance % of every input in each of the 8 models (average of the Ridge |standardised coef| share and the XGBoost mean|SHAP| share)
    and the Ridge equation (raw units) of each of the 8 models."""
    cols, eqs = {}, []
    allfits = {f"N+{h}": mfits[h] for h in HORIZONS if h in mfits} | {q: f for q, f in qfits.items()}
    for model in MODELS:
        if model not in allfits:
            continue
        r, x, tr = allfits[model]
        cols[model] = ((s2.pct(s2.ridge_std_coef(r, feats)) + s2.pct(s2.shap_importance(x, tr[feats]))) / 2).to_numpy()
        eqs.append(s2.equation_frame(mt, f"Ridge of model {model} ({len(feats)} inputs)", r, feats, model))
    imp = pd.DataFrame(cols, index=feats)
    imp.insert(0, "Group", [s1.FEATURE_GROUP[f] for f in feats])
    imp.insert(0, "Feature", feats)
    imp["Average over the 8 models %"] = imp[[m for m in MODELS if m in imp.columns]].mean(axis=1)
    imp = imp.sort_values("Average over the 8 models %", ascending=False).reset_index(drop=True)
    imp.insert(0, "Milk Type", mt.title())
    return imp, pd.concat(eqs, ignore_index=True)


def save_models(path: Path, mt: str, origin: pd.Timestamp, feats: List[str], mfits: Dict, qfits: Dict, rfit) -> None:
    """The fitted models of the final origin (so a forecast can be repeated without refitting): joblib file per milk type."""
    import joblib
    pack = {"milk_type": mt, "trained_through": origin, "features": feats,
            "monthly": {f"N+{h}": {"ridge": f[0], "xgb": f[1]} for h, f in mfits.items()},
            "quarterly": {q: {"ridge": f[0], "xgb": f[1]} for q, f in qfits.items()},
            "recursive_one_step": {"ridge": rfit[0], "xgb": rfit[1]}, "xgb_is_real": s2.XGB_IS_REAL}
    joblib.dump(pack, path)


# ===================================================================== main
def main(cfg: s1.Config, outdir: Path, origin: str | None, level: float, weather_band: bool, default_m: str, default_q: str,
         min_n: int, margin: float) -> None:
    global DEFAULT_MONTHLY, DEFAULT_QUARTER
    DEFAULT_MONTHLY, DEFAULT_QUARTER = default_m, default_q
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "models").mkdir(exist_ok=True)
    if not s2.XGB_IS_REAL:
        log.warning("xgboost NOT installed: a stand-in boosted-tree model is used. Every XGB / Blend / ensemble number is INDICATIVE only (pip install xgboost shap).")
    raw = s1.load_raw(cfg)
    panels, market = s1.build_panels(raw, cfg)
    last_actual = raw.loc[raw["row_valid"], s1.DATE].max()
    O = pd.Timestamp(origin) if origin else last_actual
    held = origin is not None and O < last_actual
    log.info("origin %s | last actual %s | %s", f"{O:%b %Y}", f"{last_actual:%b %Y}", "HELD-OUT mode (months after the origin are scored)" if held else "live forecast")
    wb = None
    if weather_band and cfg.forward_inputs:
        wb = s1.weather_bands_at(raw, O, level, cfg.max_horizon)                    # TASK 1: expected weather + seasonal band, data <= origin
    origins = pd.date_range(FIRST_ORIGIN, last_actual - pd.DateOffset(months=1), freq="MS")
    HOLDOUT_SET[:] = s2.holdout_origins(last_actual, s2.HOLDOUT_N, s2.HOLDOUT_STEP, s2.HOLDOUT_LAST)    # multi-holdout batches (same as step 2, PART D)
    log.info("multi-holdout batches (origins): %s", ", ".join(f"{o:%b-%y}" for o in HOLDOUT_SET))

    M_all, Q_all, live_m, live_q, picks_m, picks_q = [], [], {}, {}, {}, {}
    imps, eqs, fitpack = {}, {}, {}
    mad_all, qad_all = [], []
    for mt in s1.MILK_TYPES:
        panel = panels[mt]
        table = s1.build_training_table(panel, market, cfg)
        spec = s1.spec_for(mt, forward=cfg.forward_inputs, flags=cfg.use_flags)
        feats = spec["all"]
        recx = s2.prepare_recx(panel, market, table, cfg, spec)
        log.info("%s: %d inputs (%s flags) | backtest of the 8 models, %d origins", mt, len(feats), "with" if cfg.use_flags else "NO", len(origins))
        mdet, qdet = run_backtest(mt, panel, market, table, recx, cfg, origins)
        mdet = mdet[mdet["rate_actual"].notna()].reset_index(drop=True)
        qdet = qdet[qdet["rate_actual"].notna()].reset_index(drop=True)

        # ---- nested method choice (monthly first, then the quarters, which may average the selected monthly forecasts)
        # NEW: cow chooses among baselines + the new cow members (default cow_model.METHOD); buffalo keeps the original lists and defaults
        cands_m, cands_q, dflt_m = (COW_MONTHLY_CANDS, COW_QUARTER_CANDS, COW_DEFAULT_MONTHLY) if mt == "COW" else (MONTHLY_CANDS, QUARTER_CANDS, default_m)
        # mad = apply_nested(mdet, MONTHLY_CANDS, default_m, min_n, margin)                         # (original)
        mad = apply_nested(mdet, cands_m, dflt_m, min_n, margin)
        qdet["Avg of selected monthly"] = avg_selected_monthly(mad, qdet)
        # qad = apply_nested(qdet, QUARTER_CANDS, default_q, min_n, margin)                         # (original)
        qad = apply_nested(qdet, cands_q, default_q, min_n, margin)
        mad_all.append(mad), qad_all.append(qad)

        # ---- live forecast at origin O (fits kept for importance / equations / saving), with the weather scenarios
        scn = qscn = None
        if wb is not None:
            cal = panel.index.get_level_values(s1.DATE)
            ext = pd.date_range(cal.min(), cal.max() + pd.DateOffset(months=cfg.max_horizon), freq="MS")
            dists = sorted(panel.index.get_level_values(s1.DISTRICT).unique())
            scn = {SCN_LOW: s1.scenario_table(panel, cfg, table, O, s1.scenario_shifts(ext, dists, wb[0], "lo")),
                   SCN_HIGH: s1.scenario_table(panel, cfg, table, O, s1.scenario_shifts(ext, dists, wb[0], "hi"))}
            qscn = {nm: {qid: quarter_table(tb[tb["origin"] == O], feats, hs) for qid, hs in QUARTERS.items()} for nm, tb in scn.items()}
        districts = s1.forecastable_districts(panel, O, cfg.forecast_min_months)
        # dl, fits = s2.forecast_origin(panel, table, O, cfg.max_horizon, cfg, recx, districts, keep_fits=True, scenarios=scn)    # (original)
        dl, fits = s2.forecast_origin(panel, table, O, cfg.max_horizon, cfg, recx, districts, keep_fits=True, scenarios=scn, cow_extra=(mt == "COW"))
        dl = monthly_frame(dl, mt)
        qtabs = {qid: quarter_table(table, feats, hs) for qid, hs in QUARTERS.items()}
        ql, qfits = quarter_forecast(panel, qtabs, O, feats, dl, qscn, keep_fits=True)

        # pm = pick_methods(mdet, O, MONTHLY_CANDS, default_m, min_n, margin)                       # (original)
        pm = pick_methods(mdet, O, cands_m, dflt_m, min_n, margin)
        # 'Avg of selected monthly' for the live quarters: mean of the monthly forecasts of the methods picked at O (also under LOW / HIGH weather)
        for nm in (None,) + (SCNS if wb is not None else ()):
            col = "_sel" + sfx(nm)
            dl[col] = np.nan
            for model, idx in dl.groupby("model").groups.items():
                meth = pm.get((mt, model), {}).get("method", dflt_m)           # (original default_m; NEW dflt_m = cow default for cow)
                dl.loc[idx, col] = get_col(dl.loc[idx], meth, nm).to_numpy()
            for qid, hs in QUARTERS.items():
                m = ql["model"] == qid
                ql.loc[m, "Avg of selected monthly" + sfx(nm)] = avg_of_months(dl, hs, col).reindex(ql.loc[m, s1.DISTRICT]).to_numpy()
        qdet_k = qdet.copy()
        # pq = pick_methods(qdet_k, O, QUARTER_CANDS, default_q, min_n, margin)                     # (original)
        pq = pick_methods(qdet_k, O, cands_q, default_q, min_n, margin)
        picks_m.update(pm), picks_q.update(pq)
        live_m[mt], live_q[mt] = dl, ql
        M_all.append(mdet.assign(milk_type=mt)), Q_all.append(qdet.assign(milk_type=mt))
        mfits = {h: fits[("Direct", h)] for h in HORIZONS if ("Direct", h) in fits}
        imps[mt], eqs[mt] = importance_tables(mt, mfits, qfits, feats)
        save_models(outdir / "models" / f"{mt.lower()}_8_models.joblib", mt, O, feats, mfits, qfits, fits[("RecursiveX", 1)])

    MD, QD = pd.concat(M_all, ignore_index=True), pd.concat(Q_all, ignore_index=True)
    MAD, QAD = pd.concat(mad_all, ignore_index=True), pd.concat(qad_all, ignore_index=True)
    MD.to_csv(outdir / "horizon_backtest_monthly.csv", index=False)
    QD.to_csv(outdir / "horizon_backtest_quarterly.csv", index=False)

    fc = pd.concat([forecast_rows(mt, live_m[mt], live_q[mt], picks_m, picks_q, MD, QD, O, held) for mt in s1.MILK_TYPES], ignore_index=True)
    val = validation_table(MAD, QAD, pd.Timestamp("2025-12-01") if not held else O)
    hold_tab = holdout_table(MAD, QAD, HOLDOUT_SET)
    mscores = method_scores(MD, QD, O)
    cov = band_coverage(fc) if held else pd.DataFrame()

    # ---- summary: one row per milk type x model
    rec = []
    for mt in s1.MILK_TYPES:
        for model in MODELS:
            p = (picks_m if not model.startswith("Q") else picks_q).get((mt, model))
            if p is None:
                continue
            g = fc[(fc["Milk Type"] == mt.title()) & (fc["Model"] == model)]
            if g.empty:
                continue
            row = {"Milk Type": mt.title(), "Model": model, "What it predicts": ("rate of month " + g["Forecast period"].iloc[0]) if not model.startswith("Q") else
                   "average rate of " + g["Forecast period"].iloc[0], "Method used": p["method"], "Errors behind the choice (n)": p["n"],
                   "Districts forecast": g["District"].nunique(), "Avg weather band width": g["Weather band width"].mean(),
                   "Avg TOTAL band width": (g["TOTAL band high (weather + model error)"] - g["TOTAL band low (weather + model error)"]).mean()}
            for scope in val["Scope"].unique():
                v = val[(val["Scope"] == scope) & (val["Milk Type"] == mt.title()) & (val["Model"] == model)]
                row[f"MAE, scope {scope[:1]}"] = float(v["MAE"].iloc[0]) if len(v) else np.nan
                row[f"Aggregate MAE, scope {scope[:1]}"] = float(v["Aggregate MAE"].iloc[0]) if len(v) else np.nan
            rec.append(row)
    summary = pd.DataFrame(rec)

    notes = pd.DataFrame({"Notes": [
        f"Origin {O:%b %Y}. Models trained only on rows whose target month (quarters: LAST month of the quarter) is <= origin (assert_no_leakage).",
        f"XGBoost is {'REAL' if s2.XGB_IS_REAL else 'a STAND-IN (xgboost not installed): XGB / Blend / ensemble numbers are INDICATIVE only'}.",
        "N+h = Direct model of horizon h (its own training rows). Q1 / Q2 = dedicated quarterly models (average of the 3 months), or the average of the monthly forecasts - whichever the track record prefers.",
        f"Method choice: lowest MAE over errors known at the origin; default '{default_m}' (monthly) / '{default_q}' (quarter) replaced only if a challenger is >= {margin:.0%} better and >= {min_n} rows exist.",
        "Validation uses the NESTED choice (the method chosen at each origin from errors known then), so it is an out-of-sample estimate. Scope A = origins Dec-24..Nov-25, B = held-out origin (ONE origin = one test month per horizon, do not quote it alone), C = all origins, D = the multi-holdout batches (6 origins, 2 months apart; same as step 2 PART D).",
        "Sheet Holdout_6_Batches = the evidence for the client: N+1..N+6, Q1, Q2 over 6 batches, MAE / weighted MAE / aggregate MAE, MAE of each batch, batches inside target, and the simple baselines on the same rows.",
        "Targets: N+1 <= 2.0 Rs/kg TS (CBE), Q1 <= 5.0. Q1 is shown two ways: error of the quarter AVERAGE (what the Q1 model predicts) and the average MONTHLY error inside the quarter (step 2's definition).",
        "Aggregate MAE = error of the volume-weighted all-district rate (the client's 2.37 may be measured there). District MAE is the stricter number.",
        "Weather band: same fitted models run with LOW / EXPECTED / HIGH weather (sheet Weather_Band_Rules). Fat % and SNF % stay at expected values in all three runs. Weather band is NOT a forecast interval; use the TOTAL band.",
        "Intervention flags run_up_share / crash_share are inputs of all 8 models; they are 0 for forecast months after the 2022-24 swing. Dates are data-detected until the business confirms them (--run-up / --crash).",
        "Only districts with a rate in each of the last 6 months are forecast; few districts => results move a lot with one district.",
        f"COW uses its own method ({cow_model.METHOD}, cow_model.py): average of 2 HistGradientBoosting models and a momentum Ridge per horizon; the cow choice is only among the naive baselines and the cow members (old Ridge / XGBoost / RecursiveX can not be picked for cow). BUFFALO is unchanged. Importance_COW / Ridge_Equations still describe the OLD shared models (see cow_model.py for the cow inputs).",
        "Selected models are saved in outputs/models/*.joblib (Ridge + XGBoost of N+1..N+6, Q1, Q2 and the one-step RecursiveX model).",
    ]})
    sheets = {"Summary": summary, "Holdout_6_Batches": hold_tab, "Validation": val}
    for model in MODELS:
        sheets[f"Model_{model}"] = fc[fc["Model"] == model].drop(columns="Model")
    sheets["Forecast_All_Models"] = fc
    sheets["Method_Scores"] = mscores
    if len(cov):
        sheets["Band_Coverage"] = cov
    if wb is not None:
        live = sorted(set(fc["District"]))
        sheets["Weather_Band"] = wb[1][wb[1]["District"].isin(live)].rename(columns={"low": "Low", "expected": "Expected", "high": "High"})
        sheets["Weather_Band_Rules"] = wb[0]
    sheets["Intervention_Windows"] = s1.intervention_table(panels, cfg)
    sheets["Importance_COW"], sheets["Importance_BUFFALO"] = imps["COW"], imps["BUFFALO"]
    sheets["Ridge_Equations"] = pd.concat([eqs["COW"], eqs["BUFFALO"]], ignore_index=True)
    sheets["Notes"] = notes
    fname = f"Horizon_Forecast_from_{O:%b%Y}_heldout.xlsx" if held else "Horizon_Forecast.xlsx"
    s2.write_excel(outdir / fname, sheets)

    # ---- console report
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(f"\n=== VALIDATION (nested method choice; MAE Rs/kg TS) - district level and all-district aggregate ===")
    show = val[~val["Model"].str.contains("avg monthly")][["Scope", "Milk Type", "Model", "n", "districts", "MAE", "Aggregate MAE", "Bias (fcst-actual)",
                                                           "Direction acc %", "MAE if rate stays flat", "Target", "Check (district level)", "Check (aggregate level)"]]
    print(show.round(2).to_string(index=False))
    print("\nQ1 / Q2 as the AVERAGE MONTHLY error inside the quarter (step 2 definition):")
    print(val[val["Model"].str.contains("avg monthly")][["Scope", "Milk Type", "Model", "n", "MAE", "Aggregate MAE", "Target", "Check (district level)"]].round(2).to_string(index=False))
    print(f"\n=== MULTI-HOLDOUT: {len(HOLDOUT_SET)} batches ({', '.join(f'{o:%b-%y}' for o in HOLDOUT_SET)}) - nested forecast of each of the 8 models ===")
    print(hold_tab.round(2).to_string(index=False))
    print("\nMETHOD USED AT ORIGIN", f"{O:%b %Y}")
    print(summary[["Milk Type", "Model", "What it predicts", "Method used", "Errors behind the choice (n)", "Avg weather band width", "Avg TOTAL band width"]].round(1).to_string(index=False))
    if held:
        print(f"\nHELD-OUT: the 8 models fitted at {O:%b %Y} vs the actuals that followed (band coverage, %):")
        print(cov.round(1).to_string(index=False))
    print("\nsaved ->", outdir / fname)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Step 4: the 8 horizon models (N+1..N+6 monthly, Q1 / Q2 quarterly) with weather bands, flags and validation")
    ap.add_argument("--data", default=s1.Config.data_xlsx)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--origin", default=None, help="held-out check: forecast from this origin, e.g. 2025-12-01 (default: last actual month)")
    ap.add_argument("--level", type=float, default=95.0, help="weather band level in %% (default 95)")
    ap.add_argument("--no-weather-band", action="store_true", help="skip the LOW / HIGH weather runs")
    ap.add_argument("--default-monthly", default=DEFAULT_MONTHLY, choices=MONTHLY_CANDS, help="monthly method used until a better one has a track record")
    ap.add_argument("--default-quarter", default=DEFAULT_QUARTER, choices=QUARTER_CANDS, help="quarter method used until a better one has a track record")
    ap.add_argument("--min-select-n", type=int, default=MIN_SELECT_N)
    ap.add_argument("--switch-margin", type=float, default=SWITCH_MARGIN, help="a challenger must beat the default by this share (0.05 = 5%%)")
    ap.add_argument("--holdout-n", type=int, default=s2.HOLDOUT_N, help="number of multi-holdout batches (default 6)")
    ap.add_argument("--holdout-step", type=int, default=s2.HOLDOUT_STEP, help="months between batch origins (default 2)")
    ap.add_argument("--holdout-last", default=None, help="last batch origin, e.g. 2025-12-01 (default: latest origin with 6 actual months after it)")
    ap.add_argument("--no-forward", action="store_true", help="old behaviour: weather / fat / SNF observed at the origin")
    ap.add_argument("--no-flags", action="store_true", help="do not use the intervention flags as inputs")
    ap.add_argument("--keep-2021", action="store_true", help="keep training rows whose target month is before 2022")
    s1.add_swing_args(ap)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s2.HOLDOUT_N, s2.HOLDOUT_STEP, s2.HOLDOUT_LAST = a.holdout_n, a.holdout_step, a.holdout_last
    main(s1.config_from_args(a), s1.resolve(a.outdir), a.origin, a.level, not a.no_weather_band, a.default_monthly, a.default_quarter,
         a.min_select_n, a.switch_margin)

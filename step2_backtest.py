"""
COW v2 (NEW): COW is forecast by its own method, "Cow Ensemble v2" (cow_model.py, must sit in the same folder).  BUFFALO and every shared
function are unchanged (cow_extra=False is the original behaviour); the replaced cow lines are kept as comments marked "(original)".
step2_backtest.py - FILE 2 of 4 (MERGED: step2_walk_forward + step2c_variant_comparison + stepB_intervention_report)
=====================================================================================================
backtest in three parts, COW and BUFFALO separately, DISTRICT level, Rs/kg TS
=====================================================================================================
INPUT LISTS: COW 42 / BUFFALO 46 inputs in the full set; if selected_features.json exists (no longer used)
the pruned lists are used instead.  Lists live in step1: FEATURES_BY_TYPE.
The small 10 / 12-input models of the first version (PriceOnly, Recursive, RecursiveONI) are removed.

One engine, ``forecast_origin()``: "standing at the end of month t, train only on what was known at t,
forecast t+1 ... t+H".  Everything below is built from it.

TEST A  (2025 backtest, 1 month ahead, retrained every month)
    train Jan-21..Dec-24 -> predict Jan-25 ; train ..Jan-25 -> predict Feb-25 ; ... ; train ..Nov-25 -> predict Dec-25
    12 runs, every forecast is h=1.  Metrics: MAE, RMSE, MAPE, bias, directional accuracy, volatility score, trust score.

TEST B  (March origin)
    train ..Mar-26 -> forecast Apr, May, Jun 2026 (h=1,2,3) and compare with actuals.
    CBE check: April (h=1) MAE <= 2.0.      Q1 check: Apr-May-Jun MAE <= 5.0.
    The same window one year earlier (train ..Mar-25 -> Apr-Jun 2025) shows whether things are improving.

METHOD COMPARISON  (origins Dec-24 .. May-26, h = 1..6) - decides the method used for the final forecast
    DIRECT       all inputs (42 cow / 46 buffalo, +2 intervention flags), one model per horizon.  Weather and fat / SNF are
                 values FOR THE TARGET MONTH: actual up to the origin, same-month average of earlier years after it (step 1,
                 forward_inputs); supply / incentive are observed at the origin; market / ONI are projected to the target month.
    RECURSIVEX   ONE one-step model with the same inputs.  Forecast t+1, treat it as if it were an actual, recompute the
                 price inputs, forecast t+2, and so on.  Observed inputs (weather, quality, supply, incentive) are taken
                 6 months before the month being forecast (so they are always known); market / ONI are projected.
    each with Ridge, XGBoost and Blend (plain average of the two predicted changes), plus 3 naive baselines.

Rules: training rows for origin t only have target month <= t (asserted); weather is never forecast; model
settings are fixed (nothing is tuned on a test window).

MULTI-HOLDOUT REPORT (PART D, new)  -  "6 data points per horizon"
    One forecast origin gives ONE test point per horizon (N+1 = one month only).  The report therefore repeats the whole forecast from
    HOLDOUT_N = 6 different origins ("batches", default every 2 months over one year, the last one = last origin that still has 6 actual months).
    Every batch is retrained only on data known at its origin (the engine above asserts this).  For each of N+1 .. N+6 and Q1 / Q2 there are
    then 6 errors per district; MAE and VOLUME-WEIGHTED MAE are reported next to the baselines the client can understand
    (last month repeated, rolling 3-month average, rolling 6-month average, ARIMA(1,1,0)).  -> MultiHoldout_Report.xlsx

    python step2_backtest.py                 (main backtest + multi-holdout report)
    python step2_backtest.py --variants --intervention-report   (also the 5-variant comparison and the intervention report)
    python step2_backtest.py --holdout-n 6 --holdout-step 2 --holdout-last 2025-12-01 --headline-method "RecursiveX Blend"
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import step1_data_features as s1
import cow_model                                      # NEW: separate COW method (cow_model.py); buffalo never uses it

try:
    from xgboost import XGBRegressor
    XGB_IS_REAL = True
except ImportError:                                   # only so the code can be smoke-tested without xgboost
    from sklearn.ensemble import GradientBoostingRegressor
    XGB_IS_REAL = False

warnings.filterwarnings("ignore")
log = logging.getLogger("milk")

# ------------------------------------------------------------------ settings
TEST_A_FIRST_ORIGIN, TEST_A_LAST_ORIGIN = "2024-12-01", "2025-11-01"   # targets Jan-25 .. Dec-25
TEST_B_ORIGIN, TEST_B_PRIOR_ORIGIN = "2026-03-01", "2025-03-01"        # targets Apr-Jun 2026 / Apr-Jun 2025
COMPARE_FIRST_ORIGIN = "2024-12-01"
CBE_TARGET, Q1_TARGET, CLIENT_CBE_MAE = 2.0, 5.0, 2.37                 # Rs/kg TS
MIN_TRAIN_ROWS = 25
N_BOOT, BOOT_SEED = 2000, 0
# ---- multi-holdout report (PART D)
HOLDOUT_N, HOLDOUT_STEP, HOLDOUT_LAST = 6, 2, None      # 6 batches, origins 2 months apart, last = last origin with 6 actual months after it
HEADLINE_METHOD = "RecursiveX Blend"                      # the method shown as "our model" in the client tables (fixed in advance, not picked on the test)
# NEW (COW only): cow has its own method - "Cow Ensemble v2" (cow_model.py).  BUFFALO keeps HEADLINE_METHOD above, unchanged.
HEADLINE_BY_TYPE = {"COW": cow_model.METHOD}
HOLDOUT_H = 6
HOLDOUT_QUARTERS = {"Q1": (1, 2, 3), "Q2": (4, 5, 6)}
HOLDOUT_QUARTER_TARGET = {"Q1": 5.0, "Q2": 8.0}

# RecursiveX = ONE one-step model whose inputs are all known on the forecast date:
#   price inputs   : recomputed from the rate history, including already-forecast months
#   observed inputs: weather / quality / supply / incentive observed RECX_LAG months before the month being forecast
#                    (RECX_LAG >= the longest horizon, so October weather feeds a March forecast as a 5-month lag or more)
#   market inputs  : last value + k x average of the last 6 monthly changes (projected, see step 1)
#   For the 12-month backtest, the recursion must be able to use inputs known up to 12 months ahead, therefore the lag
#   must match the full forecast horizon rather than the previous 6-month assumption.
RECX_LAG = 12
FAMILIES = ["Direct", "RecursiveX"]
LEARNERS = ["Ridge", "XGB", "Blend"]
BASELINES = ["Naive", "Seasonal-trend naive", "Drift (6m avg)"]
# extra comparison methods for the client report ("compare 4-5 models"): flat forecasts / simple time-series, no inputs
COMPARATORS = ["Rolling 3M avg", "Rolling 6M avg", "ARIMA(1,1,0)"]
LEARNED = [f"{f} {m}" for f in FAMILIES for m in LEARNERS]
ALL_METHODS = BASELINES + LEARNED


# -------------------------------------------------------------------- models
def make_ridge():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         RidgeCV(alphas=np.logspace(-1, 3, 13)))


if XGB_IS_REAL:
    def make_xgb():
        return XGBRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, min_child_weight=3,
                            subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
                            random_state=42, n_jobs=1)
else:
    class _StandIn(GradientBoostingRegressor):         # NOT XGBoost - numbers from it are indicative only
        def _fill(self, X):
            X = np.array(X, dtype=float)
            i = np.where(np.isnan(X))
            X[i] = np.take(self._med, i[1])
            return X

        def fit(self, X, y):
            self._med = np.nan_to_num(np.nanmedian(np.asarray(X, dtype=float), axis=0))
            return super().fit(self._fill(X), y)

        def predict(self, X):
            return super().predict(self._fill(X))

    def make_xgb():
        return _StandIn(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, max_features=0.7,
                        min_samples_leaf=3, random_state=42)


def fit_pair(train: pd.DataFrame, feats: List[str]):
    X, y = train[feats], train["target"]
    return make_ridge().fit(X, y), make_xgb().fit(X, y)


def baseline_changes(rows: pd.DataFrame) -> Dict[str, np.ndarray]:
    h = rows["horizon"].to_numpy()
    return {"Naive": np.zeros(len(rows)),
            "Seasonal-trend naive": rows["ly_change"].fillna(0.0).to_numpy(),
            "Drift (6m avg)": rows["diff_mean6"].fillna(0.0).to_numpy() * h}


# ------------------------------------------------------- recursive machinery
def price_feature_frame(R: pd.DataFrame, target_month: pd.Timestamp) -> pd.DataFrame:
    """Price + calendar inputs at the LAST row of the wide rate table R (dates x districts), for forecasting the next
    month.  Uses the SAME price_block() as step 1 (checked in ``self_check``)."""
    F = pd.DataFrame({k: W.iloc[-1] for k, W in s1.price_block(R).items()})
    F["ly_change"] = R.iloc[-12] - R.iloc[-13]                 # h = 1:  Rate(t-11) - Rate(t-12)
    for k, v in s1.calendar_features(pd.DatetimeIndex([target_month])).items():
        F[k] = v[0]
    return F


def prepare_recx(panel: pd.DataFrame, market: pd.DataFrame, table: pd.DataFrame, cfg: s1.Config, spec: Dict) -> Dict:
    """Training table and lookups for the RecursiveX model (one-step rows; table-fed inputs for the target month, lagged observed
    inputs, market at origin).  Table-fed = forward weather / fat / SNF and the intervention flags: for step k of a recursion that
    starts at origin t they are read from the table row (origin t, horizon k), so they never depend on the recursion itself."""
    of = s1.origin_features(panel, market)
    wx = of[spec["observed"]]
    t1 = table[table["horizon"] == 1].copy()
    idx = pd.MultiIndex.from_arrays([t1[s1.DISTRICT], t1["origin"]])
    # target month s = origin + 1 ; inputs observed RECX_LAG months before s = RECX_LAG-1 months before the origin
    if spec["observed"]:
        lagged = wx.groupby(level=s1.DISTRICT).shift(RECX_LAG - 1).reindex(idx)
        for c in lagged.columns:
            t1[c] = lagged[c].to_numpy()
    m0 = s1.projected_market_features(market, 0, cfg.proj_window).reindex(t1["origin"])   # actual values at the step origin
    for c in spec["market"]:
        t1[c] = m0[c].to_numpy()
    return {"train": t1, "wx": wx, "market": market, "cfg": cfg, "spec": spec, "table": table}


def paths_from_fit(panel: pd.DataFrame, ridge, xgb, t: pd.Timestamp, H: int, feats: List[str], exog):
    """Recursive forecasts of the RATE for every district alive at t, from ALREADY FITTED one-step models.
    Returns {learner: {k: Series(district)}}.  ``exog(k)`` gives the observed / market / table-fed inputs of step k."""
    base = panel["rate"].unstack(s1.DISTRICT).sort_index().loc[:t]
    out: Dict[str, Dict[int, pd.Series]] = {}
    for kind in LEARNERS:
        R, paths = base.copy(), {}
        for k in range(1, H + 1):
            d = R.index[-1]
            tgt = d + pd.DateOffset(months=1)
            F = pd.concat([price_feature_frame(R, tgt), exog(k).reindex(R.columns)], axis=1)
            live = (R.loc[d].notna() & F["diff_1"].notna()).to_numpy()
            pr, px = ridge.predict(F[feats]), xgb.predict(F[feats])
            chg = {"Ridge": pr, "XGB": px, "Blend": (pr + px) / 2}[kind]
            new = pd.Series(np.nan, index=R.columns)
            new[live] = R.loc[d].to_numpy()[live] + chg[live]
            R.loc[tgt] = new                                   # the forecast is fed back as if it were an actual
            paths[k] = new
        out[kind] = paths
    return out


def recursive_paths(panel: pd.DataFrame, train1: pd.DataFrame, t: pd.Timestamp, H: int, feats: List[str], exog):
    """Fit the one-step Ridge + XGBoost on ``train1`` and run the recursion (kept for callers that want both in one call)."""
    ridge, xgb = fit_pair(train1, feats)
    return paths_from_fit(panel, ridge, xgb, t, H, feats, exog), (ridge, xgb)


def make_exog(recx: Dict, t: pd.Timestamp):
    """Observed + market inputs for step k of a recursion that starts at origin t (everything known at t)."""
    wx, market, cfg, spec = recx["wx"], recx["market"], recx["cfg"], recx["spec"]
    cache = {}
    fed = spec["table_fed"]
    tbl = recx["table"]
    at_t = tbl[tbl["origin"] == t].set_index([s1.DISTRICT, "horizon"])[fed] if fed else None

    def exog(k: int) -> pd.DataFrame:
        s_lag = t + pd.DateOffset(months=k - RECX_LAG)          # observed month = target month - RECX_LAG  (<= t since k <= RECX_LAG)
        assert s_lag <= t, "lag shorter than the horizon: input would not be known"
        w = wx.xs(s_lag, level=s1.DATE)
        if k - 1 not in cache:                                  # market at the step origin t+k-1, projected from t
            cache[k - 1] = s1.projected_market_features(market, k - 1, cfg.proj_window).loc[t]
        w = w.assign(**{c: cache[k - 1][c] for c in spec["market"]})
        if fed:                                                 # forward weather / fat / SNF + flags of the target month t+k, known at t
            blk = at_t.xs(k, level="horizon").copy()
            flags = [c for c in fed if c in s1.FLAG_FEATURES]
            if flags and k > 1:                                 # the one-step model was trained on "is month t+1 inside the window" (0/1),
                prev = at_t.xs(k - 1, level="horizon")          # the table row of horizon k holds the SHARE of months t+1..t+k -> convert:
                for c in flags:                                 #   indicator(t+k) = k x share(k) - (k-1) x share(k-1)
                    blk[c] = (k * blk[c] - (k - 1) * prev[c].reindex(blk.index)).clip(0.0, 1.0).round(6)
            w = pd.concat([w, blk], axis=1)
        return w
    return exog


# ------------------------------------------------------- simple comparison forecasts
def arima110(y: np.ndarray, H: int) -> np.ndarray:
    """ARIMA(1,1,0) with a constant, fitted by conditional least squares (numpy only, no statsmodels needed):
    d_t = c + phi * d_(t-1) on the monthly changes d, forecast = last rate + cumulated predicted changes."""
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < 12:
        return np.full(H, np.nan)
    d = np.diff(y)
    x, z = d[:-1], d[1:]
    xm, zm, var = x.mean(), z.mean(), ((x - x.mean()) ** 2).sum()
    phi = float(np.clip(((x - xm) * (z - zm)).sum() / var, -0.95, 0.95)) if var > 0 else 0.0
    c = zm - phi * xm
    last_d, level, out = d[-1], y[-1], []
    for _ in range(H):
        last_d = c + phi * last_d
        level += last_d
        out.append(level)
    return np.array(out)


def comparator_forecasts(panel: pd.DataFrame, t: pd.Timestamp, districts: List[str], H: int) -> pd.DataFrame:
    """Rolling 3M / 6M average (flat forecast = mean of the last 3 / 6 rates) and ARIMA(1,1,0) for every district and horizon.
    Only rates dated <= t are used.  Index = (district, horizon)."""
    R = panel["rate"].unstack(s1.DISTRICT).sort_index().loc[:t]
    rows = []
    for dist in districts:
        s = R[dist]
        r3, r6 = s.iloc[-3:].mean(), s.iloc[-6:].mean()
        ar = arima110(s.dropna().to_numpy(), H)
        for h in range(1, H + 1):
            rows.append({s1.DISTRICT: dist, "horizon": h, "Rolling 3M avg": r3, "Rolling 6M avg": r6, "ARIMA(1,1,0)": ar[h - 1]})
    return pd.DataFrame(rows).set_index([s1.DISTRICT, "horizon"])


# ------------------------------------------------------------------- engine
def forecast_origin(panel: pd.DataFrame, table: pd.DataFrame, t, H: int, cfg: s1.Config, recx: Dict,
                    districts: List[str] | None = None, keep_fits: bool = False,
                    scenarios: Dict[str, pd.DataFrame] | None = None, cow_extra: bool = False):
    """Stand at the end of month t and forecast t+1..t+H for every eligible district.

    ``recx`` carries the milk type's feature spec.  Returns (DataFrame with one row per district x horizon and one
    column per method, fits dict).

    ``scenarios`` = {name: table}: copies of ``table`` whose forecast rows (origin t) carry other forward-weather inputs
    (see step 1 scenario_table).  The models are fitted ONCE; every scenario only changes the inputs of the forecast rows and adds
    columns "<learned method> | <name>".
    """
    scenarios = scenarios or {}
    t = pd.Timestamp(t)
    feats = recx["spec"]["all"]
    if districts is None:
        districts = s1.forecastable_districts(panel, t, cfg.forecast_min_months)
    fits, parts = {}, []
    comps = comparator_forecasts(panel, t, districts, H)

    trx = recx["train"][(recx["train"]["target_month"] <= t) & recx["train"]["target"].notna()]
    s1.assert_no_leakage(trx, t)
    r1, x1 = fit_pair(trx, feats)
    recx_fit = (r1, x1)
    recx_paths = paths_from_fit(panel, r1, x1, t, H, feats, make_exog(recx, t))
    scn_paths = {nm: paths_from_fit(panel, r1, x1, t, H, feats, make_exog({**recx, "table": tb}, t)) for nm, tb in scenarios.items()}
    if keep_fits:
        fits[("RecursiveX", 1)] = (*recx_fit, trx)

    for h in range(1, H + 1):
        th = table[table["horizon"] == h]
        train = th[(th["target_month"] <= t) & th["target"].notna()]
        s1.assert_no_leakage(train, t)
        test = th[(th["origin"] == t) & th[s1.DISTRICT].isin(districts)]
        if test.empty or len(train) < MIN_TRAIN_ROWS:
            continue
        res = test[[s1.DISTRICT, "origin", "horizon", "target_month", "rate_t", "rate_actual"]].copy()
        res["n_train"] = len(train)
        for m, c in baseline_changes(test).items():
            res[m] = test["rate_t"].to_numpy() + c
        cdf = comps.reindex(pd.MultiIndex.from_arrays([test[s1.DISTRICT], test["horizon"]]))
        for m in COMPARATORS:
            res[m] = cdf[m].to_numpy()
        r, x = fit_pair(train, feats)
        pr, px = r.predict(test[feats]), x.predict(test[feats])
        for name, c in (("Ridge", pr), ("XGB", px), ("Blend", (pr + px) / 2)):
            res[f"Direct {name}"] = test["rate_t"].to_numpy() + c
        if keep_fits:
            fits[("Direct", h)] = (r, x, train)
        for kind in LEARNERS:
            res[f"RecursiveX {kind}"] = recx_paths[kind][h].reindex(test[s1.DISTRICT]).to_numpy()
        for nm, tb in scenarios.items():                        # same fitted models, other forward-weather inputs
            ts = tb[(tb["horizon"] == h) & (tb["origin"] == t)].set_index(s1.DISTRICT).reindex(test[s1.DISTRICT])
            pr_s, px_s = r.predict(ts[feats]), x.predict(ts[feats])
            for kind, c in (("Ridge", pr_s), ("XGB", px_s), ("Blend", (pr_s + px_s) / 2)):
                res[f"Direct {kind} | {nm}"] = test["rate_t"].to_numpy() + c
                res[f"RecursiveX {kind} | {nm}"] = scn_paths[nm][kind][h].reindex(test[s1.DISTRICT]).to_numpy()
        vol = panel["ts_kg"].reindex(pd.MultiIndex.from_arrays([test[s1.DISTRICT], test["target_month"]]))
        res["weight_ts_kg"] = vol.to_numpy()
        parts.append(res)
    # return pd.concat(parts, ignore_index=True), fits                       # (original return - kept for BUFFALO: cow_extra=False gives exactly this)
    out = pd.concat(parts, ignore_index=True)
    if cow_extra:                                                            # NEW, COW only: add the columns of the separate cow model (cow_model.py)
        cw = cow_model.cow_columns(table, t, districts, H, scenarios)
        out = out.merge(cw, on=[s1.DISTRICT, "horizon"], how="left")
    return out, fits


def fiscal_quarter(month: pd.Timestamp) -> str:
    fy = month.year if month.month >= 4 else month.year - 1
    return f"FY{fy % 100}-{(fy + 1) % 100} Q{((month.month - 4) % 12) // 3 + 1}"


def run_backtest(mt: str, panel: pd.DataFrame, table: pd.DataFrame, cfg: s1.Config, recx: Dict, H: int = 6) -> pd.DataFrame:
    last_actual = panel["rate"].dropna().index.get_level_values(s1.DATE).max()
    origins = pd.date_range(COMPARE_FIRST_ORIGIN, last_actual - pd.DateOffset(months=1), freq="MS")
    rows = []
    for t in origins:
        # d, _ = forecast_origin(panel, table, t, H, cfg, recx)                # (original call)
        d, _ = forecast_origin(panel, table, t, H, cfg, recx, cow_extra=(mt == "COW"))   # NEW: COW also gets the separate cow model's columns
        d["milk_type"] = mt
        rows.append(d)
        log.info("  %s origin %s done", mt, f"{t:%b %Y}")
    d = pd.concat(rows, ignore_index=True)
    d = d[d["rate_actual"].notna()].copy()                       # score only months whose actual has arrived
    d["fiscal_quarter"] = d["target_month"].map(fiscal_quarter)
    return d


def self_check(panel: pd.DataFrame, table: pd.DataFrame, spec: Dict) -> float:
    """Recursive feature builder must reproduce step 1's h=1 price / calendar inputs (guards against silent drift)."""
    t = pd.Timestamp("2025-06-01")
    ref = table[(table["horizon"] == 1) & (table["origin"] == t)].set_index(s1.DISTRICT)
    R = panel["rate"].unstack(s1.DISTRICT).sort_index().loc[:t]
    cols = spec["price_cal"]
    F = price_feature_frame(R, t + pd.DateOffset(months=1)).loc[ref.index, cols]
    return float(np.nanmax(np.abs(F.to_numpy() - ref[cols].to_numpy())))


# ------------------------------------------------------------------ metrics
def _dir_acc(g: pd.DataFrame, m: str) -> float:
    if m == "Naive":
        return np.nan
    return float((np.sign(g[m] - g["rate_t"]) == np.sign(g["rate_actual"] - g["rate_t"])).mean() * 100)


def volume_weighted_mae(g: pd.DataFrame, method: str) -> float:
    """Volume-weighted MAE using the district-month procurement volume as the weight.
    The panel already stores the total procured solids volume per district-month, so this is not a
    chilling-centre re-aggregation and avoids double-counting within the same state-month bucket."""
    w = g["weight_ts_kg"].fillna(0.0)
    mask = g["rate_actual"].notna() & g[method].notna() & (w > 0)
    if not mask.any():
        return np.nan
    err = (g.loc[mask, method] - g.loc[mask, "rate_actual"]).abs()
    return float((err * w.loc[mask]).sum() / w.loc[mask].sum())


def metric_row(g: pd.DataFrame, m: str) -> Dict[str, float]:
    e = g[m] - g["rate_actual"]
    mae = e.abs().mean()
    vmae = volume_weighted_mae(g, m) if "weight_ts_kg" in g.columns else np.nan
    naive_mae = (g["Naive"] - g["rate_actual"]).abs().mean()
    chg_f, chg_a = g[m] - g["rate_t"], g["rate_actual"] - g["rate_t"]
    vol = chg_f.std() / chg_a.std() if len(g) > 2 and chg_a.std() > 0 else np.nan
    dacc = _dir_acc(g, m)
    skill = 1 - mae / naive_mae if naive_mae > 0 else np.nan
    trust = np.nan
    if m != "Naive" and not np.isnan(skill):
        trust = 100 * (0.4 * min(1.0, CBE_TARGET / mae) + 0.3 * float(np.clip(skill, 0, 1))
                       + 0.3 * float(np.clip((dacc / 100 - 0.5) / 0.5, 0, 1)))
    return {"n": len(g), "MAE": mae, "VW-MAE": vmae, "RMSE": float(np.sqrt((e ** 2).mean())),
            "MAPE %": (e.abs() / g["rate_actual"]).mean() * 100, "Bias (fcst-actual)": e.mean(),
            "Directional acc %": dacc, "Volatility score": vol, "Skill vs naive": skill, "Trust score": trust}


def metrics_table(d: pd.DataFrame, by: List[str], methods: List[str] | None = None) -> pd.DataFrame:
    methods = methods or [m for m in ALL_METHODS if m in d.columns]
    rec = []
    for keys, g in d.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for m in methods:
            rec.append({**dict(zip(by, keys)), "method": m, **metric_row(g, m)})
    return pd.DataFrame(rec)


def aggregate_mae(d: pd.DataFrame, by: List[str], methods: List[str] | None = None) -> pd.DataFrame:
    """MAE of the VOLUME-WEIGHTED all-district rate (one aggregate rate per target month and horizon)."""
    methods = methods or [m for m in ALL_METHODS if m in d.columns]
    rec = []
    for keys, g in d.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for m in methods:
            w = g["weight_ts_kg"].fillna(0)
            gg = g.assign(_w=w, _a=g["rate_actual"] * w, _f=g[m] * w)
            s = gg.groupby(["origin", "horizon"])[["_w", "_a", "_f"]].sum()
            s = s[s["_w"] > 0]
            err = (s["_f"] / s["_w"] - s["_a"] / s["_w"]).abs()
            rec.append({**dict(zip(by, keys)), "method": m, "n_months": len(err), "Aggregate MAE": err.mean(),
                        "Volume-weighted MAE": err.mean()})
    return pd.DataFrame(rec)


def pivot(t: pd.DataFrame, index: List[str], value: str = "MAE") -> pd.DataFrame:
    cols = [m for m in ALL_METHODS if m in set(t["method"])]
    return t.pivot_table(index=index, columns="method", values=value)[cols].reset_index()


def bootstrap_gain(d: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    """Gain over naive = MAE(naive) - MAE(method); 95% interval from resampling ORIGIN months."""
    rng = np.random.default_rng(BOOT_SEED)
    rec = []
    for mt, g in d.groupby("milk_type"):
        origins = np.array(sorted(g["origin"].unique()))
        idx = {o: g["origin"] == o for o in origins}
        en = {o: (g.loc[idx[o], "Naive"] - g.loc[idx[o], "rate_actual"]).abs().sum() for o in origins}
        cnt = {o: idx[o].sum() for o in origins}
        for m in methods:
            if m == "Naive":
                continue
            em = {o: (g.loc[idx[o], m] - g.loc[idx[o], "rate_actual"]).abs().sum() for o in origins}
            gains = []
            for _ in range(N_BOOT):
                pick = rng.choice(origins, size=len(origins), replace=True)
                n = sum(cnt[o] for o in pick)
                gains.append((sum(en[o] for o in pick) - sum(em[o] for o in pick)) / n)
            tot = sum(cnt.values())
            rec.append({"milk_type": mt, "method": m, "gain vs naive (Rs/kg TS)": (sum(en.values()) - sum(em.values())) / tot,
                        "95% low": np.percentile(gains, 2.5), "95% high": np.percentile(gains, 97.5),
                        "clear gain?": "YES" if np.percentile(gains, 2.5) > 0 else "not clear"})
    return pd.DataFrame(rec)


# ------------------------------------------------------------------- Test A
def test_a(detail: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    a = detail[(detail["horizon"] == 1) & (detail["origin"] >= TEST_A_FIRST_ORIGIN) & (detail["origin"] <= TEST_A_LAST_ORIGIN)].copy()
    a["month"] = a["target_month"].dt.strftime("%Y-%m")
    met = metrics_table(a, ["milk_type"])
    agg = aggregate_mae(a, ["milk_type"])
    by_q = metrics_table(a, ["milk_type", "fiscal_quarter"])
    best = {mt: g[g["method"].isin(LEARNED)].sort_values("MAE").iloc[0]["method"] for mt, g in met.groupby("milk_type")}
    ctl = []
    for mt, g in by_q.groupby("milk_type"):
        for m in dict.fromkeys([best[mt], "Naive"]):
            for _, r in g[g["method"] == m].iterrows():
                ctl.append({"milk_type": mt, "method": m, "fiscal_quarter": r["fiscal_quarter"], "n": r["n"], "MAE": r["MAE"],
                            "target": CBE_TARGET, "status": "IN CONTROL" if r["MAE"] <= CBE_TARGET else "OVER TARGET"})
    return {
        "A_Metrics_2025": met,
        "A_Aggregate_Level": agg,
        "A_By_District": pivot(metrics_table(a, ["milk_type", s1.DISTRICT]), ["milk_type", s1.DISTRICT]),
        "A_By_Month": pivot(metrics_table(a, ["milk_type", "month"]), ["milk_type", "month"]),
        "A_By_Quarter": pivot(by_q, ["milk_type", "fiscal_quarter"]),
        "A_Control_Check": pd.DataFrame(ctl),
        "A_Bootstrap": bootstrap_gain(a, LEARNED),
        "A_Forecast_vs_Actual": a.drop(columns=["month"]).assign(origin=a["origin"].dt.date, target_month=a["target_month"].dt.date),
    }


# ------------------------------------------------------------------- Test B
def test_b(detail: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    rec, rows = [], []
    for label, origin in (("2026: train to Mar-26, forecast Apr-Jun 2026", TEST_B_ORIGIN),
                          ("2025: train to Mar-25, forecast Apr-Jun 2025", TEST_B_PRIOR_ORIGIN)):
        b = detail[(detail["origin"] == origin) & (detail["horizon"] <= 3)].copy()
        b["window"] = label
        rows.append(b)
        for mt, g in b.groupby("milk_type"):
            apr = g[g["horizon"] == 1]
            for m in [x for x in ALL_METHODS if x in g.columns]:
                e_apr = (apr[m] - apr["rate_actual"]).abs().mean()
                e_q1 = (g[m] - g["rate_actual"]).abs().mean()
                a_apr = aggregate_mae(apr, ["milk_type"], [m])["Aggregate MAE"].iloc[0]
                a_q1 = aggregate_mae(g, ["milk_type"], [m])["Aggregate MAE"].iloc[0]
                rec.append({"window": label, "milk_type": mt, "method": m,
                            "n districts": len(apr), "April MAE (h1)": e_apr, "CBE target": CBE_TARGET,
                            "CBE check": "MEETS" if e_apr <= CBE_TARGET else "OVER",
                            "Q1 MAE (Apr-Jun)": e_q1, "Q1 target": Q1_TARGET,
                            "Q1 check": "MEETS" if e_q1 <= Q1_TARGET else "OVER",
                            "Agg April MAE": a_apr, "Agg Q1 MAE": a_q1})
    summ = pd.DataFrame(rec)
    fva = pd.concat(rows).assign(origin=lambda x: x["origin"].dt.date, target_month=lambda x: x["target_month"].dt.date)
    return {"B_Summary": summ, "B_Forecast_vs_Actual": fva}


# ---------------------------------------------------------- method comparison
def comparison(detail: pd.DataFrame, last_actual: pd.Timestamp, cfg: s1.Config):
    """Like-for-like: for h=1..K only origins whose month t+K already has an actual are scored."""
    d4 = detail[(detail["horizon"] <= 4) & (detail["origin"] <= last_actual - pd.DateOffset(months=4))]
    d6 = detail[detail["origin"] <= last_actual - pd.DateOffset(months=6)]
    h1_4 = metrics_table(d4, ["milk_type"])
    by_h = pivot(metrics_table(d6, ["milk_type", "horizon"]), ["milk_type", "horizon"])
    sheets = {"C_Method_h1-4": h1_4, "C_MAE_by_Horizon_h1-6": by_h, "C_Bootstrap_h1-4": bootstrap_gain(d4, LEARNED)}
    best = {mt: g[g["method"].isin(LEARNED)].sort_values("MAE").iloc[0]["method"] for mt, g in h1_4.groupby("milk_type")}
    bands: Dict = {}
    for (mt, h), g in detail.groupby(["milk_type", "horizon"]):
        for m in LEARNED:
            e = (g["rate_actual"] - g[m]).dropna().to_numpy()
            if len(e) >= 5:
                bands.setdefault(mt, {}).setdefault(m, {})[str(int(h))] = [float(np.percentile(e, 10)), float(np.percentile(e, 90))]
    return sheets, best, bands


# --------------------------------------------------------------- excel out
def write_excel(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter
    good, bad = {"IN CONTROL", "MEETS", "YES"}, {"OVER TARGET", "OVER"}
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            for c in ws[1]:
                c.font, c.fill = Font(name="Arial", bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1F4E78")
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    c.font = Font(name="Arial")
                    if isinstance(c.value, float):
                        c.number_format = "0.00"
                    if c.value in good:
                        c.fill = PatternFill("solid", fgColor="C6EFCE")
                    if c.value in bad:
                        c.fill = PatternFill("solid", fgColor="FFC7CE")
            for i, col in enumerate(df.columns, 1):
                ws.column_dimensions[get_column_letter(i)].width = max(12, min(30, len(str(col)) + 3))
            ws.freeze_panes = "A2"


def backtest_main(cfg: s1.Config, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if not XGB_IS_REAL:
        log.warning("xgboost NOT installed: a stand-in boosted-tree model is used. XGB / Blend numbers are INDICATIVE only.")
    raw = s1.load_raw(cfg)
    panels, market = s1.build_panels(raw, cfg)
    last_actual = raw.loc[raw["row_valid"], s1.DATE].max()

    log.info("Input lists: %s", {k: len(v) for k, v in s1.FEATURES_BY_TYPE.items()} | {"pruned": s1.USING_SELECTED})
    parts = []
    for mt in s1.MILK_TYPES:
        spec = s1.spec_for(mt, forward=cfg.forward_inputs, flags=cfg.use_flags)
        table = s1.build_training_table(panels[mt], market, cfg)
        log.info("%s: %d inputs | recursive feature check, max abs gap vs step 1 = %.2e",
                 mt, len(spec["all"]), self_check(panels[mt], table, spec))
        recx = prepare_recx(panels[mt], market, table, cfg, spec)
        parts.append(run_backtest(mt, panels[mt], table, cfg, recx))
    detail = pd.concat(parts, ignore_index=True)

    A, B = test_a(detail), test_b(detail)
    C, best, bands = comparison(detail, last_actual, cfg)
    meta = {"best_method": best, "bands": bands, "xgb_is_real": XGB_IS_REAL}
    (outdir / "backtest_meta.json").write_text(json.dumps(meta, indent=2))
    detail.to_csv(outdir / "backtest_detail.csv", index=False)

    notes = pd.DataFrame({"Notes": [
        f"XGBoost is {'REAL' if XGB_IS_REAL else 'a STAND-IN (xgboost not installed) - XGB/Blend numbers indicative only'}.",
        f"Inputs used: COW {len(s1.FEATURES_BY_TYPE['COW'])} of {len(s1.FULL_FEATURES_BY_TYPE['COW'])}, BUFFALO {len(s1.FEATURES_BY_TYPE['BUFFALO'])} of {len(s1.FULL_FEATURES_BY_TYPE['BUFFALO'])} ({'PRUNED via DROP_BY_TYPE in step1_prepare_data.py: ' + str({k: v for k, v in s1.DROP_BY_TYPE.items() if v}) if s1.USING_SELECTED else 'full set'}). Full list: Feature_Dictionary.xlsx.",
        "Test A: origins Dec-24..Nov-25, h=1, targets Jan-25..Dec-25; model retrained every month on rows with target month <= origin.",
        "Test B: origin Mar-26, h=1..3 = Apr, May, Jun 2026. April (h1) vs 2.0; Apr-Jun average absolute error vs 5.0. Same window of 2025 shown for comparison. NOTE: this is ONE origin = ONE test month per horizon; use MultiHoldout_Report.xlsx (6 batches) as the evidence for the client.",
        "MultiHoldout_Report.xlsx: 6 forecast origins x N+1..N+6 and Q1 / Q2, MAE and volume-weighted MAE, compared with last-month repeat, rolling 3 / 6-month average and ARIMA.",
        "Method comparison: origins Dec-24..May-26, h=1..6. h1-4 summary uses only origins whose t+4 actual exists (like-for-like across horizons).",
        "Final method per milk type = lowest h1-4 MAE among the learned methods on the SAME data it is scored on, so it is slightly optimistic.",
        f"Settings of this run: forward_inputs={cfg.forward_inputs}, use_flags={cfg.use_flags}, training rows with target month >= {cfg.train_from_target}.",
        "Direct: inputs for a forecast of month t+h: weather and fat/SNF are for the TARGET month (actual up to the origin, same-calendar-month average of EARLIER years after it; derived 3/6-month rain, temperature, THI, fat/SNF trends recomputed from that series); supply / incentive = values observed at origin t; market/ONI = PROJECTED to the target month (last value + k x average of last 6 monthly changes); price history = actuals up to t; intervention flags = 0 for every forecast made after the 2022-24 swing. Same recipe in training, backtest and final forecast. With forward_inputs=False the old 'observed at the origin' weather is used.",
        "RecursiveX = ONE one-step model fed back its own forecasts: price inputs recomputed from the extended rate history; forward weather / fat/SNF / flags of the target month read from the table row (origin, horizon k); supply / incentive taken 6 months before the month being forecast; market/ONI projected from the last value + average of the last 6 monthly changes.",
        "Districts: only those with a rate in each of the 6 months before the origin are forecast and scored -> samples are small (cow ~7 districts, buffalo ~6, only 2 with full history).",
        "Many inputs (market, ONI, THI ...) are the same for every district in a month, so ~66 months carry them. More inputs can overfit: judge only by the backtest.",
        "Directional accuracy = % of forecasts with the right up/down vs last actual (blank for naive). Volatility score = std(forecast change)/std(actual change).",
        "Trust score = 100 x (0.4 x min(1, 2.0/MAE) + 0.3 x skill vs naive (0-1) + 0.3 x direction score (accuracy 50%->0, 100%->1)). Definitions are ours, to be confirmed with the client.",
        "Aggregate level = volume-weighted all-district rate per target month; the client's 2.37 may be measured there.",
        "Expected weather = average of the same calendar month in earlier years (a district without an earlier year uses the other districts' average); it is an estimate, see step1_data_features.py (part 2) for its error band.",
        "Bootstrap resamples origin months; 'clear gain' means the 95% interval of the gain over naive is above zero.",
    ]})
    write_excel(outdir / "Backtest_Results.xlsx", {**B, **A, **C, "Notes": notes})
    holdout_main(detail, last_actual, outdir)          # PART D: 6 holdout batches x N+1..N+6, Q1, Q2 (MAE, weighted MAE, baselines)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = ["Naive"] + LEARNED
    print("\n=== TEST A: 2025, 1 month ahead, district level ===")
    print(A["A_Metrics_2025"][lambda x: x["method"].isin(show)].round(2).to_string(index=False))
    print("\nAggregate level"); print(A["A_Aggregate_Level"][lambda x: x["method"].isin(show)].round(2).to_string(index=False))
    print("\n=== TEST B: train to Mar-26 ===")
    cols = ["window", "milk_type", "method", "n districts", "April MAE (h1)", "CBE check", "Q1 MAE (Apr-Jun)", "Q1 check", "Agg April MAE", "Agg Q1 MAE"]
    print(B["B_Summary"][lambda x: x["method"].isin(show)][cols].round(2).to_string(index=False))
    print("\n=== METHOD COMPARISON h1-4 ===")
    print(C["C_Method_h1-4"][lambda x: x["method"].isin(show)][["milk_type", "method", "n", "MAE", "Bias (fcst-actual)"]].round(2).to_string(index=False))
    print("\nFinal method per milk type:", best)


# =====================================================================================================================
# PART B (was step2c_variant_comparison.py) - do the two changes (forward weather, intervention flags) really lower the error?
# =====================================================================================================================
FIRST_ORIGIN, HELD_OUT_ORIGIN, SELECTION_LAST_TARGET = "2024-12-01", "2025-12-01", "2025-12-01"
VARIANTS = {   # name -> (forward_inputs, use_flags, train_from_target)
    "V0 current": (False, False, None),
    "V0b current, rows 2022+": (False, False, "2022-01-01"),
    "V1 forward weather": (True, False, "2022-01-01"),
    "V2 flags only": (False, True, "2022-01-01"),
    "V3 forward + flags": (True, True, "2022-01-01"),
}
SHOW = ["Naive", "Direct Ridge", "Direct XGB", "Direct Blend", "RecursiveX Ridge", "RecursiveX XGB", "RecursiveX Blend"]


def run_variant(mt: str, panel, market, base: s1.Config, spec_args) -> pd.DataFrame:
    fwd, flg, start = spec_args
    cfg = replace(base, forward_inputs=fwd, use_flags=flg, train_from_target=start)
    table = s1.build_training_table(panel, market, cfg)
    recx = prepare_recx(panel, market, table, cfg, s1.spec_for(mt, forward=fwd, flags=flg))
    rows = []
    for t in pd.date_range(FIRST_ORIGIN, HELD_OUT_ORIGIN, freq="MS"):
        d, _ = forecast_origin(panel, table, t, 6, cfg, recx)
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d["milk_type"] = mt
    d["n_train_rows_known"] = int(table["target"].notna().sum())
    return d[d["rate_actual"].notna()].copy()


def mae_of(g: pd.DataFrame, m: str) -> float:
    return float((g[m] - g["rate_actual"]).abs().mean())


def summarise(d: pd.DataFrame, label: str) -> pd.DataFrame:
    """MAE per method: h=1, average of h=1..3, average of h=1..6 (each horizon weighted equally by row)."""
    rec = []
    for mt, g in d.groupby("milk_type"):
        for m in [x for x in SHOW if x in g.columns]:
            rec.append({"set": label, "milk_type": mt, "method": m, "n h1": int((g.horizon == 1).sum()),
                        "MAE h1": mae_of(g[g.horizon == 1], m), "MAE h1-3": mae_of(g[g.horizon <= 3], m), "MAE h1-6": mae_of(g, m),
                        "Bias h1-6": float((g[m] - g["rate_actual"]).mean()),
                        "Dir.acc % h1-6": np.nan if m == "Naive" else float((np.sign(g[m] - g["rate_t"]) == np.sign(g["rate_actual"] - g["rate_t"])).mean() * 100)})
    return pd.DataFrame(rec)


def held_out_table(d: pd.DataFrame) -> pd.DataFrame:
    h = d[d["origin"] == pd.Timestamp(HELD_OUT_ORIGIN)]
    rec = []
    for mt, g in h.groupby("milk_type"):
        for m in [x for x in SHOW if x in g.columns]:
            q1, q2 = g[g.horizon <= 3], g[g.horizon >= 4]
            agg = aggregate_mae(g, ["milk_type"], [m])["Aggregate MAE"].iloc[0]
            rec.append({"milk_type": mt, "method": m, "districts": g[s1.DISTRICT].nunique(),
                        **{f"h{k}": mae_of(g[g.horizon == k], m) for k in range(1, 7)},
                        "Q1 Jan-Mar": mae_of(q1, m), "Q2 Apr-Jun": mae_of(q2, m), "All 6": mae_of(g, m), "State-level All 6": agg,
                        "h1<=2.0": "MEETS" if mae_of(g[g.horizon == 1], m) <= CBE_TARGET else "OVER",
                        "Q1<=5.0": "MEETS" if mae_of(q1, m) <= Q1_TARGET else "OVER"})
    return pd.DataFrame(rec)


def variants_main(cfg: s1.Config, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if not XGB_IS_REAL:
        log.warning("xgboost NOT installed: a stand-in boosted-tree model is used. XGB / Blend numbers are INDICATIVE only.")
    raw = s1.load_raw(cfg)
    panels, market = s1.build_panels(raw, cfg)
    details = []
    for name, args in VARIANTS.items():
        for mt in s1.MILK_TYPES:
            log.info("running %s / %s", name, mt)
            d = run_variant(mt, panels[mt], market, cfg, args)
            d["variant"] = name
            details.append(d)
    detail = pd.concat(details, ignore_index=True)
    sel = detail[detail["target_month"] <= pd.Timestamp(SELECTION_LAST_TARGET)]
    held = detail[detail["origin"] == pd.Timestamp(HELD_OUT_ORIGIN)]

    sel_tab = pd.concat([summarise(g, "SELECTION").assign(variant=v) for v, g in sel.groupby("variant")])
    rows = detail.groupby(["variant", "milk_type"])["n_train_rows_known"].first().reset_index()
    held_tab = pd.concat([held_out_table(g).assign(variant=v) for v, g in held.groupby("variant")])
    by_h = pd.concat([summarise(g[g.horizon == h], f"SELECTION h{h}").assign(variant=v, horizon=h)
                      for v, g0 in sel.groupby("variant") for h, g in g0.groupby("horizon")])
    detail_out = detail.assign(origin=detail["origin"].dt.date, target_month=detail["target_month"].dt.date)

    with pd.ExcelWriter(outdir / "Variant_Comparison.xlsx", engine="openpyxl") as xw:
        sel_tab.to_excel(xw, sheet_name="Selection_2025", index=False)
        held_tab.to_excel(xw, sheet_name="HeldOut_Jan-Jun2026", index=False)
        by_h.to_excel(xw, sheet_name="Selection_by_horizon", index=False)
        rows.to_excel(xw, sheet_name="Training_rows", index=False)
        detail_out.to_excel(xw, sheet_name="Detail", index=False)
    detail_out.to_csv(outdir / "variant_detail.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    key = ["Naive", "Direct XGB", "Direct Blend", "RecursiveX XGB", "RecursiveX Blend"]
    print("\nTRAINING ROWS with a known target (all horizons):\n", rows.pivot(index="variant", columns="milk_type", values="n_train_rows_known").to_string())
    for mt in s1.MILK_TYPES:
        print(f"\n=== {mt}: SELECTION set (origins Dec-24..Nov-25, targets <= Dec-25) - MAE Rs/kg TS ===")
        t = sel_tab[(sel_tab.milk_type == mt) & sel_tab.method.isin(key)]
        print(t.pivot(index="variant", columns="method", values="MAE h1-6")[key].round(2).to_string(), "   <- h1-6")
        print(t.pivot(index="variant", columns="method", values="MAE h1")[key].round(2).to_string(), "   <- h1")
        print(f"\n=== {mt}: HELD-OUT origin Dec-25, Jan-Jun 2026 ===")
        t = held_tab[(held_tab.milk_type == mt) & held_tab.method.isin(key)]
        print(t[["variant", "method", "districts", "h1", "Q1 Jan-Mar", "Q2 Apr-Jun", "All 6", "State-level All 6", "h1<=2.0", "Q1<=5.0"]].round(2).to_string(index=False))
    print("\nDECISION RULE: keep a change only if it lowers the SELECTION MAE (h1-6 and h1) for most methods; the held-out table is for reporting only.")
    print("saved ->", outdir / "Variant_Comparison.xlsx")


# =====================================================================================================================
# PART C (was stepB_intervention_report.py + helpers moved from step3) - TASK 2: intervention flags
# =====================================================================================================================
# ------------------------------------------------------------- equations
def ridge_equation(ridge, feats) -> Tuple[float, pd.Series, float]:
    """Pipeline (median impute -> scale -> ridge) as raw-unit coefficients: change = b0 + sum b_j * x_j."""
    sc, rg = ridge.steps[1][1], ridge.steps[2][1]
    coef = rg.coef_ / sc.scale_
    b0 = rg.intercept_ - np.sum(rg.coef_ * sc.mean_ / sc.scale_)
    return float(b0), pd.Series(coef, index=feats), float(rg.alpha_)


def equation_frame(mt: str, label: str, ridge, feats, horizon) -> pd.DataFrame:
    b0, coef, alpha = ridge_equation(ridge, feats)
    return pd.DataFrame({"Milk Type": mt.title(), "Model": label, "Horizon": horizon,
                         "Term": ["Intercept"] + list(feats), "Coefficient": [b0] + coef.tolist(),
                         "Median used if missing": [np.nan] + ridge.steps[0][1].statistics_.tolist(), "Ridge alpha": alpha})


# ------------------------------------------------------------- importance
_SHAP_WARNED = False


def shap_importance(xgb, X: pd.DataFrame) -> pd.Series:
    global _SHAP_WARNED
    try:
        import shap
        return pd.Series(np.abs(shap.TreeExplainer(xgb).shap_values(X)).mean(axis=0), index=X.columns)
    except Exception as exc:
        if not _SHAP_WARNED:
            log.warning("SHAP unavailable (%s); using model feature importance instead (pip install shap)", exc)
            _SHAP_WARNED = True
        return pd.Series(xgb.feature_importances_, index=X.columns)


def ridge_std_coef(ridge, feats) -> pd.Series:
    """Coefficient on the STANDARDISED input = change in forecast (Rs/kg TS) for a 1-standard-deviation rise of the input."""
    return pd.Series(ridge.steps[2][1].coef_, index=feats)


def pct(s: pd.Series) -> pd.Series:
    t = s.abs().sum()
    return s.abs() / t * 100 if t > 0 else s * 0.0


B_SELECTION_LAST_TARGET = pd.Timestamp("2025-12-01")
PAIRS = [("V1 forward weather", "V3 forward + flags", "flags added to the forward-weather model"),
         ("V0b current, rows 2022+", "V2 flags only", "flags added to the current (observed-at-origin) model")]


def flag_effects(mt: str, panel: pd.DataFrame, market: pd.DataFrame, cfg: s1.Config, origin: pd.Timestamp) -> pd.DataFrame:
    """Fit the Direct models (one per horizon) at ``origin`` and measure what they do with each flag."""
    feats = s1.spec_for(mt, forward=cfg.forward_inputs, flags=True)["all"]
    table = s1.build_training_table(panel, market, cfg)
    rows = []
    for h in range(1, cfg.max_horizon + 1):
        th = table[table["horizon"] == h]
        tr = th[(th["target_month"] <= origin) & th["target"].notna()]
        if len(tr) < MIN_TRAIN_ROWS:
            continue
        ridge, xgb = fit_pair(tr, feats)
        _, coef, _ = ridge_equation(ridge, feats)
        imp = pct(shap_importance(xgb, tr[feats]))
        rank = imp.rank(ascending=False, method="min")
        for f in s1.FLAG_FEATURES:
            X1, X0 = tr[feats].copy(), tr[feats].copy()
            X1[f], X0[f] = 1.0, 0.0
            rows.append({"Milk Type": mt.title(), "Horizon": h, "Flag": f, "Training rows with flag > 0": int((tr[f] > 0).sum()), "Training rows": len(tr),
                         "Ridge: effect of flag 0 -> 1 (Rs/kg TS)": float(coef[f]),
                         "XGB: avg effect of flag 0 -> 1 (Rs/kg TS)": float(np.mean(xgb.predict(X1) - xgb.predict(X0))),
                         "XGB importance share %": float(imp[f]), "XGB importance rank": int(rank[f]), "Inputs": len(feats)})
    return pd.DataFrame(rows)


def paired_evidence(detail: pd.DataFrame) -> pd.DataFrame:
    """MAE without vs with the flags, same model / origins / districts (selection set)."""
    d = detail[pd.to_datetime(detail["target_month"]) <= B_SELECTION_LAST_TARGET]
    methods = [m for m in ["Direct Ridge", "Direct XGB", "Direct Blend", "RecursiveX Ridge", "RecursiveX XGB", "RecursiveX Blend"] if m in d.columns]
    rec = []
    for mt, g in d.groupby("milk_type"):
        for a, b, label in PAIRS:
            ga, gb = g[g["variant"] == a], g[g["variant"] == b]
            if ga.empty or gb.empty:
                continue
            for m in methods:
                for scope, sa, sb in (("h1", ga[ga.horizon == 1], gb[gb.horizon == 1]), ("h1-6", ga, gb)):
                    ma, mb = (sa[m] - sa["rate_actual"]).abs().mean(), (sb[m] - sb["rate_actual"]).abs().mean()
                    rec.append({"Milk Type": mt.title(), "Comparison": label, "Method": m, "Scope": scope, "MAE without flags": ma, "MAE with flags": mb,
                                "Gain (Rs/kg TS, + = flags help)": ma - mb, "Flags help?": "YES" if mb < ma else "NO", "n": len(sb)})
    return pd.DataFrame(rec)


def reading(ev: pd.DataFrame, eff: pd.DataFrame, win: pd.DataFrame) -> pd.DataFrame:
    lines = ["WHAT THE FLAGS ARE: run_up_share / crash_share = share of the forecast months inside the 2022-24 run-up / crash. They are 0 for every forecast made after the swing.",
             "WHY: without them the model reads the swing as normal behaviour and carries it into future forecasts (the client's 'spurious outcome').",
             "DATES: found from the data (peak and trough of the reference districts). The business must validate the dates and give the reason; override with --run-up / --crash."]
    for _, r in win.iterrows():
        lines.append(f"  {r['Milk Type']} {r['Window']}: {r['Start']} to {r['End']} ({r['Months']} months), reference rate {r['Move (Rs/kg TS)']:+.1f} Rs/kg TS ({r['Move %']:+.0f}%).")
    if len(ev):
        for (mt, comp), g in ev[ev["Scope"] == "h1-6"].groupby(["Milk Type", "Comparison"]):
            lines.append(f"EVIDENCE {mt}, {comp}: flags lower the h1-6 error for {int((g['Flags help?'] == 'YES').sum())} of {len(g)} methods "
                         f"(average gain {g['Gain (Rs/kg TS, + = flags help)'].mean():+.2f} Rs/kg TS).")
        lines.append("CAUTION: 12 origins, a handful of districts, one swing - the evidence is thin. The flags are kept because they are the structurally correct treatment, "
                     "and they helped in most method / milk-type combinations, not because the test proves a large gain.")
    if len(eff):
        c = eff[eff["Flag"] == "crash_share"].groupby("Milk Type")[["Ridge: effect of flag 0 -> 1 (Rs/kg TS)", "XGB: avg effect of flag 0 -> 1 (Rs/kg TS)"]].mean()
        for mt, r in c.iterrows():
            lines.append(f"MODEL READING {mt}: with every forecast month inside the crash, the models lower the forecast change by about "
                         f"{r.iloc[0]:+.1f} (Ridge) / {r.iloc[1]:+.1f} (XGBoost) Rs/kg TS on average - that part of the history is explained by the intervention, not by weather / market.")
    lines.append("LIMIT: after the swing the flags are 0, so they cannot move a Jan-Jun 2026 forecast by themselves; they work by cleaning what the model learns from 2022-24.")
    return pd.DataFrame({"Reading": lines})


def intervention_main(cfg: s1.Config, outdir: Path, origin: str | None, variants: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    raw = s1.load_raw(cfg)
    panels, market = s1.build_panels(raw, cfg)
    org = pd.Timestamp(origin) if origin else raw.loc[raw["row_valid"], s1.DATE].max()
    win = s1.intervention_table(panels, cfg)
    eff = pd.concat([flag_effects(mt, panels[mt], market, cfg, org) for mt in s1.MILK_TYPES], ignore_index=True)
    ev = paired_evidence(pd.read_csv(variants)) if variants.exists() else pd.DataFrame()
    if ev.empty:
        log.warning("%s not found: run step2_backtest.py --variants to get the 'do flags help' evidence", variants)
    summ = reading(ev, eff, win)
    sheets = {"Windows": win, "Flag_Effect": eff}
    if len(ev):
        sheets["Flags_Help_Backtest"] = ev
    sheets["Reading"] = summ
    write_excel(outdir / "Intervention_Report.xlsx", sheets)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 220)
    print("\n1. WINDOWS\n", win.drop(columns=["Reference districts"]).round(1).to_string(index=False))
    print(f"\n2. FLAG EFFECT (models fitted at origin {org:%b %Y}; effect of the flag going 0 -> 1 on the forecast change, Rs/kg TS)")
    print(eff.round(2).to_string(index=False))
    if len(ev):
        print("\n3. DO FLAGS HELP? MAE h1-6, selection set (target <= Dec-25)")
        print(ev[ev["Scope"] == "h1-6"].round(2).to_string(index=False))
    print("\n4. READING"); [print(" -", x) for x in summ["Reading"]]
    print("\nsaved ->", outdir / "Intervention_Report.xlsx")


# =====================================================================================================================
# PART D (new) - MULTI-HOLDOUT REPORT: 6 batches (forecast origins) x N+1..N+6 and Q1 / Q2, MAE and volume-weighted MAE, vs simple baselines
# =====================================================================================================================
# WHY: one origin = ONE test point per horizon (N+1 is a single month).  The client / expert needs the error of N+1, N+2 ... N+6 and of the
# two quarters over several different starting months, so that one lucky or unlucky month cannot decide.  Nothing is refitted differently:
# the walk-forward engine (forecast_origin) already retrains at every origin using only rows whose target month <= origin, so the batches are
# read from the same ``detail`` table that Test A / B / C use.
# HOLDOUT_METHODS = ["Naive", "Rolling 3M avg", "Rolling 6M avg", "ARIMA(1,1,0)", "Seasonal-trend naive", "Drift (6m avg)"] + LEARNED     # (original)
HOLDOUT_METHODS = ["Naive", "Rolling 3M avg", "Rolling 6M avg", "ARIMA(1,1,0)", "Seasonal-trend naive", "Drift (6m avg)"] + LEARNED + cow_model.COW_COLUMNS   # NEW: + cow-only columns
HOLDOUT_LABEL = {"Naive": "Last month repeated", "Rolling 3M avg": "Rolling 3-month average", "Rolling 6M avg": "Rolling 6-month average"}
MODELS_HO = [f"N+{h}" for h in range(1, HOLDOUT_H + 1)] + list(HOLDOUT_QUARTERS)


def holdout_origins(last_actual: pd.Timestamp, n: int = HOLDOUT_N, step: int = HOLDOUT_STEP, last: str | None = None) -> List[pd.Timestamp]:
    """n forecast origins ``step`` months apart.  The last one is ``last`` (default: the latest origin that still has HOLDOUT_H actual months after it)."""
    latest = last_actual - pd.DateOffset(months=HOLDOUT_H)
    end = pd.Timestamp(last) if last else latest
    if end > latest:
        raise ValueError(f"holdout origin {end:%b %Y} has fewer than {HOLDOUT_H} actual months after it (last actual {last_actual:%b %Y})")
    origins = [end - pd.DateOffset(months=step * i) for i in range(n - 1, -1, -1)]
    if origins[0] < pd.Timestamp(COMPARE_FIRST_ORIGIN):
        raise ValueError(f"first holdout origin {origins[0]:%b %Y} is before the first backtest origin {COMPARE_FIRST_ORIGIN}: use a smaller --holdout-n / --holdout-step")
    return origins


def _stats(g: pd.DataFrame, m: str) -> Dict[str, float]:
    g = g[g[m].notna() & g["rate_actual"].notna()]
    if g.empty:
        return {"n": 0, "MAE": np.nan, "VW-MAE": np.nan, "Aggregate MAE": np.nan, "MAPE %": np.nan, "Bias (fcst-actual)": np.nan}
    e = g[m] - g["rate_actual"]
    w = g["weight_ts_kg"].fillna(0.0)
    vw = float((e.abs() * w).sum() / w.sum()) if w.sum() > 0 else np.nan
    s = g.assign(_w=w, _a=g["rate_actual"] * w, _f=g[m] * w).groupby("origin")[["_w", "_a", "_f"]].sum()
    s = s[s["_w"] > 0]
    agg = float((s["_f"] / s["_w"] - s["_a"] / s["_w"]).abs().mean()) if len(s) else np.nan   # error of the volume-weighted all-district rate
    return {"n": len(g), "MAE": float(e.abs().mean()), "VW-MAE": vw, "Aggregate MAE": agg,
            "MAPE %": float((e.abs() / g["rate_actual"]).mean() * 100), "Bias (fcst-actual)": float(e.mean())}


def _quarter_frame(d: pd.DataFrame, hs: Tuple[int, ...], methods: List[str]) -> pd.DataFrame:
    """One row per (district, origin): average forecast and average actual of the quarter's months (only when ALL months are present)."""
    g = d[d["horizon"].isin(hs)]
    key = [s1.DISTRICT, "origin"]
    g = g[g.groupby(key)["horizon"].transform("size") == len(hs)]
    if g.empty:
        return pd.DataFrame()
    q = g.groupby(key).agg(rate_t=("rate_t", "first"), rate_actual=("rate_actual", "mean"), weight_ts_kg=("weight_ts_kg", "mean")).reset_index()
    idx = pd.MultiIndex.from_frame(q[key])
    for m in methods:
        q[m] = g.groupby(key)[m].apply(lambda s: s.mean() if s.notna().all() else np.nan).reindex(idx).to_numpy()
    return q


def holdout_sheets(detail: pd.DataFrame, origins: List[pd.Timestamp], headline: str = HEADLINE_METHOD) -> Dict[str, pd.DataFrame]:
    """All sheets of the multi-holdout report.  ``detail`` = the walk-forward detail table (one row per milk type, district, origin, horizon)."""
    origins = [pd.Timestamp(o) for o in origins]
    d = detail[detail["origin"].isin(origins) & (detail["horizon"] <= HOLDOUT_H) & detail["rate_actual"].notna()].copy()
    methods = [m for m in HOLDOUT_METHODS if m in d.columns]
    if headline not in d.columns:
        raise ValueError(f"headline method '{headline}' not in the detail table: {methods}")
    batch_no = {o: i + 1 for i, o in enumerate(origins)}

    # ---- batches
    brows = []
    for o in origins:
        brows.append({"Batch": batch_no[o], "Trained on data up to": o.date(),
                      **{f"N+{h}": f"{o + pd.DateOffset(months=h):%b-%Y}" for h in range(1, HOLDOUT_H + 1)},
                      "Q1 (N+1..N+3)": f"{o + pd.DateOffset(months=1):%b}-{o + pd.DateOffset(months=3):%b %Y}",
                      "Q2 (N+4..N+6)": f"{o + pd.DateOffset(months=4):%b}-{o + pd.DateOffset(months=6):%b %Y}"})
    batches = pd.DataFrame(brows)

    long_rows, per_batch, fva = [], [], []
    for mt, dm in d.groupby("milk_type"):
        hl = HEADLINE_BY_TYPE.get(str(mt).upper(), headline)                    # NEW: headline method of THIS milk type (cow = Cow Ensemble v2, buffalo = unchanged)
        methods_mt = [m for m in methods if dm[m].notna().any()]               # NEW: a method that exists for the other milk type only is skipped (no empty rows)
        # qf = {q: _quarter_frame(dm, hs, methods) for q, hs in HOLDOUT_QUARTERS.items()}     # (original)
        qf = {q: _quarter_frame(dm, hs, methods_mt) for q, hs in HOLDOUT_QUARTERS.items()}
        # for m in methods:                                                                  # (original)
        for m in methods_mt:
            for h in range(1, HOLDOUT_H + 1):
                g = dm[dm["horizon"] == h]
                st = _stats(g, m)
                tgt = CBE_TARGET if h == 1 else np.nan
                long_rows.append({"Milk Type": mt.title(), "Method": m, "Model": f"N+{h}", "Measure": f"error of month t+{h}", "Batches": g["origin"].nunique(),
                                  **st, "Target": tgt})
            for q, hs in HOLDOUT_QUARTERS.items():
                tgt = HOLDOUT_QUARTER_TARGET[q]
                if len(qf[q]):
                    st = _stats(qf[q], m)
                    long_rows.append({"Milk Type": mt.title(), "Method": m, "Model": q, "Measure": "error of the quarter AVERAGE", "Batches": qf[q]["origin"].nunique(), **st, "Target": tgt})
                g = dm[dm["horizon"].isin(hs)]
                long_rows.append({"Milk Type": mt.title(), "Method": m, "Model": q, "Measure": "average MONTHLY error inside the quarter (step 2 definition)",
                                  "Batches": g["origin"].nunique(), **_stats(g, m), "Target": tgt})
        # ---- per batch (district-average MAE of each batch) for the headline model and the two simplest baselines
        # for m in dict.fromkeys([headline, "Naive", "Rolling 3M avg"]):                       # (original)
        for m in dict.fromkeys([hl, "Naive", "Rolling 3M avg"]):
            for o in origins:
                go = dm[dm["origin"] == o]
                row = {"Milk Type": mt.title(), "Method": m, "Batch": batch_no[o], "Origin": o.date()}
                for h in range(1, HOLDOUT_H + 1):
                    row[f"N+{h} MAE"] = _stats(go[go["horizon"] == h], m)["MAE"]
                for q, hs in HOLDOUT_QUARTERS.items():
                    qo = qf[q][qf[q]["origin"] == o] if len(qf[q]) else pd.DataFrame()
                    row[f"{q} MAE (quarter avg)"] = _stats(qo, m)["MAE"] if len(qo) else np.nan
                per_batch.append(row)
        # ---- forecast vs actual of the headline model, with the two simple baselines side by side (the table the expert asked for)
        # x = dm[[s1.DISTRICT, "origin", "horizon", "target_month", "rate_t", "rate_actual", headline, "Rolling 3M avg", "Naive", "weight_ts_kg"]].copy()   # (original)
        x = dm[[s1.DISTRICT, "origin", "horizon", "target_month", "rate_t", "rate_actual", hl, "Rolling 3M avg", "Naive", "weight_ts_kg"]].copy()
        x.insert(0, "Milk Type", mt.title())
        x.insert(1, "Batch", x["origin"].map(batch_no))
        x["Abs error model"] = (x[hl] - x["rate_actual"]).abs()
        x["MAPE % model"] = x["Abs error model"] / x["rate_actual"] * 100
        x["Abs error rolling 3M"] = (x["Rolling 3M avg"] - x["rate_actual"]).abs()
        x["Abs error last month repeat"] = (x["Naive"] - x["rate_actual"]).abs()
        x = x.rename(columns={hl: f"Model forecast ({hl})", "rate_actual": "Actual", "rate_t": "Last actual at origin",
                              "Rolling 3M avg": "Last-3-month average", "Naive": "Last-month repeat", "weight_ts_kg": "TS kg (weight)"})
        x["origin"], x["target_month"] = x["origin"].dt.date, x["target_month"].dt.date
        fva.append(x.sort_values(["Batch", "horizon", s1.DISTRICT]))
    long = pd.DataFrame(long_rows)
    long["Check (district MAE)"] = np.where(long["Target"].isna(), "n/a", np.where(long["MAE"] <= long["Target"], "MEETS", "OVER"))
    long["Check (weighted MAE)"] = np.where(long["Target"].isna(), "n/a", np.where(long["VW-MAE"] <= long["Target"], "MEETS", "OVER"))
    order = {m: i for i, m in enumerate(MODELS_HO)}
    morder = {m: i for i, m in enumerate(methods)}
    long["_o"], long["_m"] = long["Model"].map(order), long["Method"].map(morder)
    long = long.sort_values(["Milk Type", "_o", "Measure", "_m"]).drop(columns=["_o", "_m"]).reset_index(drop=True)

    # ---- the client table: our model vs the baselines, MAE and weighted MAE, N+1..N+6, Q1, Q2 (quarter = average of its months)
    def wide(value: str, measure: str) -> pd.DataFrame:
        sub = long[(long["Measure"].isin([f"error of month t+{h}" for h in range(1, HOLDOUT_H + 1)] + [measure]))]
        p = sub.pivot_table(index=["Milk Type", "Method"], columns="Model", values=value)
        p = p[[c for c in MODELS_HO if c in p.columns]]
        p = p.reset_index()
        p["_m"] = p["Method"].map(morder)
        return p.sort_values(["Milk Type", "_m"]).drop(columns="_m")
    qm = "error of the quarter AVERAGE"
    comp_mae, comp_vw = wide("MAE", qm), wide("VW-MAE", qm)
    comp_agg = wide("Aggregate MAE", qm)

    # ---- pass / fail of the headline model against the targets + gain over the simple baselines
    chk = []
    for mt in sorted(long["Milk Type"].unique()):
        for model in MODELS_HO:
            meas = f"error of month t+{model[2:]}" if model.startswith("N+") else qm
            g = long[(long["Milk Type"] == mt) & (long["Model"] == model) & (long["Measure"] == meas)].set_index("Method")
            hl = HEADLINE_BY_TYPE.get(str(mt).upper(), headline)                # NEW: headline of this milk type
            if hl not in g.index:
                continue
            h = g.loc[hl]
            pb = pd.DataFrame(per_batch)
            pb = pb[(pb["Milk Type"] == mt) & (pb["Method"] == hl)]
            col = f"{model} MAE" if model.startswith("N+") else f"{model} MAE (quarter avg)"
            k_ok = int((pb[col] <= h["Target"]).sum()) if h["Target"] == h["Target"] else np.nan
            simple = [m for m in ("Naive", "Rolling 3M avg", "Rolling 6M avg", "ARIMA(1,1,0)") if m in g.index]
            best_simple = min(simple, key=lambda m: g.loc[m, "MAE"])
            chk.append({"Milk Type": mt, "Model": model, "Our model": hl, "Batches": int(h["Batches"]), "Test points (district x batch)": int(h["n"]),
                        "MAE": h["MAE"], "Weighted MAE": h["VW-MAE"], "Aggregate MAE": h["Aggregate MAE"], "MAPE %": h["MAPE %"], "Target": h["Target"],
                        "Check (MAE)": "n/a" if h["Target"] != h["Target"] else "MEETS" if h["MAE"] <= h["Target"] else "OVER",
                        "Check (weighted MAE)": "n/a" if h["Target"] != h["Target"] else "MEETS" if h["VW-MAE"] <= h["Target"] else "OVER",
                        "Batches within target": k_ok, "Last month repeated MAE": g.loc["Naive", "MAE"], "Rolling 3M avg MAE": g.loc["Rolling 3M avg", "MAE"],
                        "Best simple method": best_simple, "Best simple MAE": g.loc[best_simple, "MAE"],
                        "Gain over best simple method %": (1 - h["MAE"] / g.loc[best_simple, "MAE"]) * 100})
    check = pd.DataFrame(chk)

    notes = pd.DataFrame({"Notes": [
        f"Batches = {len(origins)} forecast origins, {HOLDOUT_STEP} months apart: " + ", ".join(f"{o:%b-%y}" for o in origins) + ". Every batch is retrained on data known at its origin only (assert_no_leakage in the engine).",
        "Each batch gives N+1..N+6 (months t+1..t+6) and Q1 = average of N+1..N+3, Q2 = average of N+4..N+6.  With 6 batches every horizon has 6 test months per district instead of 1.",
        "MAE = plain average absolute error over all district x batch points.  Weighted MAE = the same, each point weighted by the district's total solids (TS kg) of the target month.  Aggregate MAE = error of the volume-weighted all-district rate per batch.",
        "Quarter error is shown two ways: (a) error of the quarter AVERAGE (forecast average of the 3 months vs actual average); (b) the average MONTHLY error inside the quarter.  Targets: N+1 <= 2.0, Q1 <= 5.0, Q2 <= 8.0 Rs/kg TS.",
        "Baselines: Last month repeated (Naive), Rolling 3 / 6-month average (flat forecast), ARIMA(1,1,0) with constant (conditional least squares), seasonal-trend naive, 6-month drift.  They use no inputs.",
        f"Our model in the client tables = {headline} for BUFFALO (fixed in advance; all 6 learned methods are listed in Comparison_MAE so nothing is hidden).",
        f"Our model for COW = {HEADLINE_BY_TYPE['COW']} (cow_model.py): equal-weight average of 2 HistGradientBoosting models (one recency-weighted) and a momentum Ridge, one model per horizon, "
        "lean input list without level / ONI inputs.  Cow members and the old cow methods (RecursiveX / Direct) are all listed in Comparison_MAE.",
        "The data are at DISTRICT level (not chilling-centre level), so 'weighted' = weighted over districts by TS kg.",
        "Only districts with a rate in each of the 6 months before the origin are forecast, so the sample is small; 6 batches are the minimum, more batches (--holdout-n 8 --holdout-step 1) are more reliable.",
        "Caution: the 2022-24 price swing sits inside the training years; batches before 2025 were not used here because their targets fall in/near that swing.",
    ]})
    return {"H_Notes": notes, "H_Batches": batches, "H_Target_Check": check, "Comparison_MAE": comp_mae, "Comparison_Weighted_MAE": comp_vw,
            "Comparison_Aggregate_MAE": comp_agg, "H_Per_Batch": pd.DataFrame(per_batch), "H_Long": long,
            "H_Forecast_vs_Actual": pd.concat(fva, ignore_index=True)}


def holdout_main(detail: pd.DataFrame, last_actual: pd.Timestamp, outdir: Path) -> Dict[str, pd.DataFrame]:
    origins = holdout_origins(last_actual, HOLDOUT_N, HOLDOUT_STEP, HOLDOUT_LAST)
    sheets = holdout_sheets(detail, origins, HEADLINE_METHOD)
    write_excel(outdir / "MultiHoldout_Report.xlsx", sheets)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(f"\n=== MULTI-HOLDOUT: {len(origins)} batches, origins " + ", ".join(f"{o:%b-%y}" for o in origins) + f" | our model = {HEADLINE_METHOD} (buffalo) / {HEADLINE_BY_TYPE['COW']} (cow) ===")
    print(sheets["H_Batches"].to_string(index=False))
    show = ["Milk Type", "Model", "Batches", "Test points (district x batch)", "MAE", "Weighted MAE", "Aggregate MAE", "Target", "Check (MAE)", "Check (weighted MAE)",
            "Batches within target", "Last month repeated MAE", "Rolling 3M avg MAE", "Gain over best simple method %"]
    print(sheets["H_Target_Check"][show].round(2).to_string(index=False))
    print("\nMAE by method (district level):\n", sheets["Comparison_MAE"].round(2).to_string(index=False))
    print("\nsaved ->", outdir / "MultiHoldout_Report.xlsx")
    return sheets


# =====================================================================================================================
# command line   (default = main backtest; extra switches run the other two parts)
# =====================================================================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Step 2: walk-forward backtest (+ variant comparison + intervention report)")
    ap.add_argument("--data", default=s1.Config.data_xlsx)
    ap.add_argument("--competitor", default=s1.Config.competitor_xlsx)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--no-forward", action="store_true", help="old behaviour: weather / fat / SNF observed at the origin")
    ap.add_argument("--no-flags", action="store_true", help="do not use the intervention flags as inputs")
    ap.add_argument("--keep-2021", action="store_true", help="keep training rows whose target month is before 2022")
    ap.add_argument("--variants", action="store_true", help="ALSO run the 5-variant comparison (old step2c; slow) -> Variant_Comparison.xlsx, variant_detail.csv")
    ap.add_argument("--intervention-report", action="store_true", help="ALSO write Intervention_Report.xlsx (old stepB; uses variant_detail.csv if present)")
    ap.add_argument("--report-origin", default=None, help="intervention report: fit the models at this origin month (default: last actual)")
    ap.add_argument("--holdout-n", type=int, default=HOLDOUT_N, help="number of holdout batches (forecast origins) for MultiHoldout_Report.xlsx (default 6)")
    ap.add_argument("--holdout-step", type=int, default=HOLDOUT_STEP, help="months between two batch origins (default 2 -> 6 batches cover one year)")
    ap.add_argument("--holdout-last", default=None, help="last batch origin, e.g. 2025-12-01 (default: latest origin that has 6 actual months after it)")
    ap.add_argument("--headline-method", default=HEADLINE_METHOD, choices=LEARNED, help="learned method shown as 'our model' in the holdout tables")
    ap.add_argument("--skip-backtest", action="store_true", help="do not rerun the main backtest (only the extra parts asked for)")
    s1.add_swing_args(ap)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = s1.config_from_args(a, competitor_xlsx=str(s1.resolve(a.competitor)))
    HOLDOUT_N, HOLDOUT_STEP, HOLDOUT_LAST, HEADLINE_METHOD = a.holdout_n, a.holdout_step, a.holdout_last, a.headline_method
    out = s1.resolve(a.outdir)
    if not a.skip_backtest:
        backtest_main(cfg, out)
    if a.variants:
        variants_main(cfg, out)
    if a.intervention_report:
        intervention_main(cfg, out, a.report_origin, out / "variant_detail.csv")

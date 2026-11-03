"""
cow_model.py - NEW separate COW forecasting method ("Cow Ensemble v2")        (BUFFALO IS NOT TOUCHED BY THIS FILE)
==================================================================================================================
Why a separate cow model
------------------------
The shared engine (Ridge + XGBoost, one-step RecursiveX) works for buffalo but not for cow:
  * cow rates are a staircase (flat for months, then a jump of 15-35 Rs), and in 2025-26 they rose steadily from ~231 to ~320;
  * a Ridge that has learned the 2023-24 crash mean-reverts, so the Blend forecasts sit 7-13 Rs BELOW the actual from N+2 onward;
  * level inputs (rate_t, ONI, ly_change, incentive %) let the model recognise "the period" rather than the dairy economics.
A correction of the bias from past errors was tested and REJECTED: the sign of the cow bias flips between regimes (+6 Rs in 2024, -8 Rs in 2025),
so a bias learned from the past makes the next regime worse.  The fix has to be in the model, not in a patch on its output.

What the new cow model is
-------------------------
Equal-weight average of the predicted CHANGE (target = Rate(t+h) - Rate(t), one model per horizon h, exactly the same training rows,
same leakage rule and same forward-weather / intervention-flag inputs as step 1) of THREE different learners:

  1. Cow HGB weighted    HistGradientBoosting on the LEAN input list, rows weighted by recency (half-life 12 months): the recent regime counts more.
  2. Cow HGB             the same learner, all rows weighted equally.
  3. Cow Ridge momentum  Ridge on 14 momentum / mean-reversion / season / flag inputs only (no weather, no level, no ONI).

  Cow Ensemble v2 = rate(t) + mean of the three predicted changes.

LEAN input list = the 44 cow inputs MINUS the level / regime proxies (rate_t, ly_change, oni_t, oni_l3, incentive_pct, temp_avg_t, temp_min_t, rain_t,
rain_6m, temp_max_3m, humid_t, thi_t, snf_ch3, feed_maize_ratio): fewer inputs, and none of them can act as a calendar stamp for the 2022-24 swing.
The intervention flags run_up_share / crash_share stay in (they clean the 2022-24 swing out of what the model learns; 0 when forecasting).

Rules kept: every training row has target month <= origin (assert_no_leakage); no setting is tuned on the test window; scikit-learn only
(no xgboost needed, deterministic, random_state fixed).  Weather scenarios (LOW / HIGH) are supported the same way as in step 2: the fitted models
are applied to the scenario inputs of the forecast rows and the columns are named "<method> | WEATHER LOW" / "<method> | WEATHER HIGH".

How it was chosen (honest note)
-------------------------------
About 20 variants were tried (single learners, 5 ensembles, shrink-to-naive, bias correction, recency weights).  They were compared on a DEV window
(origins Jan-24..Aug-24, targets before Mar-25 - the 2024 recovery, a different regime) and on the 6-batch holdout.  The ensemble above was the most
consistent across both; it is NOT the best single number on the holdout, so the holdout gain is only mildly optimistic.  The real test is the next months.
"""
from __future__ import annotations

import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

METHOD = "Cow Ensemble v2"                                    # the cow headline / default method (all other cow members are listed below)
MEMBERS = ["Cow HGB weighted", "Cow HGB", "Cow Ridge momentum"]
COW_COLUMNS = [METHOD] + MEMBERS                              # columns added to the step-2 detail table for COW only
MIN_TRAIN_ROWS = 25                                           # same as step 2
HALF_LIFE_MONTHS = 12.0                                       # recency weight: a row 12 months older counts half

LEAN: List[str] = ["diff_1", "diff_3", "diff_6", "diff_mean6", "diff_std6", "accel", "dev_ma3", "dev_ma6", "dev_ma12", "yoy", "range12_pos",
                   "rel_mean", "rel_ch3", "state_ch3", "tgt_month_sin", "tgt_month_cos", "tgt_summer", "smp_ch3", "smp_ch6", "ghee_ch3",
                   "feed_ch3", "maize_ch3", "soy_meal_ch3", "cpi_ch3", "fat_t", "snf_t", "temp_max_t", "rain_3m", "run_up_share", "crash_share"]
MOMENTUM: List[str] = ["diff_1", "diff_3", "diff_6", "diff_mean6", "accel", "state_ch3", "dev_ma3", "dev_ma6", "dev_ma12", "smp_ch3",
                       "tgt_month_sin", "tgt_month_cos", "run_up_share", "crash_share"]


def _hgb() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=42)


def _ridge():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=30.0))


def _recency_weights(target_month: pd.Series, t: pd.Timestamp) -> np.ndarray:
    age_months = (t - pd.to_datetime(target_month)).dt.days.to_numpy() / 30.4
    return np.exp(-np.log(2.0) * age_months / HALF_LIFE_MONTHS)


def cow_columns(table: pd.DataFrame, t, districts: List[str], H: int, scenarios: Dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Forecasts of the new cow model standing at origin ``t``.  One row per (district, horizon); columns = COW_COLUMNS
    (+ '<column> | <scenario>' for every weather scenario).  ``table`` = step-1 training table of COW.  Only rows whose
    target month is <= t are used for fitting (checked)."""
    t = pd.Timestamp(t)
    scenarios = scenarios or {}
    parts = []
    for h in range(1, H + 1):
        th = table[table["horizon"] == h]
        train = th[(th["target_month"] <= t) & th["target"].notna()]
        test = th[(th["origin"] == t) & th["district"].isin(districts)]
        if test.empty or len(train) < MIN_TRAIN_ROWS:
            continue
        assert train["target_month"].max() <= t, "leakage in cow_model"
        w = _recency_weights(train["target_month"], t)
        y = train["target"]
        m_w = _hgb().fit(train[LEAN], y, sample_weight=w)
        m_h = _hgb().fit(train[LEAN], y)
        m_r = _ridge().fit(train[MOMENTUM], y)

        def predict(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
            c = {MEMBERS[0]: m_w.predict(frame[LEAN]), MEMBERS[1]: m_h.predict(frame[LEAN]), MEMBERS[2]: m_r.predict(frame[MOMENTUM])}
            c[METHOD] = np.mean([c[k] for k in MEMBERS], axis=0)
            return c

        res = pd.DataFrame({"district": test["district"].to_numpy(), "horizon": h})
        base = test["rate_t"].to_numpy()
        for k, c in predict(test).items():
            res[k] = base + c
        for nm, tb in scenarios.items():                      # same fitted models, other forward-weather inputs of the forecast rows
            ts = tb[(tb["horizon"] == h) & (tb["origin"] == t)].set_index("district").reindex(test["district"])
            for k, c in predict(ts).items():
                res[f"{k} | {nm}"] = base + c
        parts.append(res)
    if not parts:
        return pd.DataFrame(columns=["district", "horizon"] + COW_COLUMNS)
    return pd.concat(parts, ignore_index=True)

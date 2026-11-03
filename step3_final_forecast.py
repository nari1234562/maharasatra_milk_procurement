"""
COW v2 (NEW): COW is forecast by its own method, "Cow Ensemble v2" (cow_model.py, must sit in the same folder).  BUFFALO and every shared
function are unchanged (cow_extra=False is the original behaviour); the replaced cow lines are kept as comments marked "(original)".
step3_final_forecast.py - FILE 3 of 4: the FINAL forecast (point + weather band + total band + explanation pack)
=====================================================================================================================
train Jan-2021 .. Jun-2026 (last actual), forecast Jul-Dec 2026   (PRUNED / FULL FEATURE SET)
=====================================================================================================================
* Origin = last actual month (Jun 2026).  COW and BUFFALO separately (input lists from step 1: pruned if selected_features.json exists),
  district level, Rs/kg TS.
* h = 1..4 (Jul-Oct 2026) = short-term CBE.  h = 5..6 (Nov-Dec 2026) = INDICATIVE (backtest error is much larger).
* Method for BOTH milk types = RecursiveX XGB (FINAL_METHOD below).  --auto-method goes back to "best method of the step-2
  comparison" (backtest_meta.json).  All methods are still written side by side so the choice can be checked.
* Forecast = last actual rate + predicted change.  The 80% band is the 10th-90th percentile of the BACKTEST
  errors (actual - forecast) of that method at that horizon.
* TASK 1 (weather band): the forecast is produced under three weather inputs - LOW / EXPECTED / HIGH (step1_data_features.py part 2:
  expected = same-month average of earlier years, low / high = expected + the seasonal 95% error band).  The models are fitted once;
  only the inputs of the forecast rows change.  Output columns: forecast under each weather, WEATHER BAND (min / max of the three)
  and TOTAL BAND (weather band widened by the 80% model-error band).  In held-out mode (--origin) the model-error band uses only
  backtest errors whose target month is <= origin (no look-ahead).
* TASK 2 (intervention): run_up_share / crash_share are model inputs (step 1); sheet Intervention_Windows lists the dates for business validation
  (override with --run-up / --crash).  step2_backtest.py --intervention-report quantifies what the model does with them.
* Explanation pack: Ridge equation (Direct, one per horizon), RecursiveX equation (one-step, applied month after month),
  XGBoost SHAP drivers, and FEATURE IMPORTANCE shown separately for COW and BUFFALO (sheets + PNG charts).

    python step3_final_forecast.py          (run step2_backtest.py first)
    python step3_final_forecast.py --auto-method      (use the backtest-best method per milk type instead)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import step1_data_features as s1
import step2_backtest as s2
import cow_model                                   # NEW: separate COW method (cow_model.py); buffalo never uses it

log = logging.getLogger("milk")

FINAL_METHOD = "RecursiveX XGB"      # BUFFALO method (unchanged).  (Original comment: one method for cow AND buffalo)
FINAL_METHOD_COW = cow_model.METHOD  # NEW: COW has its own method (cow_model.py)


# ------------------------------------------------------------- model-error bands
SCN_LOW, SCN_HIGH = "WEATHER LOW", "WEATHER HIGH"


def model_bands(detail: pd.DataFrame, origin: pd.Timestamp | None = None) -> Dict:
    """80% band of the BACKTEST errors (actual - forecast): {milk_type: {method: {horizon: [p10, p90, n]}}}.
    With ``origin`` only errors whose target month is <= origin are used, so a forecast made at ``origin`` never sees later actuals."""
    d = detail.copy()
    d["target_month"] = pd.to_datetime(d["target_month"])
    if origin is not None:
        d = d[d["target_month"] <= origin]
    out: Dict = {}
    for (mt, h), g in d.groupby(["milk_type", "horizon"]):
        # for m in s2.LEARNED:                                                           # (original)
        for m in s2.LEARNED + (cow_model.COW_COLUMNS if mt == "COW" else []):             # NEW: cow-only methods get their band for cow
            if m not in g.columns:
                continue
            e = (g["rate_actual"] - g[m]).dropna().to_numpy()
            if len(e) >= 5:
                out.setdefault(mt, {}).setdefault(m, {})[str(int(h))] = [float(np.percentile(e, 10)), float(np.percentile(e, 90)), len(e)]
    return out


# (ridge_equation, equation_frame, shap_importance, ridge_std_coef, pct now live in step2_backtest.py)


def feature_importance(mt: str, fits: Dict, feats: List[str], H: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Importance of every input for ONE milk type, from the two model families.

    RecursiveX (one-step model)  : Ridge |standardised coefficient|  and  XGBoost mean |SHAP|
    Direct (one model per horizon): the same two, averaged over h = 1..H
    Each is turned into a % share (sums to 100).  Overall = average of the four shares.
    Ridge shares show a LINEAR effect, XGBoost shares also catch non-linear / interaction effects.
    """
    r, x, tr = fits[("RecursiveX", 1)]
    rec_r_signed = s2.ridge_std_coef(r, feats)
    rec_r, rec_x = s2.pct(rec_r_signed), s2.pct(s2.shap_importance(x, tr[feats]))
    dr, dx, dsign = [], [], []
    for h in range(1, H + 1):
        rh, xh, th = fits[("Direct", h)]
        s = s2.ridge_std_coef(rh, feats)
        dsign.append(s)
        dr.append(s2.pct(s)), dx.append(s2.pct(s2.shap_importance(xh, th[feats])))
    dir_r, dir_x = sum(dr) / H, sum(dx) / H
    out = pd.DataFrame({
        "Feature": feats,
        "Group": [s1.FEATURE_GROUP[f] for f in feats],
        "How known at forecast time": [s1.FEATURE_KIND[f] for f in feats],
        "Recursive Ridge %": rec_r.to_numpy(), "Recursive XGB %": rec_x.to_numpy(),
        "Direct Ridge % (avg h1-6)": dir_r.to_numpy(), "Direct XGB % (avg h1-6)": dir_x.to_numpy(),
    })
    out["Overall importance %"] = out[["Recursive Ridge %", "Recursive XGB %", "Direct Ridge % (avg h1-6)", "Direct XGB % (avg h1-6)"]].mean(axis=1)
    sign = np.sign(rec_r_signed.to_numpy() + sum(dsign).to_numpy() / H)
    out["Effect on forecast (Ridge)"] = np.where(sign > 0, "rate rises when input rises", np.where(sign < 0, "rate falls when input rises", "-"))
    out["Meaning"] = [s1.FEATURE_INFO[f][2] for f in feats]
    out = out.sort_values("Overall importance %", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out) + 1))
    out.insert(0, "Milk Type", mt.title())
    grp = (out.groupby("Group")[["Overall importance %", "Recursive Ridge %", "Recursive XGB %",
                                 "Direct Ridge % (avg h1-6)", "Direct XGB % (avg h1-6)"]].sum()
           .join(out.groupby("Group")["Feature"].count().rename("Inputs")).sort_values("Overall importance %", ascending=False)
           .reset_index())
    grp.insert(0, "Milk Type", mt.title())
    return out, grp


def importance_chart(mt: str, imp: pd.DataFrame, grp: pd.DataFrame, path: Path, top: int = 20) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = list(dict.fromkeys(s1.FEATURE_GROUP.values()))
    cmap = {g: plt.cm.tab20(i) for i, g in enumerate(groups)}
    fig, ax = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [1.5, 1]})
    t = imp.head(top).iloc[::-1]
    ax[0].barh(t["Feature"], t["Overall importance %"], color=[cmap[g] for g in t["Group"]])
    ax[0].set_title(f"{mt.title()} - top {top} inputs of {len(imp)} (overall importance %)")
    ax[0].set_xlabel("share of total importance (%)")
    for i, v in enumerate(t["Overall importance %"]):
        ax[0].text(v + 0.05, i, f"{v:.1f}", va="center", fontsize=8)
    g = grp.iloc[::-1]
    ax[1].barh(g["Group"], g["Overall importance %"], color=[cmap[x] for x in g["Group"]])
    ax[1].set_title(f"{mt.title()} - importance by group (%)")
    for i, (v, n) in enumerate(zip(g["Overall importance %"], g["Inputs"])):
        ax[1].text(v + 0.2, i, f"{v:.1f}  ({int(n)} inputs)", va="center", fontsize=8)
    ax[1].set_xlim(0, g["Overall importance %"].max() * 1.35)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------- forecast
def final_for(mt: str, panel: pd.DataFrame, table: pd.DataFrame, cfg: s1.Config, meta: Dict, recx: Dict,
              origin_override: pd.Timestamp | None = None, scenarios: Dict[str, pd.DataFrame] | None = None):
    spec = recx["spec"]
    feats = spec["all"]
    origin = origin_override if origin_override is not None else panel["rate"].dropna().index.get_level_values(s1.DATE).max()
    districts = s1.forecastable_districts(panel, origin, cfg.forecast_min_months)
    method = meta["best_method"][mt]
    log.info("%s: origin %s | %d inputs | districts: %s | method: %s", mt, f"{origin:%b %Y}", len(feats), districts, method)
    # d, fits = s2.forecast_origin(panel, table, origin, cfg.max_horizon, cfg, recx, districts, keep_fits=True, scenarios=scenarios)   # (original)
    d, fits = s2.forecast_origin(panel, table, origin, cfg.max_horizon, cfg, recx, districts, keep_fits=True, scenarios=scenarios, cow_extra=(mt == "COW"))

    out = pd.DataFrame({
        "Milk Type": mt.title(), "District": d[s1.DISTRICT], "Horizon (months)": d["horizon"],
        "Forecast Month": d["target_month"].dt.date, "Last Actual (Rs/kg TS)": d["rate_t"], "Forecast (Rs/kg TS)": d[method]})
    lo_hi = [meta["bands"][mt][method].get(str(int(h)), [np.nan, np.nan, np.nan]) for h in d["horizon"]]
    lo = np.minimum([x[0] for x in lo_hi], 0.0)        # keep the point forecast inside its own band
    hi = np.maximum([x[1] for x in lo_hi], 0.0)
    out["Lower 80% (Rs/kg TS)"], out["Upper 80% (Rs/kg TS)"] = out["Forecast (Rs/kg TS)"] + lo, out["Forecast (Rs/kg TS)"] + hi
    out["Model-error band n"] = [x[2] if len(x) > 2 else np.nan for x in lo_hi]
    if scenarios:                                      # TASK 1: forecast under the LOW / HIGH weather band
        f0 = d[method].to_numpy()
        f_lo, f_hi = d[f"{method} | {SCN_LOW}"].to_numpy(), d[f"{method} | {SCN_HIGH}"].to_numpy()
        stack = np.vstack([f0, f_lo, f_hi])
        out["Forecast, LOW weather (Rs/kg TS)"], out["Forecast, HIGH weather (Rs/kg TS)"] = f_lo, f_hi
        out["Weather band low (Rs/kg TS)"], out["Weather band high (Rs/kg TS)"] = stack.min(axis=0), stack.max(axis=0)
        out["Weather band width (Rs/kg TS)"] = out["Weather band high (Rs/kg TS)"] - out["Weather band low (Rs/kg TS)"]
        out["TOTAL band low (weather + model error)"] = out["Weather band low (Rs/kg TS)"] + lo
        out["TOTAL band high (weather + model error)"] = out["Weather band high (Rs/kg TS)"] + hi
    out["Method"] = method
    if origin_override is not None:                       # held-out mode: the months after the origin are known -> score them
        out["Actual (Rs/kg TS)"] = d["rate_actual"].to_numpy()
        out["Abs error (Rs/kg TS)"] = (out["Forecast (Rs/kg TS)"] - out["Actual (Rs/kg TS)"]).abs()
    out["Status"] = np.where(d["horizon"] <= cfg.short_horizon, "Short-term (CBE)", "INDICATIVE - outside short-term horizon")
    for m in s2.ALL_METHODS + [c for c in cow_model.COW_COLUMNS if c in d.columns]:   # NEW: + the cow-model columns (cow only)
        out[m] = d[m].to_numpy()

    eq = [s2.equation_frame(mt, f"Direct Ridge ({len(feats)} inputs)", fits[("Direct", h)][0], feats, h) for h in range(1, cfg.max_horizon + 1)]
    rx = fits[("RecursiveX", 1)]
    rec_eq = s2.equation_frame(mt, f"RecursiveX one-step Ridge ({len(feats)} inputs)", rx[0], feats, 1)
    sh = [pd.DataFrame({"Milk Type": mt.title(), "Model": "Direct XGB", "Horizon": h, "Feature": feats,
                        "Mean |SHAP|": s2.shap_importance(fits[("Direct", h)][1], fits[("Direct", h)][2][feats]).to_numpy()})
          for h in range(1, cfg.max_horizon + 1)]
    sh.append(pd.DataFrame({"Milk Type": mt.title(), "Model": "RecursiveX one-step XGB", "Horizon": 1, "Feature": feats,
                            "Mean |SHAP|": s2.shap_importance(rx[1], rx[2][feats]).to_numpy()}))
    imp, grp = feature_importance(mt, fits, feats, cfg.max_horizon)
    return out, pd.concat(eq), rec_eq, pd.concat(sh), imp, grp


def main(cfg: s1.Config, outdir: Path, override: Dict[str, str] | None = None, origin: str | None = None,
         band_level: float = 95.0, weather_band: bool = True) -> None:
    origin_ts = pd.Timestamp(origin) if origin else None
    meta_path = outdir / "backtest_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("Run step2_backtest.py first (backtest_meta.json missing)")
    meta = json.loads(meta_path.read_text())
    override = {k: v for k, v in (override or {}).items() if v}
    for mt, m in override.items():
        valid = s2.LEARNED + (cow_model.COW_COLUMNS if mt == "COW" else [])                # NEW: cow may use the cow-model methods
        if m not in valid:
            raise ValueError(f"Unknown method '{m}'. Choose one of: {valid}")
        # if m not in meta["bands"].get(mt, {}):                                              # (original)
        if m not in meta["bands"].get(mt, {}) and not (m in cow_model.COW_COLUMNS and (outdir / "backtest_detail.csv").exists()):   # NEW: cow bands come from backtest_detail.csv
            raise ValueError(f"No backtest bands for {mt} / {m} - rerun step2")
        log.warning("%s: method FORCED to '%s' (backtest-best was '%s')", mt, m, meta["best_method"][mt])
        meta["best_method"][mt] = m
    if not s2.XGB_IS_REAL or not meta.get("xgb_is_real", True):
        log.warning("XGBoost is a STAND-IN in this run or in the backtest: XGB/Blend rows, importance and the method choice are INDICATIVE")

    detail_path = outdir / "backtest_detail.csv"
    if detail_path.exists():                                   # model-error band: only errors known at the origin (no look-ahead)
        mb = model_bands(pd.read_csv(detail_path), origin_ts)
        if all(mb.get(mt, {}).get(m) for mt in s1.MILK_TYPES for m in s2.LEARNED):
            meta["bands"] = mb
            log.info("model-error band from backtest errors%s", f" with target month <= {origin_ts:%b %Y}" if origin_ts is not None else "")
    elif origin_ts is not None:
        log.warning("backtest_detail.csv missing: the model-error band comes from backtest_meta.json and includes errors AFTER the origin")

    raw = s1.load_raw(cfg)
    panels, market = s1.build_panels(raw, cfg)
    origin_eff = origin_ts if origin_ts is not None else raw.loc[raw["row_valid"], s1.DATE].max()
    wb = None
    if weather_band and not cfg.forward_inputs:
        log.warning("weather band needs forward inputs (--no-forward given): no weather scenarios")
    elif weather_band:                                         # TASK 1: expected weather + seasonal band, from data <= origin
        wb = s1.weather_bands_at(raw, origin_eff, band_level, cfg.max_horizon)
        log.info("weather band: level %.0f%%, seasons: %s", band_level, ", ".join(sorted(set(s1.SEASON_OF_MONTH.values()))))
    fc, eq, req, sh, imps, grps = [], [], [], [], {}, {}
    for mt in s1.MILK_TYPES:
        table = s1.build_training_table(panels[mt], market, cfg)
        recx = s2.prepare_recx(panels[mt], market, table, cfg, s1.spec_for(mt, forward=cfg.forward_inputs, flags=cfg.use_flags))
        scn = None
        if wb is not None:
            p = panels[mt]
            cal = p.index.get_level_values(s1.DATE)
            ext = pd.date_range(cal.min(), cal.max() + pd.DateOffset(months=cfg.max_horizon), freq="MS")
            dists = sorted(p.index.get_level_values(s1.DISTRICT).unique())
            o = origin_eff if origin_ts is not None else p["rate"].dropna().index.get_level_values(s1.DATE).max()
            scn = {SCN_LOW: s1.scenario_table(p, cfg, table, o, s1.scenario_shifts(ext, dists, wb[0], "lo")),
                   SCN_HIGH: s1.scenario_table(p, cfg, table, o, s1.scenario_shifts(ext, dists, wb[0], "hi"))}
        a, b, c, e, imp, grp = final_for(mt, panels[mt], table, cfg, meta, recx, origin_ts, scn)
        fc.append(a), eq.append(b), req.append(c), sh.append(e)
        imps[mt], grps[mt] = imp, grp
        imp.to_csv(outdir / f"feature_importance_{mt.lower()}.csv", index=False)
        importance_chart(mt, imp, grp, outdir / f"feature_importance_{mt.lower()}.png")
    forecast = pd.concat(fc).sort_values(["Milk Type", "District", "Horizon (months)"]).reset_index(drop=True)
    cmp_cols = [f"{f} {k}" for k in ("Blend", "XGB", "Ridge") for f in ("Direct", "RecursiveX")]
    compare = forecast[["Milk Type", "District", "Horizon (months)", "Forecast Month", "Last Actual (Rs/kg TS)"] + cmp_cols]
    compare = compare.assign(**{"Direct minus RecursiveX (Blend)": compare["Direct Blend"] - compare["RecursiveX Blend"]})
    shap_df = pd.concat(sh)
    shap_top = (shap_df.groupby(["Milk Type", "Model", "Feature"])["Mean |SHAP|"].mean().reset_index()
                .sort_values(["Milk Type", "Model", "Mean |SHAP|"], ascending=[True, True, False]))
    shap_top["Rank"] = shap_top.groupby(["Milk Type", "Model"]).cumcount() + 1
    shap_top = shap_top[shap_top["Rank"] <= 15]

    howto = pd.DataFrame({"How to use the equations in Excel": [
        "Forecast(t+h) = Last actual rate(t) + Change, Change = Intercept + SUM(Coefficient x input).",
        f"DIRECT: use the rows of the matching milk type AND horizon (one equation per horizon; {len(s1.FEATURES_BY_TYPE['COW'])} inputs cow, {len(s1.FEATURES_BY_TYPE['BUFFALO'])} inputs buffalo).",
        "RECURSIVEX: ONE equation for the next month. Forecast month t+1, add it to the rate history as if it were an actual,",
        "  recompute the price inputs from the extended history, apply the same equation for t+2, and so on.",
        "Inputs are values at the last month in the history; tgt_month_sin/cos/summer/monsoon describe the month being forecast.",
        "Price inputs are defined in Feature_Dictionary.xlsx (diff_*, dev_ma*, yoy, range12_pos, rel_*, state_ch3, accel, diff_std6, ly_change).",
        "If an input is missing use the 'Median used if missing' value.",
        "Weather / humidity / THI / fat / SNF are values FOR THE TARGET MONTH: actual up to the last actual month, the average of the same calendar month",
        "  in earlier years after it; 3/6-month rain, 3-month max temperature, THI and fat/SNF trends are recomputed from that monthly series (step 1).",
        "  Supply and incentive are the values observed at the origin (RecursiveX: 6 months before the month being forecast).",
        "run_up_share / crash_share = intervention flags for the 2022-24 price swing; they are 0 for every forecast month (the swing is over).",
        "Market inputs are PROJECTED: proj(t+k) = value(t) + k x drift, drift = (value(t) - value(t-6)) / 6; the % changes, ratios and ONI",
        "  inputs are then computed from the projected series (see step 1, projected_market_features).",
        "Blend = plain average of the Ridge and XGBoost predicted changes. XGBoost has no closed form; SHAP shows its drivers.",
        "Feature_Importance_COW / _BUFFALO: % share of each input. Ridge share = size of its standardised coefficient (linear effect);",
        "  XGB share = mean |SHAP| (also non-linear effects). Overall = average of Recursive Ridge, Recursive XGB, Direct Ridge, Direct XGB.",
        "WEATHER BAND: the same fitted models are run three times - with the EXPECTED weather, with expected + lower band edge (LOW) and with expected + upper edge (HIGH).",
        "  Band edges = seasonal percentiles of past errors (actual - expected), see sheet Weather_Band_Rules.  Weather band = min / max of the three forecasts.",
        "  TOTAL band = weather band widened by the 80% model-error band of the backtest (target month <= origin when --origin is used).",
        "  fat % and SNF % stay at their expected values in all three runs (the client's band is for weather only).",
        "Importance says what the model LEANS on, not what causes price moves. Inputs that are identical for all districts in a month can act as a time stamp.",
    ]})
    extra = {}
    if wb is not None:
        live = sorted(set(forecast["District"]))
        extra["Weather_Band"] = wb[1][wb[1]["District"].isin(live)].rename(columns={"low": "Low", "expected": "Expected", "high": "High"})
        extra["Weather_Band_Rules"] = wb[0]
    extra["Intervention_Windows"] = s1.intervention_table(panels, cfg)
    sheets = {"Forecast": forecast, "Compare_Direct_vs_RecX": compare, **extra,
              "Feature_Importance_COW": imps["COW"], "Feature_Importance_BUFFALO": imps["BUFFALO"],
              "Importance_by_Group": pd.concat([grps["COW"], grps["BUFFALO"]]),
              "Ridge_Equation_Direct": pd.concat(eq), "Recursive_Equation": pd.concat(req),
              "XGB_SHAP_Top15": shap_top, "How_to_use": howto}
    fname = f"Forecast_from_{origin_ts:%b%Y}_heldout.xlsx" if origin_ts is not None else "Final_Forecast.xlsx"
    s2.write_excel(outdir / fname, sheets)
    if origin_ts is not None:
        print(f"\nHELD-OUT CHECK: trained only on rows with target month <= {origin_ts:%b %Y}; forecast of the next 6 months vs actual")
        print("(model-error band: backtest errors with target month <= origin only; weather band: expected weather +- seasonal 95% error band from data <= origin)")
        print("(the WEATHER band alone is the effect of weather uncertainty on the forecast, not a forecast interval: it is not meant to cover 80-95% of actuals)")
        sc = forecast.dropna(subset=["Actual (Rs/kg TS)"])
        for mt in s1.MILK_TYPES:
            g = sc[sc["Milk Type"] == mt.title()]
            if g.empty:
                continue
            h = g["Horizon (months)"]
            naive = (g["Last Actual (Rs/kg TS)"] - g["Actual (Rs/kg TS)"]).abs()
            print(f"{mt}: districts {sorted(g['District'].unique())}")
            print(f"   MAE by month ahead      : " + ", ".join(f"h{k}={g.loc[h == k, 'Abs error (Rs/kg TS)'].mean():.2f}" for k in sorted(h.unique())))
            print(f"   next month (target 2.0) : {g.loc[h == 1, 'Abs error (Rs/kg TS)'].mean():.2f}   | rate-stays-flat {naive[h == 1].mean():.2f}")
            print(f"   Q1 = months 1-3 (5.0)   : {g.loc[h <= 3, 'Abs error (Rs/kg TS)'].mean():.2f}   | rate-stays-flat {naive[h <= 3].mean():.2f}")
            print(f"   Q2 = months 4-6         : {g.loc[h >= 4, 'Abs error (Rs/kg TS)'].mean():.2f}   | rate-stays-flat {naive[h >= 4].mean():.2f}")
            if "Weather band low (Rs/kg TS)" in g.columns:
                act = g["Actual (Rs/kg TS)"]
                in_w = ((act >= g["Weather band low (Rs/kg TS)"]) & (act <= g["Weather band high (Rs/kg TS)"]))
                in_t = ((act >= g["TOTAL band low (weather + model error)"]) & (act <= g["TOTAL band high (weather + model error)"]))
                in_m = ((act >= g["Lower 80% (Rs/kg TS)"]) & (act <= g["Upper 80% (Rs/kg TS)"]))
                print(f"   band width (Rs/kg TS)   : weather only {g['Weather band width (Rs/kg TS)'].mean():.2f} | model-error 80% {(g['Upper 80% (Rs/kg TS)'] - g['Lower 80% (Rs/kg TS)']).mean():.2f}"
                      f" | total {(g['TOTAL band high (weather + model error)'] - g['TOTAL band low (weather + model error)']).mean():.2f}")
                print(f"   actual inside the band  : weather band {in_w.mean() * 100:.0f}% | model-error 80% band {in_m.mean() * 100:.0f}% | total band {in_t.mean() * 100:.0f}%   (n={len(g)})")

    pd.set_option("display.width", 220)
    cols = ["Milk Type", "District", "Forecast Month", "Last Actual (Rs/kg TS)", "Forecast (Rs/kg TS)",
            "Weather band low (Rs/kg TS)", "Weather band high (Rs/kg TS)", "TOTAL band low (weather + model error)",
            "TOTAL band high (weather + model error)"] if wb is not None else \
           ["Milk Type", "District", "Forecast Month", "Last Actual (Rs/kg TS)", "Forecast (Rs/kg TS)", "Lower 80% (Rs/kg TS)", "Upper 80% (Rs/kg TS)"]
    cols = cols + ["Method"]
    print("\nFINAL FORECAST (Rs/kg TS): h=1-4 short-term, h=5-6 INDICATIVE")
    print(forecast[cols].round(1).to_string(index=False))
    for mt in s1.MILK_TYPES:
        print(f"\nFEATURE IMPORTANCE - {mt} ({len(imps[mt])} inputs): top 15")
        print(imps[mt][["Rank", "Feature", "Group", "Overall importance %"]].head(15).round(2).to_string(index=False))
        print(grps[mt][["Group", "Inputs", "Overall importance %"]].round(1).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Final forecast Jul-Dec 2026, Rs/kg TS, district level")
    ap.add_argument("--data", default=s1.Config.data_xlsx)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--buffalo-method", default=None, help=f'method for buffalo (default "{FINAL_METHOD}")')
    # ap.add_argument("--cow-method", default=None, help=f'method for cow (default "{FINAL_METHOD}")')    # (original)
    ap.add_argument("--cow-method", default=None, help=f'method for cow (default "{FINAL_METHOD_COW}"; old cow method: "{FINAL_METHOD}")')
    ap.add_argument("--origin", default=None, help="forecast from this origin month instead of the last actual, e.g. 2025-12-01 for Jan-Jun 2026 (held-out check)")
    ap.add_argument("--no-forward", action="store_true", help="old behaviour: weather / fat / SNF observed at the origin")
    ap.add_argument("--no-flags", action="store_true", help="do not use the intervention flags as inputs")
    ap.add_argument("--keep-2021", action="store_true", help="keep training rows whose target month is before 2022")
    ap.add_argument("--band-level", type=float, default=95.0, help="weather band level in %% (default 95)")
    ap.add_argument("--no-weather-band", action="store_true", help="skip the LOW / HIGH weather runs (point forecast only)")
    s1.add_swing_args(ap)
    ap.add_argument("--auto-method", action="store_true", help="use the backtest-best method of each milk type instead of the fixed one")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ov = {"BUFFALO": a.buffalo_method, "COW": a.cow_method} if a.auto_method else \
         {"BUFFALO": a.buffalo_method or FINAL_METHOD, "COW": a.cow_method or FINAL_METHOD_COW}   # NEW cow default (original: a.cow_method or FINAL_METHOD)
    main(s1.config_from_args(a), s1.resolve(a.outdir), ov, a.origin, a.band_level, not a.no_weather_band)
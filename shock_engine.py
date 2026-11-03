"""
shock_engine.py - "what if the market moves?" engine for the CDPL Maharashtra milk-rate models
=================================================================================================
NOTHING in step1 / step2 / step3 / step4 / cow_model is changed.  This file only IMPORTS them.

What it does
------------
1. Builds the data + training table exactly like step3 (same Config, same features, same flags).
2. Fits the SAME models ONCE at the last actual month (the origin):
      * Direct Ridge / XGB / Blend   (one pair per horizon, step 2)
      * RecursiveX Ridge / XGB / Blend (one one-step pair fed back on itself, step 2)
      * Cow Ensemble v2 + its 3 members (cow_model.py, COW only)
3. For a scenario ("SMP +10 %", "feed +15 %", "hotter summer" ...) it changes ONLY the forecast-time inputs
   (market series of the last months / expected weather of the coming months), recomputes every derived input
   with the project's own functions (projected_market_features, forward_block, scenario_table) and runs the
   already fitted models again.  Models are never refitted on shocked data.
4. Delta = scenario forecast - baseline forecast  (Rs per kg of Total Solids).

5. feature_importance(): permutation importance of the fitted headline model, for COW and BUFFALO separately.
6. tornado(): a driver that the chosen model has NO input for is reported as "not used" (NaN) instead of a misleading 0.

The baseline of this engine is checked against step2.forecast_origin in ``self_check`` (must match to 1e-9).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import cow_model as cm
import step1_data_features as s1
import step2_backtest as s2

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ drivers a user can shock
# key -> (label, kind, unit text).  kind "pct": multiply by (1 + x/100) ; kind "abs": add x
MARKET_DRIVERS: Dict[str, Tuple[str, str, str]] = {
    "feed": ("Cattle-feed bag price", "pct", "%"),
    "maize": ("Maize (Latur)", "pct", "%"),
    "soy_meal": ("Soybean meal (Latur)", "pct", "%"),
    "smp": ("Skimmed milk powder (SMP)", "pct", "%"),
    "ghee": ("Ghee (Mumbai)", "pct", "%"),
    "butter": ("Butter (Maharashtra)", "pct", "%"),
    "cpi": ("CPI Maharashtra (inflation level)", "pct", "%"),
    "oni": ("ONI (El Nino / La Nina anomaly)", "abs", "deg C"),
    "monsoon": ("Monsoon impact index", "abs", "index pts"),
}
WEATHER_DRIVERS: Dict[str, Tuple[str, str]] = {      # key -> (label, unit)
    "temp": ("Temperature (avg / max / min)", "deg C"),
    "rain": ("Rainfall", "% of expected"),
    "humid": ("Relative humidity", "% points"),
}
# model inputs that each market driver feeds (step 1, projected_market_features)
DRIVER_FEATURES: Dict[str, List[str]] = {
    "feed": ["feed_ch3", "feed_maize_ratio"], "maize": ["maize_ch3", "feed_maize_ratio"], "soy_meal": ["soy_meal_ch3"],
    "smp": ["smp_ch3", "smp_ch6", "ghee_smp_ratio"], "ghee": ["ghee_ch3", "ghee_ch6", "ghee_smp_ratio"],
    "butter": ["butter_ch3"], "cpi": ["cpi_ch3"], "oni": ["oni_t", "oni_l3"], "monsoon": ["monsoon_t"],
}
HEADLINE = {"COW": cm.METHOD, "BUFFALO": "RecursiveX XGB"}          # same as step 3 (FINAL_METHOD_COW / FINAL_METHOD)
RATE = "Rs/kg TS"


def all_methods(mt: str) -> List[str]:
    base = [f"{f} {m}" for f in s2.FAMILIES for m in s2.LEARNERS]
    return (cm.COW_COLUMNS + base) if mt == "COW" else base


# ------------------------------------------------------------------ shocks
def shocked_market(market: pd.DataFrame, shocks: Dict[str, float], origin: pd.Timestamp, ramp: int) -> pd.DataFrame:
    """Copy of the market table in which each shocked series moves by its full shock at the origin month, building up
    linearly over the last ``ramp`` months (ramp = 1 -> one sudden jump in the origin month)."""
    m = market.astype(float).copy()
    months = list(m.index[m.index <= origin][-max(1, int(ramp)):])
    for col, val in (shocks or {}).items():
        if not val or col not in m.columns:
            continue
        kind = MARKET_DRIVERS[col][1]
        for j, d in enumerate(months, start=1):
            f = j / len(months)
            m.loc[d, col] = m.loc[d, col] * (1 + val / 100.0 * f) if kind == "pct" else m.loc[d, col] + val * f
    return m


@dataclass
class MilkState:
    mt: str
    panel: pd.DataFrame
    table: pd.DataFrame
    recx: Dict
    feats: List[str]
    origin: pd.Timestamp
    districts: List[str]
    rows: pd.DataFrame                       # table rows at the origin (forecast rows), districts x horizons
    weights: pd.Series                       # ts_kg of the origin month per district (volume weights)
    Ew: Dict[str, pd.DataFrame]              # expected weather (for % rain shocks)
    direct: Dict[int, Tuple] = field(default_factory=dict)          # h -> (ridge, xgb, train)
    recx_pair: Tuple = None                                           # (ridge, xgb) one-step models
    cow: Dict[int, Tuple] = field(default_factory=dict)              # h -> (m_w, m_h, m_r, train)


class ShockEngine:
    def __init__(self, data_path: str | None = None, H: int = 12):
        self.cfg = s1.Config(data_xlsx=data_path) if data_path else s1.Config()
        self.H = int(min(H, self.cfg.max_horizon))
        raw = s1.load_raw(self.cfg)
        self.panels, self.market = s1.build_panels(raw, self.cfg)
        self.state: Dict[str, MilkState] = {}
        for mt in s1.MILK_TYPES:
            self.state[mt] = self._build(mt)

    # ------------------------------------------------------------- fit (once)
    def _build(self, mt: str) -> MilkState:
        cfg, panel = self.cfg, self.panels[mt]
        table = s1.build_training_table(panel, self.market, cfg)
        recx = s2.prepare_recx(panel, self.market, table, cfg, s1.spec_for(mt, forward=cfg.forward_inputs, flags=cfg.use_flags))
        feats = recx["spec"]["all"]
        origin = panel["rate"].dropna().index.get_level_values(s1.DATE).max()
        districts = s1.forecastable_districts(panel, origin, cfg.forecast_min_months)
        rows = table[(table["origin"] == origin) & table[s1.DISTRICT].isin(districts) & (table["horizon"] <= self.H)].copy()
        w = panel["ts_kg"].reindex(pd.MultiIndex.from_product([districts, [origin]])).droplevel(1)
        _, Ew, _ = s1.forward_inputs_wide(panel, cfg)
        st = MilkState(mt, panel, table, recx, feats, origin, districts, rows, w.fillna(w.mean()), Ew)
        for h in range(1, self.H + 1):
            th = table[table["horizon"] == h]
            train = th[(th["target_month"] <= origin) & th["target"].notna()]
            s1.assert_no_leakage(train, origin)
            if len(train) < s2.MIN_TRAIN_ROWS:
                continue
            r, x = s2.fit_pair(train, feats)
            st.direct[h] = (r, x, train)
            if mt == "COW":
                wt = cm._recency_weights(train["target_month"], origin)
                y = train["target"]
                st.cow[h] = (cm._hgb().fit(train[cm.LEAN], y, sample_weight=wt), cm._hgb().fit(train[cm.LEAN], y),
                             cm._ridge().fit(train[cm.MOMENTUM], y), train)
        trx = recx["train"][(recx["train"]["target_month"] <= origin) & recx["train"]["target"].notna()]
        s1.assert_no_leakage(trx, origin)
        st.recx_pair = s2.fit_pair(trx, feats)
        return st

    # ------------------------------------------------------------- inputs of a scenario
    def _scenario_rows(self, st: MilkState, market: pd.DataFrame, weather: Dict[str, float] | None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """(forecast rows with scenario inputs, training table whose origin rows carry the scenario weather)."""
        weather = {k: v for k, v in (weather or {}).items() if v}
        table = st.table
        rows = st.rows.copy()
        if weather:                                              # forward weather inputs rebuilt from expected + shift
            ext = st.Ew["rain"].index
            cols = st.Ew["rain"].columns
            sh: Dict[str, pd.DataFrame] = {}
            if "temp" in weather:
                for v in ("temp_avg", "temp_max", "temp_min"):
                    sh[v] = pd.DataFrame(weather["temp"], index=ext, columns=cols)
            if "rain" in weather:
                sh["rain"] = st.Ew["rain"] * (weather["rain"] / 100.0)
            if "humid" in weather:
                sh["humid"] = pd.DataFrame(weather["humid"], index=ext, columns=cols)
            table = s1.scenario_table(st.panel, self.cfg, st.table, st.origin, sh)
            rows = table[(table["origin"] == st.origin) & table[s1.DISTRICT].isin(st.districts) & (table["horizon"] <= self.H)].copy()
        for h in range(1, self.H + 1):                           # market inputs recomputed from the (shocked) market table
            pm = s1.projected_market_features(market, h, self.cfg.proj_window).loc[st.origin]
            m = rows["horizon"] == h
            for c in pm.index:
                if c in rows.columns:
                    rows.loc[m, c] = pm[c]
        return rows, table

    # ------------------------------------------------------------- forecast under a scenario
    def forecast(self, mt: str, market_shocks: Dict[str, float] | None = None, ramp: int = 3,
                 weather: Dict[str, float] | None = None, methods: Sequence[str] | None = None,
                 H: int | None = None) -> pd.DataFrame:
        """Long table: district, horizon, target_month, rate_t + one column per method (absolute Rs/kg TS)."""
        st = self.state[mt]
        H = int(min(H or self.H, self.H))
        need = set(methods) if methods else set(all_methods(mt))
        market = shocked_market(self.market, market_shocks or {}, st.origin, ramp) if market_shocks and any(market_shocks.values()) else self.market
        rows, table = self._scenario_rows(st, market, weather)
        rows = rows[rows["horizon"] <= H]
        out = rows[[s1.DISTRICT, "horizon", "target_month", "rate_t"]].reset_index(drop=True)
        feats = st.feats

        direct_cols = {m for m in need if m.startswith("Direct")}
        cow_cols = {m for m in need if m in cm.COW_COLUMNS}
        if direct_cols or cow_cols:
            parts = []
            for h in range(1, H + 1):
                if h not in st.direct:
                    continue
                t = rows[rows["horizon"] == h]
                if t.empty:
                    continue
                res = pd.DataFrame({s1.DISTRICT: t[s1.DISTRICT].to_numpy(), "horizon": h})
                base = t["rate_t"].to_numpy()
                if direct_cols:
                    r, x, _ = st.direct[h]
                    pr, px = r.predict(t[feats]), x.predict(t[feats])
                    for nm, c in (("Ridge", pr), ("XGB", px), ("Blend", (pr + px) / 2)):
                        res[f"Direct {nm}"] = base + c
                if cow_cols and mt == "COW":
                    m_w, m_h, m_r, _ = st.cow[h]
                    c = {cm.MEMBERS[0]: m_w.predict(t[cm.LEAN]), cm.MEMBERS[1]: m_h.predict(t[cm.LEAN]), cm.MEMBERS[2]: m_r.predict(t[cm.MOMENTUM])}
                    c[cm.METHOD] = np.mean([c[k] for k in cm.MEMBERS], axis=0)
                    for k, v in c.items():
                        res[k] = base + v
                parts.append(res)
            if parts:
                out = out.merge(pd.concat(parts, ignore_index=True), on=[s1.DISTRICT, "horizon"], how="left")
        if any(m.startswith("RecursiveX") for m in need):
            ex = s2.make_exog({**st.recx, "market": market, "table": table}, st.origin)
            r1, x1 = st.recx_pair
            paths = s2.paths_from_fit(st.panel, r1, x1, st.origin, H, feats, ex)
            for kind in s2.LEARNERS:
                out[f"RecursiveX {kind}"] = [paths[kind][int(h)].get(d, np.nan) for d, h in zip(out[s1.DISTRICT], out["horizon"])]
        return out.sort_values([s1.DISTRICT, "horizon"]).reset_index(drop=True)

    # ------------------------------------------------------------- helpers for the app
    def aggregate(self, mt: str, df: pd.DataFrame, col: str, districts: Iterable[str] | None = None) -> pd.Series:
        """Volume-weighted (ts_kg of the origin month) average rate per horizon over the chosen districts."""
        st = self.state[mt]
        d = df[df[s1.DISTRICT].isin(list(districts) if districts is not None else st.districts)].dropna(subset=[col]).copy()
        d["w"] = d[s1.DISTRICT].map(st.weights)
        d["wr"] = d[col] * d["w"]
        g = d.groupby("horizon")
        return g["wr"].sum() / g["w"].sum()

    def history(self, mt: str, districts: Iterable[str] | None = None, months: int = 24) -> pd.Series:
        st = self.state[mt]
        r = st.panel["rate"].unstack(s1.DISTRICT)[list(districts) if districts is not None else st.districts]
        w = st.weights.reindex(r.columns)
        ok = r.notna()
        s = (r.fillna(0) * w).sum(axis=1) / (ok * w).sum(axis=1).replace(0, np.nan)
        return s.dropna().iloc[-months:]

    def changed_inputs(self, mt: str, market_shocks: Dict[str, float], ramp: int, weather: Dict[str, float] | None, horizon: int) -> pd.DataFrame:
        """Model inputs of the forecast rows before / after the scenario at one horizon (volume-weighted over districts)."""
        st = self.state[mt]
        market = shocked_market(self.market, market_shocks, st.origin, ramp)
        rows, _ = self._scenario_rows(st, market, weather)
        a = st.rows[st.rows["horizon"] == horizon].set_index(s1.DISTRICT)
        b = rows[rows["horizon"] == horizon].set_index(s1.DISTRICT)
        w = st.weights.reindex(a.index)
        out = []
        for f in st.feats:
            if f not in a.columns:
                continue
            va, vb = np.average(a[f].fillna(0), weights=w), np.average(b[f].fillna(0), weights=w)
            if abs(vb - va) > 1e-9:
                out.append({"Model input": f, "Group": s1.FEATURE_GROUP.get(f, ""), "Before": va, "After": vb, "Change": vb - va,
                            "Meaning": s1.FEATURE_INFO[f][2]})
        return pd.DataFrame(out)

    def used_inputs(self, mt: str, method: str) -> set:
        """Model inputs the chosen method can read (Cow Ensemble v2 uses only the LEAN / MOMENTUM lists)."""
        if method in cm.COW_COLUMNS:
            return set(cm.MOMENTUM) if method == "Cow Ridge momentum" else (set(cm.LEAN) | set(cm.MOMENTUM) if method == cm.METHOD else set(cm.LEAN))
        return set(self.state[mt].feats)

    # ------------------------------------------------------------- impact of every market driver (tornado)
    def tornado(self, mt: str, method: str, size_pct: float, ramp: int, horizon: int) -> pd.DataFrame:
        """Push every market driver up and down alone.  A driver for which the chosen model has no input at all is
        NOT run: its effect is NaN ("not used"), because 0 would wrongly suggest 'tested, no effect'."""
        used = self.used_inputs(mt, method)
        base = self.aggregate(mt, self.forecast(mt, None, ramp, None, [method], horizon), method).loc[horizon]
        rows = []
        for key, (label, kind, unit) in MARKET_DRIVERS.items():
            feeds = [f for f in DRIVER_FEATURES[key] if f in used]
            size = size_pct if kind == "pct" else round(0.5 * float(self.market[key].std()), 2)   # abs series: +-0.5 std of history
            up = dn = np.nan
            if feeds:
                res = {}
                for sign in (+1, -1):
                    f = self.forecast(mt, {key: sign * size}, ramp, None, [method], horizon)
                    res[sign] = self.aggregate(mt, f, method).loc[horizon] - base
                up, dn = res[+1], res[-1]
            rows.append({"Driver": label, "Key": key, "Shock size": f"+-{size:g} {unit}",
                         "Used by this model": "yes" if feeds else "no",
                         "Model inputs it feeds (used by this model)": ", ".join(feeds) or "none - this model has no input built from it",
                         "Rise (+) effect": up, "Fall (-) effect": dn})
        out = pd.DataFrame(rows)
        out["Swing"] = (out["Rise (+) effect"] - out["Fall (-) effect"]).abs()
        return out.sort_values("Swing", ascending=False, na_position="last").reset_index(drop=True)

    def sweep(self, mt: str, method: str, key: str, values: Sequence[float], ramp: int, horizons: Sequence[int]) -> pd.DataFrame:
        base = self.aggregate(mt, self.forecast(mt, None, ramp, None, [method], max(horizons)), method)
        rec = []
        for v in values:
            agg = self.aggregate(mt, self.forecast(mt, {key: v}, ramp, None, [method], max(horizons)), method)
            for h in horizons:
                rec.append({"Shock": v, "Horizon": f"N+{h}", "Change vs baseline": agg.loc[h] - base.loc[h]})
        return pd.DataFrame(rec)

    # ------------------------------------------------------------- impact of EVERY model input (direct models)
    def input_sensitivity(self, mt: str, horizon: int, n_std: float = 1.0) -> pd.DataFrame:
        """Push each model input alone by +- n_std standard deviations (std of the training rows of that horizon) and
        read the change of the predicted rate change.  Uses the DIRECT model of that horizon (Cow Ensemble v2 for cow,
        Direct Blend for buffalo) because it maps inputs -> forecast in one step."""
        st = self.state[mt]
        t = st.rows[st.rows["horizon"] == horizon].set_index(s1.DISTRICT)
        w = st.weights.reindex(t.index).to_numpy()
        feats = st.feats
        train = (st.cow[horizon][3] if mt == "COW" else st.direct[horizon][2])

        def predict(frame: pd.DataFrame) -> np.ndarray:
            if mt == "COW":
                m_w, m_h, m_r, _ = st.cow[horizon]
                return np.mean([m_w.predict(frame[cm.LEAN]), m_h.predict(frame[cm.LEAN]), m_r.predict(frame[cm.MOMENTUM])], axis=0)
            r, x, _ = st.direct[horizon]
            return (r.predict(frame[feats]) + x.predict(frame[feats])) / 2

        base = np.average(predict(t), weights=w)
        used = sorted(set(cm.LEAN) | set(cm.MOMENTUM)) if mt == "COW" else feats
        rec = []
        for f in used:
            sd = float(train[f].std())
            if not np.isfinite(sd) or sd == 0:
                continue
            up, dn = t.copy(), t.copy()
            up[f] = up[f] + n_std * sd
            dn[f] = dn[f] - n_std * sd
            rec.append({"Model input": f, "Group": s1.FEATURE_GROUP.get(f, ""), "1 std (history)": sd,
                        "Input +1 std": np.average(predict(up), weights=w) - base,
                        "Input -1 std": np.average(predict(dn), weights=w) - base,
                        "Meaning": s1.FEATURE_INFO[f][2]})
        out = pd.DataFrame(rec)
        out["Swing"] = (out["Input +1 std"] - out["Input -1 std"]).abs()
        return out.sort_values("Swing", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------- feature importance (cow / buffalo separately)
    def feature_importance(self, mt: str, horizons: Sequence[int] = (1, 2, 3, 4, 5, 6), n_repeat: int = 3,
                           seed: int = 0, max_rows: int = 3000) -> Tuple[pd.DataFrame, str]:
        """Permutation importance of the fitted headline model of ``mt``.

        Each model input is shuffled (breaking its link to the target) and the rise of the model's mean absolute error
        on the training rows is measured (Rs/kg TS).  Bigger rise = the model leans on that input more.
        COW     -> Cow Ensemble v2 (direct models, averaged over ``horizons``)
        BUFFALO -> RecursiveX XGB one-step model (the model that produces the buffalo headline forecast)
        Returns (table, model name).  Importance (%) = share of the total (negative values clipped to 0).
        """
        st = self.state[mt]
        rng = np.random.default_rng(seed)
        jobs = []                                              # (training frame, predict function)
        if mt == "COW":
            name = "Cow Ensemble v2 (average over horizons)"
            feats = sorted(set(cm.LEAN) | set(cm.MOMENTUM))
            for h in horizons:
                if h not in st.cow:
                    continue
                m_w, m_h, m_r, train = st.cow[h]

                def pred(fr, m_w=m_w, m_h=m_h, m_r=m_r):
                    return np.mean([m_w.predict(fr[cm.LEAN]), m_h.predict(fr[cm.LEAN]), m_r.predict(fr[cm.MOMENTUM])], axis=0)
                jobs.append((train, pred))
        else:
            name = "RecursiveX XGB (one-step model)"
            feats = list(st.feats)
            trx = st.recx["train"]
            trx = trx[(trx["target_month"] <= st.origin) & trx["target"].notna()]
            _, x = st.recx_pair
            jobs.append((trx, lambda fr: x.predict(fr[feats])))

        imp = pd.Series(0.0, index=feats)
        for train, pred in jobs:
            fr = (train.sample(max_rows, random_state=seed) if len(train) > max_rows else train).copy()
            y = fr["target"].to_numpy(dtype=float)
            base_err = float(np.mean(np.abs(pred(fr) - y)))
            for f in feats:
                if f not in fr.columns:
                    continue
                orig = fr[f].to_numpy().copy()
                errs = []
                for _ in range(n_repeat):
                    fr[f] = rng.permutation(orig)
                    errs.append(float(np.mean(np.abs(pred(fr) - y))))
                fr[f] = orig
                imp[f] += np.mean(errs) - base_err
        imp = (imp / max(len(jobs), 1)).clip(lower=0.0)
        tot = float(imp.sum())
        out = pd.DataFrame({"Model input": imp.index, "Importance (%)": (100 * imp / tot).to_numpy() if tot > 0 else 0.0,
                            "MAE rise when shuffled": imp.to_numpy()})
        out["Group"] = [s1.FEATURE_GROUP.get(f, "Other") or "Other" for f in out["Model input"]]
        out["Meaning"] = [s1.FEATURE_INFO[f][2] if f in s1.FEATURE_INFO else "" for f in out["Model input"]]
        return out.sort_values("Importance (%)", ascending=False).reset_index(drop=True), name

    # ------------------------------------------------------------- safety check
    def self_check(self, mt: str) -> float:
        """Baseline of this engine vs the project's own step2.forecast_origin (max abs difference, Rs/kg TS)."""
        st = self.state[mt]
        ref, _ = s2.forecast_origin(st.panel, st.table, st.origin, self.H, self.cfg, st.recx, st.districts, cow_extra=(mt == "COW"))
        mine = self.forecast(mt)
        m = ref.merge(mine, on=[s1.DISTRICT, "horizon"], suffixes=("_ref", ""))
        cols = [c for c in all_methods(mt) if f"{c}_ref" in m.columns]
        return float(max(np.nanmax(np.abs(m[c] - m[f"{c}_ref"])) for c in cols))
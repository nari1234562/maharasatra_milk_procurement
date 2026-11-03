"""
step1_data_features.py - FILE 1 of 4: data, target, features, intervention flags, weather expectation + weather band
(MERGED: old step1_prepare_data.py + stepA_weather_forecast.py - every function of both is kept)
==========================================================================================================
Project  : CDPL Maharashtra milk procurement rate, SHORT-TERM (CBE) forecast
Target   : Rate_TS = Base_Price_INR / TS_Kg   (Rs per kg of Total Solids)
Level    : DISTRICT x MONTH, COW and BUFFALO modelled separately

VERSION 2 (forward-looking inputs + intervention flags)  -  what is new
-----------------------------------------------------------------------
* TASK 1  Weather (temp avg / max / min, rain, humidity) and milk quality (fat %, SNF %) are now taken FOR THE TARGET MONTH.
          Months up to the origin t are the actual values; months after t are EXPECTED values =
              average of the SAME calendar month in EARLIER years  (strictly earlier years, so nothing after the origin is used).
          3- / 6-month rain, 3-month max temperature, heat index (THI) and the fat / SNF trends are RECOMPUTED from that
          monthly series (actual up to t, expected after t).  Training rows are built the same way ("train the way we forecast").
          A district with no earlier year of its own for a month uses the average of the other districts' earlier years
          (flag column share is reported).  Switch: Config.forward_inputs (False = the old "observed at origin" version).
* TASK 2  Intervention flags run_up_share / crash_share for the 2022-24 price swing.  Windows are found from the data
          (peak / trough of the districts that have a rate in every month), see swing_windows().  The flag of a training row is
          the share of the months (t, t+h] that fall inside the window; it is 0 for every forecast made after the swing.
          Switch: Config.use_flags (False = flags not used as model inputs).
* Config.train_from_target drops training rows whose target month is before that date (first year has no earlier year to average).

VERSION 3 additions (this file)
-------------------------------
* Config.run_up_override / crash_override (CLI --run-up 2022-01:2023-04 --crash 2023-05:2024-02): business-validated intervention dates replace the
  dates found from the data.  intervention_table() lists the windows for the business to validate (written to outputs/intervention_windows.csv).
* scenario_table(): rebuilds the forward-weather inputs of the FORECAST rows from expected + a band shift (LOW / HIGH weather), everything else untouched.
  Used by step3 to turn the weather band of part 2 of this file into a forecast band.

What changed versus the first version
-------------------------------------
* The model inputs grew from 27 (same list for both milk types) to
      COW      : 42 inputs
      BUFFALO  : 46 inputs
  = 36 CORE inputs shared by both + milk-type specific extras (6 for cow, 10 for buffalo).
* Raw columns that were loaded but never used before are now used: humidity, Net price (incentive), monsoon index,
  butter, procurement volume.  New derived inputs: momentum / volatility / range position of the rate, heat-stress
  index (THI), fat & SNF trends, supply trend, incentive %, fat-vs-SNF market value ratios, 6-month market changes.
* The extras are chosen by DAIRY LOGIC (see COW_EXTRA / BUFFALO_EXTRA), not by looking at test results.
* PRUNING: (old) step2b_feature_selection.py could drop inputs that hurt the one-step RecursiveX XGBoost forecast.  Its result is
  saved in selected_features.json; if that file is present, FEATURES_BY_TYPE below is the pruned list (see _load_selected).
* Logic is unchanged: every input is dated t or earlier, weather is never forecast, market / ONI inputs are projected
  to the target month, one training row = (district, origin month t, horizon h), target = Rate(t+h) - Rate(t).

What this file does
-------------------
1. Reads the merged workbook (sheet Final_Merged) and validates it.
2. Adds the row-level target column ``Rate_TS_Base`` (Rs/kg TS) and saves a copy of the workbook
   (formulas, so it recalculates) together with a district-month sheet.
3. Aggregates to district-month:  rate = SUM(Base_Price_INR) / SUM(TS_Kg)   (volume weighted, NOT an
   average of ratios).
4. Builds the direct multi-horizon training table.  Each row is "standing at origin month t, predict
   month t+h":

        target      = Rate(t+h) - Rate(t)                      (the CHANGE; forecast = last rate + change)
        every feature uses information dated t or earlier      (no leakage; see assert_no_leakage)

   Weather is NOT forecast.  Weather at t, t-1, t-2 ... is used as it already happened, so for a forecast of
   month t+h its lag is at least h.  Market / ONI inputs are PROJECTED to the target month:
   last value + k x average of the last 6 monthly changes (see projected_market_features).
5. Writes Feature_Dictionary.xlsx: every input, its group, how it is known at forecast time, and its formula.

    python step1_data_features.py             (prepares the data AND writes the weather-band report; defaults: data/ in, outputs/ out)
"""


from __future__ import annotations

import argparse
import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("milk")

HERE = Path(__file__).resolve().parent


def resolve(path: str | Path) -> Path:
    """Relative paths are taken relative to this folder, so the scripts run from any working directory."""
    p = Path(path)
    return p if p.is_absolute() else HERE / p


def find_data_file(path: str | Path) -> Path:
    """Return the data workbook.  If ``path`` does not exist, look for Maharashtra_Final_Merged*.xlsx next to the scripts
    and in a data/ sub-folder (newest file wins) and say clearly what was searched when nothing is found."""
    p = Path(path)
    if p.exists():
        return p
    folders = [HERE, HERE / "data", Path.cwd(), Path.cwd() / "data"]
    hits = sorted({f for fo in folders if fo.exists() for f in fo.glob("Maharashtra_Final_Merged*.xlsx") if not f.name.startswith("~$")},
                  key=lambda f: f.stat().st_mtime, reverse=True)
    if hits:
        log.warning("Data file %s not found - using %s", p, hits[0])
        return hits[0]
    raise FileNotFoundError(
        f"Data workbook not found: {p}\nPut your Excel file (for example Maharashtra_Final_Merged_Jan2021_dec_2025_49a7e8.xlsx) in "
        f"{HERE} (next to the scripts) and run again, or pass the exact path:  --data \"C:\\full\\path\\file.xlsx\"")


DATE, DISTRICT, MTYPE = "month", "district", "milk_type"
MILK_TYPES: Tuple[str, ...] = ("COW", "BUFFALO")

RAW_RENAME = {
    "Month": DATE,
    "District": DISTRICT,
    "Milk_Type": MTYPE,
    "Chilling Centre": "centre",
    "Feed_Price_per_bag_INR": "feed",
    "Maize_Latur_INR_per_Qtl": "maize",
    "SoybeanMeal_Latur_INR_per_Qtl": "soy_meal",
    "SMP_Maharashtra_INR_per_Kg": "smp",
    "Ghee_Mumbai_Modal_INR_per_Qtl": "ghee",
    "Butter_Maharashtra_INR_per_100g": "butter",
    "CPI_Maharashtra_Combined_2012base": "cpi",
    "ONI_C_Anomaly": "oni",
    "MH_Monsoon_Impact_Index": "monsoon",
    "Avg_Temperature_C": "temp_avg",
    "Avg_Max_Temperature_C": "temp_max",
    "Avg_Min_Temperature_C": "temp_min",
    "Total_Precipitation_mm": "rain",
    "Avg_Relative_Humidity_Pct": "humid",
}
MARKET_PCT = ["feed", "maize", "soy_meal", "smp", "ghee", "cpi", "butter"]   # price-type series: % changes
MARKET_COLS = MARKET_PCT + ["oni", "monsoon"]                                # + anomaly-type series: levels
WEATHER_COLS = ["temp_avg", "temp_max", "temp_min", "rain", "humid"]


@dataclass(frozen=True)
class Config:
    data_xlsx: str = "data/Maharashtra_Final_Merged_Jan2021_dec_2025.xlsx"
    data_sheet: str = "Final_Merged"
    max_horizon: int = 12       # quarterly / 12-month backtest requires full-year monthly horizon support
    short_horizon: int = 4
    min_ts_kg: float = 2000.0   # district-month with less total solids than this gets no rate (unstable)
    min_row_history: int = 1    # a training row needs rate(t) and rate(t-1)
    forecast_min_months: int = 6  # a district is forecast only with a rate in each of the last 6 months
    forward_inputs: bool = True   # TASK 1: weather + fat/SNF for the TARGET month (actual up to t, same-month earlier-year average after t)
    use_flags: bool = True        # TASK 2: intervention flags run_up_share / crash_share are model inputs
    fwd_min_years: int = 1        # earlier years needed for a district's own expectation (else other districts' average)
    train_from_target: str | None = "2022-01-01"   # drop training rows whose target month is earlier (None = keep all rows)
    swing_start: str = "2022-01-01"                # start of the run-up (from the expert call; business to confirm)
    swing_peak_search: Tuple[str, str] = ("2022-01-01", "2023-12-01")   # peak is searched here
    swing_end_search: str = "2024-12-01"           # trough is searched between the peak and this month
    run_up_override: Tuple[str, str] | None = None  # business-validated run-up dates ("2022-01-01", "2023-04-01"); None = found from the data
    crash_override: Tuple[str, str] | None = None   # business-validated crash dates; None = found from the data
    project_market: bool = True   # market/ONI inputs = value PROJECTED to the target month (last value + h x avg of last 6 changes)
    proj_window: int = 6          # months used for the average monthly change
    # Ghee values flagged as wrong in Column_Guide (Feb-21 = 18,400 and Jul-21 = 4,000) -> neighbours' level
    ghee_fixes: Tuple[Tuple[str, float], ...] = (("2021-02-01", 42000.0), ("2021-07-01", 42000.0))
    # Competitor quotes (optional "Model B"): Rs per litre, Apr-2024 .. Mar-2026
    competitor_xlsx: str = "data/Competitor_Rates.xlsx"
    competitor_sheet: str = "Sheet1"
    comp_gap_fill_months: int = 1     # a month with no quotes repeats the previous month (causal, max 1 month)
    comp_outlier_rel: float = 0.25    # drop a quote >25% from the centre-month median (needs >= 3 quotes)
    comp_valid_range: Tuple[Tuple[str, float, float], ...] = (("COW", 20.0, 70.0), ("BUFFALO", 30.0, 80.0))


def _parse_window(text: str | None) -> Tuple[str, str] | None:
    """'2022-01:2023-04' -> ('2022-01-01', '2023-04-01').  None / '' -> None."""
    if not text:
        return None
    lo, hi = [pd.Timestamp(x.strip()).to_period("M").to_timestamp() for x in text.split(":")]
    if hi < lo:
        raise ValueError(f"window end before start: {text}")
    return (lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d"))


def add_swing_args(ap) -> None:
    """Command-line switches for the business-validated intervention dates (used by step 1 / 2 / 2c / 3 / B)."""
    ap.add_argument("--run-up", default=None, help="override the run-up window, e.g. 2022-01:2023-04 (default: found from the data)")
    ap.add_argument("--crash", default=None, help="override the crash window, e.g. 2023-05:2024-02 (default: found from the data)")


def config_from_args(a, **extra) -> Config:
    """Build the Config from the command-line switches shared by step 2 / step 3 / the comparison script."""
    return Config(data_xlsx=str(resolve(a.data)), forward_inputs=not getattr(a, "no_forward", False),
                  use_flags=not getattr(a, "no_flags", False),
                  run_up_override=_parse_window(getattr(a, "run_up", None)), crash_override=_parse_window(getattr(a, "crash", None)),
                  train_from_target=None if getattr(a, "keep_2021", False) else Config.train_from_target, **extra)


# ------------------------------------------------------------------ loading
def load_raw(cfg: Config) -> pd.DataFrame:
    """Read Final_Merged, standardise names, validate and add the row-level target."""
    raw = pd.read_excel(find_data_file(cfg.data_xlsx), sheet_name=cfg.data_sheet).rename(columns=RAW_RENAME)
    raw[DATE] = pd.to_datetime(raw[DATE]).dt.to_period("M").dt.to_timestamp()
    raw[MTYPE] = raw[MTYPE].str.upper().str.strip()

    if raw.duplicated([DATE, "centre", "MCC_Category", MTYPE]).any():
        raise ValueError("Duplicate month / centre / category / milk-type rows")
    if not set(raw[MTYPE]) <= set(MILK_TYPES):
        raise ValueError(f"Unexpected milk types: {set(raw[MTYPE]) - set(MILK_TYPES)}")
    if (raw.groupby("centre")[DISTRICT].nunique() > 1).any():
        raise ValueError("A chilling centre appears under more than one district")

    # TS_Kg must equal Fat_Kg + SNF_Kg (checked: max gap 0.13 kg, i.e. rounding)
    gap = (raw["TS_Kg"] - (raw["Fat_Kg"] + raw["SNF_Kg"])).abs().max()
    if gap > 1.0:
        raise ValueError(f"TS_Kg differs from Fat_Kg + SNF_Kg by up to {gap:.2f} kg")

    # THE TARGET at row level: base price (NOT net - net contains agent incentives) per kg of total solids
    valid = (raw["Base_Price_INR"] > 0) & (raw["TS_Kg"] > 0)
    raw["Rate_TS_Base"] = (raw["Base_Price_INR"] / raw["TS_Kg"]).where(valid)
    n_bad = int((~valid).sum())
    if n_bad:
        log.warning("%d row(s) with zero base price or zero TS are given no rate and excluded from sums", n_bad)
    raw["row_valid"] = valid

    for stamp, value in cfg.ghee_fixes:
        raw.loc[raw[DATE] == pd.Timestamp(stamp), "ghee"] = value
    return raw


# ------------------------------------------------------------------- panels
def build_panels(raw: pd.DataFrame, cfg: Config) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """District-month panel per milk type (full monthly calendar) plus the shared market table."""
    calendar = pd.date_range(raw[DATE].min(), raw[DATE].max(), freq="MS")

    market = raw.groupby(DATE)[MARKET_COLS].first().reindex(calendar)
    if raw.groupby(DATE)[MARKET_COLS].nunique().max().max() > 1:
        raise ValueError("Market columns differ within a month")
    if market.isna().any().any():
        raise ValueError(f"Missing market values: {market.isna().sum()[lambda s: s > 0].to_dict()}")

    weather = raw.groupby([DISTRICT, DATE])[WEATHER_COLS].first()

    panels: Dict[str, pd.DataFrame] = {}
    for mt in MILK_TYPES:
        g = raw[(raw[MTYPE] == mt) & raw["row_valid"]]
        agg = g.groupby([DISTRICT, DATE]).agg(
            base=("Base_Price_INR", "sum"),
            net=("Net_Price_INR", "sum"),
            ts_kg=("TS_Kg", "sum"),
            qty_kg=("Qty_Kg", "sum"),
            qty_ltr=("Qty_Ltr", "sum"),
            fat_kg=("Fat_Kg", "sum"),
            snf_kg=("SNF_Kg", "sum"),
            n_centres=("centre", "nunique"),
        )
        grid = pd.MultiIndex.from_product([sorted(agg.index.get_level_values(0).unique()), calendar],
                                          names=[DISTRICT, DATE])
        agg = agg.reindex(grid)
        ok = agg["ts_kg"] >= cfg.min_ts_kg

        p = pd.DataFrame(index=grid)
        p["rate"] = (agg["base"] / agg["ts_kg"]).where(ok)          # Rs per kg TS, volume weighted
        p["rate_ltr"] = (agg["base"] / agg["qty_ltr"]).where(ok)     # Rs per litre (to compare with competitors)
        p["incentive_pct"] = ((agg["net"] / agg["base"] - 1) * 100).where(ok)   # agent incentive as % of base price
        p["qty_ltr"] = agg["qty_ltr"].where(ok)
        p["ts_kg"] = agg["ts_kg"]
        p["n_centres"] = agg["n_centres"]
        p["fat_pct"] = (agg["fat_kg"] / agg["qty_kg"] * 100).where(ok)
        p["snf_pct"] = (agg["snf_kg"] / agg["qty_kg"] * 100).where(ok)
        wk = pd.MultiIndex.from_arrays([p.index.get_level_values(DISTRICT), p.index.get_level_values(DATE)])
        p[WEATHER_COLS] = weather.reindex(wk).to_numpy()
        panels[mt] = p
    return panels, market


# --------------------------------------------------------- competitor data
COMP_FEATURES: List[str] = ["comp_gap_best", "comp_gap_med", "comp_best_ch1", "comp_best_ch3", "comp_med_ch1", "comp_n"]
_MILK_MAP = {"CM": "COW", "BM": "BUFFALO"}
_CENTRE_ALIAS = {"Bhalki": "Balki"}
_BRAND_ALIAS = {
    "heritage": "Heritage", "jersey": "Jersey", "jersey natural": "Jersey", "jersey heritage": "Heritage",
    "hatsun": "Hatsun", "country delight": "Country Delight", "coiuntrydelight": "Country Delight",
    "prabath": "Prabhat", "shivprasad": "Shivprasad", "shivaprasad": "Shivprasad",
    "cooperative": "Co-operative", "nilanga co-operative": "Co-operative", "mh co-operative": "Co-operative",
}


def load_competitor_quotes(cfg: Config, centre_district: Dict[str, str]) -> pd.DataFrame:
    """Clean competitor quotes (Net Rate, Rs per litre) and attach each centre's district.

    Cleaning follows milk_competitor.py: zero = no quote, plausible-range filter, >25% outlier filter,
    spelling variants merged, centres not in the main data ignored.
    """
    q = pd.read_excel(cfg.competitor_xlsx, sheet_name=cfg.competitor_sheet)
    if (q["UoM"].astype(str).str.lower().str.strip() != "inr per ltr").any():
        raise ValueError("Competitor file has a unit other than 'INR per Ltr'")
    q["centre"] = q["Name"].astype(str).str.strip().replace(_CENTRE_ALIAS)
    q[MTYPE] = q["Milk Type"].map(_MILK_MAP)
    q[DATE] = pd.to_datetime(q["Month"], format="%b %y").dt.to_period("M").dt.to_timestamp()
    key = q["Competitor Name"].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    q["brand"] = key.map(_BRAND_ALIAS).fillna(key.str.title())
    q = q.dropna(subset=[MTYPE])
    q = q[q["Net Rate"] > 0]
    bad = pd.Series(False, index=q.index)
    for mt, lo, hi in cfg.comp_valid_range:
        bad |= (q[MTYPE] == mt) & ((q["Net Rate"] < lo) | (q["Net Rate"] > hi))
    q = q[~bad]
    grp = q.groupby(["centre", MTYPE, DATE])["Net Rate"]
    med, cnt = grp.transform("median"), grp.transform("size")
    q = q[~((cnt >= 3) & ((q["Net Rate"] / med - 1).abs() > cfg.comp_outlier_rel))]
    q = q[q["centre"].isin(centre_district)].copy()
    q[DISTRICT] = q["centre"].map(centre_district)
    return q.reset_index(drop=True)


def competitor_panel(quotes: pd.DataFrame, mt: str, calendar: pd.DatetimeIndex, cfg: Config) -> pd.DataFrame:
    """(district, month) -> best / median competitor rate in Rs per litre and number of brands quoting.

    best = median over the district's centres of each centre's highest quote; med = median of all quotes.
    The panel ends at the last quote month (Mar-2026): later months stay blank, nothing is projected.
    """
    g = quotes[quotes[MTYPE] == mt]
    if g.empty:
        return pd.DataFrame(columns=["best", "med", "n"])
    centre_best = g.groupby([DISTRICT, "centre", DATE])["Net Rate"].max().groupby([DISTRICT, DATE]).median()
    a = pd.DataFrame({"best": centre_best,
                      "med": g.groupby([DISTRICT, DATE])["Net Rate"].median(),
                      "n": g.groupby([DISTRICT, DATE])["brand"].nunique().astype(float)})
    cal = calendar[calendar <= g[DATE].max()]
    a = a.reindex(pd.MultiIndex.from_product([sorted(a.index.get_level_values(0).unique()), cal], names=[DISTRICT, DATE]))
    return a.groupby(level=DISTRICT).ffill(limit=cfg.comp_gap_fill_months)   # causal fill


def add_competitor_features(d: pd.DataFrame, comp: pd.DataFrame | None) -> pd.DataFrame:
    """Competitor features at the origin month t (quotes of month t or earlier)."""
    if comp is None or comp.empty:
        for f in COMP_FEATURES:
            d[f] = np.nan
        return d
    c = comp.reindex(d.index)
    g = c.groupby(level=DISTRICT)
    d["comp_gap_best"] = d["rate_ltr"] - c["best"]      # our Rs/L minus best competitor Rs/L
    d["comp_gap_med"] = d["rate_ltr"] - c["med"]
    d["comp_best_ch1"] = c["best"] - g["best"].shift(1)
    d["comp_best_ch3"] = c["best"] - g["best"].shift(3)
    d["comp_med_ch1"] = c["med"] - g["med"].shift(1)
    d["comp_n"] = c["n"]
    return d


# ----------------------------------------------------------------- feature lists
# CORE = inputs shared by COW and BUFFALO (36).  Extras are milk-type specific, chosen by dairy logic:
#   COW     (+6): crossbred cows are SNF-driven, sensitive to heat + humidity and depend on bought concentrate feed.
#   BUFFALO (+10): buffalo milk is fat-rich (ghee / butter value), supply is strongly seasonal, heat stress is severe.
CORE: List[str] = [
    # price dynamics at the origin (16)
    "diff_1", "diff_3", "diff_6", "diff_mean6", "diff_std6", "accel", "dev_ma3", "dev_ma6", "dev_ma12", "yoy",
    "range12_pos", "rel_mean", "rel_ch3", "state_ch3", "ly_change", "rate_t",
    # calendar of the TARGET month (3)
    "tgt_month_sin", "tgt_month_cos", "tgt_summer",
    # weather already observed at the origin, district level (7)
    "temp_avg_t", "temp_max_t", "temp_min_t", "rain_t", "temp_max_3m", "rain_3m", "rain_6m",
    # milk quality at the origin (2)
    "fat_t", "snf_t",
    # market drivers, projected (8)
    "feed_ch3", "maize_ch3", "soy_meal_ch3", "smp_ch3", "ghee_ch3", "cpi_ch3", "oni_t", "oni_l3",
]
COW_EXTRA: List[str] = ["humid_t", "thi_t", "snf_ch3", "feed_maize_ratio", "smp_ch6", "incentive_pct"]
BUFFALO_EXTRA: List[str] = ["thi_t", "thi_3m", "fat_ch3", "fat_snf_ratio", "ghee_ch6", "butter_ch3", "ghee_smp_ratio",
                            "tgt_monsoon", "qty_ch3_pct", "monsoon_t"]
FULL_FEATURES_BY_TYPE: Dict[str, List[str]] = {"COW": CORE + COW_EXTRA, "BUFFALO": CORE + BUFFALO_EXTRA}
assert len(FULL_FEATURES_BY_TYPE["COW"]) == 42 and len(FULL_FEATURES_BY_TYPE["BUFFALO"]) == 46
assert all(len(set(v)) == len(v) for v in FULL_FEATURES_BY_TYPE.values())
# inputs built for the TARGET month (actual up to the origin, expected = same-month earlier-year average after it)
FORWARD_FEATURES: List[str] = ["temp_avg_t", "temp_max_t", "temp_min_t", "rain_t", "temp_max_3m", "rain_3m", "rain_6m",
                               "humid_t", "thi_t", "thi_3m", "fat_t", "snf_t", "fat_ch3", "snf_ch3", "fat_snf_ratio"]
# intervention flags (share of the months (t, t+h] inside the 2022-24 run-up / crash)
FLAG_FEATURES: List[str] = ["run_up_share", "crash_share"]
ALL_FEATURES: List[str] = list(dict.fromkeys(FULL_FEATURES_BY_TYPE["COW"] + FULL_FEATURES_BY_TYPE["BUFFALO"])) + FLAG_FEATURES

# ---- PRUNED feature lists -------------------------------------------------------------------------------------
# Inputs REMOVED per milk type.  The full lists above (COW 42 / BUFFALO 46) stay the reference.
#   COW     : nothing removed (42).  Pruning did not help cow on data not used for choosing, and made Apr-Jun 2026 worse.
#   BUFFALO : accel, ghee_smp_ratio removed (44).  Held-out Apr-Jun 2026: April 1.71 (<= 2.0), Apr-Jun 4.20 (<= 5.0).
# To change the lists, edit ONLY this dictionary.  selected_features.json is no longer read.
DROP_BY_TYPE: Dict[str, List[str]] = {
    "COW": [],
    "BUFFALO": ["accel", "ghee_smp_ratio"],
}
SELECTED_JSON = HERE / "selected_features.json"      # kept for reference only; NOT read


def _apply_drops() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for mt, full in FULL_FEATURES_BY_TYPE.items():
        drop = list(DROP_BY_TYPE.get(mt, []))
        bad = [f for f in drop if f not in full]
        if bad:
            raise ValueError(f"DROP_BY_TYPE[{mt}] has inputs that are not in the full list: {bad}")
        out[mt] = [f for f in full if f not in set(drop)]            # canonical order kept
        if len(out[mt]) < 8:
            raise ValueError(f"too few inputs left for {mt}: {len(out[mt])}")
    return out


FEATURES_BY_TYPE: Dict[str, List[str]] = _apply_drops()
USING_SELECTED = any(DROP_BY_TYPE.values())
if SELECTED_JSON.exists():
    log.warning("%s exists but is IGNORED - the lists come from DROP_BY_TYPE in step1_data_features.py", SELECTED_JSON.name)

# kind = how the input is known at forecast time
#   price    : recomputed from the (actual or already-forecast) rate history  -> usable inside the recursion
#   calendar : known in advance for the target month
#   observed : weather / quality / supply / incentive AS OBSERVED (never forecast)
#   market   : market / ONI series projected from the last value + k x average of the last 6 monthly changes
FEATURE_INFO: Dict[str, Tuple[str, str, str]] = {      # name -> (group, kind, definition)
    "diff_1": ("Price momentum", "price", "rate(t) - rate(t-1)"),
    "diff_3": ("Price momentum", "price", "rate(t) - rate(t-3)"),
    "diff_6": ("Price momentum", "price", "rate(t) - rate(t-6)"),
    "diff_mean6": ("Price momentum", "price", "average of the last 6 monthly changes"),
    "diff_std6": ("Price momentum", "price", "standard deviation of the last 6 monthly changes (how jumpy the rate is)"),
    "accel": ("Price momentum", "price", "diff_1 minus the previous month's change (is the move speeding up?)"),
    "dev_ma3": ("Price vs own history", "price", "rate(t) minus its 3-month average"),
    "dev_ma6": ("Price vs own history", "price", "rate(t) minus its 6-month average"),
    "dev_ma12": ("Price vs own history", "price", "rate(t) minus its 12-month average (mean reversion)"),
    "yoy": ("Price vs own history", "price", "rate(t) - rate(t-12)"),
    "range12_pos": ("Price vs own history", "price", "where rate(t) sits between its 12-month low (0) and high (1)"),
    "rate_t": ("Price vs own history", "price", "the rate level at the origin month"),
    "rel_mean": ("Price vs other districts", "price", "rate(t) minus the average rate of all districts in month t"),
    "rel_ch3": ("Price vs other districts", "price", "change in rel_mean over 3 months (district catching up / falling behind)"),
    "state_ch3": ("Price vs other districts", "price", "average 3-month change of all districts (state-wide momentum)"),
    "ly_change": ("Seasonal pattern", "price", "same window one year earlier: rate(t+h-12) - rate(t-12)"),
    "tgt_month_sin": ("Seasonal pattern", "calendar", "SIN(2*PI*month/12) of the month being forecast"),
    "tgt_month_cos": ("Seasonal pattern", "calendar", "COS(2*PI*month/12) of the month being forecast"),
    "tgt_summer": ("Seasonal pattern", "calendar", "1 if the month being forecast is Mar-Jun (lean / summer season) else 0"),
    "tgt_monsoon": ("Seasonal pattern", "calendar", "1 if the month being forecast is Jul-Sep else 0"),
    "temp_avg_t": ("Weather", "observed", "district average temperature"),
    "temp_max_t": ("Weather", "observed", "district average daily maximum temperature"),
    "temp_min_t": ("Weather", "observed", "district average daily minimum temperature"),
    "rain_t": ("Weather", "observed", "district rainfall in the month (mm)"),
    "temp_max_3m": ("Weather", "observed", "3-month average of the maximum temperature"),
    "rain_3m": ("Weather", "observed", "rainfall of the last 3 months"),
    "rain_6m": ("Weather", "observed", "rainfall of the last 6 months (fodder availability)"),
    "humid_t": ("Heat stress", "observed", "district average relative humidity %"),
    "thi_t": ("Heat stress", "observed", "Temperature-Humidity Index = 0.8*T + RH/100*(T-14.4) + 46.4 (above ~72 cattle are stressed)"),
    "thi_3m": ("Heat stress", "observed", "3-month average of THI"),
    "fat_t": ("Milk quality", "observed", "fat % of the milk procured"),
    "snf_t": ("Milk quality", "observed", "SNF % of the milk procured"),
    "fat_ch3": ("Milk quality", "observed", "fat % now minus fat % 3 months ago"),
    "snf_ch3": ("Milk quality", "observed", "SNF % now minus SNF % 3 months ago"),
    "fat_snf_ratio": ("Milk quality", "observed", "fat % divided by SNF % (richness of the milk)"),
    "qty_ch3_pct": ("Supply & incentive", "observed", "% change of litres procured vs 3 months ago"),
    "incentive_pct": ("Supply & incentive", "observed", "(Net price / Base price - 1) x 100: agent incentive paid on top of base"),
    "feed_ch3": ("Feed & commodity costs", "market", "3-month % change of cattle-feed bag price"),
    "maize_ch3": ("Feed & commodity costs", "market", "3-month % change of maize (Latur)"),
    "soy_meal_ch3": ("Feed & commodity costs", "market", "3-month % change of soybean meal (Latur)"),
    "feed_maize_ratio": ("Feed & commodity costs", "market", "feed bag price / maize price (feed margin over raw material)"),
    "smp_ch3": ("Dairy product prices", "market", "3-month % change of skimmed milk powder"),
    "smp_ch6": ("Dairy product prices", "market", "6-month % change of skimmed milk powder"),
    "ghee_ch3": ("Dairy product prices", "market", "3-month % change of ghee (Mumbai)"),
    "ghee_ch6": ("Dairy product prices", "market", "6-month % change of ghee (Mumbai)"),
    "butter_ch3": ("Dairy product prices", "market", "3-month % change of butter (Maharashtra)"),
    "ghee_smp_ratio": ("Dairy product prices", "market", "ghee price / SMP price (value of fat relative to SNF)"),
    "cpi_ch3": ("Macro & climate", "market", "3-month % change of CPI Maharashtra"),
    "oni_t": ("Macro & climate", "market", "ONI (El Nino / La Nina) anomaly at the target month, projected"),
    "oni_l3": ("Macro & climate", "market", "ONI anomaly 3 months before the target month, projected"),
    "monsoon_t": ("Macro & climate", "market", "Maharashtra monsoon impact index, projected"),
}
FEATURE_INFO["run_up_share"] = ("Intervention flags", "flag", "share of the months t+1..t+h inside the 2022-24 price run-up (0 for every forecast made after it)")
FEATURE_INFO["crash_share"] = ("Intervention flags", "flag", "share of the months t+1..t+h inside the 2023-24 price crash (0 for every forecast made after it)")
for _f in FORWARD_FEATURES:        # kind 'forward': value for the TARGET month, expected (same month, earlier years) where not yet observed
    _g, _k, _t = FEATURE_INFO[_f]
    FEATURE_INFO[_f] = (_g, "forward", _t + "  [FOR THE TARGET MONTH: actual up to the origin, same-month earlier-year average after it]")
assert set(ALL_FEATURES) == set(FEATURE_INFO), set(ALL_FEATURES) ^ set(FEATURE_INFO)
FEATURE_GROUP = {k: v[0] for k, v in FEATURE_INFO.items()}
FEATURE_KIND = {k: v[1] for k, v in FEATURE_INFO.items()}


def spec_for(mt: str, full: bool = False, forward: bool = True, flags: bool = True) -> Dict[str, List[str]]:
    """Feature lists of one milk type, split by how the input is known at forecast time.
    full=True gives the complete 42 / 46 list (used by the feature-selection script), otherwise the active list.
    forward=False treats the weather / quality inputs as 'observed at the origin' (old version); flags=True adds the 2 intervention flags."""
    base = FULL_FEATURES_BY_TYPE[mt] if full else FEATURES_BY_TYPE[mt]
    return spec_from(list(base) + (FLAG_FEATURES if flags else []), forward)


def spec_from(f: List[str], forward: bool = True) -> Dict[str, List[str]]:
    def kind(x: str) -> str:
        k = FEATURE_KIND[x]
        return "observed" if (k == "forward" and not forward) else k
    return {"all": list(f),
            "price_cal": [x for x in f if kind(x) in ("price", "calendar")],
            "observed": [x for x in f if kind(x) == "observed"],
            "market": [x for x in f if kind(x) == "market"],
            "table_fed": [x for x in f if kind(x) in ("forward", "flag")]}   # read straight from the training table, for the target month


# ----------------------------------------------------------------- price features (wide)
def calendar_features(month: pd.Series | pd.DatetimeIndex) -> Dict[str, np.ndarray]:
    m = np.asarray(pd.DatetimeIndex(month).month)
    return {"tgt_month_sin": np.sin(2 * np.pi * m / 12), "tgt_month_cos": np.cos(2 * np.pi * m / 12),
            "tgt_summer": ((m >= 3) & (m <= 6)).astype(float), "tgt_monsoon": ((m >= 7) & (m <= 9)).astype(float)}


def price_block(R: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Price features at EVERY date of the wide rate table R (dates x districts).

    One function feeds both the training table and the recursion (step 2 takes the last row), so the two can never
    drift apart.  ly_change and the calendar inputs depend on the horizon / target month and are added by the caller."""
    step = R.diff()
    out = {"rate_t": R, "diff_1": R - R.shift(1), "diff_3": R - R.shift(3), "diff_6": R - R.shift(6)}
    out["diff_mean6"] = step.rolling(6, min_periods=3).mean()
    out["diff_std6"] = step.rolling(6, min_periods=3).std()
    out["accel"] = step - step.shift(1)
    out["dev_ma3"] = R - R.rolling(3, min_periods=2).mean()
    out["dev_ma6"] = R - R.rolling(6, min_periods=3).mean()
    out["dev_ma12"] = R - R.rolling(12, min_periods=6).mean()
    out["yoy"] = R - R.shift(12)
    hi, lo = R.rolling(12, min_periods=6).max(), R.rolling(12, min_periods=6).min()
    out["range12_pos"] = ((R - lo) / (hi - lo)).where(hi > lo)
    rel = R.sub(R.mean(axis=1), axis=0)
    out["rel_mean"] = rel
    out["rel_ch3"] = rel - rel.shift(3)
    state3 = out["diff_3"].mean(axis=1)
    out["state_ch3"] = pd.DataFrame({c: state3 for c in R.columns})
    return out


def _long(W: pd.DataFrame, index: pd.MultiIndex) -> np.ndarray:
    """wide (dates x districts) -> values aligned to the (district, month) panel index."""
    s = W.unstack()
    s.index.set_names([DISTRICT, DATE], inplace=True)
    return s.reindex(index).to_numpy()


def origin_features(panel: pd.DataFrame, market: pd.DataFrame, comp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Features that depend only on the ORIGIN month t (everything dated t or earlier)."""
    d = panel.sort_index().copy()
    R = d["rate"].unstack(DISTRICT).sort_index()
    for name, W in price_block(R).items():
        d[name] = _long(W, d.index)
    dates = d.index.get_level_values(DATE)

    # weather that ALREADY happened (district level) and heat stress
    d["temp_avg_t"], d["temp_max_t"], d["temp_min_t"], d["rain_t"], d["humid_t"] = (
        d["temp_avg"], d["temp_max"], d["temp_min"], d["rain"], d["humid"])
    d["temp_max_3m"] = _by_district(d["temp_max"], lambda s: s.rolling(3, min_periods=2).mean())
    d["rain_3m"] = _by_district(d["rain"], lambda s: s.rolling(3, min_periods=2).sum())
    d["rain_6m"] = _by_district(d["rain"], lambda s: s.rolling(6, min_periods=3).sum())
    d["thi_t"] = 0.8 * d["temp_avg"] + d["humid"] / 100.0 * (d["temp_avg"] - 14.4) + 46.4
    d["thi_3m"] = _by_district(d["thi_t"], lambda s: s.rolling(3, min_periods=2).mean())

    # milk quality at the origin (fat and SNF %) and its 3-month trend
    d["fat_t"], d["snf_t"] = d["fat_pct"], d["snf_pct"]
    d["fat_ch3"] = d["fat_pct"] - d["fat_pct"].groupby(level=DISTRICT).shift(3)
    d["snf_ch3"] = d["snf_pct"] - d["snf_pct"].groupby(level=DISTRICT).shift(3)
    d["fat_snf_ratio"] = d["fat_pct"] / d["snf_pct"]

    # supply trend and agent incentive (incentive_pct is already in the panel)
    q = d["qty_ltr"]
    d["qty_ch3_pct"] = (q / q.groupby(level=DISTRICT).shift(3) - 1) * 100

    # market / macro drivers at month t (actual values; the training table replaces them with projections)
    mf = projected_market_features(market, 0).reindex(dates)
    for c in mf.columns:
        d[c] = mf[c].to_numpy()
    return add_competitor_features(d, comp) if comp is not None else d


def _by_district(s: pd.Series, fn) -> pd.Series:
    return s.groupby(level=DISTRICT).transform(fn)


def projected_market_features(market: pd.DataFrame, h: int, window: int = 6) -> pd.DataFrame:
    """Market / ONI inputs for a forecast of month t+h, built ONLY from values dated t or earlier.

    Future values are not known, so each series is projected:   proj(t+k) = value(t) + k x drift,
    drift = average of the last ``window`` monthly changes up to t  (= (value(t) - value(t-window)) / window).
    Inputs for the TARGET month t+h (months up to t are actual values, months after t are projected):
        <series>_ch3 = % change over the 3 months ending at t+h          [feed, maize, soy_meal, smp, ghee, cpi, butter]
        smp_ch6, ghee_ch6 = % change over the 6 months ending at t+h
        ghee_smp_ratio = ghee / smp ; feed_maize_ratio = feed / maize     (at t+h)
        oni_t, oni_l3 = ONI at t+h and at t+h-3 ; monsoon_t = monsoon index at t+h
    Same recipe in training and forecasting.
    """
    m = market[MARKET_COLS].astype(float)
    drift = (m - m.shift(window)) / window

    def proj(k: int) -> pd.DataFrame:            # value at month t+k, seen from origin t
        return m + k * drift if k > 0 else m.shift(-k)

    now, back3, back6 = proj(h), proj(h - 3), proj(h - 6)
    out = pd.DataFrame(index=market.index)
    for c in MARKET_PCT:
        out[f"{c}_ch3"] = (now[c] / back3[c] - 1) * 100
    for c in ("smp", "ghee"):
        out[f"{c}_ch6"] = (now[c] / back6[c] - 1) * 100
    out["ghee_smp_ratio"] = now["ghee"] / now["smp"]
    out["feed_maize_ratio"] = now["feed"] / now["maize"]
    out["oni_t"], out["oni_l3"], out["monsoon_t"] = now["oni"], back3["oni"], now["monsoon"]
    return out


# ----------------------------------------------------------------- TASK 1: forward-looking inputs
FWD_SERIES = ["temp_avg", "temp_max", "temp_min", "rain", "humid", "fat_pct", "snf_pct"]


def expected_wide(A: pd.DataFrame, min_years: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """EXPECTED value of every (month, district) = average of the SAME calendar month in strictly EARLIER years.

    A is a wide table (months x districts) on a contiguous monthly index.  If a district has fewer than ``min_years``
    earlier years for that month, the average of ALL districts' earlier years of that month is used instead (fallback).
    Returns (expected, fallback_flag).  Only years before the month's own year are ever used."""
    mon = A.index.month
    own = A.groupby(mon).transform(lambda s: s.expanding().mean().shift(1))
    n_prev = A.notna().astype(float).groupby(mon).transform(lambda s: s.cumsum().shift(1))
    own = own.where(n_prev >= min_years)
    row_sum, row_cnt = A.sum(axis=1), A.notna().sum(axis=1).astype(float)
    ps = row_sum.groupby(mon).transform(lambda s: s.cumsum().shift(1))
    pc = row_cnt.groupby(mon).transform(lambda s: s.cumsum().shift(1))
    pooled = ps / pc.where(pc > 0)
    fb = own.isna()
    E = own.copy()
    for c in E.columns:
        E[c] = E[c].fillna(pooled)
    return E, fb


def forward_inputs_wide(panel: pd.DataFrame, cfg: Config):
    """Actual (A) and expected (E) wide tables for the 7 forward series on a calendar extended by max_horizon months."""
    cal = panel.index.get_level_values(DATE)
    ext = pd.date_range(cal.min(), cal.max() + pd.DateOffset(months=cfg.max_horizon), freq="MS")
    Aw, Ew, FB = {}, {}, {}
    for name in FWD_SERIES:
        A = panel[name].unstack(DISTRICT).sort_index().reindex(ext)
        Aw[name] = A
        Ew[name], FB[name] = expected_wide(A, cfg.fwd_min_years)
    return Aw, Ew, FB


def _blend(A: pd.DataFrame, E: pd.DataFrame, h: int, k: int) -> pd.DataFrame:
    """Value of a series in month (t + h - k) seen from origin t, indexed by t.
    Month <= t : the actual (expected only if the actual is missing).   Month > t : the expected value."""
    j = h - k
    if j <= 0:
        return A.shift(-j).fillna(E.shift(-j))
    return E.shift(-j)


def _roll(frames: List[pd.DataFrame], how: str, min_periods: int) -> pd.DataFrame:
    arr = np.stack([f.to_numpy(dtype=float) for f in frames])
    cnt = np.sum(~np.isnan(arr), axis=0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        val = np.nansum(arr, axis=0) if how == "sum" else np.nanmean(arr, axis=0)
    return pd.DataFrame(np.where(cnt >= min_periods, val, np.nan), index=frames[0].index, columns=frames[0].columns)


def forward_block(Aw: Dict[str, pd.DataFrame], Ew: Dict[str, pd.DataFrame], h: int) -> Dict[str, pd.DataFrame]:
    """The FORWARD_FEATURES for a forecast of month t+h, as wide tables (origin t x district).
    Every derived input is recomputed from the monthly series (actual up to t, expected after t)."""
    b = lambda name, k: _blend(Aw[name], Ew[name], h, k)
    thi = lambda k: 0.8 * b("temp_avg", k) + b("humid", k) / 100.0 * (b("temp_avg", k) - 14.4) + 46.4
    out = {"temp_avg_t": b("temp_avg", 0), "temp_max_t": b("temp_max", 0), "temp_min_t": b("temp_min", 0),
           "rain_t": b("rain", 0), "humid_t": b("humid", 0), "thi_t": thi(0),
           "temp_max_3m": _roll([b("temp_max", k) for k in range(3)], "mean", 2),
           "rain_3m": _roll([b("rain", k) for k in range(3)], "sum", 2),
           "rain_6m": _roll([b("rain", k) for k in range(6)], "sum", 3),
           "thi_3m": _roll([thi(k) for k in range(3)], "mean", 2),
           "fat_t": b("fat_pct", 0), "snf_t": b("snf_pct", 0),
           "fat_ch3": b("fat_pct", 0) - b("fat_pct", 3), "snf_ch3": b("snf_pct", 0) - b("snf_pct", 3)}
    out["fat_snf_ratio"] = out["fat_t"] / out["snf_t"]
    return out


# ----------------------------------------------------------------- TASK 2: intervention flags
def swing_windows(panel: pd.DataFrame, cfg: Config) -> Dict:
    """Run-up and crash windows of the 2022-24 price swing, found from the data (business must validate the dates).

    Reference rate = average of the districts that have a rate in EVERY month of the search window.
    run_up = swing_start .. peak (highest month in swing_peak_search);  crash = month after the peak .. trough
    (lowest month between the peak and swing_end_search).  Recovery after the trough is NOT flagged (new normal)."""
    r = panel["rate"].unstack(DISTRICT).sort_index()
    win = r.loc[cfg.swing_start:cfg.swing_end_search]
    full = [c for c in win.columns if win[c].notna().all()]
    if not full:
        return {}
    S = r[full].mean(axis=1).loc[:cfg.swing_end_search]
    peak = S.loc[cfg.swing_peak_search[0]:cfg.swing_peak_search[1]].idxmax()
    trough = S.loc[peak:].idxmin()
    run_up = (pd.Timestamp(cfg.swing_start), peak)
    crash = (peak + pd.DateOffset(months=1), trough)
    src = {"run_up": "found from the data", "crash": "found from the data"}
    if cfg.run_up_override:
        run_up, src["run_up"] = (pd.Timestamp(cfg.run_up_override[0]), pd.Timestamp(cfg.run_up_override[1])), "business override"
    if cfg.crash_override:
        crash, src["crash"] = (pd.Timestamp(cfg.crash_override[0]), pd.Timestamp(cfg.crash_override[1])), "business override"
    S_all = r[full].mean(axis=1)
    return {"run_up": run_up, "crash": crash, "reference_districts": full, "peak_rate": float(S[peak]), "trough_rate": float(S[trough]),
            "start_rate": float(S_all.get(run_up[0], np.nan)), "ref_series": S_all, "source": src}


def intervention_table(panels: Dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """One row per milk type and window: dates, length, reference-rate move and average monthly change, plus the
    normal-period monthly change for comparison.  This is the table the business validates (dates and explanation)."""
    rows = []
    for mt, p in panels.items():
        w = swing_windows(p, cfg)
        if not w:
            continue
        S = w["ref_series"]
        step = S.diff()
        def avg(lo, hi):
            x = step.loc[lo:hi].dropna()
            return float(x.mean()) if len(x) else np.nan
        for name, label in (("run_up", "Run-up"), ("crash", "Crash")):
            lo, hi = w[name]
            before = S.loc[:lo - pd.DateOffset(months=1)].dropna()
            r0 = float(before.iloc[-1]) if len(before) else float(S.get(lo, np.nan))
            r1 = float(S.get(hi, np.nan))
            rows.append({"Milk Type": mt.title(), "Window": label, "Start": lo.date(), "End": hi.date(),
                         "Months": len(pd.date_range(lo, hi, freq="MS")), "Dates source": w["source"][name],
                         "Reference rate before (Rs/kg TS)": r0, "Reference rate at end (Rs/kg TS)": r1,
                         "Move (Rs/kg TS)": r1 - r0, "Move %": (r1 / r0 - 1) * 100 if r0 else np.nan,
                         "Avg monthly change in window": avg(lo, hi),
                         "Avg monthly change outside both windows": float(step[~((step.index >= w["run_up"][0]) & (step.index <= w["crash"][1]))].dropna().mean()),
                         "Reference districts": ", ".join(w["reference_districts"])})
    return pd.DataFrame(rows)


def flag_shares(win: Dict, ext: pd.DatetimeIndex, h: int) -> Dict[str, pd.Series]:
    """Share of the months t+1..t+h inside each window, indexed by the origin month t."""
    out = {}
    for name, key in (("run_up_share", "run_up"), ("crash_share", "crash")):
        ind = pd.Series(0.0, index=ext)
        if win:
            lo, hi = win[key]
            ind[(ind.index >= lo) & (ind.index <= hi)] = 1.0
        out[name] = sum(ind.shift(-j) for j in range(1, h + 1)).fillna(0.0) / h
    return out


def movement_category(run_up_share: float, crash_share: float, price_change: float | None) -> str:
    """Interpretable movement label for a single observed district-month row."""
    if price_change is None or pd.isna(price_change):
        return "missing_change"
    if abs(price_change) <= 1e-6:
        return "flat"
    if run_up_share > 0 and price_change > 0:
        return "run_up_rise"
    if crash_share > 0 and price_change < 0:
        return "crash_fall"
    if price_change > 0:
        return "recovery_or_rise"
    if price_change < 0:
        return "decline"
    return "flat"


def confidence_score(run_up_share: float, crash_share: float, prior_rate: float | None, price_change: float | None) -> float:
    """Deterministic, reproducible confidence score based on observed evidence only.

    This is not a probability or model score; it is a transparency metric that rewards complete monthly observations,
    window membership, and a valid change calculation.  It is intentionally simple and fixed so it can be reproduced exactly."""
    score = 50.0
    if run_up_share > 0 or crash_share > 0:
        score += 25.0
    if prior_rate is not None and not pd.isna(prior_rate):
        score += 15.0
    if price_change is not None and not pd.isna(price_change):
        score += 10.0
    return round(float(np.clip(score, 0.0, 100.0)), 1)


def build_run_up_crash_analysis(panels: Dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """Row-level analysis for the 2022-2024 run-up / crash period.  Only observed district-month rows are reported."""
    start = pd.Timestamp("2022-01-01")
    end = pd.Timestamp("2024-12-01")
    rows = []
    for mt, panel in panels.items():
        win = swing_windows(panel, cfg)
        if not win:
            continue
        for (district, month), rate in panel["rate"].dropna().items():
            if month < start or month > end:
                continue
            prev_key = (district, month - pd.DateOffset(months=1))
            prev_rate = panel.loc[prev_key, "rate"] if prev_key in panel.index else np.nan
            price_change = np.nan if pd.isna(prev_rate) else float(rate - prev_rate)
            pct_change = np.nan if pd.isna(prev_rate) or prev_rate == 0 else float((rate / prev_rate - 1.0) * 100.0)
            run_up_share = 1.0 if win["run_up"][0] <= month <= win["run_up"][1] else 0.0
            crash_share = 1.0 if win["crash"][0] <= month <= win["crash"][1] else 0.0
            categ = movement_category(run_up_share, crash_share, price_change)
            ref_rate = float(win["ref_series"].get(month, np.nan)) if month in win["ref_series"].index else np.nan
            observed = (
                f"Observed rate for {district} in {month:%b %Y} was {rate:.2f} Rs/kg TS. "
                f"Previous month rate was {prev_rate:.2f} Rs/kg TS." if pd.notna(prev_rate) else
                f"Observed rate for {district} in {month:%b %Y} was {rate:.2f} Rs/kg TS; no prior-month rate was available."
            )
            if pd.notna(ref_rate):
                observed += f" The reference rate for that month was {ref_rate:.2f} Rs/kg TS."
            association = (
                "This month is inside the detected run-up window, so the row is statistically aligned with the broader 2022-24 rise."
                if run_up_share > 0 else
                "This month is inside the detected crash window, so the row is statistically aligned with the broader 2022-24 drop."
                if crash_share > 0 else
                "This month lies outside the detected 2022-24 swing windows; the row is treated as a normal-period observation for comparison."
            )
            possible_causes = (
                "Possible drivers may include broader market conditions or weather conditions, but this row alone does not establish a causal mechanism."
            )
            limitations = (
                "This is a descriptive monthly analysis only: it does not prove cause, the swing windows are data-detected reference windows, "
                "missing prior-month rates are excluded from direct change calculations, and the score is a reproducible evidence-quality metric, "
                "not a statistical confidence probability."
            )
            rows.append({
                "month": month,
                "district": district,
                "milk_type": mt,
                "rate_rs_per_kg_ts": float(rate),
                "prior_month_rate_rs_per_kg_ts": float(prev_rate) if pd.notna(prev_rate) else np.nan,
                "price_change_rs_per_kg_ts": float(price_change) if pd.notna(price_change) else np.nan,
                "pct_change_vs_previous_month": float(pct_change) if pd.notna(pct_change) else np.nan,
                "run_up_share": round(float(run_up_share), 6),
                "crash_share": round(float(crash_share), 6),
                "movement_category": categ,
                "reference_rate_rs_per_kg_ts": float(ref_rate) if pd.notna(ref_rate) else np.nan,
                "window_label": "Run-up" if run_up_share > 0 else "Crash" if crash_share > 0 else "Outside swing",
                "confidence_score": confidence_score(run_up_share, crash_share, prev_rate, price_change),
                "observed_facts": observed,
                "statistical_association": association,
                "possible_causes": possible_causes,
                "limitations": limitations,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["milk_type", "district", "month"]).reset_index(drop=True)


def write_run_up_crash_analysis(panels: Dict[str, pd.DataFrame], cfg: Config, path: Path) -> pd.DataFrame:
    """Write the row-level 2022-2024 run-up / crash analysis and the supporting summary sheets to Excel."""
    analysis = build_run_up_crash_analysis(panels, cfg)
    windows = intervention_table(panels, cfg)
    missing = []
    for mt, panel in panels.items():
        idx = panel["rate"].dropna().index
        complete = pd.MultiIndex.from_product([sorted(panel.index.get_level_values(DISTRICT).unique()), pd.date_range(start="2022-01-01", end="2024-12-01", freq="MS")], names=[DISTRICT, DATE])
        present = pd.Index([(d, m) for d, m in idx if pd.Timestamp(m) >= pd.Timestamp("2022-01-01") and pd.Timestamp(m) <= pd.Timestamp("2024-12-01")])
        if len(present) == 0:
            continue
        months = pd.date_range(start="2022-01-01", end="2024-12-01", freq="MS")
        for district in sorted(panel.index.get_level_values(DISTRICT).unique()):
            for month in months:
                key = (district, month)
                if key not in panel.index:
                    missing.append({"milk_type": mt, "district": district, "month": month, "missing_rate": True, "reason": "district-month no valid rate"})
                elif pd.isna(panel.loc[key, "rate"]):
                    missing.append({"milk_type": mt, "district": district, "month": month, "missing_rate": True, "reason": "valid rate missing"})
    missing_df = pd.DataFrame(missing)
    method_df = pd.DataFrame({
        "Field": ["Scope", "Window detection", "Observed rows", "Change calculation", "Confidence score", "Interpretation", "Limitations"],
        "Definition": [
            "2022-01 to 2024-12 district-month rows, split by COW and BUFFALO.",
            "Reference windows are detected from the reference series and then validated by business context.",
            "Only rows with a valid observed rate are included in the row-level analysis.",
            "Price change = current month rate minus previous month rate; percent change is relative to the previous month.",
            "Score is deterministic: 50 base + 25 if the row is inside the detected swing + 15 if there is a prior month + 10 if a valid change exists; values are capped at 100.",
            "Observed facts are listed separately from the statistical association and possible causes; no causal claim is made.",
            "The analysis is descriptive only; it is not forecast or causal analysis and excludes 2025-2026 backtesting and forecast bands."
        ]
    })
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        analysis.to_excel(xw, sheet_name="Row_Level_Analysis", index=False)
        windows.to_excel(xw, sheet_name="Window_Summary", index=False)
        missing_df.to_excel(xw, sheet_name="Missing_Data_Summary", index=False)
        method_df.to_excel(xw, sheet_name="Methodology", index=False)
    analysis.to_csv(path.with_suffix(".csv"), index=False)
    return analysis


def build_training_table(panel: pd.DataFrame, market: pd.DataFrame, cfg: Config,
                         comp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stack horizons h = 1..max_horizon: one row per (district, origin month, horizon).

    Columns: all candidate features + ``origin``, ``horizon``, ``target_month``, ``rate_t`` (last known rate),
    ``rate_actual`` (rate at t+h, NaN if unknown) and ``target`` (= rate_actual - rate_t).
    A milk type's model uses only its own list (FEATURES_BY_TYPE).
    """
    d = origin_features(panel, market, comp)
    r = d["rate"]
    g = r.groupby(level=DISTRICT)
    Aw, Ew, _ = forward_inputs_wide(panel, cfg)
    win = swing_windows(panel, cfg)
    ext = Aw["rain"].index
    frames = []
    for h in range(1, cfg.max_horizon + 1):
        x = d.copy()
        x["horizon"] = h
        x["origin"] = x.index.get_level_values(DATE)
        if cfg.project_market:                              # market inputs projected to the target month
            pm = projected_market_features(market, h, cfg.proj_window).reindex(x["origin"])
            for c in pm.columns:
                x[c] = pm[c].to_numpy()
        x["target_month"] = x["origin"] + pd.DateOffset(months=h)
        x["rate_actual"] = g.shift(-h)                      # rate at t+h
        x["target"] = x["rate_actual"] - x["rate_t"]
        # same window one year earlier: Rate(t+h-12) - Rate(t-12). Known at t because h <= 12.
        x["ly_change"] = g.shift(12 - h) - g.shift(12)
        for k, v in calendar_features(x["target_month"]).items():
            x[k] = v
        if cfg.forward_inputs:                              # TASK 1: weather / fat / SNF of the TARGET month (actual <= t, expected > t)
            for name, W in forward_block(Aw, Ew, h).items():
                x[name] = _long(W, x.index)
        for name, sh in flag_shares(win, ext, h).items():   # TASK 2: intervention flags
            x[name] = sh.reindex(x["origin"]).to_numpy()
        frames.append(x)
    t = pd.concat(frames).reset_index().rename(columns={DATE: "origin_idx"}).drop(columns="origin_idx")
    t = t[t["rate_t"].notna() & t["diff_1"].notna()]       # need a rate now and one month earlier
    if cfg.train_from_target:                               # first year has no earlier year to average -> drop those targets
        t = t[t["target_month"] >= pd.Timestamp(cfg.train_from_target)]
    return t.sort_values([DISTRICT, "origin", "horizon"]).reset_index(drop=True)


# ----------------------------------------------------------------- TASK 1: weather SCENARIOS (band -> forecast)
WEATHER_LIMITS = {"rain": (0.0, None), "humid": (0.0, 100.0)}      # physical limits applied after a band shift


def scenario_table(panel: pd.DataFrame, cfg: Config, table: pd.DataFrame, origin: pd.Timestamp,
                   shifts: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Copy of the training table in which the forward weather inputs of the rows at ``origin`` are rebuilt from
    EXPECTED + SHIFT instead of EXPECTED.

    shifts : {weather series -> wide frame (months x districts) of additive shifts}, e.g. the lower / upper edge of the weather band.
    Only months AFTER the origin are shifted (months up to the origin are actual).  Rows of every other origin are untouched, so the
    models fitted from this table are identical to the base ones - only the inputs of the forecast rows change.
    """
    Aw, Ew, _ = forward_inputs_wide(panel, cfg)
    Es = {k: v.copy() for k, v in Ew.items()}
    for name, S in shifts.items():
        S = S.reindex(index=Es[name].index, columns=Es[name].columns).fillna(0.0)
        S.loc[S.index <= origin] = 0.0
        E = Es[name] + S
        lo, hi = WEATHER_LIMITS.get(name, (None, None))
        Es[name] = E.clip(lower=lo, upper=hi)
    out = table.copy()
    for h in range(1, cfg.max_horizon + 1):
        m = (out["origin"] == origin) & (out["horizon"] == h)
        if not m.any():
            continue
        dist = out.loc[m, DISTRICT]
        for name, W in forward_block(Aw, Es, h).items():
            out.loc[m, name] = W.loc[origin].reindex(dist).to_numpy()
    return out


def assert_no_leakage(train: pd.DataFrame, origin: pd.Timestamp) -> None:
    """Every training row's target month must be known at ``origin``."""
    if len(train) and train["target_month"].max() > origin:
        raise AssertionError(f"leakage: training reaches {train['target_month'].max():%b %Y} > origin {origin:%b %Y}")


def forecastable_districts(panel: pd.DataFrame, origin: pd.Timestamp, n_months: int) -> List[str]:
    """Districts with a rate in each of the last ``n_months`` months up to ``origin``."""
    window = pd.date_range(end=origin, periods=n_months, freq="MS")
    r = panel["rate"].unstack(DATE).reindex(columns=window)
    return sorted(r.index[r.notna().all(axis=1)])


# ------------------------------------------------------------- excel output
def write_target_workbook(cfg: Config, panels: Dict[str, pd.DataFrame], out_path: Path) -> None:
    """Copy the source workbook, add Rate_TS_Base as a FORMULA column and a district-month sheet."""
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = load_workbook(find_data_file(cfg.data_xlsx))
    ws = wb[cfg.data_sheet]
    headers = {c.value: c.column for c in ws[1]}
    col_ts, col_base = get_column_letter(headers["TS_Kg"]), get_column_letter(headers["Base_Price_INR"])
    new_col = ws.max_column + 1
    head = ws.cell(row=1, column=new_col, value="Rate_TS_Base_INR_per_kgTS")
    head.font, head.fill = Font(name="Arial", bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1F4E78")
    for r in range(2, ws.max_row + 1):
        ws.cell(row=r, column=new_col,
                value=f'=IF(AND({col_ts}{r}>0,{col_base}{r}>0),{col_base}{r}/{col_ts}{r},"")').number_format = "0.00"
    ws.column_dimensions[get_column_letter(new_col)].width = 26

    # district-month sheet: SUMIFS formulas (sum of base / sum of TS over valid rows)
    dm = wb.create_sheet("District_Month_Rate")
    cols = ["Month", "District", "Milk_Type", "Base_Price_INR (sum)", "TS_Kg (sum)", "Rate_TS (Rs/kg TS)", "Centres"]
    for i, c in enumerate(cols, 1):
        cell = dm.cell(row=1, column=i, value=c)
        cell.font, cell.fill = Font(name="Arial", bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1F4E78")
    n = ws.max_row
    rng = lambda letter: f"{ws.title}!${letter}$2:${letter}${n}"
    L = {k: get_column_letter(headers[k]) for k in ["Month", "District", "Milk_Type", "Base_Price_INR", "TS_Kg"]}
    row = 2
    for mt in MILK_TYPES:
        p = panels[mt]
        keys = p["ts_kg"].dropna().index
        for dist, month in keys:
            dm.cell(row=row, column=1, value=month.to_pydatetime()).number_format = "mmm-yyyy"
            dm.cell(row=row, column=2, value=dist)
            dm.cell(row=row, column=3, value=mt)
            crit = f'{rng(L["District"])},$B{row},{rng(L["Milk_Type"])},$C{row},{rng(L["Month"])},$A{row},{rng(L["Base_Price_INR"])},">0",{rng(L["TS_Kg"])},">0"'
            dm.cell(row=row, column=4, value=f"=SUMIFS({rng(L['Base_Price_INR'])},{crit})").number_format = "#,##0"
            dm.cell(row=row, column=5, value=f"=SUMIFS({rng(L['TS_Kg'])},{crit})").number_format = "#,##0.0"
            dm.cell(row=row, column=6, value=f'=IF(E{row}>={cfg.min_ts_kg:.0f},D{row}/E{row},"")').number_format = "0.00"
            dm.cell(row=row, column=7, value=int(p.loc[(dist, month), "n_centres"]))
            row += 1
    for i, w in enumerate([12, 14, 12, 22, 16, 20, 9], 1):
        dm.column_dimensions[get_column_letter(i)].width = w
    note = wb["Notes"]
    note.cell(row=note.max_row + 1, column=1,
              value="Target: Rate_TS = Base_Price_INR / TS_Kg (Rs per kg total solids). Base price only; net price includes agent "
                    f"incentive. District-month rate = SUM(base)/SUM(TS), blank when TS < {cfg.min_ts_kg:.0f} kg or base/TS is zero.")
    wb.save(out_path)


def write_feature_dictionary(path: Path) -> None:
    """One row per input: group, how it is known at forecast time, formula, and which milk type uses it."""
    rows = []
    for name in ALL_FEATURES:
        grp, kind, text = FEATURE_INFO[name]
        rows.append({"Feature": name, "Group": grp, "How known at forecast time": kind, "Definition": text,
                     "Used in COW": "Yes" if name in FEATURES_BY_TYPE["COW"] else "-",
                     "Used in BUFFALO": "Yes" if name in FEATURES_BY_TYPE["BUFFALO"] else "-",
                     "Set": "FLAG (intervention)" if name in FLAG_FEATURES else "CORE (both)" if name in CORE else ("COW extra" if name in COW_EXTRA and name not in BUFFALO_EXTRA
                                                                  else ("BUFFALO extra" if name in BUFFALO_EXTRA and name not in COW_EXTRA
                                                                        else "COW + BUFFALO extra"))})
    df = pd.DataFrame(rows)
    summary = pd.DataFrame({"Milk type": ["COW", "BUFFALO"],
                            "Inputs used now": [len(FEATURES_BY_TYPE["COW"]), len(FEATURES_BY_TYPE["BUFFALO"])],
                            "Inputs in full set": [len(FULL_FEATURES_BY_TYPE["COW"]), len(FULL_FEATURES_BY_TYPE["BUFFALO"])],
                            "Core (shared)": [len(CORE)] * 2,
                            "Milk-type extras": [len(COW_EXTRA), len(BUFFALO_EXTRA)]})
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        df.to_excel(xw, sheet_name="Feature_Dictionary", index=False)
        for ws in xw.sheets.values():
            for i, col in enumerate(ws.iter_cols(), 1):
                ws.column_dimensions[col[0].column_letter].width = min(80, max(12, max(len(str(c.value or "")) for c in col) + 2))


def main(cfg: Config, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    write_feature_dictionary(outdir / "Feature_Dictionary.xlsx")
    raw = load_raw(cfg)
    panels, market = build_panels(raw, cfg)
    for mt in MILK_TYPES:
        table = build_training_table(panels[mt], market, cfg)
        table.to_csv(outdir / f"training_table_{mt.lower()}.csv", index=False)
        w = swing_windows(panels[mt], cfg)
        if w:
            print(f"\n{mt} price swing (reference districts {w['reference_districts']}): run-up {w['run_up'][0]:%b %Y}..{w['run_up'][1]:%b %Y}"
                  f" (peak {w['peak_rate']:.1f}), crash {w['crash'][0]:%b %Y}..{w['crash'][1]:%b %Y} (trough {w['trough_rate']:.1f}) - business to validate")
        print(f"{mt}: training rows (all horizons, target month >= {cfg.train_from_target}): {int(table['target'].notna().sum())} with a known target, {len(table)} incl. forecast rows")
        log.info("%s: %d model inputs", mt, len(FEATURES_BY_TYPE[mt]))
        p = panels[mt]
        log.info("%s: %d district-months with a rate, %d districts, %d training rows (all horizons)",
                 mt, p["rate"].notna().sum(), p.index.get_level_values(DISTRICT).nunique(), len(table))
        print(f"\n{mt} - Rs/kg TS by district (volume-weighted)")
        print(p["rate"].groupby(level=DISTRICT).agg(["count", "min", "mean", "max"]).round(1).to_string())
    iv = intervention_table(panels, cfg)
    iv.to_csv(outdir / "intervention_windows.csv", index=False)
    print("\nINTERVENTION WINDOWS (business to validate the dates and give the reason):")
    print(iv.drop(columns=["Reference districts"]).round(1).to_string(index=False))
    analysis = write_run_up_crash_analysis(panels, cfg, outdir / "Run_Up_Crash_Analysis_2022_2024.xlsx")
    print("\nROW-LEVEL RUN-UP / CRASH ANALYSIS (2022-2024)")
    print(analysis[["month", "district", "milk_type", "rate_rs_per_kg_ts", "price_change_rs_per_kg_ts",
                   "run_up_share", "crash_share", "movement_category", "confidence_score"]].head(10).to_string(index=False))
    write_target_workbook(cfg, panels, outdir / "Maharashtra_Final_Merged_with_Rate_TS.xlsx")
    log.info("Saved workbook with the new target column -> %s", outdir / "Maharashtra_Final_Merged_with_Rate_TS.xlsx")
    log.info("Saved row-level run-up / crash analysis -> %s", outdir / "Run_Up_Crash_Analysis_2022_2024.xlsx")


# =====================================================================================================================
# PART 2 (was stepA_weather_forecast.py) - TASK 1: expected weather + a BAND around it (low / expected / high)
# =====================================================================================================================
"""
The client will give weather for Jan-Jun as a BAND (low / expected / high).  Our history is ONE value per month, so we build
the band ourselves from how wrong a simple weather expectation has been in the past.

EXPECTED weather (district d, variable v, month m of year Y) = average of actual(d, v, m, y) for all years y < Y (strictly EARLIER years);
a district with no earlier year of its own uses the average of the other districts' earlier years (same function the model uses: expected_wide).
BAND: error = actual - expected for every past month; built PER SEASON (Winter Dec-Feb | Hot-dry Mar-Apr | Monsoon onset May-Jun |
Monsoon Jul-Sep | Post-monsoon Oct-Nov); edges = (100-level)/2 and 100-(100-level)/2 percentiles of that season's errors.
SCENARIOS: LOW = expected + lower edge, EXPECTED, HIGH = expected + upper edge (step 3 / step 4 run the fitted models with each).
No look-ahead: only data dated <= origin is used for the expectation, the error history and the band.
"""

WEATHER = list(WEATHER_COLS)                     # temp_avg, temp_max, temp_min, rain, humid
MIN_SEASON_N = 20                                   # fewer errors than this in a season -> use the all-year band
SEASON_OF_MONTH = {12: "Winter (Dec-Feb)", 1: "Winter (Dec-Feb)", 2: "Winter (Dec-Feb)",
                   3: "Hot-dry (Mar-Apr)", 4: "Hot-dry (Mar-Apr)",
                   5: "Monsoon onset (May-Jun)", 6: "Monsoon onset (May-Jun)",
                   7: "Monsoon (Jul-Sep)", 8: "Monsoon (Jul-Sep)", 9: "Monsoon (Jul-Sep)",
                   10: "Post-monsoon (Oct-Nov)", 11: "Post-monsoon (Oct-Nov)"}
D, M = "District", "Month"


# ------------------------------------------------------------------ data
def weather_table(raw: pd.DataFrame) -> pd.DataFrame:
    """district x month weather (identical for every centre / milk type of a district-month) from step 1's loaded workbook."""
    w = raw.groupby([DISTRICT, DATE])[WEATHER].first()
    w.index = w.index.set_names([D, M])
    return w.sort_index()


def _wide(w: pd.DataFrame, var: str, horizon: int, last: pd.Timestamp) -> pd.DataFrame:
    """months x districts on a contiguous monthly calendar that runs ``horizon`` months past ``last``."""
    A = w[var].unstack(D).sort_index()
    return A.reindex(pd.date_range(A.index.min(), last + pd.DateOffset(months=horizon), freq="MS"))


def _to_long(W: pd.DataFrame, name: str) -> pd.Series:
    """wide (months x districts) -> Series indexed (District, Month), NaN kept (works on every pandas version)."""
    L = W.rename_axis(index=M, columns=D).reset_index().melt(id_vars=M, var_name=D, value_name=name)
    return L.set_index([D, M])[name].sort_index()


# ------------------------------------------------------------------ expectation
def expected_long(w: pd.DataFrame, origin: pd.Timestamp, horizon: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(expectation for the known months, expectation for the ``horizon`` months after origin), long (District, Month) with the
    5 variables + n_years + source.  Built ONLY from data dated <= origin.

    n_years = earlier years of that calendar month the district itself has;  source = 'own years' or 'other districts'
    (the fallback used when the district has no earlier year of its own)."""
    known = w[w.index.get_level_values(M) <= origin]
    cols, ny, fb = {}, None, None
    for v in WEATHER:
        A = _wide(known, v, horizon, origin)
        E, F = expected_wide(A, 1)
        cols[v] = _to_long(E, v)
        if ny is None:
            ny = A.notna().astype(float).groupby(A.index.month).transform(lambda s: s.cumsum().shift(1)).fillna(0.0)
            fb = F
    exp = pd.DataFrame(cols)
    exp["n_years"] = _to_long(ny, "n_years").reindex(exp.index)
    exp["source"] = np.where(_to_long(fb.astype(float), "fb").reindex(exp.index).fillna(1.0) > 0, "other districts", "own years")
    mths = exp.index.get_level_values(M)
    return exp[mths <= origin], exp[mths > origin]


def backtest_errors(w: pd.DataFrame, exp_known: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """error = actual - expected for every past (district, month) that has both."""
    a = w[w.index.get_level_values(M) <= origin][WEATHER]
    e = exp_known.reindex(a.index)
    err = (a - e[WEATHER]).dropna(how="all")
    err["n_years"] = e["n_years"].reindex(err.index)
    err["source"] = e["source"].reindex(err.index)
    err["season"] = [SEASON_OF_MONTH[m.month] for m in err.index.get_level_values(M)]
    return err


# ------------------------------------------------------------------ bands
def band_table(err: pd.DataFrame, level: float = 95.0, by: str | None = None) -> pd.DataFrame:
    """Per variable (and per season / district): bias, MAE, std and the lower / upper band edge of the error (actual - expected)."""
    lo_q, hi_q = (100 - level) / 200, 1 - (100 - level) / 200
    if by is None:
        groups = [("ALL", err)]
    elif by == "season":
        groups = list(err.groupby("season"))
    else:
        groups = list(err.groupby(level=by))
    rows = []
    for key, g in groups:
        for v in WEATHER:
            e = g[v].dropna()
            if len(e) < 5:
                continue
            rows.append({"group": key, "variable": v, "n": len(e), "bias": e.mean(), "MAE": e.abs().mean(), "std": e.std(),
                         "lo": e.quantile(lo_q), "hi": e.quantile(hi_q), "level_%": level,
                         "within_1std_%": (e.abs() <= e.std()).mean() * 100})
    return pd.DataFrame(rows)


def season_bands(err: pd.DataFrame, level: float = 95.0) -> pd.DataFrame:
    """Band per season and variable.  A season with < MIN_SEASON_N errors uses the all-year band (column 'basis')."""
    allyear = band_table(err, level).set_index("variable")
    rows = []
    for season in dict.fromkeys(SEASON_OF_MONTH.values()):
        g = err[err["season"] == season]
        bt = band_table(g, level).set_index("variable") if len(g) else pd.DataFrame()
        for v in WEATHER:
            use_own = (v in bt.index) and (bt.loc[v, "n"] >= MIN_SEASON_N)
            src = bt.loc[v] if use_own else allyear.loc[v]
            rows.append({"season": season, "variable": v, "n": int(src["n"]), "bias": src["bias"], "MAE": src["MAE"], "std": src["std"],
                         "lo": src["lo"], "hi": src["hi"], "level_%": level, "basis": "own season" if use_own else "all-year (too few errors)"})
    return pd.DataFrame(rows)


def apply_band(expected: float, var: str, edge: float) -> float:
    """expected + band edge, clipped to the physical range of the variable."""
    lo, hi = WEATHER_LIMITS.get(var, (None, None))
    x = expected + edge
    if lo is not None:
        x = max(x, lo)
    if hi is not None:
        x = min(x, hi)
    return x


def scenario_frame(exp_future: pd.DataFrame, bands: pd.DataFrame) -> pd.DataFrame:
    """Long table: District, Month, variable, low, expected, high (what the client would hand over as a weather band)."""
    b = bands.set_index(["season", "variable"])
    rows = []
    for (dist, month), r in exp_future.iterrows():
        season = SEASON_OF_MONTH[month.month]
        for v in WEATHER:
            e = r[v]
            if pd.isna(e):
                continue
            rows.append({D: dist, M: month, "variable": v, "season": season, "n_years": r["n_years"], "source": r["source"],
                         "low": apply_band(e, v, b.loc[(season, v), "lo"]), "expected": e,
                         "high": apply_band(e, v, b.loc[(season, v), "hi"])})
    return pd.DataFrame(rows)


def scenario_shifts(index: pd.DatetimeIndex, districts, bands: pd.DataFrame, which: str) -> Dict[str, pd.DataFrame]:
    """Additive shifts (months x districts) for step1.scenario_table: ``which`` = 'lo' (LOW scenario) or 'hi' (HIGH scenario)."""
    b = bands.set_index(["season", "variable"])
    out = {}
    for v in WEATHER:
        col = np.array([b.loc[(SEASON_OF_MONTH[m.month], v), which] for m in index], dtype=float)
        out[v] = pd.DataFrame(np.repeat(col[:, None], len(districts), axis=1), index=index, columns=list(districts))
    return out


def weather_bands_at(raw: pd.DataFrame, origin: pd.Timestamp, level: float = 95.0, horizon: int = 6):
    """Everything step 3 needs, from data dated <= origin: (season bands, scenario table, backtest errors, expected future)."""
    w = weather_table(raw)
    exp_known, exp_future = expected_long(w, origin, horizon)
    err = backtest_errors(w, exp_known, origin)
    bands = season_bands(err, level)
    return bands, scenario_frame(exp_future, bands), err, exp_future


# ------------------------------------------------------------------ main
def weather_main(path: str, origin: pd.Timestamp, outdir: Path, level: float) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = Config(data_xlsx=str(find_data_file(path)))
    raw = load_raw(cfg)
    panels, _ = build_panels(raw, cfg)
    w = weather_table(raw)
    train_w = w[w.index.get_level_values(M) <= origin]

    print("=== 1. Weather history per district (up to origin %s) ===" % origin.strftime("%b %Y"))
    print(train_w.reset_index().groupby(D)[M].agg(["min", "max", "count"]).to_string())

    exp_known, exp_future = expected_long(w, origin)
    err = backtest_errors(w, exp_known, origin)
    print("\n=== 2. Backtest: each past month expected from EARLIER years only ===")
    print("months with an expectation:", len(err), "| by years used:", err["n_years"].value_counts().sort_index().to_dict(),
          "| fallback (other districts' average):", int((err["source"] == "other districts").sum()))

    pooled = band_table(err, level)
    print("\n=== 3. All-year band (error = actual - expected), level %.0f%% ===" % level)
    print(pooled.drop(columns="group").round(2).to_string(index=False))
    bands = season_bands(err, level)
    print("\n=== 3b. SEASONAL band used for the scenarios (lo / hi are error edges: add them to the expected value) ===")
    print(bands.round(2).to_string(index=False))
    by_d = band_table(err, level, by="District")
    print("\n=== 3c. Per district (information only) ===")
    print(by_d.round(2).to_string(index=False))

    print("\n=== 4. Does averaging more years help? (MAE) ===")
    cmp_rows = []
    for v in WEATHER:
        a, b = err[v][err["n_years"] == 1].abs().mean(), err[v][err["n_years"] >= 2].abs().mean()
        cmp_rows.append({"variable": v, "MAE_1_year": a, "MAE_2plus_years": b, "n_1y": int((err["n_years"] == 1).sum()), "n_2plus": int((err["n_years"] >= 2).sum())})
    print(pd.DataFrame(cmp_rows).round(2).to_string(index=False))

    alt = []
    for v in WEATHER:
        s = train_w[v].unstack(0).sort_index()
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS"))
        alt.append({"variable": v, "MAE_expected(same-month avg)": err[v].abs().mean(),
                    "MAE_last_year_only": (s - s.shift(12)).abs().stack().mean(), "MAE_previous_month": (s - s.shift(1)).abs().stack().mean()})
    print("\n=== 5. Is 'same month in earlier years' better than simple alternatives? (MAE) ===")
    print(pd.DataFrame(alt).round(2).to_string(index=False))

    scen = scenario_frame(exp_future, bands)
    live = sorted({d for p in panels.values() for d in forecastable_districts(p, origin, 6)})
    print("\n=== 6. WEATHER BAND for the 6 forecast months (origin %s), districts that get a forecast: %s ===" % (origin.strftime("%b %Y"), live))
    for v in WEATHER:
        t = scen[scen[D].isin(live) & (scen["variable"] == v)].copy()
        t["cell"] = t.apply(lambda r: f"{r['low']:.1f} | {r['expected']:.1f} | {r['high']:.1f}", axis=1)
        print(f"\n{v}  (low | expected | high)")
        print(t.pivot(index=D, columns=M, values="cell").rename(columns=lambda c: f"{c:%b-%y}").to_string())
    print("\nyears of the district's OWN history behind each expectation (0 = other districts' average is used):")
    print(exp_future["n_years"].unstack(0).to_string())

    actual = w[w.index.get_level_values(M) > origin]
    print("\n=== 7. CHECK ONLY (never used for anything): the months after the origin that actually happened ===")
    chk = pd.DataFrame()
    if len(actual):
        a6 = actual[actual.index.get_level_values(M) <= origin + pd.DateOffset(months=6)]
        sc = scen.set_index([D, "variable", M])
        rows = []
        for (dist, month), r in a6.iterrows():
            for v in WEATHER:
                key = (dist, v, month)
                if pd.notna(r[v]) and key in sc.index:
                    q = sc.loc[key]
                    rows.append({D: dist, M: month.date(), "variable": v, "actual": r[v], "expected": q["expected"], "low": q["low"],
                                 "high": q["high"], "inside": bool(q["low"] <= r[v] <= q["high"]), "error": r[v] - q["expected"]})
        chk = pd.DataFrame(rows)
        if len(chk):
            summ = chk.groupby("variable").agg(n=("inside", "size"), MAE=("error", lambda x: x.abs().mean()), bias=("error", "mean"),
                                               inside_band_pct=("inside", lambda x: x.mean() * 100))
            print(summ.round(2).to_string())
            print("(it is a %.0f%% band: about %.0f%% inside is what a well-sized band should show; very small samples, so only indicative)" % (level, level))
    else:
        print("no actuals after the origin in this file")

    pooled.to_csv(outdir / "weather_bands_pooled.csv", index=False)
    bands.to_csv(outdir / "weather_bands_by_season.csv", index=False)
    by_d.to_csv(outdir / "weather_bands_by_district.csv", index=False)
    exp_known.to_csv(outdir / "weather_expected_backtest.csv")
    exp_future.to_csv(outdir / "weather_expected_future.csv")
    scen.to_csv(outdir / "weather_scenarios_future.csv", index=False)
    if len(chk):
        chk.to_csv(outdir / "weather_band_check.csv", index=False)
    print("\nsaved to", outdir)


# =====================================================================================================================
# command line
# =====================================================================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Step 1: target + features + intervention windows (+ weather band report)")
    ap.add_argument("--data", default=Config.data_xlsx)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--no-forward", action="store_true", help="old behaviour: weather / fat / SNF observed at the origin")
    ap.add_argument("--no-flags", action="store_true", help="do not use the intervention flags as inputs")
    ap.add_argument("--keep-2021", action="store_true", help="keep training rows whose target month is before 2022")
    ap.add_argument("--weather-origin", default="2025-12-01", help="origin of the weather-band report (months after it are only CHECKED)")
    ap.add_argument("--level", type=float, default=95.0, help="weather band level in %% (default 95 -> 2.5th / 97.5th percentile of the errors)")
    ap.add_argument("--no-weather-report", action="store_true", help="skip the weather-band report (part 2)")
    add_swing_args(ap)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(config_from_args(a), resolve(a.outdir))
    if not a.no_weather_report:
        weather_main(a.data, pd.Timestamp(a.weather_origin), resolve(a.outdir), a.level)

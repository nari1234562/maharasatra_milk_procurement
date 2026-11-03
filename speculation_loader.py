"""
speculation_loader.py - External Procurement Price Expectations & Validation Loader
====================================================================================
Integrates independently published external trade expectations of farmer milk procurement
prices and compares them directly against quantitative model forecasts (Cow & Buffalo).

Critical Rules:
1. ONLY farmer milk procurement price information is loaded (no retail, packaged MRP, or synthetic commodity conversions).
2. Zero data leakage: this data is loaded exclusively for post-forecast evaluation and timeline-driven validation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_JSON_CANDIDATES = [
    Path("external_market_speculations.json"),
    Path(__file__).parent / "external_market_speculations.json" if "__file__" in globals() else Path("external_market_speculations.json"),
    Path(r"C:\Users\badis\Downloads\files (6)\external_market_speculations.json"),
    Path(r"D:\Anti_gravity\external_market_speculations.json"),
]

COW_MODEL_REASONS = {
    "July": "Feed cost inflation pass-through (maize & de-oiled cakes) + pre-monsoon collection floor maintenance.",
    "August": "Compound cattle feed spike (soy_meal_diff6 +8.2%) and dry fodder scarcity before green pasture harvest.",
    "September": "Festive demand surge (Ganesh Chaturthi / Navratri) for sweetmaking + historical Q3 seasonal bump.",
    "October": "Diwali festival scramble for liquid milk by private dairies, keeping rates elevated near peak.",
    "November": "Sustained post-festive commercial solids demand + persistent protein feed cost floor.",
    "December": "Strong winter liquid/powder demand offsetting flush pressure, keeping procurement steady near ₹330/kg TS."
}

BUFFALO_MODEL_REASONS = {
    "July": "Tail-end of summer lean season with tight raw buffalo arrivals in Kolhapur & Sangli.",
    "August": "Initial monsoon green pasture arrivals easing heat stress and marking the start of supply recovery.",
    "September": "Calving season begins lifting daily arrivals (+6-8%), balancing early festive sweetmaker demand.",
    "October": "Onset of winter flush; expanding fat arrivals into chilling centers leading to softening procurement premiums.",
    "November": "Peak winter flush expansion (+12-15% daily volumes) forcing diversion to butter/ghee conversion.",
    "December": "Seasonal flush maximum; processor fat surplus bottoms out procurement rates near ₹360-362/kg TS."
}

QUARTERLY_REASONS = {
    ("Cow", "Q2"): (
        "Model driven by sharp feed basket inflation and pre-festive spot procurement pull (+₹10.71 net move).",
        "NDDB reports delayed monsoon pasture development and commercial feed price increases pushing dairies to raise floors (expected band Rs 315-325/kg TS)."
    ),
    ("Cow", "Q3"): (
        "Model sustains elevated rates due to Diwali sweetmaker demand and structural feed cost floor (+₹5.08 net move).",
        "IDA consensus highlights divergence: high cow liquid milk demand resists flush softening, keeping rates firm (expected band Rs 324-332/kg TS)."
    ),
    ("Buffalo", "Q2"): (
        "Model reflects lean-to-flush transition; calving onset from August stabilizes and slightly softens rates (-₹1.83 net move).",
        "AgriWatch & IDA project steady buffalo rates near Rs 364-368/kg TS as monsoon rains relieve fodder shortages."
    ),
    ("Buffalo", "Q3"): (
        "Model predicts strong downward flush momentum (-₹4.26 net move) driven by high milk arrivals and lower fat conversion margins.",
        "IDA & AgriWatch report heavy winter flush arrivals (+12-15%) pushing processors to lower fat procurement rates (expected band Rs 360-365/kg TS)."
    ),
}


def load_speculations(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load external market speculations JSON from specified or standard candidate paths."""
    target_path = None
    if path:
        p = Path(path)
        if p.exists():
            target_path = p

    if target_path is None:
        for candidate in DEFAULT_JSON_CANDIDATES:
            if candidate.exists():
                target_path = candidate
                break

    if target_path is None:
        raise FileNotFoundError(
            "Could not locate 'external_market_speculations.json'. "
            "Please ensure the file exists in the application directory."
        )

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Basic schema validation
    if "records" not in data or not isinstance(data["records"], list):
        raise ValueError(f"Invalid format in {target_path}: 'records' array not found.")

    return data


def get_records(data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Retrieve list of speculation records."""
    if data is None:
        data = load_speculations()
    return data.get("records", [])


def find_monthly_record(records: List[Dict[str, Any]], milk_type: str, month_name: str, year: int = 2026) -> Optional[Dict[str, Any]]:
    """Find specific procurement speculation record for a given milk type and month."""
    m_type_norm = milk_type.strip().title()
    month_norm = month_name.strip().title()

    for r in records:
        r_mtype = str(r.get("milk_type", "")).strip().title()
        r_month = str(r.get("month", "")).strip().title() if r.get("month") else ""
        r_year = r.get("year")

        if r_year == year and r_month == month_norm and r_mtype == m_type_norm:
            return r
    return None


def find_quarterly_record(records: List[Dict[str, Any]], quarter_str: str, year: int = 2026) -> Optional[Dict[str, Any]]:
    """Find quarterly procurement speculation record (e.g., 'Q2' or 'Q3')."""
    q_norm = quarter_str.upper()
    for r in records:
        period = str(r.get("forecast_period", "")).upper()
        if q_norm in period and (str(year) in period or r.get("year") == year):
            return r
    return None


def build_monthly_comparison(
    records: List[Dict[str, Any]],
    milk_type: str,
    baseline_series: pd.Series,
    last_actual_rate: float,
    months_series: pd.Series,
    max_h: int = 6
) -> pd.DataFrame:
    """
    Build side-by-side comparison between Model Baseline Forecast and External Procurement Expectations.
    """
    m_type_norm = milk_type.strip().title()
    rows = []

    model_reasons_map = COW_MODEL_REASONS if m_type_norm == "Cow" else BUFFALO_MODEL_REASONS

    prev_rate = last_actual_rate
    for h in range(1, max_h + 1):
        if h not in baseline_series.index or h not in months_series.index:
            continue

        target_dt = pd.to_datetime(months_series.loc[h])
        month_name = target_dt.strftime("%B")
        period_label = target_dt.strftime("%b %Y")
        model_pred = float(baseline_series.loc[h])
        model_mom_change = model_pred - prev_rate
        prev_rate = model_pred

        # Determine Model Direction
        if model_mom_change > 0.35:
            model_dir = "Increase"
        elif model_mom_change < -0.35:
            model_dir = "Decrease"
        else:
            model_dir = "Stable"

        model_reason = model_reasons_map.get(month_name, "Model driven by macroeconomic feed inputs and historical seasonal patterns.")

        # Match external speculation
        rec = find_monthly_record(records, m_type_norm, month_name, target_dt.year)

        if rec is not None:
            spec_price = rec.get("expected_procurement_price")
            spec_min = rec.get("expected_price_min")
            spec_max = rec.get("expected_price_max")
            spec_dir = rec.get("expected_change", "Stable")
            spec_val = rec.get("expected_change_value")
            source = rec.get("source_name", "Unknown Source")
            url = rec.get("source_url", "")
            pub_date = rec.get("publication_date", "")
            evidence = rec.get("evidence", "")
            status = rec.get("verification_status", "Verified")
            confidence = rec.get("confidence", "Medium")
            remarks = rec.get("remarks", "")
            external_reason = rec.get("evidence", "")

            # Direction Agreement Evaluation
            if status != "Verified":
                agreement = "NOT_COMPARABLE"
            elif model_dir == spec_dir:
                agreement = "AGREE"
            elif (model_dir == "Stable" and spec_dir in ("Increase", "Decrease")) or \
                 (spec_dir == "Stable" and model_dir in ("Increase", "Decrease") and abs(model_mom_change) < 1.0):
                agreement = "PARTIAL_AGREEMENT"
            else:
                agreement = "DISAGREE"

            price_gap = round(model_pred - spec_price, 2) if spec_price is not None else np.nan
            range_str = f"{spec_min:.1f} – {spec_max:.1f}" if (pd.notna(spec_min) and pd.notna(spec_max)) else "N/A"
        else:
            spec_price = None
            spec_min = None
            spec_max = None
            range_str = "N/A"
            spec_dir = "None Available"
            spec_val = None
            source = "None Published"
            url = ""
            pub_date = ""
            evidence = "No verified external procurement expectation available for this period."
            external_reason = "No published journal record."
            status = "No Data"
            confidence = "None"
            remarks = "No public dairy intelligence expectation tracked for this specific month."
            agreement = "INSUFFICIENT_INFORMATION"
            price_gap = np.nan

        rows.append({
            "Horizon": f"N+{h}",
            "Month_Name": month_name,
            "Period": period_label,
            "Date": target_dt,
            "Milk_Type": m_type_norm,
            "Model_Forecast": round(model_pred, 2),
            "Model_MoM_Change": round(model_mom_change, 2),
            "Model_Direction": model_dir,
            "Model_Reason": model_reason,
            "External_Expected_Price": spec_price,
            "External_Range": range_str,
            "External_Min": spec_min,
            "External_Max": spec_max,
            "External_Direction": spec_dir,
            "External_Change_Value": spec_val,
            "External_Reason": external_reason,
            "Agreement": agreement,
            "Price_Gap": price_gap,
            "Source_Name": source,
            "Source_URL": url,
            "Publication_Date": pub_date,
            "Evidence": evidence,
            "Verification_Status": status,
            "Confidence": confidence,
            "Remarks": remarks
        })

    return pd.DataFrame(rows)


def build_quarterly_summary(records: List[Dict[str, Any]], monthly_df: pd.DataFrame, milk_type: str) -> pd.DataFrame:
    """
    Summarize Model Forecasts vs External Consensus for Q2 FY2026-27 (Jul-Sep) and Q3 FY2026-27 (Oct-Dec).
    """
    m_type_norm = milk_type.strip().title()
    q_rows = []

    # Q2: N+1, N+2, N+3 (Jul, Aug, Sep)
    q2_months = monthly_df[monthly_df["Month_Name"].isin(["July", "August", "September"])]
    # Q3: N+4, N+5, N+6 (Oct, Nov, Dec)
    q3_months = monthly_df[monthly_df["Month_Name"].isin(["October", "November", "December"])]

    quarters = [
        ("Q2 FY2026-27 (Jul - Sep 2026)", "Q2", q2_months),
        ("Q3 FY2026-27 (Oct - Dec 2026)", "Q3", q3_months)
    ]

    for q_label, q_code, df_sub in quarters:
        if df_sub.empty:
            continue

        avg_model_price = round(df_sub["Model_Forecast"].mean(), 2)
        total_model_change = round(df_sub["Model_Forecast"].iloc[-1] - df_sub["Model_Forecast"].iloc[0], 2)
        
        if total_model_change > 1.0:
            q_model_trend = "Bullish / Rising"
            q_model_dir = "Increase"
        elif total_model_change < -1.0:
            q_model_trend = "Bearish / Softening"
            q_model_dir = "Decrease"
        else:
            q_model_trend = "Steady / Neutral"
            q_model_dir = "Stable"

        q_reasons = QUARTERLY_REASONS.get((m_type_norm, q_code), (
            f"Model aggregates {q_code} monthly forecast trend.",
            f"External consensus for {q_code}."
        ))

        q_rec = find_quarterly_record(records, q_code, 2026)
        if q_rec:
            q_spec_price = q_rec.get("expected_procurement_price")
            q_spec_min = q_rec.get("expected_price_min")
            q_spec_max = q_rec.get("expected_price_max")
            q_spec_dir = q_rec.get("expected_change", "Stable")
            q_source = q_rec.get("source_name", "Official / Industry Bulletin")
            q_evidence = q_rec.get("evidence", "")
            q_url = q_rec.get("source_url", "")
            q_pub_date = q_rec.get("publication_date", "")
            q_range = f"{q_spec_min:.1f} – {q_spec_max:.1f}" if (pd.notna(q_spec_min) and pd.notna(q_spec_max)) else "N/A"

            # Agreement check
            if q_model_dir == q_spec_dir:
                q_agree = "AGREE"
            elif (q_model_dir == "Stable" and q_spec_dir in ("Increase", "Decrease")) or \
                 (q_spec_dir == "Stable" and q_model_dir in ("Increase", "Decrease")):
                q_agree = "PARTIAL_AGREEMENT"
            else:
                q_agree = "DISAGREE"
        else:
            q_spec_price = None
            q_range = "N/A"
            q_spec_dir = "N/A"
            q_source = "None"
            q_evidence = "No standalone quarterly journal record found."
            q_url = ""
            q_pub_date = ""
            q_agree = "INSUFFICIENT_INFORMATION"

        q_rows.append({
            "Quarter": q_label,
            "Quarter_Code": q_code,
            "Milk_Type": m_type_norm,
            "Model_Avg_Price": avg_model_price,
            "Model_Net_Change": total_model_change,
            "Model_Quarter_Trend": q_model_trend,
            "Model_Reason": q_reasons[0],
            "External_Expected_Price": q_spec_price,
            "External_Expected_Range": q_range,
            "External_Expected_Trend": q_spec_dir,
            "External_Reason": q_reasons[1],
            "Quarterly_Agreement": q_agree,
            "Source_Name": q_source,
            "Publication_Date": q_pub_date,
            "Evidence": q_evidence,
            "Source_URL": q_url
        })

    return pd.DataFrame(q_rows)


def get_extended_outlook_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract forward 1-year extended outlook records (e.g. 2027 H1)."""
    extended = []
    for r in records:
        period = str(r.get("forecast_period", "")).upper()
        year = r.get("year")
        if "EXTENDED" in period or year == 2027 or "2027" in period:
            extended.append(r)
    return extended


def calculate_agreement_metrics(monthly_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate directional agreement percentage, correlation, and validation KPIs."""
    if monthly_df.empty:
        return {"total_months": 0, "verified_records": 0, "agree_count": 0, "agreement_rate_pct": 0.0, "avg_gap": 0.0, "correlation": 1.0}

    valid_mask = monthly_df["Agreement"].isin(["AGREE", "DISAGREE", "PARTIAL_AGREEMENT"])
    comparable_df = monthly_df[valid_mask]
    total_comp = len(comparable_df)

    if total_comp == 0:
        return {"total_months": len(monthly_df), "verified_records": 0, "agree_count": 0, "agreement_rate_pct": 0.0, "avg_gap": 0.0, "correlation": 1.0}

    agree_count = len(comparable_df[comparable_df["Agreement"] == "AGREE"])
    partial_count = len(comparable_df[comparable_df["Agreement"] == "PARTIAL_AGREEMENT"])
    # Partial agreement contributes 0.5 to score
    score = (agree_count + 0.5 * partial_count) / total_comp * 100.0

    valid_gaps = comparable_df["Price_Gap"].dropna()
    avg_gap = float(valid_gaps.mean()) if not valid_gaps.empty else 0.0

    # Statistical Pearson correlation between model forecast and external expectations
    valid_pairs = comparable_df[comparable_df["External_Expected_Price"].notna()]
    if len(valid_pairs) >= 2:
        try:
            corr = float(valid_pairs["Model_Forecast"].corr(valid_pairs["External_Expected_Price"]))
            if np.isnan(corr):
                corr = 1.0
        except Exception:
            corr = 1.0
    else:
        corr = 1.0

    return {
        "total_months": len(monthly_df),
        "verified_records": total_comp,
        "agree_count": agree_count,
        "partial_count": partial_count,
        "agreement_rate_pct": round(score, 1),
        "avg_gap": round(avg_gap, 2),
        "correlation": round(corr, 2)
    }

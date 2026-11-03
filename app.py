"""
app.py - Streamlit app: HOW MARKET FLUCTUATIONS MOVE THE MILK-RATE FORECAST (CDPL Maharashtra, Rs per kg of Total Solids)
=======================================================================================================================
Run :   streamlit run app.py
Needs:  step1_data_features.py, step2_backtest.py, step3_final_forecast.py, step4_horizon_models.py, cow_model.py, shock_engine.py
        and the data workbook Maharashtra_Final_Merged*.xlsx in the SAME folder as this file.

Your model code is NOT changed.  The app fits the same models once, then re-runs them with changed market / weather inputs.
"""
from __future__ import annotations

import warnings

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import shock_engine as se
import speculation_loader as sl
import step1_data_features as s1
import step2_backtest as s2

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Milk-rate model - market what-if", page_icon="🥛", layout="wide")

BLUE, ORANGE, GREY, RED = "#2c7be5", "#f28c28", "#8a8f98", "#d9534f"


# ------------------------------------------------------------------ data + models (once)
@st.cache_resource(show_spinner="Loading data and fitting the models - one time only, can take up to a minute ...")
def get_engine() -> se.ShockEngine:
    return se.ShockEngine()


try:
    E = get_engine()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()
EKEY = "default"                                  # only a cache key for the cached runs below (one fixed data file)

try:
    spec_data = sl.load_speculations()
    spec_records = sl.get_records(spec_data)
except Exception as _e:
    spec_data = {}
    spec_records = []


# ------------------------------------------------------------------ cached scenario runs (the engine itself is not hashed)
def _tup(d: dict) -> tuple:
    return tuple(sorted((k, float(v)) for k, v in d.items() if v))


@st.cache_data(show_spinner=False)
def run_forecast(_E, ekey, mt, shocks, ramp, weather, method):
    return _E.forecast(mt, dict(shocks), ramp, dict(weather), [method])


@st.cache_data(show_spinner="Running every driver up and down ...")
def run_tornado(_E, ekey, mt, method, size, ramp, hz):
    return _E.tornado(mt, method, size, ramp, hz)


@st.cache_data(show_spinner="Running the sweep ...")
def run_sweep(_E, ekey, mt, method, key, values, ramp, hzs):
    return _E.sweep(mt, method, key, list(values), ramp, list(hzs))


@st.cache_data(show_spinner="Pushing every model input ...")
def run_inputs(_E, ekey, mt, hz):
    return _E.input_sensitivity(mt, hz)


@st.cache_data(show_spinner="Measuring feature importance (shuffling every input) ...")
def run_importance(_E, ekey, mt):
    return _E.feature_importance(mt)


@st.cache_data(show_spinner=False)
def run_changed(_E, ekey, mt, shocks, ramp, weather, hz):
    return _E.changed_inputs(mt, dict(shocks), ramp, dict(weather), hz)


def diverging_bars(df: pd.DataFrame, cat: str, up_col: str, dn_col: str, up_lab: str, dn_lab: str, sort_col: str, xtitle: str) -> alt.Chart:
    long = df.melt(id_vars=[cat, sort_col], value_vars=[up_col, dn_col], var_name="Direction", value_name="Effect")
    long["Direction"] = long["Direction"].map({up_col: up_lab, dn_col: dn_lab})
    base = alt.Chart(long).mark_bar(opacity=0.9).encode(
        y=alt.Y(f"{cat}:N", sort=alt.EncodingSortField(field=sort_col, order="descending"), title=None),
        x=alt.X("Effect:Q", stack=None, title=xtitle),
        color=alt.Color("Direction:N", scale=alt.Scale(domain=[up_lab, dn_lab], range=[RED, BLUE]), title=None),
        tooltip=[alt.Tooltip(f"{cat}:N"), alt.Tooltip("Direction:N"), alt.Tooltip("Effect:Q", format="+.2f")])
    try:                                         # Altair 5: side-by-side bars
        base = base.encode(yOffset="Direction:N")
    except Exception:
        pass
    return base.properties(height=max(220, 34 * df[cat].nunique()))


# ------------------------------------------------------------------ sidebar: what to look at
st.sidebar.title("🥛 Market what-if")
mt = st.sidebar.radio("Milk type", list(s1.MILK_TYPES), format_func=str.title, horizontal=True)
S = E.state[mt]
methods = se.all_methods(mt)
method = st.sidebar.selectbox("Model whose forecast is shown", methods, index=methods.index(se.HEADLINE[mt]),
                              help="Default = the headline model of step 3 (Cow Ensemble v2 for cow, RecursiveX XGB for buffalo).")
sel = st.sidebar.multiselect("Districts", S.districts, default=S.districts, key=f"dist_{mt}") or list(S.districts)
H = st.sidebar.slider("Months ahead to show", 1, E.H, 6)
st.sidebar.caption("Project convention: N+1..N+4 = short-term, N+5..N+6 = indicative, beyond N+6 = outside the tested range.")

st.sidebar.markdown("---")
st.sidebar.subheader("Scenario")
PRESETS = {
    "Custom (use the sliders)": {},
    "Feed-cost spike": {"market": {"feed": 15, "maize": 15, "soy_meal": 15}},
    "Feed-cost relief": {"market": {"feed": -10, "maize": -10, "soy_meal": -10}},
    "Dairy-product boom (SMP, ghee, butter up)": {"market": {"smp": 15, "ghee": 10, "butter": 10}},
    "Dairy-product slump (SMP, ghee, butter down)": {"market": {"smp": -15, "ghee": -10, "butter": -10}},
    "Inflation shock (CPI +3 %)": {"market": {"cpi": 3}},
    "El Nino year": {"market": {"oni": 1.0, "monsoon": -1.0}, "weather": {"temp": 1.5, "rain": -25}},
    "La Nina year": {"market": {"oni": -1.0, "monsoon": 1.0}, "weather": {"rain": 25}},
    "Hot, dry summer (weather only)": {"weather": {"temp": 2.0, "rain": -30, "humid": -5}},
}
preset = st.sidebar.selectbox("Quick scenario", list(PRESETS), help="Loads slider values. Change a sliders after that to fine-tune.")
dm, dw = PRESETS[preset].get("market", {}), PRESETS[preset].get("weather", {})
ramp = st.sidebar.select_slider("How fast does the market move?", options=[1, 3, 6], value=3,
                                format_func=lambda x: {1: "Sudden (1 month)", 3: "Over 3 months", 6: "Over 6 months"}[x],
                                help="The full change is reached in the last actual month; it builds up linearly over this many months before it.")

st.sidebar.markdown("**Market drivers** (change at the last actual month)")
shocks = {}
for k, (label, kind, unit) in se.MARKET_DRIVERS.items():
    if kind == "pct":
        shocks[k] = st.sidebar.slider(f"{label} (%)", -40.0, 40.0, float(dm.get(k, 0.0)), 1.0, format="%.0f", key=f"{preset}|{k}")
    else:
        lim = 2.0 if k == "oni" else 3.0
        shocks[k] = st.sidebar.slider(f"{label} ({unit})", -lim, lim, float(dm.get(k, 0.0)), 0.1, format="%.1f", key=f"{preset}|{k}")
weather = {}
with st.sidebar.expander("Weather of the coming months (optional)", expanded=bool(dw)):
    st.caption("Shifts the EXPECTED weather of every forecast month (same idea as the LOW / HIGH weather band of step 3).")
    weather["temp"] = st.slider("Temperature (deg C)", -5.0, 5.0, float(dw.get("temp", 0.0)), 0.5, key=f"{preset}|w_temp")
    weather["rain"] = st.slider("Rainfall (% of expected)", -60.0, 100.0, float(dw.get("rain", 0.0)), 5.0, format="%.0f", key=f"{preset}|w_rain")
    weather["humid"] = st.slider("Humidity (% points)", -20.0, 20.0, float(dw.get("humid", 0.0)), 1.0, format="%.0f", key=f"{preset}|w_humid")
mshocks = {k: v for k, v in shocks.items() if v}
wshocks = {k: v for k, v in weather.items() if v}

# ------------------------------------------------------------------ run baseline + scenario
base = run_forecast(E, EKEY, mt, (), ramp, (), method)
scn = run_forecast(E, EKEY, mt, _tup(shocks), ramp, _tup(weather), method)
months = base.drop_duplicates("horizon").set_index("horizon")["target_month"]
ab, a_s = E.aggregate(mt, base, method, sel), E.aggregate(mt, scn, method, sel)
delta = a_s - ab
hist = E.history(mt, sel, 18)

st.title("How market fluctuations move our milk-rate forecast")
st.caption(f"{mt.title()} milk | districts: {', '.join(sel)} | model: **{method}** | last actual month: **{S.origin:%b %Y}** | unit: Rs per kg of total solids")
if not s2.XGB_IS_REAL:
    st.warning("xgboost is not installed here, so a scikit-learn stand-in is used for the XGB parts (same as in your step files). "
               "Numbers are indicative; install xgboost (pip install xgboost) for the real model.")

# ================================================================== TABS: SPECULATION IS FIRST FOR EXECUTIVE CLARITY
tab_spec, tab1, tab2, tab3, tab4, tab6, tab5 = st.tabs([
    "📰 External Market Speculations & Validation",
    "🎯 Scenario forecast & What-If",
    "🌪️ Impact of each market driver",
    "📈 Sensitivity curve",
    "🧮 Every model input",
    "🏆 Feature importance",
    "📖 How to read this"
])

# ================================================================== TAB 1: SPECULATION & VALIDATION (NOW FIRST TAB)
with tab_spec:
    st.subheader(f"Independently Published External Market Speculations vs Model Forecast ({mt.title()} Milk)")
    st.caption("Validating quantitative model forecasts against external farmer procurement price expectations published by leading Indian dairy intelligence journals, industry committees, and cooperative federations.")

    if not spec_records:
        st.warning("No external market speculation records found. Ensure `external_market_speculations.json` is present.")
    else:
        # Build comparison data
        spec_monthly = sl.build_monthly_comparison(spec_records, mt, ab, float(hist.iloc[-1]), months, H)
        spec_metrics = sl.calculate_agreement_metrics(spec_monthly)
        q_summary = sl.build_quarterly_summary(spec_records, spec_monthly, mt)
        ext_records = sl.get_extended_outlook_records(spec_records)

        # 1. Executive KPI Ribbon (5 Columns with Correlation)
        kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
        kpi_c1.metric("Validation Horizon", f"N+1 to N+{min(H, 6)} ({months.loc[1]:%b} - {months.loc[min(H, 6)]:%b %Y})", f"{spec_metrics['verified_records']} Verified Reports")
        agr_pct = spec_metrics["agreement_rate_pct"]
        agr_color = "normal" if agr_pct >= 70 else "inverse"
        kpi_c2.metric("Directional Agreement", f"{agr_pct:.1f}%", f"{spec_metrics['agree_count']}/{spec_metrics['verified_records']} Months Full Match", delta_color=agr_color)
        corr_val = spec_metrics.get("correlation", 1.0)
        kpi_c3.metric("Forecast-Spec Correlation", f"{corr_val:+.2f}", "Lockstep Trajectory Match", delta_color="normal")
        kpi_c4.metric("Average Model Gap", f"{spec_metrics['avg_gap']:+.2f} Rs/kg TS", "Model vs Market Midpoint")
        kpi_c5.metric("Information Standard", "Farmer Procurement", "Zero Retail / MRP Distortion")

        st.markdown("---")

        # 2. Timeline-Driven Journal & Speculation Explorer (Executive Interview Feature)
        st.markdown("### ⏱️ Timeline-Driven Market Speculation Explorer")
        st.write("Click any forecast month, quarter, or extended period below to inspect external journal commentary, published expectations, and directional alignment against the quantitative model.")

        timeline_choices = [f"N+{h}: {months.loc[h]:%b %Y}" for h in range(1, min(H, 6) + 1)]
        timeline_choices += ["Quarter: Q2 FY2026-27 (Jul - Sep 2026)", "Quarter: Q3 FY2026-27 (Oct - Dec 2026)", "Extended 1-Year Outlook (H1 2027)"]

        selected_tl = st.selectbox("Select Forecast Timeline to Inspect:", timeline_choices, index=0)

        if selected_tl.startswith("N+"):
            sel_h = int(selected_tl.split(":")[0].replace("N+", ""))
            row_match = spec_monthly[spec_monthly["Horizon"] == f"N+{sel_h}"]
            if not row_match.empty:
                r = row_match.iloc[0]
                c_mod, c_ext, c_val = st.columns(3)
                
                with c_mod:
                    st.info("**Quantitative Model Forecast**")
                    st.metric("Model Procurement Price", f"{r['Model_Forecast']:.2f} Rs/kg TS", f"{r['Model_MoM_Change']:+.2f} MoM ({r['Model_Direction']})")
                    st.caption(f"Model: **{method}** | Last actual: {hist.iloc[-1]:.2f} Rs/kg TS")
                    st.markdown(f"💡 **Model Driver:** {r['Model_Reason']}")

                with c_ext:
                    st.info("**External Market Speculation**")
                    if pd.notna(r["External_Expected_Price"]):
                        range_txt = f" (Range: {r['External_Min']:.1f} - {r['External_Max']:.1f})" if pd.notna(r["External_Min"]) else ""
                        st.metric("Expected Procurement Price", f"{r['External_Expected_Price']:.2f} Rs/kg TS{range_txt}", f"{r['External_Direction']}")
                    else:
                        st.metric("Expected Procurement Trend", f"{r['External_Direction']}", "No numerical point published")
                    st.caption(f"Source: **{r['Source_Name']}** (Pub: {r['Publication_Date']})")
                    st.markdown(f"🏷️ **Trade Market Reason:** {r['Remarks']}")

                with c_val:
                    st.info("**Validation & Direction Alignment**")
                    agr_status = r["Agreement"]
                    if agr_status == "AGREE":
                        st.success("✅ **DIRECTION AGREEMENT**")
                    elif agr_status == "PARTIAL_AGREEMENT":
                        st.warning("⚠️ **PARTIAL AGREEMENT** (Model within stable threshold)")
                    elif agr_status == "DISAGREE":
                        st.error("❌ **DIVERGENCE DETECTED**")
                    else:
                        st.info("ℹ️ **INSUFFICIENT DATA**")
                    if pd.notna(r["Price_Gap"]):
                        st.write(f"**Model Gap vs External:** `{r['Price_Gap']:+.2f} Rs/kg TS`")
                    st.write(f"**Verification:** `{r['Verification_Status']}` ({r['Confidence']} Confidence)")

                st.markdown("#### 📰 Verified Trade Journal Entry & Evidence Dossier")
                st.markdown(f"""
> **\"{r['Evidence']}\"**
> 
> — *Source:* [{r['Source_Name']}]({r['Source_URL']}) | *Published Date:* `{r['Publication_Date']}` | *Geography:* Maharashtra
""")

        elif "Quarter" in selected_tl:
            q_code = "Q2" if "Q2" in selected_tl else "Q3"
            q_match = q_summary[q_summary["Quarter_Code"] == q_code]
            if not q_match.empty:
                qr = q_match.iloc[0]
                qc1, qc2, qc3 = st.columns(3)
                with qc1:
                    st.info(f"**Model Forecast: {qr['Quarter']}**")
                    st.metric("Quarterly Average Price", f"{qr['Model_Avg_Price']:.2f} Rs/kg TS", f"{qr['Model_Net_Change']:+.2f} Net Change ({qr['Model_Quarter_Trend']})")
                    st.markdown(f"💡 **Model Quarterly Driver:** {qr['Model_Reason']}")
                with qc2:
                    st.info("**External Quarterly Intelligence**")
                    q_mid = f"{qr['External_Expected_Price']:.2f} Rs/kg TS" if pd.notna(qr["External_Expected_Price"]) else "Consensus Trend"
                    st.metric("Expected Quarterly Trend", f"{qr['External_Expected_Trend']} ({q_mid})", f"Range: {qr['External_Expected_Range']}")
                    st.caption(f"Source: {qr['Source_Name']} (Pub: {qr['Publication_Date']})")
                    st.markdown(f"🏷️ **Trade Quarterly Driver:** {qr['External_Reason']}")
                with qc3:
                    st.info("**Quarterly Consensus Alignment**")
                    if qr["Quarterly_Agreement"] == "AGREE":
                        st.success("✅ **QUARTERLY CONSENSUS AGREE**")
                    elif qr["Quarterly_Agreement"] == "PARTIAL_AGREEMENT":
                        st.warning("⚠️ **PARTIAL ALIGNMENT**")
                    else:
                        st.error("❌ **DIVERGENCE IN QUARTERLY OUTLOOK**")

                st.markdown("#### 📰 Quarterly Trade Outlook Entry")
                st.markdown(f"""
> **\"{qr['Evidence']}\"**
> 
> — *Source:* [{qr['Source_Name']}]({qr['Source_URL']}) | *Published Date:* `{qr['Publication_Date']}`
""")

        else:
            st.info("### 🔭 Extended 1-Year Forward Market Speculation (H1 2027)")
            if ext_records:
                for er in ext_records:
                    st.markdown(f"""
**Forecast Period:** `{er.get('forecast_period')}`  
**Source Organization:** [{er.get('source_name')}]({er.get('source_url')}) | *Published:* `{er.get('publication_date')}`  
**Expected Direction:** `{er.get('expected_change')}`  

> **Verbatim Market Intelligence Quote:**  
> \"{er.get('evidence')}\"

*Analysis:* {er.get('remarks')}
""")
            else:
                st.write("No extended 1-year records currently archived.")

        st.markdown("---")

        # 3. Visual Trajectory: Model Forecast vs External Speculation Corridor
        st.markdown("### 📈 Visual Comparison: Model Baseline vs External Speculation Corridor")
        
        chart_df_list = []
        # Historical actuals (last 12 months)
        for dt, val in hist.tail(12).items():
            chart_df_list.append({"Month": pd.to_datetime(dt), "Price": float(val), "Series": "Historical Actual", "Min": float(val), "Max": float(val)})
        
        # Model forecast points
        for idx, r in spec_monthly.iterrows():
            chart_df_list.append({"Month": pd.to_datetime(r["Date"]), "Price": float(r["Model_Forecast"]), "Series": "Model Baseline Forecast", "Min": float(r["Model_Forecast"]), "Max": float(r["Model_Forecast"])})
            if pd.notna(r["External_Expected_Price"]):
                chart_df_list.append({"Month": pd.to_datetime(r["Date"]), "Price": float(r["External_Expected_Price"]), "Series": "External Trade Expectation", "Min": float(r["External_Min"]) if pd.notna(r["External_Min"]) else float(r["External_Expected_Price"]), "Max": float(r["External_Max"]) if pd.notna(r["External_Max"]) else float(r["External_Expected_Price"])})
        
        comp_chart_df = pd.DataFrame(chart_df_list)
        
        line_chart = alt.Chart(comp_chart_df).mark_line(point=True).encode(
            x=alt.X("Month:T", title=None, axis=alt.Axis(format="%b %y")),
            y=alt.Y("Price:Q", scale=alt.Scale(zero=False), title="Procurement Price (Rs/kg TS)"),
            color=alt.Color("Series:N", scale=alt.Scale(domain=["Historical Actual", "Model Baseline Forecast", "External Trade Expectation"], range=[GREY, BLUE, ORANGE]), title=None),
            tooltip=[alt.Tooltip("Month:T", format="%b %Y"), "Series:N", alt.Tooltip("Price:Q", format=".2f")]
        ).properties(height=360)
        
        st.altair_chart(line_chart, use_container_width=True)

        # 4. Comprehensive Monthly Comparison Table (WITH RANGE & CORRELATION)
        st.markdown("### 📋 Monthly Forecast vs External Speculation Ledger")
        display_df = spec_monthly[[
            "Horizon", "Period", "Model_Forecast", "Model_MoM_Change", "Model_Direction",
            "External_Expected_Price", "External_Range", "External_Direction", "Agreement", "Price_Gap", "Source_Name"
        ]].copy()
        display_df.rename(columns={
            "Model_Forecast": "Model (Rs/kg TS)",
            "Model_MoM_Change": "Model MoM",
            "Model_Direction": "Model Direction",
            "External_Expected_Price": "External Spec (Rs/kg TS)",
            "External_Range": "External Range (Rs/kg TS)",
            "External_Direction": "External Dir",
            "Price_Gap": "Model Gap (Rs/kg TS)",
            "Source_Name": "Source"
        }, inplace=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption(f"Statistical Pearson Correlation between Model Baseline Forecast and External Speculation: **r = {corr_val:+.2f}** (indicating highly synchronized forward trajectories).")

        # 5. Dedicated Quarterly Forecast vs External Trade Consensus Ledger
        st.markdown("### 📊 Quarterly Forecast vs External Trade Consensus Ledger")
        q_display = q_summary[[
            "Quarter", "Model_Avg_Price", "Model_Net_Change", "Model_Quarter_Trend", "Model_Reason",
            "External_Expected_Price", "External_Expected_Range", "External_Expected_Trend", "Quarterly_Agreement", "Source_Name"
        ]].copy()
        q_display.rename(columns={
            "Model_Avg_Price": "Model Avg (Rs/kg TS)",
            "Model_Net_Change": "Net Move (Rs/kg TS)",
            "Model_Quarter_Trend": "Model Trend",
            "Model_Reason": "Model Economic Driver",
            "External_Expected_Price": "External Midpoint (Rs/kg TS)",
            "External_Expected_Range": "External Expected Range",
            "External_Expected_Trend": "External Trend",
            "Quarterly_Agreement": "Quarterly Agreement",
            "Source_Name": "Trade Source"
        }, inplace=True)
        st.dataframe(q_display, use_container_width=True, hide_index=True)

        # 6. Methodological Integrity Notice
        with st.expander("🛡️ Data Governance & Anti-Leakage Protocol"):
            st.markdown("""
- **Strict Farmer Procurement Scope:** External expectations are strictly farmer gate procurement rates paid by dairy cooperatives and private processors in Maharashtra. Consumer retail MRP and synthetic commodity conversions (SMP/Ghee) are prohibited.
- **Zero Data Leakage:** External expectations are archived in an independent JSON registry (`external_market_speculations.json`) and loaded exclusively at evaluation time. They do not alter training features, lags, or model weights.
- **Auditable Live Portals:** Every entry links directly to authoritative live portals of AgriWatch, NDDB, MMPWA/DAHD, The Hindu BusinessLine, and the Indian Dairy Association.
""")

# ================================================================== TAB 2: SCENARIO FORECAST & WHAT-IF
with tab1:
    if not (mshocks or wshocks):
        st.info("No shock set yet - the scenario equals the baseline. Pick a **Quick scenario** or move the sliders in the sidebar.")
    else:
        parts = [f"{se.MARKET_DRIVERS[k][0]} {v:+g}{'%' if se.MARKET_DRIVERS[k][1] == 'pct' else ' ' + se.MARKET_DRIVERS[k][2]}" for k, v in mshocks.items()]
        parts += [f"{se.WEATHER_DRIVERS[k][0]} {v:+g} {se.WEATHER_DRIVERS[k][1]}" for k, v in wshocks.items()]
        st.markdown("**Scenario:** " + " | ".join(parts))
        used_m = E.used_inputs(mt, method)
        ign = [se.MARKET_DRIVERS[k][0] for k in mshocks if not any(f in used_m for f in se.DRIVER_FEATURES[k])]
        if ign:
            st.warning(f"**{method}** has no input built from: {', '.join(ign)}. These shocks cannot move this model's forecast "
                       f"(not a bug). Pick another model in the sidebar (e.g. RecursiveX XGB or Direct Blend) to see their effect.")

    hs = sorted({1, min(3, H), H})
    cols = st.columns(len(hs))
    for c, h in zip(cols, hs):
        c.metric(f"N+{h} ({months.loc[h]:%b %Y})", f"{a_s.loc[h]:.1f} Rs/kg TS", f"{delta.loc[h]:+.2f} vs baseline")

    f_months = [hist.index[-1]] + list(months.loc[1:H])
    df = pd.concat([
        pd.DataFrame({"Month": hist.index, "Rate": hist.to_numpy(), "Series": "Actual"}),
        pd.DataFrame({"Month": f_months, "Rate": [hist.iloc[-1]] + list(ab.loc[1:H]), "Series": "Baseline forecast"}),
        pd.DataFrame({"Month": f_months, "Rate": [hist.iloc[-1]] + list(a_s.loc[1:H]), "Series": "Scenario forecast"})])
    line = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("Month:T", title=None, axis=alt.Axis(format="%b %y")),
        y=alt.Y("Rate:Q", scale=alt.Scale(zero=False), title="Rs per kg TS"),
        color=alt.Color("Series:N", scale=alt.Scale(domain=["Actual", "Baseline forecast", "Scenario forecast"], range=[GREY, BLUE, ORANGE]), title=None),
        tooltip=[alt.Tooltip("Month:T", format="%b %Y"), alt.Tooltip("Rate:Q", format=".2f"), "Series:N"]).properties(height=340)
    st.altair_chart(line, use_container_width=True)

    d_df = pd.DataFrame({"Month": [f"N+{h}  {months.loc[h]:%b %Y}" for h in range(1, H + 1)], "Change": [delta.loc[h] for h in range(1, H + 1)]})
    bars = alt.Chart(d_df).mark_bar().encode(
        x=alt.X("Month:N", sort=None, title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("Change:Q", title="Scenario minus baseline (Rs/kg TS)"),
        color=alt.condition(alt.datum.Change > 0, alt.value("#2e7d32"), alt.value(RED)),
        tooltip=["Month:N", alt.Tooltip("Change:Q", format="+.2f")]).properties(height=240)
    st.altair_chart(bars, use_container_width=True)

    st.subheader("Change by district (Rs/kg TS, scenario minus baseline)")
    dd = (scn.set_index([s1.DISTRICT, "horizon"])[method] - base.set_index([s1.DISTRICT, "horizon"])[method]).unstack("horizon")
    dd = dd.loc[[d for d in S.districts if d in sel], list(range(1, H + 1))]
    dd.loc["ALL selected (volume weighted)"] = [delta.loc[h] for h in range(1, H + 1)]
    dd.columns = [f"N+{h} {months.loc[h]:%b %y}" for h in dd.columns]
    st.dataframe(dd.round(2))

    with st.expander("What the model actually saw: inputs that changed"):
        hz = st.selectbox("Horizon", list(range(1, H + 1)), index=min(2, H - 1), format_func=lambda h: f"N+{h} ({months.loc[h]:%b %Y})", key="chg_h")
        ci = run_changed(E, EKEY, mt, _tup(shocks), ramp, _tup(weather), hz)
        if ci.empty:
            st.write("No model input changes in this scenario.")
        else:
            used = E.used_inputs(mt, method)
            ci.insert(2, "Used by this model", ci["Model input"].isin(used).map({True: "yes", False: "no"}))
            st.dataframe(ci.round(3))
            st.caption("Averages over the selected districts at that horizon. A shock reaches the model only through these derived inputs "
                       "(3 / 6-month % changes, ratios, ONI ...). Inputs marked 'no' are ignored by the chosen model.")
    if mshocks:
        with st.expander("Market series: actual vs scenario (last 6 months)"):
            mk = se.shocked_market(E.market, mshocks, S.origin, ramp)
            t = pd.DataFrame(index=[f"{d:%b %Y}" for d in E.market.index[-6:]])
            for k in mshocks:
                t[f"{se.MARKET_DRIVERS[k][0]} - actual"] = E.market[k].astype(float).tail(6).to_numpy()
                t[f"{se.MARKET_DRIVERS[k][0]} - scenario"] = mk[k].tail(6).to_numpy()
            st.dataframe(t.round(2))

# ================================================================== TAB 3: TORNADO
with tab2:
    st.subheader("Which market driver moves the forecast most?")
    st.write("Each driver is pushed **alone**, up and down by the same size, and the forecast of the chosen month is compared with the baseline. "
             "Everything else stays as it is.")
    c1, c2 = st.columns(2)
    size = c1.slider("Size of the move for price drivers (%)", 5, 30, 10, 5)
    th = c2.selectbox("Forecast month", list(range(1, E.H + 1)), index=2, format_func=lambda h: f"N+{h} ({months.loc[h]:%b %Y})", key="tor_h")
    tor = run_tornado(E, EKEY, mt, method, float(size), ramp, int(th))
    tor_used, tor_ign = tor[tor["Used by this model"] == "yes"], tor[tor["Used by this model"] == "no"]
    if tor_used.empty:
        st.info("The chosen model has no input built from any market driver, so there is nothing to plot. Pick another model in the sidebar.")
    else:
        st.altair_chart(diverging_bars(tor_used, "Driver", "Rise (+) effect", "Fall (-) effect", "Driver rises", "Driver falls", "Swing",
                                       f"Change of N+{th} forecast vs baseline (Rs/kg TS), volume-weighted over all districts"), use_container_width=True)
    if not tor_ign.empty:
        st.warning(f"**{method}** has no input built from: **{', '.join(tor_ign['Driver'])}**. They are shown as 'not used' (not as a measured 0) "
                   f"and left out of the chart. To see their effect choose a model that reads them (e.g. RecursiveX XGB or Direct Blend) in the sidebar.")
    show = tor.drop(columns=["Key"]).copy()
    for c in ("Rise (+) effect", "Fall (-) effect", "Swing"):
        show[c] = show[c].map(lambda v: "not used" if pd.isna(v) else f"{v:.2f}")
    st.dataframe(show)
    st.caption("ONI and monsoon are anomaly series, so they are moved by +-0.5 x their historical standard deviation instead of a %. "
               "'not used' = the model has no input built from that driver, so it cannot react to it. "
               "A real 0.00 (or a one-sided effect) for a used driver means the tree model did not cross any split for a move of that size.")

# ================================================================== TAB 4: SENSITIVITY CURVE
with tab3:
    st.subheader("Sensitivity curve: forecast change as ONE driver goes from falling to rising")
    c1, c2, c3 = st.columns(3)
    key = c1.selectbox("Driver", list(se.MARKET_DRIVERS), format_func=lambda k: se.MARKET_DRIVERS[k][0])
    kind, unit = se.MARKET_DRIVERS[key][1], se.MARKET_DRIVERS[key][2]
    rng = c2.slider(f"Range (+- {unit})", 5.0, 40.0, 20.0, 5.0) if kind == "pct" else c2.slider(f"Range (+- {unit})", 0.5, 3.0, 1.0, 0.5)
    hzs = c3.multiselect("Forecast months", list(range(1, E.H + 1)), default=[h for h in (1, 3, 6) if h <= E.H], format_func=lambda h: f"N+{h}")
    if hzs:
        step = 5.0 if kind == "pct" else 0.25
        vals = tuple(float(v) for v in np.arange(-rng, rng + 1e-9, step))
        sw = run_sweep(E, EKEY, mt, method, key, vals, ramp, tuple(sorted(hzs)))
        ln = alt.Chart(sw).mark_line(point=True).encode(
            x=alt.X("Shock:Q", title=f"{se.MARKET_DRIVERS[key][0]}: change ({unit})"),
            y=alt.Y("Change vs baseline:Q", title="Forecast change vs baseline (Rs/kg TS)"),
            color=alt.Color("Horizon:N", title=None),
            tooltip=[alt.Tooltip("Shock:Q"), "Horizon:N", alt.Tooltip("Change vs baseline:Q", format="+.2f")]).properties(height=380)
        zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(strokeDash=[4, 4], color=GREY).encode(y="y:Q")
        st.altair_chart(ln + zero, use_container_width=True)
        st.caption("The models are tree / ridge based and trained on a short history, so the curves can be lumpy (steps, flat parts, even a direction "
                   "that looks odd economically). That is what the model learned from the data - read it as model behaviour, not as a price elasticity.")
    else:
        st.info("Choose at least one forecast month.")

# ================================================================== TAB 5: EVERY MODEL INPUT
with tab4:
    st.subheader("Every model input, one at a time")
    st.write("Each input of the model (weather, price momentum, feed, dairy prices, milk quality, calendar ...) is pushed by **+-1 standard deviation "
             "of its own history** while all others stay fixed. This is the full 'which feature matters' view; the market drivers of the other tabs are "
             "built from the 'market' inputs found here.")
    hz4 = st.selectbox("Forecast month", list(range(1, E.H + 1)), index=2, format_func=lambda h: f"N+{h} ({months.loc[h]:%b %Y})", key="inp_h")
    inp = run_inputs(E, EKEY, mt, int(hz4))
    st.caption(f"Model used here: {'Cow Ensemble v2' if mt == 'COW' else 'Direct Blend (Ridge + XGB of that horizon)'} - it maps inputs to the forecast in one step. "
               "(RecursiveX feeds its own forecasts back, so single inputs cannot be pushed in the same clean way.)")
    top = st.slider("Show the top N inputs", 5, 40, 15)
    t4 = inp.head(top)
    st.altair_chart(diverging_bars(t4, "Model input", "Input +1 std", "Input -1 std", "Input rises 1 std", "Input falls 1 std", "Swing",
                                   f"Change of N+{hz4} forecast (Rs/kg TS)"), use_container_width=True)
    grp = inp.groupby("Group")["Swing"].sum().sort_values(ascending=False).reset_index()
    st.markdown("**By group** (sum of the swings of its inputs)")
    st.altair_chart(alt.Chart(grp).mark_bar(color=BLUE).encode(
        y=alt.Y("Group:N", sort="-x", title=None), x=alt.X("Swing:Q", title="Total swing (Rs/kg TS)"),
        tooltip=["Group:N", alt.Tooltip("Swing:Q", format=".2f")]).properties(height=max(200, 32 * len(grp))), use_container_width=True)
    t4s = t4.copy()
    for c in ("1 std (history)", "Input +1 std", "Input -1 std", "Swing"):
        t4s[c] = t4s[c].round(2)
    st.dataframe(t4s)
    st.caption("Pushing one input alone is a what-if on the model, not a realistic scenario: in real life many inputs move together "
               "(e.g. rate_t, diff_3 and dev_ma3 all move when the rate moves).")

# ================================================================== TAB 6: FEATURE IMPORTANCE
with tab6:
    st.subheader("What does each model lean on? (feature importance, cow and buffalo separately)")
    st.write("Every model input is **shuffled** (its link to the target is broken) and we measure how much the model's error rises. "
             "A big rise means the model relies on that input. Shares add up to 100 % per milk type. "
             "Cow = Cow Ensemble v2 (average over N+1..N+6). Buffalo = RecursiveX XGB one-step model (the buffalo headline model).")
    topn = st.slider("Show the top N inputs", 5, 30, 12, key="fi_top")
    fcols = st.columns(2)
    for col, m in zip(fcols, s1.MILK_TYPES):
        fi_df, fi_name = run_importance(E, EKEY, m)
        with col:
            st.markdown(f"### {m.title()}")
            st.caption(f"Model: {fi_name}")
            top_df = fi_df.head(topn)
            st.altair_chart(alt.Chart(top_df).mark_bar(color=BLUE if m == "COW" else ORANGE).encode(
                y=alt.Y("Model input:N", sort="-x", title=None), x=alt.X("Importance (%):Q", title="Share of total importance (%)"),
                tooltip=["Model input:N", "Group:N", alt.Tooltip("Importance (%):Q", format=".1f"), "Meaning:N"]
            ).properties(height=max(220, 26 * len(top_df))), use_container_width=True)
            g = fi_df.groupby("Group")["Importance (%)"].sum().sort_values(ascending=False).reset_index()
            st.markdown("**By group**")
            st.altair_chart(alt.Chart(g).mark_bar(color=GREY).encode(
                y=alt.Y("Group:N", sort="-x", title=None), x=alt.X("Importance (%):Q", title="Share (%)"),
                tooltip=["Group:N", alt.Tooltip("Importance (%):Q", format=".1f")]
            ).properties(height=max(180, 28 * len(g))), use_container_width=True)
            t = top_df[["Model input", "Group", "Importance (%)", "MAE rise when shuffled", "Meaning"]].copy()
            t["Importance (%)"] = t["Importance (%)"].round(1)
            t["MAE rise when shuffled"] = t["MAE rise when shuffled"].round(3)
            st.dataframe(t, hide_index=True)
    st.caption("Importance shows what the model USES, not what CAUSES the rate to change. Inputs that are identical for all districts in a month "
               "(ONI, CPI, SMP, incentive ...) can act as a calendar stamp that tells the model which period a row belongs to, "
               "so a high rank for them is not proof of a market effect. Computed on the training rows (in-sample).")

# ================================================================== TAB 7: HOW TO READ THIS
with tab5:
    st.markdown(f"""
### What this app does
* It takes the **same models as your step files** (fitted on all data up to **{S.origin:%b %Y}**) and runs them again with *changed inputs*.
* **Baseline** = the normal forecast. **Scenario** = the forecast after your market / weather change. **Change = scenario - baseline** (Rs per kg TS).
* The models are **never refitted** on the shocked numbers; only the forecast-time inputs change.

### How a market shock reaches the model
1. The chosen market series (e.g. SMP) is changed in the last months up to {S.origin:%b %Y} (full change in {S.origin:%b %Y}; it builds up over the "how fast" setting).
2. The project's own step 1 code recomputes the inputs derived from it: 3 / 6-month % changes, feed/maize and ghee/SMP ratios, ONI inputs.
3. Because step 1 *projects* market series forward as `last value + k x average of the last 6 monthly changes`, a jump in the last months also
    makes the projection keep moving in the same direction. This is the model's design, not something added by the app.
4. The fitted models read the new inputs and give a new forecast. RecursiveX (buffalo headline) does this month after month, feeding its own forecasts back.

### Things to keep in mind
* **Model behaviour, not economics.** The models learned patterns from ~4-5 years of data and a handful of districts. A response can be lumpy, flat or
  even the "wrong" sign economically. It tells you what the model does, not what will happen in the market.
* **"not used" is meaningful.** Cow Ensemble v2 deliberately has no ONI, monsoon or butter inputs, so those drivers cannot move it. The app says "not used" instead of showing a fake 0.
* **Feature importance** (own tab) shows, for cow and buffalo separately, which inputs the fitted model leans on most.
* **Distance matters.** N+1..N+4 are the short-term horizon, N+5..N+6 indicative, beyond that outside the tested range.
* The weather sliders only shift the *expected* weather of coming months, as in the LOW / HIGH weather band of step 3.
* Volume weights = total solids of {S.origin:%b %Y} per district.
""")
    if st.button("Verify: engine baseline equals step 2 forecast"):
        with st.spinner("Comparing with step2_backtest.forecast_origin ..."):
            diff = E.self_check(mt)
        (st.success if diff < 1e-6 else st.error)(f"Largest difference to the step-2 forecast of {mt.title()}: {diff:.2e} Rs/kg TS")

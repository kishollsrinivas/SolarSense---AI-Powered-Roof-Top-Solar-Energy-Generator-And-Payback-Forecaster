import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re

from utils.simulation import simulate_24h_dispatch, get_available_dataset_dates, predict_battery_state

st.set_page_config(
    page_title="SolarSense: AI Microgrid Management Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .terminal-box {
        background-color: #0b1320;
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 16px;
        font-family: 'Courier New', Courier, monospace;
        color: #10b981;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .metric-title { color: #9CA3AF; font-size: 0.82rem; font-weight: 500; margin-bottom: 3px; }
    .metric-val { color: #F9FAFB; font-size: 1.45rem; font-weight: 700; }
    .metric-sub-green { color: #10B981; font-size: 0.75rem; margin-top: 3px; }
    .metric-sub-red { color: #EF4444; font-size: 0.75rem; margin-top: 3px; }
    .metric-sub-blue { color: #38BDF8; font-size: 0.75rem; margin-top: 3px; }
    .compact-docket-ribbon {
        background: #111827;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 20px;
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        align-items: center;
        justify-content: space-between;
    }
    .docket-item { display: flex; flex-direction: column; }
    .docket-lbl { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }
    .docket-val { font-size: 1.05rem; font-weight: 700; color: #f8fafc; }
</style>
""", unsafe_allow_html=True)

if "engine_started" not in st.session_state:
    st.session_state["engine_started"] = False

if "show_inverter_diag" not in st.session_state:
    st.session_state["show_inverter_diag"] = False

if "last_uploaded_files" not in st.session_state:
    st.session_state["last_uploaded_files"] = []

def parse_single_tangedco_bill(uploaded_file):
    extracted = {
        "valid": False, "filename": uploaded_file.name, "consumer_no": "Unknown",
        "date_from": "N/A", "date_to": "N/A", "billing_days": 60,
        "total_units": 0, "daily_demand": 0.0, "bill_amount": 0.0
    }
    try:
        uploaded_file.seek(0)
        filename = uploaded_file.name.lower()
        text_content = ""
        
        if filename.endswith(".txt"):
            text_content = uploaded_file.read().decode("utf-8", errors="ignore")
        elif filename.endswith(".pdf"):
            try:
                import pypdf
                reader = pypdf.PdfReader(uploaded_file)
                for page in reader.pages:
                    txt = page.extract_text()
                    if txt: text_content += txt + "\n"
            except Exception: pass
        elif filename.endswith(".docx"):
            try:
                import docx
                doc = docx.Document(uploaded_file)
                for p in doc.paragraphs:
                    if p.text: text_content += p.text + "\n"
                for table in doc.tables:
                    for row in table.rows:
                        text_content += " | ".join(cell.text.strip() for cell in row.cells) + "\n"
            except Exception: pass

        if text_content:
            upper_text = text_content.upper()
            has_org_signature = any(sig in upper_text for sig in ["TANGEDCO", "TAMIL NADU GENERATION", "ELECTRICITY BILL", "LT-1A"])
            has_unit_signature = any(sig in upper_text for sig in ["TOTAL UNITS", "BILLED UNITS", "METER READING", "CONSUMPTION"])
            
            if not (has_org_signature and has_unit_signature):
                return extracted

            extracted["valid"] = True
            cons_match = re.search(r'Consumer\s+Service\s+No\.?[\s|:]+([A-Za-z0-9\-]+)', text_content, re.IGNORECASE)
            if cons_match: extracted["consumer_no"] = cons_match.group(1).strip()
            
            prev_d = re.search(r'Previous\s+Reading\s+Date[\s|:]+([0-9]{2}/[0-9]{2}/[0-9]{4})', text_content, re.IGNORECASE)
            curr_d = re.search(r'Current\s+Reading\s+Date[\s|:]+([0-9]{2}/[0-9]{2}/[0-9]{4})', text_content, re.IGNORECASE)
            if prev_d: extracted["date_from"] = prev_d.group(1)
            if curr_d: extracted["date_to"] = curr_d.group(1)

            days_match = re.search(r'Billing\s+Days[\s|:]+([0-9]+)', text_content, re.IGNORECASE)
            if days_match: extracted["billing_days"] = int(days_match.group(1))

            units_match = re.search(r'Total\s+Units\s+Consumed[\s|:]+([0-9,]+)', text_content, re.IGNORECASE)
            if units_match:
                extracted["total_units"] = int(units_match.group(1).replace(",", ""))
            else:
                gen_match = re.search(r'(?:Billed\s+Units|Total\s+Units)[\s|:]+([0-9,]+)', text_content, re.IGNORECASE)
                if gen_match: extracted["total_units"] = int(gen_match.group(1).replace(",", ""))

            daily_match = re.search(r'Average\s+Daily\s+Consumption[\s|:]+([0-9\.]+)', text_content, re.IGNORECASE)
            if daily_match:
                extracted["daily_demand"] = float(daily_match.group(1))
            else:
                extracted["daily_demand"] = round(extracted["total_units"] / max(1, extracted["billing_days"]), 2)

            amt_match = re.search(r'NET\s+AMOUNT\s+PAYABLE.*?([0-9,]+\.?[0-9]*)', text_content, re.IGNORECASE)
            if amt_match: extracted["bill_amount"] = float(amt_match.group(1).replace(",", ""))
    except Exception: pass
    return extracted

def calculate_tangedco_domestic_bill(units):
    if units <= 100:
        return 0
    elif units <= 200:
        energy_charges = (units - 100) * 2.35
    elif units <= 400:
        energy_charges = (100 * 2.35) + ((units - 200) * 4.70)
    elif units <= 500:
        energy_charges = (100 * 2.35) + (200 * 4.70) + ((units - 400) * 6.30)
    else:
        energy_charges = (100 * 2.35) + (200 * 4.70) + (100 * 6.30) + ((units - 500) * 8.40)
    
    electricity_duty = energy_charges * 0.05
    fixed_surcharge = (units - 100) * 0.10
    return int(round(energy_charges + electricity_duty + fixed_surcharge))

# --- SIDEBAR: INGESTION & AUTO-RESET MANAGEMENT ---
st.sidebar.subheader("📥 Ingestion Dock (TANGEDCO)")
uploaded_bills = st.sidebar.file_uploader(
    "Upload TANGEDCO Bills (.pdf, .docx, .txt)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if not uploaded_bills:
    st.session_state["engine_started"] = False
    st.session_state["show_inverter_diag"] = False
    st.session_state["last_uploaded_files"] = []
    st.title("⚡ SolarSense: AI Microgrid Predictive Engine")
    st.info("Please upload your TANGEDCO electricity bill to initialize telemetry models.")
    st.stop()

current_file_signature = [f"{b.name}_{b.size}" for b in uploaded_bills]
if current_file_signature != st.session_state["last_uploaded_files"]:
    st.session_state["engine_started"] = False
    st.session_state["show_inverter_diag"] = False
    st.session_state["last_uploaded_files"] = current_file_signature

parsed_results = [parse_single_tangedco_bill(b) for b in uploaded_bills[:4]]
invalid_files = [b["filename"] for b in parsed_results if not b["valid"]]

if invalid_files:
    st.session_state["engine_started"] = False
    st.sidebar.error(f"❌ Invalid Bill Detected: {', '.join(invalid_files)}")
    st.stop()

parsed_bills_list = parsed_results
num_bills = len(parsed_bills_list)
primary_bill = parsed_bills_list[0]

total_all_units = sum(b["total_units"] for b in parsed_bills_list)
total_all_days = sum(b["billing_days"] for b in parsed_bills_list)
overall_avg_daily_demand = round(total_all_units / max(1, total_all_days), 2)
effective_monthly_units = int(round(overall_avg_daily_demand * 30.0))
active_cycle_days = max(1, int(round(total_all_days / num_bills)))

dynamic_array_kw = round(float(np.clip(overall_avg_daily_demand / 4.0, 1.0, 15.0)), 1)
dynamic_rooftop_sqft = int(dynamic_array_kw * 100)
dynamic_battery_kwh = round(float(np.clip(overall_avg_daily_demand * 0.55, 1.0, 20.0)), 1)

st.sidebar.success(f"Verified & Loaded {num_bills} TANGEDCO Bill(s)")

if not st.session_state["engine_started"]:
    if st.sidebar.button("🚀 Start Prediction", type="primary", width="stretch"):
        st.session_state["engine_started"] = True
        st.rerun()

if not st.session_state["engine_started"]:
    st.title("⚡ SolarSense: AI Microgrid Predictive Engine")
    st.info("Files loaded successfully. Click 'Start Prediction' in sidebar to run dashboard.")
    st.stop()

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("---")
st.sidebar.subheader("🏠 Rooftop & Solar Sizing")
usable_area = st.sidebar.number_input("Usable Rooftop Area (sq.ft)", min_value=100, max_value=2500, value=dynamic_rooftop_sqft, step=25)
recommended_pv_kw = round(usable_area / 100.0, 1)

st.sidebar.subheader("⚙️ Battery Bank Specs")
selected_battery_kwh = st.sidebar.slider("Target Battery Capacity (kWh)", 1.0, 20.0, dynamic_battery_kwh, 0.1)
reserve_floor_pct = st.sidebar.slider("Emergency Reserve Floor (%)", 10, 50, 30, 5)

timeline_resolution = st.sidebar.radio("Sampling Interval:", ["15-Min (Plant SCADA)", "5-Min (Inverter Native)"], index=0)
toggle_inverter_loss_cols = st.sidebar.toggle("⚡ Show DC-to-AC Efficiency Loss", value=True)

selected_weather = st.sidebar.selectbox("Weather Based Data:", get_available_dataset_dates(), index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Diagnostic Stress Testing")
battery_bms_fault = st.sidebar.checkbox("Simulate Battery Fault / Zero Charge (BMS Trip)", value=False)

if "fan_count" not in st.session_state: st.session_state["fan_count"] = 2
if "light_count" not in st.session_state: st.session_state["light_count"] = 4
if "fridge_count" not in st.session_state: st.session_state["fridge_count"] = 1
if "router_count" not in st.session_state: st.session_state["router_count"] = 1

total_appliance_load_kw = (
    (st.session_state["fan_count"] * 75) +
    (st.session_state["light_count"] * 14) +
    (st.session_state["fridge_count"] * 180) +
    (st.session_state["router_count"] * 15)
) / 1000.0

df_dispatch, summary, df_5min, df_15min = simulate_24h_dispatch(
    plant_capacity_kw=recommended_pv_kw,
    nominal_battery_kwh=selected_battery_kwh,
    emergency_reserve_floor_pct=reserve_floor_pct,
    weather_profile=selected_weather,
    daily_demand_kwh=overall_avg_daily_demand,
    battery_fault=battery_bms_fault,
    selected_emergency_load_kw=max(0.05, total_appliance_load_kw),
    resolution=timeline_resolution
)

# 1. Summary Ribbon Card
st.markdown(f"""
<div class="compact-docket-ribbon">
    <div class="docket-item"><span class="docket-lbl">CONSUMER NO</span><span class="docket-val">{primary_bill['consumer_no']}</span></div>
    <div class="docket-item"><span class="docket-lbl">BILLING UNITS</span><span class="docket-val">{effective_monthly_units} kWh/Mo</span></div>
    <div class="docket-item"><span class="docket-lbl">DAILY DEMAND</span><span class="docket-val">{overall_avg_daily_demand:.2f} kWh/Day</span></div>
    <div class="docket-item"><span class="docket-lbl">ROOFTOP AREA</span><span class="docket-val">{usable_area} sq.ft</span></div>
    <div class="docket-item"><span class="docket-lbl">RECOMMENDED ARRAY</span><span class="docket-val" style="color:#60a5fa;">{recommended_pv_kw} kW</span></div>
    <div class="docket-item"><span class="docket-lbl">SELECTED STORAGE</span><span class="docket-val" style="color:#a78bfa;">{selected_battery_kwh} kWh</span></div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# DUAL EB BILL PREDICTION ENGINE
# =====================================================================
if num_bills == 1 and primary_bill["bill_amount"] > 0:
    pred_bill_baseline = int(round(primary_bill["bill_amount"]))
    projected_cycle_units = primary_bill["total_units"]
else:
    projected_cycle_units = int(round(overall_avg_daily_demand * float(active_cycle_days)))
    pred_bill_baseline = calculate_tangedco_domestic_bill(projected_cycle_units)

daily_grid_units = summary['total_import_kwh']
cycle_grid_import_kwh = daily_grid_units * float(active_cycle_days)
post_solar_payable_units = int(round(cycle_grid_import_kwh))
pred_bill_post_solar = calculate_tangedco_domestic_bill(post_solar_payable_units)

net_savings_cycle = max(0, pred_bill_baseline - pred_bill_post_solar)
savings_pct = round((net_savings_cycle / max(1, pred_bill_baseline)) * 100, 1)

st.markdown("### 📊 Dual EB Prediction: Before vs After Solar Installation")

d_col1, d_col2, d_col3, d_col4 = st.columns([1, 1, 1.2, 1.2])

with d_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Night Grid Draw (Daily)</div>
        <div class="metric-val" style="color: #F87171;">{summary['total_import_kwh']} <span style="font-size:1rem;">kWh</span></div>
        <div class="metric-sub-red">↑ Auto-Switched to EB Grid</div>
    </div>
    <div class="metric-card" style="margin-top: 10px;">
        <div class="metric-title">Bi-Monthly Total Savings</div>
        <div class="metric-val" style="color: #34D399;">₹ {net_savings_cycle:,}</div>
        <div class="metric-sub-green">↓ {savings_pct}% Bill Reduction</div>
    </div>
    """, unsafe_allow_html=True)

with d_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Solar Net Surplus Export</div>
        <div class="metric-val" style="color: #38BDF8;">{summary['total_export_kwh']} <span style="font-size:1rem;">kWh</span></div>
        <div class="metric-sub-blue">↑ Fed to TANGEDCO Grid</div>
    </div>
    <div class="metric-card" style="margin-top: 10px;">
        <div class="metric-title">Post-Solar {active_cycle_days}-Day Units</div>
        <div class="metric-val" style="color: #FBBF24;">{post_solar_payable_units} <span style="font-size:1rem;">kWh</span></div>
        <div class="metric-sub-green">↓ Net Grid Draw (Down from {projected_cycle_units} kWh)</div>
    </div>
    """, unsafe_allow_html=True)

with d_col3:
    st.markdown(f"""
    <div class="metric-card" style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 8px;">
        <div class="metric-title" style="text-align: center; width: 100%;">Baseline Upcoming Bill ({active_cycle_days} Days)</div>
    """, unsafe_allow_html=True)
    
    gauge_val_pre = min(100.0, (pred_bill_baseline / max(1.0, pred_bill_baseline * 1.1)) * 100.0)
    fig_ring_pre = go.Figure(go.Pie(
        values=[gauge_val_pre, 100 - gauge_val_pre],
        hole=0.78,
        sort=False,
        direction='clockwise',
        showlegend=False,
        hoverinfo='none',
        textinfo='none',
        marker=dict(colors=['#06b6d4', '#1e293b'], line=dict(width=0))
    ))
    fig_ring_pre.update_layout(
        height=145, margin=dict(l=0, r=0, t=5, b=5),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(
            text=f"<b style='font-size:1.15rem; color:#f8fafc;'>₹ {pred_bill_baseline:,}</b><br><span style='font-size:0.72rem; color:#94a3b8;'>Pre-Solar Cost</span>",
            x=0.5, y=0.5, showarrow=False
        )]
    )
    st.plotly_chart(fig_ring_pre, width="stretch", config={'displayModeBar': False})
    st.markdown(f'<div class="metric-sub-red" style="margin-top:-10px; text-align:center;">Full EB Tariff Load</div></div>', unsafe_allow_html=True)

with d_col4:
    st.markdown(f"""
    <div class="metric-card" style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 8px; border: 1px solid #059669;">
        <div class="metric-title" style="text-align: center; width: 100%; color: #34D399;">Post-Solar Net EB Bill </div>
    """, unsafe_allow_html=True)
    
    gauge_val_post = min(100.0, (pred_bill_post_solar / max(1.0, pred_bill_baseline)) * 100.0)
    fig_ring_post = go.Figure(go.Pie(
        values=[gauge_val_post, 100 - gauge_val_post],
        hole=0.74,
        sort=False,
        direction='counterclockwise',
        showlegend=False,
        hoverinfo='none',
        textinfo='none',
        marker=dict(colors=['#10b981', '#3f3f46'], line=dict(color='#059669', width=2))
    ))
    fig_ring_post.update_layout(
        height=145, margin=dict(l=0, r=0, t=5, b=5),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(
            text=f"<b style='font-size:1.2rem; color:#34d399;'>₹ {pred_bill_post_solar:,}</b><br><span style='font-size:0.72rem; color:#cbd5e1;'>Net Payable</span>",
            x=0.5, y=0.5, showarrow=False
        )]
    )
    st.plotly_chart(fig_ring_post, width="stretch", config={'displayModeBar': False})
    st.markdown(f'<div class="metric-sub-green" style="margin-top:-10px; text-align:center;">Saved ₹ {net_savings_cycle:,} ({savings_pct}% Off)</div></div>', unsafe_allow_html=True)

with st.expander("📋 View Ingested Bill Breakdown (Consumer No, Billing Dates & Day Usage)"):
    breakdown_data = []
    for idx, b in enumerate(parsed_bills_list, 1):
        breakdown_data.append({
            "Bill Entry": f"Cycle {idx} ({b['filename']})",
            "Consumer No": b["consumer_no"],
            "Billing Period": f"{b['date_from']} to {b['date_to']}",
            "Billing Days": b["billing_days"],
            "Total kWh Consumed": f"{b['total_units']} kWh",
            "Daily Usage Rate": f"{b['daily_demand']:.2f} kWh / Day",
            "Recorded Bill (₹)": f"₹ {b['bill_amount']:,.2f}"
        })
    st.dataframe(pd.DataFrame(breakdown_data), width="stretch")

# =====================================================================
# INVERTER HEALTH STATUS: 5-MIN INVERTER NATIVE INSPECTOR
# =====================================================================
arrow_icon = "▴" if st.session_state["show_inverter_diag"] else "▾"
inverter_button_label = f"🩺 INVERTER HEALTH STATUS  {arrow_icon}"

if st.button(inverter_button_label, width="stretch"):
    st.session_state["show_inverter_diag"] = not st.session_state["show_inverter_diag"]
    st.rerun()

if st.session_state["show_inverter_diag"]:
    if selected_weather == "None":
        st.markdown("""
        <div class="terminal-box" style="border-color: #F59E0B; color: #F59E0B;">
            <div>┌────────────────────────────────────────────────────────────────────────────────────────┐</div>
            <div>│  ⚠️ Telemetry Inactive: Please select a historical weather data                        │</div>
            <div>└────────────────────────────────────────────────────────────────────────────────────────┘</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_sl1, col_sl2 = st.columns([3, 1])
        with col_sl1:
            time_options_5m = df_5min["Timestamp"].tolist()
            inspect_time_5m = st.select_slider(
                "🕒 Inverter Hardware Telemetry Inspector (5-Min Native)",
                options=time_options_5m,
                value=time_options_5m[len(time_options_5m)//2]
            )
        with col_sl2:
            weather_label = selected_weather.split(' ')[0]
            st.info(f"Replaying: **{weather_label}** (5-Min Native)")

        row_now_5m = df_5min[df_5min["Timestamp"] == inspect_time_5m].iloc[0]

        live_irr = row_now_5m["Irradiance_W_m2"]
        live_dc_amps = row_now_5m["DC_Current_Amps"]
        live_mod_temp = row_now_5m["Module_Temp_C"]
        live_core_temp = row_now_5m["Inverter_Temp_C"]
        live_eff = row_now_5m["Inverter_Eff_%"]
        live_loss_w = row_now_5m["Inverter_Loss_W"]
        live_pv_kw = row_now_5m["Solar_Gen_kW"]
        live_voltage = row_now_5m["Grid_Voltage_V"]
        live_grid_st = row_now_5m["Grid_Status"]

        if live_core_temp >= 65.0:
            health_msg = f"⚠️ INVERTER HEALTH: THERMAL OVERHEAT TRIP @ {inspect_time_5m} ({live_core_temp}°C)"
            color_code = "#EF4444"
        elif live_grid_st in ["Blackout", "Brownout"]:
            health_msg = f"⚠️ INVERTER HEALTH: ISLANDED GRID TRIP ({live_grid_st.upper()} @ {live_voltage}V)"
            color_code = "#F59E0B"
        elif live_irr <= 5:
            health_msg = f"🌙 INVERTER HEALTH: NIGHT STANDBY MODE @ {inspect_time_5m}"
            color_code = "#60A5FA"
        else:
            health_msg = f"✅ INVERTER HEALTH: OPTIMAL RUNNING CONDITION @ {inspect_time_5m}"
            color_code = "#10B981"

        st.markdown(f"""
        <div class="terminal-box" style="border-color: {color_code}; color: {color_code};">
            <div>┌────────────────────────────────────────────────────────────────────────────────────────┐</div>
            <div>│  {health_msg}</div>
            <div>├────────────────────────────────────────────────────────────────────────────────────────┤</div>
            <div>│  • Telemetry Time        : {inspect_time_5m} Hrs  [Profile: {selected_weather}]</div>
            <div>│  • Solar Irradiance (IRR): {live_irr} W/m²  │ DC Current: {live_dc_amps} A  │ Module Temp: {live_mod_temp}°C</div>
            <div>│  • PV AC Generated       : {live_pv_kw:.3f} kW ({int(live_pv_kw*1000)} Watts)</div>
            <div>│  • Inverter Core Temp    : {live_core_temp}°C  ({"Thermal Stress Overload" if live_core_temp>=65 else "Healthy / Well-Cooled"})</div>
            <div>│  • Conversion Efficiency : {live_eff:.2f}%  (Dynamic ML Joule Dissipation: ~{live_loss_w:.1f} Watts)</div>
            <div>│  • Grid Synchronization  : {live_voltage:.1f} V @ 50.02 Hz ({live_grid_st})</div>
            <div>└────────────────────────────────────────────────────────────────────────────────────────┘</div>
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# SOLAR SCADA TELEMETRY INSPECTOR: 15-MIN STREAM
# =====================================================================
col_scada1, col_scada2 = st.columns([3, 1])
with col_scada1:
    time_options_15m = df_15min["Timestamp"].tolist()
    inspect_time_15m = st.select_slider(
        "🕒 Solar SCADA Telemetry Inspector (15-Min SCADA Stream)",
        options=time_options_15m,
        value=time_options_15m[len(time_options_15m)//2]
    )
with col_scada2:
    st.info(f"⚡ Slot: **{inspect_time_15m}** (15-Min)")

row_scada = df_15min[df_15min["Timestamp"] == inspect_time_15m].iloc[0]

scada_dc_kw = row_scada["DC_Gen_kW"]
scada_eff_pct = row_scada["Inverter_Eff_%"]
scada_ac_kw = row_scada["AC_Dispatched_kW"]
scada_loss_w = row_scada["Conversion_Wasted_W"]

f1, f2, f3, f4 = st.columns(4)
with f1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">DC Generation</div><div class="metric-val">{scada_dc_kw:.3f} kW</div><div class="metric-sub-green">↑ Raw Solar PV @ {inspect_time_15m}</div></div>', unsafe_allow_html=True)
with f2:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Inverter Efficiency</div><div class="metric-val">{scada_eff_pct:.2f}%</div><div class="metric-sub-green">↑ Pure Sine Wave @ {inspect_time_15m}</div></div>', unsafe_allow_html=True)
with f3:
    st.markdown(f'<div class="metric-card"><div class="metric-title">AC Dispatched</div><div class="metric-val">{scada_ac_kw:.3f} kW</div><div class="metric-sub-green">↑ Delivered Load @ {inspect_time_15m}</div></div>', unsafe_allow_html=True)
with f4:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Conversion Wasted</div><div class="metric-val">{scada_loss_w:.1f} W</div><div class="metric-sub-red">↑ Heat Dissipation @ {inspect_time_15m}</div></div>', unsafe_allow_html=True)

# =====================================================================
# 4. BATTERY STORAGE PERFORMANCE & CAPACITY TRACKER
# =====================================================================
st.markdown("### 🔋 Battery Storage Performance & Capacity Tracker")

if battery_bms_fault:
    st.markdown("""
    <div class="terminal-box" style="border-color: #EF4444; color: #EF4444; margin-bottom: 12px; background: rgba(239, 68, 68, 0.08);">
        <div>🚨 <b>CRITICAL HARDWARE ALERT: BMS TRIP DETECTED - BATTERY STORAGE ISOLATED</b></div>
        <div style="font-size: 0.85rem; color: #FCA5A5; margin-top: 4px;">
            Battery charging is locked to 0.00 kW. Internal contactors opened. Automatic STS Grid Bypass engaged to supply load.
        </div>
    </div>
    """, unsafe_allow_html=True)

time_options_bat = df_5min["Timestamp"].tolist()
default_idx = min(len(time_options_bat) - 1, 144)
inspect_bat_time = st.select_slider(
    "🕒 5-Min Battery Charging & Energy Transition Timeline",
    options=time_options_bat,
    value=time_options_bat[default_idx]
)

row_bat = df_5min[df_5min["Timestamp"] == inspect_bat_time].iloc[0]

live_soc_pct = float(row_bat["Battery_SoC_Pct"]) if not battery_bms_fault else 0.0
live_stored_kwh = float(row_bat["Battery_Stored_kWh"]) if not battery_bms_fault else 0.0
live_battery_flow_kw = float(row_bat["Battery_Flow_kW"]) if not battery_bms_fault else 0.0
live_grid_imp_kw = float(row_bat["Grid_Import_kW"])
live_status = str(row_bat["Dispatch_Status"])
st.progress(int(np.clip(live_soc_pct, 0, 100)))

b1, b2, b3, b4 = st.columns(4)
with b1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Current Battery Charge @ {inspect_bat_time}</div>
        <div class="metric-val">{live_soc_pct:.1f}%</div>
        <div class="metric-sub-green">↑ {live_stored_kwh:.2f} / {selected_battery_kwh} kWh</div>
    </div>
    """, unsafe_allow_html=True)

with b2:
    inflow_label = f"↑ {live_battery_flow_kw:.2f} kW Inflow" if live_battery_flow_kw > 0 else (f"↓ {abs(live_battery_flow_kw):.2f} kW Discharge" if live_battery_flow_kw < 0 else "Idle / Standby")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Battery Active Flow Rate</div>
        <div class="metric-val">{abs(live_battery_flow_kw):.2f} <span style="font-size:1rem;">kW</span></div>
        <div class="metric-sub-green">{inflow_label}</div>
    </div>
    """, unsafe_allow_html=True)

with b3:
    if battery_bms_fault:
        bypass_status = "ACTIVE (FAILOVER)"
        bypass_color = "#EF4444"
        bypass_sub = "⚡ BMS Tripped: Load Switched to EB Grid"
    else:
        bypass_status = "INACTIVE (NORMAL ONLINE)"
        bypass_color = "#10B981"
        bypass_sub = "✓ Battery / Solar Inverter Online"

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Grid Bypass Status</div>
        <div class="metric-val" style="font-size: 1.15rem; color: {bypass_color};">{bypass_status}</div>
        <div style="font-size: 0.72rem; color: {bypass_color}; margin-top: 3px;">{bypass_sub}</div>
    </div>
    """, unsafe_allow_html=True)

with b4:
    if battery_bms_fault or live_grid_imp_kw > 0.05:
        night_grid_val = f"{summary['total_import_kwh']} kWh"
        night_grid_sub = f"⚡ Drawing {live_grid_imp_kw:.2f} kW from EB Grid"
        night_color = "#F59E0B" if not battery_bms_fault else "#EF4444"
    else:
        night_grid_val = "0.00 kWh"
        night_grid_sub = "✓ 100% Clean Battery/Solar Powered"
        night_color = "#10B981"

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Night EB Grid Consumed</div>
        <div class="metric-val" style="color: {night_color};">{night_grid_val}</div>
        <div style="font-size: 0.72rem; color: {night_color}; margin-top: 3px;">{night_grid_sub}</div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# 5. EB GRID EXPORT & SURPLUS REVENUE DASHBOARD
# =====================================================================
st.markdown("### ⚡ EB Grid Export & Surplus Feed-in Telemetry")

daily_export_kwh = summary["total_export_kwh"]
cycle_export_kwh = round(daily_export_kwh * float(active_cycle_days), 2)
feed_in_tariff_rate = 3.10  # Standard TANGEDCO Solar Feed-in Tariff per Unit (INR)
cycle_export_revenue = int(round(cycle_export_kwh * feed_in_tariff_rate))
annual_export_revenue = int(round(daily_export_kwh * 365.0 * feed_in_tariff_rate))

instant_export_kw = float(row_bat["Grid_Export_kW"])
peak_export_kw = float(df_5min["Grid_Export_kW"].max())

ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)
with ex_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Daily Solar Fed to Grid</div>
        <div class="metric-val" style="color: #38BDF8;">{daily_export_kwh} <span style="font-size:1rem;">kWh</span></div>
        <div class="metric-sub-blue">↑ Total Daily Surplus Export</div>
    </div>
    """, unsafe_allow_html=True)

with ex_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{active_cycle_days}-Day Export Volume</div>
        <div class="metric-val" style="color: #38BDF8;">{cycle_export_kwh} <span style="font-size:1rem;">kWh</span></div>
        <div class="metric-sub-green">↑ Net Metering Units ({active_cycle_days} Days)</div>
    </div>
    """, unsafe_allow_html=True)

with ex_col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Live Export Rate @ {inspect_bat_time}</div>
        <div class="metric-val" style="color: #34D399;">{instant_export_kw:.3f} <span style="font-size:1rem;">kW</span></div>
        <div class="metric-sub-green">↑ Peak Export: {peak_export_kw:.2f} kW</div>
    </div>
    """, unsafe_allow_html=True)

with ex_col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Surplus Feed-in Credits</div>
        <div class="metric-val" style="color: #FBBF24;">₹ {cycle_export_revenue:,}</div>
        <div class="metric-sub-blue">↑ ~₹ {annual_export_revenue:,}/Year Feed-in Credit</div>
    </div>
    """, unsafe_allow_html=True)

fig_export = go.Figure()
fig_export.add_trace(go.Scatter(
    x=df_dispatch["Timestamp"],
    y=df_dispatch["Grid_Export_kW"],
    mode="lines",
    name="Surplus Solar Export (kW)",
    line=dict(color="#38BDF8", width=2.5),
    fill="tozeroy",
    fillcolor="rgba(56, 189, 248, 0.18)",
    hovertemplate="<b>%{x}</b><br>Solar Export Rate: <b>%{y:.3f} kW</b><extra></extra>"
))
fig_export.update_layout(
    height=240,
    margin=dict(l=15, r=15, t=25, b=15),
    template="plotly_dark",
    xaxis=dict(title="Timeline (Hours)", gridcolor="#1f2937"),
    yaxis=dict(title="Exported Power (kW)", gridcolor="#1f2937", range=[0, max(0.5, peak_export_kw * 1.15)]),
    showlegend=False
)
st.plotly_chart(fig_export, width="stretch", key="eb_export_telemetry_chart")

# =====================================================================
# 6. BACKUP APPLIANCE LOADS
# =====================================================================
st.markdown("### Configure Essential Backup Loads")
col_c1, col_c2, col_c3, col_c4 = st.columns(4)
with col_c1:
    st.session_state["fan_count"] = st.number_input("🌀 Ceiling Fans Count (75W ea)", min_value=0, max_value=10, value=st.session_state["fan_count"], step=1)
with col_c2:
    st.session_state["light_count"] = st.number_input("💡 LED Lights Count (14W ea)", min_value=0, max_value=20, value=st.session_state["light_count"], step=1)
with col_c3:
    st.session_state["fridge_count"] = st.number_input("❄️ Inverter Refrigerator (180W ea)", min_value=0, max_value=3, value=st.session_state["fridge_count"], step=1)
with col_c4:
    st.session_state["router_count"] = st.number_input("📶 Wi-Fi Router (15W ea)", min_value=0, max_value=3, value=st.session_state["router_count"], step=1)

avail_emergency_kwh = live_stored_kwh if not battery_bms_fault else 0.0
combined_runtime = avail_emergency_kwh / max(0.05, total_appliance_load_kw)

st.markdown(f"#### ⏱️ Combined Backup Duration: **{combined_runtime:.1f} Hours** (with all selected loads active: ~{int(total_appliance_load_kw*1000)} Watts @ {inspect_bat_time})")
app_c1, app_c2, app_c3, app_c4 = st.columns(4)
with app_c1:
    fan_load = max(0.001, (st.session_state["fan_count"] * 75) / 1000.0)
    fan_runtime = avail_emergency_kwh / fan_load
    st.markdown(f"""
    <div class="metric-card" style="text-align: center;">
        <div class="metric-title">🌀 {st.session_state['fan_count']} Fan(s)</div>
        <div style="color: #9CA3AF; font-size: 0.8rem;">Active Load: {int(fan_load*1000)} W</div>
        <div class="metric-val" style="color: #60A5FA; margin-top: 6px;">{fan_runtime:.1f} hrs</div>
        <div class="metric-sub-green">Dedicated Runtime</div>
    </div>
    """, unsafe_allow_html=True)

with app_c2:
    light_load = max(0.001, (st.session_state["light_count"] * 14) / 1000.0)
    light_runtime = avail_emergency_kwh / light_load
    st.markdown(f"""
    <div class="metric-card" style="text-align: center;">
        <div class="metric-title">💡 {st.session_state['light_count']} Light(s)</div>
        <div style="color: #9CA3AF; font-size: 0.8rem;">Active Load: {int(light_load*1000)} W</div>
        <div class="metric-val" style="color: #FBBF24; margin-top: 6px;">{light_runtime:.1f} hrs</div>
        <div class="metric-sub-green">Dedicated Runtime</div>
    </div>
    """, unsafe_allow_html=True)

with app_c3:
    fridge_load = max(0.001, (st.session_state["fridge_count"] * 180) / 1000.0)
    fridge_runtime = avail_emergency_kwh / fridge_load
    st.markdown(f"""
    <div class="metric-card" style="text-align: center;">
        <div class="metric-title">❄️ {st.session_state['fridge_count']} Refrigerator</div>
        <div style="color: #9CA3AF; font-size: 0.8rem;">Active Load: {int(fridge_load*1000)} W</div>
        <div class="metric-val" style="color: #34D399; margin-top: 6px;">{fridge_runtime:.1f} hrs</div>
        <div class="metric-sub-green">Dedicated Runtime</div>
    </div>
    """, unsafe_allow_html=True)

with app_c4:
    router_load = max(0.001, (st.session_state["router_count"] * 15) / 1000.0)
    router_runtime = avail_emergency_kwh / router_load
    st.markdown(f"""
    <div class="metric-card" style="text-align: center;">
        <div class="metric-title">📶 {st.session_state['router_count']} Wi-Fi Router</div>
        <div style="color: #9CA3AF; font-size: 0.8rem;">Active Load: {int(router_load*1000)} W</div>
        <div class="metric-val" style="color: #A78BFA; margin-top: 6px;">{router_runtime:.1f} hrs</div>
        <div class="metric-sub-green">Dedicated Runtime</div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# 7. CAPITAL INVESTMENT AND PAYBACK FORECASTER (WATERFALL FORECAST)
# =====================================================================
st.markdown("### 💰 Capital Investment and Payback Forecaster")

cap1, cap2, cap3, cap4 = st.columns(4)
with cap1: st.markdown(f'<div class="metric-card"><div class="metric-title">Gross Capital Cost</div><div class="metric-val">₹ {summary["gross_capex"]:,}</div><div class="metric-sub-green">↑ ₹60,000/kW x {recommended_pv_kw} kW</div></div>', unsafe_allow_html=True)
with cap2: st.markdown(f'<div class="metric-card"><div class="metric-title">PM Surya Ghar Subsidy</div><div class="metric-val" style="color:#34D399;">- ₹ {summary["pm_subsidy"]:,}</div><div class="metric-sub-green">↑ Direct Central DBT Subsidy</div></div>', unsafe_allow_html=True)
with cap3: st.markdown(f'<div class="metric-card"><div class="metric-title">Net User Investment</div><div class="metric-val">₹ {summary["net_investment"]:,}</div><div class="metric-sub-green">↑ Out-of-Pocket Capex</div></div>', unsafe_allow_html=True)
with cap4: st.markdown(f'<div class="metric-card"><div class="metric-title">Estimated Payback Period</div><div class="metric-val">{summary["payback_years"]} Years</div><div class="metric-sub-green">↑ ₹{summary["annual_savings_rs"]:,}/Year Saved</div></div>', unsafe_allow_html=True)

# --- WATERFALL PAYBACK FORECAST CHART ---
net_capex = float(summary["net_investment"])
annual_savings = float(summary["annual_savings_rs"])
payback_yrs = float(summary["payback_years"])

max_proj_years = max(5, int(np.ceil(payback_yrs)) + 2)
wf_x = ["(Initial Capex)"] + [f"Year {yr}" for yr in range(1, max_proj_years + 1)]
wf_measure = ["absolute"] + ["relative"] * max_proj_years
wf_y = [-net_capex] + [annual_savings] * max_proj_years

fig_waterfall = go.Figure(go.Waterfall(
    name="Payback Flow",
    orientation="v",
    measure=wf_measure,
    x=wf_x,
    textposition="outside",
    text=[f"-₹{int(net_capex):,}"] + [f"+₹{int(annual_savings):,}" for _ in range(max_proj_years)],
    y=wf_y,
    connector={"line": {"color": "rgb(63, 63, 70)"}},
    decreasing={"marker": {"color": "#EF4444"}},
    increasing={"marker": {"color": "#10B981"}},
    totals={"marker": {"color": "#38BDF8"}}
))

fig_waterfall.add_hline(
    y=0, line_dash="dash", line_color="#FBBF24", line_width=1.8,
    annotation_text=f"🎯 Break-Even Point (~{payback_yrs} Years)",
    annotation_position="top left",
    annotation_font=dict(color="#FBBF24", size=12)
)

fig_waterfall.update_layout(
    title=f"<b>Cumulative Capital Recovery & Payback Forecast (Break-Even in ~{payback_yrs} Years)</b>",
    title_font=dict(size=14, color="#94A3B8"),
    height=360,
    margin=dict(l=20, r=20, t=40, b=20),
    template="plotly_dark",
    yaxis=dict(title="Net Balance (₹)", gridcolor="#1f2937"),
    xaxis=dict(gridcolor="#1f2937"),
    showlegend=False
)
st.plotly_chart(fig_waterfall, width="stretch", key="capex_payback_waterfall")

# =====================================================================
# 8. 24-HOUR ENERGY DISPATCH
# =====================================================================
st.markdown("### 📈 24-Hour Energy Dispatch & Storage Flow")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_dispatch["Timestamp"], y=df_dispatch["House_Load_kW"],
    name="House Demand (kWh)", line=dict(color="#EF4444", width=2)
))
fig.add_trace(go.Scatter(
    x=df_dispatch["Timestamp"], y=df_dispatch["Solar_Gen_kW"],
    name="Solar AC Output (kWh)", line=dict(color="#10B981", width=2.5),
    fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.2)'
))
fig.add_trace(go.Scatter(
    x=df_dispatch["Timestamp"], y=df_dispatch["Battery_SoC_Pct"],
    name="Battery SoC (%)", line=dict(color="#38BDF8", width=2, dash='dot'),
    yaxis="y2"
))

fig.add_hline(
    y=reserve_floor_pct, line_dash="dash", line_color="#F59E0B",
    annotation_text=f"Emergency Cutoff Reserve ({reserve_floor_pct}%)",
    annotation_position="bottom right", yref="y2"
)

fig.update_layout(
    height=450,
    margin=dict(l=20, r=20, t=30, b=20),
    template="plotly_dark",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis=dict(title="Energy (kWh)"),
    yaxis2=dict(title="Battery SoC (%)", overlaying="y", side="right", range=[0, 105])
)
st.plotly_chart(fig, width="stretch")

# =====================================================================
# 9. ADVANCED ANALYTICS (CORRELATION, SCATTER & ENERGY BALANCE)
# =====================================================================
st.markdown("### 🔬 System Telemetry & Energy Dispatch Analytics")

tab_heat, tab_scatter, tab_bar = st.tabs([
    "🔥 Correlation Heatmap", 
    "📈 Irradiance vs DC Scatterplot", 
    "📊 Hourly Energy Balance Bar"
])

# 1. CORRELATION HEATMAP (NaN FREE CORRELATION MATRIX)
with tab_heat:
    st.caption("Solar Irradiance, Module Temp, Inverter Efficiency, DC Power, House Load and Grid Import/Export correlation matrix.")
    corr_cols = [
        "Irradiance_W_m2", "Module_Temp_C", "Inverter_Eff_%", 
        "DC_Gen_kW", "House_Load_kW", "Grid_Import_kW", "Grid_Export_kW"
    ]
    valid_cols = [c for c in corr_cols if c in df_dispatch.columns]
    
    # Filter daylight hours where solar generation occurs to compute realistic efficiency correlation
    daylight_df = df_dispatch[df_dispatch["DC_Gen_kW"] > 0.01][valid_cols]
    if len(daylight_df) > 5:
        corr_df = daylight_df.corr().fillna(0.0)
    else:
        corr_df = df_dispatch[valid_cols].corr().fillna(0.0)

    # Ensure diagonal values are strictly 1.0
    for col in corr_df.columns:
        corr_df.loc[col, col] = 1.0

    fig_heat = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale="Viridis",
        zmin=-1, zmax=1,
        text=np.round(corr_df.values, 2),
        texttemplate="%{text}",
        hoverongaps=False
    ))
    fig_heat.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        template="plotly_dark",
        xaxis=dict(tickangle=-25)
    )
    st.plotly_chart(fig_heat, width="stretch", key="analytics_corr_heatmap")

# 2. SCATTERPLOT WITH TRENDLINE (IRRADIANCE VS DC POWER)
with tab_scatter:
    st.caption("Solar Radiation vs Raw DC Generation (Module Temperature impact linearity check).")
    
    x_irr = df_dispatch["Irradiance_W_m2"].values
    y_dc = df_dispatch["DC_Gen_kW"].values
    if len(x_irr) > 1 and np.max(x_irr) > 0:
        slope, intercept = np.polyfit(x_irr, y_dc, 1)
        trend_x = np.linspace(np.min(x_irr), np.max(x_irr), 50)
        trend_y = slope * trend_x + intercept
    else:
        trend_x, trend_y = [], []

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=df_dispatch["Irradiance_W_m2"],
        y=df_dispatch["DC_Gen_kW"],
        mode="markers",
        name="Telemetry Points",
        marker=dict(
            size=7,
            color=df_dispatch["Module_Temp_C"],
            colorscale="Inferno",
            showscale=True,
            colorbar=dict(title="Module Temp (°C)")
        ),
        text=[f"Time: {t}<br>Eff: {eff}%" for t, eff in zip(df_dispatch["Timestamp"], df_dispatch["Inverter_Eff_%"])],
        hoverinfo="text+x+y"
    ))

    if len(trend_x) > 0:
        fig_scatter.add_trace(go.Scatter(
            x=trend_x,
            y=trend_y,
            mode="lines",
            name="Linear Trendline",
            line=dict(color="#38BDF8", width=2.5, dash="dash")
        ))

    fig_scatter.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        template="plotly_dark",
        xaxis=dict(title="Solar Irradiance (W/m²)", gridcolor="#1f2937"),
        yaxis=dict(title="Raw DC Power Output (kW)", gridcolor="#1f2937"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_scatter, width="stretch", key="analytics_power_scatter")

# 3. HOURLY ENERGY BALANCE BAR CHART
with tab_bar:
    st.caption("Hourly Generation, Consumption, Battery Charging and Grid Export distribution.")
    df_hourly = df_dispatch.copy()
    df_hourly["Hour_Block"] = df_hourly["Timestamp"].apply(lambda x: str(x).split(":")[0] + ":00")
    
    df_hourly["Battery_Charge_kW"] = df_hourly["Battery_Flow_kW"].apply(lambda val: max(0.0, val))
    
    grouped = df_hourly.groupby("Hour_Block")[
        ["Solar_Gen_kW", "House_Load_kW", "Battery_Charge_kW", "Grid_Export_kW"]
    ].mean().reset_index()

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=grouped["Hour_Block"], y=grouped["Solar_Gen_kW"], name="Solar Generation", marker_color="#10B981"))
    fig_bar.add_trace(go.Bar(x=grouped["Hour_Block"], y=grouped["House_Load_kW"], name="House Demand", marker_color="#EF4444"))
    fig_bar.add_trace(go.Bar(x=grouped["Hour_Block"], y=grouped["Battery_Charge_kW"], name="Battery Storage Inflow", marker_color="#A78BFA"))
    fig_bar.add_trace(go.Bar(x=grouped["Hour_Block"], y=grouped["Grid_Export_kW"], name="Surplus Grid Export", marker_color="#38BDF8"))

    fig_bar.update_layout(
        barmode="group",
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        template="plotly_dark",
        hovermode="x unified",
        xaxis=dict(title="24-Hour Timeline (Hour Blocks)", gridcolor="#1f2937"),
        yaxis=dict(title="Average Power Flow (kW)", gridcolor="#1f2937"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_bar, width="stretch", key="analytics_hourly_bar")

# =====================================================================
# 10. 24-HOUR TELEMETRY LEDGER TABLE
# =====================================================================
with st.expander("🔍 Inspect Full 24-Hour Telemetry & Loss Ledger"):
    st.caption(f"Showing **{len(df_dispatch)} records** at **{timeline_resolution}** native resolution.")
    if toggle_inverter_loss_cols:
        display_cols = [
            "Timestamp", "DC_Gen_kW", "Inverter_Eff_%", "AC_Dispatched_kW",
            "Conversion_Wasted_W", "House_Load_kW", "Battery_Flow_kW",
            "Battery_Stored_kWh", "Battery_SoC_Pct", "Grid_Import_kW", "Grid_Export_kW"
        ]
    else:
        display_cols = [
            "Timestamp", "AC_Dispatched_kW", "House_Load_kW", "Battery_Flow_kW",
            "Battery_Stored_kWh", "Battery_SoC_Pct", "Grid_Import_kW", "Grid_Export_kW"
        ]

    st.dataframe(
        df_dispatch[display_cols].style.format({
            "DC_Gen_kW": "{:.3f}", "Inverter_Eff_%": "{:.2f}%", "AC_Dispatched_kW": "{:.3f}",
            "Conversion_Wasted_W": "{:.1f} W", "House_Load_kW": "{:.3f}", "Battery_Flow_kW": "{:.3f}",
            "Battery_Stored_kWh": "{:.2f}", "Battery_SoC_Pct": "{:.1f}%", "Grid_Import_kW": "{:.3f}",
            "Grid_Export_kW": "{:.3f}"
        }),
        width="stretch",
        height=450
    )
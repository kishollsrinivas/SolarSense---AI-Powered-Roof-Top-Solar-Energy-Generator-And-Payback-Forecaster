import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import warnings

# Suppress feature name validation warnings from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

@st.cache_resource
def load_ml_models():
    models_dir = "models"
    model_files = {
        "solar_granular": "solar_generation_model.pkl",
        "solar_macro": "solar_macro_model.pkl",
        "load_model": "house_load_model.pkl",
        "inverter_model": "inverter_temp_model.pkl",
        "inverter_eff": "inverter_efficiency_model.pkl",
        "battery_model": "battery_health_model.pkl",
        "grid_classifier": "grid_stability_classifier.pkl"
    }
    loaded = {}
    for key, filename in model_files.items():
        path = os.path.join(models_dir, filename)
        loaded[key] = joblib.load(path) if os.path.exists(path) else None
    return loaded

MODELS = load_ml_models()

def get_available_dataset_dates():
    return [
        "None",
        "2020-05-15 (Peak Clear Summer)",
        "2020-06-18 (Heavy Monsoon / Rain Cloud)",
        "2020-07-04 (Intermittent Passing Clouds)"
    ]

def fetch_weather_profile(profile_type="None"):
    if profile_type == "2020-06-18 (Heavy Monsoon / Rain Cloud)":
        base_irr = [0, 0, 0, 0, 0, 10, 45, 95, 140, 180, 210, 195, 160, 130, 110, 70, 30, 5, 0, 0, 0, 0, 0, 0]
        amb_t = [22, 22, 21, 21, 21, 22, 23, 24, 25, 26, 26, 26, 25, 25, 24, 24, 23, 23, 22, 22, 22, 22, 22, 22]
        mod_t = [t + (irr * 0.012) for t, irr in zip(amb_t, base_irr)]
        grid_v = [231, 230, 230, 229, 228, 227, 224, 220, 215, 204, 201, 198, 205, 212, 218, 222, 225, 227, 228, 229, 230, 230, 231, 231]
    elif profile_type == "2020-07-04 (Intermittent Passing Clouds)":
        base_irr = [0, 0, 0, 0, 0, 20, 120, 380, 620, 290, 880, 310, 890, 720, 410, 330, 120, 10, 0, 0, 0, 0, 0, 0]
        amb_t = [24, 23, 23, 22, 22, 23, 26, 28, 30, 29, 33, 31, 34, 33, 32, 30, 28, 27, 26, 25, 25, 24, 24, 24]
        mod_t = [t + (irr * 0.026) for t, irr in zip(amb_t, base_irr)]
        grid_v = [232, 231, 231, 230, 230, 229, 227, 225, 223, 226, 221, 228, 220, 222, 225, 227, 228, 229, 230, 230, 231, 231, 232, 232]
    elif profile_type == "2020-05-15 (Peak Clear Summer)":
        base_irr = [0, 0, 0, 0, 0, 25, 160, 440, 710, 880, 960, 990, 970, 880, 690, 430, 170, 18, 0, 0, 0, 0, 0, 0]
        amb_t = [25, 24, 23, 23, 23, 24, 26, 29, 32, 34, 36, 37, 37, 36, 35, 33, 31, 29, 28, 27, 26, 26, 25, 25]
        mod_t = [t + (irr * 0.032) for t, irr in zip(amb_t, base_irr)]
        grid_v = [231, 231, 230, 230, 230, 229, 231, 234, 238, 244, 249, 255, 256, 252, 246, 241, 236, 233, 231, 231, 231, 231, 231, 231]
    else:
        base_irr = [0, 0, 0, 0, 0, 20, 150, 420, 680, 850, 920, 950, 930, 840, 650, 410, 160, 15, 0, 0, 0, 0, 0, 0]
        amb_t = [24, 23, 23, 22, 22, 23, 25, 28, 31, 33, 35, 36, 36, 35, 34, 32, 30, 28, 27, 26, 25, 25, 24, 24]
        mod_t = [t + (irr * 0.030) for t, irr in zip(amb_t, base_irr)]
        grid_v = [230.8] * 24
    return base_irr, mod_t, amb_t, grid_v

def classify_grid_stability(voltage=230.0, frequency=50.0, current=5.0):
    if voltage <= 5.0:
        return "Blackout", 0.0
    if MODELS["grid_classifier"] is not None:
        try:
            feats = pd.DataFrame([[voltage, frequency, current]],
                                 columns=["Grid Voltage (V)", "Grid Frequency (Hz)", "Grid Current (A)"])
            return str(MODELS["grid_classifier"].predict(feats)[0]), voltage
        except Exception:
            pass
    if voltage < 207.0:
        return "Brownout", voltage
    elif voltage > 253.0:
        return "Surge", voltage
    return "Normal Sync", voltage

def predict_battery_state(cycle_count=380, ir_ohms=0.0165, avg_temp=30.0, max_temp=38.0, charge_time=1.8):
    soh = None
    if MODELS["battery_model"] is not None:
        try:
            feats = pd.DataFrame([[cycle_count, ir_ohms, avg_temp, max_temp, charge_time]],
                                 columns=["cycle", "IR", "Tavg", "Tmax", "chargetime"])
            pred = float(MODELS["battery_model"].predict(feats)[0])
            if 40.0 <= pred <= 100.0:
                soh = pred
        except Exception:
            pass
            
    if soh is None:
        cycle_fade = (cycle_count - 380) * 0.0072
        ir_fade = max(0.0, (ir_ohms - 0.0165) * 450.0)
        soh = 89.2 - cycle_fade - ir_fade

    soh_final = round(float(np.clip(soh, 45.0, 99.5)), 1)
    return soh_final, ir_ohms

@st.cache_data(show_spinner=False)
def simulate_24h_dispatch(plant_capacity_kw=3.0,
                          nominal_battery_kwh=10.0,
                          emergency_reserve_floor_pct=30.0,
                          weather_profile="None",
                          daily_demand_kwh=12.0,
                          battery_fault=False,
                          selected_emergency_load_kw=0.45,
                          resolution="15-Min (Plant SCADA)"):
    
    steps_per_day = 288
    step_hours = 5.0 / 60.0

    soh_pct, ir_ohms = predict_battery_state(cycle_count=380)
    effective_battery_kwh = nominal_battery_kwh * (soh_pct / 100.0)
    emergency_reserve_kwh = effective_battery_kwh * (emergency_reserve_floor_pct / 100.0)
    
    current_stored_kwh = 0.0 if battery_fault else emergency_reserve_kwh
    hourly_irr, hourly_mod_temp, hourly_amb_temp, hourly_grid_v = fetch_weather_profile(weather_profile)
    v_battery_nom = 48.0
    
    xp = np.arange(24)
    x_new = np.linspace(0, 23.999, steps_per_day)

    interp_irr = np.interp(x_new, xp, hourly_irr)
    interp_mod_t = np.interp(x_new, xp, hourly_mod_temp)
    interp_amb_t = np.interp(x_new, xp, hourly_amb_temp)
    interp_grid_v = np.interp(x_new, xp, hourly_grid_v)

    raw_base_curve = np.array([
        0.35, 0.30, 0.28, 0.28, 0.35, 0.65, 
        1.20, 1.60, 1.25, 0.95, 0.85, 0.90, 1.05, 0.95, 0.88, 1.10, 1.55, 2.15, 
        2.55, 2.30, 1.80, 1.25, 0.75, 0.45
    ])
    interp_hourly_weights = np.interp(x_new, xp, raw_base_curve)
    norm_factor = daily_demand_kwh / max(0.001, (interp_hourly_weights.sum() * step_hours))
    step_demand_kw = interp_hourly_weights * norm_factor

    irr_arr = interp_irr
    pv_gen_arr = np.where(irr_arr <= 5.0, 0.0, (irr_arr / 1000.0) * plant_capacity_kw * 0.85)
    dc_current_arr = np.where(irr_arr <= 5.0, 0.0, (irr_arr / 1000.0) * (plant_capacity_kw * 1000.0 / 360.0))
    dc_gen_arr = pv_gen_arr * 1.025

    core_temp_arr = np.where(
        pv_gen_arr * 1000.0 <= 5.0,
        interp_amb_t + 2.0,
        interp_amb_t + 5.0 + ((pv_gen_arr * 1000.0) / max(1.0, plant_capacity_kw * 1000.0)) * 28.0
    )
    if weather_profile == "2020-05-15 (Peak Clear Summer)":
        peak_mask = (x_new >= 11.5) & (x_new <= 14.5)
        core_temp_arr[peak_mask] = np.maximum(core_temp_arr[peak_mask], 66.2)

    # Dynamic Thermal Efficiency Calculation (Variance non-zero for correlation)
    eff_list = []
    for gen_kw, c_temp in zip(pv_gen_arr, core_temp_arr):
        if gen_kw * 1000.0 <= 5.0:
            eff_list.append(98.4)  # Base standby efficiency
        else:
            temp_delta = max(0.0, c_temp - 40.0)
            load_factor = gen_kw / max(0.5, plant_capacity_kw)
            # Inverter curves: peaks at 40-60% load, slight drop at max thermal/overload
            calculated_eff = 98.4 - (temp_delta * 0.055) - (max(0.0, load_factor - 0.7) * 1.2)
            eff_list.append(round(float(np.clip(calculated_eff, 93.0, 98.4)), 2))

    eff_arr = np.array(eff_list)
    loss_arr = np.where(
        pv_gen_arr * 1000.0 <= 5.0, 
        0.0, 
        (pv_gen_arr * 1000.0) * (1.0 - (eff_arr / 100.0))
    )

    records = []
    total_battery_joule_loss_kwh = 0.0
    max_charge_rate_kw = effective_battery_kwh * 0.5

    full_charge_reached_time = "Not Reached"
    charge_start_time = "N/A"

    for i in range(steps_per_day):
        total_mins = i * 5
        time_str = f"{total_mins // 60:02d}:{total_mins % 60:02d}"
        
        step_grid_v = float(interp_grid_v[i])
        step_grid_status, _ = classify_grid_stability(voltage=step_grid_v)
        step_is_cut = step_grid_status in ["Blackout", "Brownout"]

        load_demand_kw = float(selected_emergency_load_kw) if step_is_cut else float(max(0.05, step_demand_kw[i]))
        pv_gen_kw = float(pv_gen_arr[i])
        
        battery_flow_kw = 0.0
        grid_import_kw = 0.0
        grid_export_kw = 0.0
        dispatch_status = ""

        if battery_fault:
            current_stored_kwh = 0.0
            battery_flow_kw = 0.0
            net = pv_gen_kw - load_demand_kw
            if net > 0:
                grid_export_kw = net if not step_is_cut else 0.0
                dispatch_status = "BMS Trip: Solar Load + Export"
            else:
                grid_import_kw = abs(net) if not step_is_cut else 0.0
                dispatch_status = "BMS Trip: Failover to EB Grid"
        else:
            if pv_gen_kw >= load_demand_kw:
                surplus_solar = pv_gen_kw - load_demand_kw
                available_capacity_kwh = max(0.0, effective_battery_kwh - current_stored_kwh)
                
                if available_capacity_kwh > 0.02:
                    if charge_start_time == "N/A":
                        charge_start_time = time_str
                    
                    charge_kw = min(surplus_solar, max_charge_rate_kw, available_capacity_kwh / step_hours)
                    charge_current = (charge_kw * 1000.0) / v_battery_nom
                    joule_loss_kw = ((charge_current ** 2) * ir_ohms) / 1000.0
                    total_battery_joule_loss_kwh += (joule_loss_kw * step_hours)
                    
                    net_increase = max(0.0, (charge_kw - joule_loss_kw) * step_hours)
                    current_stored_kwh = min(effective_battery_kwh, current_stored_kwh + net_increase)
                    battery_flow_kw = charge_kw
                    
                    residual_export = surplus_solar - charge_kw
                    if residual_export > 0.01 and not step_is_cut:
                        grid_export_kw = residual_export
                        dispatch_status = "Solar Powering House + Battery Charging + Grid Export"
                    else:
                        dispatch_status = "Solar Powering House + Battery Charging"
                        
                    if current_stored_kwh >= effective_battery_kwh * 0.98 and full_charge_reached_time == "Not Reached":
                        full_charge_reached_time = time_str
                else:
                    battery_flow_kw = 0.0
                    grid_export_kw = surplus_solar if not step_is_cut else 0.0
                    dispatch_status = "Battery 100% Full: Solar Powering House + Active Grid Export"
            else:
                deficit = load_demand_kw - pv_gen_kw
                usable_battery_energy = max(0.0, current_stored_kwh - (emergency_reserve_kwh if not step_is_cut else 0.02))
                
                if usable_battery_energy > 0.01:
                    drain_kw = min(deficit, usable_battery_energy / step_hours, max_charge_rate_kw)
                    discharge_current = (drain_kw * 1000.0) / v_battery_nom
                    joule_loss_kw = ((discharge_current ** 2) * ir_ohms) / 1000.0
                    total_battery_joule_loss_kwh += (joule_loss_kw * step_hours)
                    
                    current_stored_kwh = max(0.0, current_stored_kwh - ((drain_kw + joule_loss_kw) * step_hours))
                    battery_flow_kw = -drain_kw
                    deficit -= drain_kw
                
                if deficit > 0.01 and not step_is_cut:
                    grid_import_kw = deficit
                    dispatch_status = "Battery Floor Reached: Drawing from EB Grid"
                else:
                    dispatch_status = "100% Clean Solar + Battery Supply"

        current_soc_pct = float(np.clip((current_stored_kwh / max(0.1, effective_battery_kwh)) * 100.0, 0.0, 100.0)) if not battery_fault else 0.0
        step_eff_val = round(float(eff_arr[i]), 2)

        records.append({
            "Timestamp": time_str,
            "Hour": time_str,
            "DC_Gen_kW": round(float(dc_gen_arr[i]), 3),
            "Inverter_Eff_%": step_eff_val,
            "Inverter_Efficiency": step_eff_val,
            "AC_Dispatched_kW": round(pv_gen_kw, 3),
            "Conversion_Wasted_W": round(float(loss_arr[i]), 1),
            "Solar_Gen_kW": round(pv_gen_kw, 3),
            "House_Load_kW": round(load_demand_kw, 3),
            "Battery_Flow_kW": round(battery_flow_kw, 3),
            "Battery_Stored_kWh": round(current_stored_kwh, 3),
            "Battery_SoC_Pct": round(current_soc_pct, 1),
            "Grid_Import_kW": round(grid_import_kw, 3),
            "Grid_Export_kW": round(grid_export_kw, 3),
            "Irradiance_W_m2": round(float(irr_arr[i]), 1),
            "Module_Temp_C": round(float(interp_mod_t[i]), 1),
            "DC_Current_Amps": round(float(dc_current_arr[i]), 2),
            "Inverter_Temp_C": round(float(core_temp_arr[i]), 1),
            "Inverter_Loss_W": round(float(loss_arr[i]), 1),
            "Grid_Voltage_V": round(step_grid_v, 1),
            "Grid_Status": step_grid_status,
            "Dispatch_Status": dispatch_status
        })

    df_5min = pd.DataFrame(records)
    df_15min = df_5min.iloc[::3].copy().reset_index(drop=True)
    df_primary = df_5min if str(resolution).startswith("5-Min") else df_15min

    eval_df = df_primary
    eval_step_hours = 5.0/60.0 if str(resolution).startswith("5-Min") else 15.0/60.0

    total_grid_import_kwh = (eval_df["Grid_Import_kW"] * eval_step_hours).sum()
    total_grid_export_kwh = (eval_df["Grid_Export_kW"] * eval_step_hours).sum()
    total_solar_ac_kwh = (eval_df["Solar_Gen_kW"] * eval_step_hours).sum()
    
    gross_capex = plant_capacity_kw * 60000.0
    if plant_capacity_kw >= 3.0:
        pm_subsidy = 78000.0
    elif plant_capacity_kw >= 2.0:
        pm_subsidy = 60000.0 + ((plant_capacity_kw - 2.0) * 18000.0)
    else:
        pm_subsidy = plant_capacity_kw * 30000.0
        
    net_investment = max(0.0, gross_capex - pm_subsidy)
    annual_gen_kwh = total_solar_ac_kwh * 365.0
    annual_export_kwh = total_grid_export_kwh * 365.0
    annual_savings_rs = int((annual_gen_kwh * 6.50) + (annual_export_kwh * 3.10))
    payback_years = round(net_investment / max(100.0, annual_savings_rs), 1)

    total_inv_loss_kwh = (eval_df["Inverter_Loss_W"] * eval_step_hours).sum() / 1000.0
    total_dc_kwh = total_solar_ac_kwh + total_inv_loss_kwh

    overall_grid_status = "Normal Sync"
    if "Blackout" in df_primary["Grid_Status"].values:
        overall_grid_status = "Blackout"
    elif "Brownout" in df_primary["Grid_Status"].values:
        overall_grid_status = "Brownout"
    elif "Surge" in df_primary["Grid_Status"].values:
        overall_grid_status = "Surge"

    daylight_efficiencies = [x for x in eff_arr if x > 0.0]
    avg_operating_efficiency = round(float(np.mean(daylight_efficiencies)), 2) if daylight_efficiencies else 97.5

    metrics = {
        "total_solar_kwh": round(total_solar_ac_kwh, 2),
        "total_dc_kwh": round(total_dc_kwh, 2),
        "total_load_kwh": round((eval_df["House_Load_kW"] * eval_step_hours).sum(), 2),
        "total_import_kwh": round(total_grid_import_kwh, 2),
        "total_export_kwh": round(total_grid_export_kwh, 2),
        "battery_soh_pct": round(soh_pct, 1),
        "effective_battery_kwh": round(effective_battery_kwh, 2),
        "emergency_reserve_kwh": round(emergency_reserve_kwh, 2),
        "battery_joule_loss_kwh": round(total_battery_joule_loss_kwh, 3),
        "max_inv_temp": round(eval_df["Inverter_Temp_C"].max(), 1),
        "avg_inv_eff": avg_operating_efficiency,
        "total_inv_loss_kwh": round(total_inv_loss_kwh, 3),
        "grid_status": overall_grid_status,
        "gross_capex": int(gross_capex),
        "pm_subsidy": int(pm_subsidy),
        "net_investment": int(net_investment),
        "annual_savings_rs": annual_savings_rs,
        "payback_years": payback_years,
        "charge_start_time": charge_start_time,
        "full_charge_reached_time": full_charge_reached_time
    }
    return df_primary, metrics, df_5min, df_15min
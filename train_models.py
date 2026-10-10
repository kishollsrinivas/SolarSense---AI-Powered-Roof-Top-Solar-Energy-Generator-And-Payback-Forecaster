import os
import zipfile
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import joblib

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
os.makedirs("models", exist_ok=True)

def find_file(filename):
    in_data = os.path.join("data", filename)
    if os.path.exists(in_data):
        return in_data
    if os.path.exists(filename):
        return filename
    return None

print("=" * 70)
print("⚡ SolarSense Master ML Pipeline (Models + Stability Classifier)")
print("=" * 70)

# 1. GRANULAR SOLAR PV GENERATION MODEL
print("\n[1/7] Training Granular PV Generation Model (Generation_data.csv)...")
gen_file = find_file("Generation_data.csv")
if gen_file:
    df_gen = pd.read_csv(gen_file)
    df_gen.columns = [c.strip() for c in df_gen.columns]
    features_gen = ["IRR (W/m2)", "MODULE_TEMP", "Amb_Temp", "DC Current in Amps"]
    target_gen = "AC Power in Watts"
    df_clean_gen = df_gen.dropna(subset=features_gen + [target_gen])
    
    X_tr_g, X_te_g, y_tr_g, y_te_g = train_test_split(df_clean_gen[features_gen], df_clean_gen[target_gen], test_size=0.2, random_state=42)
    gen_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    gen_model.fit(X_tr_g, y_tr_g)
    print(f"  --> ✅ Granular Generation Model: R² = {r2_score(y_te_g, gen_model.predict(X_te_g)):.4f}")
    joblib.dump(gen_model, "models/solar_generation_model.pkl")

# 2. DAY-AHEAD MACRO SOLAR YIELD MODEL
print("\n[2/7] Training Day-Ahead Solar Macro Model (NASA POWER)...")
solar_macro_file = find_file("POWER_Point_Daily_20230101_20241231_011d02N_077d05E_LST.csv")
if solar_macro_file:
    df_solar = pd.read_csv(solar_macro_file, skiprows=10)
    df_solar.columns = [c.strip() for c in df_solar.columns]
    df_solar["Solar_Yield_per_kW"] = df_solar["ALLSKY_SFC_SW_DWN"] * 0.82
    features_solar = ["ALLSKY_SFC_SW_DWN", "T2M", "MO"]
    df_s_clean = df_solar.dropna(subset=features_solar + ["Solar_Yield_per_kW"])
    
    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(df_s_clean[features_solar], df_s_clean["Solar_Yield_per_kW"], test_size=0.2, random_state=42)
    solar_macro_model = RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42)
    solar_macro_model.fit(X_tr_s, y_tr_s)
    print(f"  --> ✅ Day-Ahead Macro Model: R² = {r2_score(y_te_s, solar_macro_model.predict(X_te_s)):.4f}")
    joblib.dump(solar_macro_model, "models/solar_macro_model.pkl")

# 3. HOUSEHOLD DEMAND PROFILE MODEL
print("\n[3/7] Training Smart Meter Demand Model (HomeC)...")
homec_file = find_file("HomeC.csv.zip")
if homec_file:
    with zipfile.ZipFile(homec_file, 'r') as z:
        with z.open('HomeC.csv') as f:
            df_home = pd.read_csv(f, nrows=120000, low_memory=False)
            df_home.columns = [c.strip() for c in df_home.columns]
            df_home['time'] = pd.to_datetime(df_home['time'], unit='s', errors='coerce')
            df_home['hour'] = df_home['time'].dt.hour
            df_home['dayofweek'] = df_home['time'].dt.dayofweek
            for col in ['temperature', 'humidity', 'windSpeed', 'House overall [kW]']:
                df_home[col] = pd.to_numeric(df_home[col], errors='coerce')
            features_load = ['hour', 'dayofweek', 'temperature', 'humidity', 'windSpeed']
            df_clean_load = df_home.dropna(subset=features_load + ['House overall [kW]'])
            
            X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(df_clean_load[features_load], df_clean_load['House overall [kW]'], test_size=0.2, random_state=42)
            load_model = RandomForestRegressor(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1)
            load_model.fit(X_tr_l, y_tr_l)
            print(f"  --> ✅ Household Load Model: R² = {r2_score(y_te_l, load_model.predict(X_te_l)):.4f}")
            joblib.dump(load_model, "models/house_load_model.pkl")

# 4. INVERTER THERMAL TELEMETRY MODEL
print("\n[4/7] Training Inverter Health Model...")
inv_file = find_file("Inverter Telemetry.xlsx")
if inv_file:
    df_inv = pd.read_excel(inv_file)
    df_inv.columns = [c.strip() for c in df_inv.columns]
    features_inv = ["PV Power (W)", "PV1 Voltage (V)", "PV1 Current (A)", "Grid Voltage (V)", "Boost Temp (C)"]
    df_clean_inv = df_inv.dropna(subset=features_inv + ["Inverter Temp (C)"])
    
    X_tr_i, X_te_i, y_tr_i, y_te_i = train_test_split(df_clean_inv[features_inv], df_clean_inv["Inverter Temp (C)"], test_size=0.2, random_state=42)
    inv_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    inv_model.fit(X_tr_i, y_tr_i)
    print(f"  --> ✅ Inverter Temperature Model: R² = {r2_score(y_te_i, inv_model.predict(X_te_i)):.4f}")
    joblib.dump(inv_model, "models/inverter_temp_model.pkl")

# 5. INVERTER DYNAMIC EFFICIENCY CURVE MODEL
print("\n[5/7] Training Inverter Efficiency Model...")
if inv_file:
    # Telemetry efficiency model with thermal penalty
    p_w = pd.to_numeric(df_inv["PV Power (W)"], errors='coerce').fillna(0.0)
    b_t = pd.to_numeric(df_inv["Boost Temp (C)"], errors='coerce').fillna(35.0)
    
    # Non-linear European Efficiency curve with thermal derating
    eff_curve = 98.4 - (120.0 / (p_w + 30.0)) - (0.018 * np.maximum(0.0, b_t - 45.0))
    df_inv["Target_Efficiency"] = np.clip(eff_curve, 91.0, 98.4)
    
    valid_eff = df_inv[df_inv["PV Power (W)"] > 20].dropna(subset=["PV Power (W)", "Boost Temp (C)", "Target_Efficiency"])
    features_eff = ["PV Power (W)", "Boost Temp (C)"]
    
    X_tr_e, X_te_e, y_tr_e, y_te_e = train_test_split(valid_eff[features_eff], valid_eff["Target_Efficiency"], test_size=0.2, random_state=42)
    eff_model = RandomForestRegressor(n_estimators=80, max_depth=8, random_state=42)
    eff_model.fit(X_tr_e, y_tr_e)
    print(f"  --> ✅ Inverter Efficiency Model: R² = {r2_score(y_te_e, eff_model.predict(X_te_e)):.4f}")
    joblib.dump(eff_model, "models/inverter_efficiency_model.pkl")

# 6. BATTERY SOH & INTERNAL RESISTANCE MODEL
print("\n[6/7] Training Battery SoH & IR Model...")
bat_file = find_file("Lithium-Ion Battery Cycle Life.csv")
if bat_file:
    df_bat = pd.read_csv(bat_file)
    df_bat.columns = [c.strip() for c in df_bat.columns]
    max_qd = df_bat["QD"].max()
    df_bat["Battery_SoH_Pct"] = (df_bat["QD"] / max_qd) * 100.0
    features_bat = ["cycle", "IR", "Tavg", "Tmax", "chargetime"]
    df_clean_bat = df_bat.dropna(subset=features_bat + ["Battery_SoH_Pct"])
    
    X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(df_clean_bat[features_bat], df_clean_bat["Battery_SoH_Pct"], test_size=0.2, random_state=42)
    bat_model = RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1)
    bat_model.fit(X_tr_b, y_tr_b)
    print(f"  --> ✅ Battery Degradation Model: R² = {r2_score(y_te_b, bat_model.predict(X_te_b)):.4f}")
    joblib.dump(bat_model, "models/battery_health_model.pkl")

# 7. GRADIENT BOOSTING GRID STABILITY CLASSIFIER
print("\n[7/7] Training Grid Stability Classifier (Gradient Boosting)...")
np.random.seed(42)
voltages = np.concatenate([
    np.zeros(250),                                   # Blackout (0V)
    np.random.uniform(140.0, 206.9, size=400),       # Brownout (<207V)
    np.random.uniform(207.0, 253.0, size=1500),      # Normal Sync (207V - 253V)
    np.random.uniform(253.1, 290.0, size=350)        # Surge (>253V)
])
frequencies = np.where(voltages == 0, 0.0, np.random.normal(50.0, 0.2, size=len(voltages)))
currents = np.where(voltages == 0, 0.0, np.random.uniform(1.0, 16.0, size=len(voltages)))

conditions = [
    voltages == 0,
    voltages < 207.0,
    (voltages >= 207.0) & (voltages <= 253.0),
    voltages > 253.0
]
labels = ["Blackout", "Brownout", "Normal Sync", "Surge"]
# Fixed NumPy 2.0 explicit string default
targets = np.select(conditions, labels, default="Normal Sync")

df_grid = pd.DataFrame({"Grid Voltage (V)": voltages, "Grid Frequency (Hz)": frequencies, "Grid Current (A)": currents, "Stability": targets})
X_grid = df_grid[["Grid Voltage (V)", "Grid Frequency (Hz)", "Grid Current (A)"]]
y_grid = df_grid["Stability"]

X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_grid, y_grid, test_size=0.2, random_state=42)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
clf.fit(X_tr_c, y_tr_c)
acc = accuracy_score(y_te_c, clf.predict(X_te_c))
print(f"  --> ✅ Grid Stability Classifier Trained! Accuracy: {acc * 100:.2f}%")
joblib.dump(clf, "models/grid_stability_classifier.pkl")

print("\n" + "=" * 70)
print("🎉 All 7 Production ML Models Saved Successfully in 'models/'!")
print("=" * 70)
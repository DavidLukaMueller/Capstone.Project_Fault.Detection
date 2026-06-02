import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import os

# =====================================================================
# 1. LOAD DATA & PREPROCESSING
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "Datasets", "MZVAV-2-2_Numbered.csv")
# file_path = os.path.join(BASE_DIR, "Datasets", "MZVAV-2-1_Numbered.csv")
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Standardize fault column
if "Fault Detection Ground Truth" in df.columns:
    df = df.rename(columns={"Fault Detection Ground Truth": "Fault"})

# Fallback check if Heating Coil column is missing
if "AHU: Heating Coil Valve Control Signal" not in df.columns:
    df["AHU: Heating Coil Valve Control Signal"] = 0.0

# Create an ID for each continuous block of occupancy/unoccupancy
df["Occ_Block"] = (df["Occupancy Mode Indicator"] != df["Occupancy Mode Indicator"].shift()).cumsum()

# Minutes active counter
df["Minutes_Active"] = df.groupby("Occ_Block").cumcount() + 1


# =====================================================================
# 2. SMOOTH SENSORS
# =====================================================================
def smooth_sensor(df, col):
    roll = df.groupby("Occ_Block")[col].transform(lambda x: x.rolling(6, min_periods=6).mean())
    return roll.fillna(df[col])

# Sensors for Fault 1 (Damper)
df["Roll_OAT"] = smooth_sensor(df, "AHU: Outdoor Air Temperature")
df["Roll_RAT"] = smooth_sensor(df, "AHU: Return Air Temperature")
df["Roll_MAT"] = smooth_sensor(df, "AHU: Mixed Air Temperature")
df["Roll_Static_Press"] = smooth_sensor(df, "AHU: Supply Air Duct Static Pressure")
df["Roll_Static_Press_SP"] = smooth_sensor(df, "AHU: Supply Air Duct Static Pressure Set Point")
df["Roll_OAD"] = smooth_sensor(df, "AHU: Outdoor Air Damper Control Signal")
df["Roll_RAD"] = smooth_sensor(df, "AHU: Return Air Damper Control Signal")
df["Roll_SAF"] = smooth_sensor(df, "AHU: Supply Air Fan Speed Control Signal")

# Sensors for Fault 2 (Cooling)
df["Roll_Supply_Temp"] = smooth_sensor(df, "AHU: Supply Air Temperature")
df["Roll_Set_Point"]   = smooth_sensor(df, "AHU: Supply Air Temperature Set Point")
df["Roll_Heat_Valve"]  = smooth_sensor(df, "AHU: Heating Coil Valve Control Signal")

# Sensors for Fault 3 (Heating)
df["Roll_Cool_Valve"] = smooth_sensor(df, "AHU: Cooling Coil Valve Control Signal")

if "AHU: Supply Air Fan Status" not in df.columns:
    df["AHU: Supply Air Fan Status"] = (df["AHU: Supply Air Fan Speed Control Signal"] > 0.05).astype(float)

df["Roll_Fan_Status"] = smooth_sensor(df, "AHU: Supply Air Fan Status")

# =====================================================================
# 3. CALCULATE ROW-BY-ROW PHYSICAL FEATURES
# =====================================================================
# Fault 1 Features
df["Ideal_MAT"] = (df["Roll_OAD"] * df["Roll_OAT"]) + (df["Roll_RAD"] * df["Roll_RAT"])
df["MAT_Error"] = (df["Roll_MAT"] - df["Ideal_MAT"]).abs()
df["Fan_Effort_Ratio"] = df["Roll_Static_Press"] / (df["Roll_SAF"] + 0.01)
df["Pressure_Error"] = df["Roll_Static_Press"] - df["Roll_Static_Press_SP"]

# Fault 2 Features
df["Heating_Demand_Mismatch"] = df["Roll_Heat_Valve"] * (df["Roll_Set_Point"] - df["Roll_Supply_Temp"])
df["Over_Cooling_Error"] = df["Roll_Set_Point"] - df["Roll_Supply_Temp"]

# NEW: Hardware Fault Features (Fan Mismatch)
# If command is > 5% but status is 0 (or vice versa), this will flag as 1
df["Fan_Command_State"] = (df["Roll_SAF"] > 0.05).astype(float)
df["Fan_Status_State"] = (df["Roll_Fan_Status"] > 0.5).astype(float)
df["Fan_Mismatch_Error"] = (df["Fan_Command_State"] - df["Fan_Status_State"]).abs()


# =====================================================================
# 4. FILTER DATA (Steady State Only)
# =====================================================================
df_steady = df[(df["Occupancy Mode Indicator"] == 1) & (df["Minutes_Active"] > 60)].copy()


# =====================================================================
# 5. COMPRESS THE DAY (Block-Level Aggregation)
# =====================================================================
def p95(x): return x.quantile(0.95)

# Extract the maximum fault code present in the block
def get_block_fault(x): return int(x.max()) 

agg_functions = {
    # F1
    "MAT_Error": p95,
    "Fan_Effort_Ratio": "mean",
    "Pressure_Error": "mean",
    # F2
    "Heating_Demand_Mismatch": p95,
    "Over_Cooling_Error": "mean",
    # F3
    "Roll_Cool_Valve": "mean",
    # New Aggregations needed for new rules
    "Fan_Mismatch_Error": "mean",
    "Roll_Supply_Temp": "mean",
    "Roll_Set_Point": "mean",
    "Roll_Heat_Valve": "mean",
    # Target
    "Fault": get_block_fault
}

df_compressed = df_steady.groupby("Occ_Block").agg(agg_functions).reset_index()

# Rename columns for clarity based on their aggregations
df_compressed = df_compressed.rename(columns={
    "MAT_Error": "MAT_Error_p95",
    "Fan_Effort_Ratio": "Fan_Effort_Ratio_mean",
    "Pressure_Error": "Pressure_Error_mean",
    "Heating_Demand_Mismatch": "Heating_Demand_Mismatch_p95",
    "Over_Cooling_Error": "Over_Cooling_Error_mean",
    "Roll_Cool_Valve": "Roll_Cool_Valve_mean",
    
    "Fan_Mismatch_Error": "Fan_Mismatch_Error_mean",
    "Roll_Supply_Temp": "Roll_Supply_Temp_mean",
    "Roll_Set_Point": "Roll_Set_Point_mean",
    "Roll_Heat_Valve": "Roll_Heat_Valve_mean",
    "Fault": "True_Fault"
})


# =====================================================================
# 6. UNIFIED FAULT DETECTION FUNCTION
# =====================================================================
def detect_fault(row):
    """
    Evaluates physical features for a steady-state block.
    Returns:
      1 : Fault 1 (Damper)
      2 : Fault 2 (Cooling)
      3 : Fault 3 (Heating)
      4 : Fault 4 (Fan / Hardware Mismatch)
      5 : Fault 5 (Low Pressure - Leak/Broken Belt)
      6 : Fault 6 (High Pressure - Clogged Filter)
      0 : Normal Operation (No Fault)
    """
    # --- Check Fault 1 (Damper) ---
    F1_LOW_FAN = 1.05
    F1_HIGH_PRESS = 1.13
    F1_PERFECT_TEMP = 2.41
    
    if (row["Fan_Effort_Ratio_mean"] <= F1_LOW_FAN) or \
       (row["Fan_Effort_Ratio_mean"] > F1_LOW_FAN and 
        row["Pressure_Error_mean"] > F1_HIGH_PRESS and 
        row["MAT_Error_p95"] <= F1_PERFECT_TEMP):
        return 1

    # --- Check Fault 2 (Cooling) ---
    F2_REHEAT_FIGHT = 9.62
    F2_UNDER_COOLING = -15.68
    
    if (row["Heating_Demand_Mismatch_p95"] > F2_REHEAT_FIGHT) or \
       (row["Over_Cooling_Error_mean"] <= F2_UNDER_COOLING):
        return 2
    # NEW RULE: Air is too warm (needs cooling), cooling valve is closed, and heating is OFF
    elif (row["Roll_Supply_Temp_mean"] > row["Roll_Set_Point_mean"] + 2.0) and \
         (row["Roll_Cool_Valve_mean"] < 0.05) and \
         (row["Roll_Heat_Valve_mean"] < 0.05):
        return 2

    # --- Check Fault 3 (Heating) ---
    F3_VALVE_THRESH = 0.4865
    
    if row["Roll_Cool_Valve_mean"] > F3_VALVE_THRESH:
        return 3
    # NEW RULE: Air is too cold (needs heating), heating valve is closed, and cooling is OFF
    elif (row["Roll_Supply_Temp_mean"] < row["Roll_Set_Point_mean"] - 2.0) and \
         (row["Roll_Heat_Valve_mean"] < 0.05) and \
         (row["Roll_Cool_Valve_mean"] < 0.05):
        return 3

    # --- Check NEW Fault 4 (Fan Hardware Mismatch) ---
    # Trigger if fan signal and fan status disagree for > 10% of the block
    if row["Fan_Mismatch_Error_mean"] > 0.1:
        return 4 
    
    # --- Check NEW Fault 5 & 6 (Pressure Faults) ---
    # Placed below F1-F3 so it doesn't accidentally override your primary models
    # Thresholds (0.5 / -0.5 in wc) can be adjusted as needed
    if row["Pressure_Error_mean"] > 0.5:
        return 6  # High Pressure (Clogged Vent/Filter)
    elif row["Pressure_Error_mean"] < -0.5:
        return 5  # Low Pressure (Air leak, Broken fan blades)

    # --- Default to Normal ---
    return 0

# Apply the function to get predictions
df_compressed["Predicted_Fault"] = df_compressed.apply(detect_fault, axis=1)


# =====================================================================
# 7. EVALUATE TRANSLATION ACCURACY (ALL FAULTS)
# =====================================================================
# Expanded labels and names to support the new fault types
labels = [0, 1, 2, 3, 4, 5, 6]
class_names = [
    "0: Normal", 
    "1: Damper", 
    "2: Cooling", 
    "3: Heating", 
    "4: Fan HW", 
    "5: Low Press", 
    "6: High Press"
]

print("\n=========================================================")
print("--- Unified Physical Rule Confusion Matrix ---")
print("=========================================================")
cm = confusion_matrix(df_compressed["True_Fault"], df_compressed["Predicted_Fault"], labels=labels)
cm_df = pd.DataFrame(
    cm, 
    index=[f"Actual {name}" for name in class_names], 
    columns=[f"Pred {name}" for name in class_names]
)
print(cm_df.to_string())

print("\n=========================================================")
print("--- Unified Physical Rule Classification Report ---")
print("=========================================================")
print(classification_report(
    df_compressed["True_Fault"], 
    df_compressed["Predicted_Fault"], 
    labels=labels,
    target_names=class_names,
    zero_division=0
))
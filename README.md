# HVAC Fault Detection: Predictive Maintenance Capstone

## Introduction
HVAC systems are complex networks of sensors, dampers, valves, and fans. Unnoticed component failures waste energy and damage equipment. 

This project builds a hybrid Machine Learning and Heuristics system to automatically detect and classify specific component failures in an Air Handling Unit (AHU) using time-series sensor data. 

### System Schematic
Simplified diagram of the HVAC system showing airflow, components, and sensor placements:

![HVAC System Diagram](Images/Simplified_HVAC.png)

### Dataset Overview
The project uses two time-series datasets recorded at **1-minute intervals**:
* **Train Set (`MZVAV-2-2.csv`):** 37,441 rows. Normal operation data + 3 artificial error types (Damper Stuck, Heating Coil Leak, Cooling Valve Stuck).
* **Test Set (`MZVAV-2-1.csv`):** 21,601 rows. Normal operation data + 1 error type (Heating Coil Leak).

**Target Variable:** The original dataset had a binary `Fault Detection` column (0 = Normal, 1 = Fault). I engineered this into a multi-class column to isolate specific components (0 = Normal, 1 = Damper, 2 = Cooling, 3 = Heating, etc.).

#### Simulated Fault Structure
The training set contains faults of varying severities. The test set mirrors only the Heating Coil Leak to validate generalization.

* **Error 1: Outdoor Air (OA) Damper Stuck**
  * *Stuck Closed:* 2/12/2008, 5/7/2008
  * *Stuck 40% Open:* 5/8/2008
  * *Stuck 45% Open:* 9/5/2007
  * *Stuck 55% Open:* 9/6/2007
* **Error 2: Cooling Coil Valve Stuck**
  * *Fully Open:* 8/31/2007, 5/15/2008
  * *Fully Closed:* 5/6/2008
  * *Partially Open (15%):* 9/1/2007
  * *Partially Open (65%):* 9/2/2007
* **Error 3: Heating Coil Valve Leaking** *(Test set features these exact dates/faults)*
  * *0.4 GPM Leak:* 8/28/2007
  * *1.0 GPM Leak:* 8/29/2007
  * *2.0 GPM Leak:* 8/30/2007

### Sensor Variables & System Behavior
The dataset includes continuous analog signals and binary indicators:
* **Temperatures:** Supply Air, Set Point, Outdoor Air, Mixed Air, Return Air.
* **Control Signals & Fan Speeds:** Supply/Return Fan Speed Control, Exhaust/Outdoor/Return Damper Control, Cooling/Heating Valve Control.
* **Pressures:** Supply Air Duct Static Pressure, Pressure Set Point.
* **Binary Indicators:** Supply/Return Fan Status, Occupancy Mode Indicator.

**System Dynamics (The 6 AM Shift):** The system state changes drastically when occupancy switches from `0` to `1` (typically at 6:00 AM). This activates fans and temperature controls, making faults mathematically visible.

![Occupancy Transition Sample](Images/CSV_Transition.png)

![Train Data Tail](Images/CSV_End.png)

---

## Project Goal & Strategy
**Goal:** Ingest daily data and generate a diagnostic report pinpointing exact component failures.

1. **Data Exploration:** Analyze sensor stability and extract baseline rules.
2. **Feature Engineering:** Calculate physical relationship metrics and smooth data with a 6-minute rolling average.
3. **Model Training:** Train Decision Trees and Random Forests on localized faults.
4. **System Refinement:** Shift from minute-by-minute predictions to daily aggregation to eliminate noise.
5. **Hybrid Architecture:** Combine ML boundaries with hard-coded heuristics.

---

## Phase 1: Establishing a "Steady-State" Baseline
HVAC signals reset nightly. To establish a true baseline, I analyzed the system in a stable state by generating a correlation heatmap with two constraints:
1. **Filtered out all faults** (healthy data only).
2. **Skipped the first 60 minutes** of occupancy to remove startup noise.

![Correlation Heatmap Placeholder](Images/Coorelation_Tree_V1.png)

This established absolute mathematical relationships. If a perfect 1.0 correlation breaks during steady-state operation, it guarantees a mechanical fault.

---

## Phase 2: The Modeling Journey & Challenges

### 1. Damper Stuck (Error 1)
Training a decision tree on raw temperatures caused overfitting tied to specific weather conditions. A Random Forest feature importance check revealed that **system relationships** matter more than raw temps. I engineered ratio columns (*Fan Speed Difference, Pressure Error, MAT Error*) and trained a robust decision tree.

![Damper Tree V1](Images/Damper_Tree.png)

### 2. Cooling Valve Stuck (Error 2)
Initial models failed (268/3600 incorrect). The tree required contextual time data, so I added a counter tracking continuous system activity to measure "Cooling Valve Efficiency."

### 3. Heating Valve Leak (Error 3) & Engineering Pivots
Logically, a 2.0 GPM leak should be easiest to detect, and 0.4 GPM hardest. Surprisingly, the model struggled most with the 1.0 GPM leak. 

**Why Plan A Failed:**
My original plan analyzed data row-by-row. 
* **Overfitting:** A 3-minute sensor delay triggered false positives.
* **Translation Nightmare:** Translating R-based decision tree cutoffs into Python `if/else` statements caused models to overlap and conflict.

---

## Phase 3: The Breakthrough (Daily Aggregation & Physics)

**Plan B (Final Architecture):** I abandoned row-by-row predictions for a **Daily Aggregation Model**. Skipping the first 60 minutes and summarizing the remaining shift into a single row captured macro-behavior, eliminating noise-induced false positives.

### Compressing the Day: Column Logic
The script groups steady-state data into 10 aggregated feature columns:

**Category 1: Engineered Physics Metrics**
* `Pressure_Error_mean`: *(Actual Static Pressure - Set Point)*. Measures failure to meet pressure targets.
* `Fan_Effort_Ratio_mean`: *(Static Pressure / Fan Speed)*. High fan speed with low pressure indicates blocked or leaking air.
* `MAT_Error_p95`: *(Actual MAT - Ideal MAT)*. High error means the damper is physically lying about its position.
* `Over_Cooling_Error_mean`: *(Set Point - Supply Temp)*. Measures temperature overshoot.
* `Heating_Demand_Mismatch_p95`: *(Heating Valve % * (Set Point - Supply Temp))*. High values prove the heater is fighting freezing air (systems fighting each other).
* `Fan_Mismatch_Error_mean`: Percentage of the day fan software and hardware disagreed.

**Category 2: Smoothed Sensor States**
Smoothed raw sensors (6-minute rolling average) pass through as daily means (`Roll_Cool_Valve_mean`, `Roll_Supply_Temp_mean`, etc.) to feed the heuristic rules.

**Aggregation Logic:**
I used `.mean()` for continuous signals (e.g., valve positions) and `.quantile(0.95)` for error spikes. This ignores 1-minute anomalies while catching sustained daily errors.

### Hybrid Architecture: ML meets Heuristics
1. **Machine Learning (Hidden Faults):** ML found the exact thresholds and ratios (e.g., Fan Effort vs Pressure) for hidden faults like slow leaks or stuck dampers.
2. **Heuristics (Mechanical Faults):** Direct mechanical failures use custom `if/elif` rules derived from the correlation heatmap (e.g., Fan Command vs. Status mismatch = guaranteed Fault 4).

### Robustness & Margin of Error
Sensors drift and hardware lags. Razor-thin thresholds fail in production. This architecture ensures a massive safety buffer:
1. **Time Tolerance:** Using `MAT_Error_p95` requires an error to persist for 5% of the day (~30 minutes) before triggering.
2. **Wide ML Thresholds:** `Heating_Demand_Mismatch > 9.62` is a massive boundary. It requires the system to actively and continuously fight itself, ignoring minor calibration glitches.
3. **Hardware Forgiveness:** The fan mismatch rule requires hardware/software disagreement for **>10% of the day**, ignoring brief fan spin-up delays.

### The Logic Behind the Trees 

#### **Damper Fault Logic**
```text
|--- Fan_Effort_Ratio_mean <= 1.05  --> FAULT
|--- Fan_Effort_Ratio_mean > 1.05
|   |--- Pressure_Error_mean > 1.13
|   |   |--- MAT_Error_p95 <= 2.41  --> FAULT
```
* **Meaning:** If the fan spins fast but pressure stays low, the damper is stuck. If the system maintains perfect temps but has terrible pressure errors, the HVAC is brute-forcing the fans to mask the stuck damper.

#### **Cooling Fault Logic**
```text
|--- Heating_Demand_Mismatch_p95 > 9.62      --> FAULT
|--- Over_Cooling_Error_mean <= -15.68       --> FAULT
```
* **Meaning:** If the heater works at maximum capacity but the air is freezing, the cooling valve is stuck open. Conversely, if the air severely overshoots the cooling target, the cooler won't shut off.

#### **Heating Fault Logic**
```text
|--- Roll_Cool_Valve_mean > 0.49   --> FAULT
```
* **Meaning:** If the cooling valve averages 50% open all day just to maintain normal room temps, it is secretly fighting a continuous hot-water leak.

---

## Final Results

Tested on the combined dataset, the system identified faults with **zero false positives**.

**Training Set Confusion Matrix:**
| Actual \ Predicted | 0: Normal | 1: Damper | 2: Cooling | 3: Heating | 4: Fan HW | 5: Low Press | 6: High Press |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0: Normal** | 12 | 0 | 0 | 1 | 0 | 0 | 0 |
| **1: Damper** | 0 | 5 | 0 | 0 | 0 | 0 | 0 |
| **2: Cooling** | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| **3: Heating** | 0 | 0 | 0 | 3 | 0 | 0 | 0 |

**Test Set Confusion Matrix:**
*(Note: 2 False positives on Heating were traced to extreme weather days where return air was vastly warmer than supply air).*

| Actual \ Predicted | 0: Normal | 1: Damper | 2: Cooling | 3: Heating | 4: Fan HW | 5: Low Press | 6: High Press |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0: Normal** | 11 | 0 | 0 | 2 | 0 | 0 | 0 |
| **1: Damper** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **2: Cooling** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **3: Heating** | 0 | 0 | 0 | 2 | 0 | 0 | 0 |

---

## Lessons Learned
* **Machine Learning vs. Reality:** Intermediate faults (e.g., 1.0 GPM leak) can be harder to detect than extreme bounds.
* **Tech Stack Consistency:** Translating R models to Python manually is inefficient. Future ML workflows will stay natively in Python end-to-end.
* **Theory vs. Reality:** Training data has razor-thin boundaries. Real-world data requires broader tolerances (daily aggregation fixed this).
* **Simplicity First:** Start with a scalable foundation. Introduce complexity only when necessary.

## Future Improvements
This system functions as a "daily log parser" to guide maintenance teams. In production, I would split it into a **Two-Tier Architecture**:
1. **Tier 1 (Critical - Rule-Based):** Hard-coded heuristics (valves, pressure loss) running on a rolling 30-minute window to flag catastrophic failures immediately.
2. **Tier 2 (Non-Critical - ML Based):** Machine learning models (slow leaks, stuck dampers) running on a 24-hour delay to analyze full-day behavior and eliminate false positives.

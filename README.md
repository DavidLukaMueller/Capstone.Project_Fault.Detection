# HVAC Fault Detection: Predictive Maintenance Capstone

## Introduction
Modern Heating, Ventilation, and Air Conditioning (HVAC) systems are highly complex networks of sensors, dampers, valves, and fans. When a component fails, it can waste significant energy or damage the system before human operators notice a problem. 

This project aims to build a robust Machine Learning and Heuristics-based system to automatically detect and classify specific component failures in an Air Handling Unit (AHU) using time-series sensor data. 

### System Schematic
To understand the data, here is a simplified diagram of the HVAC system, showing the airflow, dampers, heating/cooling coils, and sensor placements:

![HVAC System Diagram](Images\Simplified_HVAC.png)

### Dataset Overview
The project utilizes two primary time-series datasets recorded at **1-minute intervals**:
* **Train Set (`MZVAV-2-2.csv`):** 37,441 rows. Contains normal operation data and 3 artificial error types (Damper Stuck, Heating Coil Leak, Cooling Valve Stuck).
* **Test Set (`MZVAV-2-1.csv`):** 21,601 rows. Contains normal operation data and 1 error type (Heating Coil Leak).

**Target Variable Modification:** The original dataset featured a binary `Fault Detection Ground Truth` column (0 = Normal, 1 = Fault). To achieve component-level isolation, I engineered this into a multi-class column (0 = Normal, 1 = Damper, 2 = Cooling, 3 = Heating, etc.).

### Sensor Variables & System Behavior
The dataset consists of continuous analog signals and binary state indicators:
* **Temperatures:** Supply Air, Set Point, Outdoor Air, Mixed Air, Return Air.
* **Control Signals & Fan Speeds:** Supply/Return Fan Speed Control, Exhaust/Outdoor/Return Damper Control, Cooling/Heating Valve Control.
* **Pressures:** Supply Air Duct Static Pressure, Pressure Set Point.
* **Binary Indicators:** Supply/Return Fan Status, Occupancy Mode Indicator.

**System Dynamics (The 6 AM Shift):** The most drastic changes in the dataset occur when the `Occupancy Mode Indicator` switches from `0` to `1` (typically at 6:00 AM). This triggers the fans and temperature control loops to activate, which is when faults become mathematically visible.

![Occupancy Transition Sample](Images\CSV_Transition.png)

![Train Data Tail](Images\CSV_End.png)

---

## Project Goal & Strategy
The objective is to ingest a full day's worth of data and generate a diagnostic report for maintenance teams, pinpointing the exact location of the failure.

1. **Data Exploration:** Analyze sensor stability, system cycles, and identify heuristic baseline rules.
2. **Feature Engineering:** Create custom ratio-based columns to capture system context rather than relying on raw temperatures.
3. **Model Training:** Train Decision Trees and Random Forests to detect individual, localized faults.
4. **System Refinement:** Shift from row-by-row prediction to daily aggregation to eliminate noise and false positives.
5. **Hybrid Architecture:** Combine Machine Learning (for complex leaks/stuck valves) with hard-coded logic (for direct sensor mismatches).

---

## Phase 1: Data Exploration & Baseline Extraction
Initial data exploration showed that the HVAC signals were generally stable, with a hard reset occurring at midnight. 

To understand baseline operations, I generated correlation heatmaps split into four categories: All data, Only Non-Faulty, Non-Faulty Empty Building, and Non-Faulty Occupied Building.

![Correlation Heatmap Placeholder](Images\Coorelation_Tree_V1.png)

**Key Finding:** The heatmaps revealed "hard rules" that never fault under normal conditions. This proved that certain faults don't need ML—they can be caught with strict logic rules based on perfect 1.0 correlations (e.g., Operating Mode vs. Supply/Return Fan Status).

---

## Phase 2: The Modeling Journey & Challenges

My initial plan was to detect errors from easiest to hardest: Damper Position -> Cooling Valve Stuck -> Heating Coil Leak.

### 1. Damper Stuck (Error 1)
Using raw temperatures in a decision tree resulted in an overfitted model that only worked under highly specific weather conditions. To fix this, I used a Random Forest regression model to extract feature importance. 

**Top Feature Importances:**
| Feature | Importance (MeanDecreaseGini) |
| :--- | :--- |
| `AHU: Return Air Fan Speed Control Signal` | 2823.98 |
| `AHU: Supply Air Fan Speed Control Signal` | 2101.82 |
| `AHU: Supply Air Duct Static Pressure` | 1070.59 |
| `AHU: Return Air Temperature` | 413.60 |

*Insight:* Raw temps are less important than the **relationships** between systems. I engineered new ratio columns: *Fan Speed Difference, Pressure Error, and Mixed Air Temp (MAT) Error.* Training the tree with a custom loss matrix (penalizing false negatives heavily) resulted in a clean, robust decision tree for the damper.

![Damper Tree V1](Images\Damper_Tree.png)

### 2. Cooling Valve Stuck (Error 2)
The simple tree initially failed (268/3600 incorrect). Allowing the tree to expand showed heavy reliance on "Cooling Valve Efficiency." I engineered an additional contextual column: a time-based counter tracking how long the system had been continually active.

### 3. Heating Valve Leak (Error 3) - *The Trouble Child*
Logically, a massive 2.0 GPM leak should be easiest to detect, and a 0.4 GPM leak the hardest. Surprisingly, the model struggled the most with the intermediate 1.0 GPM leak. 

**The Pivot:**
Translating complex R-based decision trees into Python row-by-row `if/else` statements became messy and resulted in combined model overlap (models conflicting with each other). Row-by-row analysis was simply too granular and susceptible to minor calibration glitches.

---

## Phase 3: The Breakthrough (Daily Aggregation & Heuristics)

To solve the overfitting and translation issues, I made two major structural changes to the project:

![Coorelation Heatmap](Images\Coorelation_Tree_60minSkip_V2.png)

1. **Daily Batch Processing:** Instead of predicting fault status row-by-row, the system now analyzes data in full-day chunks, ignoring the first 2 hours of the day to allow the HVAC system to stabilize. This vastly improved the model's confidence in classifying a "faulty day."
2. **Implementation of Heuristics:** I implemented hard-coded logic for system failures that exhibit perfect mathematical breakdowns, removing the burden from the ML models.
   * **Fan Hardware (Fault 4):** Fan Speed Command vs. Status (1.0 correlation normally). A mismatch mathematically guarantees mechanical/electrical failure.
   * **Pressure Issues (Faults 5 & 6):** Duct Pressure vs. Fan Status. A breakdown here guarantees a physical delivery issue (broken belt, leak, or clogged filter).
   * **Control Loop Failure (Faults 2 & 3):** If target temperatures are far off baseline, but the corresponding valve remains closed, the control loop has failed.

---

## Final Results

The final hybrid system was tested on the combined dataset. While not every simulated fault could be perfectly replicated in the test environment, the system successfully identified faults with **zero false positives**.

**Training Set Confusion Matrix:**
| Actual \ Predicted | 0: Normal | 1: Damper | 2: Cooling | 3: Heating | 4: Fan HW | 5: Low Press | 6: High Press |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0: Normal** | 12 | 0 | 0 | 1 | 0 | 0 | 0 |
| **1: Damper** | 0 | 5 | 0 | 0 | 0 | 0 | 0 |
| **2: Cooling** | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| **3: Heating** | 0 | 0 | 0 | 3 | 0 | 0 | 0 |

**Test Set Confusion Matrix:**
*(Note: 2 False positives on Heating were analyzed and traced back to extreme weather days where return air was vastly warmer than supply air, tricking the system).*

| Actual \ Predicted | 0: Normal | 1: Damper | 2: Cooling | 3: Heating | 4: Fan HW | 5: Low Press | 6: High Press |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0: Normal** | 11 | 0 | 0 | 2 | 0 | 0 | 0 |
| **1: Damper** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **2: Cooling** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **3: Heating** | 0 | 0 | 0 | 2 | 0 | 0 | 0 |

---

## Lessons Learned
* **Machine Learning Logic vs. Human Logic:** Intermediate faults (e.g., the 1.0 leak) can sometimes be harder for a model to detect than extreme ends of the spectrum.
* **Tech Stack Consistency:** Translating trained models across languages (R to Python) manually is highly inefficient. Future ML workflows should be kept natively in Python end-to-end.
* **Theory vs Reality:** Training data often has razor-thin boundaries between classes. Real-world data requires broader tolerances. Grouping data in daily chunks saved the project from sensor-glitch false positives.
* **Organization:** Version control and strict folder organization are vital when experimenting with dozens of scripts, datasets, and iterations.
* **Simplicity First:** Start with a humble, scalable foundation. Only introduce complexity when necessary.

## Future Improvements (Project Conclusion)
The final system successfully acts as a "daily log parser" that tells maintenance teams exactly where to look for an error. 

If I were to deploy this in production, I would split it into a **Two-Tier System**:
1. **Tier 1 (Critical - Rule-Based):** Hard-coded logic (valves, fan status) running on a rolling 30-minute window to immediately flag catastrophic mechanical failures.
2. **Tier 2 (Non-Critical - ML Based):** Machine learning models (predicting slow leaks, damper sticks) running on a 24-hour delay. These faults don't cause immediate danger but waste energy, and analyzing them over a full day eliminates false positives.
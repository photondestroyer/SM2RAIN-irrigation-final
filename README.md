# SM2RAIN Irrigation Detection System

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Background](#theoretical-background)
3. [Calibration Methodology](#calibration-methodology)
4. [Data Requirements](#data-requirements)
5. [File Structure](#file-structure)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [Parameters and Bounds](#parameters-and-bounds)
9. [Performance Metrics](#performance-metrics)
10. [Calibration Results](#calibration-results)
11. [Output Files](#output-files)
12. [Troubleshooting](#troubleshooting)
13. [References](#references)

---

## Overview

This system implements the SM2RAIN algorithm for detecting irrigation from satellite soil moisture observations. The methodology inverts the soil water balance equation to estimate total water input (precipitation plus irrigation) from observed changes in satellite-derived soil moisture. Irrigation is then isolated as the positive residual between the SM2RAIN-estimated total water input and a satellite rainfall reference.

The implementation uses a **4-parameter calibration model**:

| Symbol | Parameter | Units |
|--------|-----------|-------|
| Z\*   | Effective soil depth | mm |
| Ks    | Saturated hydraulic conductivity | mm/day |
| lambda | Shape parameter for the drainage function | dimensionless |
| Kc    | Crop coefficient for potential evapotranspiration | dimensionless |

The **exponential filter time constant T** is a user-configured constant (not calibrated) that smooths near-surface soil moisture observations to approximate root-zone soil moisture. The default value is **T = 5 days**.

---

## Theoretical Background

### Soil Water Balance Equation

The SM2RAIN algorithm inverts the soil water balance:

```
P(t) = Z* * dS/dt + D(t) + ET(t)
```

where:
- `P(t)` = total water input (precipitation + irrigation) [mm/day]
- `Z*` = effective soil depth [mm]
- `dS/dt` = rate of change of relative soil moisture [1/day]
- `D(t)` = drainage [mm/day]
- `ET(t)` = actual evapotranspiration [mm/day]

### Drainage Term

Drainage is modelled by a nonlinear power law:

```
D(t) = Ks * S(t)^lambda
```

where `Ks` is saturated hydraulic conductivity [mm/day], `S(t)` is relative soil moisture [-], and `lambda` is the shape parameter [-].

### Evapotranspiration

Actual evapotranspiration is computed as:

```
ET(t) = Kc * PET(t) * S(t)
```

Potential evapotranspiration (PET) is estimated using the **Hargreaves-Samani method** with dynamic extraterrestrial radiation Ra computed following FAO Irrigation and Drainage Paper 56 (Allen et al., 1998):

```
PET(t) = 0.0023 * (Ta(t) + 17.8) * Ra(t)
```

where `Ta` is daily mean air temperature [degrees C] and `Ra` is date- and latitude-specific extraterrestrial radiation [mm/day], calculated as:

```
Ra = (24*60/pi) * Gsc * dr * [omega_s * sin(phi)*sin(delta) + cos(phi)*cos(delta)*sin(omega_s)] * 0.408
```

with `Gsc = 0.0820` MJ/(m^2 min), `dr` the inverse relative Earth-Sun distance, `delta` solar declination, `omega_s` sunset hour angle, and `phi` the station latitude in radians. This replaces the static Blaney-Criddle formulation used in earlier versions.

### Exponential Filter

Near-surface satellite soil moisture (representative of approximately 0-5 cm) is converted to a root-zone soil moisture proxy using an exponential filter (Wagner et al., 1999):

```
SWI(t) = SWI(t-1) + K * [SM(t) - SWI(t-1)]
K = 1 / (1 + T/dt)
```

where SWI is the Soil Water Index, SM is the surface soil moisture observation, `dt` is the time step (1 day), and `T` is the characteristic time length [days]. A larger `T` corresponds to a deeper, slower-responding soil layer.

### Soil Moisture Normalization

Raw volumetric soil moisture observations are normalized to a relative wetness index S in [0, 1]:

```
S(t) = (theta(t) - theta_min) / (theta_max - theta_min)
```

where `theta_min` and `theta_max` are derived from the **full available SMAP time series** at each grid point (not only the calibration sub-period), ensuring that S = 0 corresponds to the residual/wilting moisture and S = 1 corresponds to the saturation extreme.

### Irrigation Detection

Irrigation is isolated as the positive residual between SM2RAIN-estimated total water input and the GPM IMERG reference rainfall:

```
I(t) = max(0, P_SM2RAIN(t) - P_ref(t))
```

Events below a minimum threshold (default 2.5 mm/day) are discarded as noise.

---

## Calibration Methodology

### Overview

Rather than fitting a single global parameter set, the system applies a **class-based (per-class) calibration** strategy. Grid points are first grouped into classes by rainfall intensity regime; independent parameter sets are then optimised for each class. This accounts for systematic differences in the soil moisture-rainfall relationship across climatically distinct sub-regions.

### Step 1 - Rainfall Intensity Classification

Daily GPM IMERG precipitation time series at each SMAP grid point are summarised as long-term statistics (mean annual rainfall, wet-day frequency, percentile values). A **Gaussian Mixture Model (GMM)** is fitted to these statistics and grid points are assigned to one of **5 rainfall intensity classes**. The class boundaries are validated against an independent K-means / DBSCAN run to confirm cluster stability.

Classification output is stored in `calibration/SM_classes/SM_classes_ludhiana.csv`, with class labels 1 (lowest rainfall regime) through 5 (highest rainfall regime).

### Step 2 - Calibration Date Selection (Rainfall-Proximity Filter)

Calibration is performed only on SMAP observation dates that fall within a defined window around a confirmed rainfall event. This ensures that the optimizer fits SM2RAIN parameters under conditions where the soil moisture response is primarily driven by precipitation rather than irrigation, minimising parameter bias. The filtered date lists are stored per class in `calibration/calib_dates_rainfall_filtered/`.

### Step 3 - Per-Class Parameter Optimisation

For each of the 5 classes, parameters (Z\*, Ks, lambda, Kc) are optimised by minimising a scalar objective function against GPM IMERG daily rainfall.

**Optimizer**: `scipy.optimize.differential_evolution` - a global, population-based evolutionary algorithm that avoids entrapment in local minima (Storn and Price, 1997).

**Optimiser settings** (as used in the final calibration run):

| Setting | Value |
|---------|-------|
| Strategy | best1bin |
| Population size | 30 |
| Maximum iterations | 2500 |
| Tolerance (tol, atol) | 1e-3 |
| Mutation factor | (0.5, 1.5) |
| Recombination factor | 0.9 |
| Updating | deferred |
| Random seed | 83 |

**Calibration modes** (configurable):

| Mode | Description |
|------|-------------|
| DAILY | Each calibration date is treated as an independent daily observation |
| AGGREGATED | Soil moisture and rainfall are aggregated over 5-day windows centred on each rainfall event, reducing noise and improving signal-to-noise in the objective |

**Calibration strategies** (configurable):

| Strategy | Description |
|----------|-------------|
| per_point | Each grid point within a class is calibrated independently; the point with the best objective score is selected as the class representative |
| pooled | All grid points within a class are merged into a single dataset and a single optimisation is performed; the optimizer sees N_stations x N_events data points |

The `per_point` strategy was used for the final Ludhiana calibration.

### Step 4 - Best-Parameter Selection

The best parameter set for each class is the one that achieves the lowest loss value on the primary objective (KGE-based loss). Both RMSE and KGE are reported for each grid point and each class to enable comparison.

---

## Data Requirements

### 1. SMAP Soil Moisture (NetCDF)

**File:** `data/SMAP_data/SPL3SMP_E_Ludhiana_ROI.nc`

**Source:** NASA SMAP Level-3 Enhanced Product (SPL3SMP_E), EASE-Grid, approximately 9 km resolution.

**Variables required:**

| Variable | Description |
|----------|-------------|
| `time` | Seconds since J2000 epoch |
| `latitude` | Grid-point latitude |
| `longitude` | Grid-point longitude |
| `soil_moisture_am` | AM overpass volumetric soil moisture [m3/m3] |
| `soil_moisture_pm` | PM overpass volumetric soil moisture [m3/m3] |

Both AM and PM retrievals are averaged (`SM_SOURCE = 'BOTH'`) to maximise temporal coverage. Fill values (-9999) are excluded before averaging.

### 2. Temperature Data (NetCDF)

**File:** `data/temp_data/AgERA5_daily_mean_ludhiana.nc`

**Source:** Copernicus agERA5 reanalysis, daily 2-m mean air temperature.

**Variable:** `Temperature_Air_2m_Mean_24h` in Kelvin; converted internally to Celsius by subtracting 273.15.

### 3. Precipitation Reference (NetCDF)

**File:** `data/precipitation_data/GPM_IMERG_ludhiana_final_run.nc`

**Source:** GPM IMERG Final Run, daily accumulated precipitation [mm/day].

**Variable:** `precipitation`

Both temperature and precipitation grids are resampled to the SMAP EASE-Grid using nearest-neighbour interpolation via a k-d tree (`scipy.spatial.cKDTree`).

### 4. Rainfall Classification File (CSV)

**File:** `calibration/SM_classes/SM_classes_ludhiana.csv`

**Columns:** `latitude`, `longitude`, `rainfall_class` (integer 1-5)

### 5. Calibration Date Files (CSV, one per class)

**Directory:** `calibration/calib_dates_rainfall_filtered/`

**Files:** `class_1_filtered_dates.csv` ... `class_5_filtered_dates.csv`

**Columns:** `date` (YYYY-MM-DD), `event_date` (the triggering rainfall event date)

---

## File Structure

```
SM2RAIN-irrigation_Final/
|
|-- main.py                          # Core SM2RAIN algorithm and SM2RAINCalibrator class
|-- data_preprocessor.py             # Data loading utilities (NetCDF/CSV)
|-- gridded_sm2rain_runner.py         # Gridded analysis pipeline (Phase 1 + Phase 2)
|-- utils.py                         # Shared utility functions
|
|-- data/
|   |-- SMAP_data/
|   |   |-- SPL3SMP_E_Ludhiana_ROI.nc       # SMAP L3 Enhanced soil moisture
|   |-- temp_data/
|   |   |-- AgERA5_daily_mean_ludhiana.nc   # agERA5 daily mean temperature
|   |-- precipitation_data/
|       |-- GPM_IMERG_ludhiana_final_run.nc  # GPM IMERG daily precipitation
|
|-- calibration/
|   |-- 1_clustering_rainfall_IMERG.ipynb   # Rainfall classification (GMM)
|   |-- 2_SMAP_class_assigner.ipynb         # Assign SMAP grid points to classes
|   |-- 3_SMAP_calib_dates_per_class.ipynb  # Identify calibration date windows
|   |-- 4-1_SMAP_rainfall_filter.ipynb      # Rainfall-proximity date filter
|   |-- 5_per_class_optimizer.ipynb         # Interactive per-class calibration
|   |-- 5_per_class_optimizer.py            # Script version of the optimizer
|   |
|   |-- SM_classes/
|   |   |-- SM_classes_ludhiana.csv         # Grid-point class assignments
|   |
|   |-- calib_dates_rainfall_filtered/
|   |   |-- class_{1..5}_filtered_dates.csv # Rainfall-proximity filtered dates
|   |
|   |-- per_class_parameters_ludhiana_correct_temp/
|       |-- all_classes_best_parameters.csv          # Final best parameters per class
|       |-- all_classes_calibration_results.json     # Full calibration output (all points)
|       |-- class_{1..5}_calibration_results.json    # Per-class detailed results
|
|-- rainfall_classes/
|   |-- rainfall_classification_5_bins.csv   # 5-class GMM classification results
|   |-- rainfall_classification_5_bins_GMM.csv
|
|-- irrigation_output/                       # Daily and monthly irrigation estimates
|-- irrigation_output_new/
```

---

## Configuration

The following constants in `calibration/5_per_class_optimizer.py` control the calibration run:

```python
# Data paths
SMAP_FILE           = Path(r'data/SMAP_data/SPL3SMP_E_Ludhiana_ROI.nc')
CLASSIFICATION_FILE = Path(r'calibration/SM_classes/SM_classes_ludhiana.csv')
DATES_DIR           = Path(r'calibration/calib_dates_rainfall_filtered')
TEMP_FILE           = Path(r'data/temp_data/AgERA5_daily_mean_ludhiana.nc')
PRECIP_FILE         = Path(r'data/precipitation_data/GPM_IMERG_ludhiana_final_run.nc')
OUTPUT_DIR          = Path(r'calibration/per_class_parameters_ludhiana_correct_temp')

# SM2RAIN settings
T_EXPONENTIAL_FILTER  = 5.0       # Exponential filter time constant [days]
FILL_VALUE            = -9999.0   # SMAP missing-data flag
SM_SOURCE             = 'BOTH'    # 'AM', 'PM', or 'BOTH' (AM/PM average)
NUM_CLASSES           = 5         # Number of rainfall intensity classes

# Calibration settings
OBJECTIVE_FUNCTION     = 'KGE'   # Primary objective: 'KGE' or 'RMSE'
USE_AGGREGATED_WINDOWS = True     # True: 5-day aggregated; False: daily
POOLED_CALIBRATION     = False    # True: pooled; False: per-point
```

### Exponential Filter Time Constant (T)

| Soil Type | Recommended T [days] | Rationale |
|-----------|---------------------|-----------|
| Sandy | 1-2 | Fast drainage, shallow effective layer |
| Loamy | 3-4 | Moderate retention |
| Clay | 5-8 | Slow drainage, deep effective layer |
| Irrigated agriculture | 3-5 | Depends on irrigation method |

---

## Usage

### Running the Per-Class Calibration

```bash
python calibration/5_per_class_optimizer.py
```

Or open and step through the interactive notebook `calibration/5_per_class_optimizer.ipynb`.

### Running the Gridded Irrigation Detection Pipeline

```bash
python gridded_sm2rain_runner.py
```

Or via Python API:

```python
from gridded_sm2rain_runner import run_sm2rain_analysis

results = run_sm2rain_analysis(
    output_dir="irrigation_output_new",
    T_filter=5.0,        # Must match the value used during calibration
    max_workers=8,
    filter_ndvi=False
)
```

### Two-Phase Analysis Pipeline

**Phase 1 - Per-Class Calibration**

1. Load GPM IMERG and assign rainfall intensity classes via GMM clustering.
2. Filter SMAP observation dates to rainfall-proximity windows.
3. For each class, optimise (Z\*, Ks, lambda, Kc) by differential evolution against GPM IMERG.
4. Select the best-scoring grid point per class and record parameters.

**Phase 2 - Irrigation Detection**

1. Apply class-specific parameters to all grid points.
2. Reconstruct total water input using SM2RAIN for the full study period.
3. Subtract GPM IMERG reference rainfall; retain positive residuals as irrigation.
4. Aggregate to daily and monthly time series; write output files.

---

## Parameters and Bounds

### Calibrated Parameters

| Parameter | Symbol | Units | Optimisation Bounds | Description |
|-----------|--------|-------|---------------------|-------------|
| Effective soil depth | Z* | mm | [0.001, 100] | Depth of soil layer contributing to SM observations |
| Saturated hydraulic conductivity | Ks | mm/day | [0.001, 50] | Maximum drainage rate |
| Drainage shape parameter | lambda | - | [3.0, 15.0] | Controls nonlinearity of drainage term |
| Crop coefficient | Kc | - | [0.001, 25] | Scales PET to actual ET |

### Fixed Parameters

| Parameter | Symbol | Units | Default | Description |
|-----------|--------|-------|---------|-------------|
| Exponential filter constant | T | days | 5.0 | Controls root-zone SM estimation depth |
| Irrigation detection threshold | - | mm/day | 2.5 | Minimum detectable irrigation event |

---

## Performance Metrics

The calibration reports three complementary performance measures.

### Kling-Gupta Efficiency (KGE)

```
KGE = 1 - sqrt((r-1)^2 + (beta-1)^2 + (gamma-1)^2)
```

where `r` is the Pearson correlation coefficient, `beta = mean_sim / mean_obs` is the bias ratio, and `gamma = CV_sim / CV_obs` is the variability ratio (Gupta et al., 2009). A perfect simulation gives KGE = 1; values below -0.41 indicate that the mean of the observations is a better predictor than the model.

The optimizer minimises `(1 - KGE)` so that a perfect fit corresponds to a loss of zero. KGE is the **primary selection criterion** for the best parameter set.

### Root Mean Square Error (RMSE)

```
RMSE = sqrt( (1/N) * sum( (P_hat(t) - P_ref(t))^2 ) )
```

Units: mm/day. Reported alongside KGE for each grid point and class but not used as the primary selection criterion.

### Pearson Correlation Coefficient (r)

Measures linear association between simulated and observed rainfall, independent of bias. Reported as a diagnostic alongside KGE and RMSE; not used directly in the optimisation objective.

---

## Calibration Results

Best-parameter results for the Ludhiana study region (per-point strategy, KGE objective, 5-day aggregated windows, T = 5 days). Source: `calibration/per_class_parameters_ludhiana_correct_temp/all_classes_best_parameters.csv`.

| Class | Z* (mm) | Ks (mm/day) | lambda | Kc | Best KGE | Best RMSE (mm/day) | Mean KGE | Mean RMSE (mm/day) |
|-------|---------|-------------|--------|----|----------|--------------------|----------|--------------------|
| 1 | 47.13 | 14.34 | 4.16 | 0.00 | 0.387 | 13.93 | 0.294 | 12.85 |
| 2 | 61.24 | 16.86 | 4.83 | 0.00 | 0.377 | 13.96 | 0.306 | 14.94 |
| 3 | 22.68 | 20.59 | 4.76 | 0.00 | 0.367 | 15.69 | 0.263 | 18.45 |
| 4 | 55.52 | 36.38 | 5.91 | 0.00 | 0.461 | 18.61 | 0.321 | 16.65 |
| 5 | 153.52 | 59.73 | 15.00 | 0.00 | 0.498 | 17.18 | 0.294 | 19.68 |

**Notes:**
- "Best" refers to the single grid point within each class that achieved the highest KGE.
- "Mean" is the arithmetic mean across all 25 grid points per class.
- Kc = 0 across all classes indicates that the Hargreaves PET term does not improve the KGE objective relative to the SM-only water balance terms for this dataset.
- Class 5 (highest rainfall regime) achieves the highest KGE (0.498), consistent with larger soil moisture responses providing greater discriminating power.

---

## Output Files

### Calibration Outputs (`calibration/per_class_parameters_ludhiana_correct_temp/`)

| File | Description |
|------|-------------|
| `all_classes_best_parameters.csv` | Best parameter set per class with KGE, RMSE, and Pearson r |
| `all_classes_calibration_results.json` | Complete calibration results for all classes combined |
| `class_{1..5}_calibration_results.json` | Per-class results for all grid points: parameters, KGE, RMSE, Pearson r, SM normalisation bounds |

### Irrigation Detection Outputs (`irrigation_output/` or `irrigation_output_new/`)

| File | Description |
|------|-------------|
| `daily_irrigation_estimates/` | Daily irrigation time series per grid point |
| `monthly_irrigation_estimates/` | Monthly aggregated irrigation estimates |
| `figures/` | Diagnostic plots |

### Column Descriptions for Monthly Output CSV

| Column | Description |
|--------|-------------|
| `grid_id` | Grid point identifier |
| `latitude`, `longitude` | Coordinates |
| `year`, `month`, `year_month` | Time identifiers |
| `irrigation_volume_mm` | Monthly irrigation total [mm] |
| `irrigation_events` | Number of days with detected irrigation |
| `irrigation_frequency` | Fraction of days with detected irrigation |
| `total_water_input_mm` | SM2RAIN-estimated total water input [mm] |
| `reference_rainfall_mm` | GPM IMERG reference precipitation [mm] |
| `avg_soil_moisture` | Monthly mean relative soil moisture [-] |
| `avg_temperature_c` | Monthly mean air temperature [degrees C] |

---

## Troubleshooting

### "No class data found for class N"

The calibration date files in `calib_dates_rainfall_filtered/` may be missing or named differently. The optimizer tries both `class_{i}_filtered_dates.csv` and `class_{i}_calib_dates.csv` (in that order). Verify that `4-1_SMAP_rainfall_filter.ipynb` has been executed and that its output was written to the expected directory.

### "Insufficient data for calibration (< 20 points)"

Fewer than 20 valid overlapping records exist for a given grid point after rainfall-proximity filtering. Consider widening the rainfall window in the filter notebook, or verify that all three datasets share a common temporal extent.

### High RMSE or Low KGE After Calibration

- Confirm that temperature is in Celsius before being passed to the Hargreaves PET function. The data loader subtracts 273.15 K from agERA5 output automatically; verify that no double-conversion has occurred.
- Confirm that SMAP soil moisture fill values (-9999) have been fully excluded prior to normalization.
- Consider increasing `maxiter` or `popsize` in `differential_evolution` for a more thorough global search.

### Zero Irrigation Detected

- Verify that SM2RAIN-estimated rainfall exceeds GPM IMERG in at least some time steps.
- Confirm that class-specific parameters are being applied correctly in Phase 2.
- Check the irrigation threshold (default 2.5 mm/day); lowering it will increase detected event counts.

### Grid Alignment Warnings

GPM IMERG and agERA5 have different native grids. Both are resampled to the SMAP EASE-Grid using nearest-neighbour matching with a coordinate tolerance of 0.001 degrees. If warnings about large resampling distances appear, verify that the input files cover the correct spatial extent.

---

## References

1. Brocca, L., Tarpanelli, A., Filippucci, P., Dorigo, W., Zaussinger, F., Gruber, A., and Fernandez-Prieto, D. (2018). How much water is used for irrigation? A new approach exploiting coarse resolution satellite soil moisture products. *International Journal of Applied Earth Observation and Geoinformation*, 73, 752-766. https://doi.org/10.1016/j.jag.2018.08.023

2. Brocca, L., Moramarco, T., Melone, F., and Wagner, W. (2013). A new method for rainfall estimation through soil moisture observations. *Geophysical Research Letters*, 40, 853-858. https://doi.org/10.1002/grl.50173

3. Wagner, W., Lemoine, G., and Rott, H. (1999). A method for estimating soil moisture from ERS scatterometer and soil data. *Remote Sensing of Environment*, 70(2), 191-207. https://doi.org/10.1016/S0034-4257(99)00036-X

4. Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling. *Journal of Hydrology*, 377(1-2), 80-91. https://doi.org/10.1016/j.jhydrol.2009.08.003

5. Allen, R. G., Pereira, L. S., Raes, D., and Smith, M. (1998). *Crop Evapotranspiration - Guidelines for Computing Crop Water Requirements*. FAO Irrigation and Drainage Paper 56. Food and Agriculture Organization of the United Nations, Rome.

6. Hargreaves, G. H., and Samani, Z. A. (1985). Reference crop evapotranspiration from temperature. *Applied Engineering in Agriculture*, 1(2), 96-99. https://doi.org/10.13031/2013.26773

7. Storn, R., and Price, K. (1997). Differential Evolution - A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces. *Journal of Global Optimization*, 11, 341-359. https://doi.org/10.1023/A:1008202821328

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 3.0 | 2025 | Class-based calibration (5-class GMM), Hargreaves-Samani PET with FAO-56 Ra, differential evolution optimizer, KGE primary objective, rainfall-proximity date filter, global SM normalization, 5-day aggregated calibration windows |
| 2.0 | 2024 | 4-parameter model (Z*, Ks, lambda, Kc), NetCDF/CSV data loading, Blaney-Criddle PET |
| 1.0 | 2023 | Original implementation with CSV data files |

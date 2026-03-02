# SM2RAIN Irrigation Detection System - Comprehensive Guide

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Background](#theoretical-background)
3. [Data Requirements](#data-requirements)
4. [File Structure](#file-structure)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Parameters](#parameters)
8. [Output Files](#output-files)
9. [Troubleshooting](#troubleshooting)
10. [References](#references)

---

## Overview

This system implements the SM2RAIN algorithm for detecting irrigation from satellite soil moisture data. The methodology is based on inverting the soil water balance equation to estimate total water input (precipitation + irrigation) from observed changes in soil moisture.

The implementation uses a **4-parameter calibration model**:
- **Z\*** (Zstar): Effective soil depth [mm]
- **Ks**: Saturated hydraulic conductivity [mm/day]
- **lambda**: Shape parameter for the drainage function [-]
- **Kc**: Crop coefficient for evapotranspiration [-]

Additionally, the **exponential filter time constant T** is a configurable constant (not calibrated) that controls the smoothing of near-surface soil moisture to estimate root-zone soil moisture.

---

## Theoretical Background

### Water Balance Equation

The SM2RAIN algorithm is based on the soil water balance equation:

```
P(t) = Z* * dS/dt + D(t) + ET(t)
```

Where:
- `P(t)` = Total water input (precipitation + irrigation) [mm/day]
- `Z*` = Effective soil depth [mm]
- `dS/dt` = Rate of change of relative soil moisture [-/day]
- `D(t)` = Drainage term [mm/day]
- `ET(t)` = Actual evapotranspiration [mm/day]

### Drainage Term

The drainage is modeled using a non-linear function:

```
D(t) = Ks * S(t)^lambda
```

Where:
- `Ks` = Saturated hydraulic conductivity [mm/day]
- `S(t)` = Relative soil moisture at time t [-]
- `lambda` = Shape parameter (typically 1-3) [-]

### Evapotranspiration

Actual evapotranspiration is computed as:

```
ET(t) = Kc * ETpot(t) * S(t)
```

Where:
- `Kc` = Crop coefficient [-]
- `ETpot(t)` = Potential evapotranspiration [mm/day]
- `S(t)` = Relative soil moisture [-]

Potential evapotranspiration is estimated using the Blaney-Criddle method:

```
ETpot = p * (0.46 * Ta + 8.13)
```

Where:
- `p` = Mean daily percentage of annual daytime hours (latitude-dependent)
- `Ta` = Mean daily air temperature [degrees C]

### Exponential Filter

Near-surface soil moisture from satellites (typically 0-5 cm) is converted to root-zone soil moisture (0-100 cm) using an exponential filter:

```
SWI(t) = SWI(t-1) + K * (SM(t) - SWI(t-1))
```

Where:
- `SWI` = Soil Water Index (root-zone soil moisture estimate)
- `SM` = Surface soil moisture observation
- `K` = Gain factor: `K = 1 / (1 + T/dt)` with `T` being the characteristic time length

The parameter `T` (in days) controls the depth of the soil layer being represented:
- Smaller T (1-2 days): Shallower soil layer, faster response
- Larger T (5-10 days): Deeper soil layer, slower response

### Irrigation Detection

Irrigation is detected as the residual between SM2RAIN-estimated total water input and reference rainfall:

```
Irrigation(t) = max(0, P_sm2rain(t) - P_reference(t))
```

Events below a threshold (default 2.5 mm/day) are filtered out.

---

## Data Requirements

### Input Files

The system requires three main data sources:

#### 1. Temperature Data (NetCDF)

**File path:** `G:\sm2rain-irrigation\agERA5_ROI_main\combined\agERA5_24_hour_mean_combined.nc`

**Required variables:**
- `time`: Time coordinate (datetime)
- `lat`: Latitude coordinate
- `lon`: Longitude coordinate
- `Temperature_Air_2m_Mean_24h`: 24-hour mean air temperature [K]

**Source:** ERA5-Land reanalysis (agERA5)

#### 2. Precipitation Data (NetCDF)

**File path:** `G:\sm2rain-irrigation\GPM_IMERG_Daily_ROI_main.nc`

**Required variables:**
- `time`: Time coordinate (datetime)
- `lat`: Latitude coordinate
- `lon`: Longitude coordinate
- `precipitation`: Daily precipitation [mm/day]

**Source:** GPM IMERG satellite precipitation

#### 3. Soil Moisture Data (CSV)

**File path:** `G:\sm2rain-irrigation\data\SMAP_data\SMAP_SPL3SMP_E_30N-31N_75E-76E_2017-2021.csv`

**Required columns:**
- `Date`: Date string (YYYY-MM-DD format)
- `Time`: Time string (optional)
- `Latitude`: Latitude coordinate
- `Longitude`: Longitude coordinate
- `Soil_Moisture`: Volumetric soil moisture [m3/m3]

**Source:** SMAP L3 Enhanced soil moisture product

#### 4. NDVI Dates File (Text)

**File path:** `G:\sm2rain-irrigation\ndvi_dates_mean.txt`

**Format:** One date per line (YYYY-MM-DD)

**Purpose:** Calibration is performed only on dates when NDVI observations are available, ensuring consistency with vegetation conditions.

---

## File Structure

```
sm2rain-irrigation/
|
|-- main.py                      # Core SM2RAIN algorithm (4-parameter model)
|-- data_preprocessor.py         # Data loading from NetCDF and CSV files
|-- gridded_sm2rain_runner.py    # Gridded analysis pipeline
|-- utils.py                     # Utility functions
|
|-- ndvi_dates_mean.txt          # NDVI observation dates for calibration
|
|-- agERA5_ROI_main/
|   |-- combined/
|       |-- agERA5_24_hour_mean_combined.nc  # Temperature data
|
|-- data/
|   |-- SMAP_data/
|       |-- SMAP_SPL3SMP_E_30N-31N_75E-76E_2017-2021.csv  # Soil moisture
|
|-- GPM_IMERG_Daily_ROI_main.nc  # Precipitation data
|
|-- results/
    |-- sm2rain_4param_results/  # Output directory
```

---

## Configuration

### Key Configuration Constants

Edit these values in `gridded_sm2rain_runner.py`:

```python
# File paths
TEMP_PATH = r"G:\sm2rain-irrigation\agERA5_ROI_main\combined\agERA5_24_hour_mean_combined.nc"
PRECIP_PATH = r"G:\sm2rain-irrigation\GPM_IMERG_Daily_ROI_main.nc"
SOIL_MOISTURE_PATH = r"G:\sm2rain-irrigation\data\SMAP_data\SMAP_SPL3SMP_E_30N-31N_75E-76E_2017-2021.csv"
NDVI_DATES_FILE = r"G:\sm2rain-irrigation\ndvi_dates_mean.txt"

# Exponential Filter Time Constant
# Typical values: 1-10 days depending on soil type
# This is NOT calibrated - it is a fixed constant
T_EXPONENTIAL_FILTER = 3.0  # days

# Region of Interest
ROI = {
    'min_lat': 30.0,
    'max_lat': 31.0,
    'min_lon': 75.0,
    'max_lon': 76.0
}
```

### Changing the Exponential Filter Time Constant (T)

The `T` parameter can be modified in two ways:

1. **Edit the constant in `gridded_sm2rain_runner.py`:**
   ```python
   T_EXPONENTIAL_FILTER = 5.0  # Change from default 3.0 to 5.0 days
   ```

2. **Pass as argument when calling functions:**
   ```python
   results = run_sm2rain_analysis(
       output_dir="path/to/output",
       T_filter=5.0,  # Override default
       max_workers=8,
       filter_ndvi=True
   )
   ```

### Recommended T Values by Soil Type

| Soil Type | Recommended T (days) | Rationale |
|-----------|---------------------|-----------|
| Sandy | 1-2 | Fast drainage, shallow effective layer |
| Loamy | 3-4 | Moderate retention |
| Clay | 5-8 | Slow drainage, deep effective layer |
| Irrigated agriculture | 3-5 | Depends on irrigation method |

---

## Usage

### Basic Usage

```python
from gridded_sm2rain_runner import run_sm2rain_analysis

# Run with default settings
results = run_sm2rain_analysis(
    output_dir="results/sm2rain_output",
    T_filter=3.0,       # Exponential filter constant
    max_workers=8,      # Parallel processing workers
    filter_ndvi=True    # Only calibrate on NDVI dates
)
```

### Command Line Execution

```bash
python gridded_sm2rain_runner.py
```

### Two-Phase Analysis

The analysis proceeds in two phases:

**Phase 1: Global Calibration**
- Calibrates the 4 parameters (Z*, Ks, lambda, Kc) for each grid point
- Selects the best parameters (lowest RMSE) as global parameters
- Uses fixed T value for exponential filtering

**Phase 2: Irrigation Detection**
- Applies global parameters to all grid points
- Estimates total water input using SM2RAIN
- Computes irrigation as residual (total water - reference rainfall)
- Generates daily and monthly irrigation estimates

---

## Parameters

### Calibrated Parameters (4-Parameter Model)

| Parameter | Symbol | Units | Typical Range | Description |
|-----------|--------|-------|---------------|-------------|
| Effective soil depth | Z* | mm | 10-200 | Depth of soil layer contributing to SM observations |
| Saturated hydraulic conductivity | Ks | mm/day | 0.1-50 | Maximum drainage rate |
| Shape parameter | lambda | - | 0.1-5 | Controls non-linearity of drainage |
| Crop coefficient | Kc | - | 0.5-2.0 | Scales potential to actual ET |

### Fixed Parameters

| Parameter | Symbol | Units | Default | Description |
|-----------|--------|-------|---------|-------------|
| Exponential filter constant | T | days | 3.0 | Controls root-zone SM estimation |
| Irrigation threshold | - | mm/day | 2.5 | Minimum irrigation event size |

### Parameter Bounds for Calibration

The optimization uses the following bounds:

```python
bounds = {
    'Zstar': (10.0, 200.0),   # mm
    'Ks': (0.1, 50.0),        # mm/day
    'lam': (0.1, 5.0),        # dimensionless
    'Kc': (0.5, 2.0)          # dimensionless
}
```

---

## Output Files

The analysis generates several output files in the specified output directory:

### Phase 1 Outputs (Calibration)

| File | Description |
|------|-------------|
| `phase1_calibration_results.json` | Detailed results for each grid point |
| `phase1_calibration_summary.json` | Summary statistics of calibration |
| `phase1_calibration_summary.txt` | Human-readable summary |

### Phase 2 Outputs (Irrigation Detection)

| File | Description |
|------|-------------|
| `phase2_irrigation_results.json` | Detailed irrigation results per grid |
| `phase2_irrigation_summary.json` | Summary of irrigation detection |
| `phase2_irrigation_summary.txt` | Human-readable summary |
| `monthly_irrigation_all_grids.csv` | Monthly irrigation estimates for all grids |
| `daily_time_series_all_grids.csv` | Daily time series data |
| `irrigation_heatmap_data.csv` | Spatial irrigation data for mapping |

### Final Results

| File | Description |
|------|-------------|
| `final_results.json` | Complete analysis results and configuration |

### Output CSV Column Descriptions

**monthly_irrigation_all_grids.csv:**
- `grid_id`: Grid point identifier
- `latitude`, `longitude`: Coordinates
- `year`, `month`, `year_month`: Time identifiers
- `irrigation_volume_mm`: Monthly irrigation total [mm]
- `irrigation_events`: Number of irrigation events
- `irrigation_frequency`: Fraction of days with irrigation
- `total_water_input_mm`: SM2RAIN estimated water input [mm]
- `reference_rainfall_mm`: Reference precipitation [mm]
- `avg_soil_moisture`: Monthly mean relative soil moisture
- `avg_temperature_c`: Monthly mean temperature [C]

---

## Troubleshooting

### Common Issues

#### 1. "No common grid points found"

**Cause:** Grid alignment mismatch between datasets (GPM and ERA5 have different grid centers).

**Solution:** The system uses a diagonal tolerance of approximately 0.07 degrees to handle the ~0.05 degree offset between GPM (centers at .05, .15, ...) and ERA5 (centers at .0, .1, .2, ...) grids.

#### 2. "Insufficient data for calibration"

**Cause:** Less than 20 overlapping dates between all three datasets.

**Solution:** 
- Check that NDVI dates overlap with the data period
- Verify that all input files cover the same time period
- Consider setting `filter_ndvi=False` to use all available dates

#### 3. High RMSE values

**Cause:** Poor fit between SM2RAIN estimates and reference rainfall.

**Possible solutions:**
- Adjust the T parameter for your soil type
- Check for data quality issues (missing values, outliers)
- Verify that soil moisture data is in correct units (volumetric, 0-0.6 range)

#### 4. Zero irrigation detected

**Cause:** SM2RAIN estimates consistently lower than reference rainfall.

**Possible solutions:**
- Check if the region actually has irrigation
- Verify temperature data is in correct units (Kelvin vs Celsius)
- Adjust the irrigation threshold parameter

### Data Quality Checks

Before running the analysis, verify:

1. **Soil moisture range:** Should be 0.0-0.6 for volumetric measurements
2. **Temperature units:** Should be in Kelvin for ERA5 data
3. **Precipitation units:** Should be in mm/day
4. **Temporal overlap:** All datasets should cover the same time period
5. **Spatial coverage:** All datasets should cover the region of interest

---

## References

1. Brocca, L., Tarpanelli, A., Filippucci, P., Dorigo, W., Zaussinger, F., Gruber, A., and Fernandez-Prieto, D. (2018). How much water is used for irrigation? A new approach exploiting coarse resolution satellite soil moisture products. International Journal of Applied Earth Observation and Geoinformation, 73, 752-766.

2. Brocca, L., Moramarco, T., Melone, F., and Wagner, W. (2013). A new method for rainfall estimation through soil moisture observations. Geophysical Research Letters, 40, 853-858.

3. Wagner, W., Lemoine, G., and Rott, H. (1999). A method for estimating soil moisture from ERS scatterometer and soil data. Remote Sensing of Environment, 70(2), 191-207.

4. Blaney, H.F., and Criddle, W.D. (1950). Determining water requirements in irrigated areas from climatological and irrigation data. USDA Soil Conservation Service Technical Paper 96.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2024 | 4-parameter model (Z*, Ks, lambda, Kc), NetCDF/CSV data loading |
| 1.0 | 2023 | Original implementation with CSV data files |

---

## Contact

For questions about this implementation, refer to the code documentation or the referenced publications for theoretical details.

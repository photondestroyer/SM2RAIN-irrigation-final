# SM2RAIN Irrigation Detection algorithm


## Overview

This project implements the SM2RAIN algorithm for detecting irrigation from satellite soil moisture observations. The methodology inverts the soil water balance equation to estimate total water input (precipitation plus irrigation) from observed changes in satellite-derived soil moisture. Irrigation is then isolated as the positive residual between the SM2RAIN-estimated total water input and a satellite rainfall reference.

The SM2RAIN algorithm is implemented for 11 years: 2015 - 2025.

The implementation uses a **4-parameter calibration model**:

| Symbol | Parameter | Units |
|--------|-----------|-------|
| Z\*   | Effective soil depth | mm |
| Ks    | Saturated hydraulic conductivity | mm/day |
| lambda | Shape parameter for the drainage function | dimensionless |
| Kc    | Crop coefficient for potential evapotranspiration | dimensionless |

The **exponential filter time constant T** is a user configured (not calibrated) value that smooths near-surface soil moisture observations to approximate root-zone soil moisture. The default value is **T = 5 days**.

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

with `Gsc = 0.0820` MJ/(m^2 min), `dr` the inverse relative Earth-Sun distance, `delta` solar declination, `omega_s` sunset hour angle, and `phi` the station latitude in radians.

### Irrigation Detection

Irrigation is isolated as the positive residual between SM2RAIN-estimated total water input and the GPM IMERG reference rainfall:

```
I(t) = max(0, P_SM2RAIN(t) - P_ref(t))
```

Events below a minimum threshold (default 2.5 mm/day) are discarded as noise.

---
## Data products used
- Copernicus AgERA5- for day time mean temperature
- NASA IMERG Final Run V7 - for precipitation values
- NASA SMAP - SPL3SMP_E - for SM values 

## Calibration Methodology

### Overview

Rather than fitting a single global parameter set, the calibration algorithm applies a **class-based (per-class) calibration** strategy. Grid points are first grouped into classes by rainfall intensity regime; independent parameter sets are then optimised for each class. This accounts for systematic differences in the soil moisture-rainfall relationship across climatically distinct sub-regions.

### Step 1 - Rainfall Intensity Classification

Daily GPM IMERG precipitation time series at each SMAP grid point are summarised as long-term statistics (mean annual rainfall, wet-day frequency, percentile values). A **Gaussian Mixture Model (GMM)** is fitted to these statistics and grid points are assigned to one of **5 rainfall intensity classes**. The class boundaries are validated against an independent K-means / DBSCAN run to confirm cluster stability.


### Step 2 - Calibration Date Selection (Rainfall-Proximity Filter)

Calibration is performed only on SMAP observation dates that fall within a defined window around a confirmed rainfall event. This ensures that the optimizer fits SM2RAIN parameters under conditions where the soil moisture response is primarily driven by precipitation rather than irrigation, minimising parameter bias.

### Step 3 - Per-Class Parameter Optimisation

For each of the 5 classes, parameters (Z\*, Ks, lambda, Kc) are optimised by minimising a scalar objective function against GPM IMERG daily rainfall.

**Optimizer**: `scipy.optimize.differential_evolution` - a global, population-based evolutionary algorithm that avoids entrapment in local minima (Storn and Price, 1997).


### Step 4 - Best-Parameter Selection

The best parameter set for each class is the one that achieves the lowest loss value on the primary objective (KGE-based loss). Both RMSE and KGE are reported for each grid point and each class to enable comparison.


## Parameters and Bounds

### Calibrated Parameters

| Parameter | Symbol | Units | Optimisation Bounds | Description |
|-----------|--------|-------|---------------------|-------------|
| Effective soil depth | Z* | mm | [0.001, 100] | Depth of soil layer contributing to SM observations |
| Saturated hydraulic conductivity | Ks | mm/day | [0.001, 50] | Maximum drainage rate |
| Drainage shape parameter | lambda | - | [3.0, 15.0] | Controls nonlinearity of drainage term |
| Crop coefficient | Kc | - | [0.001, 25] | Scales PET to actual ET |

---


## Calibration Results

Best-parameter results for the Ludhiana study region (per-point strategy, KGE objective, 5-day aggregated windows, T = 5 days).

| Class | Z* (mm) | Ks (mm/day) | lambda | Kc | Best KGE | Best RMSE (mm/day) | Mean KGE | Mean RMSE (mm/day) |
|-------|---------|-------------|--------|----|----------|--------------------|----------|--------------------|
| 1 | 47.13 | 14.34 | 4.16 | 0.00 | 0.387 | 13.93 | 0.294 | 12.85 |
| 2 | 61.24 | 16.86 | 4.83 | 0.00 | 0.377 | 13.96 | 0.306 | 14.94 |
| 3 | 22.68 | 20.59 | 4.76 | 0.00 | 0.367 | 15.69 | 0.263 | 18.45 |
| 4 | 55.52 | 36.38 | 5.91 | 0.00 | 0.461 | 18.61 | 0.321 | 16.65 |
| 5 | 153.52 | 59.73 | 15.00 | 0.00 | 0.498 | 17.18 | 0.294 | 19.68 |

**Notes:**
- "Best" refers to the single grid point within each class that achieved the highest KGE.
- "Mean" is the arithmetic mean across all ~25 grid points per class.
- Kc = 0 across all classes indicates that the Hargreaves PET term does not improve the KGE objective relative to the SM-only water balance terms for this dataset.
- Class 5 (highest rainfall regime) achieves the highest KGE (0.498), consistent with larger soil moisture responses providing greater discriminating power.



## References

1. Brocca, L., Tarpanelli, A., Filippucci, P., Dorigo, W., Zaussinger, F., Gruber, A., and Fernandez-Prieto, D. (2018). How much water is used for irrigation? A new approach exploiting coarse resolution satellite soil moisture products. *International Journal of Applied Earth Observation and Geoinformation*, 73, 752-766. https://doi.org/10.1016/j.jag.2018.08.023

2. Brocca, L., Moramarco, T., Melone, F., and Wagner, W. (2013). A new method for rainfall estimation through soil moisture observations. *Geophysical Research Letters*, 40, 853-858. https://doi.org/10.1002/grl.50173

3. Wagner, W., Lemoine, G., and Rott, H. (1999). A method for estimating soil moisture from ERS scatterometer and soil data. *Remote Sensing of Environment*, 70(2), 191-207. https://doi.org/10.1016/S0034-4257(99)00036-X

4. Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling. *Journal of Hydrology*, 377(1-2), 80-91. https://doi.org/10.1016/j.jhydrol.2009.08.003

5. Allen, R. G., Pereira, L. S., Raes, D., and Smith, M. (1998). *Crop Evapotranspiration - Guidelines for Computing Crop Water Requirements*. FAO Irrigation and Drainage Paper 56. Food and Agriculture Organization of the United Nations, Rome.

6. Hargreaves, G. H., and Samani, Z. A. (1985). Reference crop evapotranspiration from temperature. *Applied Engineering in Agriculture*, 1(2), 96-99. https://doi.org/10.13031/2013.26773

7. Storn, R., and Price, K. (1997). Differential Evolution - A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces. *Journal of Global Optimization*, 11, 341-359. https://doi.org/10.1023/A:1008202821328

---


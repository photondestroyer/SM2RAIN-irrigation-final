import numpy as np
import pandas as pd
import xarray as xr
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.stats import pearsonr
import datetime
import warnings
from typing import Dict, Tuple, Optional, Union, List
import logging
from pathlib import Path
import seaborn as sns
warnings.filterwarnings('ignore')
from utils import detect_irrigation_events

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ROI = 30,75, 31,76
# TIME = 2020-01-01 00:00:00 start
# TIME = 2020-01-31 23:59:59 till

def load_data(soil_moisture_path, rainfall_path, temp_path, var_names=None) -> dict:
    """
    Read the time series you need (satellite soil moisture, rainfall reference, air temperature).
    Following the SM2RAIN paper methodology for data loading and preprocessing.
    
    Args:+
        soil_moisture_path: Path to soil moisture data file (NetCDF, CSV, or TXT)
        rainfall_path: Path to rainfall reference data file
        temp_path: Path to temperature data file
        var_names: Dictionary with variable names for each dataset
    
    Returns:
        dict with keys: 'sm' (pd.Series), 'rain' (pd.Series), 'Ta' (pd.Series)
        all series are daily and aligned (same index).
    """
    data = {}
    
    # Load soil moisture data
    if soil_moisture_path and os.path.exists(soil_moisture_path):
        if soil_moisture_path.endswith('.nc') or soil_moisture_path.endswith('.nc4'):
            try:
                sm_ds = xr.open_dataset(soil_moisture_path)
                if var_names and 'sm' in var_names:
                    sm_var = var_names['sm']
                else:
                    # Common variable names for soil moisture
                    possible_sm_vars = ['sm', 'soil_moisture', 'SM', 'swvl1', 'theta', 'volumetric_soil_water']
                    sm_var = next((var for var in possible_sm_vars if var in sm_ds.variables), None)
                
                if sm_var:
                    sm_data = sm_ds[sm_var]
                    # Handle multi-dimensional data (select spatial point or average)
                    if len(sm_data.dims) > 1:
                        # If spatial dimensions exist, take the first point or average
                        for dim in sm_data.dims:
                            if dim not in ['time', 'date']:
                                sm_data = sm_data.isel({dim: 0})
                    
                    sm_data = sm_data.to_pandas()
                    if isinstance(sm_data, pd.DataFrame):
                        sm_data = sm_data.iloc[:, 0]  # Take first column if DataFrame
                    data['sm'] = sm_data
                sm_ds.close()
            except Exception as e:
                logger.error(f"Error loading soil moisture NetCDF file: {e}")
        elif soil_moisture_path.endswith('.csv') or soil_moisture_path.endswith('.txt'):
            try:
                sm_data = pd.read_csv(soil_moisture_path, index_col=0, parse_dates=True).squeeze()
                data['sm'] = sm_data
            except Exception as e:
                logger.error(f"Error loading soil moisture CSV/TXT file: {e}")
    
    # Load rainfall data
    if rainfall_path and os.path.exists(rainfall_path):
        if rainfall_path.endswith('.nc') or rainfall_path.endswith('.nc4'):
            try:
                rain_ds = xr.open_dataset(rainfall_path)
                if var_names and 'rain' in var_names:
                    rain_var = var_names['rain']
                else:
                    possible_rain_vars = ['precipitation', 'rain', 'tp', 'precip', 'pr']
                    rain_var = next((var for var in possible_rain_vars if var in rain_ds.variables), None)
                
                if rain_var:
                    rain_data = rain_ds[rain_var]
                    # Handle multi-dimensional data (select spatial point or average)
                    if len(rain_data.dims) > 1:
                        # If spatial dimensions exist, take the first point or average
                        for dim in rain_data.dims:
                            if dim not in ['time', 'date']:
                                rain_data = rain_data.isel({dim: 0})
                    
                    rain_data = rain_data.to_pandas()
                    if isinstance(rain_data, pd.DataFrame):
                        rain_data = rain_data.iloc[:, 0]
                    data['rain'] = rain_data
                rain_ds.close()
            except Exception as e:
                logger.error(f"Error loading rainfall NetCDF file: {e}")
        elif rainfall_path.endswith('.csv') or rainfall_path.endswith('.txt'):
            try:
                rain_data = pd.read_csv(rainfall_path, index_col=0, parse_dates=True).squeeze()
                data['rain'] = rain_data
            except Exception as e:
                logger.error(f"Error loading rainfall CSV/TXT file: {e}")
    
    # Load temperature data
    if temp_path and os.path.exists(temp_path):
        if temp_path.endswith('.nc') or temp_path.endswith('.nc4'):
            try:
                temp_ds = xr.open_dataset(temp_path)
                if var_names and 'temp' in var_names:
                    temp_var = var_names['temp']
                else:
                    possible_temp_vars = ['temperature', 'temp', 't2m', 'Ta', 'air_temperature']
                    temp_var = next((var for var in possible_temp_vars if var in temp_ds.variables), None)
                
                if temp_var:
                    temp_data = temp_ds[temp_var]
                    # Handle multi-dimensional data (select spatial point or average)
                    if len(temp_data.dims) > 1:
                        # If spatial dimensions exist, take the first point or average
                        for dim in temp_data.dims:
                            if dim not in ['time', 'date']:
                                temp_data = temp_data.isel({dim: 0})
                    
                    temp_data = temp_data.to_pandas()
                    if isinstance(temp_data, pd.DataFrame):
                        temp_data = temp_data.iloc[:, 0]
                    data['Ta'] = temp_data
                temp_ds.close()
            except Exception as e:
                logger.error(f"Error loading temperature NetCDF file: {e}")
        elif temp_path.endswith('.csv') or temp_path.endswith('.txt'):
            try:
                temp_data = pd.read_csv(temp_path, index_col=0, parse_dates=True).squeeze()
                data['Ta'] = temp_data
            except Exception as e:
                logger.error(f"Error loading temperature CSV/TXT file: {e}")
    
    # Align all series to common time index
    if len(data) > 1:
        common_index = None
        for key, series in data.items():
            if common_index is None:
                common_index = series.index
            else:
                common_index = common_index.intersection(series.index)
        
        for key in data.keys():
            data[key] = data[key].reindex(common_index)
    
    # Log data loading summary
    logger.info(f"Data loaded successfully:")
    for key, series in data.items():
        logger.info(f"  {key}: {len(series)} records from {series.index.min()} to {series.index.max()}")
    
    # Validate data quality
    for key, series in data.items():
        missing_pct = series.isna().sum() / len(series) * 100
        if missing_pct > 50:
            logger.warning(f"{key}: {missing_pct:.1f}% missing data")
        elif missing_pct > 10:
            logger.info(f"{key}: {missing_pct:.1f}% missing data")
    
    return data


def preprocess_soil_moisture(sm_series: pd.Series, tau_days: float=3.0) -> pd.Series:
    """
    filter the soil moisture to remove high-frequency noise but keep irrigation signal (paper uses Wagner et al. exponential filter)
    """
    # exponential low-pass filter: S_filtered[t] = alpha*S[t] + (1-alpha)*S_filtered[t-1]
    # where alpha = 1/(1+tau) or similar depending on discretization.
    # use discrete exponential smoothing with alpha = dt/(tau+dt) where dt=1 day.
    dt = 1.0  # 1 day
    alpha = dt / (tau_days + dt)
    
    # Initialize filtered series
    sm_filtered = sm_series.copy()
    
    # Apply exponential filter
    for i in range(1, len(sm_series)):
        if pd.notna(sm_series.iloc[i]) and pd.notna(sm_filtered.iloc[i-1]):
            sm_filtered.iloc[i] = alpha * sm_series.iloc[i] + (1 - alpha) * sm_filtered.iloc[i-1]
    
    return sm_filtered

def compute_dSdt(S_series: pd.Series) -> pd.Series:
    """
    Compute time derivative of soil moisture dS/dt
    """
    dt = 1.0  # 1 day
    return S_series.diff() / dt


def normalize_soil_moisture(sm_series: pd.Series) -> Tuple[pd.Series, float, float]:
    """
    Normalize soil moisture to 0-1 range based on actual data min/max.
    
    This rescales the input soil moisture values to a normalized range [0, 1]
    using the minimum and maximum values in the data, ensuring consistent
    scaling for SM2RAIN calibration and irrigation detection.
    
    Parameters
    ----------
    sm_series : pd.Series
        Raw soil moisture time series
        
    Returns
    -------
    tuple
        (normalized_sm, sm_min, sm_max) where:
        - normalized_sm: pd.Series with values in [0, 1]
        - sm_min: minimum value used for normalization
        - sm_max: maximum value used for normalization
    """
    # Get valid (non-NaN) values for computing min/max
    valid_data = sm_series.dropna()
    
    if len(valid_data) == 0:
        logger.warning("No valid soil moisture data for normalization")
        return sm_series, 0.0, 1.0
    
    sm_min = float(valid_data.min())
    sm_max = float(valid_data.max())
    
    # Handle case where min equals max (constant values)
    if sm_max - sm_min < 1e-10:
        logger.warning(f"Constant soil moisture values detected (value={sm_min})")
        return pd.Series(0.5, index=sm_series.index), sm_min, sm_max
    
    # Normalize to 0-1 range
    normalized_sm = (sm_series - sm_min) / (sm_max - sm_min)
    
    # Clip to ensure values are strictly within [0, 1]
    normalized_sm = normalized_sm.clip(0, 1)
    
    logger.debug(f"Soil moisture normalized: original range [{sm_min:.4f}, {sm_max:.4f}] -> [0, 1]")
    
    return normalized_sm, sm_min, sm_max


def compute_relative_soil_moisture(theta_series, theta_min, theta_sat) -> pd.Series:
    """
    Compute relative soil moisture S according to Eq. (1) in the SM2RAIN paper:
    S = (theta - theta_min) / (theta_sat - theta_min)
    
    Note: For data-driven normalization, use normalize_soil_moisture() instead.
    """
    # Compute relative soil moisture according to Eq. (1) in the paper
    S = (theta_series - theta_min) / (theta_sat - theta_min)
    # Clip values between 0 and 1 to ensure physical consistency
    S = S.clip(0, 1)
    return S

def separate_irrigation_rainfall(total_water_input, reference_rainfall, method='threshold'):
    """
    Separate irrigation from total water input based on different methods
    
    Args:
        total_water_input: Total water input from SM2RAIN (rainfall + irrigation)
        reference_rainfall: Reference rainfall data
        method: Method for separation ('threshold', 'correlation', 'residual')
    
    Returns:
        Dictionary with separated irrigation and rainfall estimates
    """
    # Simple threshold method - assume irrigation when total > reference
    if method == 'threshold':
        estimated_irrigation = np.maximum(0, total_water_input - reference_rainfall)
        estimated_rainfall = np.minimum(total_water_input, reference_rainfall)
        
    elif method == 'residual':
        # Residual method - irrigation is the difference
        estimated_rainfall = reference_rainfall.copy()
        estimated_irrigation = np.maximum(0, total_water_input - reference_rainfall)
        
    elif method == 'correlation':
        # Correlation-based method (more sophisticated approach)
        # This would need additional meteorological data
        estimated_irrigation = np.maximum(0, total_water_input - reference_rainfall)
        estimated_rainfall = reference_rainfall.copy()
    
    return {
        'estimated_irrigation': pd.Series(estimated_irrigation, index=total_water_input.index),
        'estimated_rainfall': pd.Series(estimated_rainfall, index=total_water_input.index),
        'total_water_input': total_water_input
    }
def compute_ETpot(Ta_series: pd.Series, xi: float, Kc: float) -> pd.Series:
    """
    Compute potential evapotranspiration according to Eq. (4) in the paper:
    ETpot(t) = Kc * xi * (-2 + 1.26*(0.46*Ta + 8.13))
    
    This is a simplified temperature-based ET formula used in the SM2RAIN algorithm.
    
    Args:
        Ta_series: Air temperature series [°C]
        xi: Parameter for ET calculation (calibrated parameter)
        Kc: Crop coefficient (default = 1.0 for reference conditions)
    
    Returns:
        ETpot series [mm/day]
    """
    # Eq. (4) from the paper
    ETpot = Kc * xi * (-2 + 1.26 * (0.46 * Ta_series + 8.13))
    
    # FIX: Ensure realistic ET values (typically 0-10 mm/day)
    ETpot = ETpot.clip(lower=0, upper=15)  # Reasonable ET bounds
    
    return ETpot


def drainage_term(S_series: pd.Series, Ks: float, lam: float) -> pd.Series:
    """
    Compute drainage term g(S) according to Eq. (2):
    g(S) = Ks * S^(3/2 + λ)
    
    Args:
        S_series: Relative soil moisture series [-]
        Ks: Saturated hydraulic conductivity [mm/day]
        lam: Shape parameter [-]
    
    Returns:
        Drainage term [mm/day]
    """
    exponent = 1.5 + lam  # 3/2 + λ
    
    # FIX: Handle edge cases and ensure realistic drainage
    S_clipped = S_series.clip(lower=0.01, upper=1.0)  # Avoid zero/negative values
    drainage = Ks * (S_clipped ** exponent)
    
    # Apply realistic upper bound for drainage
    drainage = drainage.clip(upper=50)  # Cap at 50 mm/day
    
    return drainage

def sm2rain_forward(S_series, dSdt_series, ETpot_series, params) -> pd.Series:
    """
    SM2RAIN forward model to compute total water input (rainfall + irrigation)
    According to Eq. (5) in the paper:
    r(t) + i(t) = Z* * dS/dt + g(S) + ETpot(t) * S(t)
    """
    # Compute drainage term g(S)
    g_S = drainage_term(S_series, params['Ks'], params['lam'])
    
    # Compute total water input according to Eq. (5)
    r_plus_i = params['Zstar'] * dSdt_series + g_S + ETpot_series * S_series
    
    # FIX: Ensure realistic values and proper scaling
    r_plus_i = r_plus_i.clip(lower=0)  # Remove negative values
    
    # Apply realistic upper bound (very high rainfall/irrigation events are rare)
    r_plus_i = r_plus_i.clip(upper=100)  # Cap at 100 mm/day
    
    return r_plus_i

def objective_calibrate(x, S, dSdt, ETpot, ref_rain):
    """
    Objective function for SM2RAIN calibration
    
    Args:
        x: Parameter vector [Zstar, Ks, lam, xi]
        S: Relative soil moisture series
        dSdt: Time derivative of soil moisture
        ETpot: Potential evapotranspiration
        ref_rain: Reference rainfall data
    
    Returns:
        RMSE between estimated and reference rainfall
    """
    from scipy.stats import pearsonr
    
    # Unpack parameters
    Zstar, Ks, lam, xi = x
    
    # Create params dictionary
    params = {
        'Zstar': Zstar,
        'Ks': Ks,
        'lam': lam
    }
    
    # Estimate rainfall using SM2RAIN
    r_est_daily = sm2rain_forward(S, dSdt, ETpot, params)
    
    # Remove negative values (set to 0)
    r_est_daily = r_est_daily.clip(lower=0)
    
    # Convert to numpy arrays and remove NaN values
    valid_mask = ~(np.isnan(r_est_daily) | np.isnan(ref_rain))
    r_est_valid = r_est_daily[valid_mask]
    ref_rain_valid = ref_rain[valid_mask]
    
    if len(r_est_valid) == 0:
        return np.inf
    
    # Compute RMSE
    rmse = np.sqrt(np.mean((r_est_valid - ref_rain_valid) ** 2))
    
    return rmse

def calibrate_sm2rain(S_series, Ta_series, ref_rain, theta_min=0.0, theta_sat=0.5, 
                     tau_days=3.0, Kc=1.0):
    """
    Calibrate SM2RAIN parameters using optimization
    
    Args:
        S_series: Soil moisture time series
        Ta_series: Air temperature time series
        ref_rain: Reference rainfall data
        theta_min: Minimum soil moisture content
        theta_sat: Saturated soil moisture content
        tau_days: Time constant for exponential filter
        Kc: Crop coefficient
    
    Returns:
        Optimized parameters and results
    """
    from scipy.optimize import minimize
    
    # Preprocess soil moisture
    S_filtered = preprocess_soil_moisture(S_series, tau_days)
    
    # Compute relative soil moisture
    S_rel = compute_relative_soil_moisture(S_filtered, theta_min, theta_sat)
    
    # Compute time derivative
    dSdt = compute_dSdt(S_rel)
    
    # Initial parameter guess and bounds
    x0 = [50.0, 10.0, 0.5, 1.0]  # [Zstar, Ks, lam, xi]
    bounds = [(1.0, 950.0),   # Zstar: effective soil depth [mm]
              (0.001, 1000.0),   # Ks: saturated hydraulic conductivity [mm/day]
              (-10.0, 10.0),    # lam: shape parameter [-]
              (0.0001, 10.0)]     # xi: ET parameter [-]
    
    # Compute potential ET for initial guess
    ETpot = compute_ETpot(Ta_series, x0[3], Kc)
    
    # Define objective function wrapper
    def objective_wrapper(x):
        # Update ETpot with current xi parameter
        ETpot_current = compute_ETpot(Ta_series, x[3], Kc)
        return objective_calibrate(x, S_rel, dSdt, ETpot_current, ref_rain)
    
    # Optimize parameters
    result = minimize(objective_wrapper, x0, bounds=bounds, method='L-BFGS-B')
    
    # Extract optimized parameters
    Zstar_opt, Ks_opt, lam_opt, xi_opt = result.x
    
    # Compute final results with optimized parameters
    ETpot_opt = compute_ETpot(Ta_series, xi_opt, Kc)
    params_opt = {'Zstar': Zstar_opt, 'Ks': Ks_opt, 'lam': lam_opt}
    r_est = sm2rain_forward(S_rel, dSdt, ETpot_opt, params_opt)
    r_est = r_est.clip(lower=0)
    
    return {
        'parameters': {
            'Zstar': Zstar_opt,
            'Ks': Ks_opt,
            'lam': lam_opt,
            'xi': xi_opt,
            'Kc': Kc
        },
        'estimated_rainfall': r_est,
        'optimization_result': result,
        'processed_data': {
            'S_relative': S_rel,
            'dSdt': dSdt,
            'ETpot': ETpot_opt
        }
    }

def evaluate_performance(estimated_rain, reference_rain):
    """
    Evaluate SM2RAIN performance using various metrics
    
    Args:
        estimated_rain: Estimated rainfall series
        reference_rain: Reference rainfall series
    
    Returns:
        Dictionary with performance metrics
    """
    from scipy.stats import pearsonr
    
    # Remove NaN values
    valid_mask = ~(np.isnan(estimated_rain) | np.isnan(reference_rain))
    est_valid = estimated_rain[valid_mask]
    ref_valid = reference_rain[valid_mask]
    
    if len(est_valid) == 0:
        return {}
    
    # Compute metrics
    correlation, p_value = pearsonr(est_valid, ref_valid)
    rmse = np.sqrt(np.mean((est_valid - ref_valid) ** 2))
    mae = np.mean(np.abs(est_valid - ref_valid))
    bias = np.mean(est_valid - ref_valid)
    
    # Nash-Sutcliffe efficiency
    nse = 1 - np.sum((est_valid - ref_valid) ** 2) / np.sum((ref_valid - np.mean(ref_valid)) ** 2)
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'rmse': rmse,
        'mae': mae,
        'bias': bias,
        'nse': nse,
        'n_samples': len(est_valid)
    }

def aggregate_to_monthly(daily_series, method='sum'):
    """
    Aggregate daily data to monthly with enhanced functionality
    """
    if method == 'sum':
        monthly = daily_series.resample('M').sum()
    elif method == 'mean':
        monthly = daily_series.resample('M').mean()
    elif method == 'count':
        monthly = daily_series.resample('M').count()
    else:
        raise ValueError("Method must be 'sum', 'mean', or 'count'")
    
    return monthly

def calculate_monthly_irrigation_stats(irrigation_ts, reference_rain_ts):
    """
    Calculate comprehensive monthly irrigation statistics
    
    Args:
        irrigation_ts: Daily irrigation time series
        reference_rain_ts: Daily reference rainfall time series
    
    Returns:
        Dictionary with monthly irrigation statistics
    """
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'irrigation': irrigation_ts,
        'reference_rain': reference_rain_ts,
        'total_water': irrigation_ts + reference_rain_ts,
        'irrigation_flag': (irrigation_ts > 0).astype(int)
    })
    
    # Monthly aggregations
    monthly_stats = df.resample('M').agg({
        'irrigation': ['sum', 'mean', 'std', 'max'],
        'reference_rain': ['sum', 'mean'],
        'total_water': ['sum', 'mean'],
        'irrigation_flag': ['sum', 'count']
    })
    
    # Flatten column names
    monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns.values]
    
    # Calculate derived metrics
    monthly_stats['irrigation_frequency'] = (
        monthly_stats['irrigation_flag_sum'] / monthly_stats['irrigation_flag_count']
    )
    monthly_stats['irrigation_intensity'] = (
        monthly_stats['irrigation_sum'] / monthly_stats['irrigation_flag_sum'].replace(0, 1)
    )
    
    return monthly_stats

class SM2RAINCalibrator:
    """
    Complete SM2RAIN calibration class following the methodology from the paper
    """
    def __init__(self, sm_data: pd.Series, temp_data: pd.Series, rain_ref: pd.Series, 
                 theta_min: float = 0.0, theta_sat: float = 0.5, Kc: float = 1.0, tau_days:float = 5.0,
                 normalize_sm: bool = True):
        """
        Initialize SM2RAIN calibrator
        
        Args:
            sm_data: Soil moisture time series [m³/m³]
            temp_data: Air temperature time series [°C]
            rain_ref: Reference rainfall time series [mm/day]
            theta_min: Minimum soil moisture content [m³/m³] (used if normalize_sm=False)
            theta_sat: Saturated soil moisture content [m³/m³] (used if normalize_sm=False)
            Kc: Crop coefficient [-]
            tau_days: Exponential filter time constant [days]
            normalize_sm: If True, normalize soil moisture to 0-1 using data min/max
        """
        self.sm_data = sm_data
        self.temp_data = temp_data
        self.rain_ref = rain_ref
        self.theta_min = theta_min
        self.theta_sat = theta_sat
        self.Kc = Kc
        self.tau_days = tau_days
        self.normalize_sm = normalize_sm
        
        # Normalization parameters (stored for consistency)
        self.sm_norm_min = None
        self.sm_norm_max = None
        
        # Processed data storage
        self.sm_filtered = None
        self.S_rel = None
        self.dSdt = None
        self.ETpot = None
        self.optimized_params = None
        
    def preprocess_data(self, tau_days: float = 3.0):
        """
        Preprocess soil moisture data according to Wagner et al. exponential filter.
        
        If normalize_sm=True, the soil moisture is normalized to [0, 1] using the
        actual data min/max values before computing relative soil moisture.
        """
        # Filter soil moisture to remove noise
        self.sm_filtered = preprocess_soil_moisture(self.sm_data, tau_days)
        
        # Normalize soil moisture to 0-1 range if enabled
        if self.normalize_sm:
            # Use data-driven normalization
            self.S_rel, self.sm_norm_min, self.sm_norm_max = normalize_soil_moisture(self.sm_filtered)
            logger.debug(f"SM normalized using data range: [{self.sm_norm_min:.4f}, {self.sm_norm_max:.4f}]")
        else:
            # Use fixed theta_min/theta_sat (traditional approach)
            self.S_rel = compute_relative_soil_moisture(
                self.sm_filtered, self.theta_min, self.theta_sat
            )
        
        # Compute time derivative
        self.dSdt = compute_dSdt(self.S_rel)
        
        # logger.info(f"Data preprocessing completed. Relative SM range: [{self.S_rel.min():.3f}, {self.S_rel.max():.3f}]")
        # print()
    
    def calibrate_model(self, parameter_bounds=None):
        """
        Calibrate SM2RAIN model with 5 parameters: Zstar, Ks, lam, xi, tau_days
        """
        if self.S_rel is None or self.dSdt is None:
            self.preprocess_data()
        
        # Define 5-parameter optimization bounds
        if parameter_bounds is None:
            parameter_bounds = [
                (1.0, 200.0),     # Zstar: effective soil depth [mm]
                (0.1, 100.0),     # Ks: saturated hydraulic conductivity [mm/day]
                (-2.0, 5.0),      # lam: shape parameter [-]
                (0.1, 5.0),       # xi: ET parameter [-]
                (1.0, 10.0)       # tau_days: exponential filter time constant [days]
            ]
        
        # Initial parameter guess
        x0 = [50.0, 10.0, 0.5, 1.0, 3.0]  # [Zstar, Ks, lam, xi, tau_days]
        
        # Define objective function for 5 parameters
        def objective_5param(x):
            Zstar, Ks, lam, xi, tau_days = x
            
            try:
                # Reprocess with current tau_days
                sm_filtered_current = preprocess_soil_moisture(self.sm_data, tau_days)
                
                # Normalize soil moisture to 0-1 range if enabled
                if self.normalize_sm:
                    S_rel_current, _, _ = normalize_soil_moisture(sm_filtered_current)
                else:
                    S_rel_current = compute_relative_soil_moisture(
                        sm_filtered_current, self.theta_min, self.theta_sat
                    )
                dSdt_current = compute_dSdt(S_rel_current)
                
                # Compute ET with current xi
                ETpot_current = compute_ETpot(self.temp_data, xi, self.Kc)
                
                # SM2RAIN forward model
                params_current = {'Zstar': Zstar, 'Ks': Ks, 'lam': lam}
                r_est = sm2rain_forward(S_rel_current, dSdt_current, ETpot_current, params_current)
                r_est = r_est.clip(lower=0)
                
                # Compute RMSE with reference rainfall
                valid_mask = ~(np.isnan(r_est) | np.isnan(self.rain_ref))
                if np.sum(valid_mask) < 10:  # Need minimum valid points
                    return 1000.0
                
                r_est_valid = r_est[valid_mask]
                rain_ref_valid = self.rain_ref[valid_mask]
                
                rmse = np.sqrt(np.mean((r_est_valid - rain_ref_valid) ** 2))
                return rmse
                
            except Exception as e:
                logger.debug(f"Objective function error: {e}")
                return 1000.0
        
        # Optimize using scipy minimize
        from scipy.optimize import minimize
        result = minimize(
            objective_5param, 
            x0, 
            bounds=parameter_bounds, 
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        
        # Extract optimized parameters
        Zstar_opt, Ks_opt, lam_opt, xi_opt, tau_opt = result.x
        
        # Store optimized parameters
        self.optimized_params = {
            'Zstar': float(Zstar_opt),
            'Ks': float(Ks_opt),
            'lam': float(lam_opt),
            'xi': float(xi_opt),
            'tau_days': float(tau_opt),
            'Kc': float(self.Kc)
        }
        
        # Reprocess data with optimized tau_days
        self.preprocess_data(tau_days=tau_opt)
        
        # Compute final ETpot with optimized xi
        self.ETpot = compute_ETpot(self.temp_data, xi_opt, self.Kc)
        
        # logger.info(f"5-parameter. RMSE: {result.fun:.4f}")
        # logger.info(f"Optimized parameters: {self.optimized_params}")
            
        return {
            'parameters': self.optimized_params,
            'optimization_result': result,
            'rmse': float(result.fun),
            'success': result.success
        }
    
    def apply_global_parameters(self, global_params):
        """
        Apply globally optimized parameters to this calibrator instance
        """
        self.optimized_params = global_params.copy()
        
        # Reprocess data with global tau_days
        tau_days = global_params.get('tau_days', 3.0)
        self.preprocess_data(tau_days=tau_days)
        
        # Compute ETpot with global xi
        xi = global_params.get('xi', 1.0)
        self.ETpot = compute_ETpot(self.temp_data, xi, self.Kc)
        
        logger.debug(f"Applied global parameters: {global_params}")
    
    def sm2rain_forward(self, Zstar, Ks, lam, Kc=None, xi=None):
        """
        Run SM2RAIN forward model with given parameters.
        
        This method uses the preprocessed (and normalized if normalize_sm=True) 
        soil moisture data to estimate total water input.
        
        Parameters
        ----------
        Zstar : float
            Effective soil depth [mm]
        Ks : float
            Saturated hydraulic conductivity [mm/day]
        lam : float
            Shape parameter for drainage [-]
        Kc : float, optional
            Crop coefficient [-]. If None, uses self.Kc
        xi : float, optional
            ET parameter. If None, uses value from optimized_params or default 1.0
            
        Returns
        -------
        pd.Series
            Total water input time series [mm/day]
        """
        if self.S_rel is None or self.dSdt is None:
            self.preprocess_data()
        
        # Get xi parameter
        if xi is None:
            xi = self.optimized_params.get('xi', 1.0) if self.optimized_params else 1.0
        
        # Get Kc parameter
        if Kc is None:
            Kc = self.Kc
        
        # Compute ETpot
        ETpot = compute_ETpot(self.temp_data, xi, Kc)
        
        # Create params dict for forward model
        params = {'Zstar': Zstar, 'Ks': Ks, 'lam': lam}
        
        # Run SM2RAIN forward model (uses normalized S_rel)
        total_water = sm2rain_forward(self.S_rel, self.dSdt, ETpot, params)
        
        return total_water.clip(lower=0)

    def estimate_water_input(self):
        """
        Estimate total water input (rainfall + irrigation) using calibrated model
        
        Returns:
            pd.Series: Total water input time series [mm/day]
        """
        if self.optimized_params is None:
            raise ValueError("Model must be calibrated first")
        
        if self.ETpot is None:
            xi = self.optimized_params.get('xi', 1.0)
            self.ETpot = compute_ETpot(self.temp_data, xi, self.Kc)
        
        # Prepare parameters for forward model
        params = {
            'Zstar': self.optimized_params['Zstar'],
            'Ks': self.optimized_params['Ks'],
            'lam': self.optimized_params['lam']
        }
        
        # Run SM2RAIN forward model
        total_water = sm2rain_forward(self.S_rel, self.dSdt, self.ETpot, params)
        return total_water.clip(lower=0)  # Ensure non-negative values
    
    def detect_irrigation(self, irrigation_threshold: float = 2.5):
        """
        Detect irrigation events based on calibrated model
        
        Args:
            irrigation_threshold: Minimum irrigation amount to detect [mm/day]
            
        Returns:
            Dictionary with irrigation detection results
        """
        total_water = self.estimate_water_input()
        
        # Detect irrigation using multiple methods
        results = {
            'total_water_input': total_water,
            'threshold_method': separate_irrigation_rainfall(
                total_water, self.rain_ref, method='threshold'
            ),
            'residual_method': separate_irrigation_rainfall(
                total_water, self.rain_ref, method='residual'
            ),
            'irrigation_events': detect_irrigation_events(
                total_water, self.rain_ref, irrigation_threshold
            )
        }
        
        return results
    
    def validate_model(self):
        """
        Validate SM2RAIN model assumptions and data quality
        """
        if self.optimized_params is None:
            raise ValueError("Model must be calibrated first")
        
        total_water = self.estimate_water_input()
        return validate_sm2rain_assumptions(self.S_rel, self.dSdt, total_water, self.rain_ref)
    
    def get_water_balance_components(self):
        """
        Get all water balance components
        """
        if self.optimized_params is None:
            raise ValueError("Model must be calibrated first")
        
        params = {
            'Zstar': self.optimized_params['Zstar'],
            'Ks': self.optimized_params['Ks'],
            'lam': self.optimized_params['lam']
        }
        
        return compute_water_balance_components(self.S_rel, self.dSdt, self.ETpot, params)
    
    def get_monthly_irrigation_estimates(self, irrigation_threshold: float = 2.5):
        """
        Get monthly irrigation estimates from calibrated model
        
        Args:
            irrigation_threshold: Minimum threshold for irrigation detection
            
        Returns:
            Dictionary with monthly irrigation statistics
        """
        if self.optimized_params is None:
            raise ValueError("Model must be calibrated first")
        
        # Get total water input
        total_water = self.estimate_water_input()
        
        # Estimate irrigation as excess over reference rainfall
        irrigation_estimate = np.maximum(0, total_water - self.rain_ref - irrigation_threshold)
        
        # Calculate monthly statistics
        monthly_stats = calculate_monthly_irrigation_stats(irrigation_estimate, self.rain_ref)
        
        return {
            'monthly_statistics': monthly_stats,
            'daily_irrigation_estimate': irrigation_estimate,
            'total_water_input': total_water
        }

def run_sm2rain_analysis(soil_moisture_path, rainfall_path, temp_path, 
                         theta_min=0.0, theta_sat=0.5, var_names=None):
    """
    Complete SM2RAIN analysis pipeline following the paper methodology
    
    Args:
        soil_moisture_path: Path to soil moisture data file
        rainfall_path: Path to rainfall reference data file  
        temp_path: Path to temperature data file
        theta_min: Minimum soil moisture content
        theta_sat: Saturated soil moisture content
        var_names: Dictionary with variable names for each dataset
    
    Returns:
        Dictionary with results including parameters, estimates, and metrics
    """
    logger.info("Starting SM2RAIN analysis pipeline...")
    
    # Load and validate data
    logger.info("Loading input data...")
    data = load_data(soil_moisture_path, rainfall_path, temp_path, var_names)
    
    if 'sm' not in data or 'rain' not in data or 'Ta' not in data:
        raise ValueError("Required data (soil moisture, rainfall, temperature) not found")
    
    # Initialize SM2RAIN calibrator
    logger.info("Initializing SM2RAIN calibrator...")
    calibrator = SM2RAINCalibrator(data['sm'], data['Ta'], data['rain'], 
                                  theta_min=theta_min, theta_sat=theta_sat)
    
    # Calibrate model
    logger.info("Calibrating SM2RAIN model...")
    calibration_results = calibrator.calibrate_model()
    
    # Estimate water input
    logger.info("Estimating total water input...")
    total_water = calibrator.estimate_water_input()
    
    # Evaluate performance
    logger.info("Evaluating model performance...")
    metrics = evaluate_performance(total_water, data['rain'])
    
    # Detect irrigation
    logger.info("Detecting irrigation events...")
    irrigation_results = calibrator.detect_irrigation()
    
    # Validate model assumptions
    logger.info("Validating model assumptions...")
    validation = calibrator.validate_model()
    
    # Get water balance components
    water_balance = calibrator.get_water_balance_components()
    
def plot_comprehensive_results(calibrator, data):
    """
    Generate comprehensive plots for SM2RAIN analysis results
    
    Args:
        calibrator: Calibrated SM2RAINCalibrator object
        data: Input data dictionary
    """
    # Get results
    total_water = calibrator.estimate_water_input()
    irrigation_results = calibrator.detect_irrigation()
    water_balance = calibrator.get_water_balance_components()
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Time series comparison
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(data['rain'].index, data['rain'].values, 'b-', label='Reference Rainfall', alpha=0.7)
    ax1.plot(total_water.index, total_water.values, 'r-', label='SM2RAIN Estimate', alpha=0.7)
    ax1.set_ylabel('Rainfall [mm/day]')
    ax1.set_title('Rainfall Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Scatter plot
    ax2 = plt.subplot(3, 3, 2)
    valid_mask = ~(np.isnan(total_water) | np.isnan(data['rain']))
    est_valid = total_water[valid_mask]
    ref_valid = data['rain'][valid_mask]
    ax2.scatter(ref_valid, est_valid, alpha=0.5)
    max_val = max(np.max(ref_valid), np.max(est_valid))
    ax2.plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
    ax2.set_xlabel('Reference Rainfall [mm/day]')
    ax2.set_ylabel('Estimated Rainfall [mm/day]')
    ax2.set_title('Scatter Plot')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Soil moisture time series
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(calibrator.S_rel.index, calibrator.S_rel.values, 'g-', label='Relative SM')
    ax3.set_ylabel('Relative SM [-]')
    ax3.set_title('Soil Moisture')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Water balance components
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(water_balance['drainage'].index, water_balance['drainage'].values, 
             label='Drainage', alpha=0.7)
    ax4.plot(water_balance['evapotranspiration'].index, water_balance['evapotranspiration'].values, 
             label='ET', alpha=0.7)
    ax4.plot(water_balance['storage_change'].index, water_balance['storage_change'].values, 
             label='Storage Change', alpha=0.7)
    ax4.set_ylabel('Water Flux [mm/day]')
    ax4.set_title('Water Balance Components')
    ax4.legend()
    ax4.grid(True)
    
    # Plot 5: Monthly aggregation
    ax5 = plt.subplot(3, 3, 5)
    ref_monthly = aggregate_to_monthly(data['rain'], 'sum')
    est_monthly = aggregate_to_monthly(total_water, 'sum')
    ax5.plot(ref_monthly.index, ref_monthly.values, 'b-o', label='Reference', markersize=4)
    ax5.plot(est_monthly.index, est_monthly.values, 'r-s', label='Estimated', markersize=4)
    ax5.set_ylabel('Monthly Rainfall [mm/month]')
    ax5.set_title('Monthly Totals')
    ax5.legend()
    ax5.grid(True)
    
    # Plot 6: Irrigation detection
    ax6 = plt.subplot(3, 3, 6)
    irrigation_events = irrigation_results['irrigation_events']
    ax6.scatter(data['rain'].index[irrigation_events['irrigation_flags']], 
               irrigation_events['irrigation_amount'][irrigation_events['irrigation_flags']], 
               c='red', alpha=0.7, label='Detected Irrigation')
    ax6.set_ylabel('Irrigation [mm/day]')
    ax6.set_title('Irrigation Events')
    ax6.legend()
    ax6.grid(True)
    
    # Plot 7: Performance metrics text
    ax7 = plt.subplot(3, 3, 7)
    metrics = evaluate_performance(total_water, data['rain'])
    metrics_text = 'Performance Metrics:\n\n'
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            metrics_text += f'{key}: {value:.3f}\n'
    ax7.text(0.1, 0.9, metrics_text, transform=ax7.transAxes, 
             verticalalignment='top', fontfamily='monospace', fontsize=10)
    ax7.set_title('Performance Metrics')
    ax7.axis('off')
    
    # Plot 8: Parameter values
    ax8 = plt.subplot(3, 3, 8)
    params_text = 'Calibrated Parameters:\n\n'
    for key, value in calibrator.optimized_params.items():
        params_text += f'{key}: {value:.3f}\n'
    ax8.text(0.1, 0.9, params_text, transform=ax8.transAxes, 
             verticalalignment='top', fontfamily='monospace', fontsize=10)
    ax8.set_title('Model Parameters')
    ax8.axis('off')
    
    # Plot 9: Residuals
    ax9 = plt.subplot(3, 3, 9)
    residuals = total_water - data['rain']
    ax9.plot(residuals.index, residuals.values, 'k-', alpha=0.7)
    ax9.axhline(y=0, color='r', linestyle='--')
    ax9.set_ylabel('Residuals [mm/day]')
    ax9.set_title('Model Residuals')
    ax9.grid(True)
    
    plt.tight_layout()
    plt.suptitle('SM2RAIN Comprehensive Analysis Results', y=0.98, fontsize=16)
    plt.show()


def save_results_to_csv(results_dict, output_dir):
    """
    Save SM2RAIN analysis results to CSV files and text summaries
    """
    import os
    from utils import convert_numpy_types, safe_txt_dump
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy types to avoid serialization issues
    clean_results = convert_numpy_types(results_dict)
    
    # Save calibrated parameters
    if 'calibration_results' in clean_results and 'parameters' in clean_results['calibration_results']:
        params_df = pd.DataFrame([clean_results['calibration_results']['parameters']])
        params_df.to_csv(os.path.join(output_dir, 'calibrated_parameters.csv'), index=False)
    
    # Save performance metrics
    if 'performance_metrics' in clean_results:
        metrics_df = pd.DataFrame([clean_results['performance_metrics']])
        metrics_df.to_csv(os.path.join(output_dir, 'performance_metrics.csv'), index=False)
    
    # Save time series results
    if 'calibrator' in results_dict:
        calibrator = results_dict['calibrator']
        try:
            total_water = calibrator.estimate_water_input()
            
            ts_results = pd.DataFrame({
                'date': total_water.index,
                'reference_rainfall': results_dict['input_data']['rain'].values,
                'estimated_total_water': total_water.values,
                'soil_moisture': calibrator.S_rel.values,
                'temperature': results_dict['input_data']['Ta'].values
            })
            ts_results.to_csv(os.path.join(output_dir, 'time_series_results.csv'), index=False)
        except Exception as e:
            logger.warning(f"Could not save time series results: {e}")
    
    # Save irrigation detection results
    if 'irrigation_detection' in clean_results and 'irrigation_events' in clean_results['irrigation_detection']:
        irrigation_events = clean_results['irrigation_detection']['irrigation_events']
        try:
            irrigation_df = pd.DataFrame({
                'date': irrigation_events['irrigation_flags'].index if hasattr(irrigation_events['irrigation_flags'], 'index') else range(len(irrigation_events['irrigation_flags'])),
                'irrigation_flag': irrigation_events['irrigation_flags'],
                'irrigation_amount': irrigation_events['irrigation_amount']
            })
            irrigation_df.to_csv(os.path.join(output_dir, 'irrigation_detection.csv'), index=False)
        except Exception as e:
            logger.warning(f"Could not save irrigation detection results: {e}")
    
    # Save summary as text file
    safe_txt_dump(clean_results, os.path.join(output_dir, 'complete_results.txt'), "Complete SM2RAIN Analysis Results")
    
    logger.info(f"Results saved to {output_dir}")


def generate_summary_report(results_dict):
    """
    Generate a summary report of SM2RAIN analysis
    
    Args:
        results_dict: Dictionary containing analysis results
        
    Returns:
        String with summary report
    """
    calibrator = results_dict['calibrator']
    metrics = results_dict['performance_metrics']
    irrigation_results = results_dict['irrigation_detection']
    
    report = "SM2RAIN ANALYSIS SUMMARY REPORT\n"
    report += "=" * 50 + "\n\n"
    
    # Model parameters
    report += "CALIBRATED PARAMETERS:\n"
    report += "-" * 25 + "\n"
    for param, value in calibrator.optimized_params.items():
        report += f"{param:10s}: {value:8.3f}\n"
    report += "\n"
    
    # Performance metrics
    report += "MODEL PERFORMANCE:\n"
    report += "-" * 20 + "\n"
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            report += f"{metric:15s}: {value:8.3f}\n"
    report += "\n"
    
    # Irrigation detection summary
    irrigation_events = irrigation_results['irrigation_events']
    report += "IRRIGATION DETECTION:\n"
    report += "-" * 22 + "\n"
    report += f"Total events detected: {irrigation_events['irrigation_frequency']}\n"
    report += f"Total irrigation volume: {irrigation_events['total_irrigation_volume']:.2f} mm\n"
    report += f"Average irrigation per event: {irrigation_events['total_irrigation_volume']/max(1, irrigation_events['irrigation_frequency']):.2f} mm\n"
    report += "\n"
    
    # Data quality
    validation = results_dict['model_validation']
    report += "DATA QUALITY:\n"
    report += "-" * 15 + "\n"
    report += f"SM data completeness: {(1-validation['sm_missing_fraction'])*100:.1f}%\n"
    report += f"Rain data completeness: {(1-validation['rain_missing_fraction'])*100:.1f}%\n"
    report += f"SM-Rain correlation: {validation.get('water_rain_correlation', 'N/A')}\n"
    
    return report


def create_sm2rain_config(output_path=None):
    """
    Create a configuration file template for SM2RAIN analysis
    
    Args:
        output_path: Path to save config file (optional)
    
    Returns:
        Dictionary with default configuration
    """
    config = {
        'data_paths': {
            'soil_moisture': 'path/to/soil_moisture.nc',
            'rainfall': 'path/to/rainfall.nc', 
            'temperature': 'path/to/temperature.nc'
        },
        'variable_names': {
            'sm': 'soil_moisture',
            'rain': 'precipitation',
            'temp': 'temperature'
        },
        'parameters': {
            'theta_min': 0.0,
            'theta_sat': 0.5,
            'tau_days': 3.0,
            'Kc': 1.0
        },
        'calibration': {
            'method': 'L-BFGS-B',
            'max_iter': 1000
        },
        'irrigation_detection': {
            'threshold': 2.0,
            'method': 'threshold'
        },
        'output': {
            'save_results': True,
            'output_dir': 'sm2rain_results',
            'generate_plots': True
        }
    }
    
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration template saved to {output_path}")
    
    return config


def run_sm2rain_from_config(config_path):
    """
    Run SM2RAIN analysis from configuration file
    
    Args:
        config_path: Path to configuration JSON file
    
    Returns:
        Analysis results dictionary
    """
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract paths and parameters
    data_paths = config['data_paths']
    var_names = config.get('variable_names', None)
    params = config['parameters']
    
    # Run analysis
    results = run_sm2rain_analysis(
        data_paths['soil_moisture'],
        data_paths['rainfall'], 
        data_paths['temperature'],
        theta_min=params['theta_min'],
        theta_sat=params['theta_sat'],
        var_names=var_names
    )
    
    # Save results if requested
    if config['output']['save_results']:
        save_results_to_csv(results, config['output']['output_dir'])
    
    return results


def compare_irrigation_methods(calibrator, reference_rain):
    """
    Compare different irrigation detection methods
    
    Args:
        calibrator: Calibrated SM2RAINCalibrator object
        reference_rain: Reference rainfall data
    
    Returns:
        Comparison results dictionary
    """
    total_water = calibrator.estimate_water_input()
    
    # Different detection methods
    methods = {
        'threshold': separate_irrigation_rainfall(total_water, reference_rain, 'threshold'),
        'residual': separate_irrigation_rainfall(total_water, reference_rain, 'residual'),
        'correlation': separate_irrigation_rainfall(total_water, reference_rain, 'correlation')
    }
    
    # Advanced detection based on water balance
    advanced_detection = detect_irrigation_events(
        total_water, reference_rain
    )
    methods['advanced'] = {
        'estimated_irrigation': advanced_detection['irrigation_amount'],
        'estimated_rainfall': reference_rain.copy()
    }
    
    # Compare results
    comparison = {}
    for method_name, method_results in methods.items():
        irrigation_total = method_results['estimated_irrigation'].sum()
        irrigation_events = (method_results['estimated_irrigation'] > 0).sum()
        
        comparison[method_name] = {
            'total_irrigation_volume': irrigation_total,
            'irrigation_events_count': irrigation_events,
            'average_irrigation_per_event': irrigation_total / max(1, irrigation_events),
            'irrigation_frequency': irrigation_events / len(reference_rain)
        }
        irrigation_events = (method_results['estimated_irrigation'] > 0).sum()
        
        comparison[method_name] = {
            'total_irrigation_volume': irrigation_total,
            'irrigation_events_count': irrigation_events,
            'average_irrigation_per_event': irrigation_total / max(1, irrigation_events),
            'irrigation_frequency': irrigation_events / len(reference_rain),
            'total_irrigation_volume': irrigation_total,
            'irrigation_events_count': irrigation_events,
            'average_irrigation_per_event': irrigation_total / max(1, irrigation_events),
            'irrigation_frequency': irrigation_events / len(reference_rain)
        }
            
"""
Gridded SM2RAIN Runner

This module provides the gridded analysis pipeline for SM2RAIN irrigation detection.
It uses the 4-parameter calibration model (Z*, Ks, lambda, Kc) and processes 
NetCDF and CSV data files.

Data Files:
    - Temperature: NetCDF (agERA5_24_hour_mean_combined.nc)
    - Precipitation: NetCDF (GPM_IMERG_Daily_ROI_main.nc)
    - Soil Moisture: CSV (SMAP_SPL3SMP_E_30N-31N_75E-76E_2017-2021.csv)
    
Configuration:
    - T (exponential filter time constant) is a configurable constant
    - Calibration is performed only on NDVI dates
    
Reference:
    Brocca et al. (2018) SM2RAIN methodology
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import our modules
from data_preprocessor import GriddedDataProcessor, load_ndvi_dates
from main import SM2RAINCalibrator, normalize_soil_moisture
from utils import convert_numpy_types, safe_txt_dump

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION - File Paths
# =============================================================================
TEMP_PATH = r"agERA5_ROI_main/combined/agERA5_24_hour_mean_combined.nc"#r"G:\sm2rain-irrigation\agERA5_ROI_main\combined\agERA5_24_hour_mean_combined.nc"
PRECIP_PATH = r"/media/sentoki/kishodai/sm2rain-irrigation/GPM_IMERG_Daily_ROI_main.nc"#r"G:\sm2rain-irrigation\GPM_IMERG_Daily_ROI_main.nc"
SOIL_MOISTURE_PATH = r"data/SMAP_data/SMAP_SPL3SMP_E_30N-31N_75E-76E_2017-2021.csv" #r"G:\sm2rain-irrigation\data\SMAP_data\SMAP_SPL3SMP_E_30N-31N_75E-76E_2017-2021.csv"
NDVI_DATES_FILE = r"/media/sentoki/kishodai/sm2rain-irrigation/ndvi_dates_mean.txt"#r"G:\sm2rain-irrigation\ndvi_dates_mean.txt"

# =============================================================================
# CONFIGURATION - Exponential Filter Time Constant
# =============================================================================
# T is the characteristic time length of the soil layer (days)
# Typical values: 1-10 days depending on soil type
# This is NOT calibrated - it is a fixed constant that you can change here
T_EXPONENTIAL_FILTER = 5.0  # days


# =============================================================================
# CONFIGURATION - Default Region of Interest
# =============================================================================
ROI = {
    'min_lat': 30.0,
    'max_lat': 31.0,
    'min_lon': 75.0,
    'max_lon': 76.0
}


def load_all_data(temp_path=None, precip_path=None, soil_moisture_path=None, 
                  ndvi_dates_file=None, filter_ndvi=True):
    """
    Load all data from NetCDF and CSV files and prepare gridded data.
    
    Parameters
    ----------
    temp_path : str, optional
        Path to temperature NetCDF file
    precip_path : str, optional
        Path to precipitation NetCDF file
    soil_moisture_path : str, optional
        Path to soil moisture CSV file
    ndvi_dates_file : str, optional
        Path to NDVI dates file
    filter_ndvi : bool
        Whether to filter to NDVI dates only
        
    Returns
    -------
    dict
        Dictionary containing gridded data for each common grid point
    """
    temp_path = temp_path or TEMP_PATH
    precip_path = precip_path or PRECIP_PATH
    soil_moisture_path = soil_moisture_path or SOIL_MOISTURE_PATH
    ndvi_dates_file = ndvi_dates_file or NDVI_DATES_FILE
    
    # logger.info("Loading data from files...")
    # logger.info(f"  Temperature: {temp_path}")
    # logger.info(f"  Precipitation: {precip_path}")
    # logger.info(f"  Soil Moisture: {soil_moisture_path}")
    
    preprocessor = GriddedDataProcessor(
        temp_path=temp_path,
        precip_path=precip_path,
        sm_path=soil_moisture_path,
        ndvi_dates_path=ndvi_dates_file if filter_ndvi else None
    )
    
    # Prepare gridded data for all common grid points
    gridded_data = preprocessor.prepare_gridded_data(
        filter_ndvi=filter_ndvi,
        min_data_points=20
    )
    
    logger.info(f"Prepared gridded data for {len(gridded_data)} grid points")
    
    return gridded_data


def find_common_grid_points(data, tolerance=0.1):
    """
    Find grid points that have data in all three datasets.
    
    Parameters
    ----------
    data : dict
        Dictionary containing temperature, precipitation, and soil_moisture DataFrames
    tolerance : float
        Spatial tolerance for matching grid points (degrees)
        
    Returns
    -------
    list
        List of (lat, lon) tuples for common grid points
    """
    from scipy.spatial import cKDTree
    
    logger.info("Finding common grid points across datasets...")
    
    # Get unique coordinates from each dataset
    temp_df = data['temperature']
    precip_df = data['precipitation']
    sm_df = data['soil_moisture']
    
    # Get unique coordinates
    temp_coords = temp_df[['lat', 'lon']].drop_duplicates().values
    precip_coords = precip_df[['lat', 'lon']].drop_duplicates().values
    sm_coords = sm_df[['Latitude', 'Longitude']].drop_duplicates().values
    
    logger.info(f"  Temperature unique grid points: {len(temp_coords)}")
    logger.info(f"  Precipitation unique grid points: {len(precip_coords)}")
    logger.info(f"  Soil Moisture unique grid points: {len(sm_coords)}")
    
    # Use diagonal tolerance for staggered grids
    # GPM grids are offset by 0.05 degrees from ERA5 grids
    diagonal_tolerance = tolerance * np.sqrt(2) / 2 + 0.01
    
    # Build KD-trees for efficient spatial matching
    temp_tree = cKDTree(temp_coords)
    precip_tree = cKDTree(precip_coords)
    
    # Find soil moisture points that have matches in both temp and precip
    common_points = []
    
    for sm_point in sm_coords:
        # Find nearest temperature point
        temp_dist, temp_idx = temp_tree.query(sm_point)
        # Find nearest precipitation point
        precip_dist, precip_idx = precip_tree.query(sm_point)
        
        if temp_dist <= diagonal_tolerance and precip_dist <= diagonal_tolerance:
            # Use soil moisture coordinates as the reference
            common_points.append({
                'lat': sm_point[0],
                'lon': sm_point[1],
                'temp_lat': temp_coords[temp_idx][0],
                'temp_lon': temp_coords[temp_idx][1],
                'precip_lat': precip_coords[precip_idx][0],
                'precip_lon': precip_coords[precip_idx][1],
                'temp_dist': temp_dist,
                'precip_dist': precip_dist
            })
    
    logger.info(f"Found {len(common_points)} common grid points")
    
    return common_points


def prepare_grid_data(data, common_points, ndvi_dates=None):
    """
    Prepare time series data for each common grid point.
    
    Parameters
    ----------
    data : dict
        Dictionary containing loaded data
    common_points : list
        List of common grid point dictionaries
    ndvi_dates : list, optional
        List of NDVI dates to filter to
        
    Returns
    -------
    dict
        Dictionary keyed by grid_id containing time series for each point
    """
    logger.info("Preparing grid data for analysis...")
    
    temp_df = data['temperature']
    precip_df = data['precipitation']
    sm_df = data['soil_moisture']
    
    # Convert dates to datetime
    temp_df['date'] = pd.to_datetime(temp_df['time'])
    precip_df['date'] = pd.to_datetime(precip_df['time'])
    sm_df['date'] = pd.to_datetime(sm_df['Date'])
    
    gridded_data = {}
    tolerance = 0.05  # Coordinate matching tolerance
    
    for i, point in enumerate(common_points):
        grid_id = i + 1
        
        try:
            # Extract temperature for this point
            temp_mask = (
                (np.abs(temp_df['lat'] - point['temp_lat']) < tolerance) &
                (np.abs(temp_df['lon'] - point['temp_lon']) < tolerance)
            )
            temp_grid = temp_df[temp_mask][['date', 'Temperature_Air_2m_Mean_24h']].copy()
            temp_grid = temp_grid.rename(columns={'Temperature_Air_2m_Mean_24h': 'temperature'})
            temp_grid = temp_grid.set_index('date').sort_index()
            
            # Extract precipitation for this point
            precip_mask = (
                (np.abs(precip_df['lat'] - point['precip_lat']) < tolerance) &
                (np.abs(precip_df['lon'] - point['precip_lon']) < tolerance)
            )
            precip_grid = precip_df[precip_mask][['date', 'precipitation']].copy()
            precip_grid = precip_grid.set_index('date').sort_index()
            
            # Extract soil moisture for this point
            sm_mask = (
                (np.abs(sm_df['Latitude'] - point['lat']) < tolerance) &
                (np.abs(sm_df['Longitude'] - point['lon']) < tolerance)
            )
            sm_grid = sm_df[sm_mask][['date', 'Soil_Moisture']].copy()
            sm_grid = sm_grid.rename(columns={'Soil_Moisture': 'soil_moisture'})
            sm_grid = sm_grid.set_index('date').sort_index()
            
            # Remove duplicate dates (take mean)
            temp_grid = temp_grid.groupby(temp_grid.index).mean()
            precip_grid = precip_grid.groupby(precip_grid.index).mean()
            sm_grid = sm_grid.groupby(sm_grid.index).mean()
            
            # Find common dates
            common_dates = temp_grid.index.intersection(
                precip_grid.index
            ).intersection(
                sm_grid.index
            )
            
            # Filter to NDVI dates if provided
            if ndvi_dates is not None and len(ndvi_dates) > 0:
                ndvi_datetime = pd.to_datetime(ndvi_dates)
                common_dates = common_dates.intersection(ndvi_datetime)
            
            if len(common_dates) < 20:
                logger.debug(f"Grid {grid_id}: Insufficient data ({len(common_dates)} days)")
                continue
            
            # Create aligned time series
            gridded_data[grid_id] = {
                'sm': sm_grid['soil_moisture'].reindex(common_dates),
                'rain': precip_grid['precipitation'].reindex(common_dates),
                'Ta': temp_grid['temperature'].reindex(common_dates),
                'coordinates': {
                    'lat': point['lat'],
                    'lon': point['lon'],
                    'lat_center': point['lat'],
                    'lon_center': point['lon']
                }
            }
            
        except Exception as e:
            logger.warning(f"Failed to process grid {grid_id}: {e}")
            continue
    
    logger.info(f"Successfully prepared data for {len(gridded_data)} grid points")
    return gridded_data


def calibrate_single_grid_point(grid_data_tuple, T_filter=None):
    """
    Calibrate SM2RAIN parameters for a single grid point.
    
    Parameters
    ----------
    grid_data_tuple : tuple
        (grid_id, grid_data) tuple
    T_filter : float, optional
        Exponential filter time constant (days)
        
    Returns
    -------
    dict
        Calibration results for this grid point
    """
    grid_id, grid_data = grid_data_tuple
    T_filter = T_filter or T_EXPONENTIAL_FILTER
    
    try:
        sm_data = grid_data['sm']
        rain_data = grid_data['rain']
        temp_data = grid_data['Ta']
        coordinates = grid_data['coordinates']
        
        # Filter to only days where soil moisture is available (not NaN)
        valid_sm_mask = ~sm_data.isna()
        valid_dates = sm_data[valid_sm_mask].index
        
        sm_data = sm_data[valid_sm_mask]
        rain_data = rain_data.reindex(valid_dates)
        temp_data = temp_data.reindex(valid_dates)
        
        # Data quality checks
        if len(sm_data) < 20:
            return {
                'grid_id': grid_id,
                'status': 'insufficient_data',
                'error': f'Only {len(sm_data)} days of data (minimum 20 required)',
                'coordinates': coordinates
            }
        
        sm_missing_pct = sm_data.isna().sum() / len(sm_data)
        rain_missing_pct = rain_data.isna().sum() / len(rain_data)
        temp_missing_pct = temp_data.isna().sum() / len(temp_data)
        
        if max(sm_missing_pct, rain_missing_pct, temp_missing_pct) > 0.3:
            return {
                'grid_id': grid_id,
                'status': 'poor_data_quality',
                'error': f'Excessive missing data',
                'coordinates': coordinates
            }
        
        # Validate soil moisture range (should be volumetric, typically 0-0.6)
        sm_valid = sm_data.dropna()
        if len(sm_valid) == 0:
            return {
                'grid_id': grid_id,
                'status': 'invalid_sm_range',
                'error': 'No valid soil moisture data',
                'coordinates': coordinates
            }
        
        # Create calibrator with 4-parameter model
        # normalize_sm=True ensures soil moisture is rescaled to 0-1 range
        calibrator = SM2RAINCalibrator(
            sm_data=sm_data,
            temp_data=temp_data,
            rain_ref=rain_data,
            tau_days=T_filter,  # Use fixed exponential filter constant
            normalize_sm=True   # Rescale soil moisture to 0-1 range
        )
        
        # Preprocess data (includes normalization)
        calibrator.preprocess_data()
        
        if calibrator.S_rel is None or len(calibrator.S_rel.dropna()) < 15:
            return {
                'grid_id': grid_id,
                'status': 'preprocessing_failed',
                'error': 'Insufficient valid data after preprocessing',
                'coordinates': coordinates
            }
        
        # Run calibration (4 parameters: Z*, Ks, lambda, Kc)
        calibration_results = calibrator.calibrate_model()
        
        if calibration_results['rmse'] > 50:
            return {
                'grid_id': grid_id,
                'status': 'poor_calibration',
                'error': f'High RMSE: {calibration_results["rmse"]:.2f}',
                'coordinates': coordinates,
                'rmse': calibration_results['rmse']
            }
        
        # Extract calibrated parameters
        calibrated_params = {
            'Zstar': calibration_results['parameters']['Zstar'],
            'Ks': calibration_results['parameters']['Ks'],
            'lam': calibration_results['parameters']['lam'],
            'Kc': calibration_results['parameters']['Kc'],
            'T': T_filter  # Include the fixed T value
        }
        
        return {
            'grid_id': grid_id,
            'coordinates': coordinates,
            'status': 'success',
            'rmse': calibration_results['rmse'],
            'correlation': calibration_results.get('correlation', np.nan),
            'nse': calibration_results.get('nse', np.nan),
            'parameters': calibrated_params,
            'data_quality': {
                'n_days': len(sm_data),
                'sm_missing_pct': sm_missing_pct,
                'rain_missing_pct': rain_missing_pct,
                'temp_missing_pct': temp_missing_pct
            }
        }
        
    except Exception as e:
        return {
            'grid_id': grid_id,
            'status': 'error',
            'error': str(e),
            'coordinates': coordinates if 'coordinates' in locals() else {}
        }


def detect_irrigation_single_grid(grid_data_tuple, global_params, T_filter=None):
    """
    Detect irrigation for a single grid point using global parameters.
    
    Parameters
    ----------
    grid_data_tuple : tuple
        (grid_id, grid_data) tuple
    global_params : dict
        Global calibrated parameters
    T_filter : float, optional
        Exponential filter time constant
        
    Returns
    -------
    dict
        Irrigation detection results
    """
    grid_id, grid_data = grid_data_tuple
    T_filter = T_filter or T_EXPONENTIAL_FILTER
    
    try:
        sm_data = grid_data['sm']
        rain_data = grid_data['rain']
        temp_data = grid_data['Ta']
        coordinates = grid_data['coordinates']
        
        # Filter to only days where soil moisture is available (not NaN)
        valid_sm_mask = ~sm_data.isna()
        valid_dates = sm_data[valid_sm_mask].index
        
        sm_data = sm_data[valid_sm_mask]
        rain_data = rain_data.reindex(valid_dates)
        temp_data = temp_data.reindex(valid_dates)
        
        # Create calibrator with global parameters
        # normalize_sm=True ensures soil moisture is rescaled to 0-1 range
        calibrator = SM2RAINCalibrator(
            sm_data=sm_data,
            temp_data=temp_data,
            rain_ref=rain_data,
            tau_days=T_filter,
            normalize_sm=True  # Rescale soil moisture to 0-1 range
        )
        
        # Preprocess (includes normalization)
        calibrator.preprocess_data()
        
        # Estimate total water input using global parameters
        total_water_input = calibrator.sm2rain_forward(
            Zstar=global_params['Zstar'],
            Ks=global_params['Ks'],
            lam=global_params['lam'],
            Kc=global_params['Kc']
        )
        
        if total_water_input is None or total_water_input.isna().all():
            return {
                'grid_id': grid_id,
                'status': 'water_estimation_failed',
                'error': 'Total water input estimation failed',
                'coordinates': coordinates
            }
        
        # Compute irrigation as residual
        aligned_rain = rain_data.reindex(total_water_input.index).fillna(0)
        irrigation_threshold = 2.5  # mm/day minimum
        
        daily_irrigation_mm = np.maximum(0, total_water_input - aligned_rain)
        daily_irrigation_mm = np.where(
            daily_irrigation_mm > irrigation_threshold,
            daily_irrigation_mm,
            0
        )
        irrigation_flags = daily_irrigation_mm > 0.1
        
        # Generate monthly estimates
        monthly_estimates = generate_monthly_grid_estimates(
            grid_id=grid_id,
            coordinates=coordinates,
            total_water_input=total_water_input,
            rain_data=aligned_rain,
            daily_irrigation_mm=daily_irrigation_mm,
            irrigation_flags=irrigation_flags,
            soil_moisture=calibrator.S_rel,
            temp_data=temp_data
        )
        
        # Create daily time series
        daily_time_series = pd.DataFrame({
            'date': total_water_input.index,
            'grid_id': grid_id,
            'latitude': coordinates.get('lat', np.nan),
            'longitude': coordinates.get('lon', np.nan),
            'total_water_input': total_water_input.values,
            'reference_rainfall': aligned_rain.values,
            'irrigation_mm': daily_irrigation_mm,
            'irrigation_events': irrigation_flags.astype(int),
            'soil_moisture': calibrator.S_rel.reindex(total_water_input.index).values,
            'temperature': temp_data.reindex(total_water_input.index).values
        })
        
        total_events = int(np.sum(irrigation_flags))
        total_volume = float(np.sum(daily_irrigation_mm))
        
        # Calculate performance metrics
        correlation = np.nan
        rmse = np.nan
        nse = np.nan
        
        valid_mask = ~(total_water_input.isna() | aligned_rain.isna())
        if valid_mask.sum() > 10:
            try:
                correlation = np.corrcoef(
                    total_water_input[valid_mask],
                    aligned_rain[valid_mask]
                )[0, 1]
                rmse = np.sqrt(np.mean(
                    (total_water_input[valid_mask] - aligned_rain[valid_mask])**2
                ))
                mean_rain = aligned_rain[valid_mask].mean()
                if mean_rain > 0:
                    ss_res = np.sum((aligned_rain[valid_mask] - total_water_input[valid_mask])**2)
                    ss_tot = np.sum((aligned_rain[valid_mask] - mean_rain)**2)
                    nse = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            except Exception:
                pass
        
        return {
            'grid_id': grid_id,
            'coordinates': coordinates,
            'status': 'success',
            'performance': {
                'correlation': correlation,
                'rmse': rmse,
                'nse': nse
            },
            'irrigation': {
                'total_events': total_events,
                'total_volume': total_volume,
                'average_per_event': total_volume / max(1, total_events)
            },
            'daily_time_series': daily_time_series,
            'monthly_estimates': monthly_estimates,
            'water_balance': {
                'total_water_input_mean': float(total_water_input.mean()),
                'total_water_input_sum': float(total_water_input.sum()),
                'reference_rain_sum': float(aligned_rain.sum())
            }
        }
        
    except Exception as e:
        return {
            'grid_id': grid_id,
            'status': 'error',
            'error': str(e),
            'coordinates': coordinates if 'coordinates' in locals() else {}
        }


def generate_monthly_grid_estimates(grid_id, coordinates, total_water_input, rain_data,
                                    daily_irrigation_mm, irrigation_flags, soil_moisture,
                                    temp_data):
    """
    Generate monthly irrigation estimates for a grid point.
    
    Parameters
    ----------
    grid_id : int
        Grid point identifier
    coordinates : dict
        Grid point coordinates
    total_water_input : pd.Series
        Total water input time series
    rain_data : pd.Series
        Reference rainfall time series
    daily_irrigation_mm : np.ndarray
        Daily irrigation amounts
    irrigation_flags : np.ndarray
        Boolean irrigation flags
    soil_moisture : pd.Series
        Relative soil moisture
    temp_data : pd.Series
        Temperature data
        
    Returns
    -------
    list
        List of monthly estimate dictionaries
    """
    try:
        common_index = total_water_input.index
        
        daily_df = pd.DataFrame({
            'date': common_index,
            'total_water_input': total_water_input.values,
            'reference_rainfall': rain_data.reindex(common_index).fillna(0).values,
            'irrigation_mm': daily_irrigation_mm,
            'irrigation_events': irrigation_flags.astype(int),
            'soil_moisture': soil_moisture.reindex(common_index).fillna(0.3).values,
            'temperature': temp_data.reindex(common_index).fillna(20.0).values
        })
        daily_df.set_index('date', inplace=True)
        
        # Monthly aggregation
        monthly_aggregated = daily_df.resample('M').agg({
            'irrigation_mm': 'sum',
            'irrigation_events': 'sum',
            'total_water_input': 'sum',
            'reference_rainfall': 'sum',
            'soil_moisture': 'mean',
            'temperature': 'mean'
        })
        
        monthly_estimates = []
        for month_end, row in monthly_aggregated.iterrows():
            year = month_end.year
            month = month_end.month
            month_name = month_end.strftime('%B')
            month_start = month_end.replace(day=1)
            
            month_mask = (daily_df.index >= month_start) & (daily_df.index <= month_end)
            days_in_month = month_mask.sum()
            irrigation_days = daily_df.loc[month_mask, 'irrigation_events'].sum()
            
            monthly_estimate = {
                'grid_id': grid_id,
                'latitude': coordinates.get('lat', np.nan),
                'longitude': coordinates.get('lon', np.nan),
                'year': year,
                'month': month,
                'month_name': month_name,
                'year_month': f"{year}-{month:02d}",
                'month_start': month_start.strftime('%Y-%m-%d'),
                'month_end': month_end.strftime('%Y-%m-%d'),
                'days_in_month': int(days_in_month),
                'irrigation_volume_mm': float(row['irrigation_mm']),
                'irrigation_events': int(irrigation_days),
                'irrigation_frequency': float(irrigation_days / max(1, days_in_month)),
                'total_water_input_mm': float(row['total_water_input']),
                'reference_rainfall_mm': float(row['reference_rainfall']),
                'avg_soil_moisture': float(row['soil_moisture']),
                'avg_temperature_c': float(row['temperature']),
                'irrigation_intensity': float(row['irrigation_mm'] / max(1, irrigation_days))
            }
            monthly_estimates.append(monthly_estimate)
        
        return monthly_estimates
        
    except Exception as e:
        logger.error(f"Error generating monthly estimates for grid {grid_id}: {e}")
        return []


def phase1_global_calibration(gridded_data, output_path, T_filter=None, max_workers=8):
    """
    Phase 1: Global parameter calibration across all grid points.
    
    Parameters
    ----------
    gridded_data : dict
        Dictionary of grid point data
    output_path : Path
        Output directory path
    T_filter : float, optional
        Exponential filter time constant
    max_workers : int
        Maximum number of parallel workers
        
    Returns
    -------
    tuple
        (global_params, calibration_summary)
    """
    T_filter = T_filter or T_EXPONENTIAL_FILTER
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: GLOBAL PARAMETER CALIBRATION (4 PARAMETERS)")
    logger.info("="*80)
    logger.info(f"Parameters: Z*, Ks, lambda, Kc")
    logger.info(f"Fixed T (exponential filter): {T_filter} days")
    logger.info(f"Calibrating {len(gridded_data)} grid points...")
    
    grid_data_items = list(gridded_data.items())
    calibration_results = {}
    
    if max_workers == 1:
        for grid_item in tqdm(grid_data_items, desc="Calibrating grids"):
            result = calibrate_single_grid_point(grid_item, T_filter)
            calibration_results[result['grid_id']] = result
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_grid = {
                executor.submit(calibrate_single_grid_point, grid_item, T_filter): grid_item[0]
                for grid_item in grid_data_items
            }
            for future in tqdm(as_completed(future_to_grid),
                              total=len(grid_data_items),
                              desc="Calibrating grids"):
                result = future.result()
                calibration_results[result['grid_id']] = result
    
    successful_calibrations = {
        k: v for k, v in calibration_results.items() if v['status'] == 'success'
    }
    failed_calibrations = {
        k: v for k, v in calibration_results.items() if v['status'] != 'success'
    }
    
    logger.info(f"Calibration completed: {len(successful_calibrations)} successful, "
                f"{len(failed_calibrations)} failed")
    
    # Log first few failures for debugging
    if len(failed_calibrations) > 0:
        logger.warning("Sample of calibration failures:")
        for i, (grid_id, result) in enumerate(list(failed_calibrations.items())[:5]):
            logger.warning(f"  Grid {grid_id}: {result.get('status', 'unknown')} - {result.get('error', 'no error message')}")
    
    if not successful_calibrations:
        raise ValueError("No successful calibrations found")
    
    # Select best parameters (lowest RMSE)
    best_grid_id = min(
        successful_calibrations.keys(),
        key=lambda x: successful_calibrations[x]['rmse']
    )
    best_result = successful_calibrations[best_grid_id]
    global_params = best_result['parameters'].copy()
    
    # RMSE statistics
    rmse_values = [result['rmse'] for result in successful_calibrations.values()]
    
    calibration_summary = {
        'total_grids': len(gridded_data),
        'successful_calibrations': len(successful_calibrations),
        'failed_calibrations': len(failed_calibrations),
        'success_rate': len(successful_calibrations) / len(gridded_data),
        'best_grid_id': best_grid_id,
        'best_rmse': best_result['rmse'],
        'T_exponential_filter': T_filter,
        'rmse_statistics': {
            'mean': np.mean(rmse_values),
            'std': np.std(rmse_values),
            'min': np.min(rmse_values),
            'max': np.max(rmse_values)
        }
    }
    
    # Save results
    with open(output_path / 'phase1_calibration_results.json', 'w') as f:
        json.dump(convert_numpy_types(calibration_results), f, indent=2, default=str)
    
    with open(output_path / 'phase1_calibration_summary.json', 'w') as f:
        json.dump(convert_numpy_types(calibration_summary), f, indent=2, default=str)
    
    safe_txt_dump(calibration_summary, str(output_path / 'phase1_calibration_summary.txt'),
                  "Phase 1: Global Calibration Summary (4 Parameters)")
    
    logger.info(f"GLOBAL CALIBRATION SUMMARY:")
    logger.info(f"  Success rate: {calibration_summary['success_rate']:.1%}")
    logger.info(f"  Best RMSE: {calibration_summary['best_rmse']:.4f} (Grid {best_grid_id})")
    logger.info(f"  Global parameters:")
    logger.info(f"    Z* = {global_params['Zstar']:.2f} mm")
    logger.info(f"    Ks = {global_params['Ks']:.4f} mm/day")
    logger.info(f"    lambda = {global_params['lam']:.4f}")
    logger.info(f"    Kc = {global_params['Kc']:.4f}")
    logger.info(f"    T (fixed) = {T_filter} days")
    logger.info(f"  Results saved to: {output_path}")
    
    return global_params, calibration_summary


def phase2_irrigation_detection(gridded_data, global_params, output_path,
                                 T_filter=None, max_workers=8):
    """
    Phase 2: Irrigation detection using global parameters.
    
    Parameters
    ----------
    gridded_data : dict
        Dictionary of grid point data
    global_params : dict
        Global calibrated parameters
    output_path : Path
        Output directory path
    T_filter : float, optional
        Exponential filter time constant
    max_workers : int
        Maximum number of parallel workers
        
    Returns
    -------
    tuple
        (irrigation_results, irrigation_summary)
    """
    T_filter = T_filter or T_EXPONENTIAL_FILTER
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: IRRIGATION DETECTION WITH MONTHLY ESTIMATES")
    logger.info("="*80)
    logger.info(f"Using global parameters: Z*={global_params['Zstar']:.2f}, "
                f"Ks={global_params['Ks']:.4f}, lambda={global_params['lam']:.4f}, "
                f"Kc={global_params['Kc']:.4f}")
    logger.info(f"Processing {len(gridded_data)} grid points")
    
    grid_data_items = list(gridded_data.items())
    irrigation_results = {}
    
    if max_workers == 1:
        for grid_item in tqdm(grid_data_items, desc="Detecting irrigation"):
            result = detect_irrigation_single_grid(grid_item, global_params, T_filter)
            irrigation_results[result['grid_id']] = result
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_grid = {
                executor.submit(detect_irrigation_single_grid, grid_item, global_params, T_filter): grid_item[0]
                for grid_item in grid_data_items
            }
            for future in tqdm(as_completed(future_to_grid),
                              total=len(grid_data_items),
                              desc="Detecting irrigation"):
                result = future.result()
                irrigation_results[result['grid_id']] = result
    
    successful_detections = {
        k: v for k, v in irrigation_results.items() if v['status'] == 'success'
    }
    failed_detections = {
        k: v for k, v in irrigation_results.items() if v['status'] != 'success'
    }
    
    logger.info(f"Irrigation detection completed: {len(successful_detections)} successful, "
                f"{len(failed_detections)} failed")
    
    # Save results
    save_monthly_grid_estimates(irrigation_results, output_path)
    save_daily_time_series_results(irrigation_results, output_path)
    
    # Generate summary
    if successful_detections:
        irrigation_events = [
            result['irrigation']['total_events']
            for result in successful_detections.values()
        ]
        irrigation_volumes = [
            result['irrigation']['total_volume']
            for result in successful_detections.values()
        ]
        
        grids_with_irrigation = sum(1 for e in irrigation_events if e > 0)
        total_events = sum(irrigation_events)
        total_volume = sum(irrigation_volumes)
        
        irrigation_summary = {
            'total_grids_analyzed': len(gridded_data),
            'successful_detections': len(successful_detections),
            'failed_detections': len(failed_detections),
            'grids_with_irrigation': grids_with_irrigation,
            'T_exponential_filter': T_filter,
            'irrigation_statistics': {
                'total_irrigation_events': total_events,
                'total_irrigation_volume': total_volume,
                'mean_events_per_grid': np.mean(irrigation_events),
                'mean_volume_per_grid': np.mean(irrigation_volumes)
            },
            'regional_summary': {
                'irrigation_frequency': grids_with_irrigation / len(successful_detections) if successful_detections else 0,
                'average_irrigation_intensity': total_volume / max(1, grids_with_irrigation)
            }
        }
        
        # Performance statistics
        correlations = [
            result['performance']['correlation']
            for result in successful_detections.values()
            if not np.isnan(result['performance']['correlation'])
        ]
        if correlations:
            irrigation_summary['performance_statistics'] = {
                'mean_correlation': np.mean(correlations),
                'std_correlation': np.std(correlations),
                'min_correlation': np.min(correlations),
                'max_correlation': np.max(correlations)
            }
    else:
        irrigation_summary = {
            'total_grids_analyzed': len(gridded_data),
            'successful_detections': 0,
            'failed_detections': len(failed_detections),
            'grids_with_irrigation': 0
        }
    
    # Save summary files
    with open(output_path / 'phase2_irrigation_results.json', 'w') as f:
        json.dump(convert_numpy_types(irrigation_results), f, indent=2, default=str)
    
    with open(output_path / 'phase2_irrigation_summary.json', 'w') as f:
        json.dump(convert_numpy_types(irrigation_summary), f, indent=2, default=str)
    
    safe_txt_dump(irrigation_summary, str(output_path / 'phase2_irrigation_summary.txt'),
                  "Phase 2: Irrigation Detection Summary")
    
    logger.info(f"\nIRRIGATION DETECTION SUMMARY:")
    if successful_detections:
        logger.info(f"  Success rate: {len(successful_detections)/len(gridded_data):.1%}")
        logger.info(f"  Grids with irrigation: {irrigation_summary['grids_with_irrigation']}/{len(successful_detections)}")
        logger.info(f"  Total irrigation events: {irrigation_summary['irrigation_statistics']['total_irrigation_events']}")
        logger.info(f"  Total irrigation volume: {irrigation_summary['irrigation_statistics']['total_irrigation_volume']:.2f} mm")
    logger.info(f"  Results saved to: {output_path}")
    
    return irrigation_results, irrigation_summary


def save_monthly_grid_estimates(irrigation_results, output_path):
    """Save monthly irrigation estimates to CSV files."""
    logger.info("Saving monthly grid estimates...")
    
    successful_results = {
        k: v for k, v in irrigation_results.items()
        if v['status'] == 'success' and 'monthly_estimates' in v
    }
    
    if not successful_results:
        logger.warning("No successful results with monthly estimates to save")
        return None
    
    all_monthly_estimates = []
    for result in successful_results.values():
        all_monthly_estimates.extend(result['monthly_estimates'])
    
    monthly_df = pd.DataFrame(all_monthly_estimates)
    monthly_df = monthly_df.sort_values(['year', 'month', 'grid_id'])
    
    monthly_file = output_path / 'monthly_irrigation_all_grids.csv'
    monthly_df.to_csv(monthly_file, index=False)
    
    logger.info(f"Monthly estimates saved to {monthly_file}")
    logger.info(f"  Total records: {len(monthly_df)}")
    logger.info(f"  Unique months: {monthly_df['year_month'].nunique()}")
    logger.info(f"  Unique grids: {monthly_df['grid_id'].nunique()}")
    
    return monthly_file


def save_daily_time_series_results(irrigation_results, output_path):
    """Save daily time series results to CSV files."""
    logger.info("Saving daily time series results...")
    
    successful_results = {
        k: v for k, v in irrigation_results.items()
        if v['status'] == 'success' and 'daily_time_series' in v
    }
    
    if not successful_results:
        logger.warning("No successful results with daily time series to save")
        return None
    
    all_daily_data = []
    for result in successful_results.values():
        all_daily_data.append(result['daily_time_series'])
    
    daily_df = pd.concat(all_daily_data, ignore_index=True)
    daily_df = daily_df.sort_values(['date', 'grid_id'])
    
    daily_file = output_path / 'daily_time_series_all_grids.csv'
    daily_df.to_csv(daily_file, index=False)
    
    logger.info(f"Daily time series saved to {daily_file}")
    logger.info(f"  Total records: {len(daily_df)}")
    
    return daily_file


def create_irrigation_heatmap_data(irrigation_results, output_path):
    """Create irrigation heatmap data for visualization."""
    logger.info("Creating irrigation heatmap data...")
    
    successful_results = {
        k: v for k, v in irrigation_results.items()
        if v['status'] == 'success' and 'monthly_estimates' in v
    }
    
    if not successful_results:
        logger.warning("No successful results for heatmap generation")
        return None
    
    heatmap_data = []
    for grid_id, result in successful_results.items():
        coordinates = result.get('coordinates', {})
        monthly_estimates = result.get('monthly_estimates', [])
        
        total_volume = sum(est['irrigation_volume_mm'] for est in monthly_estimates)
        total_events = sum(est['irrigation_events'] for est in monthly_estimates)
        
        heatmap_record = {
            'grid_id': grid_id,
            'latitude': coordinates.get('lat', np.nan),
            'longitude': coordinates.get('lon', np.nan),
            'total_irrigation_volume_mm': total_volume,
            'total_irrigation_events': total_events,
            'avg_monthly_volume_mm': total_volume / max(1, len(monthly_estimates)),
            'avg_monthly_events': total_events / max(1, len(monthly_estimates)),
            'active_months': len([est for est in monthly_estimates if est['irrigation_volume_mm'] > 0])
        }
        heatmap_data.append(heatmap_record)
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_file = output_path / 'irrigation_heatmap_data.csv'
    heatmap_df.to_csv(heatmap_file, index=False)
    
    logger.info(f"Heatmap data saved to {heatmap_file}")
    return heatmap_file


def run_sm2rain_analysis(output_dir=None, T_filter=None, max_workers=8, filter_ndvi=True):
    """
    Run complete SM2RAIN irrigation detection analysis.
    
    Parameters
    ----------
    output_dir : str or Path, optional
        Output directory path
    T_filter : float, optional
        Exponential filter time constant (days)
    max_workers : int
        Maximum parallel workers
    filter_ndvi : bool
        Whether to filter to NDVI dates only
        
    Returns
    -------
    dict
        Final analysis results
    """
    T_filter = T_filter or T_EXPONENTIAL_FILTER
    output_dir = output_dir or Path(__file__).parent / "results" / "sm2rain_results"
    
    logger.info("="*80)
    logger.info("SM2RAIN IRRIGATION DETECTION ANALYSIS")
    logger.info("4-Parameter Model: Z*, Ks, lambda, Kc")
    logger.info(f"Exponential Filter T = {T_filter} days (fixed)")
    logger.info("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data - this now returns gridded data directly
    logger.info("\nLoading data...")
    gridded_data = load_all_data(filter_ndvi=filter_ndvi)
    
    if not gridded_data:
        raise ValueError("No gridded data available for analysis")
    
    logger.info(f"Loaded and prepared data for {len(gridded_data)} grid points")
    
    # Phase 1: Global calibration
    global_params, calibration_summary = phase1_global_calibration(
        gridded_data, output_path, T_filter, max_workers
    )
    
    # Phase 2: Irrigation detection
    irrigation_results, irrigation_summary = phase2_irrigation_detection(
        gridded_data, global_params, output_path, T_filter, max_workers
    )
    
    # Create heatmap data
    heatmap_file = create_irrigation_heatmap_data(irrigation_results, output_path)
    
    # Final results
    final_results = {
        'analysis_timestamp': datetime.now().isoformat(),
        'configuration': {
            'output_directory': str(output_dir),
            'T_exponential_filter': T_filter,
            'max_workers': max_workers,
            'filter_ndvi': filter_ndvi,
            'roi': ROI
        },
        'phase1_calibration': calibration_summary,
        'phase2_irrigation': irrigation_summary,
        'global_parameters': global_params,
        'total_grid_points': len(gridded_data)
    }
    
    # Save final results
    with open(output_path / 'final_results.json', 'w') as f:
        json.dump(convert_numpy_types(final_results), f, indent=2, default=str)
    
    logger.info("\n" + "="*80)
    logger.info("SM2RAIN ANALYSIS COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"Total grid points: {len(gridded_data)}")
    logger.info(f"Phase 1 success rate: {calibration_summary['success_rate']:.1%}")
    if irrigation_summary.get('successful_detections', 0) > 0:
        logger.info(f"Phase 2 success rate: {irrigation_summary['successful_detections']/len(gridded_data):.1%}")
        logger.info(f"Grids with irrigation: {irrigation_summary['grids_with_irrigation']}")
    logger.info(f"Results saved to: {output_path}")
    
    return final_results


def main():
    """Main entry point."""
    # Configuration
    output_dir = Path(__file__).parent / "results" / "sm2rain_4param_results"
    
    # Run analysis with default T value
    # You can change T_EXPONENTIAL_FILTER at the top of this file
    # or pass a different value here
    try:
        results = run_sm2rain_analysis(
            output_dir=str(output_dir),
            T_filter=T_EXPONENTIAL_FILTER,
            max_workers=8,
            filter_ndvi=True
        )
        logger.info("SM2RAIN analysis completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

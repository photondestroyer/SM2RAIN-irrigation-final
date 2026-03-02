"""
Utility functions for SM2RAIN irrigation detection analysis.

This module provides helper functions for:
- JSON serialization with numpy type conversion
- Data validation and consistency checking
- File I/O operations
- Performance metrics computation
- Parameter management

Reference:
    Brocca et al. (2018) SM2RAIN methodology
"""

import numpy as np
import pandas as pd
import json
from typing import Any, Dict, Union, List, Optional
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE CONVERSION UTILITIES
# =============================================================================

def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    
    Parameters
    ----------
    obj : Any
        Object that may contain numpy types
        
    Returns
    -------
    Any
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    else:
        return obj


# =============================================================================
# FILE I/O UTILITIES
# =============================================================================

def safe_txt_dump(data: Dict, filepath: str, title: str = "Data") -> None:
    """
    Safely dump data to a text file, handling various data types.
    
    Parameters
    ----------
    data : dict
        Dictionary to save
    filepath : str
        Output file path
    title : str
        Title for the file
    """
    try:
        clean_data = convert_numpy_types(data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{title}\n")
            f.write("=" * len(title) + "\n\n")
            
            def write_item(key, value, indent=0):
                prefix = "  " * indent
                
                if isinstance(value, dict):
                    f.write(f"{prefix}{key}:\n")
                    for k, v in value.items():
                        write_item(k, v, indent + 1)
                elif isinstance(value, (list, tuple)):
                    f.write(f"{prefix}{key}: [{len(value)} items]\n")
                    if len(value) <= 10:
                        for i, item in enumerate(value[:10]):
                            f.write(f"{prefix}  [{i}]: {item}\n")
                    else:
                        f.write(f"{prefix}  [showing first 5 of {len(value)}]\n")
                        for i, item in enumerate(value[:5]):
                            f.write(f"{prefix}  [{i}]: {item}\n")
                else:
                    f.write(f"{prefix}{key}: {value}\n")
            
            for key, value in clean_data.items():
                write_item(key, value)
                f.write("\n")
                
        logger.info(f"Data successfully saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save data to {filepath}: {e}")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"{title}\n")
                f.write("=" * len(title) + "\n\n")
                f.write(str(data))
        except Exception as e2:
            logger.error(f"Fallback save also failed: {e2}")


def safe_json_dump(data: Dict, filepath: str, **kwargs) -> bool:
    """
    Safely dump data to JSON file with numpy type conversion.
    
    Parameters
    ----------
    data : dict
        Data to save
    filepath : str
        Output file path
    **kwargs
        Additional arguments to json.dump
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        clean_data = convert_numpy_types(data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, **kwargs)
        
        logger.info(f"Data saved to JSON file: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save JSON file {filepath}: {e}")
        return False


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Parameters
    ----------
    size_bytes : int
        Size in bytes
        
    Returns
    -------
    str
        Formatted string (e.g., "1.2 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"


# =============================================================================
# DATA VALIDATION UTILITIES
# =============================================================================

def create_summary_dict(data_dict: Dict, include_stats: bool = True) -> Dict:
    """
    Create a summary dictionary with data statistics.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing data arrays/series
    include_stats : bool
        Whether to include statistical summaries
        
    Returns
    -------
    dict
        Summary dictionary safe for JSON serialization
    """
    summary = {}
    
    for key, data in data_dict.items():
        if isinstance(data, (pd.Series, np.ndarray)):
            summary[key] = {
                'type': 'array',
                'length': len(data),
                'dtype': str(data.dtype) if hasattr(data, 'dtype') else 'unknown'
            }
            
            if include_stats and np.issubdtype(
                data.dtype if hasattr(data, 'dtype') else np.float64, np.number
            ):
                valid_data = data[~pd.isna(data)]
                summary[key].update({
                    'missing_count': int(len(data) - len(valid_data)),
                    'missing_percentage': float((len(data) - len(valid_data)) / len(data) * 100),
                    'min': float(np.min(valid_data)) if len(valid_data) > 0 else None,
                    'max': float(np.max(valid_data)) if len(valid_data) > 0 else None,
                    'mean': float(np.mean(valid_data)) if len(valid_data) > 0 else None,
                    'std': float(np.std(valid_data)) if len(valid_data) > 0 else None
                })
        else:
            summary[key] = {
                'type': type(data).__name__,
                'value': convert_numpy_types(data)
            }
    
    return summary


def validate_data_consistency(data_dict: Dict) -> Dict:
    """
    Validate consistency of data arrays (same length, aligned indices, etc.).
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing data arrays/series
        
    Returns
    -------
    dict
        Dictionary with validation results
    """
    validation = {
        'consistent_lengths': True,
        'consistent_indices': True,
        'lengths': {},
        'issues': []
    }
    
    # Check lengths
    lengths = {}
    for key, data in data_dict.items():
        if hasattr(data, '__len__'):
            lengths[key] = len(data)
    
    validation['lengths'] = lengths
    
    if len(set(lengths.values())) > 1:
        validation['consistent_lengths'] = False
        validation['issues'].append(f"Inconsistent lengths: {lengths}")
    
    # Check indices for pandas objects
    indices = {}
    for key, data in data_dict.items():
        if isinstance(data, pd.Series):
            indices[key] = data.index
    
    if len(indices) > 1:
        first_index = list(indices.values())[0]
        for key, index in indices.items():
            if not first_index.equals(index):
                validation['consistent_indices'] = False
                validation['issues'].append(f"Index mismatch for {key}")
                break
    
    return validation


def validate_sm_data(sm_data: pd.Series) -> Dict:
    """
    Validate soil moisture data quality.
    
    Parameters
    ----------
    sm_data : pd.Series
        Soil moisture time series
        
    Returns
    -------
    dict
        Validation results
    """
    validation = {
        'valid': True,
        'issues': [],
        'statistics': {}
    }
    
    # Check for missing data
    missing_pct = sm_data.isna().sum() / len(sm_data) * 100
    validation['statistics']['missing_pct'] = missing_pct
    
    if missing_pct > 50:
        validation['valid'] = False
        validation['issues'].append(f"High missing data: {missing_pct:.1f}%")
    
    # Check value range
    valid_data = sm_data.dropna()
    if len(valid_data) > 0:
        sm_min = valid_data.min()
        sm_max = valid_data.max()
        validation['statistics']['min'] = sm_min
        validation['statistics']['max'] = sm_max
        
        # Soil moisture should typically be 0-0.6 for volumetric
        if sm_min < 0:
            validation['issues'].append(f"Negative soil moisture values: {sm_min:.4f}")
        if sm_max > 1.0:
            validation['issues'].append(f"Soil moisture > 1.0: {sm_max:.4f}")
    else:
        validation['valid'] = False
        validation['issues'].append("No valid soil moisture data")
    
    return validation


# =============================================================================
# IRRIGATION DETECTION UTILITIES
# =============================================================================

def detect_irrigation_events(
    total_water: pd.Series,
    reference_rain: pd.Series,
    threshold: float = 2.0
) -> Dict:
    """
    Detect irrigation events based on water balance residual.
    
    Parameters
    ----------
    total_water : pd.Series
        Total water input from SM2RAIN
    reference_rain : pd.Series
        Reference rainfall data
    threshold : float
        Minimum irrigation threshold [mm/day]
    
    Returns
    -------
    dict
        Dictionary with irrigation detection results
    """
    try:
        # Align series to common index
        common_index = total_water.index.intersection(reference_rain.index)
        total_water_aligned = total_water.reindex(common_index)
        reference_rain_aligned = reference_rain.reindex(common_index)
        
        # Calculate potential irrigation as difference
        irrigation_est = np.maximum(0, total_water_aligned - reference_rain_aligned)
        
        # Flag irrigation events above threshold
        irrigation_flags = irrigation_est > threshold
        
        # Calculate irrigation amounts only for flagged events
        irrigation_amount = np.where(irrigation_flags, irrigation_est, 0)
        
        # Summary statistics
        irrigation_frequency = int(np.sum(irrigation_flags))
        total_irrigation_volume = float(np.sum(irrigation_amount))
        
        # Get irrigation dates
        irrigation_dates = common_index[irrigation_flags].strftime('%Y-%m-%d').tolist()
        
        return {
            'irrigation_flags': irrigation_flags,
            'irrigation_amount': irrigation_amount,
            'irrigation_frequency': irrigation_frequency,
            'total_irrigation_volume': total_irrigation_volume,
            'irrigation_dates': irrigation_dates
        }
        
    except Exception as e:
        logger.error(f"Error in irrigation detection: {e}")
        n_points = len(reference_rain) if hasattr(reference_rain, '__len__') else 100
        return {
            'irrigation_flags': np.array([False] * n_points),
            'irrigation_amount': np.array([0.0] * n_points),
            'irrigation_frequency': 0,
            'total_irrigation_volume': 0.0,
            'irrigation_dates': []
        }


def compute_performance_metrics(
    estimated: pd.Series,
    observed: pd.Series
) -> Dict:
    """
    Compute performance metrics between estimated and observed values.
    
    Parameters
    ----------
    estimated : pd.Series
        Estimated values (e.g., SM2RAIN rainfall)
    observed : pd.Series
        Observed values (e.g., reference rainfall)
        
    Returns
    -------
    dict
        Dictionary with performance metrics
    """
    try:
        # Align series
        common_index = estimated.index.intersection(observed.index)
        est = estimated.reindex(common_index).values
        obs = observed.reindex(common_index).values
        
        # Remove NaN values
        valid_mask = ~(np.isnan(est) | np.isnan(obs))
        est = est[valid_mask]
        obs = obs[valid_mask]
        
        if len(est) < 10:
            return {
                'correlation': np.nan,
                'rmse': np.nan,
                'nse': np.nan,
                'bias': np.nan,
                'n_valid': len(est)
            }
        
        # Correlation
        correlation = np.corrcoef(est, obs)[0, 1]
        
        # RMSE
        rmse = np.sqrt(np.mean((est - obs)**2))
        
        # Nash-Sutcliffe Efficiency
        mean_obs = np.mean(obs)
        ss_res = np.sum((obs - est)**2)
        ss_tot = np.sum((obs - mean_obs)**2)
        nse = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        
        # Bias
        bias = np.mean(est - obs)
        
        return {
            'correlation': float(correlation),
            'rmse': float(rmse),
            'nse': float(nse),
            'bias': float(bias),
            'n_valid': int(len(est))
        }
        
    except Exception as e:
        logger.error(f"Error computing performance metrics: {e}")
        return {
            'correlation': np.nan,
            'rmse': np.nan,
            'nse': np.nan,
            'bias': np.nan,
            'n_valid': 0
        }


# =============================================================================
# PARAMETER MANAGEMENT
# =============================================================================

def get_default_parameter_bounds() -> Dict:
    """
    Get default parameter bounds for the 4-parameter SM2RAIN model.
    
    Returns
    -------
    dict
        Parameter bounds configuration for (Zstar, Ks, lam, Kc)
    """
    return {
        'Zstar': (10.0, 200.0),     # Effective soil depth [mm]
        'Ks': (0.1, 50.0),          # Saturated hydraulic conductivity [mm/day]
        'lam': (0.1, 5.0),          # Shape parameter for drainage [-]
        'Kc': (0.5, 2.0)            # Crop coefficient [-]
    }


def get_default_parameters() -> Dict:
    """
    Get default parameter values for the 4-parameter SM2RAIN model.
    
    Returns
    -------
    dict
        Default parameter values
    """
    return {
        'Zstar': 50.0,   # Effective soil depth [mm]
        'Ks': 10.0,      # Saturated hydraulic conductivity [mm/day]
        'lam': 1.0,      # Shape parameter for drainage [-]
        'Kc': 1.0        # Crop coefficient [-]
    }


def validate_parameters(params: Dict) -> Dict:
    """
    Validate SM2RAIN model parameters.
    
    Parameters
    ----------
    params : dict
        Parameters to validate
        
    Returns
    -------
    dict
        Validation results
    """
    validation = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    bounds = get_default_parameter_bounds()
    required_params = ['Zstar', 'Ks', 'lam', 'Kc']
    
    # Check required parameters
    for param in required_params:
        if param not in params:
            validation['errors'].append(f"Missing required parameter: {param}")
            validation['valid'] = False
    
    # Check parameter ranges
    for param, value in params.items():
        if param in bounds:
            min_val, max_val = bounds[param]
            if not (min_val <= value <= max_val):
                validation['warnings'].append(
                    f"{param} = {value} outside typical range [{min_val}, {max_val}]"
                )
    
    # Check physical consistency
    if 'Zstar' in params and params['Zstar'] < 10:
        validation['warnings'].append("Very low Zstar - check soil depth assumption")
    
    if 'Kc' in params and (params['Kc'] < 0.5 or params['Kc'] > 2.0):
        validation['warnings'].append("Kc outside typical crop coefficient range")
    
    return validation


def save_parameters(params: Dict, output_file: str) -> None:
    """
    Save SM2RAIN parameters with metadata.
    
    Parameters
    ----------
    params : dict
        Parameters to save
    output_file : str
        Output file path
    """
    try:
        params_with_metadata = {
            'parameters': convert_numpy_types(params),
            'parameter_descriptions': {
                'Zstar': 'Effective soil depth [mm]',
                'Ks': 'Saturated hydraulic conductivity [mm/day]',
                'lam': 'Shape parameter for drainage [-]',
                'Kc': 'Crop coefficient [-]',
                'T': 'Exponential filter time constant [days] (fixed)'
            },
            'model': 'SM2RAIN 4-parameter model',
            'reference': 'Brocca et al. (2018)',
            'save_timestamp': datetime.now().isoformat()
        }
        
        # Save as JSON
        json_file = str(output_file).replace('.txt', '.json')
        with open(json_file, 'w') as f:
            json.dump(params_with_metadata, f, indent=2)
        
        # Save as text
        safe_txt_dump(params_with_metadata, str(output_file),
                      "SM2RAIN Parameters (4-Parameter Model)")
        
        logger.info(f"Parameters saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save parameters: {e}")


def load_parameters(input_file: str) -> Dict:
    """
    Load SM2RAIN parameters from file.
    
    Parameters
    ----------
    input_file : str
        Input file path
        
    Returns
    -------
    dict
        Loaded parameters
    """
    try:
        input_path = Path(input_file)
        
        # Try JSON first
        json_file = str(input_path).replace('.txt', '.json')
        if Path(json_file).exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
                return data.get('parameters', data)
        
        elif input_path.exists():
            logger.warning(f"Loading from text file - limited functionality: {input_file}")
            return get_default_parameters()
        
        else:
            logger.warning(f"Parameters file not found: {input_file}")
            return get_default_parameters()
            
    except Exception as e:
        logger.error(f"Failed to load parameters: {e}")
        return get_default_parameters()


# =============================================================================
# PROCESSING METADATA
# =============================================================================

def create_processing_metadata(
    input_files: List[str],
    output_dir: str,
    config: Optional[Dict] = None
) -> Dict:
    """
    Create metadata for a processing run.
    
    Parameters
    ----------
    input_files : list
        List of input file paths
    output_dir : str
        Output directory path
    config : dict, optional
        Configuration dictionary
        
    Returns
    -------
    dict
        Metadata dictionary
    """
    metadata = {
        'processing_timestamp': datetime.now().isoformat(),
        'input_files': {},
        'output_directory': str(output_dir),
        'configuration': convert_numpy_types(config) if config else None
    }
    
    for file_path in input_files:
        if Path(file_path).exists():
            file_stats = Path(file_path).stat()
            metadata['input_files'][str(file_path)] = {
                'size': format_file_size(file_stats.st_size),
                'size_bytes': int(file_stats.st_size),
                'modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                'exists': True
            }
        else:
            metadata['input_files'][str(file_path)] = {
                'exists': False
            }
    
    return metadata


# =============================================================================
# SUMMARY REPORT GENERATION
# =============================================================================

def generate_summary_report(
    calibration_results: Dict,
    irrigation_results: Dict,
    global_params: Dict
) -> str:
    """
    Generate a comprehensive summary report.
    
    Parameters
    ----------
    calibration_results : dict
        Calibration phase results
    irrigation_results : dict
        Irrigation detection results
    global_params : dict
        Global optimized parameters
        
    Returns
    -------
    str
        Formatted summary report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("SM2RAIN IRRIGATION DETECTION SUMMARY REPORT")
    lines.append("4-Parameter Model: Z*, Ks, lambda, Kc")
    lines.append("=" * 80)
    lines.append("")
    
    # Global parameters
    lines.append("CALIBRATED PARAMETERS:")
    lines.append("-" * 40)
    lines.append(f"  Z* (effective soil depth):        {global_params.get('Zstar', 'N/A'):.2f} mm")
    lines.append(f"  Ks (sat. hydraulic conductivity): {global_params.get('Ks', 'N/A'):.4f} mm/day")
    lines.append(f"  lambda (shape parameter):         {global_params.get('lam', 'N/A'):.4f}")
    lines.append(f"  Kc (crop coefficient):            {global_params.get('Kc', 'N/A'):.4f}")
    if 'T' in global_params:
        lines.append(f"  T (exp. filter constant, fixed):  {global_params['T']:.1f} days")
    lines.append("")
    
    # Calibration summary
    if calibration_results:
        lines.append("CALIBRATION SUMMARY:")
        lines.append("-" * 40)
        lines.append(f"  Total grid points:     {calibration_results.get('total_grids', 'N/A')}")
        lines.append(f"  Successful:            {calibration_results.get('successful_calibrations', 'N/A')}")
        lines.append(f"  Failed:                {calibration_results.get('failed_calibrations', 'N/A')}")
        lines.append(f"  Success rate:          {calibration_results.get('success_rate', 0):.1%}")
        lines.append(f"  Best RMSE:             {calibration_results.get('best_rmse', 'N/A'):.4f}")
        lines.append("")
    
    # Irrigation summary
    if irrigation_results:
        lines.append("IRRIGATION DETECTION SUMMARY:")
        lines.append("-" * 40)
        lines.append(f"  Grids analyzed:        {irrigation_results.get('total_grids_analyzed', 'N/A')}")
        lines.append(f"  Grids with irrigation: {irrigation_results.get('grids_with_irrigation', 'N/A')}")
        
        if 'irrigation_statistics' in irrigation_results:
            stats = irrigation_results['irrigation_statistics']
            lines.append(f"  Total events:          {stats.get('total_irrigation_events', 'N/A')}")
            lines.append(f"  Total volume:          {stats.get('total_irrigation_volume', 0):.2f} mm")
        lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


def create_parameter_comparison_report(calibration_results: Dict) -> Dict:
    """
    Create a report comparing parameters across all grid points.
    
    Parameters
    ----------
    calibration_results : dict
        Results from all grid point calibrations
        
    Returns
    -------
    dict
        Comparison statistics
    """
    successful_results = {
        k: v for k, v in calibration_results.items()
        if v.get('status') == 'success'
    }
    
    if not successful_results:
        return {'error': 'No successful calibrations to compare'}
    
    param_names = ['Zstar', 'Ks', 'lam', 'Kc']
    param_data = {param: [] for param in param_names}
    
    for result in successful_results.values():
        if 'parameters' in result:
            for param in param_names:
                if param in result['parameters']:
                    param_data[param].append(result['parameters'][param])
    
    comparison_stats = {}
    for param, values in param_data.items():
        if values:
            comparison_stats[param] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'cv': float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0,
                'n_samples': len(values)
            }
    
    # Identify high variability parameters
    high_variability = [
        param for param, stats in comparison_stats.items()
        if stats['cv'] > 0.5
    ]
    
    return {
        'parameter_statistics': comparison_stats,
        'high_variability_parameters': high_variability,
        'total_successful_calibrations': len(successful_results),
        'parameter_names': param_names
    }

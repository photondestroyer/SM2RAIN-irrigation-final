"""
Data Preprocessor for SM2RAIN Irrigation Detection Analysis

This module handles loading and preprocessing of:
- SMAP Soil Moisture data (CSV format)
- GPM IMERG Precipitation data (NetCDF format)
- agERA5 Temperature data (NetCDF format)

It supports both single-point and gridded analysis across the study region.
"""

import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc
import os
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple, List, Union
from datetime import datetime
import warnings
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# =============================================================================
# DEFAULT FILE PATHS
# =============================================================================
DEFAULT_TEMP_PATH = r"G:\sm2rain-irrigation\agERA5_ROI_main\combined\agERA5_24_hour_mean_combined.nc"
DEFAULT_PRECIP_PATH = r"G:\sm2rain-irrigation\GPM_IMERG_Daily_ROI_main.nc"
DEFAULT_SM_PATH = r"G:\sm2rain-irrigation\data\SMAP_data\SMAP_SPL3SMP_E_30N-31N_75E-76E_2017-2021.csv"
DEFAULT_NDVI_DATES_PATH = r"G:\sm2rain-irrigation\ndvi_dates_mean.txt"


# =============================================================================
# NDVI DATES HANDLING
# =============================================================================

def load_ndvi_dates(ndvi_path: str = None) -> pd.DatetimeIndex:
    """
    Load NDVI calibration dates from file.
    
    The NDVI dates represent periods during the irrigation season 
    (typically April-July for Kharif and October-December for Rabi crops).
    
    Args:
        ndvi_path: Path to NDVI dates file
        
    Returns:
        DatetimeIndex with calibration dates
    """
    if ndvi_path is None:
        ndvi_path = DEFAULT_NDVI_DATES_PATH
    
    if not os.path.exists(ndvi_path):
        logger.warning(f"NDVI dates file not found: {ndvi_path}")
        return None
    
    try:
        with open(ndvi_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Parse dates (format: YYYY-MM-DDTHH:MM:SS.000000000)
        dates = pd.to_datetime([d.split('T')[0] for d in lines])
        
        logger.info(f"Loaded {len(dates)} NDVI dates from {ndvi_path}")
        logger.info(f"Date range: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
        
        return dates
        
    except Exception as e:
        logger.error(f"Error loading NDVI dates: {e}")
        return None


# =============================================================================
# NETCDF DATA LOADING
# =============================================================================

class NetCDFDataLoader:
    """
    Load and process NetCDF data files (precipitation and temperature).
    """
    
    def __init__(self, file_path: str, var_name: str = None):
        """
        Initialize NetCDF loader.
        
        Args:
            file_path: Path to NetCDF file
            var_name: Variable name to extract (auto-detected if None)
        """
        self.file_path = file_path
        self.var_name = var_name
        self.data = None
        self.lat = None
        self.lon = None
        self.time = None
        
        self._load_file()
    
    def _load_file(self):
        """Load NetCDF file and extract metadata."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"NetCDF file not found: {self.file_path}")
        
        ds = nc.Dataset(self.file_path, 'r')
        
        # Get coordinates
        self.lat = ds.variables['lat'][:]
        self.lon = ds.variables['lon'][:]
        
        # Get time and convert to datetime
        time_var = ds.variables['time']
        time_units = time_var.units if hasattr(time_var, 'units') else 'days since 1970-01-01'
        
        # Clean up time units string (remove any brackets or extra characters)
        if isinstance(time_units, (list, tuple)):
            time_units = time_units[0]
        time_units = str(time_units).strip("[]'\"")
        
        self.time = nc.num2date(time_var[:], units=time_units)
        self.time = pd.to_datetime([str(t)[:10] for t in self.time])
        
        # Auto-detect variable name
        if self.var_name is None:
            data_vars = [v for v in ds.variables.keys() if v not in ['lat', 'lon', 'time']]
            self.var_name = data_vars[0]
        
        # Load data
        self.data = ds.variables[self.var_name][:]
        
        logger.info(f"Loaded {self.var_name} from {self.file_path}")
        logger.info(f"Shape: {self.data.shape}")
        logger.info(f"Lat range: {self.lat.min():.4f} to {self.lat.max():.4f}")
        logger.info(f"Lon range: {self.lon.min():.4f} to {self.lon.max():.4f}")
        logger.info(f"Time range: {self.time.min()} to {self.time.max()}")
        
        ds.close()
    
    def get_grid_points(self) -> pd.DataFrame:
        """
        Get all grid point coordinates.
        
        Returns:
            DataFrame with Latitude and Longitude columns
        """
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
        
        return pd.DataFrame({
            'Latitude': lat_grid.flatten(),
            'Longitude': lon_grid.flatten()
        })
    
    def extract_point_timeseries(self, lat: float, lon: float, tolerance: float = 0.1) -> pd.Series:
        """
        Extract time series for a specific grid point.
        
        Args:
            lat: Target latitude
            lon: Target longitude
            tolerance: Spatial tolerance for matching
            
        Returns:
            Time series as pandas Series
        """
        # Find nearest indices
        lat_idx = np.argmin(np.abs(self.lat - lat))
        lon_idx = np.argmin(np.abs(self.lon - lon))
        
        # Check if within tolerance
        lat_diff = np.abs(self.lat[lat_idx] - lat)
        lon_diff = np.abs(self.lon[lon_idx] - lon)
        
        if lat_diff > tolerance or lon_diff > tolerance:
            logger.warning(f"Nearest point ({self.lat[lat_idx]:.4f}, {self.lon[lon_idx]:.4f}) "
                         f"exceeds tolerance ({lat_diff:.4f}, {lon_diff:.4f})")
        
        # Extract time series
        values = self.data[:, lat_idx, lon_idx]
        
        # Handle masked arrays
        if hasattr(values, 'filled'):
            values = values.filled(np.nan)
        
        ts = pd.Series(values, index=self.time, name=self.var_name)
        
        logger.debug(f"Extracted time series at ({self.lat[lat_idx]:.4f}, {self.lon[lon_idx]:.4f})")
        
        return ts
    
    def extract_all_points(self) -> Dict[Tuple[float, float], pd.Series]:
        """
        Extract time series for all grid points.
        
        Returns:
            Dictionary mapping (lat, lon) tuples to time series
        """
        result = {}
        
        for i, lat in enumerate(self.lat):
            for j, lon in enumerate(self.lon):
                values = self.data[:, i, j]
                if hasattr(values, 'filled'):
                    values = values.filled(np.nan)
                
                ts = pd.Series(values, index=self.time, name=self.var_name)
                result[(float(lat), float(lon))] = ts
        
        logger.info(f"Extracted {len(result)} time series")
        
        return result


# =============================================================================
# SMAP CSV DATA LOADING
# =============================================================================

class SMAPDataLoader:
    """
    Load and process SMAP soil moisture data from CSV file.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize SMAP loader.
        
        Args:
            file_path: Path to SMAP CSV file
        """
        self.file_path = file_path
        self.data = None
        self.grid_points = None
        
        self._load_file()
    
    def _load_file(self):
        """Load SMAP CSV file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"SMAP CSV file not found: {self.file_path}")
        
        self.data = pd.read_csv(self.file_path)
        
        # Parse date - handle DD-MM-YYYY format
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%d-%m-%Y', dayfirst=True)
        
        # Get unique grid points
        self.grid_points = self.data[['Latitude', 'Longitude']].drop_duplicates().reset_index(drop=True)
        
        logger.info(f"Loaded SMAP data from {self.file_path}")
        logger.info(f"Records: {len(self.data)}")
        logger.info(f"Unique grid points: {len(self.grid_points)}")
        logger.info(f"Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        logger.info(f"Lat range: {self.grid_points['Latitude'].min():.4f} to {self.grid_points['Latitude'].max():.4f}")
        logger.info(f"Lon range: {self.grid_points['Longitude'].min():.4f} to {self.grid_points['Longitude'].max():.4f}")
    
    def get_grid_points(self) -> pd.DataFrame:
        """
        Get all unique grid point coordinates.
        
        Returns:
            DataFrame with Latitude and Longitude columns
        """
        return self.grid_points.copy()
    
    def extract_point_timeseries(self, lat: float, lon: float, tolerance: float = 0.1) -> pd.Series:
        """
        Extract time series for a specific grid point.
        
        Args:
            lat: Target latitude
            lon: Target longitude
            tolerance: Spatial tolerance for matching
            
        Returns:
            Time series as pandas Series
        """
        # Find nearest grid point
        distances = np.sqrt(
            (self.grid_points['Latitude'] - lat)**2 + 
            (self.grid_points['Longitude'] - lon)**2
        )
        
        nearest_idx = distances.idxmin()
        nearest_lat = self.grid_points.loc[nearest_idx, 'Latitude']
        nearest_lon = self.grid_points.loc[nearest_idx, 'Longitude']
        
        if distances[nearest_idx] > tolerance * np.sqrt(2):
            logger.warning(f"Nearest point ({nearest_lat:.4f}, {nearest_lon:.4f}) "
                         f"exceeds tolerance (distance: {distances[nearest_idx]:.4f})")
        
        # Filter data for this location
        mask = (self.data['Latitude'] == nearest_lat) & (self.data['Longitude'] == nearest_lon)
        point_data = self.data[mask].copy()
        
        # Create time series
        ts = pd.Series(
            data=point_data['Soil_Moisture'].values,
            index=pd.to_datetime(point_data['Date']),
            name='soil_moisture'
        )
        
        # Sort and remove duplicates
        ts = ts.sort_index()
        ts = ts[~ts.index.duplicated(keep='first')]
        
        logger.debug(f"Extracted SMAP time series at ({nearest_lat:.4f}, {nearest_lon:.4f}): {len(ts)} points")
        
        return ts
    
    def extract_all_points(self) -> Dict[Tuple[float, float], pd.Series]:
        """
        Extract time series for all grid points.
        
        Returns:
            Dictionary mapping (lat, lon) tuples to time series
        """
        result = {}
        
        for _, row in self.grid_points.iterrows():
            lat, lon = row['Latitude'], row['Longitude']
            ts = self.extract_point_timeseries(lat, lon)
            result[(lat, lon)] = ts
        
        logger.info(f"Extracted {len(result)} time series")
        
        return result


# =============================================================================
# COMMON GRID FINDER
# =============================================================================

class CommonGridFinder:
    """
    Find common grid points across multiple datasets with spatial tolerance.
    """
    
    def __init__(self, tolerance: float = 0.1):
        """
        Initialize common grid finder.
        
        Args:
            tolerance: Spatial tolerance for matching (in degrees)
        """
        self.tolerance = tolerance
        self.common_points = None
        self.matched_coordinates = None
    
    def find_common_grids(
        self, 
        sm_grid: pd.DataFrame, 
        precip_grid: pd.DataFrame, 
        temp_grid: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Find common grid points across all three datasets.
        
        Uses KD-tree for efficient nearest neighbor matching.
        
        Args:
            sm_grid: SMAP grid points (Latitude, Longitude)
            precip_grid: Precipitation grid points (Latitude, Longitude)
            temp_grid: Temperature grid points (Latitude, Longitude)
            
        Returns:
            DataFrame with matched coordinates
        """
        logger.info("Finding common grid points across datasets...")
        logger.info(f"SMAP: {len(sm_grid)} points")
        logger.info(f"Precipitation: {len(precip_grid)} points")
        logger.info(f"Temperature: {len(temp_grid)} points")
        logger.info(f"Using tolerance: {self.tolerance} degrees")
        
        # Step 1: Match SMAP with Precipitation
        precip_tree = cKDTree(precip_grid[['Latitude', 'Longitude']].values)
        sm_coords = sm_grid[['Latitude', 'Longitude']].values
        
        precip_distances, precip_indices = precip_tree.query(sm_coords)
        precip_mask = precip_distances <= self.tolerance * np.sqrt(2)
        
        # Step 2: Match those with Temperature
        temp_tree = cKDTree(temp_grid[['Latitude', 'Longitude']].values)
        
        matched_coords = []
        
        for i, (sm_lat, sm_lon) in enumerate(sm_coords):
            if precip_mask[i]:
                precip_lat = precip_grid.iloc[precip_indices[i]]['Latitude']
                precip_lon = precip_grid.iloc[precip_indices[i]]['Longitude']
                
                # Find nearest temperature point
                temp_dist, temp_idx = temp_tree.query([[sm_lat, sm_lon]])
                
                if temp_dist[0] <= self.tolerance * np.sqrt(2):
                    temp_lat = temp_grid.iloc[temp_idx[0]]['Latitude']
                    temp_lon = temp_grid.iloc[temp_idx[0]]['Longitude']
                    
                    matched_coords.append({
                        'sm_lat': sm_lat,
                        'sm_lon': sm_lon,
                        'precip_lat': precip_lat,
                        'precip_lon': precip_lon,
                        'temp_lat': temp_lat,
                        'temp_lon': temp_lon,
                        'Latitude': sm_lat,  # Use SMAP as reference
                        'Longitude': sm_lon
                    })
        
        self.matched_coordinates = pd.DataFrame(matched_coords)
        
        logger.info(f"Found {len(self.matched_coordinates)} common grid points")
        
        if len(self.matched_coordinates) > 0:
            logger.info(f"Latitude range: {self.matched_coordinates['Latitude'].min():.4f} "
                       f"to {self.matched_coordinates['Latitude'].max():.4f}")
            logger.info(f"Longitude range: {self.matched_coordinates['Longitude'].min():.4f} "
                       f"to {self.matched_coordinates['Longitude'].max():.4f}")
        
        return self.matched_coordinates


# =============================================================================
# GRIDDED DATA PROCESSOR
# =============================================================================

class GriddedDataProcessor:
    """
    Process gridded data from all three sources for SM2RAIN analysis.
    
    Handles:
    - Loading data from NetCDF (precipitation, temperature) and CSV (SMAP)
    - Finding common grid points
    - Extracting aligned time series
    - Filtering to NDVI calibration dates
    """
    
    def __init__(
        self,
        sm_path: str = None,
        precip_path: str = None,
        temp_path: str = None,
        ndvi_dates_path: str = None,
        spatial_tolerance: float = 0.1
    ):
        """
        Initialize gridded data processor.
        
        Args:
            sm_path: Path to SMAP soil moisture CSV
            precip_path: Path to precipitation NetCDF
            temp_path: Path to temperature NetCDF
            ndvi_dates_path: Path to NDVI dates file
            spatial_tolerance: Tolerance for spatial matching
        """
        self.sm_path = sm_path or DEFAULT_SM_PATH
        self.precip_path = precip_path or DEFAULT_PRECIP_PATH
        self.temp_path = temp_path or DEFAULT_TEMP_PATH
        self.ndvi_dates_path = ndvi_dates_path or DEFAULT_NDVI_DATES_PATH
        self.spatial_tolerance = spatial_tolerance
        
        # Data loaders
        self.sm_loader = None
        self.precip_loader = None
        self.temp_loader = None
        
        # Common grids
        self.common_grids = None
        self.ndvi_dates = None
        
        self._initialize_loaders()
    
    def _initialize_loaders(self):
        """Initialize data loaders."""
        logger.info("Initializing data loaders...")
        
        self.sm_loader = SMAPDataLoader(self.sm_path)
        self.precip_loader = NetCDFDataLoader(self.precip_path, 'precipitation')
        self.temp_loader = NetCDFDataLoader(self.temp_path, 'Temperature_Air_2m_Mean_24h')
        
        # Load NDVI dates
        self.ndvi_dates = load_ndvi_dates(self.ndvi_dates_path)
        
        logger.info("Data loaders initialized")
    
    def find_common_grids(self) -> pd.DataFrame:
        """
        Find common grid points across all datasets.
        
        Returns:
            DataFrame with common grid coordinates
        """
        sm_grid = self.sm_loader.get_grid_points()
        precip_grid = self.precip_loader.get_grid_points()
        temp_grid = self.temp_loader.get_grid_points()
        
        grid_finder = CommonGridFinder(tolerance=self.spatial_tolerance)
        self.common_grids = grid_finder.find_common_grids(sm_grid, precip_grid, temp_grid)
        
        return self.common_grids
    
    def load_point_data(
        self, 
        lat: float, 
        lon: float,
        filter_ndvi: bool = True
    ) -> Dict[str, pd.Series]:
        """
        Load aligned data for a specific grid point.
        
        Args:
            lat: Latitude
            lon: Longitude
            filter_ndvi: Whether to filter to NDVI dates
            
        Returns:
            Dictionary with 'sm', 'rain', 'Ta' time series
        """
        # Extract from each dataset
        sm_ts = self.sm_loader.extract_point_timeseries(lat, lon, self.spatial_tolerance)
        rain_ts = self.precip_loader.extract_point_timeseries(lat, lon, self.spatial_tolerance)
        temp_ts = self.temp_loader.extract_point_timeseries(lat, lon, self.spatial_tolerance)
        
        # Remove duplicate dates by taking mean of duplicates
        if sm_ts.index.duplicated().any():
            sm_ts = sm_ts.groupby(sm_ts.index).mean()
        if rain_ts.index.duplicated().any():
            rain_ts = rain_ts.groupby(rain_ts.index).mean()
        if temp_ts.index.duplicated().any():
            temp_ts = temp_ts.groupby(temp_ts.index).mean()
        
        # Convert temperature from Kelvin to Celsius if needed
        if temp_ts.mean() > 200:
            temp_ts = temp_ts - 273.15
            logger.debug("Converted temperature from Kelvin to Celsius")
        
        # Find common dates
        common_dates = sm_ts.index.intersection(rain_ts.index).intersection(temp_ts.index)
        
        data = {
            'sm': sm_ts.reindex(common_dates),
            'rain': rain_ts.reindex(common_dates),
            'Ta': temp_ts.reindex(common_dates)
        }
        
        # logger.info(f"Loaded data for ({lat:.4f}, {lon:.4f}): {len(common_dates)} common dates")
        
        # Filter to NDVI dates if requested
        if filter_ndvi and self.ndvi_dates is not None:
            data = self._filter_to_ndvi(data)
        
        return data
    
    def _filter_to_ndvi(self, data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Filter data to NDVI calibration dates."""
        ndvi_dates_normalized = pd.to_datetime(self.ndvi_dates).normalize()
        
        filtered_data = {}
        for key, series in data.items():
            series_dates = pd.to_datetime(series.index).normalize()
            mask = series_dates.isin(ndvi_dates_normalized)
            filtered_data[key] = series[mask]
        
        # Find common filtered dates
        common_filtered = filtered_data['sm'].index
        for key in ['rain', 'Ta']:
            common_filtered = common_filtered.intersection(filtered_data[key].index)
        
        for key in filtered_data:
            filtered_data[key] = filtered_data[key].reindex(common_filtered)
        
        # logger.info(f"Filtered to {len(common_filtered)} NDVI dates")
        
        return filtered_data
    
    def prepare_gridded_data(
        self,
        filter_ndvi: bool = True,
        min_data_points: int = 30
    ) -> Dict[int, Dict]:
        """
        Prepare data for all common grid points.
        
        Args:
            filter_ndvi: Whether to filter to NDVI dates
            min_data_points: Minimum required data points per grid
            
        Returns:
            Dictionary mapping grid_id to data dictionaries
        """
        if self.common_grids is None:
            self.find_common_grids()
        
        if len(self.common_grids) == 0:
            logger.error("No common grid points found")
            return {}
        
        logger.info(f"Preparing data for {len(self.common_grids)} grid points...")
        
        gridded_data = {}
        
        for idx, row in self.common_grids.iterrows():
            lat = row['Latitude']
            lon = row['Longitude']
            
            try:
                data = self.load_point_data(lat, lon, filter_ndvi)
                
                # Check data quality
                n_points = len(data['sm'])
                if n_points >= min_data_points:
                    gridded_data[idx] = {
                        'sm': data['sm'],
                        'rain': data['rain'],
                        'Ta': data['Ta'],
                        'coordinates': {
                            'lat': lat,
                            'lon': lon,
                            'lat_center': lat,
                            'lon_center': lon
                        },
                        'grid_id': idx
                    }
                else:
                    logger.warning(f"Grid {idx} at ({lat:.4f}, {lon:.4f}): "
                                 f"insufficient data ({n_points} < {min_data_points})")
                    
            except Exception as e:
                logger.warning(f"Error processing grid {idx} at ({lat:.4f}, {lon:.4f}): {e}")
        
        logger.info(f"Successfully prepared data for {len(gridded_data)} grid points")
        
        return gridded_data


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_all_data(
    sm_path: str = None,
    precip_path: str = None,
    temp_path: str = None
) -> Dict[str, Union[SMAPDataLoader, NetCDFDataLoader]]:
    """
    Load all data sources.
    
    Args:
        sm_path: Path to SMAP CSV
        precip_path: Path to precipitation NetCDF
        temp_path: Path to temperature NetCDF
        
    Returns:
        Dictionary with data loaders
    """
    return {
        'sm': SMAPDataLoader(sm_path or DEFAULT_SM_PATH),
        'precip': NetCDFDataLoader(precip_path or DEFAULT_PRECIP_PATH, 'precipitation'),
        'temp': NetCDFDataLoader(temp_path or DEFAULT_TEMP_PATH, 'Temperature_Air_2m_Mean_24h')
    }


def get_study_region_info() -> Dict:
    """
    Get information about the study region from the data files.
    
    Returns:
        Dictionary with region information
    """
    processor = GriddedDataProcessor()
    common_grids = processor.find_common_grids()
    
    return {
        'n_common_grids': len(common_grids),
        'lat_range': (common_grids['Latitude'].min(), common_grids['Latitude'].max()),
        'lon_range': (common_grids['Longitude'].min(), common_grids['Longitude'].max()),
        'sm_date_range': (
            processor.sm_loader.data['Date'].min(),
            processor.sm_loader.data['Date'].max()
        ),
        'ndvi_dates': len(processor.ndvi_dates) if processor.ndvi_dates is not None else 0
    }


if __name__ == "__main__":
    # Test the data loading
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("Testing data preprocessor...")
    
    processor = GriddedDataProcessor()
    common_grids = processor.find_common_grids()
    
    print(f"\nFound {len(common_grids)} common grid points")
    
    if len(common_grids) > 0:
        # Test loading one point
        lat = common_grids.iloc[0]['Latitude']
        lon = common_grids.iloc[0]['Longitude']
        
        data = processor.load_point_data(lat, lon, filter_ndvi=True)
        
        print(f"\nData for point ({lat:.4f}, {lon:.4f}):")
        print(f"  Soil Moisture: {len(data['sm'])} points, range: {data['sm'].min():.3f} - {data['sm'].max():.3f}")
        print(f"  Rainfall: {len(data['rain'])} points, range: {data['rain'].min():.3f} - {data['rain'].max():.3f}")
        print(f"  Temperature: {len(data['Ta'])} points, range: {data['Ta'].min():.1f} - {data['Ta'].max():.1f}")

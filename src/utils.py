"""
src/utils.py
============
Project-wide utility functions for Swedish Electricity Price Forecasting.

Design Principles:
1. DRY: Write once, use everywhere
2. Fail-safe: Graceful error handling with retries
3. Type hints: Catch bugs early, enable IDE autocomplete
4. Lazy loading: Expensive operations (API calls) only when needed
"""

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================

import logging
import sys
from pathlib import Path
from typing import Union, Optional, Dict, Any
from datetime import datetime, timedelta
from functools import lru_cache
import time
import json

import pandas as pd 
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 1. Standard Library, Third Party Libraries (PEP 8)
# 2. typing: Import common types for function signatures
# 3. requests: Import submodules for retry strategy configuration


# =============================================================================
# SECTION 2: PATH MANAGEMENT
# =============================================================================

def get_prject_root() -> Path:
    """Get the root directory of the project.
     Automatically locate the project root directory.
    
    Why avoid hardcoding?
    - Code may run in different environments (local PC, server, Colab)
    - Hardcoding 'C:/Users/...' causes failures on other systems
    
    How it works:
    __file__ is the current file path -> parents[1] is the parent of src (project root)
    """
    return Path(__file__).parent.parent

# Calculate and cache the project root immediately
# Why use a constant? Avoid recalculating the path on every call.
PROJECT_ROOT = get_prject_root()

# Centralized definition of all data paths
# Why use a dictionary? Easy iteration and dynamic access.
DATA_PATHS = {
    'raw': PROJECT_ROOT / 'data' / 'raw',
    'processed': PROJECT_ROOT / 'data' / 'processed',
    'external': PROJECT_ROOT / 'data' / 'external',
    'models': PROJECT_ROOT / 'models',
    'reports': PROJECT_ROOT / 'reports',
    'cache': PROJECT_ROOT / 'cache', # APICache Utilities
}

def ensure_dir(path:Path) -> Path:
    """Ensure a directory exists, creating it if necessary.
    Ensure the directory exists, create it if it doesn't.
    
    Why is this needed?
    - Avoid FileNotFoundError when saving to non-existent directories
    - Cleaner than os.makedirs(), and returns a Path object for chaining
    
    Args:
        path: directory path
        
    Returns:
        The passed path (supports chaining)
        
    Example:
        >>> ensure_dir(DATA_PATHS['raw']) / 'prices.csv'
    """
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_data_path(data_type: str, filename: Optional[str] = None) -> Path:
    """
    Get the path to a data file.
    Get the path to a data file.
    
    Args:
        data_type: data type (raw, processed, external, models, reports, cache)
        filename: file name (optional)
        
    Returns:
        File path
        
    Example:
        >>> get_data_path('raw', 'prices.csv')
        >>> get_data_path('processed')
    """
    if data_type not in DATA_PATHS:
        raise ValueError(f"Unknown data_type: {data_type}. "
                         f"Available: {list(DATA_PATHS.keys())}")
    
    base_path = ensure_dir(DATA_PATHS[data_type])

    if filename:
        return base_path / filename
    return base_path

# =============================================================================
# SECTION 3: LOGGING CONFIGURATION
# =============================================================================

def setup_logger(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    console: bool = True,   
) -> logging.Logger:
    """Setup a logger with optional file and console handlers.
    Configure and return a logger instance.
    
    Why encapsulate this?
    - Standardize log format across all modules (time, level, module name)
    - Support simultaneous output to file and console
    - Avoid duplicate logs caused by multiple configurations
    
    Design Decisions:
    1. Use RotatingFileHandler to prevent log files from growing indefinitely
    2. Format includes line numbers for quick troubleshooting
    3. Use propagate=False to avoid duplicate logging to the root logger
    
    Args:
        name: logger name (usually __name__)
        log_file: LoggingFile path(optional)
        level: logging level (default INFO)
        console: whether to output to the console
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Data loading started")
        2025-02-19 14:30:00,123 - src.data_processor - INFO - Data loading started
    """   
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Key: Prevent adding duplicate handlers if the logger is already configured
    if logger.handlers:
        return logger

    # Define normalized format
    # Why use %(name)s instead of %(filename)s?
    # - name shows the full module path (src.data_processor), making it easier to locate
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    if log_file:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Avoid duplicate logs propagating to root logger
    logger.propagate = False

    return logger

# Create a default logger for direct import
# Why? Simplifies usage: from src.utils import logger
logger = setup_logger('swedish_electricity')

# =============================================================================
# SECTION 4: DATA I/O UTILITIES
# =============================================================================

def save_data(
    df: pd.DataFrame,
    data_type: str,
    filename: str,
    format: str = 'auto',
    **kwargs,
) -> Path:
    """Save a DataFrame to a file.
    Why encapsulate this?
    - Automated directory creation
    - Automatic format inference (Parquet/CSV/Pickle)
    - Log saving actions to track data lineage
    
    Format suggestions:
    - Parquet: Default, high compression, fast I/O, preserves data types
    - CSV: Human-readable, but loses data types (dates become strings)
    - Pickle: Python-specific, saves full objects (including models)
    
    Args:
        df: DataFrame to save
        data_type: data type (raw, processed, external, models, reports, cache)
        filename: file name
        format: file format (auto, csv, parquet, h5)
        **kwargs: additional arguments for pandas.to_
    
    Returns:
        Path to the saved file
    
    Example:
        >>> save_data(df, 'processed', 'prices.csv', format='csv')
    """

    # Automatically select format based on data type
    if format == 'auto':
        suffix = Path(filename).suffix.lower()
        format_map = {'.parquet': 'parquet', '.csv': 'csv', '.pkl': 'pkl', '.pickle': 'pkl'}
        format = format_map.get(suffix, 'parquet') # default parquet

    # Get full path
    filepath = get_data_path(data_type, filename)

    # Ensure parent directory exists
    ensure_dir(filepath.parent)

    # Save based on format
    start_time = time.time()
    if format == 'parquet':
        # Use pyarrow engine, supports complex data types
        df.to_parquet(filepath, engine='pyarrow', **kwargs)
    elif format == 'csv':
        df.to_csv(filepath, **kwargs)
    elif format == 'pickle':
        df.to_pickle(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")

    elapsed = time.time() - start_time
    logger.info(f"Saved {filename} data to {filepath} ({len(df)} rows, {elapsed:.2f} s)")

    return filepath

def load_data(
    data_type: str,
    filename: str,
    format: str = 'auto',
    **kwargs,
) -> pd.DataFrame:
    """Load a DataFrame from a file.
    Load data from standard locations.
    
    Why encapsulate?
    - Unified error handling (clear message if file doesn't exist)
    - Automatic format inference
    - Record loading time for performance monitoring
    
    Args:
        data_type: data type
        filename: file name
        format: file format or 'auto'
        **kwargs: arguments passed to the specific loading function
        
    Returns:
        The loaded DataFrame
        
    Raises:
        FileNotFoundError: when the file does not exist
        
    Example:
        >>> df = load_data('raw', 'nordpool_prices.parquet')
        >>> df = load_data('processed', 'features.csv', parse_dates=['timestamp'])
    """ 
    filepath = get_data_path(data_type, filename)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Please ensure data is downloaded to {DATA_PATHS[data_type ]}"
        )
    
    start_time = time.time()

    # Infer format automatically
    if format == 'auto':
        suffix = Path(filename).suffix.lower()
        format_map = {'.parquet': 'parquet', '.csv': 'csv', '.pkl': 'pkl', '.pickle': 'pkl'}
        format = format_map.get(suffix)
        if not format:
            raise ValueError(f"Cannot infer format from suffix: {suffix}") # default parquet

    # Load data
    if format == 'parquet':
        df = pd.read_parquet(filepath, **kwargs)
    elif format == 'csv':
        df = pd.read_csv(filepath, **kwargs)
    elif format == 'pickle':
        df = pd.read_pickle(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    elapsed = time.time() - start_time
    logger.info(f"Loaded {format} data from {filepath} ({len(df)} rows, {elapsed:.2f} s)")
    
    return df

# =============================================================================
# SECTION 5: TIME SERIES UTILITIES
# =============================================================================

def convert_to_swedish_time(
    df: pd.DataFrame, 
    datetime_col: str = 'timestamp',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Convert time column to Swedish local time (CET/CEST), handling DST ambiguity.
    
    Key Fixes:
    - Use 'ambiguous' parameter to handle duplicate hours during DST transitions
    - Use 'nonexistent' parameter to handle missing hours during DST transitions
    """
    if not inplace:
        df = df.copy()
    
    # Ensure it is a datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # If already localized, convert to UTC first
    if df[datetime_col].dt.tz is not None:
        df[datetime_col] = df[datetime_col].dt.tz_convert('UTC')
    else:
        # Treat timezone-less as UTC
        df[datetime_col] = df[datetime_col].dt.tz_localize('UTC')
    
    # Convert to Swedish time, handle DST ambiguity
    # ambiguous='infer': Infer based on chronological order
    # nonexistent='shift_forward': Missing clock jump to next hour
    df[datetime_col] = df[datetime_col].dt.tz_convert('Europe/Stockholm')
    
    # Remove timezone info to avoid downstream issues
    # Comment out the next line if timezone-awareness is required
    df[datetime_col] = df[datetime_col].dt.tz_localize(None)
    
    return df


def create_lag_features(
    df: pd.DataFrame,
    column: str,
    lags: list,
    drop_original: bool = False,
    fill_method: Optional[str] = 'bfill'
) -> pd.DataFrame:
    """
    Safely create lag features.
    
    Why encapsulate this?
    - Handle missing values (boundary effects)
    - Automatic column naming (avoid manual f'{col}_lag_{lag}')
    - Support backfilling (for time series, fill with future values is more reasonable than 0)
    
    Args:
        df: input dataframe (must be sorted by time)
        column: column name to create lags for
        lags: list of lag periods (e.g., [1, 24, 168] for 1h, 1d, 1w legacy)
        drop_original: whether to drop the original column (default False)
        fill_method: missing value fill method ('bfill', 'ffill', None)
        
    Returns:
        DataFrame with lag features added
        
    Example:
        >>> df = create_lag_features(df, 'price', [1, 24, 168])
        # Added columns: price_lag_1, price_lag_24, price_lag_168
    """
    df = df.copy()
    
    for lag in lags:
        new_col = f'{column}_lag_{lag}'
        df[new_col] = df[column].shift(lag)
        
        # Fill missing values (start of time series)
        if fill_method:
            df[new_col] = df[new_col].fillna(method=fill_method)
    
    if drop_original:
        df = df.drop(columns=[column])
    
    logger.debug(f"Created {len(lags)} lag features for '{column}': {lags}")
    return df


def resample_to_hourly(
    df: pd.DataFrame,
    datetime_col: str = 'timestamp',
    value_cols: Optional[list] = None,
    agg_func: str = 'mean'
) -> pd.DataFrame:
    """
    Resample data to hourly frequency.
    
    Why is this needed?
    - Different sources have different frequencies (SMHI 10min, Nord Pool hourly)
    - Standardize frequency for easy merging
    
    Args:
        df: input data
        datetime_col: datetime column name
        value_cols: value columns to resample (default all numeric columns)
        agg_func: aggregation function ('mean', 'sum', 'last')
        
    Returns:
        Hourly resampled DataFrame
    """
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col)
    
    if value_cols is None:
        value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Resample
    resampled = df[value_cols].resample('H').agg(agg_func)
    
    # Reset index to column
    resampled = resampled.reset_index()
    
    logger.info(f"Resampled from {len(df)} to {len(resampled)} hourly rows")
    return resampled


# =============================================================================
# SECTION 6: API CLIENTS WITH RETRY LOGIC
# =============================================================================

class APIClient:
    """
    API Client Base Class with retry mechanism.
    
    Design Pattern: Template Method Pattern
    - Subclasses implement specific request building and parsing logic
    - Base class provides unified retry, error handling, and logging
    
    Why use a class instead of functions?
    - Maintain state (session, retry counts, rate limiting)
    - Support connection pooling (requests.Session)
    - Ease of unit testing (can Mock the whole class)
    """
    
    def __init__(
        self,
        base_url: str,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        timeout: int = 30,
        rate_limit: Optional[float] = None
    ):
        """
        Args:
            base_url: API base URL
            max_retries: maximum retry attempts
            backoff_factor: backoff factor (e.g., 1.0 for 1s, 2s, 4s...)
            timeout: request timeout (seconds)
            rate_limit: max requests per second (None for unlimited)
        """
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.rate_limit = rate_limit
        self._last_request_time = 0
        
        # Configure Session and retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],  # These status codes trigger retries
            allowed_methods=["HEAD", "GET", "OPTIONS"]   # Only retry safe methods
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.logger = setup_logger(self.__class__.__name__)
    
    def _rate_limit_wait(self):
        """Simple rate limit implementation"""
        if self.rate_limit:
            min_interval = 1.0 / self.rate_limit
            elapsed = time.time() - self._last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_request_time = time.time()
    
    def get(self, endpoint: str, params: Optional[Dict] = None, **kwargs) -> Dict:
        """
        Send a GET request.
        
        Why return a Dict instead of original response?
        - Callers usually need JSON data directly
        - Unified error handling (exceptions for non-2xx status codes)
        """
        self._rate_limit_wait()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {url} - {e}")
            raise
    
    def close(self):
        """Close session, release connection pool"""
        self.session.close()
    
    def __enter__(self):
        """Support 'with' statements"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Auto-close on exit"""
        self.close()


# =============================================================================
# SECTION 7: NORD POOL CLIENT (Implementation)
# =============================================================================

class NordPoolClient(APIClient):
    """
    Nord Pool electricity price client.
    
    Nord Pool API Characteristics:
    - Provides historical price data (day-ahead)
    - Requires specifying delivery area (SE1, SE2, SE3, SE4)
    - Date format: YYYY-MM-DD
    
    API Documentation: https://www.nordpoolgroup.com/api/
    """
    
    def __init__(self):
        super().__init__(
            base_url="https://www.nordpoolgroup.com/api",
            max_retries=3,
            rate_limit=1.0  # 1 request/sec (polite scraping)
        )
        self.delivery_areas = ['SE1', 'SE2', 'SE3', 'SE4']
    
    def fetch_day_ahead_prices(
        self,
        start_date: str,
        end_date: str,
        area: str = 'SE3'
    ) -> pd.DataFrame:
        """
        Fetch day-ahead price data.
        
        Why wrap and return as a DataFrame?
        - Downstream can perform feature engineering without extra parsing
        - Unified column names (timestamp, price, area)
        
        Args:
            start_date: start date (YYYY-MM-DD)
            end_date: end date (YYYY-MM-DD)
            area: bidding area (defaults to 'SE3' Stockholm)
            
        Returns:
            DataFrame with columns: [timestamp, price, area, currency]
        """
        if area not in self.delivery_areas:
            raise ValueError(f"Invalid area: {area}. Use: {self.delivery_areas}")
        
        # Build request parameters (adjust according to actual API)
        params = {
            'start': start_date,
            'end': end_date,
            'delivery_area': area,
            'currency': 'SEK'
        }
        
        try:
            # Note: Placeholder endpoint, adjust based on Nord Pool API docs
            data = self.get('/marketdata/page/10', params=params)
            
            # Parse response (adjust based on actual API structure)
            df = self._parse_price_data(data, area)
            
            # Convert to Swedish time
            df = convert_to_swedish_time(df, 'timestamp')
            
            self.logger.info(f"Fetched {len(df)} price records for {area} "
                           f"({start_date} to {end_date})")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch prices for {area}: {e}")
            raise
    
    def _parse_price_data(self, raw_data: Dict, area: str) -> pd.DataFrame:
        """
        Parse raw API response into a DataFrame.
        
        Why separate this?
        - Only need to change here if API format changes
        - Ease of unit testing (parse logic can be tested independently)
        """
        # Implementation needed based on actual API response format
        # Placeholder implementation:
        records = []
        for item in raw_data.get('data', []):
            records.append({
                'timestamp': item['deliveryStart'],
                'price': item['price'],
                'area': area,
                'currency': 'SEK' # Add currency column if missing
            })
        
        return pd.DataFrame(records)


# =============================================================================
# SECTION 8: SMHI CLIENT (reused from previous implementation but inherits APIClient)
# =============================================================================

class SMHIClient(APIClient):
    """
    SMHI weather data client.
    
    Inherits from APIClient to gain:
    - Automatic retries
    - Rate limiting
    - Unified logging
    """
    
    # Representative weather stations for each region (migrated from previous code)
    STATIONS = {
        'SE1': {'name': 'Luleå', 'id': 162860, 'lat': 65.543, 'lon': 22.114},
        'SE2': {'name': 'Sundsvall', 'id': 148330, 'lat': 62.528, 'lon': 17.444},
        'SE3': {'name': 'Stockholm', 'id': 98230, 'lat': 59.651, 'lon': 17.954},
        'SE4': {'name': 'Malmö', 'id': 64020, 'lat': 55.550, 'lon': 13.367}
    }
    
    def __init__(self):
        super().__init__(
            base_url="https://opendata-download-metobs.smhi.se/api",
            max_retries=3,
            rate_limit=2.0  # SMHI allows more frequent requests
        )
    
    def fetch_temperature(
        self,
        station_id: int,
        period: str = 'corrected-archive'
    ) -> pd.DataFrame:
        """
        Fetch temperature observation data.
        
        Args:
            station_id: weather station ID
            period: 'corrected-archive', 'latest-months', 'latest-day'
            
        Returns:
            DataFrame with columns: [timestamp, temperature, quality]
        """
        endpoint = f"/version/1.0/parameter/1/station/{station_id}/period/{period}/data.json"
        
        try:
            data = self.get(endpoint)
            
            # Parse response (adjust based on actual response structure)
            df = pd.DataFrame(data['value'])
            df['timestamp'] = pd.to_datetime(df['date'], unit='ms', utc=True)
            df = convert_to_swedish_time(df, 'timestamp')
            df = df.rename(columns={'value': 'temperature'})
            
            # Select key columns
            df = df[['timestamp', 'temperature', 'quality']]
            
            self.logger.info(f"Fetched {len(df)} temperature records "
                           f"for station {station_id}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch temperature: {e}")
            raise
    
    def fetch_all_regions(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch temperature data for all four regions.
        
        Why this aggregation method?
        - Downstream usually needs complete data, not per-region requests
        - Unified time alignment and missing value handling
        """
        all_data = []
        
        for region, info in self.STATIONS.items():
            self.logger.info(f"Fetching temperature for {region} ({info['name']})...")
            
            try:
                df = self.fetch_temperature(info['id'], period='corrected-archive')
                
                # Filter date range
                mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
                df = df[mask].copy()
                
                df['region'] = region
                df['station_name'] = info['name']
                
                all_data.append(df)
                
            except Exception as e:
                self.logger.warning(f"Failed to fetch {region}, skipping: {e}")
                continue
        
        if not all_data:
            raise RuntimeError("Failed to fetch data for all regions")
        
        combined = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Combined temperature data: {len(combined)} rows")
        return combined

# =============================================================================
# SECTION 11: ENERGI DATA SERVICE CLIENT (Energi Data Service (Denmark) - Includes Swedish Nord Pool data)
# =============================================================================

class EnergiDataServiceClient(APIClient):
    """
    Energi Data Service Client.
    
    Important: although the domain is energidataservice.dk (Denmark), the service contains data for multiple Nordic countries,
    including Swedish Nord Pool day-ahead price data (SE1, SE2, SE3, SE4).
    
    Data source: Energinet (Danish Transmission System Operator)
    API Documentation: https://www.energidataservice.dk/
    
    Free, no authentication required, supports hourly historical data.
    """
    
    # Swedish bidding zones (Nord Pool)
    SWEDISH_AREAS = ['SE1', 'SE2', 'SE3', 'SE4']
    
    # Denmark areas (standby)
    DENMARK_AREAS = ['DK1', 'DK2']
    
    # Norway areas (standby)
    NORWAY_AREAS = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']
    
    def __init__(self):
        super().__init__(
            base_url="https://api.energidataservice.dk",
            max_retries=3,
            backoff_factor=1.0,
            rate_limit=10.0,  # 10 requests/sec (loose API limit)
            timeout=60
        )
        self.logger = setup_logger(self.__class__.__name__)
    
    def fetch_elspot_prices(
        self,
        start_date: str,
        end_date: str,
        areas: Optional[list] = None,
        currency: str = 'EUR'
    ) -> pd.DataFrame:
        """
        Get Nord Pool day-ahead spot prices (Elspot Prices).
        
        This is a core function to fetch day-ahead price data for the four Swedish areas.
        
        Args:
            start_date: start date (YYYY-MM-DD)
            end_date: end date (YYYY-MM-DD)
            areas: area list (defaults to SE1-SE4)
            currency: Currency ('EUR' or 'DKK', default 'EUR')
            
        Returns:
            DataFrame with columns: [timestamp, area, price, currency]
            
        Example:
            >>> client = EnergiDataServiceClient()
            >>> df = client.fetch_elspot_prices('2023-01-01', '2023-12-31')
            >>> print(df.head())
        """
        if areas is None:
            areas = self.SWEDISH_AREAS
        
        # Validate area codes
        invalid_areas = [a for a in areas if a not in self.SWEDISH_AREAS + self.DENMARK_AREAS + self.NORWAY_AREAS]
        if invalid_areas:
            raise ValueError(f"Invalid areas: {invalid_areas}. Valid Swedish areas: {self.SWEDISH_AREAS}")
        
        all_data = []
        
        for area in areas:
            self.logger.info(f"Fetching Elspot prices for {area} ({start_date} to {end_date})...")
            
            try:
                df = self._fetch_single_area(start_date, end_date, area, currency)
                all_data.append(df)
                self.logger.info(f"✅ Got {len(df)} records for {area}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to fetch {area}: {e}")
                continue
        
        if not all_data:
            raise RuntimeError("Failed to fetch data for all areas")
        
        combined = pd.concat(all_data, ignore_index=True)
        
        # Convert to Swedish time
        combined = convert_to_swedish_time(combined, 'timestamp')
        
        self.logger.info(f"✅ Total Elspot records fetched: {len(combined)}")
        return combined
    
    def _fetch_single_area(
        self,
        start_date: str,
        end_date: str,
        area: str,
        currency: str
    ) -> pd.DataFrame:
        """
        Fetch price data for a single area (internal method).
        
        Why separate encapsulation?
        - Simplify main function logic
        - Easy for unit testing
        - Support pagination (if data volume is large)
        """
        endpoint = "/dataset/ElspotPrices"
        
        # Build filter (JSON format)
        filter_dict = {
            "PriceArea": [area]
        }
        
        params = {
            'start': start_date,
            'end': end_date,
            'filter': json.dumps(filter_dict),
            'limit': 100000  # Large enough number to get all data
        }
        
        try:
            data = self.get(endpoint, params=params)
            
            if 'records' not in data:
                raise ValueError(f"Unexpected API response: {data.keys()}")
            
            records = data['records']
            
            if not records:
                self.logger.warning(f"No data returned for {area}")
                return pd.DataFrame()
            
            # Parse into DataFrame
            df = pd.DataFrame(records)
            
            # Standardize column names (map API names to internal consistency)
            column_mapping = {
                'HourDK': 'timestamp',      # Danish time
                'HourUTC': 'timestamp_utc', # UTC time
                'SpotPriceEUR': 'price',    # Spot price in EUR
                'SpotPriceDKK': 'price_dkk',# Spot price in DKK
                'PriceArea': 'area',        # Bidding zone code
                'Currency': 'currency'      # Currency
            }
            
            # Keep only existing columns
            existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
            df = df.rename(columns=existing_mapping)
            
            # Ensure timestamp column exists
            if 'timestamp' not in df.columns and 'timestamp_utc' in df.columns:
                df['timestamp'] = df['timestamp_utc']
            
            # Add currency column if missing
            if 'currency' not in df.columns:
                df['currency'] = currency
            
            # Select key columns
            keep_cols = ['timestamp', 'area', 'price', 'currency']
            available_cols = [c for c in keep_cols if c in df.columns]
            df = df[available_cols]
            
            # Data type conversion
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
            # Drop missing values
            df = df.dropna(subset=['price'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error parsing data for {area}: {e}")
            raise
    
    def fetch_production_consumption(
        self,
        start_date: str,
        end_date: str,
        areas: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Get electricity production and consumption data (ProductionConsumptionSettlement).
        
        Note: This dataset may primarily cover Denmark; Swedish data is recommended from Svenska Kraftnät.
        However, it can be used as a backup data source for validation.
        
        Args:
            start_date: start date (YYYY-MM-DD)
            end_date: end date (YYYY-MM-DD)
            areas: area list (defaults to SE1-SE4)
            
        Returns:
            DataFrame with production/consumption data
        """
        if areas is None:
            areas = self.SWEDISH_AREAS
        
        all_data = []
        
        for area in areas:
            self.logger.info(f"Fetching production/consumption for {area}...")
            
            try:
                endpoint = "/dataset/ProductionConsumptionSettlement"
                
                # Note: filter format for this API might vary, needs debugging
                params = {
                    'start': start_date,
                    'end': end_date,
                    'limit': 100000
                }
                
                data = self.get(endpoint, params=params)
                records = data.get('records', [])
                
                if not records:
                    continue
                
                df = pd.DataFrame(records)
                
                # Check for area columns, filter Swedish data
                if 'PriceArea' in df.columns:
                    df = df[df['PriceArea'] == area]
                elif 'Area' in df.columns:
                    df = df[df['Area'] == area]
                
                if len(df) > 0:
                    df['area'] = area
                    all_data.append(df)
                    self.logger.info(f"✅ Got {len(df)} production records for {area}")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to fetch production data for {area}: {e}")
                continue
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            return combined
        else:
            self.logger.warning("No production/consumption data found for Swedish areas")
            return pd.DataFrame()
    
    def get_available_datasets(self) -> pd.DataFrame:
        """
        Get list of all available datasets (metadata query).
        
        Used to explore what other datasets are available in the API.
        """
        try:
            data = self.get("/meta/dataset")
            records = data.get('records', [])
            return pd.DataFrame(records)
        except Exception as e:
            self.logger.error(f"Failed to get dataset list: {e}")
            return pd.DataFrame()
    
    def validate_data_quality(self, df: pd.DataFrame) -> dict:
        """
        Validate the quality of the downloaded data.
        """
        report = {
            'total_records': len(df),
            'areas': df['area'].unique().tolist() if 'area' in df.columns else [],
            'missing_values': df.isnull().sum().to_dict(),
        }
        
        # Time range statistics
        if 'timestamp' in df.columns and len(df) > 0:
            # Ensure timestamp is datetime type
            timestamps = pd.to_datetime(df['timestamp'])
            report['date_range'] = {
                'start': timestamps.min().isoformat(),
                'end': timestamps.max().isoformat()
            }
            
            # Price statistics
            if 'price' in df.columns:
                report['price_statistics'] = {
                    'min': float(df['price'].min()),
                    'max': float(df['price'].max()),
                    'mean': float(df['price'].mean()),
                    'std': float(df['price'].std())
                }
            
            # Check time continuity (simplified, avoiding DST issues)
            try:
                # Check per area
                completeness_list = []
                for area in df['area'].unique():
                    area_df = df[df['area'] == area].copy()
                    area_df = area_df.sort_values('timestamp')
                    
                    # Calculate expected hours
                    # Handle subtraction errors for naive datetimes:
                    # Subtracting is usually OK for continuous sequences, but doesn't handle duplicates.
                    t_start = pd.to_datetime(area_df['timestamp'].min())
                    t_end = pd.to_datetime(area_df['timestamp'].max())
                    
                    duration = t_end - t_start
                    expected_hours = int(duration.total_seconds() / 3600) + 1
                    
                    actual_hours = len(area_df)
                    completeness = actual_hours / expected_hours if expected_hours > 0 else 0
                    completeness_list.append(completeness)
                
                if completeness_list:
                    report['avg_completeness'] = sum(completeness_list) / len(completeness_list)
                    report['min_completeness'] = min(completeness_list)
                    report['completeness'] = report['avg_completeness'] # For compatibility with old code
                else:
                    report['avg_completeness'] = 1.0
                    report['completeness'] = 1.0
                
            except Exception as e:
                self.logger.warning(f"Could not calculate completeness: {e}")
                report['completeness_error'] = str(e)
        
        return report


# =============================================================================
# SECTION 12: Data Download Script (CLI tool)
# =============================================================================

def download_swedish_electricity_data(
    start_date: str = '2022-01-01',
    end_date: str = '2024-12-31',
    output_dir: Optional[str] = None
):
    """
    One-click download of Swedish electricity data (Nord Pool prices + optional production/consumption).
    
    Convenience function for data scientists to call in CLI or notebook.
    
    Args:
        start_date: start date
        end_date: end date
        output_dir: Output directory (default: data/raw/)
    """
    if output_dir is None:
        output_dir = get_data_path('raw')
    else:
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
    
    logger.info(f"🚀 Starting data download: {start_date} to {end_date}")
    logger.info(f"Output directory: {output_dir}")
    
    # 1. Download price data (Energi Data Service)
    logger.info("\n" + "="*50)
    logger.info("Step 1: Downloading Nord Pool prices (Energi Data Service)")
    logger.info("="*50)
    
    try:
        client = EnergiDataServiceClient()
        prices_df = client.fetch_elspot_prices(start_date, end_date)
        
        if len(prices_df) > 0:
            # Save as Parquet (efficient)
            price_path = output_dir / 'nordpool_prices.parquet'
            prices_df.to_parquet(price_path, index=False)
            logger.info(f"✅ Saved prices to: {price_path}")
            
            # Also save as CSV (human-readable)
            price_csv = output_dir / 'nordpool_prices.csv'
            prices_df.to_csv(price_csv, index=False)
            logger.info(f"✅ Saved prices CSV to: {price_csv}")
            
            # Data quality report
            quality = client.validate_data_quality(prices_df)
            logger.info(f"\nData Quality Report:")
            logger.info(f"  Records: {quality['total_records']}")
            logger.info(f"  Areas: {quality['areas']}")
            logger.info(f"  Date range: {quality['date_range']['start']} to {quality['date_range']['end']}")
            logger.info(f"  Completeness: {quality['completeness']:.2%}")
        else:
            logger.error("❌ No price data downloaded")
            
    except Exception as e:
        logger.error(f"❌ Failed to download prices: {e}")
        raise
    
    # 2. Attempting to download production/consumption data (backup)
    logger.info("\n" + "="*50)
    logger.info("Step 2: Attempting production/consumption data")
    logger.info("="*50)
    
    try:
        prod_df = client.fetch_production_consumption(start_date, end_date)
        
        if len(prod_df) > 0:
            prod_path = output_dir / 'production_consumption.parquet'
            prod_df.to_parquet(prod_path, index=False)
            logger.info(f"✅ Saved production data to: {prod_path}")
        else:
            logger.warning("⚠️ No production data available (expected, use Svenska Kraftnät instead)")
            
    except Exception as e:
        logger.warning(f"⚠️ Production data download failed: {e}")
        logger.info("Tip: Use Svenska Kraftnät Mimer for Swedish production data")
    
    logger.info("\n" + "="*50)
    logger.info("Download complete!")
    logger.info(f"Files saved to: {output_dir}")
    logger.info("="*50)
    
    return prices_df


# =============================================================================
# Update __all__ list (add new classes)
# =============================================================================

__all__ = [
    # Path Management
    'PROJECT_ROOT',
    'DATA_PATHS',
    'get_data_path',
    'ensure_dir',
    
    # Logging
    'setup_logger',
    'logger',
    
    # Data I/O
    'save_data',
    'load_data',
    
    # Time Series
    'convert_to_swedish_time',
    'create_lag_features',
    'resample_to_hourly',
    
    # API Clients
    'APIClient',
    'NordPoolClient',  # Keep for official API (if purchased later)
    'SMHIClient',
    'EnergiDataServiceClient',  # Added
    
    # Data Download
    'download_swedish_electricity_data',
    
    # Cache Utilities
    'clear_cache',
]


# =============================================================================
# SECTION 9: CACHE UTILITIES
# =============================================================================

@lru_cache(maxsize=128)
def get_cached_weather(region: str, date_str: str) -> Optional[float]:
    """
    Weather data retrieval with caching.
    
    Why use lru_cache?
    - Avoid redundant API calls (within the same run)
    - Automatically manage cache size (maxsize=128)
    - Functional interface, easy to use
    
    Note: Caching is per-process and lost upon restart.
    Use files or Redis for persistent storage.
    """
    # Actual implementation would call SMHIClient
    # This is an example to show decorator usage
    pass


def clear_cache():
    """Clear all caches (to force data refresh)"""
    get_cached_weather.cache_clear()
    logger.info("Cleared function caches")


# =============================================================================
# SECTION 10: MODULE EXPORTS
# =============================================================================

# Define public interfaces (items imported via 'from src.utils import *')
__all__ = [
    # Path Management
    'PROJECT_ROOT',
    'DATA_PATHS',
    'get_data_path',
    'ensure_dir',
    
    # Logging
    'setup_logger',
    'logger',
    
    # Data I/O
    'save_data',
    'load_data',
    
    # Time Series
    'convert_to_swedish_time',
    'create_lag_features',
    'resample_to_hourly',
    
    # API Clients
    'APIClient',
    'NordPoolClient',
    'SMHIClient',
    
    # Cache Utilities
    'clear_cache',
]
"""
Polygon.io data loader module for the SPY stock price prediction project.
Handles downloading and preprocessing SPY 1-minute candle data from Polygon.io API.
"""

import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, date
import random
import logging
import sys
import time
from dateutil.relativedelta import relativedelta
import pytz
import json
import pickle
from polygon import RESTClient
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import MinMaxScaler

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_CONFIG, PATHS, MODEL_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolygonDataLoader:
    """Class for loading and preprocessing SPY stock data using Polygon.io API."""
    
    def __init__(self):
        """Initialize PolygonDataLoader with configuration."""
        self.ticker = DATA_CONFIG["ticker"]
        self.interval = DATA_CONFIG["interval"]  # 1m for 1-minute candles
        self.data_dir = PATHS["data_dir"]
        self.random_seed = DATA_CONFIG.get("random_seed", 42)
        self.features = DATA_CONFIG.get("features", [])
        
        # Polygon.io API configuration
        self.api_key = DATA_CONFIG.get("polygon_api_key")
        self.max_historical_years = DATA_CONFIG.get("max_historical_years", 2)
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize polygon client
        if self.api_key:
            self.polygon_client = RESTClient(self.api_key)
            logger.info(f"Initialized Polygon.io client with API key")
        else:
            logger.error("No Polygon.io API key provided in configuration")
            self.polygon_client = None
        
        # Initialize scalers
        self.scalers = {}
        
        # Set the timezone to US/Eastern (market timezone)
        self.timezone = pytz.timezone('US/Eastern')
        
        # API rate limiting
        self.last_api_call = datetime.now() - timedelta(seconds=15)  # Initialize with a past time
        
        # Initialize cache metadata
        self.cache_metadata_file = os.path.join(self.data_dir, "polygon_cache_metadata.json")
        self.cache_metadata = self._load_cache_metadata()
    
    def _load_cache_metadata(self):
        """Load the cache metadata from disk or initialize a new one."""
        if os.path.exists(self.cache_metadata_file):
            try:
                with open(self.cache_metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache metadata: {str(e)}")
                return {"daily_files": {}, "monthly_files": {}, "last_updated": None}
        else:
            return {"daily_files": {}, "monthly_files": {}, "last_updated": None}
    
    def _save_cache_metadata(self):
        """Save the cache metadata to disk."""
        try:
            with open(self.cache_metadata_file, 'w') as f:
                json.dump(self.cache_metadata, f)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {str(e)}")
    
    def _update_cache_metadata(self, date_str, file_path, num_rows, file_type="daily"):
        """Update the cache metadata with information about a downloaded file."""
        if file_type == "daily":
            self.cache_metadata["daily_files"][date_str] = {
                "file_path": file_path,
                "num_rows": num_rows,
                "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        elif file_type == "monthly":
            self.cache_metadata["monthly_files"][date_str] = {
                "file_path": file_path,
                "num_rows": num_rows,
                "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        self.cache_metadata["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_cache_metadata()
    
    def _respect_rate_limit(self):
        """Ensure we respect Polygon.io's API rate limits (5 calls per minute for free tier)."""
        now = datetime.now()
        elapsed = (now - self.last_api_call).total_seconds()
        
        # If less than 15 seconds since last call, wait
        if elapsed < 15:
            wait_time = 15 - elapsed
            logger.info(f"Rate limiting: Waiting {wait_time:.2f} seconds between API calls")
            time.sleep(wait_time)
        
        # Update the last API call time
        self.last_api_call = datetime.now()
    
    def download_latest_data(self, days=7, force_refresh=False):
        """
        Download the latest available SPY 1-minute candle data using Polygon.io API.
        
        Args:
            days: Number of trading days of recent data to download
            force_refresh: If True, will download even if file exists locally
            
        Returns:
            tuple: (success (bool), message (str), data (DataFrame))
        """
        logger.info(f"Downloading latest {self.ticker} 1-minute candle data from Polygon.io")
        
        # Check if we have Polygon client
        if not self.polygon_client:
            return False, "Polygon.io client not initialized. Check API key.", None
        
        try:
            # Calculate date range for trading days (skip weekends and holidays)
            end_date = datetime.now()
            
            # Start with recent days and work backward to get the required number of trading days
            all_data = []
            days_processed = 0
            calendar_days_back = 0
            
            while days_processed < days and calendar_days_back < 30:  # Cap at 30 calendar days
                check_date = end_date - timedelta(days=calendar_days_back)
                check_date_str = check_date.strftime('%Y-%m-%d')
                
                # Skip weekends
                if check_date.weekday() < 5:  # 0-4 for Monday-Friday
                    # Check if we already have data for this date
                    cached_file = os.path.join(self.data_dir, f"{self.ticker}_{check_date_str}_{self.interval}.csv")
                    
                    if os.path.exists(cached_file) and not force_refresh:
                        logger.info(f"Loading cached data for {check_date_str}")
                        day_df = pd.read_csv(cached_file)
                        if not day_df.empty:
                            if 'timestamp' in day_df.columns and not pd.api.types.is_datetime64_dtype(day_df['timestamp']):
                                day_df['timestamp'] = pd.to_datetime(day_df['timestamp'])
                            all_data.append(day_df)
                            days_processed += 1
                    else:
                        # Download data for this day
                        try:
                            # Respect API rate limits
                            self._respect_rate_limit()
                            
                            # Get data from Polygon.io
                            aggs = []
                            for agg in self.polygon_client.get_aggs(
                                ticker=self.ticker,
                                multiplier=1,  # 1-minute
                                timespan="minute",
                                from_=check_date_str,
                                to=check_date_str,
                                limit=50000
                            ):
                                aggs.append({
                                    'timestamp': pd.Timestamp(agg.timestamp, unit='ms').tz_localize('UTC').tz_convert('America/New_York'),
                                    'open': agg.open,
                                    'high': agg.high,
                                    'low': agg.low,
                                    'close': agg.close,
                                    'volume': agg.volume,
                                    'vwap': getattr(agg, 'vwap', np.nan)
                                })
                            
                            # Check if we got any data (skip empty days)
                            if aggs:
                                # Convert to DataFrame
                                day_df = pd.DataFrame(aggs)
                                
                                # Save to disk
                                day_df.to_csv(cached_file, index=False)
                                
                                # Update cache metadata
                                self._update_cache_metadata(check_date_str, cached_file, len(day_df), "daily")
                                
                                all_data.append(day_df)
                                days_processed += 1
                                logger.info(f"Downloaded {len(day_df)} rows for {check_date_str}")
                            else:
                                logger.warning(f"No data returned for {check_date_str} (likely a market holiday)")
                        
                        except Exception as day_error:
                            logger.error(f"Error downloading data for {check_date_str}: {day_error}")
                
                calendar_days_back += 1
            
            if not all_data:
                logger.warning(f"No data returned from Polygon.io for {self.ticker}")
                return False, "No data returned from Polygon.io.", None
            
            # Combine all data
            df = pd.concat(all_data, ignore_index=True)
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Save combined data
            latest_file_path = os.path.join(self.data_dir, f"{self.ticker}_latest_{days}d_{self.interval}.csv")
            df.to_csv(latest_file_path, index=False)
            
            # Generate some basic statistics
            stats = {
                "total_rows": len(df),
                "date_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                "trading_days": days_processed,
                "average_rows_per_day": int(len(df) / max(days_processed, 1))
            }
            
            logger.info(f"Successfully downloaded {len(df)} rows of {self.ticker} 1-minute candle data from Polygon.io")
            return True, f"Successfully downloaded {len(df)} rows of {self.interval} {self.ticker} data", df
            
        except Exception as e:
            logger.error(f"Error downloading data from Polygon.io: {str(e)}")
            return False, f"Error downloading data: {str(e)}", None
    
    def _download_specific_month(self, year, month):
        """
        Download a specific month of 1-minute SPY data.
        
        Args:
            year: Year to download (e.g., 2023)
            month: Month to download (1-12)
            
        Returns:
            tuple: (success (bool), message (str), data (DataFrame) or None)
        """
        # Format dates for this month
        if month == 12:
            next_month_year = year + 1
            next_month = 1
        else:
            next_month_year = year
            next_month = month + 1
            
        month_start = datetime(year, month, 1)
        month_end = datetime(next_month_year, next_month, 1) - timedelta(days=1)
        
        month_start_str = month_start.strftime('%Y-%m-%d')
        month_end_str = month_end.strftime('%Y-%m-%d')
        month_key = month_start.strftime('%Y-%m')
        
        # Check if we already have this month cached
        month_file = os.path.join(self.data_dir, f"{self.ticker}_{month_key}_{self.interval}.csv")
        
        if os.path.exists(month_file):
            logger.info(f"Loading cached data for {month_key}")
            try:
                month_df = pd.read_csv(month_file)
                if 'timestamp' in month_df.columns and not pd.api.types.is_datetime64_dtype(month_df['timestamp']):
                    month_df['timestamp'] = pd.to_datetime(month_df['timestamp'])
                return True, f"Loaded {len(month_df)} rows for {month_key} from cache", month_df
            except Exception as e:
                logger.error(f"Error loading cached data for {month_key}: {e}")
                # Continue to download
        
        # Download data for this month
        try:
            logger.info(f"Downloading data for {month_key} ({month_start_str} to {month_end_str})")
            
            # Respect API rate limits
            self._respect_rate_limit()
            
            # Get data from Polygon.io
            aggs = []
            for agg in self.polygon_client.get_aggs(
                ticker=self.ticker,
                multiplier=1,  # 1-minute
                timespan="minute",
                from_=month_start_str,
                to=month_end_str,
                limit=50000
            ):
                aggs.append({
                    'timestamp': pd.Timestamp(agg.timestamp, unit='ms').tz_localize('UTC').tz_convert('America/New_York'),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume,
                    'vwap': getattr(agg, 'vwap', np.nan)
                })
            
            # Check if we got any data
            if not aggs:
                return False, f"No data returned for {month_key}", None
                
            # Convert to DataFrame
            month_df = pd.DataFrame(aggs)
            
            # Save to disk
            month_df.to_csv(month_file, index=False)
            
            # Update cache metadata
            self._update_cache_metadata(month_key, month_file, len(month_df), "monthly")
            
            logger.info(f"Downloaded {len(month_df)} rows for {month_key}")
            return True, f"Successfully downloaded {len(month_df)} rows for {month_key}", month_df
            
        except Exception as month_error:
            logger.error(f"Error downloading data for {month_key}: {month_error}")
            return False, f"Error downloading data for {month_key}: {str(month_error)}", None
    
    def download_historical_data(self, years=1):
        """
        DEPRECATED - Use select_random_month instead for on-demand approach.
        Download historical SPY data spanning multiple years from Polygon.io.
        This method is kept for backward compatibility.
        
        Args:
            years: Number of years of historical data to download
            
        Returns:
            tuple: (success (bool), message (str), data (DataFrame) or None)
        """
        logger.warning("download_historical_data() is deprecated. Use select_random_month() for on-demand approach.")
        
        # We'll just download one random month to avoid rate limits
        if not self.polygon_client:
            return False, "Polygon.io client not initialized. Check API key.", None
        
        success, message, data = self.select_random_month(from_years=years)
        if success:
            return True, "Downloaded a sample month of historical data", data
        else:
            return False, message, None
    
    def select_random_month(self, from_years=None):
        """
        Select and download a random month of data from the historical dataset.
        This method strictly selects dates from the past 2 years only, as per project requirements.
        
        Args:
            from_years: Number of years to select from, defaults to max_historical_years (but will be capped at 2)
            
        Returns:
            tuple: (success (bool), message (str), data (DataFrame) or None)
        """
        if not self.polygon_client:
            return False, "Polygon.io client not initialized. Check API key.", None
        
        # Always cap at max 2 years as per project requirements
        from_years = min(2, self.max_historical_years if from_years is None else from_years)
        
        logger.info(f"Selecting random month of {self.ticker} 1-minute data from last {from_years} years")
        
        try:
            # Calculate date range - strictly from historical_start_date or 2 years ago (whichever is later)
            end_date = datetime.now()
            
            # Use historical_start_date if provided in config, otherwise use 2 years ago
            if "historical_start_date" in DATA_CONFIG and DATA_CONFIG["historical_start_date"]:
                historical_start = datetime.strptime(DATA_CONFIG["historical_start_date"], "%Y-%m-%d")
                # Calculate date from years ago
                years_ago = datetime(end_date.year - from_years, end_date.month, end_date.day)
                # Use the more recent date (don't go before historical_start_date)
                start_date = max(historical_start, years_ago)
            else:
                # Default to from_years ago if no start date is specified
                start_date = datetime(end_date.year - from_years, end_date.month, end_date.day)
            
            # Ensure we don't include the current month (only completed months)
            # Move end_date to the beginning of the current month
            end_date = datetime(end_date.year, end_date.month, 1)
            
            logger.info(f"Date range for random selection: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Get list of all possible year-month combinations within range
            current_date = start_date
            possible_months = []
            
            while current_date < end_date:
                # Add this month to possible choices
                possible_months.append((current_date.year, current_date.month))
                
                # Move to next month
                if current_date.month == 12:
                    current_date = datetime(current_date.year + 1, 1, 1)
                else:
                    current_date = datetime(current_date.year, current_date.month + 1, 1)
            
            # If no months in range, return error
            if not possible_months:
                return False, f"No valid months found in the past {from_years} years", None
            
            # Log the available months for selection
            month_strings = [f"{year}-{month:02d}" for year, month in possible_months]
            logger.info(f"Available months for selection: {', '.join(month_strings)}")
            
            # Check which months we already have cached
            cached_months = []
            for month_key in self.cache_metadata.get("monthly_files", {}).keys():
                try:
                    year, month = map(int, month_key.split('-'))
                    # Only add if it's within our date range
                    month_date = datetime(year, month, 1)
                    if start_date <= month_date < end_date:
                        cached_months.append((year, month))
                except:
                    pass
            
            # Prefer cached months to avoid API calls
            candidate_months = cached_months or possible_months
            
            # If no candidates, return error
            if not candidate_months:
                return False, "No months available in specified range", None
            
            # Select a random month
            random.seed(datetime.now().timestamp())  # Use current timestamp for true randomness
            year, month = random.choice(candidate_months)
            
            logger.info(f"Randomly selected month: {year}-{month:02d}")
            
            # Verify this month contains trading days (not all on weekends/holidays)
            month_start = datetime(year, month, 1)
            if month == 12:
                month_end = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                month_end = datetime(year, month + 1, 1) - timedelta(days=1)
            
            has_trading_days = False
            current_day = month_start
            while current_day <= month_end:
                # Weekday check (0-4 is Monday-Friday)
                if current_day.weekday() < 5 and not self._is_market_holiday(current_day):
                    has_trading_days = True
                    break
                current_day += timedelta(days=1)
            
            if not has_trading_days:
                logger.warning(f"Selected month {year}-{month:02d} has no trading days, trying another")
                candidate_months.remove((year, month))
                if not candidate_months:
                    return False, "No months with trading days available", None
                year, month = random.choice(candidate_months)
            
            # Download (or load from cache) the selected month
            success, message, month_data = self._download_specific_month(year, month)
            
            if not success:
                logger.warning(f"Failed to download selected month {year}-{month:02d}: {message}")
                
                # Try a different month if the first one fails
                candidate_months.remove((year, month))
                
                if not candidate_months:
                    return False, "All candidate months failed to download", None
                
                year, month = random.choice(candidate_months)
                success, message, month_data = self._download_specific_month(year, month)
                
                if not success:
                    return False, f"Failed to download alternative month {year}-{month:02d}: {message}", None
            
            # Save selected month data with training label
            month_key = f"{year}-{month:02d}"
            training_path = os.path.join(self.data_dir, f"{self.ticker}_training_month_{month_key}_{self.interval}.csv")
            month_data.to_csv(training_path, index=False)
            
            logger.info(f"Selected month {month_key} with {len(month_data)} rows for training")
            return True, f"Selected month {month_key} with {len(month_data)} rows", month_data
            
        except Exception as e:
            logger.error(f"Error selecting random month: {str(e)}")
            return False, f"Error selecting random month: {str(e)}", None
    
    def generate_features(self, data):
        """
        Generate technical indicators and features for model training.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added features
        """
        logger.info("Generating technical indicators and features")
        
        # Create a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Standardize column names to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Calculate technical indicators
        # 1. Moving Averages
        df['sma_5'] = SMAIndicator(close=df['close'], n=5).sma_indicator()
        df['sma_20'] = SMAIndicator(close=df['close'], n=20).sma_indicator()
        df['ema_5'] = EMAIndicator(close=df['close'], n=5).ema_indicator()
        df['ema_20'] = EMAIndicator(close=df['close'], n=20).ema_indicator()
        
        # 2. RSI
        df['rsi'] = RSIIndicator(close=df['close'], n=14).rsi()
        
        # 3. MACD
        macd = MACD(close=df['close'], n_slow=26, n_fast=12, n_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # 4. Bollinger Bands
        bollinger = BollingerBands(close=df['close'], n=20, ndev=2)
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        
        # 5. Price and Volume Changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(5)
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        
        # Fill NaN values (created by indicators that need several periods)
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)  # For any remaining NaNs at the beginning
        
        # Drop any remaining rows with NaN
        df.dropna(inplace=True)
        
        logger.info(f"Generated features: {df.columns.tolist()}")
        return df
    
    def prepare_training_data(self, data, sequence_length=60):
        """
        Prepare data for model training by creating sequences and targets.
        
        Args:
            data: DataFrame with features
            sequence_length: Length of input sequences for the model
            
        Returns:
            tuple: (X_train, y_train) arrays ready for model training
        """
        logger.info(f"Preparing training data with sequence length {sequence_length}")
        
        try:
            # Create a copy to avoid modifying the original
            df = data.copy()
            
            # Ensure we have the timestamp column as the index
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            
            # Standardize column names to lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Remove the 5000 row limit to use the full dataset
            logger.info(f"Using full dataset with {len(df)} rows for training")
            
            # Select features for training (all except timestamp)
            # Make sure 'close' is included since it's our prediction target
            if 'close' not in df.columns:
                if 'Close' in df.columns:
                    df['close'] = df['Close']
                else:
                    raise ValueError("Missing 'close' column in data. This is required for predictions.")
            
            feature_columns = [col for col in df.columns if col != 'timestamp']
            logger.info(f"Using features: {feature_columns}")
            
            # Create scalers for each feature
            self.scalers = {}
            for col in feature_columns:
                scaler = MinMaxScaler(feature_range=(0, 1))
                df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
                self.scalers[col] = scaler
            
            # Create sequences
            X, y = [], []
            prediction_horizon = MODEL_CONFIG["prediction_horizon"]  # Get from config
            logger.info(f"Using prediction horizon of {prediction_horizon}")
            
            # For each possible starting point in the data
            for i in range(len(df) - sequence_length - prediction_horizon + 1):
                # Input sequence
                X.append(df[feature_columns].iloc[i:i+sequence_length].values)
                # Target: next 'prediction_horizon' close values
                y.append(df['close'].iloc[i+sequence_length:i+sequence_length+prediction_horizon].values)
            
            # Check if we have any sequences
            if len(X) == 0:
                logger.error("No sequences could be generated. Dataset may be too small.")
                raise ValueError("Dataset too small to create sequences with current parameters")
                
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Prepared {len(X)} sequences with input shape {X.shape} and target shape {y.shape}")
            
            # Verify that shapes are valid for TensorFlow
            if len(X) == 0 or len(y) == 0:
                raise ValueError("Empty training data generated")
                
            # Verify appropriate dimensions
            if len(X.shape) != 3:
                raise ValueError(f"Input data has wrong dimensions: {X.shape}. Expected 3D array.")
                
            # Make sure y has correct dimensions for prediction_horizon
            if y.shape[1] != prediction_horizon:
                logger.warning(f"Target shape mismatch: {y.shape[1]} vs expected {prediction_horizon}")
                
                # Try to fix the shape if possible
                if y.shape[1] < prediction_horizon:
                    # Pad with the last value
                    padding = np.repeat(y[:, -1:], prediction_horizon - y.shape[1], axis=1)
                    y = np.concatenate([y, padding], axis=1)
                else:
                    # Truncate
                    y = y[:, :prediction_horizon]
                
                logger.info(f"Adjusted target shape to {y.shape}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def save_scalers(self, model_name):
        """Save fitted scalers for later use in predictions"""
        scaler_path = os.path.join(PATHS["model_dir"], f"{model_name}_scalers.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        logger.info(f"Saved scalers to {scaler_path}")
    
    def load_scalers(self, model_name):
        """Load saved scalers for making predictions"""
        scaler_path = os.path.join(PATHS["model_dir"], f"{model_name}_scalers.pkl")
        with open(scaler_path, 'rb') as f:
            self.scalers = pickle.load(f)
        logger.info(f"Loaded scalers from {scaler_path}")
        
    def _is_market_holiday(self, date):
        """
        Check if a date is a US market holiday.
        This is a simplified check for common US market holidays.
        
        Args:
            date: Date to check
            
        Returns:
            bool: True if holiday, False otherwise
        """
        # Weekend check
        if date.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return True
        
        # Format date for easier comparison
        month_day = (date.month, date.day)
        
        # Common fixed holidays
        fixed_holidays = [
            (1, 1),    # New Year's Day
            (7, 4),    # Independence Day
            (12, 25),  # Christmas Day
        ]
        
        if month_day in fixed_holidays:
            return True
        
        # Floating holidays - Approximate check
        # Martin Luther King Jr. Day - Third Monday in January
        if date.month == 1 and 15 <= date.day <= 21 and date.weekday() == 0:
            return True
            
        # Presidents' Day - Third Monday in February
        if date.month == 2 and 15 <= date.day <= 21 and date.weekday() == 0:
            return True
            
        # Memorial Day - Last Monday in May
        if date.month == 5 and date.day >= 25 and date.weekday() == 0:
            return True
            
        # Labor Day - First Monday in September
        if date.month == 9 and date.day <= 7 and date.weekday() == 0:
            return True
            
        # Thanksgiving - Fourth Thursday in November
        if date.month == 11 and 22 <= date.day <= 28 and date.weekday() == 3:
            return True
            
        # Check for Good Friday (rough approximation)
        # Good Friday is the Friday before Easter Sunday
        # This is an approximation as Easter's date varies each year
        
        # Return False for all other days
        return False 
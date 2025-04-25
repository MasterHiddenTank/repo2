"""
Data loader module for downloading and preprocessing SPY stock data.
Uses Polygon.io API for both latest and historical data.
Optimized for Polygon.io's free tier API limitations.
"""

import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, date
import random
from sklearn.preprocessing import MinMaxScaler
import logging
import sys
import time
from dateutil.relativedelta import relativedelta
import pytz
import json

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_CONFIG, PATHS, MODEL_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading and preprocessing SPY stock data using Polygon.io API."""
    
    def __init__(self):
        """Initialize DataLoader with configuration."""
        self.ticker = DATA_CONFIG["ticker"]
        self.interval = DATA_CONFIG["interval"]  # 1m
        self.period = DATA_CONFIG["period"]
        self.data_dir = PATHS["data_dir"]
        self.random_seed = DATA_CONFIG["random_seed"]
        self.features = DATA_CONFIG["features"]
        
        # Polygon.io API configuration
        self.api_key = DATA_CONFIG["polygon_api_key"]
        self.max_historical_years = DATA_CONFIG["max_historical_years"]
        
        # Polygon.io API base URL
        self.base_url = "https://api.polygon.io"
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize scalers
        self.scalers = {}
        
        # Set the timezone to US/Eastern (market timezone)
        self.timezone = pytz.timezone('US/Eastern')
        
        # Rate limiting for Polygon.io free tier (5 calls per minute)
        self.last_api_call_time = 0
        self.api_calls_today = 0
        self.api_call_day = datetime.now().day
        self.max_calls_per_day = 200  # Conservative limit for free tier
        
        # Create a log file for API calls
        self.api_log_file = os.path.join(self.data_dir, "polygon_api_calls.log")
        
        # Track data file locations
        self.data_cache_file = os.path.join(self.data_dir, "data_cache_index.json")
        self.data_cache = self._load_data_cache()
    
    def _load_data_cache(self):
        """Load the data cache index from disk or initialize a new one."""
        if os.path.exists(self.data_cache_file):
            try:
                with open(self.data_cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading data cache: {str(e)}")
                return {"daily_files": {}, "last_updated": None}
        else:
            return {"daily_files": {}, "last_updated": None}
    
    def _save_data_cache(self):
        """Save the data cache index to disk."""
        try:
            with open(self.data_cache_file, 'w') as f:
                json.dump(self.data_cache, f)
        except Exception as e:
            logger.error(f"Error saving data cache: {str(e)}")
    
    def _update_data_cache(self, date_str, file_path, num_rows):
        """Update the data cache with information about a downloaded file."""
        self.data_cache["daily_files"][date_str] = {
            "file_path": file_path,
            "num_rows": num_rows,
            "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.data_cache["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_data_cache()
    
    def _enforce_rate_limit(self):
        """
        Enforce rate limits for Polygon.io API.
        Free tier: 5 calls per minute
        Conservative approach: Wait at least 15 seconds between calls
        """
        # Check if we're on a new day and reset the counter
        current_day = datetime.now().day
        if current_day != self.api_call_day:
            self.api_calls_today = 0
            self.api_call_day = current_day
            logger.info("New day - resetting API call counter")
            
            # Log daily API call counts
            with open(self.api_log_file, 'a') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d')}: {self.api_calls_today} API calls\n")
        
        # Check if we've reached the daily limit
        if self.api_calls_today >= self.max_calls_per_day:
            logger.warning(f"Daily API call limit reached ({self.max_calls_per_day}). Try again tomorrow.")
            raise Exception("Daily API call limit reached")
        
        # Enforce minimum time between requests (15 seconds for free tier)
        current_time = time.time()
        elapsed = current_time - self.last_api_call_time
        
        if elapsed < 15:
            wait_time = 15 - elapsed
            logger.info(f"Rate limiting: Waiting {wait_time:.2f} seconds between API calls")
            time.sleep(wait_time)
        
        # Update timestamps and counters
        self.last_api_call_time = time.time()
        self.api_calls_today += 1
        
        logger.info(f"API call #{self.api_calls_today}/{self.max_calls_per_day} for today")
    
    def _polygon_get_request(self, endpoint, params=None, max_retries=3):
        """
        Make a GET request to the Polygon.io API with sophisticated retry logic.
        
        Args:
            endpoint: API endpoint to call
            params: Dictionary of query parameters
            max_retries: Maximum number of retry attempts
            
        Returns:
            JSON response if successful, None otherwise
        """
        if params is None:
            params = {}
        
        # Add API key to parameters
        params['apiKey'] = self.api_key
        
        # Make the request with retry logic
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(max_retries):
            try:
                # Enforce rate limits before making request
                self._enforce_rate_limit()
                
                logger.info(f"Polygon API request to {endpoint} (attempt {attempt+1}/{max_retries})")
                
                response = requests.get(url, params=params)
                
                # Check if we got a rate limit error
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = 60 * (attempt + 1)  # 60, 120, 180 seconds for rate limit errors
                        logger.warning(f"Rate limit exceeded (429). Waiting {wait_time} seconds before retry {attempt+2}.")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error("Rate limit exceeded (429) and max retries reached.")
                        return None
                
                # Handle other specific error cases
                if response.status_code == 403:  # Forbidden
                    logger.error(f"API access forbidden (403). Check your API key permissions.")
                    return None
                    
                if response.status_code == 404:  # Not Found
                    logger.warning(f"Data not found (404). This could be a non-trading day or incomplete data.")
                    return None
                
                # Raise error for other HTTP errors
                response.raise_for_status()
                
                # Check if we got a valid response with results
                json_response = response.json()
                if json_response.get('status') == 'ERROR':
                    logger.error(f"API returned error: {json_response.get('error')}")
                    return None
                
                return json_response
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {str(e)}")
                
                # Only retry on network-related errors that might be temporary
                if isinstance(e, (requests.exceptions.ConnectionError, 
                                  requests.exceptions.Timeout,
                                  requests.exceptions.TooManyRedirects)) and attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1)
                    logger.warning(f"Network error, retrying in {wait_time} seconds.")
                    time.sleep(wait_time)
                else:
                    logger.error("Fatal request error or max retries reached.")
                    return None
        
        return None
    
    def download_latest_data(self, days=7, force_refresh=False):
        """
        Download the latest available SPY 1-minute candle data using Polygon.io API.
        As per project rules, we need the most recent 1-minute candles for prediction.
        Prioritizes downloading the most recent day first.
        
        Args:
            days: Number of days of recent data to download
            force_refresh: If True, will download fresh data even if cached
                
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Downloading latest {self.ticker} 1-minute candle data from Polygon.io")
        
        # Calculate date range (recent days)
        end_date = datetime.now()
        
        # We need to get the last 7 *trading* days, not calendar days
        # So we might need to look back further than 7 calendar days
        trading_days_found = 0
        calendar_days_back = 0
        dates_to_download = []
        
        # Keep looking back until we have 7 trading days
        while trading_days_found < days and calendar_days_back < 30:  # Cap at 30 calendar days
            check_date = end_date - timedelta(days=calendar_days_back)
            
            # Only add weekdays (0=Monday, 4=Friday)
            if check_date.weekday() < 5:
                # Skip holidays
                if not self._is_market_holiday(check_date):
                    dates_to_download.append(check_date)
                    trading_days_found += 1
            
            calendar_days_back += 1
        
        # Sort dates from most recent to oldest
        dates_to_download.sort(reverse=True)
        
        start_date = dates_to_download[-1] if dates_to_download else end_date - timedelta(days=14)
        
        logger.info(f"Will attempt to download {len(dates_to_download)} trading days from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Prepare result dataframe
        result_df = pd.DataFrame()
        
        # First try to download just the most recent trading day
        if dates_to_download:
            latest_trading_day = dates_to_download[0]
            logger.info(f"First attempting to download the latest trading day: {latest_trading_day.strftime('%Y-%m-%d')}")
            latest_df = self._download_single_day(latest_trading_day, force_refresh=force_refresh)
            if latest_df is not None and len(latest_df) > 0:
                result_df = pd.concat([result_df, latest_df])
                logger.info(f"Successfully downloaded most recent trading day: {latest_trading_day.strftime('%Y-%m-%d')}")
            else:
                logger.warning(f"Failed to download most recent trading day: {latest_trading_day.strftime('%Y-%m-%d')}")
        
        # Now download the rest of the days
        successful_days = 1 if len(result_df) > 0 else 0  # Count the latest day if downloaded
        
        for date in dates_to_download[1:]:  # Skip the first one we already tried
            date_str = date.strftime('%Y-%m-%d')
            logger.info(f"Downloading data for: {date_str}")
            day_df = self._download_single_day(date, force_refresh=force_refresh)
            if day_df is not None and len(day_df) > 0:
                result_df = pd.concat([result_df, day_df])
                successful_days += 1
                logger.info(f"Successfully downloaded day {successful_days}/{days}: {date_str}")
            else:
                logger.warning(f"Failed to download day: {date_str}")
        
        # Check if we got data
        if len(result_df) == 0:
            logger.warning("No data returned. Market might be closed or there's an issue with the data provider.")
            return None
        
        # Sort by timestamp
        result_df.sort_index(inplace=True)
        
        # Save raw data
        raw_file_path = os.path.join(self.data_dir, f"{self.ticker}_raw_latest.csv")
        result_df.to_csv(raw_file_path)
        logger.info(f"Downloaded {len(result_df)} rows across {successful_days} trading days of {self.ticker} 1-minute candle data using Polygon.io")
        
        return result_df
    
    def _download_single_day(self, date, force_refresh=False):
        """
        Download data for a single day with improved retry logic.
        
        Args:
            date: datetime object for the day to download
            force_refresh: If True, will download fresh data even if cached
            
        Returns:
            DataFrame with data for the day, or None if download failed
        """
        date_str = date.strftime('%Y-%m-%d')
        
        # Check if we've already downloaded this day (in data cache)
        if not force_refresh and date_str in self.data_cache["daily_files"]:
            cached_info = self.data_cache["daily_files"][date_str]
            cache_file = cached_info["file_path"]
            
            if os.path.exists(cache_file):
                logger.info(f"Loading cached data for {date_str} ({cached_info['num_rows']} rows)")
                try:
                    day_df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    if len(day_df) > 0:
                        return day_df
                    else:
                        logger.warning(f"Cached file for {date_str} exists but is empty. Will re-download.")
                except Exception as e:
                    logger.warning(f"Error loading cached data for {date_str}: {str(e)}")
        
        # Construct the API endpoint for aggregates (bars/candles)
        endpoint = f"/v2/aggs/ticker/{self.ticker}/range/1/minute/{date_str}/{date_str}"
        
        # First check if we have a cached file for this day to avoid API calls
        cache_file = os.path.join(self.data_dir, f"{self.ticker}_{date_str}.csv")
        if os.path.exists(cache_file) and not force_refresh:
            logger.info(f"Loading cached file for {date_str}")
            try:
                day_df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if len(day_df) > 0:
                    return day_df
                else:
                    logger.warning(f"Cached file for {date_str} exists but is empty. Will re-download.")
            except Exception as e:
                logger.warning(f"Error loading cached file for {date_str}: {str(e)}")
        
        # Implement retry with exponential backoff
        max_retries = 3  # Reduced retries to conserve API calls
        
        for attempt in range(max_retries):
            try:
                # Make the request - rate limiting is handled inside _polygon_get_request
                response = self._polygon_get_request(endpoint)
                
                if response and response.get('results'):
                    # Convert to DataFrame
                    day_df = pd.DataFrame(response['results'])
                    
                    # Polygon uses different column names, need to rename them
                    day_df.rename(columns={
                        't': 'timestamp',
                        'o': 'Open',
                        'h': 'High',
                        'l': 'Low',
                        'c': 'Close',
                        'v': 'Volume'
                    }, inplace=True)
                    
                    # Convert timestamp (milliseconds) to datetime
                    day_df['timestamp'] = pd.to_datetime(day_df['timestamp'], unit='ms')
                    day_df.set_index('timestamp', inplace=True)
                    
                    # Only keep market hours data (9:30 AM to 4:00 PM Eastern)
                    try:
                        day_df = day_df.between_time('9:30', '16:00')
                        
                        # Check if we got enough data
                        if len(day_df) < 10:
                            logger.warning(f"Very few data points ({len(day_df)}) for {date_str}. This may be a partial trading day.")
                        
                        # Cache the data for future use
                        day_df.to_csv(cache_file)
                        
                        # Update the data cache
                        self._update_data_cache(date_str, cache_file, len(day_df))
                        
                        logger.info(f"Successfully downloaded {len(day_df)} rows for {date_str}")
                        return day_df
                    except Exception as e:
                        logger.warning(f"Error filtering market hours for {date_str}: {str(e)}")
                        return None
                else:
                    logger.warning(f"No data returned for {date_str} on attempt {attempt+1}")
                    
                    # If not successful, and this is the most recent day, check if it's a holiday
                    now = datetime.now()
                    if date.date() == now.date() or (now.hour < 9 and date.date() == (now - timedelta(days=1)).date()):
                        logger.info("Checking if today is a market holiday...")
                        if self._is_market_holiday(date):
                            logger.info(f"{date_str} is a market holiday. Skipping.")
                            return None
            
            except Exception as e:
                logger.error(f"Error downloading data for {date_str}: {str(e)}")
                
                # We'll just let the retry happen automatically next time around
                if attempt < max_retries - 1:
                    logger.info(f"Will retry download for {date_str}")
        
        logger.error(f"Failed to download data for {date_str} after {max_retries} attempts")
        return None
    
    def _is_market_holiday(self, date):
        """
        Check if a given date is a US market holiday.
        This is a simplified version that checks the most common holidays.
        
        Args:
            date: datetime object to check
            
        Returns:
            Boolean: True if it's a holiday, False otherwise
        """
        # Basic market holidays (simplified)
        year = date.year
        holidays = [
            datetime(year, 1, 1),  # New Year's Day
            datetime(year, 1, 15) + timedelta((7 - datetime(year, 1, 15).weekday()) % 7),  # MLK Day (3rd Monday in January)
            datetime(year, 2, 15) + timedelta((7 - datetime(year, 2, 15).weekday()) % 7),  # Presidents Day (3rd Monday in February)
            datetime(year, 5, 31) - timedelta((datetime(year, 5, 31).weekday() + 1) % 7),  # Memorial Day (Last Monday in May)
            datetime(year, 7, 4),  # Independence Day
            datetime(year, 9, 1) + timedelta((7 - datetime(year, 9, 1).weekday()) % 7),  # Labor Day (1st Monday in September)
            datetime(year, 11, 25) - timedelta((datetime(year, 11, 25).weekday() + 1) % 7),  # Thanksgiving (4th Thursday in November)
            datetime(year, 12, 25),  # Christmas
        ]
        
        # Check if the date is in the holidays list
        date_to_check = datetime(date.year, date.month, date.day)
        for holiday in holidays:
            if date_to_check.date() == holiday.date():
                return True
        
        return False
    
    def download_historical_data(self, years=None):
        """
        Download historical SPY data spanning multiple years using Polygon.io API.
        Used for creating the random monthly training datasets.
        
        Args:
            years: Number of years of historical data to download (defaults to max_historical_years)
            
        Returns:
            DataFrame with OHLCV data
        """
        if years is None:
            years = self.max_historical_years
        
        # Cap at max allowed by project rules
        years = min(years, self.max_historical_years)
        
        logger.info(f"Downloading {years} years of historical {self.ticker} 1-minute candle data from Polygon.io")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - relativedelta(years=years)
        
        # Format dates
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Date range: {start_date_str} to {end_date_str}")
        
        # We'll download data in monthly chunks
        result_df = pd.DataFrame()
        
        current_date = start_date
        while current_date < end_date:
            # Calculate month end
            if current_date.month == 12:
                month_end = datetime(current_date.year + 1, 1, 1) - timedelta(days=1)
            else:
                month_end = datetime(current_date.year, current_date.month + 1, 1) - timedelta(days=1)
            
            month_end = min(month_end, end_date)
            
            # Download each day in the month (Polygon API has limits)
            month_df = pd.DataFrame()
            day_date = current_date
            
            logger.info(f"Downloading month: {day_date.strftime('%B %Y')}")
            
            while day_date <= month_end:
                date_str = day_date.strftime('%Y-%m-%d')
                
                # Check if it's a weekday (market day)
                if day_date.weekday() < 5:  # 0-4 are Monday to Friday
                    # Construct the API endpoint
                    endpoint = f"/v2/aggs/ticker/{self.ticker}/range/1/minute/{date_str}/{date_str}"
                    
                    # Make the request
                    response = self._polygon_get_request(endpoint)
                    
                    if response and response.get('results'):
                        # Convert to DataFrame
                        day_df = pd.DataFrame(response['results'])
                        
                        # Polygon uses different column names
                        day_df.rename(columns={
                            't': 'timestamp',
                            'o': 'Open',
                            'h': 'High',
                            'l': 'Low',
                            'c': 'Close',
                            'v': 'Volume'
                        }, inplace=True)
                        
                        # Convert timestamp to datetime
                        day_df['timestamp'] = pd.to_datetime(day_df['timestamp'], unit='ms')
                        day_df.set_index('timestamp', inplace=True)
                        
                        # Only keep market hours data
                        day_df = day_df.between_time('9:30', '16:00')
                        
                        # Append to month data
                        month_df = pd.concat([month_df, day_df])
                
                # Move to next day
                day_date += timedelta(days=1)
                
                # Sleep a bit to respect API rate limits
                time.sleep(0.5)  # Increased delay to avoid rate limiting
            
            # Save month data to file
            month_file = os.path.join(
                self.data_dir, 
                f"{self.ticker}_{current_date.year}_{current_date.month:02d}.csv"
            )
            
            if len(month_df) > 0:
                month_df.sort_index(inplace=True)
                month_df.to_csv(month_file)
                logger.info(f"Saved {len(month_df)} rows for {current_date.strftime('%B %Y')}")
                
                # Append to result
                result_df = pd.concat([result_df, month_df])
            
            # Move to next month
            current_date = month_end + timedelta(days=1)
        
        # Save complete historical data
        if len(result_df) > 0:
            result_df.sort_index(inplace=True)
            historical_file_path = os.path.join(self.data_dir, f"{self.ticker}_historical_{years}y.csv")
            result_df.to_csv(historical_file_path)
            logger.info(f"Downloaded {len(result_df)} rows of historical {self.ticker} data spanning {years} years")
        else:
            logger.error("Failed to download any historical data")
            return None
        
        return result_df
    
    def generate_features(self, data):
        """Generate technical indicators and features from raw data."""
        if data is None or len(data) == 0:
            logger.error("No data provided for feature generation")
            return None
            
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # Exponential Moving Averages
        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        std_dev = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (std_dev * 2)
        df['BB_lower'] = df['BB_middle'] - (std_dev * 2)
        
        # Price change features
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5'] = df['Close'].pct_change(5)
        
        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        
        # Drop NaN values (resulting from indicators that use windows)
        df = df.dropna()
        
        # Save processed data
        processed_file_path = os.path.join(self.data_dir, f"{self.ticker}_processed_latest.csv")
        df.to_csv(processed_file_path)
        logger.info(f"Generated features and saved processed data with {len(df)} rows")
        
        return df
    
    def select_random_month(self, start_year=2018):
        """
        Select a random month of data for training a model.
        
        As per project rules, each model should train on randomly selected 
        1-minute SPY data that spans one month, with these months chosen randomly 
        from different years.
        
        This function first checks if the data is already downloaded, and if not,
        it will download it using the Polygon.io API.
        """
        # Get current year and month
        current_year = datetime.now().year
        
        # Choose a random year between start_year and current_year
        year = random.randint(start_year, current_year)
        
        # Choose a random month
        # If we picked the current year, only choose months that have already passed
        if year == current_year:
            month = random.randint(1, datetime.now().month - 1) if datetime.now().month > 1 else 1
        else:
            month = random.randint(1, 12)
        
        logger.info(f"Selected random month: {datetime(year, month, 1).strftime('%B %Y')}")
        
        # Check if we already have data for this month
        month_file = os.path.join(self.data_dir, f"{self.ticker}_{year}_{month:02d}.csv")
        
        if os.path.exists(month_file):
            # Load existing data
            data = pd.read_csv(month_file, index_col=0, parse_dates=True)
            logger.info(f"Loaded existing data for {datetime(year, month, 1).strftime('%B %Y')} with {len(data)} rows")
            
            if len(data) < 100:  # If not enough data, try another month
                logger.warning(f"Not enough data for {datetime(year, month, 1).strftime('%B %Y')}. Trying another month.")
                return self.select_random_month(start_year)
            
            return data
        
        # If we don't have data for this month, we need to download it
        logger.info(f"No existing data found for {datetime(year, month, 1).strftime('%B %Y')}. Downloading...")
        
        # Calculate date range for the month
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        
        start_date = datetime(year, month, 1)
        
        # Download data for this month
        month_df = pd.DataFrame()
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Check if it's a weekday (market day)
            if current_date.weekday() < 5:  # 0-4 are Monday to Friday
                # Construct the API endpoint
                endpoint = f"/v2/aggs/ticker/{self.ticker}/range/1/minute/{date_str}/{date_str}"
                
                # Make the request
                response = self._polygon_get_request(endpoint)
                
                if response and response.get('results'):
                    # Convert to DataFrame
                    day_df = pd.DataFrame(response['results'])
                    
                    # Polygon uses different column names
                    day_df.rename(columns={
                        't': 'timestamp',
                        'o': 'Open',
                        'h': 'High',
                        'l': 'Low',
                        'c': 'Close',
                        'v': 'Volume'
                    }, inplace=True)
                    
                    # Convert timestamp to datetime
                    day_df['timestamp'] = pd.to_datetime(day_df['timestamp'], unit='ms')
                    day_df.set_index('timestamp', inplace=True)
                    
                    # Only keep market hours data
                    day_df = day_df.between_time('9:30', '16:00')
                    
                    # Append to month data
                    month_df = pd.concat([month_df, day_df])
            
            # Move to next day
            current_date += timedelta(days=1)
            
            # Sleep a bit to respect API rate limits
            time.sleep(0.5)  # Increased delay to avoid rate limiting
        
        # Check if we got enough data
        if len(month_df) < 100:
            logger.warning(f"Not enough data for {datetime(year, month, 1).strftime('%B %Y')}. Trying another month.")
            return self.select_random_month(start_year)
        
        # Save the data
        month_df.sort_index(inplace=True)
        month_df.to_csv(month_file)
        logger.info(f"Downloaded and saved data for {datetime(year, month, 1).strftime('%B %Y')} with {len(month_df)} rows")
        
        return month_df
    
    def prepare_training_data(self, data, sequence_length=60):
        """
        Prepare data for model training by creating sequences.
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            sequence_length: Number of time steps to use as input sequence
            
        Returns:
            X: Input sequences
            y: Target values (next 5 candle prices)
        """
        # Select features to use
        df = data[self.features].copy()
        
        # Scale the data
        self.scalers = {}
        scaled_data = pd.DataFrame()
        
        for column in df.columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data[column] = scaler.fit_transform(df[column].values.reshape(-1, 1)).flatten()
            self.scalers[column] = scaler
        
        # Create X (input sequences) and y (next 5 candle prices)
        X, y = [], []
        
        prediction_horizon = MODEL_CONFIG["prediction_horizon"]
        
        # We need at least sequence_length + prediction_horizon data points
        for i in range(len(scaled_data) - sequence_length - prediction_horizon + 1):
            # Input sequence
            X.append(scaled_data.iloc[i:i+sequence_length].values)
            
            # Target: next prediction_horizon candle Close prices
            next_candles = []
            for j in range(1, prediction_horizon + 1):
                next_candles.append(scaled_data.iloc[i+sequence_length+j-1]['Close'])
            
            y.append(next_candles)
        
        return np.array(X), np.array(y)
    
    def save_scalers(self, model_name):
        """Save scalers for later use in predictions."""
        import joblib
        scaler_file = os.path.join(PATHS["model_dir"], f"{model_name}_scalers.joblib")
        joblib.dump(self.scalers, scaler_file)
        logger.info(f"Saved scalers for model {model_name}")
    
    def load_scalers(self, model_name):
        """Load scalers for predictions."""
        import joblib
        scaler_file = os.path.join(PATHS["model_dir"], f"{model_name}_scalers.joblib")
        self.scalers = joblib.load(scaler_file)
        logger.info(f"Loaded scalers for model {model_name}")
        return self.scalers

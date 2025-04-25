"""
Polygon.io data manager module for the SPY stock price prediction project.
Manages the PolygonDataLoader for downloading and processing SPY stock data.
"""

import os
import pandas as pd
import numpy as np
import logging
import sys
import time
from datetime import datetime, timedelta
import random
import json
import pickle

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_CONFIG, PATHS, MODEL_CONFIG
from data.polygon_data_loader import PolygonDataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolygonManager:
    """
    Class for managing Polygon.io API for SPY stock data.
    Handles downloading, processing, and preparing data for training and prediction.
    """
    
    def __init__(self, config=None):
        """
        Initialize PolygonManager with configuration.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        # Override default config if provided
        self.config = DATA_CONFIG if config is None else config
        
        # Get basic settings
        self.ticker = self.config.get("ticker", "SPY")
        self.interval = self.config.get("interval", "1m")  # 1m for 1-minute candles
        self.data_dir = PATHS["data_dir"]
        
        # Polygon.io API configuration
        self.polygon_api_key = self.config.get("polygon_api_key")
        self.max_historical_years = self.config.get("max_historical_years", 2)
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize polygon data loader
        self.polygon_loader = None
        self._init_client()
        
        logger.info(f"PolygonManager initialized for {self.ticker} {self.interval} data")
    
    def _init_client(self):
        """Initialize Polygon.io client."""
        # Initialize Polygon.io client (only if API key is provided)
        if self.polygon_api_key:
            self.polygon_loader = PolygonDataLoader()
            logger.info("Initialized Polygon.io data loader")
        else:
            logger.warning("No Polygon.io API key provided. Data loading will not be available.")
    
    def download_latest_data(self, days=7, force_refresh=False):
        """
        Download the latest available SPY 1-minute candle data.
        
        Args:
            days: Number of trading days of recent data to download
            force_refresh: If True, will download even if file exists locally
            
        Returns:
            tuple: (success (bool), message (str), data (DataFrame))
        """
        if not self.polygon_loader:
            return False, "Polygon.io data loader not initialized. Check API key.", None
        
        try:
            # Use polygon loader to download latest data
            return self.polygon_loader.download_latest_data(days=days, force_refresh=force_refresh)
        except Exception as e:
            logger.error(f"Error downloading latest data: {e}")
            return False, f"Error downloading latest data: {str(e)}", None
    
    def download_historical_data(self, years=2):
        """
        Download historical SPY data spanning multiple years.
        
        Args:
            years: Number of years of historical data to download
            
        Returns:
            tuple: (success (bool), message (str), data (DataFrame))
        """
        if not self.polygon_loader:
            return False, "Polygon.io data loader not initialized. Check API key.", None
        
        if years < 1 or years > self.max_historical_years:
            return False, f"Years must be between 1 and {self.max_historical_years}", None
        
        try:
            # Use polygon loader to download historical data
            return self.polygon_loader.download_historical_data(years=years)
        except Exception as e:
            logger.error(f"Error downloading historical data: {e}")
            return False, f"Error downloading historical data: {str(e)}", None
    
    def select_random_month(self, from_years=None):
        """
        Select a random month of data from the historical dataset.
        This is crucial for training models on diverse market conditions.
        
        Args:
            from_years: Number of years to select from, defaults to max_historical_years
            
        Returns:
            tuple: (success (bool), message (str), data (DataFrame))
        """
        if not self.polygon_loader:
            return False, "Polygon.io data loader not initialized. Check API key.", None
        
        if from_years is None:
            from_years = self.max_historical_years
        
        try:
            # Use polygon loader to select a random month
            return self.polygon_loader.select_random_month(from_years=from_years)
        except Exception as e:
            logger.error(f"Error selecting random month: {e}")
            return False, f"Error selecting random month: {str(e)}", None
    
    def generate_features(self, data):
        """
        Generate technical indicators and features for model training.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added features
        """
        if not self.polygon_loader:
            logger.error("Polygon.io data loader not initialized")
            return data
        
        try:
            # Use polygon loader to generate features
            return self.polygon_loader.generate_features(data)
        except Exception as e:
            logger.error(f"Error generating features: {e}")
            return data
    
    def prepare_training_data(self, data, sequence_length=60):
        """
        Prepare data for model training by creating sequences and targets.
        
        Args:
            data: DataFrame with features
            sequence_length: Length of input sequences for the model
            
        Returns:
            tuple: (X_train, y_train) arrays ready for model training
        """
        if not self.polygon_loader:
            logger.error("Polygon.io data loader not initialized")
            return None, None
        
        try:
            # Use polygon loader to prepare training data
            return self.polygon_loader.prepare_training_data(data, sequence_length=sequence_length)
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None
    
    def save_scalers(self, model_name):
        """Save the fitted scalers for later use in prediction."""
        try:
            import joblib
            import pickle
            
            # Ensure we have the correct models directory path
            models_dir = PATHS.get("model_dir", os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved"))
            
            # Create models/saved directory if it doesn't exist
            os.makedirs(models_dir, exist_ok=True)
            
            # Check if we have scalers to save
            if not hasattr(self, 'scalers') or not self.scalers:
                if hasattr(self, 'polygon_loader') and self.polygon_loader and hasattr(self.polygon_loader, 'scalers'):
                    self.scalers = self.polygon_loader.scalers
                else:
                    logger.warning(f"No scalers available to save for {model_name}")
                    return False
            
            # Save scalers using both joblib and pickle for compatibility
            joblib_path = os.path.join(models_dir, f"{model_name}_scalers.joblib")
            pickle_path = os.path.join(models_dir, f"{model_name}_scalers.pkl")
            
            # Save with both formats
            joblib.dump(self.scalers, joblib_path)
            with open(pickle_path, 'wb') as f:
                pickle.dump(self.scalers, f)
                
            logger.info(f"Saved scalers to {joblib_path} and {pickle_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving scalers: {str(e)}")
            return False
    
    def load_scalers(self, model_name):
        """Load saved scalers for prediction."""
        try:
            import joblib
            import pickle
            
            # Ensure we have the correct models directory path
            models_dir = PATHS.get("model_dir", os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved"))
            
            # Create a list of possible paths to try
            scaler_paths = [
                # Standard paths
                os.path.join(models_dir, f"{model_name}_scalers.joblib"),
                os.path.join(models_dir, f"{model_name}_scalers.pkl"),
                # Lowercase model name
                os.path.join(models_dir, f"{model_name.lower()}_scalers.joblib"),
                os.path.join(models_dir, f"{model_name.lower()}_scalers.pkl"),
                # With different timestamp formats that might be in the model name
                os.path.join(models_dir, f"{model_name.split('_')[0]}_scalers.joblib"),
                os.path.join(models_dir, f"{model_name.split('_')[0]}_scalers.pkl")
            ]
            
            # Try each path until we find one that works
            for path in scaler_paths:
                if os.path.exists(path):
                    logger.info(f"Found scaler file at {path}")
                    
                    if path.endswith('.joblib'):
                        self.scalers = joblib.load(path)
                    elif path.endswith('.pkl'):
                        with open(path, 'rb') as f:
                            self.scalers = pickle.load(f)
                    
                    logger.info(f"Successfully loaded scalers for {model_name}")
                    return True
            
            # If we failed to find scalers, create empty scalers dictionary
            logger.warning(f"Could not find any scaler files for {model_name}. Using empty scalers.")
            self.scalers = {}
            return False
            
        except Exception as e:
            logger.warning(f"Error loading scalers for {model_name}: {str(e)}")
            self.scalers = {}  # Initialize empty scalers to prevent None errors
            return False 
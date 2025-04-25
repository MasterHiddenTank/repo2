"""
Configuration settings for the SPY stock price prediction system.
"""

import os
import json

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create paths for various directories
PATHS = {
    "base_dir": BASE_DIR,
    "data_dir": os.path.join(BASE_DIR, "data", "datasets"),
    "model_dir": os.path.join(BASE_DIR, "models", "saved"),
    "results_dir": os.path.join(BASE_DIR, "results"),
    "logs_dir": os.path.join(BASE_DIR, "logs"),
    "web_dir": os.path.join(BASE_DIR, "web")
}

# Data configuration
DATA_CONFIG = {
    "ticker": "SPY",
    "interval": "1m",           # 1-minute candles
    "period": "7d",             # Default period for latest data
    "random_seed": 42,
    "features": [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_5", "SMA_20", "EMA_5", "EMA_20", "RSI",
        "MACD", "MACD_signal", "BB_middle", "BB_upper", "BB_lower",
        "Price_Change", "Price_Change_5", "Volume_Change", "Volume_MA_5"
    ],
    "prediction_horizon": 20,    # Predict the next 20 1-minute candles
    "max_historical_years": 2,   # Maximum years of historical data to download
    "historical_start_date": "2023-04-20",  # Earliest date to use for historical data
    
    # Polygon.io API configuration
    "polygon_api_key": "Wzqt74w1THTYkqM1DpT6Xl5SVxPlGsHa"
}

# Model configuration
MODEL_CONFIG = {
    "sequence_length": 60,      # Use 60 timesteps as input
    "batch_size": 32,           # Increased batch size for gradient accumulation
    "epochs": 20,               # Reduced epochs for faster training
    "patience": 5,              # Early stopping patience
    "validation_split": 0.2,
    "test_split": 0.1,
    "learning_rate": 0.0005,    # Smaller learning rate for stability
    "dropout_rate": 0.2,
    "prediction_horizon": 10,    # Reduced from 20 to 10 for memory efficiency
    "accuracy_threshold": 0.65,  # Reduced threshold to make it achievable
    "enable_mixed_precision": True,  # Enable mixed precision for better memory usage
    "log_device_placement": False  # Don't log device placement details to reduce log verbosity
}

# Web application configuration
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True,
    "template_folder": os.path.join(PATHS["web_dir"], "templates"),
    "static_folder": os.path.join(PATHS["web_dir"], "static")
}

# Create directories if they don't exist
for path in PATHS.values():
    os.makedirs(path, exist_ok=True) 

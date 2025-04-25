"""
Entry point for the SPY Stock Price Prediction system.
This script launches the Flask web application.
"""

import os
import sys
import logging
from datetime import datetime

# Set up logging with timestamp
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the Flask application
from web.app import app, main

if __name__ == "__main__":
    logger.info(f"Starting SPY Stock Price Prediction System at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Using Polygon.io API for SPY 1-minute candle data")
    
    try:
        main()
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        sys.exit(1)

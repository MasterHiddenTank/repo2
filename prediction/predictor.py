"""
Predictor module for making stock price predictions.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import joblib
from datetime import datetime, timedelta
import tensorflow as tf
import matplotlib.pyplot as plt
import glob

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_CONFIG, PATHS, MODEL_CONFIG
from data.polygon_manager import PolygonManager
from models.model1_bayesian_neural_network import BayesianNeuralNetwork
from models.model2_bozdogan_caic import BozdoganCAICModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockPredictor:
    """Class for making stock price predictions."""
    
    def __init__(self):
        """Initialize the stock predictor."""
        self.data_loader = PolygonManager()
        self.models = {}  # Initialize as empty dict, not None
        self.model_accuracies = {}  # Dictionary to store model accuracies
        self.sequence_length = 60  # Number of time steps to use as input
        self.prediction_horizon = MODEL_CONFIG["prediction_horizon"]
        
        # Create results directory if it doesn't exist
        os.makedirs(PATHS["results_dir"], exist_ok=True)
    
    def load_models(self):
        """
        Load all trained models.
        
        Returns:
            Dictionary of loaded models
        """
        # Initialize empty dictionaries if they don't exist
        if not hasattr(self, 'models') or self.models is None:
            self.models = {}
        
        if not hasattr(self, 'model_accuracies') or self.model_accuracies is None:
            self.model_accuracies = {}
        
        model_dir = PATHS.get("model_dir", os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved"))
        loaded_models = {}
        
        try:
            # Find all model directories
            model_dirs = [d for d in glob.glob(f"{model_dir}/*") if os.path.isdir(d)]
            
            for model_dir in model_dirs:
                try:
                    # Get model name from directory (e.g., BayesianNN_20230515_123456)
                    model_name = os.path.basename(model_dir).split('_')[0]
                    
                    # Find corresponding metadata file
                    metadata_files = glob.glob(f"{model_dir}_metadata.joblib")
                    if not metadata_files:
                        # Also try looking in the model directory
                        metadata_files = glob.glob(os.path.join(model_dir, "metadata.joblib"))
                    
                    if metadata_files:
                        metadata = joblib.load(metadata_files[0])
                        
                        # Create model instance based on model name
                        if model_name == "BayesianNN":
                            # Handle missing input_shape in metadata
                            input_shape = metadata.get("input_shape", None)
                            if input_shape is None:
                                # Use a default input shape if not available
                                logger.warning(f"No input_shape found in metadata for {model_name}, using default (60, 19)")
                                input_shape = (60, 19)  # Default sequence length and features
                            model = BayesianNeuralNetwork(input_shape)
                            # Add more model types here as they are implemented
                        elif "bozdogan" in model_name.lower() or "caic" in model_name.lower():
                            # Handle missing input_shape in metadata
                            input_shape = metadata.get("input_shape", None)
                            if input_shape is None:
                                logger.warning(f"No input_shape found in metadata for {model_name}, using default (60, 19)")
                                input_shape = (60, 19)  # Default sequence length and features
                            model = BozdoganCAICModel(input_shape)
                        else:
                            logger.warning(f"Unknown model type: {model_name}")
                            continue
                        
                        # Load model weights
                        model.load(model_dir)
                        
                        # Store model accuracy if available in metadata
                        if "accuracy" in metadata:
                            self.model_accuracies[model_name] = metadata["accuracy"]
                        else:
                            self.model_accuracies[model_name] = "Unknown"
                        
                        # Load scalers
                        try:
                            self.data_loader.load_scalers(model_name)
                        except Exception as e:
                            logger.warning(f"Could not load scalers for {model_name}: {str(e)}")
                        
                        loaded_models[model_name] = model
                        logger.info(f"Loaded model {model_name}")
                    else:
                        logger.warning(f"No metadata file found for {model_dir}")
                except Exception as model_error:
                    logger.error(f"Error loading model from {model_dir}: {str(model_error)}")
                    # Continue to next model
            
            # Update the models dictionary
            self.models.update(loaded_models)
            return self.models
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self.models  # Return empty dict instead of None
    
    def get_latest_data(self):
        """
        Get and process the latest data for prediction.
        
        Returns:
            Processed data ready for prediction
        """
        # Download the latest data
        data = self.data_loader.download_latest_data()
        if data is None:
            logger.error("Failed to download latest data")
            return None
        
        # Process the data
        processed_data = self.data_loader.generate_features(data)
        return processed_data
    
    def prepare_prediction_data(self, data):
        """
        Prepare the latest data for prediction.
        
        Args:
            data: Processed data
            
        Returns:
            X: Input data for prediction
        """
        # Select the last sequence_length data points
        if len(data) < self.sequence_length:
            logger.error(f"Not enough data points for prediction. Need at least {self.sequence_length}, got {len(data)}")
            return None
        
        # Check if scalers are available
        if not hasattr(self.data_loader, 'scalers') or not self.data_loader.scalers:
            logger.warning("No scalers available. Creating features without scaling.")
            # Just use the raw data
            selected_features = self.data_loader.features if hasattr(self.data_loader, 'features') else data.columns
            df = data[selected_features].copy()
            X = df.iloc[-self.sequence_length:].values.reshape(1, self.sequence_length, -1)
            return X
        
        # Select features and scale
        try:
            # Check if features attribute exists
            if not hasattr(self.data_loader, 'features') or not self.data_loader.features:
                # Use all available columns except timestamp if present
                features = [col for col in data.columns if col != 'timestamp']
                logger.warning(f"No features list found. Using available columns: {features}")
            else:
                features = self.data_loader.features
                
            # Filter to only use columns that exist in the data
            valid_features = [f for f in features if f in data.columns]
            if len(valid_features) < len(features):
                missing = set(features) - set(valid_features)
                logger.warning(f"Some features are missing from data: {missing}")
            
            df = data[valid_features].copy()
            
            # Scale the data using the available scalers
            scaled_data = pd.DataFrame()
            for column in df.columns:
                if column in self.data_loader.scalers:
                    scaled_data[column] = self.data_loader.scalers[column].transform(df[column].values.reshape(-1, 1)).flatten()
                else:
                    logger.warning(f"No scaler found for {column}, using raw values")
                    scaled_data[column] = df[column]
            
            # Select the last sequence_length data points
            X = scaled_data.iloc[-self.sequence_length:].values.reshape(1, self.sequence_length, -1)
            
            return X
        except Exception as e:
            logger.error(f"Error preparing prediction data: {str(e)}")
            logger.info("Falling back to unscaled data")
            # Fallback to unscaled data in case of errors
            selected_cols = [col for col in data.columns if col != 'timestamp']
            X = data[selected_cols].iloc[-self.sequence_length:].values.reshape(1, self.sequence_length, -1)
            return X
    
    def predict_next_candles(self, model_name=None):
        """
        Predict the next 5 candles.
        
        Args:
            model_name: Name of the model to use for prediction (default: use all models)
            
        Returns:
            Dictionary of predictions from each model
        """
        # Get the latest data
        data = self.get_latest_data()
        if data is None:
            return None
        
        # Prepare data for prediction
        X = self.prepare_prediction_data(data)
        if X is None:
            return None
        
        # Make predictions with all models or specific model
        predictions = {}
        if model_name is not None:
            if model_name in self.models:
                model = self.models[model_name]
                predictions[model_name] = self._make_prediction(model, X, data)
            else:
                logger.error(f"Model {model_name} not found")
        else:
            for name, model in self.models.items():
                predictions[name] = self._make_prediction(model, X, data)
        
        # Save predictions
        self._save_predictions(predictions, data)
        
        return predictions
    
    def _make_prediction(self, model, X, data):
        """
        Make prediction with a specific model.
        
        Args:
            model: Model to use for prediction
            X: Input data
            data: Original data for reference
            
        Returns:
            Dictionary with prediction details
        """
        try:
            # Make prediction
            if hasattr(model, 'predict_with_uncertainty'):
                # For models with uncertainty estimation
                y_pred, y_std = model.predict_with_uncertainty(X)
                lower_bound, upper_bound = model.get_prediction_intervals(X)
                
                # Inverse transform the predictions
                if "Close" in self.data_loader.scalers:
                    close_scaler = self.data_loader.scalers["Close"]
                    y_pred_orig = close_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    lower_bound_orig = close_scaler.inverse_transform(lower_bound.reshape(-1, 1)).flatten()
                    upper_bound_orig = close_scaler.inverse_transform(upper_bound.reshape(-1, 1)).flatten()
                else:
                    # If no scaler is available, use the predictions as-is with a warning
                    logger.warning("No Close scaler found. Using raw predictions which may not be properly scaled.")
                    y_pred_orig = y_pred.flatten()
                    lower_bound_orig = lower_bound.flatten()
                    upper_bound_orig = upper_bound.flatten()
                
                prediction = {
                    "predicted_values": y_pred_orig.tolist(),
                    "lower_bound": lower_bound_orig.tolist(),
                    "upper_bound": upper_bound_orig.tolist(),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                # For regular models
                y_pred = model.predict(X)
                
                # Inverse transform the predictions
                if "Close" in self.data_loader.scalers:
                    close_scaler = self.data_loader.scalers["Close"]
                    y_pred_orig = close_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                else:
                    # If no scaler is available, use the predictions as-is with a warning
                    logger.warning("No Close scaler found. Using raw predictions which may not be properly scaled.")
                    y_pred_orig = y_pred.flatten()
                
                prediction = {
                    "predicted_values": y_pred_orig.tolist(),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            
            # Calculate time points
            last_time = data.index[-1]
            time_delta = pd.Timedelta(minutes=5)  # 5-minute candles
            future_times = [last_time + (i+1)*time_delta for i in range(self.prediction_horizon)]
            prediction["time_points"] = [t.strftime("%Y-%m-%d %H:%M:%S") for t in future_times]
            
            return prediction
        
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None
    
    def _save_predictions(self, predictions, data):
        """
        Save predictions to disk.
        
        Args:
            predictions: Dictionary of predictions
            data: Original data for reference
        """
        # Create predictions directory if it doesn't exist
        predictions_dir = os.path.join(PATHS["results_dir"], "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Save predictions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_file = os.path.join(predictions_dir, f"predictions_{timestamp}.json")
        
        # Add last known prices for reference
        last_candle = data.iloc[-1].to_dict()
        last_prices = {
            "time": data.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "open": float(last_candle["Open"]),
            "high": float(last_candle["High"]),
            "low": float(last_candle["Low"]),
            "close": float(last_candle["Close"]),
            "volume": float(last_candle["Volume"])
        }
        
        # Create combined predictions data
        predictions_data = {
            "timestamp": timestamp,
            "last_candle": last_prices,
            "predictions": predictions
        }
        
        # Save as JSON
        import json
        with open(predictions_file, 'w') as f:
            json.dump(predictions_data, f, indent=4)
        
        logger.info(f"Saved predictions to {predictions_file}")
        
        # Plot predictions
        self._plot_predictions(predictions, data)
    
    def _plot_predictions(self, predictions, data):
        """
        Plot predictions.
        
        Args:
            predictions: Dictionary of predictions
            data: Original data for reference
        """
        plt.figure(figsize=(12, 8))
        
        # Plot historical data
        historical_prices = data["Close"].values[-20:]  # Last 20 points
        historical_times = list(range(len(historical_prices)))
        plt.plot(historical_times, historical_prices, 'b-', label='Historical')
        
        # Plot predictions for each model
        colors = ['r', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'lime', 'brown']
        next_point = len(historical_prices)
        for i, (model_name, prediction) in enumerate(predictions.items()):
            if prediction is not None:
                pred_values = prediction["predicted_values"]
                pred_times = list(range(next_point, next_point + len(pred_values)))
                color = colors[i % len(colors)]
                plt.plot(pred_times, pred_values, f'{color}-', label=f'{model_name}')
                
                # Plot prediction intervals if available
                if "lower_bound" in prediction and "upper_bound" in prediction:
                    plt.fill_between(
                        pred_times,
                        prediction["lower_bound"],
                        prediction["upper_bound"],
                        color=color,
                        alpha=0.2
                    )
        
        # Add a vertical line at the current time
        plt.axvline(x=next_point-1, color='k', linestyle='--')
        
        # Add labels and title
        plt.xlabel('Time (5-minute intervals)')
        plt.ylabel('SPY Price')
        plt.title('SPY Price Prediction')
        plt.legend()
        plt.grid(True)
        
        # Save the figure
        os.makedirs(PATHS["results_dir"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(PATHS["results_dir"], f"prediction_plot_{timestamp}.png")
        plt.savefig(fig_path)
        plt.close()
        
        logger.info(f"Prediction plot saved to {fig_path}")


def main():
    """Main function to run the predictor."""
    # Initialize predictor
    predictor = StockPredictor()
    
    # Load models
    predictor.load_models()
    
    # Make predictions
    predictions = predictor.predict_next_candles()
    
    return predictions


if __name__ == "__main__":
    main()

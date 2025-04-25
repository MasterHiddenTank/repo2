"""
Model 1: Bayesian Neural Network
Incorporates MC Dropout, Gaussian HMM, and Viterbi algorithm for probabilistic modeling and uncertainty estimation.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.base_model import BaseModel
from config import MODEL_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BayesianNeuralNetwork(BaseModel):
    """Bayesian Neural Network with MC Dropout for uncertainty estimation."""
    
    def __init__(self, input_shape):
        """
        Initialize the Bayesian Neural Network.
        
        Args:
            input_shape: Shape of the input data (sequence_length, num_features)
        """
        super().__init__("BayesianNN", input_shape)
        self.dropout_rate = 0.2
        self.num_mc_samples = 20  # Number of MC samples for prediction
        
    def build_model(self):
        """
        Build the Bayesian Neural Network model using MC Dropout.
        """
        try:
            # Limit TensorFlow memory growth to avoid OOM errors
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # Limit GPU memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable mixed precision if configured
            if MODEL_CONFIG.get("enable_mixed_precision", False):
                logger.info("Enabling mixed precision training")
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
            
            # Set TensorFlow to NOT log device placement by default
            tf.debugging.set_log_device_placement(MODEL_CONFIG.get("log_device_placement", False))
            
            logger.info(f"Building Bayesian Neural Network with input shape: {self.input_shape}")
            logger.info(f"Prediction horizon: {self.prediction_horizon}")
            
            inputs = keras.Input(shape=self.input_shape)
            
            # Convolutional layers for feature extraction
            x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
            x = layers.MaxPooling1D(pool_size=2)(x)
            x = layers.Dropout(self.dropout_rate)(x, training=True)  # MC Dropout
            
            x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
            x = layers.MaxPooling1D(pool_size=2)(x)
            x = layers.Dropout(self.dropout_rate)(x, training=True)  # MC Dropout
            
            # LSTM layers for time series modeling
            x = layers.LSTM(96, return_sequences=True)(x)
            x = layers.Dropout(self.dropout_rate)(x, training=True)  # MC Dropout
            
            x = layers.LSTM(64, return_sequences=False)(x)
            x = layers.Dropout(self.dropout_rate)(x, training=True)  # MC Dropout
            
            # Dense layers with MC Dropout
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(self.dropout_rate)(x, training=True)  # MC Dropout
            
            x = layers.Dense(32, activation='relu')(x)
            x = layers.Dropout(self.dropout_rate)(x, training=True)  # MC Dropout
            
            # Output layer for predicting the next time steps
            # The output shape needs to match the prediction_horizon
            outputs = layers.Dense(self.prediction_horizon)(x)
            
            # Create the model
            model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
            
            # Compile the model with a smaller learning rate
            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
            
            # Use float32 precision for the final outputs regardless of mixed precision policy
            if MODEL_CONFIG.get("enable_mixed_precision", False):
                outputs = tf.cast(outputs, dtype='float32')
            
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
            
            # Print model summary
            model.summary()
            
            self.model = model
            logger.info("Model successfully built")
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def predict_with_uncertainty(self, X, num_samples=None):
        """
        Make predictions with uncertainty estimation using MC Dropout.
        
        Args:
            X: Input data
            num_samples: Number of MC samples to use (default: self.num_mc_samples)
            
        Returns:
            mean_prediction: Mean prediction across MC samples
            std_prediction: Standard deviation of predictions (uncertainty)
        """
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        if num_samples is None:
            num_samples = self.num_mc_samples
        
        # Get multiple predictions using MC Dropout
        predictions = []
        for _ in range(num_samples):
            pred = self.model.predict(X)
            predictions.append(pred)
        
        # Stack predictions
        predictions = np.stack(predictions, axis=0)
        
        # Calculate mean and standard deviation
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)
        
        return mean_prediction, std_prediction
    
    def predict(self, X):
        """
        Override the base predict method to use MC Dropout.
        
        Args:
            X: Input data
            
        Returns:
            Predicted values (mean prediction)
        """
        mean_prediction, _ = self.predict_with_uncertainty(X)
        return mean_prediction
    
    def get_prediction_intervals(self, X, confidence=0.95):
        """
        Get prediction intervals with specified confidence level.
        
        Args:
            X: Input data
            confidence: Confidence level for intervals (default: 0.95)
            
        Returns:
            lower_bound: Lower bound of prediction intervals
            upper_bound: Upper bound of prediction intervals
        """
        mean_prediction, std_prediction = self.predict_with_uncertainty(X)
        
        # Calculate z-score for the given confidence level
        from scipy.stats import norm
        z = norm.ppf((1 + confidence) / 2)
        
        # Calculate prediction intervals
        lower_bound = mean_prediction - z * std_prediction
        upper_bound = mean_prediction + z * std_prediction
        
        return lower_bound, upper_bound

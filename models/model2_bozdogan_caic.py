"""
Bozdogan's Consistent Akaike Information Criterion (CAIC) Neural Network model
for SPY stock price prediction. This model integrates fuzzy logic and stochastic
processes to balance model complexity and market noise.
"""

import os
import sys
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm
import joblib
import random

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import PATHS, MODEL_CONFIG
from models.base_model import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BozdoganCAICModel(BaseModel):
    """
    Neural Network using Bozdogan's Consistent Akaike Information Criterion
    with fuzzy logic and stochastic processes for SPY price prediction.
    """
    
    def __init__(self, input_shape):
        """
        Initialize the CAIC model.
        
        Args:
            input_shape: Shape of the input data (sequence_length, num_features)
        """
        name = "BozdoganCAIC"
        super().__init__(name, input_shape)
        
        # Additional model parameters - adjusted for better performance
        self.fuzzy_membership_threshold = 0.6  # Higher threshold for more selective fuzzy membership
        self.noise_scale = 0.005  # Reduced noise for more stable learning
        self.complexity_penalty = 0.0005  # Lower initial complexity penalty
        self.dropout_rate = 0.3  # Increased dropout for better generalization
        
    def build_model(self):
        """
        Build and compile the model architecture.
        
        Returns:
            Compiled Keras model
        """
        try:
            # Configure TensorFlow for memory efficiency
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable mixed precision if configured
            if MODEL_CONFIG.get("enable_mixed_precision", False):
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
            
            logger.info(f"Building CAIC Network with input shape: {self.input_shape}")
            logger.info(f"Prediction horizon: {self.prediction_horizon}")
            
            # Define the model architecture
            inputs = keras.Input(shape=self.input_shape)
            
            # Convolutional layers for feature extraction - adjusted for better pattern recognition
            x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(pool_size=2)(x)
            x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            
            # Two layers of Bidirectional LSTM for better temporal dynamics
            x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
            x = layers.Dropout(self.dropout_rate)(x)
            x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
            x = layers.Dropout(self.dropout_rate)(x)
            
            # Attention mechanism to focus on important time steps
            attention = layers.Dense(1, activation='tanh')(x)
            attention = layers.Flatten()(attention)
            attention = layers.Activation('softmax')(attention)
            attention = layers.RepeatVector(128)(attention)
            attention = layers.Permute([2, 1])(attention)
            
            # Apply attention
            x = layers.Multiply()([x, attention])
            x = layers.GlobalAveragePooling1D()(x)
            
            # Fuzzy layer - implemented as a custom lambda layer
            x = layers.Dense(64, activation='sigmoid')(x)
            x = layers.Lambda(self._fuzzy_layer)(x)
            
            # Stochastic layer - add random noise during training
            x = layers.Lambda(self._stochastic_layer)(x)
            
            # Dense layers
            x = layers.Dense(32, activation='relu')(x)
            x = layers.Dropout(self.dropout_rate)(x)
            
            # Output layer
            outputs = layers.Dense(self.prediction_horizon)(x)
            
            # Cast outputs to float32 for consistent loss calculation
            if MODEL_CONFIG.get("enable_mixed_precision", False):
                outputs = tf.cast(outputs, dtype='float32')
            
            # Create model
            model = keras.Model(inputs, outputs, name=self.name)
            
            # Custom optimizer with CAIC regularization
            optimizer = self._create_caic_optimizer()
            
            # Compile model
            model.compile(
                optimizer=optimizer,
                loss=self._caic_loss,
                metrics=['mae']
            )
            
            # Print model summary
            model.summary()
            
            self.model = model
            logger.info("CAIC model successfully built")
            return model
            
        except Exception as e:
            logger.error(f"Error building CAIC model: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _fuzzy_layer(self, x):
        """
        Custom fuzzy logic layer that applies fuzzy membership functions.
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor with fuzzy logic applied
        """
        # Apply fuzzy membership function (sigmoid already applied before)
        # We'll enhance the fuzzy effect by sharpening the membership values
        # Cast everything to the same dtype as x to avoid type mismatches
        x_dtype = x.dtype
        threshold = tf.cast(self.fuzzy_membership_threshold, x_dtype)
        fuzzy_mask = tf.cast(x > threshold, x_dtype)
        scale_factor = tf.cast(0.3, x_dtype)
        one = tf.cast(1.0, x_dtype)
        
        fuzzy_values = x * fuzzy_mask + (one - fuzzy_mask) * x * scale_factor
        return fuzzy_values
    
    def _stochastic_layer(self, x):
        """
        Add stochastic noise to the layer during training to improve robustness.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with stochastic noise added during training
        """
        def add_noise():
            # Get the dtype of x to ensure consistent types
            x_dtype = x.dtype
            noise_scale = tf.cast(self.noise_scale, x_dtype)
            noise = tf.random.normal(
                shape=tf.shape(x), 
                mean=tf.cast(0.0, x_dtype), 
                stddev=noise_scale,
                dtype=x_dtype
            )
            return x + noise
            
        def identity():
            return x
            
        # Only add noise during training
        # Convert learning phase to bool explicitly to avoid the int32/bool type error
        is_training = tf.cast(tf.keras.backend.learning_phase(), tf.bool)
        return tf.cond(is_training, add_noise, identity)
    
    def _create_caic_optimizer(self):
        """
        Create optimizer with learning rate schedule based on CAIC principles.
        
        Returns:
            Keras optimizer
        """
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )
        return keras.optimizers.Adam(learning_rate=lr_schedule)
    
    def _caic_loss(self, y_true, y_pred):
        """
        Custom loss function incorporating Bozdogan's CAIC.
        CAIC = -2*log(likelihood) + k*(log(n) + 1)
        where k is number of parameters and n is sample size.
        
        We approximate this by adding a complexity penalty to MSE.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            CAIC loss value
        """
        # Ensure consistent data types
        y_dtype = y_pred.dtype
        y_true = tf.cast(y_true, y_dtype)
        
        # Base MSE loss
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Count model parameters for complexity penalty
        # Since we can't directly count parameters in TF 2.x custom loss,
        # we use the complexity_penalty hyperparameter as a proxy
        complexity_term = tf.cast(self.complexity_penalty, y_dtype)
        
        # Combine into CAIC-inspired loss
        caic_loss = mse + complexity_term
        return caic_loss
    
    def predict_with_uncertainty(self, X, num_samples=30):
        """
        Make predictions with uncertainty estimation using stochastic forward passes.
        
        Args:
            X: Input data
            num_samples: Number of stochastic forward passes
            
        Returns:
            mean_prediction: Mean prediction across samples
            std_prediction: Standard deviation of predictions (uncertainty)
        """
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        # Set learning phase to 1 (training) to enable stochastic behavior
        # Use backend function to ensure proper tensor dtype
        original_learning_phase = tf.keras.backend.learning_phase()
        tf.keras.backend.set_learning_phase(1)
        
        # Get multiple predictions using stochastic passes
        predictions = []
        for _ in range(num_samples):
            pred = self.model.predict(X)
            predictions.append(pred)
        
        # Reset learning phase to original value
        tf.keras.backend.set_learning_phase(original_learning_phase)
        
        # Stack predictions
        predictions = np.stack(predictions, axis=0)
        
        # Calculate mean and standard deviation
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)
        
        return mean_prediction, std_prediction
    
    def predict(self, X):
        """
        Override the base predict method to use stochastic forward passes.
        
        Args:
            X: Input data
            
        Returns:
            Predicted values (mean prediction)
        """
        # For single predictions, use fewer stochastic passes
        mean_prediction, _ = self.predict_with_uncertainty(X, num_samples=10)
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
        z = norm.ppf((1 + confidence) / 2)
        
        # Calculate prediction intervals
        lower_bound = mean_prediction - z * std_prediction
        upper_bound = mean_prediction + z * std_prediction
        
        return lower_bound, upper_bound
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model with enhanced CAIC-based regularization.
        
        Args:
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data
            y_val: Validation target data
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Create callbacks
        callbacks = []
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Dynamic complexity penalty adjustment
        class CAICCallback(keras.callbacks.Callback):
            def __init__(self, model_instance):
                super().__init__()
                self.model_instance = model_instance
            
            def on_epoch_end(self, epoch, logs=None):
                # Adjust complexity penalty based on validation performance
                # Increase penalty if model is overfitting
                if epoch > 0 and logs.get('val_loss') > logs.get('loss'):
                    self.model_instance.complexity_penalty *= 1.05
                else:
                    self.model_instance.complexity_penalty *= 0.98
                
                # Log current penalty
                logger.info(f"Epoch {epoch+1}: complexity_penalty={self.model_instance.complexity_penalty:.6f}")
        
        callbacks.append(CAICCallback(self))
        
        # Train the model
        logger.info("Starting CAIC model training")
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("CAIC model training completed")
        return history 
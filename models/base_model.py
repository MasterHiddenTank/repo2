"""
Base model class for all neural network models.
This provides common functionality for all models in the project.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import joblib

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODEL_CONFIG, PATHS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseModel:
    """Base class for all neural network models."""
    
    def __init__(self, name, input_shape):
        """
        Initialize the base model.
        
        Args:
            name: Name of the model
            input_shape: Shape of the input data (sequence_length, num_features)
        """
        self.name = name
        self.input_shape = input_shape
        self.model = None
        self.accuracy_threshold = MODEL_CONFIG["accuracy_threshold"]
        self.max_epochs = MODEL_CONFIG["epochs"]
        self.patience = MODEL_CONFIG["patience"]
        self.batch_size = MODEL_CONFIG["batch_size"]
        self.learning_rate = MODEL_CONFIG["learning_rate"]
        self.prediction_horizon = MODEL_CONFIG["prediction_horizon"]
        
        # Create model directory if it doesn't exist
        os.makedirs(PATHS["model_dir"], exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def build_model(self):
        """
        Build the neural network model.
        This method should be implemented by each subclass.
        """
        raise NotImplementedError("Subclasses must implement build_model()")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data
            y_val: Validation target data
            
        Returns:
            Training history
        """
        import tensorflow as tf
        
        if self.model is None:
            self.build_model()
        
        # Log training parameters
        logger.info(f"Starting training on {len(X_train)} samples")
        logger.info(f"Batch size: {self.batch_size}, Epochs: {self.max_epochs}")
        
        # Memory optimization: for large datasets use gradient accumulation
        use_gradient_accumulation = len(X_train) > 5000
        if use_gradient_accumulation:
            logger.info(f"Using gradient accumulation for large dataset ({len(X_train)} samples)")
            # Use a smaller virtual batch size for memory efficiency
            virtual_batch_size = self.batch_size
            # Accumulate gradients over multiple batches
            accumulation_steps = 4
            # Effective batch size will be virtual_batch_size * accumulation_steps
            effective_batch_size = virtual_batch_size * accumulation_steps
            logger.info(f"Virtual batch size: {virtual_batch_size}, Accumulation steps: {accumulation_steps}, Effective batch size: {effective_batch_size}")
            
            original_batch_size = self.batch_size
            self.batch_size = virtual_batch_size
        else:
            # Memory optimization: set small batch size if data is large
            if len(X_train) > 500 and self.batch_size > 8:
                old_batch_size = self.batch_size
                self.batch_size = 8
                logger.info(f"Reduced batch size from {old_batch_size} to {self.batch_size} for memory efficiency")
            
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=self.patience,
                monitor='val_loss' if X_val is not None else 'loss',
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=int(self.patience / 3),
                min_lr=1e-6
            ),
            # Add a termination callback to prevent running out of memory
            keras.callbacks.TerminateOnNaN(),
        ]
        
        # Train the model with a try-except block to catch memory errors
        try:
            # Train the model
            logger.info(f"Training model {self.name}...")
            
            if use_gradient_accumulation:
                # Implement custom training loop with gradient accumulation for memory efficiency
                logger.info("Using custom training loop with gradient accumulation")
                
                # Split training data for validation if not provided
                if X_val is None or y_val is None:
                    from sklearn.model_selection import train_test_split
                    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                        X_train, y_train, test_size=0.2, random_state=42
                    )
                else:
                    X_train_split, X_val_split, y_train_split, y_val_split = X_train, X_val, y_train, y_val
                
                # Setup for gradient accumulation
                from tensorflow.keras import backend as K
                optimizer = self.model.optimizer
                train_loss = keras.metrics.Mean(name='train_loss')
                train_metrics = {m.name: keras.metrics.Mean(name=m.name) for m in self.model.metrics}
                val_loss = keras.metrics.Mean(name='val_loss')
                val_metrics = {m.name: keras.metrics.Mean(name=m.name) for m in self.model.metrics}
                
                # Training history dictionary
                history = {'loss': [], 'val_loss': []}
                for metric_name in train_metrics.keys():
                    history[metric_name] = []
                    history[f'val_{metric_name}'] = []
                
                early_stopping_counter = 0
                best_val_loss = float('inf')
                best_weights = None
                
                # Custom training loop
                for epoch in range(self.max_epochs):
                    logger.info(f"Epoch {epoch+1}/{self.max_epochs}")
                    
                    # Reset metrics
                    train_loss.reset_states()
                    for m in train_metrics.values():
                        m.reset_states()
                    val_loss.reset_states()
                    for m in val_metrics.values():
                        m.reset_states()
                    
                    # Shuffle training data
                    indices = np.arange(len(X_train_split))
                    np.random.shuffle(indices)
                    X_train_shuffled = X_train_split[indices]
                    y_train_shuffled = y_train_split[indices]
                    
                    # Training loop with gradient accumulation
                    progbar = tf.keras.utils.Progbar(len(X_train_shuffled), stateful_metrics=['loss'] + list(train_metrics.keys()))
                    
                    for idx in range(0, len(X_train_shuffled), virtual_batch_size * accumulation_steps):
                        end_idx = min(idx + virtual_batch_size * accumulation_steps, len(X_train_shuffled))
                        X_batch_big = X_train_shuffled[idx:end_idx]
                        y_batch_big = y_train_shuffled[idx:end_idx]
                        
                        # Initialize gradients to zero
                        gradients = [tf.zeros_like(var) for var in self.model.trainable_variables]
                        
                        # Process in virtual batches
                        for batch_idx in range(0, len(X_batch_big), virtual_batch_size):
                            batch_end_idx = min(batch_idx + virtual_batch_size, len(X_batch_big))
                            X_virtual_batch = X_batch_big[batch_idx:batch_end_idx]
                            y_virtual_batch = y_batch_big[batch_idx:batch_end_idx]
                            
                            # Forward and backward pass
                            with tf.GradientTape() as tape:
                                y_pred = self.model(X_virtual_batch, training=True)
                                batch_loss = self.model.loss(y_virtual_batch, y_pred)
                            
                            # Calculate gradients
                            batch_gradients = tape.gradient(batch_loss, self.model.trainable_variables)
                            
                            # Accumulate gradients
                            for i, grad in enumerate(batch_gradients):
                                if grad is not None:
                                    gradients[i] += grad * (len(X_virtual_batch) / virtual_batch_size / accumulation_steps)
                            
                            # Update metrics
                            batch_loss_value = batch_loss.numpy()
                            train_loss.update_state(batch_loss_value)
                            
                            # Update other metrics
                            y_pred_numpy = y_pred.numpy()
                            for metric_name, metric in train_metrics.items():
                                if metric_name == 'mae':
                                    mae_value = np.mean(np.abs(y_virtual_batch - y_pred_numpy))
                                    metric.update_state(mae_value)
                        
                        # Apply accumulated gradients
                        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                        
                        # Update progress bar
                        progbar.update(end_idx, values=[('loss', train_loss.result().numpy())] + 
                                      [(name, metric.result().numpy()) for name, metric in train_metrics.items()])
                    
                    # Validation step
                    for idx in range(0, len(X_val_split), self.batch_size):
                        end_idx = min(idx + self.batch_size, len(X_val_split))
                        X_val_batch = X_val_split[idx:end_idx]
                        y_val_batch = y_val_split[idx:end_idx]
                        
                        y_pred = self.model(X_val_batch, training=False)
                        batch_val_loss = self.model.loss(y_val_batch, y_pred)
                        
                        # Update metrics
                        val_loss.update_state(batch_val_loss)
                        
                        # Update other metrics
                        y_pred_numpy = y_pred.numpy()
                        for metric_name, metric in val_metrics.items():
                            if metric_name == 'mae':
                                mae_value = np.mean(np.abs(y_val_batch - y_pred_numpy))
                                metric.update_state(mae_value)
                    
                    # Update history
                    history['loss'].append(train_loss.result().numpy())
                    history['val_loss'].append(val_loss.result().numpy())
                    for metric_name in train_metrics.keys():
                        history[metric_name].append(train_metrics[metric_name].result().numpy())
                        history[f'val_{metric_name}'].append(val_metrics[metric_name].result().numpy())
                    
                    # Log epoch results
                    logger.info(f"Epoch {epoch+1}/{self.max_epochs}: loss={history['loss'][-1]:.4f}, val_loss={history['val_loss'][-1]:.4f}")
                    
                    # Check for early stopping
                    current_val_loss = val_loss.result().numpy()
                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        best_weights = self.model.get_weights()
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                        if early_stopping_counter >= self.patience:
                            logger.info(f"Early stopping triggered after {epoch+1} epochs")
                            break
                
                # Restore best weights
                if best_weights:
                    self.model.set_weights(best_weights)
                
                # Convert history to tf.keras.callbacks.History object for compatibility
                history_obj = type('History', (), {})()
                history_obj.history = history
                history_obj.params = {'epochs': len(history['loss'])}
                
                # Restore original batch size
                self.batch_size = original_batch_size
                
                return history_obj
            else:
                # Standard Keras training
                if X_val is not None and y_val is not None:
                    history = self.model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=self.max_epochs,
                        batch_size=self.batch_size,
                        callbacks=callbacks,
                        verbose=1
                    )
                else:
                    # Split training data for validation
                    from sklearn.model_selection import train_test_split
                    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                        X_train, y_train, test_size=0.2, random_state=42
                    )
                    history = self.model.fit(
                        X_train_split, y_train_split,
                        validation_data=(X_val_split, y_val_split),
                        epochs=self.max_epochs,
                        batch_size=self.batch_size,
                        callbacks=callbacks,
                        verbose=1
                    )
                
                return history
        except tf.errors.ResourceExhaustedError as e:
            logger.error(f"Memory error during training: {e}")
            # Reduce batch size and try again if possible
            if self.batch_size > 1:
                old_batch_size = self.batch_size
                self.batch_size = max(1, self.batch_size // 2)
                logger.info(f"Reducing batch size from {old_batch_size} to {self.batch_size} and retrying")
                return self.train(X_train, y_train, X_val, y_val)
            else:
                logger.error("Could not train model even with batch size of 1")
                raise
        except Exception as e:
            logger.error(f"Error during training: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test input data
            y_test: Test target data
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        try:
            logger.info(f"Evaluating model {self.name} on {len(X_test)} test samples")
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            logger.info(f"Prediction shape: {y_pred.shape}, Target shape: {y_test.shape}")
            
            # Ensure shapes match for comparison
            if y_pred.shape != y_test.shape:
                logger.warning(f"Shape mismatch: y_pred: {y_pred.shape}, y_test: {y_test.shape}")
                
                # If y_test is 1D and y_pred is 2D with one column
                if len(y_test.shape) == 1 and len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
                    logger.info("Reshaping y_pred from 2D to 1D")
                    y_pred = y_pred.flatten()
                # If y_pred is 1D and y_test is 2D with one column
                elif len(y_pred.shape) == 1 and len(y_test.shape) == 2 and y_test.shape[1] == 1:
                    logger.info("Reshaping y_test from 2D to 1D")
                    y_test = y_test.flatten()
                else:
                    logger.warning("Cannot reconcile shape mismatch automatically")
            
            # Calculate metrics
            mse = np.mean(np.square(y_pred - y_test))
            mae = np.mean(np.abs(y_pred - y_test))
            
            # Calculate directional accuracy
            correct_direction = 0
            total_points = 0
            
            # Handle based on dimensionality
            if len(y_test.shape) == 1:  # 1D outputs
                # For 1D data, compare consecutive predictions
                for i in range(1, len(y_test)):
                    actual_direction = y_test[i] - y_test[i-1]
                    pred_direction = y_pred[i] - y_pred[i-1]
                    
                    if (actual_direction * pred_direction) > 0:
                        correct_direction += 1
                    
                    total_points += 1
            else:  # Multiple steps (2D outputs)
                for i in range(len(y_test)):
                    for j in range(1, y_test.shape[1]):
                        # Check if the model correctly predicted the direction of price movement
                        actual_direction = y_test[i, j] - y_test[i, j-1]
                        pred_direction = y_pred[i, j] - y_pred[i, j-1]
                        
                        if (actual_direction * pred_direction) > 0:
                            correct_direction += 1
                        
                        total_points += 1
            
            directional_accuracy = correct_direction / total_points if total_points > 0 else 0
            
            metrics = {
                "mse": float(mse),
                "mae": float(mae),
                "directional_accuracy": float(directional_accuracy)
            }
            
            logger.info(f"Model {self.name} evaluation:")
            logger.info(f"MSE: {mse:.6f}")
            logger.info(f"MAE: {mae:.6f}")
            logger.info(f"Directional Accuracy: {directional_accuracy:.2%}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return zeroed metrics on error
            return {
                "mse": 0.0,
                "mae": 0.0,
                "directional_accuracy": 0.0
            }
    
    def predict(self, X):
        """
        Make predictions with the model.
        
        Args:
            X: Input data
            
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        return self.model.predict(X)
    
    def save(self, model_path=None, metadata=None):
        """
        Save the model to disk.
        
        Args:
            model_path: Optional path to save the model (if None, uses default path)
            metadata: Optional metadata to save with the model
            
        Returns:
            Path where the model was saved
        """
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        try:
            # Create a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Use the provided path or create one based on the name and timestamp
            if model_path is None:
                # Create the models directory if it doesn't exist
                model_dir = os.path.join(PATHS.get("model_dir", os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved")))
                os.makedirs(model_dir, exist_ok=True)
                
                # Create a subdirectory for this model
                model_path = os.path.join(model_dir, f"{self.name}_{timestamp}")
                os.makedirs(model_path, exist_ok=True)
            
            logger.info(f"Saving model to {model_path}")
            
            # TensorFlow SavedModel format
            self.model.save(model_path, save_format='tf')
            
            # Save metadata
            if metadata is None:
                metadata = {}
            
            # Add basic metadata
            metadata.update({
                "name": self.name,
                "timestamp": timestamp,
                "input_shape": self.input_shape,
                "prediction_horizon": self.prediction_horizon,
                "learning_rate": self.learning_rate,
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            
            # Save the metadata
            import joblib
            metadata_path = f"{model_path}_metadata.joblib"
            joblib.dump(metadata, metadata_path)
            
            logger.info(f"Model saved to {model_path} with metadata")
            return model_path
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def load(self, model_path):
        """
        Load the model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        self.model = keras.models.load_model(model_path)
        logger.info(f"Model {self.name} loaded from {model_path}")
        return self.model
    
    def plot_training_history(self, history):
        """
        Plot the training history.
        
        Args:
            history: Training history returned by model.fit()
        """
        plt.figure(figsize=(12, 5))
        
        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'{self.name} - Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        # Plot learning rate
        if 'lr' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['lr'])
            plt.title('Learning Rate')
            plt.ylabel('Learning Rate')
            plt.xlabel('Epoch')
            plt.yscale('log')
        
        # Save the figure
        os.makedirs(PATHS["results_dir"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(PATHS["results_dir"], f"{self.name}_history_{timestamp}.png")
        plt.savefig(fig_path)
        plt.close()
        
        logger.info(f"Training history plot saved to {fig_path}")
        
        return fig_path
    
    def plot_predictions(self, X_test, y_test, num_samples=5):
        """
        Plot predictions vs actual values.
        
        Args:
            X_test: Test input data
            y_test: Test target data
            num_samples: Number of samples to plot
        """
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        # Make predictions
        y_pred = self.model.predict(X_test[:num_samples])
        
        plt.figure(figsize=(15, 10))
        for i in range(min(num_samples, len(y_test))):
            plt.subplot(num_samples, 1, i+1)
            plt.plot(y_test[i], 'b-', label='Actual')
            plt.plot(y_pred[i], 'r-', label='Predicted')
            plt.title(f'Sample {i+1}')
            plt.ylabel('Price (Scaled)')
            plt.xlabel('Time Step')
            plt.legend()
        
        plt.tight_layout()
        
        # Save the figure
        os.makedirs(PATHS["results_dir"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(PATHS["results_dir"], f"{self.name}_predictions_{timestamp}.png")
        plt.savefig(fig_path)
        plt.close()
        
        logger.info(f"Predictions plot saved to {fig_path}")
        
        return fig_path

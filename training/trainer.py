"""
Trainer module for training neural network models.
"""

import os
import sys
import numpy as np
import logging
from datetime import datetime
import time
import joblib
from sklearn.model_selection import train_test_split

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODEL_CONFIG, PATHS
from data.data_loader import DataLoader
from models.model1_bayesian_neural_network import BayesianNeuralNetwork
from models.model2_bozdogan_caic import BozdoganCAICModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class for training neural network models."""
    
    def __init__(self):
        """Initialize the model trainer."""
        self.data_loader = DataLoader()
        self.accuracy_threshold = MODEL_CONFIG["accuracy_threshold"]
        self.sequence_length = 60  # Number of time steps to use as input
        self.models = {}
        self.training_stats = {}
        self.enable_preloading = False  # Add missing attribute
    
    def initialize_models(self, input_shape=None):
        """
        Initialize the models to be trained.
        
        Args:
            input_shape: The input shape for the model (if None, it will be determined from data)
        """
        try:
            logger.info("Initializing neural network models...")
            self.models = {}
            
            # Load existing models if we had trained them before and can reuse
            if self.enable_preloading and self.check_models_exist():
                logger.info("Loading existing models for continued training...")
                try:
                    model_names = self.list_models()
                    for model_name in model_names:
                        self.models[model_name] = self.load_model(model_name)
                    logger.info(f"Successfully loaded {len(self.models)} existing models")
                except Exception as e:
                    logger.error(f"Error loading existing models: {str(e)}. Creating new ones.")
                    # Continue to create new models if loading fails
            
            # If no models were loaded or preloading is disabled, create new ones
            if not self.models:
                if input_shape is None:
                    # Default input shape as fallback
                    logger.warning("No input shape provided, using default (60, 10)")
                    input_shape = (60, 10)
                
                logger.info(f"Creating neural network models with input_shape={input_shape}")
                
                # Model 1: Bayesian Neural Network
                try:
                    from models.model1_bayesian_neural_network import BayesianNeuralNetwork
                    self.models["model1"] = BayesianNeuralNetwork(input_shape)
                    logger.info("Initialized Model 1: Bayesian Neural Network")
                except Exception as e:
                    logger.error(f"Failed to initialize BayesianNeuralNetwork: {str(e)}")
                
                # Model 2: Bozdogan Consistent AIC Model
                try:
                    from models.model2_bozdogan_caic import BozdoganCAICModel
                    self.models["model2"] = BozdoganCAICModel(input_shape)
                    logger.info("Initialized Model 2: Bozdogan Consistent AIC")
                except Exception as e:
                    logger.error(f"Failed to initialize BozdoganCAICModel: {str(e)}")
                
                # Skip attempting to load models 3-9 as they don't exist yet
                logger.info("Models 3-9 are not implemented yet, skipping initialization")
                
            return self.models
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def get_model_type(self, model_name):
        """Determine the model type from the model name."""
        if not model_name:
            return "unknown"
        
        model_name = model_name.lower()
        
        if model_name == "model1" or "bayesian" in model_name:
            return "BayesianNeuralNetwork"
        elif model_name == "model2" or "bozdogan" in model_name or "caic" in model_name or "bozdogancaic" in model_name:
            return "BozdoganCAIC"
        elif model_name == "model3" or "viterbi" in model_name or "baum" in model_name:
            return "ViterbiBaumWelch"
        elif model_name == "model4" or "gaussian" in model_name or "hmm" in model_name:
            return "GaussianHMM"
        elif model_name == "model5" or "fuzzy" in model_name:
            return "FuzzyStochastic"
        elif model_name == "model6" or "decision" in model_name or "tree" in model_name or "fork" in model_name:
            return "DecisionTree"
        elif model_name == "model7" or "kronecker" in model_name or "laplace" in model_name:
            return "KroneckerFactored"
        elif model_name == "model8" or "stochastic" in model_name or "gradient" in model_name:
            return "StochasticGradient"
        elif model_name == "model9" or "dropout" in model_name or "mc" in model_name:
            return "MCDropout"
        else:
            return "unknown"

    def train_model(self, model_name, max_attempts=3, X=None, y=None):
        """
        Train a specific model until it meets the accuracy threshold or max attempts.
        Utilizes previous training when making new attempts with different months.
        
        Args:
            model_name: The name of the model to train
            max_attempts: Maximum number of training attempts
            X: Optional training data features (if None, will load from file)
            y: Optional training data targets (if None, will load from file)
            
        Returns:
            tuple: (model, accuracy, history) of the trained model
        """
        # Standardize model_name to lowercase for case-insensitive matching
        model_name_lower = model_name.lower() if model_name else None
        
        # Handle special case for 'model2' or 'bozdogancaic'
        if model_name_lower in ["model2", "bozdogancaic", "bozdogan", "caic"]:
            model_name = "model2"  # Normalize to the standard name
        
        # Identify the model type
        model_type = self.get_model_type(model_name)
        logger.info(f"Training model: {model_name} (Type: {model_type})")
        
        # Get the model instance
        model = self.models.get(model_name)
        if model is None:
            logger.error(f"Model {model_name} not found")
            return None, 0.0, None
        
        # Track training history across attempts
        cumulative_history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}
        model_metadata = {
            "training_attempts": 0,
            "months_trained": [],
            "final_accuracy": 0.0
        }
        
        # Keep track if this is our first attempt
        first_attempt = True
        
        for attempt in range(max_attempts):
            model_metadata["training_attempts"] += 1
            logger.info(f"Training attempt {attempt+1}/{max_attempts} for {model_name}")
            
            # First attempt: initialize the model if it doesn't have weights yet
            # Subsequent attempts: keep existing weights to preserve learning
            if first_attempt and not hasattr(model, 'model') or model.model is None:
                logger.info("First training attempt - building model from scratch")
                first_attempt = False
                model.build_model()
            else:
                logger.info("Using existing model weights from previous training")
                # Model already exists, we'll continue training with the same weights
            
            try:
                # Load training data if not provided
                if X is None or y is None:
                    logger.info("Loading data for training...")
                    try:
                        # Try to load from file - select random month
                        data = self.data_loader.select_random_month(from_years=2)
                        if data is None or len(data) < 100:
                            logger.error("Failed to load enough training data")
                            continue  # Try another attempt
                        
                        # Track which month we're using for training
                        if hasattr(data, 'index') and len(data.index) > 0:
                            try:
                                # Get the first timestamp and extract month/year
                                first_date = data.index[0]
                                month_year = first_date.strftime("%Y-%m")
                                model_metadata["months_trained"].append(month_year)
                                logger.info(f"Training on month: {month_year}")
                            except Exception as e:
                                logger.warning(f"Could not determine month from data: {e}")
                            
                        # Process data into training samples
                        X, y = self.data_loader.prepare_training_data(data)
                        
                        if X is None or y is None or len(X) < 100:
                            logger.error("Failed to prepare training data")
                            continue  # Try another attempt
                        
                        logger.info(f"Prepared {len(X)} training samples")
                    except Exception as e:
                        logger.error(f"Error loading training data: {e}")
                        continue  # Try another attempt
                
                # Split data into train, validation, and test sets
                from sklearn.model_selection import train_test_split
                
                # First split: training + validation vs. test
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=self.test_split, random_state=42)
                
                # Second split: training vs. validation
                val_size = self.validation_split / (1 - self.test_split)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_size, random_state=42)
                
                logger.info(f"Training data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
                logger.info(f"Validation data shapes - X_val: {X_val.shape}, y_val: {y_val.shape}")
                logger.info(f"Test data shapes - X_test: {X_test.shape}, y_test: {y_test.shape}")
                
                # Train the model with the new data, continuing from previous training
                start_time = time.time()
                logger.info(f"Starting model training for {model_name} (attempt {attempt+1})...")
                
                # Train with a slightly higher learning rate on first attempt, lower on subsequent
                if model_metadata["training_attempts"] > 1:
                    # Reduce learning rate for fine-tuning on subsequent attempts
                    if hasattr(model, 'learning_rate'):
                        original_lr = model.learning_rate
                        model.learning_rate = model.learning_rate * 0.5
                        logger.info(f"Reduced learning rate to {model.learning_rate} for fine-tuning")
                
                # Train the model
                try:
                    history = model.train(X_train, y_train, X_val, y_val)
                    
                    # Restore original learning rate if changed
                    if model_metadata["training_attempts"] > 1 and hasattr(model, 'learning_rate') and 'original_lr' in locals():
                        model.learning_rate = original_lr
                    
                    # Extend the cumulative history
                    for key in history.history:
                        if key in cumulative_history:
                            cumulative_history[key].extend(history.history[key])
                    
                    # Training time
                    training_time = time.time() - start_time
                    logger.info(f"Training completed in {training_time:.2f} seconds")
                    
                    # Evaluate the model
                    evaluation = model.evaluate(X_test, y_test)
                    accuracy = 1.0 - evaluation.get("mae", 0.0)  # Use 1-MAE as accuracy
                    logger.info(f"Model accuracy: {accuracy:.4f}")
                    
                    # Check if the model met the accuracy threshold
                    if accuracy >= self.accuracy_threshold:
                        logger.info(f"Model met accuracy threshold of {self.accuracy_threshold}")
                        
                        # Save model metadata
                        model_metadata["final_accuracy"] = accuracy
                        
                        # Convert history to object form for compatibility
                        history_obj = type('History', (), {})()
                        history_obj.history = cumulative_history
                        history_obj.params = {'epochs': len(cumulative_history['loss'])}
                        
                        # Save the model
                        model.save(metadata=model_metadata)
                        
                        return model, accuracy, history_obj
                    else:
                        logger.info(f"Model did not meet accuracy threshold (got {accuracy:.4f}, need {self.accuracy_threshold})")
                        # Continue to the next attempt
                except Exception as training_error:
                    logger.error(f"Error during training: {training_error}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Continue to the next attempt
            except Exception as e:
                logger.error(f"Error in training attempt: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Continue to the next attempt
        
        # If we get here, all attempts failed
        logger.warning(f"All {max_attempts} training attempts failed to meet accuracy threshold")
        
        # Save the model anyway with the best accuracy we achieved
        try:
            # Convert history to object form for compatibility
            history_obj = type('History', (), {})()
            history_obj.history = cumulative_history
            history_obj.params = {'epochs': len(cumulative_history['loss'])}
            
            # Save the model with metadata
            model_metadata["final_accuracy"] = accuracy if 'accuracy' in locals() else 0.0
            model.save(metadata=model_metadata)
            
            return model, model_metadata["final_accuracy"], history_obj
        except Exception as save_error:
            logger.error(f"Error saving model after failed attempts: {save_error}")
            return model, 0.0, None
    
    def train_all_models(self, X=None, y=None):
        """
        Train all models until they reach the accuracy threshold.
        
        Args:
            X: Optional pre-loaded training data (features)
            y: Optional pre-loaded target data
            
        Returns:
            successful_models: Dictionary of successfully trained models
        """
        successful_models = {}
        
        for model_name in self.models:
            model, is_successful = self.train_model(model_name, X=X, y=y)
            if is_successful:
                successful_models[model_name] = model
        
        logger.info(f"Successfully trained {len(successful_models)}/{len(self.models)} models")
        return successful_models
    
    def _save_training_stats(self):
        """Save training statistics to disk."""
        # Create stats directory if it doesn't exist
        stats_dir = os.path.join(PATHS["results_dir"], "training_stats")
        os.makedirs(stats_dir, exist_ok=True)
        
        # Save training stats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_file = os.path.join(stats_dir, f"training_stats_{timestamp}.joblib")
        joblib.dump(self.training_stats, stats_file)
        
        logger.info(f"Saved training statistics to {stats_file}")

    def check_models_exist(self):
        """Check if saved models exist."""
        saved_models_dir = PATHS.get("models_dir")
        if not os.path.exists(saved_models_dir):
            logger.warning(f"Models directory {saved_models_dir} does not exist")
            return False
        
        # Check if any model files exist
        model_files = [f for f in os.listdir(saved_models_dir) if f.endswith('.h5') or os.path.isdir(os.path.join(saved_models_dir, f))]
        
        if not model_files:
            logger.warning("No saved models found")
            return False
        
        logger.info(f"Found {len(model_files)} potential model files/directories")
        return True

    def list_models(self):
        """List available trained models."""
        saved_models_dir = PATHS.get("models_dir")
        if not os.path.exists(saved_models_dir):
            logger.warning(f"Models directory {saved_models_dir} does not exist")
            return []
        
        # Look for model directories (TensorFlow SavedModel format)
        model_dirs = [d for d in os.listdir(saved_models_dir) 
                     if os.path.isdir(os.path.join(saved_models_dir, d)) 
                     and not d.startswith('.')]
        
        # Extract model names - typically in format ModelName_timestamp
        model_names = []
        for model_dir in model_dirs:
            parts = model_dir.split('_')
            if len(parts) > 0:
                model_name = parts[0].lower()
                # Map to standard model names
                if model_name == "bayesiannn":
                    model_names.append("model1")
                elif model_name == "bozdogancaic":
                    model_names.append("model2")
                elif model_name == "viterbibaumwelch":
                    model_names.append("model3")
                elif model_name == "gaussianhmm":
                    model_names.append("model4")
                else:
                    # Try to map other names
                    for i, std_name in enumerate(["model1", "model2", "model3", "model4"]):
                        if std_name.lower() in model_dir.lower():
                            model_names.append(std_name)
                            break
        
        unique_models = list(set(model_names))  # Remove duplicates
        logger.info(f"Found {len(unique_models)} unique models: {unique_models}")
        return unique_models

    def load_model(self, model_name):
        """Load a model by name."""
        try:
            if model_name == "model1":
                from models.model1_bayesian_neural_network import BayesianNeuralNetwork
                model = BayesianNeuralNetwork(input_shape=None)  # Input shape will be loaded from saved model
                model.load()
                return model
            elif model_name == "model2":
                from models.model2_bozdogan_caic import BozdoganCAICModel
                model = BozdoganCAICModel(input_shape=None)
                model.load()
                return model
            else:
                logger.warning(f"Model {model_name} not available - only models 1 and 2 are implemented")
                return None
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return None


def main():
    """Main function to run the model training."""
    # Create data loader and download latest data
    data_loader = DataLoader()
    data = data_loader.download_latest_data()
    
    # Process the data
    processed_data = data_loader.generate_features(data)
    
    # Prepare sample data to get input shape
    X_sample, _ = data_loader.prepare_training_data(processed_data)
    input_shape = X_sample.shape[1:]
    
    # Initialize trainer and models
    trainer = ModelTrainer()
    trainer.initialize_models(input_shape)
    
    # Train all models
    successful_models = trainer.train_all_models()
    
    return successful_models


if __name__ == "__main__":
    main()

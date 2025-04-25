"""
Flask web application for the SPY stock price prediction system.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request, redirect, url_for, send_from_directory
from datetime import datetime

# Configure Flask's reloader to ignore TensorFlow temporary files
import os
os.environ['PYTHONWARNINGS'] = "ignore"  # Suppress deprecation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow noise

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import WEB_CONFIG, PATHS, DATA_CONFIG, MODEL_CONFIG
from data.polygon_manager import PolygonManager
from prediction.predictor import StockPredictor
from training.trainer import ModelTrainer
from models.model1_bayesian_neural_network import BayesianNeuralNetwork
from models.model2_bozdogan_caic import BozdoganCAICModel

# Set up logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Starting SPY price prediction web application")

# Try creating logs directory
try:
    os.makedirs(PATHS["logs_dir"], exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(PATHS["logs_dir"], 'app.log'))
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.info("Logging to file enabled")
except Exception as e:
    logger.error(f"Could not set up file logging: {e}")
    logger.info("Continuing with console logging only")

# Initialize Flask app
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# Initialize global objects
# Using Polygon.io for SPY 1-minute data
data_loader = PolygonManager()
predictor = StockPredictor()
trainer = None  # Will be initialized when needed to save memory

# Global variables for tracking training status
training_status = {
    "in_progress": False,
    "model_name": None,
    "progress": 0,
    "status_message": "Not started",
    "last_updated": None
}

# Training background processes
training_processes = {}

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Render the dashboard page."""
    return render_template('dashboard.html', 
                          training_status=training_status)

@app.route('/api/data/download/latest', methods=['POST'])
def download_latest_data():
    """API endpoint to download the latest SPY 1-minute data"""
    try:
        # Force refresh if requested
        force_refresh = request.json.get('force_refresh', False)
        days = request.json.get('days', 7)
        
        # Download latest data
        success, message, data_or_stats = data_loader.download_latest_data(days=days, force_refresh=force_refresh)
        
        # Convert DataFrame to stats if necessary
        stats = {}
        if success and isinstance(data_or_stats, pd.DataFrame):
            df = data_or_stats
            stats = {
                "total_rows": len(df),
                "date_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                "trading_days": len(df['timestamp'].dt.date.unique()),
                "average_rows_per_day": int(len(df) / max(len(df['timestamp'].dt.date.unique()), 1))
            }
        elif success and isinstance(data_or_stats, dict):
            stats = data_or_stats
        
        return jsonify({
            'success': success,
            'message': message,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Error in download_latest_data: {e}")
        return jsonify({
            'success': False,
            'message': f"Error: {str(e)}"
        })

@app.route('/api/data/download/historical', methods=['POST'])
def download_historical_data():
    """API endpoint to download historical SPY 1-minute data for training"""
    try:
        # Get number of years to download (default to 2)
        years = request.json.get('years', 2)
        
        # Download historical data
        success, message, data_or_stats = data_loader.download_historical_data(years=years)
        
        # Convert DataFrame to stats if necessary
        stats = {}
        if success and isinstance(data_or_stats, pd.DataFrame):
            df = data_or_stats
            # Calculate statistics from the data
            stats = {
                "total_rows": len(df),
                "date_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                "years_covered": len(df['timestamp'].dt.year.unique()),
                "months_covered": len(df['timestamp'].dt.strftime('%Y-%m').unique())
            }
        elif success and isinstance(data_or_stats, dict):
            stats = data_or_stats
        
        return jsonify({
            'success': success,
            'message': message,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Error in download_historical_data: {e}")
        return jsonify({
            'success': False,
            'message': f"Error: {str(e)}"
        })

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """API endpoint to start model training."""
    global trainer, training_status
    
    try:
        # Check if training is already in progress
        if training_status["in_progress"]:
            return jsonify({
                "success": False, 
                "message": f"Training already in progress for {training_status['model_name']}"
            }), 400
        
        # Get request parameters
        data = request.get_json() or {}
        model_name = data.get("model_name")  # If None, will train all models
        
        # Update training status
        training_status = {
            "in_progress": True,
            "model_name": model_name if model_name else "all models",
            "progress": 0,
            "status_message": "Initializing training...",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Clean up memory before starting
        import gc
        import tensorflow as tf
        gc.collect()
        tf.keras.backend.clear_session()
        
        # Initialize trainer if not already done
        if trainer is None:
            logger.info("Initializing ModelTrainer")
            trainer = ModelTrainer()
        
        # Start training process
        training_status["status_message"] = "Starting model training..."
        training_status["progress"] = 60
        
        # Download and process data for training - random month from historical data
        training_status["status_message"] = "Selecting random month of historical data..."
        training_status["progress"] = 5
        success, message, data = data_loader.select_random_month(from_years=2)  # Explicitly set to 2 years
        
        if not success or data is None:
            training_status["in_progress"] = False
            training_status["status_message"] = f"Failed to load training data: {message}"
            return jsonify({"success": False, "message": message}), 500
        
        # Process the data
        training_status["status_message"] = "Generating features..."
        training_status["progress"] = 15
        try:
            processed_data = data_loader.generate_features(data)
            logger.info(f"Features generated successfully for {len(processed_data)} rows")
        except Exception as feature_error:
            logger.error(f"Error generating features: {str(feature_error)}")
            training_status["in_progress"] = False
            training_status["status_message"] = f"Error generating features: {str(feature_error)}"
            return jsonify({"success": False, "message": f"Error generating features: {str(feature_error)}"}), 500
        
        # Prepare training data
        training_status["status_message"] = "Preparing training data..."
        training_status["progress"] = 30
        
        try:
            logger.info("Preparing training data")
            try:
                # Process the full dataset in chunks to manage memory usage
                logger.info(f"Total data size: {len(processed_data)} rows")
                
                # For training, we'll use sequence-based data preparation
                # Save the scalers first with a small batch of data
                logger.info("Fitting scalers on the entire dataset")
                # Get the features to use from config
                features = DATA_CONFIG.get("features", [])
                if not features:
                    # Default features if not specified
                    features = processed_data.columns.tolist() 
                    if 'timestamp' in features:
                        features.remove('timestamp')
                else:
                    # Convert feature names to lowercase to match processed data
                    processed_columns = processed_data.columns.str.lower().tolist()
                    features_lower = [f.lower() for f in features]
                    
                    # If features don't match processed data columns, use available columns
                    if not all(f in processed_columns for f in features_lower):
                        logger.warning(f"Some configured features not found in data. Using available columns instead.")
                        features = processed_data.columns.tolist()
                        if 'timestamp' in features:
                            features.remove('timestamp')
                    else:
                        # Map the lowercase feature names to actual column names
                        column_map = {col.lower(): col for col in processed_data.columns}
                        features = [column_map.get(f.lower(), f) for f in features]
                
                logger.info(f"Using features: {features}")
                df = processed_data[features].copy()
                
                # Prepare scalers once for the entire dataset
                from sklearn.preprocessing import MinMaxScaler
                scalers = {}
                for column in df.columns:
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    # Fit the scaler on all data
                    scaler.fit(df[column].values.reshape(-1, 1))
                    scalers[column] = scaler
                    
                # Save scalers in the data_loader for later reuse
                data_loader.scalers = scalers
                
                # Sequence length from config
                sequence_length = MODEL_CONFIG.get("sequence_length", 60)
                prediction_horizon = MODEL_CONFIG.get("prediction_horizon", 10)
                
                logger.info(f"Using sequence_length={sequence_length}, prediction_horizon={prediction_horizon}")
                
                # Function to create training sequences from a chunk of data
                def create_sequences(data_chunk):
                    # Scale the data using pre-fitted scalers
                    scaled_data = pd.DataFrame()
                    for column in data_chunk.columns:
                        if column in scalers:
                            scaled_data[column] = scalers[column].transform(data_chunk[column].values.reshape(-1, 1)).flatten()
                        else:
                            scaled_data[column] = data_chunk[column]
                    
                    # Create X (input sequences) and y (next candle prices)
                    X, y = [], []
                    
                    # We need at least sequence_length + prediction_horizon data points
                    for i in range(len(scaled_data) - sequence_length - prediction_horizon + 1):
                        # Input sequence
                        X.append(scaled_data.iloc[i:i+sequence_length].values)
                        
                        # Target: next prediction_horizon candle Close prices
                        next_candles = []
                        for j in range(1, prediction_horizon + 1):
                            # Use lowercase 'close' to match column names in the data
                            next_candles.append(scaled_data.iloc[i+sequence_length+j-1]['close'])
                        
                        y.append(next_candles)
                    
                    return np.array(X), np.array(y)
                
                # First, we'll create a small dataset to initialize the model
                # Set a default chunk_size that will be overridden based on the model type
                chunk_size = 3000  # Default value
                chunk_epochs = 10  # Default epochs per chunk
                overlap = sequence_length + prediction_horizon  # Ensure continuous sequences between chunks
                
                # Get initial chunk for model initialization
                initial_data = processed_data.iloc[:chunk_size]
                X_init, y_init = create_sequences(initial_data[features])
                
                logger.info(f"Initial X shape: {X_init.shape}, y shape: {y_init.shape}")
                
                # Check for NaN values
                if np.isnan(X_init).any():
                    logger.warning("X contains NaN values - replacing with 0")
                    X_init = np.nan_to_num(X_init, nan=0.0)
                    
                if np.isnan(y_init).any():
                    logger.warning("y contains NaN values - replacing with 0")
                    y_init = np.nan_to_num(y_init, nan=0.0)
                
                # Get input shape for model initialization
                input_shape = X_init.shape[1:]
                logger.info(f"Input shape: {input_shape}")
                
                # Initialize models
                training_status["status_message"] = "Initializing models..."
                training_status["progress"] = 40
                
                logger.info(f"Initializing models with input shape: {input_shape}")
                trainer.initialize_models(input_shape)
                logger.info("Models initialized successfully")
                
                # Get the model class
                if model_name in trainer.models:
                    model = trainer.models[model_name]
                    logger.info(f"Selected model: {model_name}")
                elif model_name == "model2" or model_name.lower() == "bozdogancaic":
                    # Handle Model 2 (BozdoganCAIC) specifically since it's a common case
                    model_name = "model2"  # Ensure consistent naming
                    model = trainer.models.get(model_name)
                    if model is None and "model2" in trainer.models:
                        model = trainer.models["model2"]
                    logger.info(f"Selected model: {model_name} (BozdoganCAIC)")
                else:
                    return jsonify({"success": False, "message": f"Model {model_name} not found"}), 404
                
                # Configure model based on type
                if model_name == "model2" or isinstance(model, BozdoganCAICModel):
                    # Model 2: More complex architecture needs more epochs and larger batch
                    model.batch_size = 128  # Larger batch size for more stable gradients
                    if hasattr(model, 'epochs'):
                        model.epochs = 15  # More epochs for the complex model
                    chunk_size = 3500  # Slightly larger chunk for more context
                    chunk_epochs = 15  # More epochs per chunk
                    logger.info(f"Using optimized config for Model 2 (BozdoganCAIC): batch_size={model.batch_size}, epochs={model.epochs if hasattr(model, 'epochs') else 15}, chunk_size={chunk_size}")
                elif model_name == "model1" or isinstance(model, BayesianNeuralNetwork):
                    # Model 1: Bayesian models benefit from smaller batches but more epochs
                    model.batch_size = 64   # Moderate batch size for bayesian models
                    if hasattr(model, 'epochs'):
                        model.epochs = 12  # Slightly more epochs for bayesian convergence
                    chunk_size = 3000  # Standard chunk size
                    chunk_epochs = 12  # More epochs for proper posterior distribution
                    logger.info(f"Using optimized config for Model 1 (Bayesian): batch_size={model.batch_size}, epochs={model.epochs if hasattr(model, 'epochs') else 12}, chunk_size={chunk_size}")
                else:
                    # Default configuration for other models
                    model.batch_size = 96  # Standard batch size
                    if hasattr(model, 'epochs'):
                        model.epochs = 10  # Standard number of epochs
                    chunk_size = 3000  # Standard chunk size
                    chunk_epochs = 10  # Standard epochs per chunk
                    logger.info(f"Using standard config for {model_name}: batch_size={model.batch_size}, epochs={model.epochs if hasattr(model, 'epochs') else 10}, chunk_size={chunk_size}")
                
                logger.info(f"Model configured with batch_size={model.batch_size}")
                
                # Split initial data for validation
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(X_init, y_init, test_size=0.2, random_state=42)
                
                # Start incremental training
                training_status["status_message"] = "Starting incremental training..."
                training_status["progress"] = 50
                
                # First, build the model with our initial data
                logger.info("Building and compiling model")
                model.build_model()
                
                # Setup TensorFlow for memory efficiency
                import tensorflow as tf
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    logger.info(f"Found {len(gpus)} GPUs")
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                else:
                    logger.info("No GPUs found, using CPU")
                
                # Enable mixed precision for memory efficiency
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                
                # Prepare to train on chunks
                total_rows = len(processed_data)
                # Make sure we create enough chunks to cover the entire dataset, limited to 10 max
                num_chunks = min(10, max(3, total_rows // (chunk_size - overlap) + 1))  
                logger.info(f"Preparing to train on {num_chunks} chunks with dataset size: {total_rows} rows")
                logger.info(f"Chunk size: {chunk_size}, Overlap: {overlap}, Effective chunk size: {chunk_size - overlap}")
                
                for chunk_idx in range(num_chunks):
                    # Update status
                    training_status["status_message"] = f"Training on data chunk {chunk_idx+1}/{num_chunks}..."
                    training_status["progress"] = 50 + (chunk_idx * 50 // num_chunks)
                    
                    # Calculate start and end indices with overlap
                    if chunk_idx == 0:
                        start_idx = 0
                    else:
                        # For chunks after the first one, we need to start at the right point considering overlap
                        start_idx = chunk_idx * (chunk_size - overlap)
                    
                    end_idx = min(len(processed_data), start_idx + chunk_size)
                    
                    logger.info(f"Processing chunk {chunk_idx+1}/{num_chunks}: rows {start_idx} to {end_idx}")
                    
                    # Skip if we've reached beyond the end of the dataset
                    if start_idx >= total_rows:
                        logger.info(f"Chunk {chunk_idx+1} start index {start_idx} is beyond dataset size {total_rows}, stopping")
                        break
                    
                    # Check if there's enough data left for meaningful sequences
                    if end_idx - start_idx <= sequence_length + prediction_horizon:
                        logger.info(f"Remaining data too small for chunk {chunk_idx+1}: only {end_idx - start_idx} rows available")
                        break
                    
                    # Get data for this chunk
                    chunk_data = processed_data.iloc[start_idx:end_idx]
                    logger.info(f"Chunk {chunk_idx+1} size: {len(chunk_data)} rows")
                    
                    # Skip first chunk since we already used it
                    if chunk_idx == 0:
                        logger.info("Skipping first chunk as it was used for initialization")
                        continue
                    
                    # Create sequences for this chunk
                    X_chunk, y_chunk = create_sequences(chunk_data[features])
                    
                    if len(X_chunk) == 0:
                        logger.warning(f"Chunk {chunk_idx+1} produced no valid sequences, skipping")
                        continue
                    
                    logger.info(f"Created {len(X_chunk)} sequences from chunk {chunk_idx+1}")
                    
                    # Handle NaN values
                    if np.isnan(X_chunk).any():
                        logger.warning(f"Chunk {chunk_idx+1} X contains NaN values - replacing with 0")
                        X_chunk = np.nan_to_num(X_chunk, nan=0.0)
                        
                    if np.isnan(y_chunk).any():
                        logger.warning(f"Chunk {chunk_idx+1} y contains NaN values - replacing with 0")
                        y_chunk = np.nan_to_num(y_chunk, nan=0.0)
                    
                    # Split for validation
                    X_train_chunk, X_val_chunk, y_train_chunk, y_val_chunk = train_test_split(
                        X_chunk, y_chunk, test_size=0.2, random_state=42+chunk_idx
                    )
                    
                    logger.info(f"Chunk {chunk_idx+1} data shapes - X_train: {X_train_chunk.shape}, y_train: {y_train_chunk.shape}")
                    
                    # Train the model on this chunk
                    logger.info(f"Training on chunk {chunk_idx+1}")
                    
                    # Determine epochs for this specific chunk training
                    logger.info(f"Using {chunk_epochs} epochs for training on chunk {chunk_idx+1}")

                    # Create TensorFlow callback to update training status
                    class TrainingStatusCallback(tf.keras.callbacks.Callback):
                        def on_epoch_begin(self, epoch, logs=None):
                            training_status["status_message"] = f"Training on chunk {chunk_idx+1}/{num_chunks} (Epoch {epoch+1}/{chunk_epochs})"
                            training_status["progress"] = 50 + ((chunk_idx * chunk_epochs + epoch) * 50 // (num_chunks * chunk_epochs))
                        
                        def on_batch_end(self, batch, logs=None):
                            if batch % 10 == 0:  # Update every 10 batches to avoid too many updates
                                batch_size = logs.get('size', 32)
                                total_batches = len(X_train_chunk) // batch_size
                                if total_batches > 0:
                                    batch_progress = min(100, (batch * 100) // total_batches)
                                    # Use hasattr and getattr to safely access the epoch number
                                    if hasattr(self, 'params') and 'epoch' in self.params:
                                        epoch_num = self.params['epoch'] + 1
                                    else:
                                        epoch_num = 1  # Default to 1 if epoch isn't available
                                    training_status["status_message"] = f"Training chunk {chunk_idx+1}/{num_chunks} - Epoch {epoch_num}/{chunk_epochs} - Batch {batch}/{total_batches}"

                    history = model.model.fit(
                        X_train_chunk, y_train_chunk,
                        validation_data=(X_val_chunk, y_val_chunk),
                        epochs=chunk_epochs,
                        batch_size=model.batch_size,
                        verbose=1,
                        callbacks=[TrainingStatusCallback()]
                    )
                    
                    # Check metrics
                    val_loss = history.history.get('val_loss', [0])[-1]
                    val_mae = history.history.get('val_mae', [0])[-1]
                    logger.info(f"Chunk {chunk_idx+1} validation - Loss: {val_loss}, MAE: {val_mae}")
                    
                    # Clean up to free memory
                    import gc
                    del X_chunk, y_chunk, X_train_chunk, X_val_chunk, y_train_chunk, y_val_chunk
                    gc.collect()
                
                # Final evaluation on validation data
                training_status["status_message"] = "Final model evaluation..."
                training_status["progress"] = 95
                
                eval_results = model.model.evaluate(X_val, y_val, verbose=1)
                val_loss = eval_results[0]
                val_mae = eval_results[1]
                
                # Convert MAE to accuracy (1-MAE is a simple metric)
                accuracy = 1.0 - val_mae
                logger.info(f"Final evaluation - Loss: {val_loss}, MAE: {val_mae}, Accuracy: {accuracy:.4f}")
                
                # Save the model and scalers
                training_status["status_message"] = "Saving model..."
                training_status["progress"] = 98
                
                # Save the scalers for prediction
                try:
                    success = data_loader.save_scalers(model.name)
                    if not success:
                        logger.warning(f"Failed to save scalers for {model.name}. Predictions may not work correctly.")
                except Exception as e:
                    logger.warning(f"Error saving scalers: {str(e)}. Predictions may not work correctly.")
                
                # Save the model
                model_path = model.save()
                
                # Update model metadata to include accuracy
                import joblib
                metadata_path = f"{model_path}_metadata.joblib"
                if os.path.exists(metadata_path):
                    metadata = joblib.load(metadata_path)
                    metadata["accuracy"] = accuracy
                    metadata["val_loss"] = float(val_loss)
                    metadata["val_mae"] = float(val_mae)
                    metadata["retrain_needed"] = accuracy < 0.90  # Flag if model needs retraining
                    metadata["last_trained_month"] = processed_data.index[0].strftime("%Y-%m") if hasattr(processed_data.index[0], 'strftime') else "unknown"
                    joblib.dump(metadata, metadata_path)
                
                # Check if model reached 90% accuracy threshold
                if accuracy >= 0.90:
                    training_status["in_progress"] = False
                    training_status["progress"] = 100
                    training_status["status_message"] = f"Training completed successfully - Accuracy: {accuracy:.4f} (reached 90% threshold)"
                    
                    return jsonify({
                        "success": True,
                        "message": f"Training completed successfully - Accuracy: {accuracy:.4f}",
                        "accuracy": float(accuracy),
                        "threshold_reached": True,
                        "model_name": model.name
                    })
                else:
                    # Model did not reach threshold - set status to need retraining but mark as complete
                    training_status["in_progress"] = False
                    training_status["progress"] = 100
                    training_status["status_message"] = f"Training completed - Accuracy: {accuracy:.4f} (below 90% threshold)"
                    training_status["needs_retraining"] = True
                    training_status["model_name"] = model.name
                    
                    return jsonify({
                        "success": True,
                        "message": f"Training completed but accuracy {accuracy:.4f} is below 90% threshold",
                        "accuracy": float(accuracy),
                        "threshold_reached": False,
                        "needs_retraining": True,
                        "model_name": model.name
                    })
                
            except Exception as inner_error:
                # Add more detailed debugging for this specific step
                logger.error(f"Error during data preparation: {str(inner_error)}")
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"Detailed traceback: {error_trace}")
                
                # Do not fall back to synthetic data - only use real SPY data
                training_status["in_progress"] = False
                training_status["progress"] = 0
                training_status["status_message"] = f"Data preparation failed: {str(inner_error)}"
                
                # Return error message
                return jsonify({
                    "success": False,
                    "message": f"Failed to prepare SPY price data for training: {str(inner_error)}. No models will be trained with synthetic data.",
                    "details": "Training requires actual SPY price data to ensure accurate predictions."
                }), 500
                
        except Exception as data_prep_error:
            logger.error(f"Error preparing training data: {str(data_prep_error)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            training_status["in_progress"] = False
            training_status["status_message"] = f"Error preparing training data: {str(data_prep_error)}"
            return jsonify({"success": False, "message": f"Error preparing training data: {str(data_prep_error)}"}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in training: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        training_status["in_progress"] = False
        training_status["status_message"] = f"Unexpected error: {str(e)}"
        return jsonify({"success": False, "message": f"Unexpected error: {str(e)}"}), 500

@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    """API endpoint to get current training status."""
    # Override the URL to make sure UI can access it for correct binding
    if 'needs_retraining' in training_status and training_status['needs_retraining']:
        # Add model name for retraining confirmation
        if 'model_name' not in training_status and 'model' in training_status:
            training_status['model_name'] = training_status['model']
    
    return jsonify(training_status)

@app.route('/api/training/simplified', methods=['POST'])
def simplified_training():
    """API endpoint for simplified training testing."""
    global trainer, training_status
    
    try:
        # Check if training is already in progress
        if training_status["in_progress"]:
            return jsonify({
                "success": False, 
                "message": f"Training already in progress for {training_status['model_name']}"
            }), 400
        
        # Update training status
        training_status = {
            "in_progress": True,
            "model_name": "simplified_test",
            "progress": 0,
            "status_message": "Initializing simplified training...",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Clean up memory before starting
        import gc
        import tensorflow as tf
        gc.collect()
        tf.keras.backend.clear_session()
        
        # Initialize trainer if not already done
        if trainer is None:
            logger.info("Initializing ModelTrainer")
            trainer = ModelTrainer()
        
        # Create synthetic data for testing
        training_status["status_message"] = "Creating synthetic data..."
        training_status["progress"] = 20
        
        # Create test data with consistent shape
        sequence_length = 60
        num_features = 20  # Reasonable number of features
        
        X = np.random.random((100, sequence_length, num_features)).astype(np.float32)
        y = np.random.random((100, 10)).astype(np.float32)  # Using 10 for prediction horizon
        
        logger.info(f"Created test data X: {X.shape}, y: {y.shape}")
        input_shape = X.shape[1:]
        
        # Initialize models
        training_status["status_message"] = "Initializing models..."
        training_status["progress"] = 40
        
        try:
            logger.info(f"Initializing models with input shape: {input_shape}")
            trainer.initialize_models(input_shape)
            logger.info("Models initialized successfully")
            
        except Exception as model_error:
            logger.error(f"Error initializing models: {str(model_error)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            training_status["in_progress"] = False
            training_status["status_message"] = f"Error initializing models: {str(model_error)}"
            return jsonify({"success": False, "message": f"Error initializing models: {str(model_error)}"}), 500
        
        # Training with synthetic data
        training_status["status_message"] = "Training model with synthetic data..."
        training_status["progress"] = 60
        
        try:
            import tensorflow as tf
            logger.info("Setting up TensorFlow")
            
            # Configure TensorFlow
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                logger.info(f"Found {len(gpus)} GPUs")
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                logger.info("No GPUs found, using CPU")
                
            # Enable mixed precision for memory efficiency
            logger.info("Enabling mixed precision")
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            # Split data for validation
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            logger.info("Training data prepared")
            logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
            logger.info(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
            
            # Train model directly
            model = trainer.models.get("model1")
            
            if model is None:
                raise ValueError("Model 'model1' not found in trainer")
                
            logger.info("Building model")
            model.build_model()
            
            # Set a very small batch size
            batch_size = 4
            logger.info(f"Setting batch size to {batch_size}")
            model.batch_size = batch_size
            
            # Try fitting model directly
            logger.info("Starting model training")
            history = model.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=2,  # Just 2 epochs for testing
                batch_size=batch_size,
                verbose=1
            )
            
            logger.info("Model training completed successfully")
            
            # Check metrics
            val_loss = history.history.get('val_loss', [0])[-1]
            logger.info(f"Validation loss: {val_loss}")
            
            # Save model for testing
            model.save()
            
            training_status["in_progress"] = False
            training_status["progress"] = 100
            training_status["status_message"] = "Simplified training completed successfully"
            
            return jsonify({
                "success": True,
                "message": "Simplified training completed successfully",
                "metrics": {
                    "val_loss": float(val_loss)
                }
            })
            
        except Exception as training_error:
            logger.error(f"Error during simplified training: {str(training_error)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            training_status["in_progress"] = False
            training_status["status_message"] = f"Error during simplified training: {str(training_error)}"
            return jsonify({"success": False, "message": f"Error during simplified training: {str(training_error)}"}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in simplified training: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        training_status["in_progress"] = False
        training_status["status_message"] = f"Unexpected error: {str(e)}"
        return jsonify({"success": False, "message": f"Unexpected error: {str(e)}"}), 500

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    """API endpoint to make predictions."""
    try:
        global predictor
        
        # Make sure predictor is initialized
        if predictor is None:
            predictor = StockPredictor()
            
        # Get request parameters
        data = request.get_json() or {}
        model_name = data.get("model_name")  # If None, will use all models
        
        # Load models if not already loaded
        if predictor.models is None or len(predictor.models) == 0:
            predictor.load_models()
            if predictor.models is None or len(predictor.models) == 0:
                return jsonify({"success": False, "message": "No trained models found"}), 404
        
        # Make predictions
        predictions = predictor.predict_next_candles(model_name)
        if predictions is None:
            return jsonify({"success": False, "message": "Failed to make predictions"}), 500
        
        # Get the latest prediction plot file
        import glob
        plot_files = sorted(glob.glob(os.path.join(PATHS["results_dir"], "prediction_plot_*.png")))
        latest_plot = plot_files[-1] if plot_files else None
        
        # Format response
        response = {
            "success": True,
            "predictions": predictions,
            "plot_url": f"/plots/{os.path.basename(latest_plot)}" if latest_plot else None
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """API endpoint to get information about available models."""
    try:
        global predictor
        
        # Make sure predictor is initialized
        if predictor is None:
            predictor = StockPredictor()
        
        # Load models if not already loaded
        if predictor.models is None or len(predictor.models) == 0:
            predictor.load_models()
        
        # Get model information
        models_info = {}
        for name, model in predictor.models.items():
            # Skip None models
            if model is None:
                continue
                
            # Handle case where model_accuracies might be None
            accuracy = "Unknown"
            if hasattr(predictor, 'model_accuracies') and predictor.model_accuracies is not None:
                accuracy = predictor.model_accuracies.get(name, "Unknown")
                
            models_info[name] = {
                "name": name,
                "type": model.__class__.__name__,
                "accuracy": accuracy
            }
        
        return jsonify({
            "success": True,
            "models": models_info
        })
    
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

@app.route('/api/plots', methods=['GET'])
def get_plots():
    """API endpoint to get available plots."""
    try:
        # Get all prediction plots
        import glob
        plot_files = sorted(glob.glob(os.path.join(PATHS["results_dir"], "prediction_plot_*.png")))
        
        # Format response
        plots = []
        for plot_file in plot_files:
            timestamp = os.path.basename(plot_file).replace("prediction_plot_", "").replace(".png", "")
            plots.append({
                "url": f"/plots/{os.path.basename(plot_file)}",
                "timestamp": timestamp
            })
        
        return jsonify({
            "success": True,
            "plots": plots
        })
    
    except Exception as e:
        logger.error(f"Error getting plots: {str(e)}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

@app.route('/plots/<filename>')
def serve_plot(filename):
    """Serve plot images."""
    return send_from_directory(PATHS["results_dir"], filename)

@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler for the Flask app."""
    logger.error(f"Unhandled exception: {str(e)}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    return jsonify({
        "success": False,
        "message": f"Server error: {str(e)}"
    }), 500

@app.route('/debug/memory')
def debug_memory():
    """API endpoint to check memory usage."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage = {
            "rss": f"{memory_info.rss / (1024 * 1024):.2f} MB",
            "vms": f"{memory_info.vms / (1024 * 1024):.2f} MB",
            "percent": f"{process.memory_percent():.2f}%",
            "system_total": f"{psutil.virtual_memory().total / (1024 * 1024):.2f} MB",
            "system_available": f"{psutil.virtual_memory().available / (1024 * 1024):.2f} MB",
            "system_percent": f"{psutil.virtual_memory().percent:.2f}%"
        }
        return jsonify({"success": True, "memory_usage": memory_usage})
    except Exception as e:
        logger.error(f"Error getting memory info: {str(e)}")
        return jsonify({"success": False, "message": f"Error getting memory info: {str(e)}"}), 500

@app.route('/debug/check-tensorflow')
def debug_tensorflow():
    """API endpoint to check TensorFlow configuration."""
    try:
        import tensorflow as tf
        tf_info = {
            "version": tf.__version__,
            "keras_version": tf.keras.__version__,
            "devices": [device.name for device in tf.config.list_physical_devices()],
            "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
            "mixed_precision": tf.keras.mixed_precision.global_policy().name
        }
        return jsonify({"success": True, "tensorflow_info": tf_info})
    except Exception as e:
        logger.error(f"Error checking TensorFlow: {str(e)}")
        return jsonify({"success": False, "message": f"Error checking TensorFlow: {str(e)}"}), 500

# Add debugging endpoints to web interface
@app.route('/debug')
def debug_page():
    """Render the debug page."""
    return render_template('debug.html')

@app.route('/api/data/stats')
def get_data_stats():
    """API endpoint to get statistics about the available data."""
    try:
        import glob
        import pandas as pd
        from collections import defaultdict
        
        data_dir = PATHS["data_dir"]
        stats = {
            "available_months": [],
            "latest_data": {},
            "file_count": 0,
            "total_size_mb": 0
        }
        
        # Get all CSV files
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        stats["file_count"] = len(csv_files)
        
        # Calculate total size
        total_size = sum(os.path.getsize(f) for f in csv_files)
        stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)
        
        # Get monthly data files
        monthly_data = defaultdict(dict)
        for file in csv_files:
            filename = os.path.basename(file)
            if filename.startswith("SPY_training_month_"):
                month_key = filename.replace("SPY_training_month_", "").replace(".csv", "")
                
                # Get row count and file size
                try:
                    df = pd.read_csv(file, nrows=5)  # Just read a few rows to get columns
                    row_count = sum(1 for _ in open(file)) - 1  # Subtract header
                    file_size_mb = os.path.getsize(file) / (1024 * 1024)
                    
                    monthly_data[month_key] = {
                        "rows": row_count,
                        "size_mb": round(file_size_mb, 2),
                        "columns": len(df.columns),
                        "file": filename
                    }
                    stats["available_months"].append(month_key)
                except Exception as e:
                    logger.error(f"Error processing file {filename}: {str(e)}")
        
        stats["monthly_data"] = dict(monthly_data)
        
        # Get latest data info
        latest_file = os.path.join(data_dir, "SPY_latest_7d_1m.csv")
        if os.path.exists(latest_file):
            try:
                df = pd.read_csv(latest_file)
                stats["latest_data"] = {
                    "rows": len(df),
                    "size_mb": round(os.path.getsize(latest_file) / (1024 * 1024), 2),
                    "date_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}" if 'timestamp' in df.columns else "Unknown"
                }
            except Exception as e:
                logger.error(f"Error reading latest data file: {str(e)}")
                stats["latest_data"] = {"error": str(e)}
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting data stats: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug/cleanup', methods=['POST'])
def cleanup_memory():
    """API endpoint to force memory cleanup."""
    try:
        import gc
        import tensorflow as tf
        
        # Count objects before cleanup
        objects_before = len(gc.get_objects())
        
        # Clear TensorFlow session
        tf.keras.backend.clear_session()
        
        # Run garbage collection multiple times
        gc.collect(0)  # Generation 0
        gc.collect(1)  # Generation 1
        gc.collect(2)  # Generation 2
        
        # Count objects after cleanup
        objects_after = len(gc.get_objects())
        objects_freed = max(0, objects_before - objects_after)
        
        return jsonify({
            "success": True,
            "message": f"Memory cleanup complete. Freed approximately {objects_freed} objects.",
            "objects_before": objects_before,
            "objects_after": objects_after
        })
    except Exception as e:
        logger.error(f"Error during memory cleanup: {str(e)}")
        return jsonify({"success": False, "message": f"Error during cleanup: {str(e)}"}), 500

def main():
    """Run the web application."""
    app.run(host=WEB_CONFIG["host"], 
            port=WEB_CONFIG["port"], 
            debug=WEB_CONFIG["debug"],
            use_reloader=False)  # Disable auto-reloading to prevent interrupting TensorFlow

if __name__ == "__main__":
    main() 
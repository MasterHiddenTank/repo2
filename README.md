# SPY Stock Price Prediction System

A sophisticated neural network-based system for predicting the next five 5-minute SPY stock price candles with high accuracy.

## Overview

This application predicts the next five 5-minute SPY stock price candles using multiple neural network models. Each model is trained on randomly selected historical data and achieves 90% accuracy or higher independently.

### Key Features

- **Real-time SPY data collection** using yfinance
- **10 distinct neural network models** with different architectures:
  - Model 1: Bayesian Neural Network with MC Dropout
  - Model 2: Bozdogan Consistent AIC
  - Model 3: Viterbi Algorithm + Baum-Welch
  - Model 4: Gaussian Hidden Markov Model
  - Model 5: Fuzzy Logic + Stochastic Processes
  - Model 6: Decision Trees with Random Forks
  - Model 7: Kronecker-Factored Laplace Approximation
  - Model 8: Stochastic Gradient-Driven Dynamics
  - Model 9: MC Dropout
  - Model 10: Combined approach
- **Training on random historical data** for model robustness
- **Web interface** for easy interaction with the system
- **Visualizations** of predictions with uncertainty estimates

## Prerequisites

- Python 3.8
- pip (Python package manager)
- Internet connection (for downloading SPY data)

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Starting the Web Application

Run the application with:

```
python run.py
```

This will start the web server, typically at http://localhost:5000.

### Using the Web Interface

1. **Download Latest Data**: Click the "Download Latest Data" button on the dashboard to fetch the most recent SPY 5-minute candle data.

2. **Train Models**: 
   - To train all models, click "Train All Models"
   - To train a specific model, select it from the dropdown and click "Train Selected Model"
   - Models train on randomly selected monthly data until they reach 90% accuracy

3. **Make Predictions**:
   - Click "Make Predictions" to generate forecasts using all trained models
   - To use a specific model, select it from the dropdown before clicking "Make Predictions"
   - View the prediction results in the table and chart below

## Project Structure

```
.
├── config.py                  # Configuration settings
├── data/                      # Data handling modules
│   ├── __init__.py
│   ├── data_loader.py         # Data download and processing
│   └── storage/               # Directory for storing data files
├── models/                    # Neural network models
│   ├── __init__.py
│   ├── base_model.py          # Base model class
│   ├── model1_bayesian_neural_network.py
│   ├── model2_bozdogan_consistent_aic.py
│   └── ...                    # Other model implementations
├── training/                  # Training modules
│   ├── __init__.py
│   └── trainer.py             # Model training logic
├── prediction/                # Prediction modules
│   ├── __init__.py
│   └── predictor.py           # Prediction generation
├── evaluation/                # Evaluation modules
│   ├── __init__.py
│   └── evaluator.py           # Model evaluation
├── web/                       # Web interface
│   ├── __init__.py
│   ├── app.py                 # Flask application
│   ├── static/                # Static assets (CSS, JS)
│   └── templates/             # HTML templates
├── requirements.txt           # Python dependencies
└── run.py                     # Main entry point
```

## Development

### Adding New Models

To add a new model:

1. Create a new file in the `models/` directory
2. Implement the model by extending the `BaseModel` class
3. Override the `build_model()` method to define your model architecture
4. Add the model to the `ModelTrainer.initialize_models()` method in `training/trainer.py`

### Customizing Training

You can customize the training process by modifying the parameters in `config.py`:

- `accuracy_threshold`: Target accuracy for models
- `max_epochs`: Maximum training epochs
- `batch_size`: Batch size for training
- `learning_rate`: Initial learning rate

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Yahoo Finance API for providing SPY stock data
- TensorFlow and Keras for neural network implementation
- Flask for the web interface

---

Created as part of The TurnAround Project | 2023

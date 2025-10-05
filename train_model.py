"""
VaanamWeather - Model Training Script
Train ML models using historical NASA/INPE data

Usage:
    python train_model.py --variable temperature --model lstm
    python train_model.py --variable precipitation --model rf --retrain
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models import (
    LSTMWeatherPredictor,
    RandomForestPredictor,
    WeatherPreprocessor,
    TrendAnalyzer,
    AnomalyDetector
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetch training data from NASA APIs or generate synthetic data"""
    
    @staticmethod
    def generate_synthetic_data(
        variable: str,
        n_samples: int = 1000,
        add_seasonality: bool = True,
        add_trend: bool = True,
        add_noise: bool = True
    ) -> pd.DataFrame:
        """
        Generate synthetic weather data for training
        
        Args:
            variable: Weather variable name
            n_samples: Number of samples to generate
            add_seasonality: Add seasonal patterns
            add_trend: Add long-term trend
            add_noise: Add random noise
        
        Returns:
            DataFrame with dates and values
        """
        logger.info(f"Generating {n_samples} synthetic data points for {variable}")
        
        # Generate dates
        start_date = datetime.now() - timedelta(days=n_samples)
        dates = [start_date + timedelta(days=i) for i in range(n_samples)]
        
        # Base values by variable type
        base_values = {
            "temperature": 25,
            "precipitation": 5,
            "windspeed": 10,
            "humidity": 60,
            "pressure": 101325
        }
        
        base = base_values.get(variable, 25)
        values = np.ones(n_samples) * base
        
        # Add seasonality (annual cycle)
        if add_seasonality:
            days_of_year = np.array([d.timetuple().tm_yday for d in dates])
            seasonal = 10 * np.sin(2 * np.pi * days_of_year / 365.25)
            values += seasonal
        
        # Add long-term trend
        if add_trend:
            trend = np.linspace(0, 5, n_samples)  # Gradual increase
            values += trend
        
        # Add random noise
        if add_noise:
            noise = np.random.normal(0, 2, n_samples)
            values += noise
        
        # Add occasional extreme events (5% of data)
        extreme_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        values[extreme_indices] += np.random.normal(15, 5, len(extreme_indices))
        
        # Ensure positive values for certain variables
        if variable in ["precipitation", "windspeed", "humidity"]:
            values = np.abs(values)
        
        df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        return df
    
    @staticmethod
    def fetch_real_data(
        latitude: float,
        longitude: float,
        variable: str,
        days: int = 365
    ) -> pd.DataFrame:
        """
        Fetch real data from NASA OPeNDAP
        (Simplified version - in production, use actual API calls)
        """
        logger.info(f"Fetching real data for {variable} at ({latitude}, {longitude})")
        
        # For now, use synthetic data
        # In production, this would call the actual NASA API
        return DataFetcher.generate_synthetic_data(variable, n_samples=days)


class ModelTrainer:
    """Train and save ML models"""
    
    def __init__(self, variable: str, model_type: str):
        self.variable = variable
        self.model_type = model_type
        self.preprocessor = WeatherPreprocessor()
    
    def train_lstm(self, data: pd.DataFrame, epochs: int = 50):
        """Train LSTM model"""
        logger.info(f"Training LSTM model for {self.variable}")
        
        # Prepare data
        values = data['value'].values
        
        # Handle missing values
        data_clean = self.preprocessor.handle_missing_values(
            pd.DataFrame({'value': values})
        )
        values_clean = data_clean['value'].values
        
        # Initialize and train model
        predictor = LSTMWeatherPredictor(variable=self.variable)
        history = predictor.train(values_clean, epochs=epochs, batch_size=32)
        
        logger.info("LSTM model training complete")
        
        # Evaluate
        final_loss = history.history['loss'][-1]
        final_mae = history.history['mae'][-1]
        
        return {
            "model_type": "LSTM",
            "variable": self.variable,
            "final_loss": float(final_loss),
            "final_mae": float(final_mae),
            "epochs": epochs,
            "samples": len(values_clean)
        }
    
    def train_random_forest(self, data: pd.DataFrame):
        """Train Random Forest model"""
        logger.info(f"Training Random Forest model for {self.variable}")
        
        # Create features
        data = self.preprocessor.create_features(data)
        
        # Prepare features and target
        feature_cols = ['day_of_year', 'month', 'day', 'year', 
                        'day_sin', 'day_cos', 'month_sin', 'month_cos']
        
        X = data[feature_cols]
        y = data['value'].values
        
        # Handle missing values
        X = self.preprocessor.handle_missing_values(X)
        
        # Train model
        predictor = RandomForestPredictor(variable=self.variable)
        predictor.train(X, y)
        
        logger.info("Random Forest model training complete")
        
        # Calculate training score
        train_score = predictor.model.score(X, y)
        
        return {
            "model_type": "RandomForest",
            "variable": self.variable,
            "train_score": float(train_score),
            "n_estimators": 100,
            "samples": len(y)
        }
    
    def train_model(self, data: pd.DataFrame, **kwargs):
        """Train specified model type"""
        if self.model_type == "lstm":
            return self.train_lstm(data, **kwargs)
        elif self.model_type == "rf" or self.model_type == "randomforest":
            return self.train_random_forest(data)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


def train_all_models(variables: list = None, use_real_data: bool = False):
    """
    Train all models for specified variables
    
    Args:
        variables: List of variables to train models for
        use_real_data: Whether to use real NASA data or synthetic
    """
    if variables is None:
        variables = ["temperature", "precipitation", "windspeed", "humidity", "pressure"]
    
    results = {}
    
    for variable in variables:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training models for: {variable}")
        logger.info(f"{'='*60}\n")
        
        # Fetch data
        if use_real_data:
            data = DataFetcher.fetch_real_data(
                latitude=13.0827,
                longitude=80.2707,
                variable=variable,
                days=365*3  # 3 years of data
            )
        else:
            data = DataFetcher.generate_synthetic_data(
                variable=variable,
                n_samples=365*3
            )
        
        # Train LSTM
        try:
            lstm_trainer = ModelTrainer(variable, "lstm")
            lstm_result = lstm_trainer.train_model(data, epochs=30)
            results[f"lstm_{variable}"] = lstm_result
            logger.info(f"✓ LSTM model trained: Loss={lstm_result['final_loss']:.4f}")
        except Exception as e:
            logger.error(f"✗ LSTM training failed: {str(e)}")
            results[f"lstm_{variable}"] = {"error": str(e)}
        
        # Train Random Forest
        try:
            rf_trainer = ModelTrainer(variable, "rf")
            rf_result = rf_trainer.train_model(data)
            results[f"rf_{variable}"] = rf_result
            logger.info(f"✓ Random Forest model trained: Score={rf_result['train_score']:.4f}")
        except Exception as e:
            logger.error(f"✗ Random Forest training failed: {str(e)}")
            results[f"rf_{variable}"] = {"error": str(e)}
    
    return results


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description="Train VaanamWeather ML models")
    parser.add_argument(
        "--variable",
        type=str,
        default="all",
        choices=["temperature", "precipitation", "windspeed", "humidity", "pressure", "all"],
        help="Variable to train models for"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["lstm", "rf", "all"],
        help="Model type to train"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs for LSTM training"
    )
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Use real NASA data instead of synthetic"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain existing models"
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("VaanamWeather Model Training")
    logger.info("="*60)
    logger.info(f"Variable: {args.variable}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Real Data: {args.real_data}")
    logger.info("="*60 + "\n")
    
    # Determine variables to train
    if args.variable == "all":
        variables = ["temperature", "precipitation", "windspeed", "humidity", "pressure"]
    else:
        variables = [args.variable]
    
    # Train models
    results = train_all_models(variables, use_real_data=args.real_data)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Training Summary")
    logger.info("="*60)
    for model_name, result in results.items():
        if "error" in result:
            logger.info(f"✗ {model_name}: FAILED - {result['error']}")
        else:
            logger.info(f"✓ {model_name}: SUCCESS")
    
    logger.info("="*60)
    logger.info("All models trained and saved!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

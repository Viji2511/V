"""
VaanamWeather - Management CLI
Automates data fetching, preprocessing, model training, and deployment

Usage:
    python manage.py sync              # Full pipeline: fetch → clean → train
    python manage.py fetch-data        # Fetch data only
    python manage.py train             # Train models only
    python manage.py deploy            # Deploy/restart API
    python manage.py status            # Check system status
"""

import argparse
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import json
import time

# Import pipeline modules
from data_pipeline import DataPipeline
from train_model import ModelTrainer, DataFetcher

# Configure logging
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"manage_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VaanamManager:
    """
    Central manager for VaanamWeather automation
    """
    
    def __init__(self):
        self.pipeline = DataPipeline()
        self.config = self.load_config()
    
    def load_config(self) -> dict:
        """Load configuration from file or use defaults"""
        config_path = Path("config.json")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        default_config = {
            "default_locations": [
                {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
                {"name": "Delhi", "lat": 28.7041, "lon": 77.1025},
                {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777}
            ],
            "data_sources": ["nasa_power", "hydrology", "inpe"],
            "training": {
                "variables": ["temperature", "precipitation", "windspeed", "humidity"],
                "lstm_epochs": 50,
                "retrain_interval_days": 30
            },
            "data_fetch": {
                "days_history": 1095,  # 3 years
                "auto_update": True
            }
        }
        
        # Save default config
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"Created default config at {config_path}")
        return default_config
    
    def fetch_data(self, location: dict = None, days: int = None):
        """
        Fetch data for specified location
        
        Args:
            location: Dict with 'lat', 'lon', and optionally 'name'
            days: Number of days of history to fetch
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 1: FETCHING DATA")
        logger.info("="*60)
        
        if location is None:
            locations = self.config["default_locations"]
        else:
            locations = [location]
        
        if days is None:
            days = self.config["data_fetch"]["days_history"]
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for loc in locations:
            logger.info(f"\nFetching data for {loc.get('name', 'Location')}")
            logger.info(f"  Coordinates: ({loc['lat']}, {loc['lon']})")
            
            try:
                df = self.pipeline.run_pipeline(
                    latitude=loc['lat'],
                    longitude=loc['lon'],
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    output_filename=f"data_{loc.get('name', 'location').lower()}.csv"
                )
                
                logger.info(f"✓ Successfully fetched {len(df)} records")
                
            except Exception as e:
                logger.error(f"✗ Error fetching data: {str(e)}")
        
        logger.info("\n✓ Data fetching complete")
    
    def train_models(self, variables: list = None, force_retrain: bool = False):
        """
        Train ML models
        
        Args:
            variables: List of variables to train
            force_retrain:

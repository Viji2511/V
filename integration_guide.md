# VaanamWeather - Complete Integration Guide 🔗

This guide explains how all components work together to create a fully automated, ML-powered weather API.

---

## 📁 Complete File Structure

```
vaanamweather/
├── main.py                          # Main FastAPI app (ML-integrated)
├── ml_models.py                     # ML models (LSTM, RF, Anomaly Detection)
├── data_pipeline.py                 # Automated data fetching
├── train_model.py                   # Model training script
├── manage.py                        # CLI automation manager
├── requirements.txt                 # All dependencies
├── Dockerfile                       # Container definition
├── docker-compose.yml               # Multi-container setup
├── .env                            # Environment variables
├── config.json                      # Configuration file
│
├── models/                          # Trained ML models
│   ├── lstm_temperature.h5
│   ├── lstm_precipitation.h5
│   ├── rf_temperature.pkl
│   └── rf_precipitation.pkl
│
├── data/                           # Data storage
│   ├── raw/                        # Raw downloaded data
│   ├── cleaned/                    # Processed data
│   │   └── merged_data.csv
│   └── metadata/                   # Data metadata
│
└── logs/                           # Application logs
    └── manage_20250105.log
```

---

## 🔄 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VaanamWeather System                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │      1. DATA FETCHING LAYER         │
        │  (data_pipeline.py)                 │
        │                                     │
        │  • NASA POWER API                   │
        │  • NASA OPeNDAP                     │
        │  • Hydrology Data RODS              │
        │  • INPE Satellite Data              │
        └──────────────┬──────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │   2. DATA PROCESSING LAYER          │
        │  (data_pipeline.py)                 │
        │                                     │
        │  • Clean missing values             │
        │  • Merge multi-source data          │
        │  • Normalize & feature engineer     │
        │  • Save to /data/cleaned/           │
        └──────────────┬──────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │   3. ML TRAINING LAYER              │
        │  (train_model.py + ml_models.py)    │
        │                                     │
        │  • LSTM for time series forecast    │
        │  • Random Forest for probability    │
        │  • Isolation Forest for anomalies   │
        │  • Save to /models/                 │
        └──────────────┬──────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │   4. API SERVING LAYER              │
        │  (main.py)                          │
        │                                     │
        │  • Load trained models              │
        │  • Serve predictions via REST API   │
        │  • Cache results                    │
        │  • Real-time inference              │
        └──────────────┬──────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │   5. AUTOMATION LAYER               │
        │  (manage.py)                        │
        │                                     │
        │  • Orchestrate all components       │
        │  • Schedule periodic updates        │
        │  • Monitor system health            │
        └─────────────────────────────────────┘
```

---

## 🚀 Quick Start Integration

### Step 1: Initial Setup

```bash
# Clone/download all files into vaanamweather/
cd vaanamweather

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p models data/raw data/cleaned data/metadata logs
```

### Step 2: Configure Environment

Create `.env` file:
```bash
# Server
HOST=0.0.0.0
PORT=8000

# NASA API
NASA_EARTHDATA_TOKEN=your_token_here

# Admin
ADMIN_TOKEN=secure_admin_token_change_me

# ML
MODEL_DIR=./models

# Logging
LOG_LEVEL=INFO
```

### Step 3: Initial Data Sync

```bash
# Run full synchronization
python manage.py sync

# This will:
# 1. Fetch 3 years of weather data from NASA
# 2. Clean and process the data
# 3. Train ML models
# 4. Save everything to disk
```

### Step 4: Start API

```bash
# Start the API server
python main.py

# API will be available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

---

## 🔗 Component Integration Details

### 1. Data Pipeline → ML Training Integration

**How it works:**

```python
# In manage.py sync command:

# Step 1: Fetch data
pipeline = DataPipeline()
df = pipeline.run_pipeline(lat, lon, start_date, end_date)
# Saves to: data/cleaned/merged_data.csv

# Step 2: Train models
from train_model import train_all_models
results = train_all_models(variables=["temperature", "precipitation"])
# Loads from: data/cleaned/merged_data.csv
# Saves to: models/lstm_temperature.h5, models/rf_temperature.pkl
```

**Data flow:**
```
NASA API → DataPipeline.fetch_all_sources() → 
DataPipeline.merge_datasets() → DataPipeline.clean_data() → 
CSV file → train_model.py → Trained models
```

### 2. ML Models → FastAPI Integration

**How it works:**

```python
# In main.py:

# On startup, load all models
model_registry = ModelRegistry()  # Loads from models/ directory

# On API request:
@app.post("/api/predict")
async def predict_weather(request: PredictionRequest):
    # 1. Fetch historical data
    df = await fetch_historical_data(lat, lon, start, end)
    
    # 2. Get trained LSTM model
    lstm_model = model_registry.get_predictor(variable, "lstm")
    
    # 3. Make prediction
    result = lstm_model.predict(historical_values, n_steps=7)
    
    # 4. Return formatted response
    return {"predictions": result["predictions"], ...}
```

**Model loading sequence:**
```
App Startup → ModelRegistry.__init__() → 
Load .h5 files (LSTM) → Load .pkl files (RF) → 
Models ready in memory → Fast inference on API calls
```

### 3. NASA API → Endpoint Integration

**How it works:**

```python
# In main.py:

# NASA POWER API is called in fetch_historical_data()
async def fetch_historical_data(lat, lon, variable, start, end):
    # Use data_pipeline module
    power_data = NASAPowerAPI.fetch_data(lat, lon, start, end)
    df = NASAPowerAPI.parse_power_data(power_data)
    return df

# This is used by ALL endpoints that need data:
# - /api/probability (statistical + ML probability)
# - /api/predict (LSTM forecasting)
# - /api/trends (trend analysis)
# - /api/anomalies (anomaly detection)
```

**Request flow:**
```
Client → /api/predict → fetch_historical_data() → 
NASA POWER API → Parse response → Feed to LSTM → 
Return prediction to client
```

---

## 📡 API Endpoint Integration

### Original Endpoints (Keep Working)

All original endpoints continue to work:

```bash
# Health check
GET /api/health

# Probability (enhanced with ML)
POST /api/probability
{
  "latitude": 13.0827,
  "longitude": 80.2707,
  "variable": "temperature",
  "threshold": 35,
  "use_ml": true  # NEW: Enable ML enhancement
}

# Time series (original functionality)
POST /api/timeseries

# Source-specific endpoints (original)
GET /api/opendap
GET /api/giovanni
GET /api/rods
GET /api/worldview
GET /api/earthdata
GET /api/inpe
```

### New ML Endpoints

```bash
# ML-based prediction
POST /api/predict
{
  "latitude": 13.0827,
  "longitude": 80.2707,
  "variable": "temperature",
  "forecast_days": 7,
  "use_lstm": true
}

# Trend analysis
POST /api/trends
{
  "latitude": 13.0827,
  "longitude": 80.2707,
  "variable": "temperature",
  "start_date": "2020-01-01",
  "end_date": "2024-12-31",
  "detect_shifts": true
}

# Anomaly detection
POST /api/anomalies
{
  "latitude": 13.0827,
  "longitude": 80.2707,
  "variable": "temperature",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "detection_method": "both"
}

# Model retraining (admin only)
POST /api/retrain
{
  "variables": ["temperature", "precipitation"],
  "force": false,
  "admin_token": "your_admin_token"
}
```

---

## 🔄 Automation Workflow

### Manual Full Sync

```bash
python manage.py sync
```

**What happens:**
1. Fetches latest data from NASA APIs
2. Cleans and merges data
3. Trains/retrains ML models
4. Restarts API (if using Docker)

### Scheduled Automation (Cron)

Add to crontab for monthly updates:

```bash
# Edit crontab
crontab -e

# Add this line for monthly sync (1st of month at 2 AM)
0 2 1 * * cd /path/to/vaanamweather && /path/to/venv/bin/python manage.py sync >> logs/cron.log 2>&1
```

### Docker Automation

```bash
# Build with automation
docker-compose up -d

# Manual sync inside container
docker exec vaanam-weather-api python manage.py sync
```

---

## 🎯 Real-World Usage Examples

### Example 1: Weather App with 7-Day Forecast

```python
import requests

# Get 7-day forecast
response = requests.post("http://localhost:8000/api/predict", json={
    "latitude": 13.0827,
    "longitude": 80.2707,
    "variable": "temperature",
    "forecast_days": 7,
    "use_lstm": True,
    "include_confidence": True
})

data = response.json()
print(f"Tomorrow's temperature: {data['predictions'][0]}°C")
print(f"Confidence: {data['confidence_score']}")
```

### Example 2: Climate Change Dashboard

```python
# Analyze 10-year temperature trend
response = requests.post("http://localhost:8000/api/trends", json={
    "latitude": 28.7041,
    "longitude": 77.1025,
    "variable": "temperature",
    "start_date": "2014-01-01",
    "end_date": "2024-12-31",
    "detect_shifts": True
})

data = response.json()
trend = data['trend_analysis']
print(f"Trend: {trend['trend_direction']}")
print(f"Annual change: {trend['annual_change']}°C/year")
print(f"Predicted next year: {trend['predicted_next_year']}°C")
```

### Example 3: Extreme Weather Alert System

```python
# Check probability of extreme heat
response = requests.post("http://localhost:8000/api/probability", json={
    "latitude": 19.0760,
    "longitude": 72.8777,
    "variable": "temperature",
    "threshold": 42,
    "operator": "greater",
    "use_ml": True
})

data = response.json()
prob = data['probability_analysis']['probability']

if prob > 30:
    print(f"⚠️ High risk of extreme heat: {prob}% probability")
    # Send alert notification
```

### Example 4: Agricultural Planning

```python
# Detect drought patterns
anomaly_response = requests.post("http://localhost:8000/api/anomalies", json={
    "latitude": 20.5937,
    "longitude": 78.9629,
    "variable": "precipitation",
    "start_date": "2024-06-01",
    "end_date": "2024-09-30",
    "detection_method": "both"
})

# Get rainfall forecast
forecast_response = requests.post("http://localhost:8000/api/predict", json={
    "latitude": 20.5937,
    "longitude": 78.9629,
    "variable": "precipitation",
    "forecast_days": 14
})

# Combine for farming decisions
anomalies = anomaly_response.json()
forecast = forecast_response.json()

print(f"Recent anomalies: {anomalies['anomaly_detection']['anomaly_count']}")
print(f"Next 2 weeks average rainfall: {forecast['mean_prediction']} mm")
```

---

## 🔧 Customization & Extension

### Adding a New ML Model

1. **Add model class to `ml_models.py`:**

```python
class GRUWeatherPredictor:
    def __init__(self, seq_length: int = 30):
        self.model = self.build_model()
    
    def build_model(self):
        # Your GRU architecture
        pass
    
    def train(self, data):
        # Training logic
        pass
    
    def predict(self, data):
        # Prediction logic
        pass
```

2. **Add to ModelRegistry:**

```python
def _initialize_models(self):
    for var in variables:
        self.models[f"gru_{var}"] = GRUWeatherPredictor(variable=var)
```

3. **Add endpoint in `main.py`:**

```python
@app.post("/api/predict_gru")
async def predict_with_gru(request: PredictionRequest):
    gru_model = model_registry.get_model(f"gru_{request.variable}")
    result = gru_model.predict(historical_data)
    return result
```

### Adding a New Data Source

1. **Add client to `data_pipeline.py`:**

```python
class NewDataSourceClient:
    @staticmethod
    def fetch_data(lat, lon, start, end):
        # Fetch logic
        return dataframe
```

2. **Integrate in DataPipeline:**

```python
def fetch_all_sources(self, ...):
    if "new_source" in sources:
        new_data = NewDataSourceClient.fetch_data(...)
        results["new_source"] = new_data
```

### Adding Custom Preprocessing

```python
class WeatherPreprocessor:
    def custom_feature(self, df):
        # Add your custom feature engineering
        df['custom_feature'] = ...
        return df
```

---

## 📊 Monitoring & Maintenance

### Check System Status

```bash
python manage.py status
```

Output:
```
==============================================================
SYSTEM STATUS
==============================================================

Data Files: 3
  - weather_data_chennai.csv
  - weather_data_delhi.csv
  - weather_data_mumbai.csv

Model Files: 10
  - lstm_temperature.h5
  - rf_temperature.pkl
  ...

API Status: ✓ RUNNING
  Version: 2.0.0
  ML Models: ready

Log Files: 5
  Latest: manage_20250105.log
==============================================================
```

### View Logs

```bash
# Application logs
tail -f logs/manage_$(date +%Y%m%d).log

# API logs (if using Docker)
docker-compose logs -f
```

### Performance Monitoring

Add to `main.py`:

```python
from prometheus_client import Counter, Histogram

prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')

@app.post("/api/predict")
async def predict_weather(request: PredictionRequest):
    with prediction_duration.time():
        prediction_counter.inc()
        # ... your prediction code
```

---

## 🐛 Troubleshooting

### Issue: Models not loading

**Solution:**
```bash
# Retrain models
python manage.py train --force

# Check model directory
ls -la models/

# Restart API
python main.py
```

### Issue: NASA API timeout

**Solution:**
```bash
# Increase timeout in .env
REQUEST_TIMEOUT=60

# Use cached data
# Cached results are stored for 1 hour automatically
```

### Issue: Out of memory during training

**Solution:**
```python
# In train_model.py, reduce batch size:
history = predictor.train(data, epochs=30, batch_size=16)  # Default: 32

# Or train one variable at a time:
python manage.py train --variables temperature
python manage.py train --variables precipitation
```

---

## 🚢 Production Deployment

### Docker Deployment

```bash
# Build image
docker build -t vaanam-weather-ml:latest .

# Run with docker-compose
docker-compose up -d

# Initial sync
docker exec vaanam-weather-api python manage.py sync
```

### Cloud Deployment (GCP)

```bash
# Update deploy-gcp.sh with ML dependencies
# Deploy
./deploy-gcp.sh

# Run initial sync
gcloud run jobs create vaanam-sync \
  --image gcr.io/${PROJECT_ID}/vaanam-weather-api \
  --command "python" \
  --args "manage.py,sync" \
  --region us-central1
```

---

## 📚 Additional Resources

- **NASA POWER API Docs**: https://power.larc.nasa.gov/docs/
- **TensorFlow Docs**: https://www.tensorflow.org/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Scikit-learn Docs**: https://scikit-learn.org/

---

## ✅ Integration Checklist

- [ ] All files downloaded and in correct structure
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file configured
- [ ] Initial data sync completed (`python manage.py sync`)
- [ ] API running (`python main.py`)
- [ ] Health check passes (`curl localhost:8000/api/health`)
- [ ] ML endpoints working (`/api/predict`, `/api/trends`, `/api/anomalies`)
- [ ] Original endpoints still working (`/api/probability`, etc.)
- [ ] Cron job configured for periodic updates (optional)
- [ ] Docker deployment tested (optional)
- [ ] Cloud deployment completed (optional)

---

**System is now fully integrated and ready for production! 🎉**

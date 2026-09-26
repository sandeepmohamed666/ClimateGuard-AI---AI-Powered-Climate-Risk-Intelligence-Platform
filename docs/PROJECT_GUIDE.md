# ClimateGuard AI v3.0 - Project Guide

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Data Pipeline](#data-pipeline)
5. [Model Training](#model-training)
6. [API Integration](#api-integration)
7. [Dashboard Usage](#dashboard-usage)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)

---

## Project Overview

ClimateGuard AI v3.0 is an end-to-end AI platform for climate risk intelligence that combines historical weather data analysis with real-time weather predictions using machine learning.

### Key Features

- **Real-Time Weather Integration**: Open-Meteo API for live weather data
- **Climate Risk Scoring**: Comprehensive risk assessment (0-100)
- **ML Predictions**: Rainfall, heatwave, and anomaly detection
- **Explainable AI**: SHAP-based model explanations
- **Interactive Dashboard**: Streamlit-based visualization

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd "AI-Powered-Climate Risk Intelligence Platform"
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import streamlit; print('Streamlit installed')"
python -c "import shap; print('SHAP installed')"
```

---

## Project Structure

```
AI-Powered-Climate Risk Intelligence Platform/
├── data/                          # Historical datasets
├── notebooks/                     # Jupyter notebooks
├── src/
│   ├── data_collection/          # Data loading modules
│   │   ├── historical_loader.py
│   │   └── __init__.py
│   ├── data_preprocessing/       # Data cleaning
│   │   ├── data_cleaner.py
│   │   ├── data_understanding.py
│   │   └── __init__.py
│   ├── feature_engineering/      # Feature engineering
│   │   ├── weather_features.py
│   │   ├── feature_selection.py
│   │   └── __init__.py
│   ├── eda/                      # Exploratory analysis
│   │   ├── exploratory_analysis.py
│   │   ├── statistical_analysis.py
│   │   └── __init__.py
│   ├── models/                   # ML models
│   │   ├── climate_profiling.py
│   │   ├── rainfall_prediction.py
│   │   ├── heatwave_prediction.py
│   │   ├── climate_risk_score.py
│   │   ├── anomaly_detection.py
│   │   ├── model_evaluation.py
│   │   └── __init__.py
│   ├── api_integration/          # API clients
│   │   ├── openmeteo.py
│   │   ├── geocoding.py
│   │   ├── prediction_pipeline.py
│   │   └── __init__.py
│   ├── explainability/          # XAI modules
│   │   ├── shap_explainer.py
│   │   └── __init__.py
│   └── utils/                    # Utilities
│       └── __init__.py
├── dashboard/                    # Streamlit dashboard
│   └── app.py
├── trained_models/               # Saved models
├── reports/                      # Generated reports
├── tests/                        # Unit tests
├── docs/                         # Documentation
├── requirements.txt              # Dependencies
├── README.md                     # Project overview
├── Dockerfile                    # Docker configuration
├── docker-compose.yml          # Docker Compose
└── .gitignore                   # Git ignore rules
```

---

## Data Pipeline

### 1. Data Collection

**Historical Data:**
```python
from src.data_collection.historical_loader import HistoricalDataLoader

loader = HistoricalDataLoader("data/IndianWeatherRepository.csv")
data = loader.load_data()
info = loader.get_data_info()
```

**API Data:**
```python
from src.api_integration.openmeteo import OpenMeteoAPI
from src.api_integration.geocoding import GeocodingAPI

# Get coordinates
geocoding = GeocodingAPI()
coords = geocoding.get_coordinates("Delhi", "India")

# Fetch weather
weather_api = OpenMeteoAPI()
weather = weather_api.get_current_weather(coords['latitude'], coords['longitude'])
```

### 2. Data Preprocessing

**Data Cleaning:**
```python
from src.data_preprocessing.data_cleaner import DataCleaner

cleaner = DataCleaner(data)
cleaner.remove_duplicates()
cleaner.handle_missing_values(strategy='mean')
cleaner.remove_impossible_values()
cleaned_data = cleaner.get_cleaned_data()
```

**Data Understanding:**
```python
from src.data_preprocessing.data_understanding import DataUnderstanding

analyzer = DataUnderstanding(data)
report = analyzer.generate_comprehensive_report()
analyzer.print_report()
```

### 3. Feature Engineering

```python
from src.feature_engineering.weather_features import WeatherFeatureEngineer

engineer = WeatherFeatureEngineer(data)
engineer.add_time_features(datetime_col='date')
engineer.add_temperature_features()
engineer.add_rainfall_features()
engineer.add_wind_features()
engineer.add_humidity_features()
engineer.add_air_quality_features()
engineered_data = engineer.get_engineered_data()
```

### 4. Feature Selection

```python
from src.feature_engineering.feature_selection import FeatureSelector

selector = FeatureSelector(X, y)
selected_features = selector.combined_selection(
    methods=['variance_threshold', 'correlation', 'rf_importance'],
    min_votes=2
)
```

---

## Model Training

### Climate Profiling (Clustering)

```python
from src.models.climate_profiling import ClimateProfiler

profiler = ClimateProfiler(X)
profiler.preprocess_data()
labels = profiler.kmeans_clustering(n_clusters=5)
profiler.find_optimal_clusters(max_clusters=10)
```

### Rainfall Prediction

```python
from src.models.rainfall_prediction import RainfallPredictor

predictor = RainfallPredictor(X, y)
predictor.train_logistic_regression()
predictor.train_random_forest()
predictor.train_gradient_boosting()
predictor.select_best_model(metric='f1_score')
predictor.save_model('random_forest', 'trained_models/rainfall_model.pkl')
```

### Heatwave Prediction

```python
from src.models.heatwave_prediction import HeatwavePredictor

predictor = HeatwavePredictor(X, y)
predictor.train_random_forest()
predictor.train_gradient_boosting()
predictor.select_best_model()
```

### Climate Risk Score

```python
from src.models.climate_risk_score import ClimateRiskScorer

scorer = ClimateRiskScorer()
risk = scorer.calculate_overall_risk(
    temperature=35,
    humidity=70,
    rainfall=10,
    pm25=50,
    uv_index=8,
    wind_speed=15,
    pressure=1005
)
```

### Anomaly Detection

```python
from src.models.anomaly_detection import ClimateAnomalyDetector

detector = ClimateAnomalyDetector(X)
detector.isolation_forest_detection()
detector.one_class_svm_detection()
detector.ensemble_detection()
```

---

## API Integration

### Real-Time Prediction Pipeline

```python
from src.api_integration.prediction_pipeline import RealTimePredictionPipeline

pipeline = RealTimePredictionPipeline()

# Run by city
results = pipeline.run_complete_pipeline('Delhi', 'India')

# Run by coordinates
results = pipeline.run_pipeline_by_coordinates(28.6139, 77.2090)

# Get recommendations
recommendations = pipeline.generate_recommendations(results)
```

---

## Dashboard Usage

### Starting the Dashboard

```bash
streamlit run dashboard/app.py
```

### Dashboard Pages

1. **Home**: Project overview and quick start
2. **Live Weather**: Current conditions and forecasts
3. **Historical Data**: Historical weather patterns
4. **Rainfall Prediction**: Rainfall probability
5. **Heatwave Prediction**: Heatwave risk assessment
6. **Climate Risk Score**: Overall climate risk
7. **Air Quality**: AQI monitoring
8. **Weather Classification**: Weather condition classification
9. **Recommendations**: Personalized weather advice

---

## Deployment

### Docker Deployment

1. **Build Docker image**
```bash
docker build -t climateguard-ai .
```

2. **Run with Docker**
```bash
docker run -p 8501:8501 climateguard-ai
```

3. **Run with Docker Compose**
```bash
docker-compose up -d
```

### Streamlit Cloud Deployment

1. Push code to GitHub
2. Connect Streamlit Community Cloud
3. Deploy from repository

### Render Deployment

1. Create `requirements.txt`
2. Create `Procfile`: `web: streamlit run dashboard/app.py --server.port=$PORT`
3. Deploy to Render

---

## Troubleshooting

### Common Issues

**Issue: Module not found error**
```bash
# Solution: Ensure you're in the project directory and venv is activated
cd "AI-Powered-Climate Risk Intelligence Platform"
source venv/bin/activate
pip install -r requirements.txt
```

**Issue: Open-Meteo API timeout**
```bash
# Solution: Check internet connection and API status
# The API is free and doesn't require authentication
```

**Issue: SHAP visualization not showing**
```bash
# Solution: Ensure matplotlib is properly configured
pip install matplotlib
```

**Issue: Docker build fails**
```bash
# Solution: Clear Docker cache and rebuild
docker system prune -a
docker build --no-cache -t climateguard-ai .
```

---

## Performance Optimization

### Model Training

- Use `n_jobs=-1` for parallel processing
- Reduce `n_estimators` for faster training
- Use GPU for XGBoost/LightGBM if available

### API Calls

- Cache API responses
- Use batch requests for multiple cities
- Implement rate limiting

### Dashboard

- Use `@st.cache_data` for expensive computations
- Limit data displayed in tables
- Use lazy loading for large datasets

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## License

This project is for educational and research purposes.

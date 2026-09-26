# ClimateGuard AI v3.0

## AI-Powered Climate Risk Intelligence Platform

### Objective

Develop an end-to-end AI platform that analyzes historical weather data, integrates real-time weather information, predicts climate risks, detects anomalies, explains AI predictions, and provides actionable recommendations through an interactive dashboard.

---

## Project Workflow

```
Historical Dataset
        │
        ▼
Data Collection
        │
        ▼
Data Understanding
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Statistical Analysis
        │
        ▼
Feature Selection
        │
        ▼
Machine Learning Models
        │
        ▼
Explainable AI
        │
        ▼
Open-Meteo API Integration
        │
        ▼
Real-Time Predictions
        │
        ▼
Interactive Dashboard
        │
        ▼
Deployment
```

---

## Problem Statement

Climate change has increased the frequency of heatwaves, heavy rainfall, poor air quality, and other extreme weather events. There is a need for an intelligent platform that combines historical climate data and real-time weather information to predict risks and support informed decision-making.

---

## Objectives

- Analyze historical weather data
- Engineer meaningful climate features
- Predict rainfall and heatwaves
- Calculate a climate risk score
- Detect weather anomalies
- Predict air quality
- Classify weather conditions
- Explain AI predictions
- Integrate live weather data
- Build an interactive dashboard

---

## Project Structure

```
AI-Powered-Climate Risk Intelligence Platform/
├── data/                          # Historical datasets
├── notebooks/                     # Jupyter notebooks for analysis
├── src/
│   ├── data_collection/          # Data collection modules
│   ├── data_preprocessing/       # Data cleaning and preprocessing
│   ├── feature_engineering/      # Feature engineering modules
│   ├── eda/                      # Exploratory data analysis
│   ├── models/                   # ML models
│   ├── api_integration/          # Open-Meteo API integration
│   ├── explainability/          # Explainable AI modules
│   └── utils/                    # Utility functions
├── dashboard/                    # Streamlit dashboard
├── trained_models/               # Saved model files
├── reports/                      # Generated reports
├── tests/                        # Unit tests
├── docs/                         # Documentation
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "AI-Powered-Climate Risk Intelligence Platform"
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Data Sources

### Historical Dataset
- IndianWeatherRepository.csv

### Live Weather Integration
- Open-Meteo Weather API
- Open-Meteo Air Quality API
- Geocoding API

---

## Core Modules

1. **Data Collection & API Integration**
2. **Data Understanding**
3. **Data Cleaning & Preprocessing**
4. **Feature Engineering**
5. **Exploratory Data Analysis**
6. **Statistical Analysis**
7. **Feature Selection**
8. **Climate Profiling**
9. **Rainfall Prediction**
10. **Heatwave Prediction**
11. **Climate Risk Score**
12. **Climate Anomaly Detection**
13. **Air Quality Prediction**
14. **Weather Condition Classification**
15. **Extreme Weather Classification**
16. **Regional Climate Classification**
17. **Weather Recommendation Engine**
18. **Climate Trend Analysis**
19. **Climate Similarity Search**
20. **Explainable AI**
21. **Interactive Dashboard**
22. **Deployment & Documentation**

---

## Usage

### Run the Dashboard
```bash
streamlit run dashboard/app.py
```

### Run Data Analysis
```bash
jupyter notebook notebooks/
```

---

## Technologies

- **Python 3.9+**
- **Pandas, NumPy** - Data manipulation
- **Scikit-learn** - Machine learning
- **XGBoost, LightGBM, CatBoost** - Gradient boosting
- **SHAP, LIME** - Explainable AI
- **Streamlit** - Interactive dashboard
- **Matplotlib, Seaborn, Plotly** - Visualization
- **Open-Meteo API** - Real-time weather data

---

## Future Scope

- NASA POWER integration
- IMD integration
- Satellite imagery
- IoT weather stations
- Flood prediction
- Drought prediction
- Wildfire risk prediction
- Crop advisory system
- Mobile application
- Automated alert notifications

---

## License

This project is for educational and research purposes.

---

## Contact

For questions or contributions, please open an issue on GitHub.

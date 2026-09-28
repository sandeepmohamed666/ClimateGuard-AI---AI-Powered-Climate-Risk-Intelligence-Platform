# ClimateGuard AI Notebook Guide

This folder contains the notebooks for the complete climate-risk machine-learning pipeline. Run them in the order below from the `notebooks` directory.

## Notebook Overview

### 1. `01_data_loading_and_understanding.ipynb`
Loads `IndianWeatherRepository.csv`, inspects data types, missing values, descriptive statistics, and regional coverage, then writes `01_raw_data.csv`.

### 2. `02_data_cleaning_and_preprocessing.ipynb`
Removes duplicate records, imputes missing numeric values, checks impossible weather values, and profiles outliers. Writes `02_cleaned_data.csv`.

### 3. `03_feature_reduction.ipynb`
Removes duplicate timestamp and unit-conversion representations before feature engineering. Writes `03_feature_reduced_data.csv`.

### 4. `03_feature_engineering.ipynb`
Creates temporal, temperature, rainfall, wind, pressure, humidity, UV, visibility, cloud, air-quality, geographic, statistical, and forecasting features. Writes `03_engineered_data.csv`.

### 5. `04_eda_and_statistical_analysis.ipynb`
Performs exploratory and statistical analysis, including distributions, boxplots, correlations, scatter plots, time trends, regional summaries, confidence intervals, normality tests, and outlier analysis. Writes `04_eda_data.csv` and report plots.

### 6. `05_feature_selection.ipynb`
Selects useful numeric predictors using variance thresholding, correlation filtering, univariate F-tests, random-forest importance, recursive feature elimination, and a combined selection report. Writes `05_selected_features.csv` and `selected_features_list.txt`.

### 7. `06_climate_profiling.ipynb`
Profiles climate patterns with K-Means, DBSCAN, Gaussian Mixture, and agglomerative clustering. Evaluates cluster characteristics and adds climate-zone labels. Writes `06_climate_zones.csv`.

### 8. `07_rainfall_prediction.ipynb`
Creates a binary rainfall target from precipitation and compares Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting models. Saves the best model as `rainfall_model.pkl` and evaluation plots.

### 9. `08_heatwave_prediction.ipynb`
Creates a binary heatwave target from temperature above 35 degrees C and compares Logistic Regression, Random Forest, and Gradient Boosting models. Saves the best model as `heatwave_model.pkl` and evaluation plots.

### 10. `09_climate_risk_score.ipynb`
Calculates component and overall climate-risk scores, assigns risk categories, analyzes risk by climate zone and over time, and identifies high-risk records. Writes `09_climate_risk_scores.csv`.

### 11. `10_anomaly_detection.ipynb`
Detects unusual weather records with Isolation Forest, One-Class SVM, Local Outlier Factor, Elliptic Envelope, and majority-vote ensemble detection. Writes `10_anomaly_detection.csv` and saves detector artifacts.

### 12. `11_explainable_ai.ipynb`
Loads the rainfall model and uses SHAP-compatible explainers to calculate feature importance, summary, bar, waterfall, decision, and individual-instance explanations. Writes SHAP plots and `shap_feature_importance.csv`.

## Data Flow

```text
IndianWeatherRepository.csv
  -> 01_raw_data.csv
  -> 02_cleaned_data.csv
  -> 03_feature_reduced_data.csv
  -> 03_engineered_data.csv
  -> 04_eda_data.csv
  -> 05_selected_features.csv
  -> 06_climate_zones.csv
  -> rainfall_model.pkl and heatwave_model.pkl
  -> 09_climate_risk_scores.csv
  -> 10_anomaly_detection.csv
  -> SHAP explanations
```

## Running the Notebooks

Install the project dependencies, then open Jupyter from this directory:

```bash
pip install -r ../requirements.txt
jupyter notebook
```

The notebooks use files under `../data/processed/`, save plots and reports under `../reports/`, and save trained models under `../trained_models/`.

## Column Usage Audit

- The raw dataset contains 42 columns. `03_feature_reduction.ipynb` removes these 8 duplicate or unit-conversion columns: `last_updated`, `temperature_fahrenheit`, `wind_mph`, `pressure_in`, `precip_in`, `feels_like_fahrenheit`, `visibility_miles`, and `gust_mph`. The remaining 34 columns continue through the pipeline.
- Retained numeric columns are used for feature engineering, statistical analysis, clustering, prediction, anomaly detection, or explainability. Date/time, location, text, and categorical columns are used for profiling, grouping, feature derivation, or reporting where applicable.
- `rain_target` and `heatwave_target` are prediction targets, not input predictors. `climate_zone`, `overall_score`, `category`, `color`, `weights_used`, `is_anomaly`, and `anomaly_score` are derived outputs and are excluded from model inputs when they would cause leakage.
- Columns not used as direct numeric model predictors include `country`, `location_name`, `region`, `timezone`, `condition_text`, `wind_direction`, `sunrise`, `sunset`, `moonrise`, `moonset`, and `moon_phase`. They are retained for descriptive analysis, grouping, or possible future categorical/time feature engineering.

## Main Outputs

- Processed data: `../data/processed/`
- Reports and plots: `../reports/`
- Trained models: `../trained_models/`

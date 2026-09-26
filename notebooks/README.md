# ClimateGuard AI v3.0 - Jupyter Notebooks

This directory contains Jupyter notebooks for the complete ML pipeline of ClimateGuard AI v3.0.

## Dataset Location
- **Raw Data**: `../data/raw/IndianWeatherRepository.csv`

## Notebook Sequence

### 1. 01_data_loading_and_understanding.ipynb
**Purpose**: Load and understand the historical weather dataset

**Key Steps**:
- Load IndianWeatherRepository.csv
- Analyze dataset structure and basic statistics
- Identify missing values and data types
- Explore regional distribution
- Generate comprehensive data understanding report

**Output**: `../data/processed/01_raw_data.csv`

---

### 2. 02_data_cleaning_and_preprocessing.ipynb
**Purpose**: Clean and preprocess the weather data

**Key Steps**:
- Remove duplicate rows
- Handle missing values using mean imputation
- Remove impossible weather values
- Detect outliers using IQR and Z-score methods

**Output**: `../data/processed/02_cleaned_data.csv`

---

### 3. 03_feature_engineering.ipynb
**Purpose**: Engineer advanced climate features

**Key Steps**:
- Add time-based features (hour, day, month, season)
- Add temperature features (heat index, wind chill)
- Add rainfall features (rain intensity, dry periods)
- Add wind features (wind direction categories)
- Add pressure, humidity, UV, visibility, cloud features
- Add air quality features (AQI categories)
- Add geographic features (climate bands)

**Output**: `../data/processed/03_engineered_data.csv`

---

### 4. 04_eda_and_statistical_analysis.ipynb
**Purpose**: Perform exploratory data analysis and statistical analysis

**Key Steps**:
- Univariate distribution analysis
- Boxplots for outlier detection
- Correlation heatmap
- Scatter matrix
- Time series analysis
- Monthly aggregate analysis
- Regional analysis
- Descriptive statistics
- Confidence intervals
- Normality tests
- Outlier detection

**Output**: `../data/processed/04_eda_data.csv`, plots in `../reports/`

---

### 5. 05_feature_selection.ipynb
**Purpose**: Select most important features for ML models

**Key Steps**:
- Variance threshold selection
- Correlation-based selection
- Univariate feature selection (F-test)
- Random Forest feature importance
- Recursive Feature Elimination (RFE)
- Combined feature selection

**Output**: `../data/processed/05_selected_features.csv`, `../data/processed/selected_features_list.txt`

---

### 6. 06_climate_profiling.ipynb
**Purpose**: Climate zone classification using clustering

**Key Steps**:
- Find optimal number of clusters
- K-Means clustering
- DBSCAN clustering
- Gaussian Mixture clustering
- Agglomerative clustering
- Evaluate clustering results
- Analyze cluster characteristics

**Output**: `../data/processed/06_climate_zones.csv`, plots in `../reports/`

---

### 7. 07_rainfall_prediction.ipynb
**Purpose**: Train ML models for rainfall prediction

**Key Steps**:
- Prepare features and target (rainfall > 0)
- Train Logistic Regression
- Train Decision Tree
- Train Random Forest
- Train Gradient Boosting
- Select best model
- Evaluate model performance
- Generate confusion matrix, ROC curve, PR curve
- Analyze feature importance
- Save best model

**Output**: `../trained_models/rainfall_model.pkl`, plots in `../reports/`

---

### 8. 08_heatwave_prediction.ipynb
**Purpose**: Train ML models for heatwave prediction

**Key Steps**:
- Prepare features and target (temperature > 35°C)
- Train Logistic Regression
- Train Random Forest
- Train Gradient Boosting
- Select best model
- Evaluate model performance
- Generate confusion matrix and ROC curve
- Analyze feature importance
- Save best model

**Output**: `../trained_models/heatwave_model.pkl`, plots in `../reports/`

---

### 9. 09_climate_risk_score.ipynb
**Purpose**: Calculate comprehensive climate risk scores

**Key Steps**:
- Calculate risk scores for all records
- Analyze risk score distribution
- Visualize risk categories
- Analyze risk by climate zone
- Analyze component risk scores
- Identify high risk events
- Analyze risk score trends over time

**Output**: `../data/processed/09_climate_risk_scores.csv`, plots in `../reports/`

---

### 10. 10_anomaly_detection.ipynb
**Purpose**: Detect weather anomalies using various algorithms

**Key Steps**:
- Isolation Forest detection
- One-Class SVM detection
- Local Outlier Factor detection
- Elliptic Envelope detection
- Ensemble anomaly detection
- Analyze anomaly details
- Visualize anomaly scores
- Analyze anomalies by climate zone

**Output**: `../data/processed/10_anomaly_detection.csv`, plots in `../reports/`

---

### 11. 11_explainable_ai.ipynb
**Purpose**: Explain model predictions using SHAP

**Key Steps**:
- Load trained rainfall model
- Initialize SHAP explainer
- Calculate SHAP values
- Extract feature importance
- Generate SHAP summary plot
- Generate SHAP bar plot
- Generate SHAP waterfall plot
- Generate SHAP decision plot
- Explain single instances
- Generate comprehensive explanation report

**Output**: `../reports/shap_*.png`, `../reports/shap_feature_importance.csv`

---

## How to Run the Notebooks

### Prerequisites
1. Ensure all dependencies are installed:
```bash
pip install -r ../requirements.txt
```

2. Create necessary directories:
```bash
mkdir -p ../data/processed
mkdir -p ../reports
mkdir -p ../trained_models
```

### Running the Notebooks

**Option 1: Run sequentially**
```bash
jupyter notebook
```
Then open and run each notebook in order (01 → 11).

**Option 2: Run all at once**
```bash
jupyter nbconvert --to notebook --execute *.ipynb
```

### Important Notes

- Each notebook saves its output to `../data/processed/`
- Plots are saved to `../reports/`
- Trained models are saved to `../trained_models/`
- Notebooks must be run in sequence as each depends on the previous output

## Data Flow

```
IndianWeatherRepository.csv (raw)
    ↓
01_data_loading_and_understanding → 01_raw_data.csv
    ↓
02_data_cleaning_and_preprocessing → 02_cleaned_data.csv
    ↓
03_feature_engineering → 03_engineered_data.csv
    ↓
04_eda_and_statistical_analysis → 04_eda_data.csv
    ↓
05_feature_selection → 05_selected_features.csv
    ↓
06_climate_profiling → 06_climate_zones.csv
    ↓
07_rainfall_prediction → rainfall_model.pkl
    ↓
08_heatwave_prediction → heatwave_model.pkl
    ↓
09_climate_risk_score → 09_climate_risk_scores.csv
    ↓
10_anomaly_detection → 10_anomaly_detection.csv
    ↓
11_explainable_ai → SHAP explanations
```

## Troubleshooting

**Issue: Module not found error**
```bash
# Ensure you're in the notebooks directory
cd notebooks
# Or add src to path in each notebook (already done)
```

**Issue: Data file not found**
```bash
# Ensure the raw data is in the correct location
ls ../data/raw/IndianWeatherRepository.csv
```

**Issue: Out of memory**
- Reduce the number of samples by adding `.head(10000)` when loading data
- Use fewer features in feature selection

## Next Steps

After running all notebooks:
1. Review the generated reports in `../reports/`
2. Use the trained models in the Streamlit dashboard
3. Deploy using Docker or Streamlit Cloud

## Additional Resources

- Project README: `../README.md`
- Project Guide: `../docs/PROJECT_GUIDE.md`
- Dashboard: `../dashboard/app.py`

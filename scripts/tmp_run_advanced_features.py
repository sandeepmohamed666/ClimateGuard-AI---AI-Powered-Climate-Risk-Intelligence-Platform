import time
import pandas as pd
from src.feature_engineering.weather_features import WeatherFeatureEngineer

# Load cleaned data
p = 'data/processed/02_cleaned_data.csv'
print('Loading', p)
df = pd.read_csv(p)
print('Loaded df shape:', df.shape)

engineer = WeatherFeatureEngineer(df)
print('Initialized engineer')

start = time.time()
engineer.add_statistical_features(compute_quantiles=False, windows=[3,6,12])
print(f"Statistical features done: {time.time()-start:.2f}s")

start = time.time()
engineer.add_expanding_features()
print(f"Expanding features done: {time.time()-start:.2f}s")

start = time.time()
engineer.add_exponential_features(spans=[3,6])
print(f"Exponential features done: {time.time()-start:.2f}s")

start = time.time()
engineer.add_lag_features(lags=[1,3,6])
print(f"Lag features done: {time.time()-start:.2f}s")

start = time.time()
engineer.add_trend_features(windows=[3,6])
print(f"Trend features done: {time.time()-start:.2f}s")

start = time.time()
engineer.add_interaction_features()
print(f"Interaction features done: {time.time()-start:.2f}s")

start = time.time()
engineer.add_derived_weather_features()
print(f"Derived weather features done: {time.time()-start:.2f}s")

# prepare forecasting features if precipitation exists
target_col = 'precipitation' if 'precipitation' in engineer.df.columns else None
start = time.time()
engineer.prepare_forecasting_features(target_col=target_col, future_horizon=3, lags=[1,3], windows=[3], spans=[3])
print(f"Forecasting features done: {time.time()-start:.2f}s")

eng = engineer.get_engineered_data()
print('Final shape:', eng.shape)
eng.to_csv('data/processed/03_engineered_data_from_script.csv', index=False)
print('Saved to data/processed/03_engineered_data_from_script.csv')

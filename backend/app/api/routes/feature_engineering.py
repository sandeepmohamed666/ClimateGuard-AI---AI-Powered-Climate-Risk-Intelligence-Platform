from fastapi import APIRouter

router = APIRouter()

@router.get('')
async def feature_engineering():
    return {
        'original_features': ['temperature', 'humidity', 'pressure', 'wind', 'rainfall', 'cloud', 'visibility', 'uv', 'latitude', 'longitude', 'date', 'time'],
        'time_features': ['year', 'month', 'day', 'hour', 'minute', 'week', 'quarter', 'season', 'weekend', 'holiday', 'month_start', 'month_end', 'year_start', 'year_end', 'cyclic_month', 'cyclic_hour'],
        'weather_physics': ['heat_index', 'wind_chill', 'dew_point', 'apparent_temperature', 'vapor_pressure', 'wet_bulb', 'comfort_index', 'feels_like'],
        'risk_features': ['heatwave_flag', 'flood_flag', 'heavy_rain_flag', 'high_wind_flag', 'high_uv_flag', 'high_aqi_flag', 'drought_flag', 'extreme_weather_flag'],
        'interaction_features': ['temperature_humidity', 'temperature_wind', 'pressure_wind', 'pressure_rainfall', 'cloud_rainfall', 'humidity_wind', 'humidity_pressure', 'aqi_wind', 'uv_temperature'],
        'statistical_features': ['rolling_mean', 'rolling_std', 'rolling_max', 'rolling_min', 'rolling_median', 'rolling_range'],
        'lag_features': ['lag_1', 'lag_3', 'lag_6', 'lag_12', 'lag_24'],
        'trend_features': ['slope', 'trend', 'moving_average', 'linear_trend'],
        'forecast_features': ['previous_temperature', 'previous_rainfall', 'previous_humidity', 'previous_aqi', 'future_target', 'forecast_horizon'],
    }

"""
Weather Feature Engineering Module
Creates advanced climate features from raw weather data
"""

import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherFeatureEngineer:
    """Engineer advanced weather features"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the feature engineer
        
        Args:
            df: Input DataFrame with raw weather data
        """
        self.df = df.copy()
        self.feature_report = {}
    
    TIMESTAMP_CANDIDATES = [
        'datetime', 'date', 'last_updated', 'last_updated_epoch', 'timestamp',
        'localtime', 'observation_time', 'time', 'epoch', 'created_at', 'updated_at'
    ]

    def add_time_features(self, datetime_col: str = None) -> pd.DataFrame:
        """
        Add time-based and seasonal features to the dataset.

        Args:
            datetime_col: Optional explicit timestamp column name.

        Returns:
            DataFrame with time features added.
        """
        ts_col = datetime_col if datetime_col in self.df.columns else self._detect_timestamp_column()

        if ts_col is None:
            self.df['generated_timestamp'] = pd.Timestamp.now()
            ts_col = 'generated_timestamp'
            logger.warning("No timestamp column found. Generated current timestamp column 'generated_timestamp'.")
        else:
            self.df[ts_col] = self._parse_timestamp_column(self.df[ts_col], ts_col)
            if self.df[ts_col].isna().all():
                logger.warning(f"Timestamp parsing failed for column '{ts_col}'. Generating fallback timestamp.")
                self.df[ts_col] = pd.Timestamp.now()

        self.timestamp_col = ts_col
        self.df = self.df.sort_values(by=ts_col).reset_index(drop=True)
        dt = self.df[ts_col]

        self.df['year'] = dt.dt.year
        self.df['month'] = dt.dt.month
        self.df['day'] = dt.dt.day
        self.df['hour'] = dt.dt.hour
        self.df['minute'] = dt.dt.minute
        self.df['second'] = dt.dt.second
        self.df['day_of_week'] = dt.dt.dayofweek
        self.df['day_name'] = dt.dt.day_name()
        self.df['month_name'] = dt.dt.month_name()
        self.df['week_of_year'] = dt.dt.isocalendar().week.astype(int)
        self.df['quarter'] = dt.dt.quarter
        self.df['day_of_year'] = dt.dt.dayofyear
        self.df['days_in_month'] = dt.dt.days_in_month
        self.df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        self.df['is_weekday'] = (dt.dt.dayofweek < 5).astype(int)
        self.df['is_month_start'] = dt.dt.is_month_start.astype(int)
        self.df['is_month_end'] = dt.dt.is_month_end.astype(int)
        self.df['is_quarter_start'] = dt.dt.is_quarter_start.astype(int)
        self.df['is_quarter_end'] = dt.dt.is_quarter_end.astype(int)
        self.df['is_year_start'] = dt.dt.is_year_start.astype(int)
        self.df['is_year_end'] = dt.dt.is_year_end.astype(int)
        self.df['is_leap_year'] = dt.dt.is_leap_year.astype(int)

        self.df['meteorological_season'] = dt.dt.month.apply(self._meteorological_season)
        self.df['climate_season'] = dt.dt.month.apply(self._climate_season)
        self.df['monsoon_phase'] = dt.dt.month.apply(self._monsoon_phase)

        self.df['is_midnight'] = (dt.dt.hour == 0).astype(int)
        self.df['is_dawn'] = dt.dt.hour.between(5, 7).astype(int)
        self.df['is_morning'] = dt.dt.hour.between(8, 11).astype(int)
        self.df['is_afternoon'] = dt.dt.hour.between(12, 16).astype(int)
        self.df['is_evening'] = dt.dt.hour.between(17, 20).astype(int)
        self.df['is_night'] = dt.dt.hour.isin([21, 22, 23]).astype(int)

        self.df['business_hours'] = ((self.df['is_weekday'] == 1) & dt.dt.hour.between(9, 17)).astype(int)
        self.df['working_day'] = self.df['is_weekday']

        self.feature_report['time_features'] = True
        self.feature_report['timestamp_column'] = ts_col

        self.add_cyclical_time_features()
        return self.df

    def add_cyclical_time_features(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Add cyclical encodings for time-based fields.

        Args:
            columns: Optional list of time fields to encode.

        Returns:
            DataFrame with cyclical time features added.
        """
        if columns is None:
            columns = ['month', 'day_of_year', 'week_of_year', 'hour', 'minute']

        for col in columns:
            if col in self.df.columns:
                period = {
                    'month': 12,
                    'day_of_year': 365,
                    'week_of_year': 52,
                    'hour': 24,
                    'minute': 60,
                }.get(col, None)
                if period is None or self.df[col].isna().all():
                    continue

                self.df[f'{col}_sin'] = np.sin(2 * np.pi * self.df[col] / period)
                self.df[f'{col}_cos'] = np.cos(2 * np.pi * self.df[col] / period)

        self.feature_report['cyclical_time_features'] = True
        return self.df

    def add_statistical_features(
        self,
        columns: Optional[List[str]] = None,
        windows: List[int] = [3, 6, 12, 24, 48],
        quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95],
        compute_quantiles: bool = True
    ) -> pd.DataFrame:
        """
        Add rolling statistical features for numeric columns.

        Args:
            columns: Feature columns to transform. If None, all numeric columns are used.
            windows: Rolling window sizes.
            quantiles: Rolling quantiles to compute.

        Returns:
            DataFrame with rolling statistical features added.
        """
        numeric_cols = columns or [c for c in self.df.columns if is_numeric_dtype(self.df[c])]
        exclude = {self.timestamp_col} if hasattr(self, 'timestamp_col') else set()
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        if not numeric_cols:
            return self.df

        # operate on a numeric subset to avoid repeated dataframe access
        df_num = self.df[numeric_cols]

        for window in windows:
            roll = df_num.rolling(window=window, min_periods=1)

            # batch compute common aggregations (faster than per-column loops)
            agg = roll.aggregate(['mean', 'median', 'std', 'var', 'min', 'max', 'skew', 'kurt'])
            # agg is a MultiIndex columns: (col, stat)
            for stat in ['mean', 'median', 'std', 'var', 'min', 'max', 'skew', 'kurt']:
                stats_df = agg.xs(stat, axis=1, level=1)
                stats_df.columns = [f"{c}_roll_{stat}_{window}" for c in stats_df.columns]
                self.df = pd.concat([self.df, stats_df], axis=1)

            # range from max - min
            max_df = agg.xs('max', axis=1, level=1)
            min_df = agg.xs('min', axis=1, level=1)
            range_df = max_df - min_df
            range_df.columns = [f"{c}_roll_range_{window}" for c in range_df.columns]
            self.df = pd.concat([self.df, range_df], axis=1)

            # quantiles are more expensive; compute only when requested
            if compute_quantiles and quantiles:
                for q in quantiles:
                    qdf = roll.quantile(q)
                    qdf.columns = [f"{c}_roll_q{int(q*100)}_{window}" for c in qdf.columns]
                    self.df = pd.concat([self.df, qdf], axis=1)

        self.feature_report['statistical_features'] = True
        return self.df

    def add_expanding_features(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Add expanding set statistical features.

        Args:
            columns: Numeric columns to expand. If None, uses all numeric columns.

        Returns:
            DataFrame with expanding features added.
        """
        numeric_cols = columns or [c for c in self.df.columns if is_numeric_dtype(self.df[c])]
        exclude = {self.timestamp_col} if hasattr(self, 'timestamp_col') else set()
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        if not numeric_cols:
            return self.df

        df_num = self.df[numeric_cols]
        self.df = pd.concat([
            self.df,
            df_num.expanding(min_periods=1).mean().add_suffix('_expanding_mean'),
            df_num.expanding(min_periods=1).std().add_suffix('_expanding_std'),
            df_num.expanding(min_periods=1).min().add_suffix('_expanding_min'),
            df_num.expanding(min_periods=1).max().add_suffix('_expanding_max'),
            df_num.expanding(min_periods=1).var().add_suffix('_expanding_var')
        ], axis=1)

        self.feature_report['expanding_features'] = True
        return self.df

    def add_exponential_features(
        self,
        columns: Optional[List[str]] = None,
        spans: List[int] = [3, 6, 12]
    ) -> pd.DataFrame:
        """
        Add exponential moving average and standard deviation features.

        Args:
            columns: Numeric columns to transform. If None, uses all numeric columns.
            spans: EWM span values.

        Returns:
            DataFrame with exponential features added.
        """
        numeric_cols = columns or [c for c in self.df.columns if is_numeric_dtype(self.df[c])]
        exclude = {self.timestamp_col} if hasattr(self, 'timestamp_col') else set()
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        if not numeric_cols:
            return self.df

        df_num = self.df[numeric_cols]
        for span in spans:
            self.df = pd.concat([
                self.df,
                df_num.ewm(span=span, min_periods=1).mean().add_suffix(f'_ema_{span}'),
                df_num.ewm(span=span, min_periods=1).std().add_suffix(f'_ewm_std_{span}')
            ], axis=1)

        self.feature_report['exponential_features'] = True
        return self.df

    def add_lag_features(
        self,
        columns: Optional[List[str]] = None,
        lags: List[int] = [1, 3, 6, 12, 24]
    ) -> pd.DataFrame:
        """
        Add lag features for weather-related columns.

        Args:
            columns: Columns to create lags for. If None, uses weather and risk columns.
            lags: Lag offsets to create.

        Returns:
            DataFrame with lag features added.
        """
        default_patterns = [
            'temperature', 'humidity', 'pressure', 'wind', 'precipitation',
            'rain', 'cloud', 'uv', 'visibility', 'pm2_5', 'pm10', 'overall_score'
        ]
        if columns is None:
            columns = [c for c in self.df.columns if any(p in c.lower() for p in default_patterns) and is_numeric_dtype(self.df[c])]

        if not columns:
            return self.df

        df_cols = self.df[columns]
        for lag in lags:
            shifted = df_cols.shift(lag)
            shifted.columns = [f"{c}_lag_{lag}" for c in shifted.columns]
            self.df = pd.concat([self.df, shifted], axis=1)

        self.feature_report['lag_features'] = True
        return self.df

    def add_trend_features(
        self,
        columns: Optional[List[str]] = None,
        windows: List[int] = [3, 6, 12]
    ) -> pd.DataFrame:
        """
        Create trend and momentum features.

        Args:
            columns: Numeric columns to process. If None, uses all numeric columns.
            windows: Rolling windows for slope and trend.

        Returns:
            DataFrame with trend features added.
        """
        numeric_cols = columns or [c for c in self.df.columns if is_numeric_dtype(self.df[c])]
        exclude = {self.timestamp_col} if hasattr(self, 'timestamp_col') else set()
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        if not numeric_cols:
            return self.df

        # Basic trend features (fast)
        df_num = self.df[numeric_cols]
        diff_df = df_num.diff().add_suffix('_diff')
        pct_df = df_num.pct_change().replace([np.inf, -np.inf], np.nan).add_suffix('_pct_change')
        accel_df = df_num.diff().diff().add_suffix('_acceleration')
        momentum_df = (df_num * df_num.diff()).add_suffix('_momentum')

        self.df = pd.concat([self.df, diff_df, pct_df, accel_df, momentum_df], axis=1)

        # Moving trend (rolling mean diff) - efficient batch compute
        for window in windows:
            moving_mean = df_num.rolling(window=window, min_periods=1).mean()
            moving_trend = moving_mean.diff().add_suffix(f'_moving_trend_{window}')
            self.df = pd.concat([self.df, moving_trend], axis=1)

        # Note: slope via per-window polyfit is expensive; keep it out of the default path
        self.feature_report['trend_features'] = True

        return self.df

    def add_interaction_features(self, pairs: Optional[List[tuple]] = None) -> pd.DataFrame:
        """
        Add engineered interaction features for weather variables.

        Args:
            pairs: List of tuple pairs to multiply.

        Returns:
            DataFrame with interaction features added.
        """
        if pairs is None:
            pairs = [
                ('temperature', 'humidity'),
                ('temperature', 'wind_speed'),
                ('temperature', 'uv_index'),
                ('humidity', 'pressure'),
                ('pressure', 'wind_speed'),
                ('cloud_cover', 'precipitation'),
                ('precipitation', 'humidity'),
                ('visibility', 'cloud_cover'),
                ('wind_speed', 'precipitation'),
                ('pm2_5', 'temperature'),
                ('uv_index', 'cloud_cover')
            ]

        for a, b in pairs:
            if a in self.df.columns and b in self.df.columns:
                self.df[f'{a}_x_{b}'] = self.df[a] * self.df[b]

        self.feature_report['interaction_features'] = True
        return self.df

    def add_derived_weather_features(self) -> pd.DataFrame:
        """
        Add derived weather physics and comfort features.

        Returns:
            DataFrame with derived weather features added.
        """
        temp_col = self._find_column(['temperature', 'temp', 'Temperature'])
        humidity_col = self._find_column(['humidity', 'relative_humidity', 'Humidity'])
        wind_col = self._find_column(['wind_speed', 'wind', 'Wind Speed'])
        rain_col = self._find_column(['precipitation', 'rain', 'Rain'])
        pressure_col = self._find_column(['pressure', 'Pressure'])
        uv_col = self._find_column(['uv_index', 'uv'])
        pm25_col = self._find_column(['pm2_5', 'PM2.5', 'pm25'])

        if temp_col and humidity_col:
            self.df['dew_point'] = self._calculate_dew_point(self.df[temp_col], self.df[humidity_col])
            self.df['heat_index'] = self._calculate_heat_index(self.df[temp_col], self.df[humidity_col])
            self.df['relative_humidity_class'] = self.df[humidity_col].apply(self._relative_humidity_class)
            self.df['vapor_pressure'] = self._calculate_vapor_pressure(self.df[temp_col], self.df[humidity_col])
            self.df['absolute_humidity'] = self._calculate_absolute_humidity(self.df[temp_col], self.df[humidity_col])
            self.df['cooling_degree_days'] = np.maximum(self.df[temp_col] - 18.0, 0.0)
            self.df['heating_degree_days'] = np.maximum(18.0 - self.df[temp_col], 0.0)
            self.df['growing_degree_days'] = np.maximum((self.df[temp_col] + 10.0) / 2.0 - 10.0, 0.0)

        if temp_col and wind_col:
            self.df['wind_chill'] = self._calculate_wind_chill(self.df[temp_col], self.df[wind_col])
            self.df['apparent_temperature'] = self._calculate_apparent_temperature(
                self.df[temp_col], self.df[humidity_col] if humidity_col else self.df[temp_col], self.df[wind_col]
            )
            self.df['wet_bulb_temperature'] = self._calculate_wet_bulb_temperature(
                self.df[temp_col], self.df[humidity_col], self.df[pressure_col]
            )

        if rain_col:
            self.df['rainfall_intensity'] = np.where(self.df[rain_col] > 0, self.df[rain_col] / (self.df[rain_col].sum() + 1e-6), 0.0)

        if uv_col:
            self.df['high_uv_alert'] = (self.df[uv_col] >= 8).astype(int)

        if pm25_col:
            self.df['poor_air_quality'] = (self.df[pm25_col] > 55).astype(int)

        self.feature_report['derived_weather_features'] = True
        return self.df

    def add_risk_features(self) -> pd.DataFrame:
        """
        Add risk indicator flags for extreme weather.

        Returns:
            DataFrame with binary risk flags added.
        """
        temp_col = self._find_column(['temperature', 'temp'])
        wind_col = self._find_column(['wind_speed', 'wind'])
        rain_col = self._find_column(['precipitation', 'rain'])
        visibility_col = self._find_column(['visibility'])
        uv_col = self._find_column(['uv_index', 'uv'])
        pm25_col = self._find_column(['pm2_5', 'pm25'])
        pm10_col = self._find_column(['pm10'])

        if temp_col:
            self.df['heatwave_flag'] = (self.df[temp_col] >= 40).astype(int)
            self.df['cold_wave_flag'] = (self.df[temp_col] <= 5).astype(int)
            self.df['extreme_temperature_flag'] = ((self.df[temp_col] <= 0) | (self.df[temp_col] >= 45)).astype(int)

        if rain_col:
            self.df['heavy_rain_flag'] = (self.df[rain_col] >= 50).astype(int)
            self.df['extreme_rain_flag'] = (self.df[rain_col] >= 100).astype(int)
            self.df['storm_flag'] = (self.df[rain_col] >= 20).astype(int)
            self.df['flood_risk_flag'] = ((self.df[rain_col] >= 50) & self.df[rain_col].notna()).astype(int)
            self.df['drought_flag'] = ((self.df[rain_col] == 0) & (self.df[temp_col] >= 35)).astype(int) if temp_col is not None else 0

        if wind_col:
            self.df['high_wind_flag'] = (self.df[wind_col] >= 15).astype(int)
            self.df['cyclone_flag'] = (self.df[wind_col] >= 32).astype(int)

        if visibility_col:
            self.df['poor_visibility_flag'] = (self.df[visibility_col] < 5).astype(int)
            self.df['dense_fog_flag'] = (self.df[visibility_col] < 1).astype(int)

        if uv_col:
            self.df['high_uv_flag'] = (self.df[uv_col] >= 8).astype(int)

        if pm25_col or pm10_col:
            aqi_val = self.df[pm25_col] if pm25_col else self.df[pm10_col]
            self.df['poor_air_quality_flag'] = (aqi_val > 55).astype(int)

        self.feature_report['risk_features'] = True
        return self.df

    def prepare_forecasting_features(
        self,
        target_col: Optional[str] = None,
        future_horizon: int = 1,
        lags: List[int] = [1, 3, 6, 12, 24],
        windows: List[int] = [3, 6, 12, 24],
        spans: List[int] = [3, 6]
    ) -> pd.DataFrame:
        """
        Prepare the dataset for forecasting and time-series modeling.

        Args:
            target_col: Column name of the prediction target.
            future_horizon: Number of future periods to create.
            lags: Lag windows to generate.
            windows: Rolling windows to generate.
            spans: EWM spans to generate.

        Returns:
            DataFrame with forecasting features added.
        """
        if not hasattr(self, 'timestamp_col') or self.timestamp_col not in self.df.columns:
            self.add_time_features()

        self.df = self.df.sort_values(by=self.timestamp_col).reset_index(drop=True)
        self.df['time_index'] = np.arange(len(self.df))

        self.add_lag_features(columns=[target_col] if target_col else None, lags=lags)
        self.add_statistical_features(columns=[target_col] if target_col else None, windows=windows)
        self.add_exponential_features(columns=[target_col] if target_col else None, spans=spans)
        self.add_trend_features(columns=[target_col] if target_col else None, windows=windows)

        if target_col is not None:
            for h in range(1, future_horizon + 1):
                self.df[f'{target_col}_future_{h}'] = self.df[target_col].shift(-h)

        self.feature_report['forecasting_features'] = True
        return self.df

    def validate_timestamps(self, datetime_col: str = None) -> Dict[str, bool]:
        """
        Validate timestamp integrity in the dataset.

        Args:
            datetime_col: Optional timestamp column name.

        Returns:
            Dictionary of validation results.
        """
        ts_col = datetime_col if datetime_col in self.df.columns else getattr(self, 'timestamp_col', None)
        if ts_col is None:
            ts_col = self._detect_timestamp_column()
        if ts_col is None:
            logger.warning('No timestamp column to validate.')
            return {'has_timestamp': False}

        self.df[ts_col] = self._parse_timestamp_column(self.df[ts_col], ts_col)
        invalid = self.df[ts_col].isna().any()
        duplicates = self.df[ts_col].duplicated().any()
        out_of_order = not self.df[ts_col].is_monotonic_increasing
        timezone_mismatch = False

        if self.df[ts_col].dt.tz is not None:
            timezone_mismatch = self.df[ts_col].dt.tz.n != self.df[ts_col].dt.tz

        result = {
            'has_timestamp': True,
            'invalid_dates': bool(invalid),
            'duplicate_timestamps': bool(duplicates),
            'out_of_order_timestamps': bool(out_of_order),
            'timezone_consistency': not timezone_mismatch
        }
        self.feature_report['timestamp_validation'] = result
        return result

    def _detect_timestamp_column(self) -> Optional[str]:
        """Detect a timestamp or datetime-like column in the DataFrame."""
        normalized = {col.lower(): col for col in self.df.columns}
        for candidate in self.TIMESTAMP_CANDIDATES:
            if candidate.lower() in normalized:
                return normalized[candidate.lower()]

        for col in self.df.columns:
            lower = col.lower()
            if any(token in lower for token in ['date', 'time', 'timestamp', 'epoch', 'localtime', 'observation_time']):
                return col
        return None

    def _parse_timestamp_column(self, series: pd.Series, col_name: str) -> pd.Series:
        """Parse timestamp series robustly, handling epoch integer values."""
        if pd.api.types.is_numeric_dtype(series):
            max_val = series.max()
            if max_val > 1e12:
                return pd.to_datetime(series, unit='ms', errors='coerce')
            if max_val > 1e9:
                return pd.to_datetime(series, unit='s', errors='coerce')
        return pd.to_datetime(series, errors='coerce')

    def _meteorological_season(self, month: int) -> str:
        if month in [12, 1, 2]:
            return 'Winter'
        if month in [3, 4, 5]:
            return 'Spring'
        if month in [6, 7, 8]:
            return 'Summer'
        return 'Autumn'

    def _climate_season(self, month: int) -> str:
        if month in [12, 1, 2]:
            return 'Winter'
        if month in [3, 4, 5]:
            return 'Pre-monsoon'
        if month in [6, 7, 9]:
            return 'Monsoon'
        if month in [10, 11]:
            return 'Post-monsoon'
        return 'Unknown'

    def _monsoon_phase(self, month: int) -> str:
        if month in [3, 4, 5]:
            return 'Pre-monsoon'
        if month in [6, 7, 8, 9]:
            return 'Monsoon'
        if month in [10, 11]:
            return 'Post-monsoon'
        return 'Winter'

    def _relative_humidity_class(self, humidity: float) -> str:
        if humidity < 30:
            return 'Very Dry'
        if humidity < 50:
            return 'Dry'
        if humidity < 70:
            return 'Comfortable'
        if humidity < 90:
            return 'Humid'
        return 'Very Humid'

    def _calculate_vapor_pressure(self, temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate vapor pressure from temperature and humidity."""
        es = 6.112 * np.exp((17.67 * temp) / (temp + 243.5))
        return (humidity / 100.0) * es

    def _calculate_absolute_humidity(self, temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate absolute humidity (g/m³)."""
        vp = self._calculate_vapor_pressure(temp, humidity)
        return 216.7 * (vp / (temp + 273.15))

    def _calculate_wind_chill(self, temp: pd.Series, wind_speed: pd.Series) -> pd.Series:
        """Calculate wind chill (Celsius) for valid conditions."""
        valid = (temp <= 10) & (wind_speed > 4.8)
        wc = 13.12 + 0.6215 * temp - 11.37 * np.power(wind_speed, 0.16) + 0.3965 * temp * np.power(wind_speed, 0.16)
        return np.where(valid, wc, temp)

    def _calculate_apparent_temperature(
        self,
        temp: pd.Series,
        humidity: pd.Series,
        wind_speed: pd.Series
    ) -> pd.Series:
        """Calculate apparent temperature."""
        return temp + 0.33 * (humidity / 100.0) - 0.7 * wind_speed - 4.00

    def _calculate_wet_bulb_temperature(
        self,
        temp: pd.Series,
        humidity: pd.Series,
        pressure: Optional[pd.Series]
    ) -> pd.Series:
        """Estimate wet bulb temperature."""
        pressure = pressure if pressure is not None else 1013.25
        aw = 0.00066 * pressure
        gamma = np.log(humidity / 100.0) + (17.27 * temp) / (237.7 + temp)
        return (237.7 * gamma) / (17.27 - gamma) - aw * (temp - (237.7 * gamma) / (17.27 - gamma))
    
    def add_temperature_features(self, temp_col: str = 'temperature', humidity_col: str = 'humidity') -> pd.DataFrame:
        """
        Add temperature-related features
        
        Args:
            temp_col: Temperature column name
            humidity_col: Humidity column name
            
        Returns:
            DataFrame with temperature features added
        """
        # Normalize column names
        temp_col = self._find_column(['temperature', 'Temperature', 'temp'])
        humidity_col = self._find_column(['humidity', 'Humidity'])
        
        if temp_col:
            # Temperature category
            self.df['temp_category'] = pd.cut(
                self.df[temp_col],
                bins=[-np.inf, 10, 20, 30, 40, np.inf],
                labels=['Very Cold', 'Cold', 'Moderate', 'Hot', 'Very Hot']
            )
            
            # Temperature anomaly (deviation from mean)
            temp_mean = self.df[temp_col].mean()
            self.df['temp_anomaly'] = self.df[temp_col] - temp_mean
            
            # Rolling temperature average (7-day)
            if len(self.df) > 7:
                self.df['temp_rolling_7d'] = self.df[temp_col].rolling(window=7, min_periods=1).mean()
            
            # Temperature trend
            if len(self.df) > 3:
                self.df['temp_trend'] = self.df[temp_col].diff()
            
            # Heat Index (approximation)
            if humidity_col:
                self.df['heat_index'] = self._calculate_heat_index(self.df[temp_col], self.df[humidity_col])
                self.df['feels_like_diff'] = self.df['heat_index'] - self.df[temp_col]
            
            logger.info("Added temperature features")
            self.feature_report['temperature_features'] = True
        
        return self.df
    
    def add_rainfall_features(self, rain_col: str = 'precipitation') -> pd.DataFrame:
        """
        Add rainfall-related features
        
        Args:
            rain_col: Precipitation column name
            
        Returns:
            DataFrame with rainfall features added
        """
        rain_col = self._find_column(['precipitation', 'Precipitation', 'rain', 'Rain'])
        
        if rain_col:
            # Rain flag
            self.df['rain_flag'] = (self.df[rain_col] > 0).astype(int)
            
            # Rain category
            self.df['rain_category'] = pd.cut(
                self.df[rain_col],
                bins=[-np.inf, 0, 2.5, 10, 50, np.inf],
                labels=['No Rain', 'Light', 'Moderate', 'Heavy', 'Very Heavy']
            )
            
            # Heavy rain indicator (>10mm)
            self.df['heavy_rain'] = (self.df[rain_col] > 10).astype(int)
            
            # Rainfall intensity
            self.df['rainfall_intensity'] = np.where(
                self.df[rain_col] > 0,
                self.df[rain_col] / self.df[rain_col].sum(),
                0
            )
            
            # Rolling rainfall average (7-day)
            if len(self.df) > 7:
                self.df['rain_rolling_7d'] = self.df[rain_col].rolling(window=7, min_periods=1).mean()
            
            # Cumulative rainfall
            self.df['cumulative_rainfall'] = self.df[rain_col].cumsum()
            
            logger.info("Added rainfall features")
            self.feature_report['rainfall_features'] = True
        
        return self.df
    
    def add_wind_features(self, wind_col: str = 'wind_speed', gust_col: str = 'wind_gust') -> pd.DataFrame:
        """
        Add wind-related features
        
        Args:
            wind_col: Wind speed column name
            gust_col: Wind gust column name
            
        Returns:
            DataFrame with wind features added
        """
        wind_col = self._find_column(['wind_speed', 'Wind Speed', 'wind'])
        gust_col = self._find_column(['wind_gust', 'Wind Gust', 'gust'])
        
        if wind_col:
            # Wind category
            self.df['wind_category'] = pd.cut(
                self.df[wind_col],
                bins=[-np.inf, 5, 15, 25, 35, np.inf],
                labels=['Calm', 'Light', 'Moderate', 'Strong', 'Very Strong']
            )
            
            # Gust difference
            if gust_col:
                self.df['gust_difference'] = self.df[gust_col] - self.df[wind_col]
            
            # Storm indicator (>25 m/s)
            self.df['storm_indicator'] = (self.df[wind_col] > 25).astype(int)
            
            # Wind severity (Beaufort scale approximation)
            self.df['wind_severity'] = np.where(
                self.df[wind_col] < 1, 0,
                np.where(self.df[wind_col] < 6, 1,
                np.where(self.df[wind_col] < 12, 2,
                np.where(self.df[wind_col] < 20, 3,
                np.where(self.df[wind_col] < 29, 4, 5))))
            )
            
            logger.info("Added wind features")
            self.feature_report['wind_features'] = True
        
        return self.df
    
    def add_pressure_features(self, pressure_col: str = 'pressure') -> pd.DataFrame:
        """
        Add pressure-related features
        
        Args:
            pressure_col: Pressure column name
            
        Returns:
            DataFrame with pressure features added
        """
        pressure_col = self._find_column(['pressure', 'Pressure'])
        
        if pressure_col:
            # Pressure category
            self.df['pressure_category'] = pd.cut(
                self.df[pressure_col],
                bins=[-np.inf, 980, 1000, 1020, 1040, np.inf],
                labels=['Very Low', 'Low', 'Normal', 'High', 'Very High']
            )
            
            # Low pressure alert (<1000 hPa)
            self.df['low_pressure_alert'] = (self.df[pressure_col] < 1000).astype(int)
            
            # High pressure alert (>1030 hPa)
            self.df['high_pressure_alert'] = (self.df[pressure_col] > 1030).astype(int)
            
            logger.info("Added pressure features")
            self.feature_report['pressure_features'] = True
        
        return self.df
    
    def add_humidity_features(self, humidity_col: str = 'humidity', temp_col: str = 'temperature') -> pd.DataFrame:
        """
        Add humidity-related features
        
        Args:
            humidity_col: Humidity column name
            temp_col: Temperature column name
            
        Returns:
            DataFrame with humidity features added
        """
        humidity_col = self._find_column(['humidity', 'Humidity'])
        temp_col = self._find_column(['temperature', 'Temperature'])
        
        if humidity_col:
            # Humidity category
            self.df['humidity_category'] = pd.cut(
                self.df[humidity_col],
                bins=[-np.inf, 30, 50, 70, 90, np.inf],
                labels=['Very Dry', 'Dry', 'Comfortable', 'Humid', 'Very Humid']
            )
            
            # Comfort index
            if temp_col:
                self.df['comfort_index'] = self._calculate_comfort_index(
                    self.df[temp_col], self.df[humidity_col]
                )
            
            # Dew point approximation
            if temp_col:
                self.df['dew_point'] = self._calculate_dew_point(
                    self.df[temp_col], self.df[humidity_col]
                )
            
            logger.info("Added humidity features")
            self.feature_report['humidity_features'] = True
        
        return self.df
    
    def add_uv_features(self, uv_col: str = 'uv_index') -> pd.DataFrame:
        """
        Add UV-related features
        
        Args:
            uv_col: UV index column name
            
        Returns:
            DataFrame with UV features added
        """
        uv_col = self._find_column(['uv_index', 'UV Index', 'uv'])
        
        if uv_col:
            # UV category
            self.df['uv_category'] = pd.cut(
                self.df[uv_col],
                bins=[-np.inf, 3, 6, 8, 11, np.inf],
                labels=['Low', 'Moderate', 'High', 'Very High', 'Extreme']
            )
            
            # UV alert (>8)
            self.df['uv_alert'] = (self.df[uv_col] > 8).astype(int)
            
            logger.info("Added UV features")
            self.feature_report['uv_features'] = True
        
        return self.df
    
    def add_visibility_features(self, visibility_col: str = 'visibility') -> pd.DataFrame:
        """
        Add visibility-related features
        
        Args:
            visibility_col: Visibility column name
            
        Returns:
            DataFrame with visibility features added
        """
        visibility_col = self._find_column(['visibility', 'Visibility'])
        
        if visibility_col:
            # Fog indicator (<1 km)
            self.df['fog_indicator'] = (self.df[visibility_col] < 1).astype(int)
            
            # Low visibility alert (<5 km)
            self.df['low_visibility_alert'] = (self.df[visibility_col] < 5).astype(int)
            
            logger.info("Added visibility features")
            self.feature_report['visibility_features'] = True
        
        return self.df
    
    def add_cloud_features(self, cloud_col: str = 'cloud_cover') -> pd.DataFrame:
        """
        Add cloud-related features
        
        Args:
            cloud_col: Cloud cover column name
            
        Returns:
            DataFrame with cloud features added
        """
        cloud_col = self._find_column(['cloud_cover', 'Cloud Cover', 'clouds'])
        
        if cloud_col:
            # Cloud cover category
            self.df['cloud_category'] = pd.cut(
                self.df[cloud_col],
                bins=[-np.inf, 20, 40, 60, 80, np.inf],
                labels=['Clear', 'Few Clouds', 'Scattered', 'Broken', 'Overcast']
            )
            
            logger.info("Added cloud features")
            self.feature_report['cloud_features'] = True
        
        return self.df
    
    def add_air_quality_features(self, pm25_col: str = 'pm2_5', pm10_col: str = 'pm10') -> pd.DataFrame:
        """
        Add air quality-related features
        
        Args:
            pm25_col: PM2.5 column name
            pm10_col: PM10 column name
            
        Returns:
            DataFrame with air quality features added
        """
        pm25_col = self._find_column(['pm2_5', 'PM2.5', 'pm25'])
        pm10_col = self._find_column(['pm10', 'PM10'])
        
        if pm25_col:
            # AQI category (based on PM2.5)
            self.df['aqi_category'] = pd.cut(
                self.df[pm25_col],
                bins=[-np.inf, 12, 35, 55, 150, 250, np.inf],
                labels=['Good', 'Moderate', 'Unhealthy for Sensitive', 'Unhealthy', 
                        'Very Unhealthy', 'Hazardous']
            )
            
            # Pollution score
            self.df['pollution_score'] = self.df[pm25_col] / 50  # Normalized
            
            # PM2.5 / PM10 ratio
            if pm10_col:
                self.df['pm_ratio'] = self.df[pm25_col] / (self.df[pm10_col] + 0.1)
            
            # Pollution severity
            self.df['pollution_severity'] = pd.cut(
                self.df[pm25_col],
                bins=[-np.inf, 35, 55, 150, np.inf],
                labels=['Low', 'Moderate', 'High', 'Extreme']
            )
            
            # Air quality alert
            self.df['air_quality_alert'] = (self.df[pm25_col] > 55).astype(int)
            
            logger.info("Added air quality features")
            self.feature_report['air_quality_features'] = True
        
        return self.df
    
    def add_geographic_features(self, lat_col: str = 'latitude', lon_col: str = 'longitude') -> pd.DataFrame:
        """
        Add geographic-related features
        
        Args:
            lat_col: Latitude column name
            lon_col: Longitude column name
            
        Returns:
            DataFrame with geographic features added
        """
        lat_col = self._find_column(['latitude', 'Latitude', 'lat'])
        lon_col = self._find_column(['longitude', 'Longitude', 'lon'])
        
        if lat_col and lon_col:
            # Latitude bands
            self.df['latitude_band'] = pd.cut(
                self.df[lat_col],
                bins=[-np.inf, -30, 0, 30, 60, np.inf],
                labels=['Southern Polar', 'Southern Temperate', 'Tropical', 
                        'Northern Temperate', 'Northern Polar']
            )
            
            # Longitude bands
            self.df['longitude_band'] = pd.cut(
                self.df[lon_col],
                bins=[-np.inf, -60, 0, 60, 120, 180, np.inf],
                labels=['Western', 'Central Western', 'Central', 
                        'Central Eastern', 'Eastern', 'Far Eastern']
            )
            
            # Climate zone (simplified)
            self.df['climate_zone'] = pd.cut(
                self.df[lat_col].abs(),
                bins=[-np.inf, 23.5, 35, 66.5, np.inf],
                labels=['Tropical', 'Subtropical', 'Temperate', 'Polar']
            )
            
            logger.info("Added geographic features")
            self.feature_report['geographic_features'] = True
        
        return self.df
    
    def _find_column(self, possible_names: List[str]) -> Optional[str]:
        """Find a column by possible names"""
        for name in possible_names:
            if name in self.df.columns:
                return name
        return None
    
    def _calculate_heat_index(self, temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate heat index (Rothfusz regression)"""
        # Simplified heat index calculation
        T = temp * 9/5 + 32  # Convert to Fahrenheit
        RH = humidity
        
        HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))
        
        # More complex calculation for higher temperatures
        mask = T >= 80
        if mask.any():
            T_high = T[mask]
            RH_high = RH[mask]
            
            HI_high = (-42.379 + 2.04901523 * T_high + 10.14333127 * RH_high
                      - 0.22475541 * T_high * RH_high - 0.00683783 * T_high * T_high
                      - 0.05481717 * RH_high * RH_high + 0.00122874 * T_high * T_high * RH_high
                      + 0.00085282 * T_high * RH_high * RH_high
                      - 0.00000199 * T_high * T_high * RH_high * RH_high)
            
            HI[mask] = HI_high
        
        return (HI - 32) * 5/9  # Convert back to Celsius
    
    def _calculate_comfort_index(self, temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate comfort index (simplified)"""
        # Comfort zone: 20-26°C, 40-60% humidity
        temp_comfort = ((temp >= 20) & (temp <= 26)).astype(int)
        humidity_comfort = ((humidity >= 40) & (humidity <= 60)).astype(int)
        return (temp_comfort + humidity_comfort) / 2
    
    def _calculate_dew_point(self, temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate dew point (Magnus formula)"""
        a = 17.27
        b = 237.7
        alpha = ((a * temp) / (b + temp)) + np.log(humidity / 100.0)
        return (b * alpha) / (a - alpha)
    
    def get_feature_report(self) -> Dict:
        """Get the feature engineering report"""
        return self.feature_report
    
    def get_engineered_data(self) -> pd.DataFrame:
        """Get the engineered DataFrame"""
        return self.df


def main():
    """Test the feature engineer"""
    # Create sample data
    data = {
        'temperature': [25, 30, 35, 40, 28, 32, 38],
        'humidity': [60, 70, 80, 50, 65, 55, 75],
        'pressure': [1013, 1015, 1010, 1008, 1012, 1014, 1011],
        'wind_speed': [5, 10, 15, 20, 8, 12, 18],
        'precipitation': [0, 5, 15, 0, 2, 0, 25],
        'uv_index': [5, 7, 9, 10, 6, 8, 11],
        'visibility': [10, 8, 5, 12, 9, 11, 4],
        'cloud_cover': [30, 50, 80, 20, 40, 35, 90],
        'latitude': [28.6] * 7,
        'longitude': [77.2] * 7
    }
    df = pd.DataFrame(data)
    
    engineer = WeatherFeatureEngineer(df)
    engineer.add_temperature_features()
    engineer.add_rainfall_features()
    engineer.add_wind_features()
    engineer.add_humidity_features()
    engineer.add_uv_features()
    engineer.add_visibility_features()
    engineer.add_cloud_features()
    engineer.add_geographic_features()
    
    print("Feature Report:")
    print(engineer.get_feature_report())
    print("\nEngineered Columns:")
    print(list(engineer.get_engineered_data().columns))


if __name__ == "__main__":
    main()

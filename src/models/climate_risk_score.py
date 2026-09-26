"""
Climate Risk Score Module
Calculates comprehensive climate risk score based on multiple factors
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClimateRiskScorer:
    """Calculate climate risk score based on weather parameters"""
    
    def __init__(self):
        """Initialize the climate risk scorer"""
        self.risk_weights = {
            'temperature': 0.25,
            'humidity': 0.15,
            'rainfall': 0.15,
            'air_quality': 0.20,
            'uv_index': 0.10,
            'wind_speed': 0.10,
            'pressure': 0.05
        }
        self.risk_report = {}
    
    def calculate_temperature_risk(self, temperature: float) -> float:
        """
        Calculate temperature risk score (0-100)
        
        Args:
            temperature: Temperature in Celsius
            
        Returns:
            Risk score (0-100)
        """
        if temperature < 10:
            # Cold risk
            risk = (10 - temperature) / 10 * 30
        elif temperature <= 30:
            # Normal
            risk = 0
        elif temperature <= 35:
            # Hot
            risk = (temperature - 30) / 5 * 50
        elif temperature <= 40:
            # Very hot
            risk = 50 + (temperature - 35) / 5 * 30
        else:
            # Extreme heat
            risk = 80 + (temperature - 40) / 10 * 20
        
        return min(max(risk, 0), 100)
    
    def calculate_humidity_risk(self, humidity: float) -> float:
        """
        Calculate humidity risk score (0-100)
        
        Args:
            humidity: Humidity percentage
            
        Returns:
            Risk score (0-100)
        """
        if humidity < 30:
            # Too dry
            risk = (30 - humidity) / 30 * 40
        elif humidity <= 60:
            # Comfortable
            risk = 0
        elif humidity <= 80:
            # Humid
            risk = (humidity - 60) / 20 * 50
        else:
            # Very humid
            risk = 50 + (humidity - 80) / 20 * 50
        
        return min(max(risk, 0), 100)
    
    def calculate_rainfall_risk(self, rainfall: float) -> float:
        """
        Calculate rainfall risk score (0-100)
        
        Args:
            rainfall: Rainfall in mm
            
        Returns:
            Risk score (0-100)
        """
        if rainfall <= 2.5:
            # Light rain - low risk
            risk = rainfall / 2.5 * 20
        elif rainfall <= 10:
            # Moderate rain
            risk = 20 + (rainfall - 2.5) / 7.5 * 30
        elif rainfall <= 50:
            # Heavy rain
            risk = 50 + (rainfall - 10) / 40 * 30
        else:
            # Very heavy rain
            risk = 80 + (rainfall - 50) / 50 * 20
        
        return min(max(risk, 0), 100)
    
    def calculate_air_quality_risk(self, pm25: float) -> float:
        """
        Calculate air quality risk score (0-100)
        
        Args:
            pm25: PM2.5 concentration in µg/m³
            
        Returns:
            Risk score (0-100)
        """
        if pm25 <= 12:
            # Good
            risk = 0
        elif pm25 <= 35:
            # Moderate
            risk = (pm25 - 12) / 23 * 30
        elif pm25 <= 55:
            # Unhealthy for sensitive
            risk = 30 + (pm25 - 35) / 20 * 40
        elif pm25 <= 150:
            # Unhealthy
            risk = 70 + (pm25 - 55) / 95 * 20
        else:
            # Very unhealthy/hazardous
            risk = 90 + (pm25 - 150) / 100 * 10
        
        return min(max(risk, 0), 100)
    
    def calculate_uv_risk(self, uv_index: float) -> float:
        """
        Calculate UV risk score (0-100)
        
        Args:
            uv_index: UV Index
            
        Returns:
            Risk score (0-100)
        """
        if uv_index <= 3:
            # Low
            risk = uv_index / 3 * 10
        elif uv_index <= 6:
            # Moderate
            risk = 10 + (uv_index - 3) / 3 * 20
        elif uv_index <= 8:
            # High
            risk = 30 + (uv_index - 6) / 2 * 30
        elif uv_index <= 11:
            # Very high
            risk = 60 + (uv_index - 8) / 3 * 30
        else:
            # Extreme
            risk = 90 + (uv_index - 11) / 5 * 10
        
        return min(max(risk, 0), 100)
    
    def calculate_wind_risk(self, wind_speed: float) -> float:
        """
        Calculate wind risk score (0-100)
        
        Args:
            wind_speed: Wind speed in m/s
            
        Returns:
            Risk score (0-100)
        """
        if wind_speed <= 5:
            # Calm to light
            risk = wind_speed / 5 * 10
        elif wind_speed <= 15:
            # Moderate
            risk = 10 + (wind_speed - 5) / 10 * 30
        elif wind_speed <= 25:
            # Strong
            risk = 40 + (wind_speed - 15) / 10 * 40
        else:
            # Very strong
            risk = 80 + (wind_speed - 25) / 25 * 20
        
        return min(max(risk, 0), 100)
    
    def calculate_pressure_risk(self, pressure: float) -> float:
        """
        Calculate pressure risk score (0-100)
        
        Args:
            pressure: Pressure in hPa
            
        Returns:
            Risk score (0-100)
        """
        # Low pressure risk (storms)
        if pressure < 980:
            risk = (980 - pressure) / 20 * 100
        elif pressure < 1000:
            risk = (1000 - pressure) / 20 * 30
        elif pressure <= 1030:
            # Normal
            risk = 0
        else:
            # High pressure (less common risk)
            risk = (pressure - 1030) / 20 * 20
        
        return min(max(risk, 0), 100)
    
    def calculate_overall_risk(
        self,
        temperature: float = None,
        humidity: float = None,
        rainfall: float = None,
        pm25: float = None,
        uv_index: float = None,
        wind_speed: float = None,
        pressure: float = None
    ) -> Dict:
        """
        Calculate overall climate risk score
        
        Args:
            temperature: Temperature in Celsius
            humidity: Humidity percentage
            rainfall: Rainfall in mm
            pm25: PM2.5 concentration in µg/m³
            uv_index: UV Index
            wind_speed: Wind speed in m/s
            pressure: Pressure in hPa
            
        Returns:
            Dictionary with risk scores and category
        """
        component_scores = {}
        total_weight = 0
        weighted_score = 0
        
        # Calculate each component risk
        if temperature is not None:
            temp_risk = self.calculate_temperature_risk(temperature)
            component_scores['temperature'] = temp_risk
            weighted_score += temp_risk * self.risk_weights['temperature']
            total_weight += self.risk_weights['temperature']
        
        if humidity is not None:
            humid_risk = self.calculate_humidity_risk(humidity)
            component_scores['humidity'] = humid_risk
            weighted_score += humid_risk * self.risk_weights['humidity']
            total_weight += self.risk_weights['humidity']
        
        if rainfall is not None:
            rain_risk = self.calculate_rainfall_risk(rainfall)
            component_scores['rainfall'] = rain_risk
            weighted_score += rain_risk * self.risk_weights['rainfall']
            total_weight += self.risk_weights['rainfall']
        
        if pm25 is not None:
            aqi_risk = self.calculate_air_quality_risk(pm25)
            component_scores['air_quality'] = aqi_risk
            weighted_score += aqi_risk * self.risk_weights['air_quality']
            total_weight += self.risk_weights['air_quality']
        
        if uv_index is not None:
            uv_risk = self.calculate_uv_risk(uv_index)
            component_scores['uv_index'] = uv_risk
            weighted_score += uv_risk * self.risk_weights['uv_index']
            total_weight += self.risk_weights['uv_index']
        
        if wind_speed is not None:
            wind_risk = self.calculate_wind_risk(wind_speed)
            component_scores['wind_speed'] = wind_risk
            weighted_score += wind_risk * self.risk_weights['wind_speed']
            total_weight += self.risk_weights['wind_speed']
        
        if pressure is not None:
            press_risk = self.calculate_pressure_risk(pressure)
            component_scores['pressure'] = press_risk
            weighted_score += press_risk * self.risk_weights['pressure']
            total_weight += self.risk_weights['pressure']
        
        # Calculate overall score
        overall_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine risk category
        if overall_score <= 25:
            category = "Low"
            color = "green"
        elif overall_score <= 50:
            category = "Moderate"
            color = "yellow"
        elif overall_score <= 75:
            category = "High"
            color = "orange"
        else:
            category = "Extreme"
            color = "red"
        
        result = {
            'overall_score': round(overall_score, 2),
            'category': category,
            'color': color,
            'component_scores': component_scores,
            'weights_used': total_weight
        }
        
        self.risk_report = result
        logger.info(f"Climate risk calculated: {overall_score:.2f} ({category})")
        return result
    
    def calculate_batch_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk scores for a batch of records using vectorized operations
        
        Args:
            df: DataFrame with weather parameters
            
        Returns:
            DataFrame with risk scores added
        """
        # Extract columns as series for vectorized operations
        temperature = df.get('temperature')
        humidity = df.get('humidity')
        rainfall = df.get('precipitation')
        pm25 = df.get('pm2_5')
        uv_index = df.get('uv_index')
        wind_speed = df.get('wind_speed')
        pressure = df.get('pressure')
        
        # Vectorized risk calculations
        temp_risk = self._vectorized_temperature_risk(temperature)
        humid_risk = self._vectorized_humidity_risk(humidity)
        rain_risk = self._vectorized_rainfall_risk(rainfall)
        aqi_risk = self._vectorized_air_quality_risk(pm25)
        uv_risk = self._vectorized_uv_risk(uv_index)
        wind_risk = self._vectorized_wind_risk(wind_speed)
        press_risk = self._vectorized_pressure_risk(pressure)
        
        # Calculate weighted scores
        weighted_score = (
            temp_risk * self.risk_weights['temperature'] +
            humid_risk * self.risk_weights['humidity'] +
            rain_risk * self.risk_weights['rainfall'] +
            aqi_risk * self.risk_weights['air_quality'] +
            uv_risk * self.risk_weights['uv_index'] +
            wind_risk * self.risk_weights['wind_speed'] +
            press_risk * self.risk_weights['pressure']
        )
        
        # Calculate total weight (only for non-null columns)
        total_weight = 0
        if temperature is not None:
            total_weight += self.risk_weights['temperature']
        if humidity is not None:
            total_weight += self.risk_weights['humidity']
        if rainfall is not None:
            total_weight += self.risk_weights['rainfall']
        if pm25 is not None:
            total_weight += self.risk_weights['air_quality']
        if uv_index is not None:
            total_weight += self.risk_weights['uv_index']
        if wind_speed is not None:
            total_weight += self.risk_weights['wind_speed']
        if pressure is not None:
            total_weight += self.risk_weights['pressure']
        
        # Calculate overall score
        overall_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine risk category
        category = pd.cut(overall_score, 
                         bins=[-np.inf, 25, 50, 75, np.inf],
                         labels=['Low', 'Moderate', 'High', 'Extreme'])
        color = pd.cut(overall_score,
                      bins=[-np.inf, 25, 50, 75, np.inf],
                      labels=['green', 'yellow', 'orange', 'red'])
        
        # Add results to dataframe
        result_df = df.copy()
        result_df['overall_score'] = overall_score.round(2)
        result_df['category'] = category
        result_df['color'] = color
        result_df['temperature_risk'] = temp_risk
        result_df['humidity_risk'] = humid_risk
        result_df['rainfall_risk'] = rain_risk
        result_df['air_quality_risk'] = aqi_risk
        result_df['uv_index_risk'] = uv_risk
        result_df['wind_speed_risk'] = wind_risk
        result_df['pressure_risk'] = press_risk
        
        logger.info(f"Batch risk scores calculated for {len(df)} records")
        return result_df
    
    def _vectorized_temperature_risk(self, temperature: pd.Series) -> pd.Series:
        """Vectorized temperature risk calculation"""
        if temperature is None:
            return pd.Series(0.0)
        temp = pd.to_numeric(temperature, errors='coerce').fillna(0)
        risk = pd.Series(0.0, index=temp.index)
        
        # Cold risk
        mask_cold = temp < 10
        risk[mask_cold] = (10 - temp[mask_cold]) / 10 * 30
        
        # Normal
        mask_normal = (temp >= 10) & (temp <= 30)
        risk[mask_normal] = 0
        
        # Hot
        mask_hot = (temp > 30) & (temp <= 35)
        risk[mask_hot] = (temp[mask_hot] - 30) / 5 * 50
        
        # Very hot
        mask_very_hot = (temp > 35) & (temp <= 40)
        risk[mask_very_hot] = 50 + (temp[mask_very_hot] - 35) / 5 * 30
        
        # Extreme heat
        mask_extreme = temp > 40
        risk[mask_extreme] = 80 + (temp[mask_extreme] - 40) / 10 * 20
        
        return risk.clip(0, 100)
    
    def _vectorized_humidity_risk(self, humidity: pd.Series) -> pd.Series:
        """Vectorized humidity risk calculation"""
        if humidity is None:
            return pd.Series(0.0)
        humid = pd.to_numeric(humidity, errors='coerce').fillna(0)
        risk = pd.Series(0.0, index=humid.index)
        
        # Too dry
        mask_dry = humid < 30
        risk[mask_dry] = (30 - humid[mask_dry]) / 30 * 40
        
        # Comfortable
        mask_comfort = (humid >= 30) & (humid <= 60)
        risk[mask_comfort] = 0
        
        # Humid
        mask_humid = (humid > 60) & (humid <= 80)
        risk[mask_humid] = (humid[mask_humid] - 60) / 20 * 50
        
        # Very humid
        mask_very_humid = humid > 80
        risk[mask_very_humid] = 50 + (humid[mask_very_humid] - 80) / 20 * 50
        
        return risk.clip(0, 100)
    
    def _vectorized_rainfall_risk(self, rainfall: pd.Series) -> pd.Series:
        """Vectorized rainfall risk calculation"""
        if rainfall is None:
            return pd.Series(0.0)
        rain = pd.to_numeric(rainfall, errors='coerce').fillna(0)
        risk = pd.Series(0.0, index=rain.index)
        
        # Light rain
        mask_light = rain <= 2.5
        risk[mask_light] = rain[mask_light] / 2.5 * 20
        
        # Moderate rain
        mask_moderate = (rain > 2.5) & (rain <= 10)
        risk[mask_moderate] = 20 + (rain[mask_moderate] - 2.5) / 7.5 * 30
        
        # Heavy rain
        mask_heavy = (rain > 10) & (rain <= 50)
        risk[mask_heavy] = 50 + (rain[mask_heavy] - 10) / 40 * 30
        
        # Very heavy rain
        mask_very_heavy = rain > 50
        risk[mask_very_heavy] = 80 + (rain[mask_very_heavy] - 50) / 50 * 20
        
        return risk.clip(0, 100)
    
    def _vectorized_air_quality_risk(self, pm25: pd.Series) -> pd.Series:
        """Vectorized air quality risk calculation"""
        if pm25 is None:
            return pd.Series(0.0)
        aqi = pd.to_numeric(pm25, errors='coerce').fillna(0)
        risk = pd.Series(0.0, index=aqi.index)
        
        # Good
        mask_good = aqi <= 12
        risk[mask_good] = 0
        
        # Moderate
        mask_moderate = (aqi > 12) & (aqi <= 35)
        risk[mask_moderate] = (aqi[mask_moderate] - 12) / 23 * 30
        
        # Unhealthy for sensitive
        mask_sensitive = (aqi > 35) & (aqi <= 55)
        risk[mask_sensitive] = 30 + (aqi[mask_sensitive] - 35) / 20 * 40
        
        # Unhealthy
        mask_unhealthy = (aqi > 55) & (aqi <= 150)
        risk[mask_unhealthy] = 70 + (aqi[mask_unhealthy] - 55) / 95 * 20
        
        # Very unhealthy/hazardous
        mask_hazardous = aqi > 150
        risk[mask_hazardous] = 90 + (aqi[mask_hazardous] - 150) / 100 * 10
        
        return risk.clip(0, 100)
    
    def _vectorized_uv_risk(self, uv_index: pd.Series) -> pd.Series:
        """Vectorized UV risk calculation"""
        if uv_index is None:
            return pd.Series(0.0)
        uv = pd.to_numeric(uv_index, errors='coerce').fillna(0)
        risk = pd.Series(0.0, index=uv.index)
        
        # Low
        mask_low = uv <= 3
        risk[mask_low] = uv[mask_low] / 3 * 10
        
        # Moderate
        mask_moderate = (uv > 3) & (uv <= 6)
        risk[mask_moderate] = 10 + (uv[mask_moderate] - 3) / 3 * 20
        
        # High
        mask_high = (uv > 6) & (uv <= 8)
        risk[mask_high] = 30 + (uv[mask_high] - 6) / 2 * 30
        
        # Very high
        mask_very_high = (uv > 8) & (uv <= 11)
        risk[mask_very_high] = 60 + (uv[mask_very_high] - 8) / 3 * 30
        
        # Extreme
        mask_extreme = uv > 11
        risk[mask_extreme] = 90 + (uv[mask_extreme] - 11) / 5 * 10
        
        return risk.clip(0, 100)
    
    def _vectorized_wind_risk(self, wind_speed: pd.Series) -> pd.Series:
        """Vectorized wind risk calculation"""
        if wind_speed is None:
            return pd.Series(0.0)
        wind = pd.to_numeric(wind_speed, errors='coerce').fillna(0)
        risk = pd.Series(0.0, index=wind.index)
        
        # Calm to light
        mask_light = wind <= 5
        risk[mask_light] = wind[mask_light] / 5 * 10
        
        # Moderate
        mask_moderate = (wind > 5) & (wind <= 15)
        risk[mask_moderate] = 10 + (wind[mask_moderate] - 5) / 10 * 30
        
        # Strong
        mask_strong = (wind > 15) & (wind <= 25)
        risk[mask_strong] = 40 + (wind[mask_strong] - 15) / 10 * 40
        
        # Very strong
        mask_very_strong = wind > 25
        risk[mask_very_strong] = 80 + (wind[mask_very_strong] - 25) / 25 * 20
        
        return risk.clip(0, 100)
    
    def _vectorized_pressure_risk(self, pressure: pd.Series) -> pd.Series:
        """Vectorized pressure risk calculation"""
        if pressure is None:
            return pd.Series(0.0)
        press = pd.to_numeric(pressure, errors='coerce').fillna(1013)
        risk = pd.Series(0.0, index=press.index)
        
        # Low pressure (storms)
        mask_low = press < 980
        risk[mask_low] = (980 - press[mask_low]) / 20 * 100
        
        # Below normal
        mask_below = (press >= 980) & (press < 1000)
        risk[mask_below] = (1000 - press[mask_below]) / 20 * 30
        
        # Normal
        mask_normal = (press >= 1000) & (press <= 1030)
        risk[mask_normal] = 0
        
        # High pressure
        mask_high = press > 1030
        risk[mask_high] = (press[mask_high] - 1030) / 20 * 20
        
        return risk.clip(0, 100)
    
    def get_risk_report(self) -> Dict:
        """Get the last risk calculation report"""
        return self.risk_report
    
    def print_report(self):
        """Print a formatted risk report"""
        report = self.risk_report
        
        print("=" * 80)
        print("CLIMATE RISK SCORE REPORT")
        print("=" * 80)
        
        print(f"\nOverall Risk Score: {report['overall_score']}/100")
        print(f"Risk Category: {report['category']}")
        print(f"Risk Level: {report['color']}")
        
        print("\nComponent Scores:")
        for component, score in report['component_scores'].items():
            weight = self.risk_weights.get(component, 0)
            print(f"  {component}: {score:.2f} (weight: {weight:.2f})")
        
        print("\n" + "=" * 80)


def main():
    """Test the climate risk scorer"""
    scorer = ClimateRiskScorer()
    
    # Test with various conditions
    print("Test 1: Normal conditions")
    risk = scorer.calculate_overall_risk(
        temperature=25, humidity=60, rainfall=0,
        pm25=10, uv_index=5, wind_speed=5, pressure=1013
    )
    scorer.print_report()
    
    print("\nTest 2: Extreme heat")
    risk = scorer.calculate_overall_risk(
        temperature=42, humidity=40, rainfall=0,
        pm25=15, uv_index=10, wind_speed=3, pressure=1010
    )
    scorer.print_report()
    
    print("\nTest 3: Heavy rain and poor air quality")
    risk = scorer.calculate_overall_risk(
        temperature=28, humidity=85, rainfall=60,
        pm25=80, uv_index=3, wind_speed=15, pressure=995
    )
    scorer.print_report()


if __name__ == "__main__":
    main()

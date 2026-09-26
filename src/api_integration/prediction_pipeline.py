"""
Real-Time Prediction Pipeline
Integrates API data, feature engineering, and ML models for real-time predictions
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Optional, Tuple
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api_integration.openmeteo import OpenMeteoAPI, OpenMeteoAirQualityAPI
from src.api_integration.geocoding import GeocodingAPI
from src.feature_engineering.weather_features import WeatherFeatureEngineer
from src.models.climate_risk_score import ClimateRiskScorer
from src.models.rainfall_prediction import RainfallPredictor
from src.models.heatwave_prediction import HeatwavePredictor
from src.models.anomaly_detection import ClimateAnomalyDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimePredictionPipeline:
    """End-to-end real-time prediction pipeline"""
    
    def __init__(self, model_dir: str = 'trained_models'):
        """
        Initialize the prediction pipeline
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = model_dir
        self.weather_api = OpenMeteoAPI()
        self.aqi_api = OpenMeteoAirQualityAPI()
        self.geocoding_api = GeocodingAPI()
        self.risk_scorer = ClimateRiskScorer()
        
        # Load trained models if available
        self.rainfall_model = None
        self.heatwave_model = None
        
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models"""
        try:
            rainfall_path = os.path.join(self.model_dir, 'rainfall_model.pkl')
            if os.path.exists(rainfall_path):
                self.rainfall_model = joblib.load(rainfall_path)
                logger.info("Rainfall model loaded")
        except Exception as e:
            logger.warning(f"Could not load rainfall model: {e}")
        
        try:
            heatwave_path = os.path.join(self.model_dir, 'heatwave_model.pkl')
            if os.path.exists(heatwave_path):
                self.heatwave_model = joblib.load(heatwave_path)
                logger.info("Heatwave model loaded")
        except Exception as e:
            logger.warning(f"Could not load heatwave model: {e}")
    
    def get_coordinates(self, city: str, country: str = None) -> Optional[Dict]:
        """
        Get coordinates for a city
        
        Args:
            city: City name
            country: Country name (optional)
            
        Returns:
            Dictionary with coordinates
        """
        return self.geocoding_api.get_coordinates(city, country)
    
    def fetch_weather_data(self, latitude: float, longitude: float) -> Dict:
        """
        Fetch current weather data from Open-Meteo
        
        Args:
            latitude: Latitude
            longitude: Longitude
            
        Returns:
            Dictionary with weather data
        """
        weather_data = self.weather_api.get_current_weather(latitude, longitude)
        return weather_data
    
    def fetch_air_quality(self, latitude: float, longitude: float) -> Dict:
        """
        Fetch air quality data from Open-Meteo
        
        Args:
            latitude: Latitude
            longitude: Longitude
            
        Returns:
            Dictionary with air quality data
        """
        aqi_data = self.aqi_api.get_air_quality(latitude, longitude)
        return aqi_data
    
    def prepare_features(self, weather_data: Dict, aqi_data: Dict) -> pd.DataFrame:
        """
        Prepare features from API data
        
        Args:
            weather_data: Weather data from API
            aqi_data: Air quality data from API
            
        Returns:
            DataFrame with engineered features
        """
        # Extract current weather
        current = weather_data.get('current', {})
        current_aqi = aqi_data.get('current', {})
        
        # Create feature dictionary
        features = {
            'temperature': current.get('temperature_2m'),
            'humidity': current.get('relative_humidity_2m'),
            'pressure': current.get('pressure_msl'),
            'wind_speed': current.get('wind_speed_10m'),
            'wind_direction': current.get('wind_direction_10m'),
            'wind_gust': current.get('wind_gusts_10m'),
            'precipitation': current.get('precipitation'),
            'cloud_cover': current.get('cloud_cover'),
            'visibility': current.get('visibility'),
            'uv_index': current.get('uv_index'),
            'pm2_5': current_aqi.get('pm2_5'),
            'pm10': current_aqi.get('pm10'),
            'carbon_monoxide': current_aqi.get('carbon_monoxide'),
            'nitrogen_dioxide': current_aqi.get('nitrogen_dioxide'),
            'sulphur_dioxide': current_aqi.get('sulphur_dioxide'),
            'ozone': current_aqi.get('ozone'),
            'latitude': latitude if 'latitude' in locals() else None,
            'longitude': longitude if 'longitude' in locals() else None
        }
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Apply feature engineering
        engineer = WeatherFeatureEngineer(df)
        engineer.add_temperature_features()
        engineer.add_rainfall_features()
        engineer.add_wind_features()
        engineer.add_pressure_features()
        engineer.add_humidity_features()
        engineer.add_uv_features()
        engineer.add_visibility_features()
        engineer.add_cloud_features()
        engineer.add_air_quality_features()
        
        return engineer.get_engineered_data()
    
    def calculate_climate_risk(self, weather_data: Dict, aqi_data: Dict) -> Dict:
        """
        Calculate climate risk score
        
        Args:
            weather_data: Weather data
            aqi_data: Air quality data
            
        Returns:
            Dictionary with risk score
        """
        current = weather_data.get('current', {})
        current_aqi = aqi_data.get('current', {})
        
        risk = self.risk_scorer.calculate_overall_risk(
            temperature=current.get('temperature_2m'),
            humidity=current.get('relative_humidity_2m'),
            rainfall=current.get('precipitation'),
            pm25=current_aqi.get('pm2_5'),
            uv_index=current.get('uv_index'),
            wind_speed=current.get('wind_speed_10m'),
            pressure=current.get('pressure_msl')
        )
        
        return risk
    
    def predict_rainfall(self, features_df: pd.DataFrame) -> Optional[Dict]:
        """
        Predict rainfall using trained model
        
        Args:
            features_df: Engineered features
            
        Returns:
            Dictionary with rainfall prediction
        """
        if self.rainfall_model is None:
            logger.warning("Rainfall model not available")
            return None
        
        try:
            # Select relevant features (adjust based on model training)
            feature_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'cloud_cover']
            X = features_df[feature_cols].fillna(0)
            
            prediction, probability = self.rainfall_model.predict(X)
            
            return {
                'rain_prediction': int(prediction[0]),
                'rain_probability': float(probability[0]),
                'will_rain': bool(prediction[0])
            }
        except Exception as e:
            logger.error(f"Rainfall prediction error: {e}")
            return None
    
    def predict_heatwave(self, features_df: pd.DataFrame) -> Optional[Dict]:
        """
        Predict heatwave using trained model
        
        Args:
            features_df: Engineered features
            
        Returns:
            Dictionary with heatwave prediction
        """
        if self.heatwave_model is None:
            logger.warning("Heatwave model not available")
            return None
        
        try:
            # Select relevant features
            feature_cols = ['temperature', 'humidity', 'uv_index', 'heat_index']
            X = features_df[feature_cols].fillna(0)
            
            prediction, probability = self.heatwave_model.predict(X)
            
            return {
                'heatwave_prediction': int(prediction[0]),
                'heatwave_probability': float(probability[0]),
                'is_heatwave': bool(prediction[0])
            }
        except Exception as e:
            logger.error(f"Heatwave prediction error: {e}")
            return None
    
    def run_complete_pipeline(self, city: str, country: str = None) -> Dict:
        """
        Run the complete prediction pipeline for a city
        
        Args:
            city: City name
            country: Country name (optional)
            
        Returns:
            Dictionary with all predictions and analysis
        """
        logger.info(f"Running pipeline for {city}, {country}")
        
        # Step 1: Get coordinates
        coords = self.get_coordinates(city, country)
        if not coords:
            return {'error': 'Could not find coordinates for city'}
        
        lat, lon = coords['latitude'], coords['longitude']
        
        # Step 2: Fetch weather data
        weather_data = self.fetch_weather_data(lat, lon)
        if not weather_data:
            return {'error': 'Could not fetch weather data'}
        
        # Step 3: Fetch air quality data
        aqi_data = self.fetch_air_quality(lat, lon)
        if not aqi_data:
            aqi_data = {'current': {}}
        
        # Step 4: Prepare features
        features_df = self.prepare_features(weather_data, aqi_data)
        
        # Step 5: Calculate climate risk
        risk_score = self.calculate_climate_risk(weather_data, aqi_data)
        
        # Step 6: Predict rainfall
        rainfall_pred = self.predict_rainfall(features_df)
        
        # Step 7: Predict heatwave
        heatwave_pred = self.predict_heatwave(features_df)
        
        # Compile results
        results = {
            'location': {
                'city': coords['name'],
                'country': coords['country'],
                'latitude': lat,
                'longitude': lon
            },
            'current_weather': weather_data.get('current', {}),
            'air_quality': aqi_data.get('current', {}),
            'climate_risk': risk_score,
            'rainfall_prediction': rainfall_pred,
            'heatwave_prediction': heatwave_pred,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"Pipeline completed for {city}")
        return results
    
    def run_pipeline_by_coordinates(self, latitude: float, longitude: float) -> Dict:
        """
        Run pipeline using coordinates directly
        
        Args:
            latitude: Latitude
            longitude: Longitude
            
        Returns:
            Dictionary with all predictions and analysis
        """
        logger.info(f"Running pipeline for coordinates: {latitude}, {longitude}")
        
        # Fetch weather data
        weather_data = self.fetch_weather_data(latitude, longitude)
        if not weather_data:
            return {'error': 'Could not fetch weather data'}
        
        # Fetch air quality data
        aqi_data = self.fetch_air_quality(latitude, longitude)
        if not aqi_data:
            aqi_data = {'current': {}}
        
        # Prepare features
        features_df = self.prepare_features(weather_data, aqi_data)
        
        # Calculate climate risk
        risk_score = self.calculate_climate_risk(weather_data, aqi_data)
        
        # Predict rainfall
        rainfall_pred = self.predict_rainfall(features_df)
        
        # Predict heatwave
        heatwave_pred = self.predict_heatwave(features_df)
        
        # Compile results
        results = {
            'location': {
                'latitude': latitude,
                'longitude': longitude
            },
            'current_weather': weather_data.get('current', {}),
            'air_quality': aqi_data.get('current', {}),
            'climate_risk': risk_score,
            'rainfall_prediction': rainfall_pred,
            'heatwave_prediction': heatwave_pred,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"Pipeline completed for coordinates")
        return results
    
    def generate_recommendations(self, results: Dict) -> List[str]:
        """
        Generate weather recommendations based on predictions
        
        Args:
            results: Pipeline results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Weather-based recommendations
        current = results.get('current_weather', {})
        temp = current.get('temperature_2m', 25)
        precip = current.get('precipitation', 0)
        wind = current.get('wind_speed_10m', 0)
        
        if temp > 35:
            recommendations.append("🔥 Extreme heat warning: Stay hydrated, avoid outdoor activities during peak hours.")
        elif temp > 30:
            recommendations.append("☀️ Hot weather: Wear light clothing, stay hydrated, use sunscreen.")
        
        if precip > 10:
            recommendations.append("🌧️ Heavy rain expected: Carry an umbrella, avoid waterlogged areas.")
        elif precip > 0:
            recommendations.append("🌦️ Light rain: Carry an umbrella, be cautious while driving.")
        
        if wind > 15:
            recommendations.append("💨 Strong winds: Secure loose objects, avoid outdoor activities.")
        
        # Risk-based recommendations
        risk = results.get('climate_risk', {})
        risk_category = risk.get('category', 'Low')
        
        if risk_category == 'Extreme':
            recommendations.append("⚠️ Extreme climate risk: Stay indoors, avoid all outdoor activities.")
        elif risk_category == 'High':
            recommendations.append("⚠️ High climate risk: Limit outdoor exposure, stay updated on weather alerts.")
        
        # Air quality recommendations
        aqi = results.get('air_quality', {})
        pm25 = aqi.get('pm2_5', 0)
        
        if pm25 > 55:
            recommendations.append("😷 Poor air quality: Wear N95 mask, limit outdoor activities.")
        elif pm25 > 35:
            recommendations.append("😐 Moderate air quality: Sensitive groups should limit outdoor exposure.")
        
        # UV recommendations
        uv = current.get('uv_index', 0)
        if uv > 8:
            recommendations.append("🧴 High UV index: Wear sunscreen, seek shade, wear protective clothing.")
        
        # Rainfall prediction
        rain_pred = results.get('rainfall_prediction', {})
        if rain_pred and rain_pred.get('will_rain'):
            recommendations.append("🌧️ Rain predicted: Plan indoor activities, carry rain gear.")
        
        # Heatwave prediction
        heat_pred = results.get('heatwave_prediction', {})
        if heat_pred and heat_pred.get('is_heatwave'):
            recommendations.append("🔥 Heatwave predicted: Stay indoors, use air conditioning, stay hydrated.")
        
        return recommendations if recommendations else ["Weather conditions are normal. No specific recommendations."]


def main():
    """Test the prediction pipeline"""
    pipeline = RealTimePredictionPipeline()
    
    # Test with Delhi
    print("Testing pipeline for Delhi...")
    results = pipeline.run_complete_pipeline('Delhi', 'India')
    
    print("\nPipeline Results:")
    print(f"Location: {results.get('location', {}).get('city')}")
    print(f"Temperature: {results.get('current_weather', {}).get('temperature_2m')}°C")
    print(f"Climate Risk: {results.get('climate_risk', {}).get('category')} ({results.get('climate_risk', {}).get('overall_score')}/100)")
    
    print("\nRecommendations:")
    for rec in pipeline.generate_recommendations(results):
        print(f"  {rec}")


if __name__ == "__main__":
    main()

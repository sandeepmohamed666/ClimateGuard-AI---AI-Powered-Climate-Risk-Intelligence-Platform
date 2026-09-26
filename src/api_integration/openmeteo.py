"""
Open-Meteo API Integration Module
Fetches real-time weather data from Open-Meteo API
"""

import requests
import logging
from typing import Dict, Optional, List
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenMeteoAPI:
    """Open-Meteo Weather API Client"""
    
    BASE_URL = "https://api.open-meteo.com/v1"
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_current_weather(
        self,
        latitude: float,
        longitude: float,
        hourly: bool = False,
        daily: bool = False
    ) -> Optional[Dict]:
        """
        Get current weather data for a location
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            hourly: Include hourly forecast
            daily: Include daily forecast
            
        Returns:
            Dictionary containing weather data
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,pressure_msl,"
                      "wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
                      "precipitation,cloud_cover,visibility,uv_index"
        }
        
        if hourly:
            params["hourly"] = "temperature_2m,relative_humidity_2m,precipitation"
        
        if daily:
            params["daily"] = "temperature_2m_max,temperature_2m_min,precipitation_sum"
        
        try:
            response = self.session.get(f"{self.BASE_URL}/forecast", params=params)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Successfully fetched weather data for {latitude}, {longitude}")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching weather data: {e}")
            return None
    
    def get_historical_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str
    ) -> Optional[Dict]:
        """
        Get historical weather data for a location
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary containing historical weather data
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                    "wind_speed_10m_max,uv_index_max"
        }
        
        try:
            response = self.session.get(f"{self.BASE_URL}/archive", params=params)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Successfully fetched historical data for {latitude}, {longitude}")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching historical data: {e}")
            return None


class OpenMeteoAirQualityAPI:
    """Open-Meteo Air Quality API Client"""
    
    BASE_URL = "https://air-quality-api.open-meteo.com/v1"
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_air_quality(
        self,
        latitude: float,
        longitude: float,
        hourly: bool = False
    ) -> Optional[Dict]:
        """
        Get air quality data for a location
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            hourly: Include hourly data
            
        Returns:
            Dictionary containing air quality data
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,"
                      "sulphur_dioxide,ozone,dust,aerosol_optical_depth"
        }
        
        if hourly:
            params["hourly"] = "pm10,pm2_5"
        
        try:
            response = self.session.get(f"{self.BASE_URL}/air-quality", params=params)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Successfully fetched air quality data for {latitude}, {longitude}")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching air quality data: {e}")
            return None


def main():
    """Test the API clients"""
    weather_api = OpenMeteoAPI()
    aqi_api = OpenMeteoAirQualityAPI()
    
    # Test with Delhi coordinates
    lat, lon = 28.6139, 77.2090
    
    print("Fetching current weather...")
    weather_data = weather_api.get_current_weather(lat, lon)
    if weather_data:
        print(weather_data)
    
    print("\nFetching air quality...")
    aqi_data = aqi_api.get_air_quality(lat, lon)
    if aqi_data:
        print(aqi_data)


if __name__ == "__main__":
    main()

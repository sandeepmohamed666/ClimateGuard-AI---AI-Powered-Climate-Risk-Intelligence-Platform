"""
Geocoding API Integration Module
Converts city names to latitude and longitude coordinates
"""

import requests
import logging
from typing import Optional, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeocodingAPI:
    """Geocoding API Client using Open-Meteo Geocoding"""
    
    BASE_URL = "https://geocoding-api.open-meteo.com/v1"
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_coordinates(self, city: str, country: str = None) -> Optional[Dict]:
        """
        Get latitude and longitude for a city
        
        Args:
            city: Name of the city
            country: Optional country name for better accuracy
            
        Returns:
            Dictionary containing latitude, longitude, and other location details
        """
        params = {
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json"
        }
        
        if country:
            params["country"] = country
        
        try:
            response = self.session.get(f"{self.BASE_URL}/search", params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("results") and len(data["results"]) > 0:
                result = data["results"][0]
                location_info = {
                    "latitude": result.get("latitude"),
                    "longitude": result.get("longitude"),
                    "name": result.get("name"),
                    "country": result.get("country"),
                    "admin1": result.get("admin1"),
                    "timezone": result.get("timezone")
                }
                logger.info(f"Found coordinates for {city}: {location_info['latitude']}, {location_info['longitude']}")
                return location_info
            else:
                logger.warning(f"No results found for city: {city}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching geocoding data: {e}")
            return None
    
    def get_coordinates_batch(self, cities: list) -> Dict[str, Optional[Dict]]:
        """
        Get coordinates for multiple cities
        
        Args:
            cities: List of city names
            
        Returns:
            Dictionary mapping city names to their coordinate information
        """
        results = {}
        for city in cities:
            results[city] = self.get_coordinates(city)
        return results


def main():
    """Test the geocoding API"""
    geocoding = GeocodingAPI()
    
    # Test with some Indian cities
    cities = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"]
    
    for city in cities:
        coords = geocoding.get_coordinates(city)
        if coords:
            print(f"{city}: {coords['latitude']}, {coords['longitude']}, {coords['country']}")


if __name__ == "__main__":
    main()

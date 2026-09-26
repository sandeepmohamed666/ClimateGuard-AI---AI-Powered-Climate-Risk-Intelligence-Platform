"""
Historical Data Loader Module
Loads and validates historical weather dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalDataLoader:
    """Load and validate historical weather data"""
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader
        
        Args:
            data_path: Path to the historical dataset CSV file
        """
        self.data_path = Path(data_path)
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load historical weather data from CSV
        
        Returns:
            DataFrame containing the historical weather data
        """
        logger.info(f"Loading historical data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Successfully loaded {len(self.data)} records")
            logger.info(f"Columns: {list(self.data.columns)}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_data_info(self) -> dict:
        """
        Get basic information about the dataset
        
        Returns:
            Dictionary containing dataset information
        """
        if self.data is None:
            self.load_data()
        
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicate_rows': self.data.duplicated().sum(),
            'memory_usage': self.data.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        return info
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for numerical columns
        
        Returns:
            DataFrame with summary statistics
        """
        if self.data is None:
            self.load_data()
        
        return self.data.describe(include='all')
    
    def get_unique_regions(self) -> list:
        """
        Get list of unique regions in the dataset
        
        Returns:
            List of unique region names
        """
        if self.data is None:
            self.load_data()
        
        if 'region' in self.data.columns:
            return self.data['region'].unique().tolist()
        elif 'Region' in self.data.columns:
            return self.data['Region'].unique().tolist()
        else:
            logger.warning("Region column not found")
            return []


def main():
    """Test the data loader"""
    loader = HistoricalDataLoader("data/IndianWeatherRepository.csv")
    data = loader.load_data()
    print(loader.get_data_info())
    print("\nSummary Statistics:")
    print(loader.get_summary_statistics())


if __name__ == "__main__":
    main()

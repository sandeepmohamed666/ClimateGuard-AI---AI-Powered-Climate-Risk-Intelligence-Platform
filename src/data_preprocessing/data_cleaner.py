"""
Data Cleaning Module
Handles data cleaning, missing value imputation, and outlier detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and preprocess weather data"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the data cleaner
        
        Args:
            df: Input DataFrame to clean
        """
        self.df = df.copy()
        self.cleaning_report = {}
    
    def remove_duplicates(self) -> pd.DataFrame:
        """Remove duplicate rows from the dataset"""
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        duplicates_removed = initial_count - len(self.df)
        self.cleaning_report['duplicates_removed'] = duplicates_removed
        logger.info(f"Removed {duplicates_removed} duplicate rows")
        return self.df
    
    def handle_missing_values(self, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            strategy: Strategy for imputation ('mean', 'median', 'mode', 'drop')
            
        Returns:
            Cleaned DataFrame
        """
        missing_before = self.df.isnull().sum().sum()
        
        for column in self.df.columns:
            if self.df[column].isnull().sum() > 0:
                if self.df[column].dtype in ['int64', 'float64']:
                    if strategy == 'mean':
                        self.df[column].fillna(self.df[column].mean(), inplace=True)
                    elif strategy == 'median':
                        self.df[column].fillna(self.df[column].median(), inplace=True)
                    elif strategy == 'drop':
                        self.df.dropna(subset=[column], inplace=True)
                else:
                    # For categorical columns, use mode
                    self.df[column].fillna(self.df[column].mode()[0], inplace=True)
        
        missing_after = self.df.isnull().sum().sum()
        self.cleaning_report['missing_values_handled'] = missing_before - missing_after
        logger.info(f"Handled {missing_before - missing_after} missing values")
        return self.df
    
    def remove_impossible_values(self) -> pd.DataFrame:
        """Remove physically impossible weather values"""
        initial_count = len(self.df)
        
        # Temperature: -50 to 60 Celsius
        if 'temperature' in self.df.columns:
            self.df = self.df[(self.df['temperature'] >= -50) & (self.df['temperature'] <= 60)]
        elif 'Temperature' in self.df.columns:
            self.df = self.df[(self.df['Temperature'] >= -50) & (self.df['Temperature'] <= 60)]
        
        # Humidity: 0 to 100%
        if 'humidity' in self.df.columns:
            self.df = self.df[(self.df['humidity'] >= 0) & (self.df['humidity'] <= 100)]
        elif 'Humidity' in self.df.columns:
            self.df = self.df[(self.df['Humidity'] >= 0) & (self.df['Humidity'] <= 100)]
        
        # Pressure: 870 to 1085 hPa
        if 'pressure' in self.df.columns:
            self.df = self.df[(self.df['pressure'] >= 870) & (self.df['pressure'] <= 1085)]
        elif 'Pressure' in self.df.columns:
            self.df = self.df[(self.df['Pressure'] >= 870) & (self.df['Pressure'] <= 1085)]
        
        # Wind Speed: 0 to 150 m/s
        if 'wind_speed' in self.df.columns:
            self.df = self.df[(self.df['wind_speed'] >= 0) & (self.df['wind_speed'] <= 150)]
        elif 'Wind Speed' in self.df.columns:
            self.df = self.df[(self.df['Wind Speed'] >= 0) & (self.df['Wind Speed'] <= 150)]
        
        removed_count = initial_count - len(self.df)
        self.cleaning_report['impossible_values_removed'] = removed_count
        logger.info(f"Removed {removed_count} rows with impossible values")
        return self.df
    
    def detect_outliers_iqr(self, column: str, multiplier: float = 1.5) -> pd.Series:
        """
        Detect outliers using IQR method
        
        Args:
            column: Column name to check for outliers
            multiplier: IQR multiplier (default 1.5)
            
        Returns:
            Boolean series indicating outliers
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        logger.info(f"Detected {outliers.sum()} outliers in {column} using IQR method")
        return outliers
    
    def detect_outliers_zscore(self, column: str, threshold: float = 3) -> pd.Series:
        """
        Detect outliers using Z-score method
        
        Args:
            column: Column name to check for outliers
            threshold: Z-score threshold (default 3)
            
        Returns:
            Boolean series indicating outliers
        """
        z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
        outliers = z_scores > threshold
        logger.info(f"Detected {outliers.sum()} outliers in {column} using Z-score method")
        return outliers
    
    def get_cleaning_report(self) -> Dict:
        """Get the cleaning report"""
        return self.cleaning_report
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """Get the cleaned DataFrame"""
        return self.df


def main():
    """Test the data cleaner"""
    # Create sample data
    data = {
        'temperature': [25, 30, 35, -100, 40, 45, 50],  # -100 is impossible
        'humidity': [60, 70, 80, 90, 110, 50, 65],  # 110 is impossible
        'pressure': [1013, 1015, 1010, 1008, 1012, 1014, 1011],
        'wind_speed': [5, 10, 15, 20, 25, 30, 35]
    }
    df = pd.DataFrame(data)
    
    cleaner = DataCleaner(df)
    cleaner.remove_duplicates()
    cleaner.remove_impossible_values()
    
    print("Cleaning Report:")
    print(cleaner.get_cleaning_report())
    print("\nCleaned Data:")
    print(cleaner.get_cleaned_data())


if __name__ == "__main__":
    main()

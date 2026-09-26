"""
Data Understanding Module
Provides comprehensive data analysis and understanding capabilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataUnderstanding:
    """Comprehensive data understanding and analysis"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the data understanding module
        
        Args:
            df: Input DataFrame to analyze
        """
        self.df = df.copy()
        self.analysis_report = {}
    
    def get_dataset_shape(self) -> Tuple[int, int]:
        """Get dataset shape (rows, columns)"""
        return self.df.shape
    
    def get_data_types(self) -> Dict:
        """Get data types for all columns"""
        return self.df.dtypes.to_dict()
    
    def get_missing_values(self) -> pd.Series:
        """Get count of missing values per column"""
        return self.df.isnull().sum()
    
    def get_missing_percentage(self) -> pd.Series:
        """Get percentage of missing values per column"""
        return (self.df.isnull().sum() / len(self.df) * 100).round(2)
    
    def get_duplicate_count(self) -> int:
        """Get count of duplicate rows"""
        return self.df.duplicated().sum()
    
    def get_unique_regions(self) -> List:
        """Get list of unique regions"""
        region_cols = [col for col in self.df.columns if 'region' in col.lower()]
        if region_cols:
            return self.df[region_cols[0]].unique().tolist()
        return []
    
    def get_feature_descriptions(self) -> Dict:
        """
        Get descriptions for common weather features
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        descriptions = {
            'temperature': 'Air temperature in degrees Celsius',
            'humidity': 'Relative humidity as a percentage',
            'pressure': 'Atmospheric pressure in hPa/hectopascals',
            'wind_speed': 'Wind speed in meters per second',
            'wind_direction': 'Wind direction in degrees',
            'precipitation': 'Amount of rainfall in millimeters',
            'visibility': 'Visibility distance in kilometers',
            'cloud_cover': 'Cloud cover as a percentage',
            'uv_index': 'UV Index (0-11+)',
            'pm2_5': 'Particulate Matter 2.5 micrometers in µg/m³',
            'pm10': 'Particulate Matter 10 micrometers in µg/m³',
            'latitude': 'Geographic latitude',
            'longitude': 'Geographic longitude',
            'region': 'Geographic region or state'
        }
        
        # Map to actual column names (case-insensitive)
        feature_desc = {}
        for col in self.df.columns:
            col_lower = col.lower()
            for key, desc in descriptions.items():
                if key in col_lower:
                    feature_desc[col] = desc
                    break
        
        return feature_desc
    
    def get_statistical_summary(self) -> pd.DataFrame:
        """Get statistical summary for numerical columns"""
        return self.df.describe(include='all')
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix for numerical columns"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        return self.df[numerical_cols].corr()
    
    def get_target_distribution(self, target_col: str) -> pd.Series:
        """
        Get distribution of target variable
        
        Args:
            target_col: Name of target column
            
        Returns:
            Series with value counts
        """
        if target_col in self.df.columns:
            return self.df[target_col].value_counts()
        else:
            logger.warning(f"Target column {target_col} not found")
            return pd.Series()
    
    def generate_comprehensive_report(self) -> Dict:
        """
        Generate a comprehensive data understanding report
        
        Returns:
            Dictionary containing all analysis results
        """
        self.analysis_report = {
            'dataset_shape': self.get_dataset_shape(),
            'data_types': self.get_data_types(),
            'missing_values': self.get_missing_values().to_dict(),
            'missing_percentage': self.get_missing_percentage().to_dict(),
            'duplicate_rows': self.get_duplicate_count(),
            'unique_regions': self.get_unique_regions(),
            'feature_descriptions': self.get_feature_descriptions(),
            'statistical_summary': self.get_statistical_summary().to_dict(),
            'correlation_matrix': self.get_correlation_matrix().to_dict()
        }
        
        return self.analysis_report
    
    def print_report(self):
        """Print a formatted data understanding report"""
        report = self.generate_comprehensive_report()
        
        print("=" * 80)
        print("DATA UNDERSTANDING REPORT")
        print("=" * 80)
        
        print(f"\nDataset Shape: {report['dataset_shape'][0]} rows, {report['dataset_shape'][1]} columns")
        
        print(f"\nDuplicate Rows: {report['duplicate_rows']}")
        
        print(f"\nUnique Regions: {len(report['unique_regions'])}")
        if report['unique_regions']:
            print(f"Regions: {', '.join(map(str, report['unique_regions'][:10]))}")
        
        print("\n" + "-" * 80)
        print("MISSING VALUES")
        print("-" * 80)
        missing_df = pd.DataFrame({
            'Count': report['missing_values'],
            'Percentage': report['missing_percentage']
        })
        print(missing_df[missing_df['Count'] > 0])
        
        print("\n" + "-" * 80)
        print("DATA TYPES")
        print("-" * 80)
        for col, dtype in report['data_types'].items():
            print(f"{col}: {dtype}")
        
        print("\n" + "-" * 80)
        print("STATISTICAL SUMMARY")
        print("-" * 80)
        print(report['statistical_summary'])
        
        print("\n" + "-" * 80)
        print("FEATURE DESCRIPTIONS")
        print("-" * 80)
        for col, desc in report['feature_descriptions'].items():
            print(f"{col}: {desc}")
        
        print("\n" + "=" * 80)


def main():
    """Test the data understanding module"""
    # Create sample data
    data = {
        'temperature': [25, 30, 35, 40, 28, 32, 38, None, 29, 31],
        'humidity': [60, 70, 80, 50, 65, 55, 75, 68, 72, 58],
        'pressure': [1013, 1015, 1010, 1008, 1012, 1014, 1011, 1013, 1010, 1016],
        'wind_speed': [5, 10, 15, 20, 8, 12, 18, 7, 9, 11],
        'precipitation': [0, 5, 15, 0, 2, 0, 25, 1, 3, 0],
        'region': ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Delhi', 'Mumbai', 
                   'Bangalore', 'Chennai', 'Delhi', 'Mumbai']
    }
    df = pd.DataFrame(data)
    
    # Add duplicate
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    
    analyzer = DataUnderstanding(df)
    analyzer.print_report()


if __name__ == "__main__":
    main()

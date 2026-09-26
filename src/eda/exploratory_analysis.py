"""
Exploratory Data Analysis Module
Provides comprehensive EDA capabilities for weather data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExploratoryDataAnalysis:
    """Comprehensive exploratory data analysis"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the EDA module
        
        Args:
            df: Input DataFrame to analyze
        """
        self.df = df.copy()
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def plot_univariate_distributions(self, columns: List[str] = None, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot univariate distributions (histograms and KDE)
        
        Args:
            columns: List of columns to plot (default: all numerical)
            figsize: Figure size
        """
        if columns is None:
            columns = self.numerical_cols[:6]  # Limit to first 6 columns
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if col in self.df.columns:
                axes[i].hist(self.df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        logger.info("Univariate distributions plot displayed")
    
    def plot_boxplots(self, columns: List[str] = None, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot boxplots for numerical columns
        
        Args:
            columns: List of columns to plot (default: all numerical)
            figsize: Figure size
        """
        if columns is None:
            # Filter out datetime columns from numerical columns
            columns = [col for col in self.numerical_cols if not pd.api.types.is_datetime64_any_dtype(self.df[col])]
            columns = columns[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if col in self.df.columns:
                axes[i].boxplot(self.df[col].dropna())
                axes[i].set_title(f'Boxplot of {col}')
                axes[i].set_ylabel(col)
        
        plt.tight_layout()
        plt.show()
        logger.info("Boxplots displayed")
    
    def plot_correlation_heatmap(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Plot correlation heatmap
        
        Args:
            figsize: Figure size
        """
        if len(self.numerical_cols) < 2:
            logger.warning("Not enough numerical columns for correlation heatmap")
            return
        
        corr_matrix = self.df[self.numerical_cols].corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()
        logger.info("Correlation heatmap displayed")
    
    def plot_scatter_matrix(self, columns: List[str] = None, figsize: Tuple[int, int] = (15, 15)):
        """
        Plot scatter matrix (pairplot)
        
        Args:
            columns: List of columns to plot (default: first 4 numerical)
            figsize: Figure size
        """
        if columns is None:
            columns = self.numerical_cols[:4]
        
        if len(columns) < 2:
            logger.warning("Need at least 2 columns for scatter matrix")
            return
        
        df_subset = self.df[columns].dropna()
        
        fig, axes = plt.subplots(len(columns), len(columns), figsize=figsize)
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i == j:
                    axes[i, j].hist(df_subset[col1], bins=20, alpha=0.7)
                else:
                    axes[i, j].scatter(df_subset[col2], df_subset[col1], alpha=0.5)
                
                if i == len(columns) - 1:
                    axes[i, j].set_xlabel(col2)
                if j == 0:
                    axes[i, j].set_ylabel(col1)
        
        plt.tight_layout()
        plt.show()
        logger.info("Scatter matrix displayed")
    
    def plot_time_series(self, date_col: str, value_cols: List[str], figsize: Tuple[int, int] = (15, 8)):
        """
        Plot time series for specified columns
        
        Args:
            date_col: Date column name
            value_cols: List of value columns to plot
            figsize: Figure size
        """
        if date_col not in self.df.columns:
            logger.warning(f"Date column {date_col} not found")
            return
        
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        df_sorted = self.df.sort_values(date_col)
        
        fig, axes = plt.subplots(len(value_cols), 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        for i, col in enumerate(value_cols):
            if col in self.df.columns:
                axes[i].plot(df_sorted[date_col], df_sorted[col], linewidth=1)
                axes[i].set_title(f'{col} Over Time')
                axes[i].set_xlabel('Date')
                axes[i].set_ylabel(col)
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        logger.info("Time series plot displayed")
    
    def plot_monthly_aggregates(self, date_col: str, value_cols: List[str], figsize: Tuple[int, int] = (15, 8)):
        """
        Plot monthly aggregates for specified columns
        
        Args:
            date_col: Date column name
            value_cols: List of value columns to aggregate
            figsize: Figure size
        """
        if date_col not in self.df.columns:
            logger.warning(f"Date column {date_col} not found")
            return
        
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df['month'] = self.df[date_col].dt.month
        self.df['year'] = self.df[date_col].dt.year
        
        fig, axes = plt.subplots(len(value_cols), 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        for i, col in enumerate(value_cols):
            if col in self.df.columns:
                monthly_avg = self.df.groupby('month')[col].mean()
                axes[i].bar(monthly_avg.index, monthly_avg.values)
                axes[i].set_title(f'Monthly Average {col}')
                axes[i].set_xlabel('Month')
                axes[i].set_ylabel(col)
                axes[i].set_xticks(range(1, 13))
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        logger.info("Monthly aggregates plot displayed")
    
    def plot_regional_analysis(self, region_col: str, value_col: str, figsize: Tuple[int, int] = (12, 6)):
        """
        Plot regional analysis for a value column
        
        Args:
            region_col: Region column name
            value_col: Value column to analyze
            figsize: Figure size
        """
        if region_col not in self.df.columns or value_col not in self.df.columns:
            logger.warning(f"Columns {region_col} or {value_col} not found")
            return
        
        regional_avg = self.df.groupby(region_col)[value_col].mean().sort_values(ascending=False)
        
        plt.figure(figsize=figsize)
        regional_avg.plot(kind='bar', color='steelblue')
        plt.title(f'Average {value_col} by Region')
        plt.xlabel('Region')
        plt.ylabel(value_col)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        logger.info("Regional analysis plot displayed")
    
    def plot_categorical_distribution(self, col: str, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot distribution of categorical variable
        
        Args:
            col: Categorical column name
            figsize: Figure size
        """
        if col not in self.df.columns:
            logger.warning(f"Column {col} not found")
            return
        
        value_counts = self.df[col].value_counts()
        
        plt.figure(figsize=figsize)
        value_counts.plot(kind='bar', color='coral')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        logger.info(f"Categorical distribution for {col} displayed")
    
    def generate_eda_report(self, date_col: str = None) -> Dict:
        """
        Generate a comprehensive EDA report
        
        Args:
            date_col: Optional date column for time series analysis
            
        Returns:
            Dictionary containing EDA results
        """
        report = {
            'numerical_columns': self.numerical_cols,
            'categorical_columns': self.categorical_cols,
            'plots_generated': []
        }
        
        # Generate plots
        if self.numerical_cols:
            self.plot_univariate_distributions()
            report['plots_generated'].append('univariate_distributions')
            
            self.plot_boxplots()
            report['plots_generated'].append('boxplots')
            
            self.plot_correlation_heatmap()
            report['plots_generated'].append('correlation_heatmap')
            
            self.plot_scatter_matrix()
            report['plots_generated'].append('scatter_matrix')
        
        if date_col and self.numerical_cols:
            self.plot_time_series(date_col, self.numerical_cols[:3])
            report['plots_generated'].append('time_series')
            
            self.plot_monthly_aggregates(date_col, self.numerical_cols[:3])
            report['plots_generated'].append('monthly_aggregates')
        
        # Regional analysis if region column exists
        region_cols = [col for col in self.df.columns if 'region' in col.lower()]
        if region_cols and self.numerical_cols:
            self.plot_regional_analysis(region_cols[0], self.numerical_cols[0])
            report['plots_generated'].append('regional_analysis')
        
        # Categorical distributions
        for col in self.categorical_cols[:3]:
            self.plot_categorical_distribution(col)
            report['plots_generated'].append(f'categorical_{col}')
        
        return report


def main():
    """Test the EDA module"""
    # Create sample data with date
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    data = {
        'date': dates,
        'temperature': np.random.normal(28, 5, 365),
        'humidity': np.random.normal(65, 15, 365),
        'pressure': np.random.normal(1013, 10, 365),
        'wind_speed': np.random.normal(10, 5, 365),
        'precipitation': np.random.exponential(2, 365),
        'region': np.random.choice(['Delhi', 'Mumbai', 'Bangalore', 'Chennai'], 365)
    }
    df = pd.DataFrame(data)
    
    eda = ExploratoryDataAnalysis(df)
    report = eda.generate_eda_report(date_col='date')
    
    print("EDA Report:")
    print(f"Numerical columns: {report['numerical_columns']}")
    print(f"Categorical columns: {report['categorical_columns']}")
    print(f"Plots generated: {report['plots_generated']}")


if __name__ == "__main__":
    main()

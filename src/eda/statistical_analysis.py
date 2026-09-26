"""
Statistical Analysis Module
Provides comprehensive statistical analysis capabilities
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalAnalysis:
    """Comprehensive statistical analysis"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the statistical analysis module
        
        Args:
            df: Input DataFrame to analyze
        """
        self.df = df.copy()
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
    
    def calculate_descriptive_stats(self) -> pd.DataFrame:
        """
        Calculate descriptive statistics
        
        Returns:
            DataFrame with descriptive statistics
        """
        stats_df = self.df[self.numerical_cols].describe()
        
        # Add additional statistics
        stats_df.loc['skewness'] = self.df[self.numerical_cols].skew()
        stats_df.loc['kurtosis'] = self.df[self.numerical_cols].kurt()
        stats_df.loc['range'] = stats_df.loc['max'] - stats_df.loc['min']
        
        return stats_df
    
    def calculate_confidence_interval(self, column: str, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for a column
        
        Args:
            column: Column name
            confidence: Confidence level (default 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if column not in self.numerical_cols:
            logger.warning(f"Column {column} is not numerical")
            return (0, 0)
        
        data = self.df[column].dropna()
        mean = data.mean()
        std_err = stats.sem(data)
        
        ci = stats.t.interval(confidence, len(data) - 1, loc=mean, scale=std_err)
        
        return ci
    
    def calculate_correlation(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation DataFrame
        """
        return self.df[self.numerical_cols].corr(method=method)
    
    def calculate_covariance(self) -> pd.DataFrame:
        """
        Calculate covariance matrix
        
        Returns:
            Covariance DataFrame
        """
        return self.df[self.numerical_cols].cov()
    
    def perform_hypothesis_test(
        self, 
        column1: str, 
        column2: str = None, 
        test_type: str = 'ttest'
    ) -> Dict:
        """
        Perform hypothesis testing
        
        Args:
            column1: First column
            column2: Second column (optional, for two-sample tests)
            test_type: Type of test ('ttest', 'anova', 'mannwhitney')
            
        Returns:
            Dictionary with test results
        """
        result = {'test_type': test_type}
        
        if test_type == 'ttest':
            if column2:
                # Two-sample t-test
                stat, p_value = stats.ttest_ind(
                    self.df[column1].dropna(), 
                    self.df[column2].dropna()
                )
                result['statistic'] = stat
                result['p_value'] = p_value
                result['significant'] = p_value < 0.05
            else:
                # One-sample t-test (test against mean)
                stat, p_value = stats.ttest_1samp(self.df[column1].dropna(), 0)
                result['statistic'] = stat
                result['p_value'] = p_value
                result['significant'] = p_value < 0.05
        
        elif test_type == 'mannwhitney':
            if column2:
                stat, p_value = stats.mannwhitneyu(
                    self.df[column1].dropna(), 
                    self.df[column2].dropna()
                )
                result['statistic'] = stat
                result['p_value'] = p_value
                result['significant'] = p_value < 0.05
        
        elif test_type == 'anova':
            if column2:
                # One-way ANOVA
                groups = [self.df[column1].dropna(), self.df[column2].dropna()]
                stat, p_value = stats.f_oneway(*groups)
                result['statistic'] = stat
                result['p_value'] = p_value
                result['significant'] = p_value < 0.05
        
        return result
    
    def perform_normality_test(self, column: str) -> Dict:
        """
        Perform normality test (Shapiro-Wilk)
        
        Args:
            column: Column to test
            
        Returns:
            Dictionary with test results
        """
        if column not in self.numerical_cols:
            logger.warning(f"Column {column} is not numerical")
            return {}
        
        data = self.df[column].dropna()
        
        # Shapiro-Wilk test (limited to 5000 samples)
        if len(data) > 5000:
            data = data.sample(5000, random_state=42)
        
        stat, p_value = stats.shapiro(data)
        
        return {
            'test': 'Shapiro-Wilk',
            'statistic': stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
    
    def detect_outliers_statistical(self, column: str, method: str = 'zscore') -> pd.Series:
        """
        Detect outliers using statistical methods
        
        Args:
            column: Column to analyze
            method: Method to use ('zscore', 'iqr')
            
        Returns:
            Boolean series indicating outliers
        """
        if column not in self.numerical_cols:
            logger.warning(f"Column {column} is not numerical")
            return pd.Series([False] * len(self.df))
        
        data = self.df[column]
        
        if method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = z_scores > 3
        elif method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = (data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)
        else:
            logger.warning(f"Unknown method: {method}")
            outliers = pd.Series([False] * len(self.df))
        
        return outliers
    
    def calculate_percentiles(self, column: str, percentiles: List[float] = None) -> Dict:
        """
        Calculate percentiles for a column
        
        Args:
            column: Column to analyze
            percentiles: List of percentiles to calculate (default: [5, 25, 50, 75, 95])
            
        Returns:
            Dictionary with percentile values
        """
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]
        
        if column not in self.numerical_cols:
            logger.warning(f"Column {column} is not numerical")
            return {}
        
        values = {}
        for p in percentiles:
            values[f'p{p}'] = self.df[column].quantile(p / 100)
        
        return values
    
    def generate_statistical_report(self) -> Dict:
        """
        Generate a comprehensive statistical report
        
        Returns:
            Dictionary containing all statistical analysis results
        """
        report = {
            'descriptive_statistics': self.calculate_descriptive_stats().to_dict(),
            'correlation_matrix': self.calculate_correlation().to_dict(),
            'covariance_matrix': self.calculate_covariance().to_dict(),
            'confidence_intervals': {},
            'normality_tests': {},
            'outlier_detection': {}
        }
        
        # Confidence intervals for numerical columns
        for col in self.numerical_cols:
            ci = self.calculate_confidence_interval(col)
            report['confidence_intervals'][col] = {
                'lower': ci[0],
                'upper': ci[1]
            }
        
        # Normality tests
        for col in self.numerical_cols:
            normality = self.perform_normality_test(col)
            report['normality_tests'][col] = normality
        
        # Outlier detection
        for col in self.numerical_cols:
            outliers_zscore = self.detect_outliers_statistical(col, 'zscore')
            outliers_iqr = self.detect_outliers_statistical(col, 'iqr')
            report['outlier_detection'][col] = {
                'zscore_outliers': int(outliers_zscore.sum()),
                'iqr_outliers': int(outliers_iqr.sum())
            }
        
        return report
    
    def print_report(self):
        """Print a formatted statistical report"""
        report = self.generate_statistical_report()
        
        print("=" * 80)
        print("STATISTICAL ANALYSIS REPORT")
        print("=" * 80)
        
        print("\n" + "-" * 80)
        print("DESCRIPTIVE STATISTICS")
        print("-" * 80)
        print(pd.DataFrame(report['descriptive_statistics']))
        
        print("\n" + "-" * 80)
        print("CONFIDENCE INTERVALS (95%)")
        print("-" * 80)
        for col, ci in report['confidence_intervals'].items():
            print(f"{col}: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
        
        print("\n" + "-" * 80)
        print("NORMALITY TESTS (Shapiro-Wilk)")
        print("-" * 80)
        for col, test in report['normality_tests'].items():
            print(f"{col}: p-value = {test['p_value']:.4f}, Normal = {test['is_normal']}")
        
        print("\n" + "-" * 80)
        print("OUTLIER DETECTION")
        print("-" * 80)
        for col, outliers in report['outlier_detection'].items():
            print(f"{col}: Z-score outliers = {outliers['zscore_outliers']}, "
                  f"IQR outliers = {outliers['iqr_outliers']}")
        
        print("\n" + "=" * 80)


def main():
    """Test the statistical analysis module"""
    # Create sample data
    np.random.seed(42)
    data = {
        'temperature': np.random.normal(28, 5, 1000),
        'humidity': np.random.normal(65, 15, 1000),
        'pressure': np.random.normal(1013, 10, 1000),
        'wind_speed': np.random.normal(10, 5, 1000),
        'precipitation': np.random.exponential(2, 1000)
    }
    df = pd.DataFrame(data)
    
    analyzer = StatisticalAnalysis(df)
    analyzer.print_report()


if __name__ == "__main__":
    main()

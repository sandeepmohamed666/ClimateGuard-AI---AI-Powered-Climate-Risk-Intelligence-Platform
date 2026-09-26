"""
Feature Selection Module
Provides various feature selection techniques for ML models
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, f_regression,
    mutual_info_classif, mutual_info_regression, RFE
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """Comprehensive feature selection"""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Initialize the feature selector
        
        Args:
            X: Feature DataFrame
            y: Target Series (optional)
        """
        self.X = X.copy()
        self.y = y.copy() if y is not None else None
        self.selected_features = []
        self.selection_report = {}
    
    def variance_threshold_selection(self, threshold: float = 0.01) -> List[str]:
        """
        Select features using variance threshold
        
        Args:
            threshold: Variance threshold (default 0.01)
            
        Returns:
            List of selected feature names
        """
        numerical_cols = self.X.select_dtypes(include=[np.number]).columns
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(self.X[numerical_cols])
        
        selected = numerical_cols[selector.get_support()].tolist()
        self.selection_report['variance_threshold'] = {
            'threshold': threshold,
            'selected_count': len(selected),
            'removed_count': len(numerical_cols) - len(selected)
        }
        
        logger.info(f"Variance threshold selected {len(selected)} features")
        return selected
    
    def correlation_selection(self, threshold: float = 0.95) -> List[str]:
        """
        Select features by removing highly correlated features
        
        Args:
            threshold: Correlation threshold (default 0.95)
            
        Returns:
            List of selected feature names
        """
        numerical_cols = self.X.select_dtypes(include=[np.number]).columns
        corr_matrix = self.X[numerical_cols].corr().abs()
        
        # Find highly correlated pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_tri.columns 
                   if any(upper_tri[column] > threshold)]
        
        selected = [col for col in numerical_cols if col not in to_drop]
        
        self.selection_report['correlation'] = {
            'threshold': threshold,
            'selected_count': len(selected),
            'removed_count': len(to_drop),
            'removed_features': to_drop
        }
        
        logger.info(f"Correlation selection removed {len(to_drop)} features")
        return selected
    
    def univariate_selection(
        self, 
        k: int = 10, 
        score_func: str = 'f_classif'
    ) -> List[str]:
        """
        Select features using univariate statistical tests
        
        Args:
            k: Number of features to select
            score_func: Score function ('f_classif', 'f_regression', 'mutual_info')
            
        Returns:
            List of selected feature names
        """
        if self.y is None:
            logger.warning("Target variable not provided for univariate selection")
            return []
        
        numerical_cols = self.X.select_dtypes(include=[np.number]).columns
        X_numerical = self.X[numerical_cols].fillna(0)
        
        # Encode target if categorical
        if self.y.dtype == 'object':
            le = LabelEncoder()
            y_encoded = le.fit_transform(self.y.astype(str))
        else:
            y_encoded = self.y
        
        # Select score function
        if score_func == 'f_classif':
            func = f_classif
        elif score_func == 'f_regression':
            func = f_regression
        elif score_func == 'mutual_info':
            func = mutual_info_classif if len(np.unique(y_encoded)) < 20 else mutual_info_regression
        else:
            func = f_classif
        
        selector = SelectKBest(func, k=min(k, len(numerical_cols)))
        selector.fit(X_numerical, y_encoded)
        
        selected = numerical_cols[selector.get_support()].tolist()
        scores = selector.scores_[selector.get_support()]
        
        self.selection_report['univariate'] = {
            'k': k,
            'score_func': score_func,
            'selected_count': len(selected),
            'feature_scores': dict(zip(selected, scores))
        }
        
        logger.info(f"Univariate selection selected {len(selected)} features")
        return selected
    
    def recursive_feature_elimination(
        self, 
        n_features: int = 10, 
        estimator_type: str = 'random_forest'
    ) -> List[str]:
        """
        Select features using Recursive Feature Elimination
        
        Args:
            n_features: Number of features to select
            estimator_type: Type of estimator ('random_forest')
            
        Returns:
            List of selected feature names
        """
        if self.y is None:
            logger.warning("Target variable not provided for RFE")
            return []
        
        numerical_cols = self.X.select_dtypes(include=[np.number]).columns
        X_numerical = self.X[numerical_cols].fillna(0)
        
        # Encode target if categorical
        if self.y.dtype == 'object':
            le = LabelEncoder()
            y_encoded = le.fit_transform(self.y.astype(str))
        else:
            y_encoded = self.y
        
        # Create estimator
        if len(np.unique(y_encoded)) < 20:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # RFE
        rfe = RFE(estimator=estimator, n_features_to_select=min(n_features, len(numerical_cols)))
        rfe.fit(X_numerical, y_encoded)
        
        selected = numerical_cols[rfe.support_].tolist()
        rankings = dict(zip(numerical_cols, rfe.ranking_))
        
        self.selection_report['rfe'] = {
            'n_features': n_features,
            'selected_count': len(selected),
            'feature_rankings': rankings
        }
        
        logger.info(f"RFE selected {len(selected)} features")
        return selected
    
    def random_forest_importance(
        self, 
        n_features: int = 10, 
        threshold: float = None
    ) -> List[str]:
        """
        Select features based on Random Forest importance
        
        Args:
            n_features: Number of top features to select
            threshold: Importance threshold (optional)
            
        Returns:
            List of selected feature names
        """
        if self.y is None:
            logger.warning("Target variable not provided for RF importance")
            return []
        
        numerical_cols = self.X.select_dtypes(include=[np.number]).columns
        X_numerical = self.X[numerical_cols].fillna(0)
        
        # Encode target if categorical
        if self.y.dtype == 'object':
            le = LabelEncoder()
            y_encoded = le.fit_transform(self.y.astype(str))
        else:
            y_encoded = self.y
        
        # Train Random Forest
        if len(np.unique(y_encoded)) < 20:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
        
        rf.fit(X_numerical, y_encoded)
        
        # Get feature importances
        importances = pd.Series(rf.feature_importances_, index=numerical_cols)
        importances = importances.sort_values(ascending=False)
        
        # Select features
        if threshold:
            selected = importances[importances >= threshold].index.tolist()
        else:
            selected = importances.head(n_features).index.tolist()
        
        self.selection_report['rf_importance'] = {
            'n_features': n_features,
            'threshold': threshold,
            'selected_count': len(selected),
            'feature_importances': importances.to_dict()
        }
        
        logger.info(f"Random Forest importance selected {len(selected)} features")
        return selected
    
    def combined_selection(
        self, 
        methods: List[str] = None, 
        min_votes: int = 2
    ) -> List[str]:
        """
        Combine multiple feature selection methods
        
        Args:
            methods: List of methods to use
            min_votes: Minimum number of votes for a feature to be selected
            
        Returns:
            List of selected feature names
        """
        if methods is None:
            methods = ['variance_threshold', 'correlation', 'rf_importance']
        
        all_features = {}
        
        for method in methods:
            if method == 'variance_threshold':
                features = self.variance_threshold_selection()
            elif method == 'correlation':
                features = self.correlation_selection()
            elif method == 'univariate':
                features = self.univariate_selection()
            elif method == 'rfe':
                features = self.recursive_feature_elimination()
            elif method == 'rf_importance':
                features = self.random_forest_importance()
            else:
                continue
            
            for feature in features:
                if feature not in all_features:
                    all_features[feature] = 0
                all_features[feature] += 1
        
        # Select features with minimum votes
        selected = [feat for feat, votes in all_features.items() if votes >= min_votes]
        
        self.selection_report['combined'] = {
            'methods': methods,
            'min_votes': min_votes,
            'selected_count': len(selected),
            'feature_votes': all_features
        }
        
        logger.info(f"Combined selection selected {len(selected)} features")
        return selected
    
    def get_selection_report(self) -> Dict:
        """Get the feature selection report"""
        return self.selection_report
    
    def print_report(self):
        """Print a formatted feature selection report"""
        report = self.selection_report
        
        print("=" * 80)
        print("FEATURE SELECTION REPORT")
        print("=" * 80)
        
        for method, results in report.items():
            print(f"\n{method.upper()}:")
            print(f"  Selected: {results.get('selected_count', 0)} features")
            if 'removed_count' in results:
                print(f"  Removed: {results['removed_count']} features")
            if 'feature_scores' in results:
                print("  Top features by score:")
                for feat, score in sorted(results['feature_scores'].items(), 
                                          key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    {feat}: {score:.4f}")
            if 'feature_importances' in results:
                print("  Top features by importance:")
                for feat, imp in sorted(results['feature_importances'].items(), 
                                        key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    {feat}: {imp:.4f}")
        
        print("\n" + "=" * 80)


def main():
    """Test the feature selector"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 0.01, n_samples),  # Low variance
        'feature4': np.random.normal(0, 1, n_samples),
        'feature5': np.random.normal(0, 1, n_samples),
    }
    X = pd.DataFrame(data)
    
    # Create target
    y = (X['feature1'] + X['feature2'] + np.random.normal(0, 0.1, n_samples) > 0).astype(int)
    
    selector = FeatureSelector(X, y)
    
    # Test different methods
    print("Variance Threshold Selection:")
    print(selector.variance_threshold_selection())
    
    print("\nRandom Forest Importance:")
    print(selector.random_forest_importance(n_features=3))
    
    print("\nCombined Selection:")
    print(selector.combined_selection())
    
    selector.print_report()


if __name__ == "__main__":
    main()

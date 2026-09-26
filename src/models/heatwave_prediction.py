"""
Heatwave Prediction Module
Machine learning models for heatwave prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from typing import Dict, Optional, Tuple
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeatwavePredictor:
    """Heatwave prediction using various ML models"""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        """
        Initialize the heatwave predictor
        
        Args:
            X: Feature DataFrame
            y: Target Series (1 for heatwave, 0 for no heatwave)
        """
        self.X = X.copy()
        self.y = y.copy()
        self.scaler = StandardScaler()
        self.models = {}
        self.model_report = {}
        self.best_model = None
        self.best_model_name = None
    
    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess data: handle missing values and scale
        
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        # Fill missing values
        X_filled = self.X.fillna(self.X.median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_filled, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        logger.info("Data preprocessed and split")
        return X_train_scaled, X_test_scaled
    
    def train_logistic_regression(self, C: float = 1.0, max_iter: int = 1000) -> Dict:
        """
        Train Logistic Regression model
        
        Args:
            C: Regularization parameter
            max_iter: Maximum iterations
            
        Returns:
            Dictionary with model performance metrics
        """
        X_train_scaled, X_test_scaled = self.preprocess_data()
        
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        model.fit(X_train_scaled, self.y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        self.models['logistic_regression'] = model
        self.model_report['logistic_regression'] = metrics
        
        logger.info(f"Logistic Regression trained - Accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def train_random_forest(self, n_estimators: int = 100, max_depth: int = 10) -> Dict:
        """
        Train Random Forest model
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            
        Returns:
            Dictionary with model performance metrics
        """
        X_train_scaled, X_test_scaled = self.preprocess_data()
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, self.y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        # Feature importance
        feature_importance = dict(zip(self.X.columns, model.feature_importances_))
        metrics['feature_importance'] = feature_importance
        
        self.models['random_forest'] = model
        self.model_report['random_forest'] = metrics
        
        logger.info(f"Random Forest trained - Accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def train_gradient_boosting(self, n_estimators: int = 100, learning_rate: float = 0.1) -> Dict:
        """
        Train Gradient Boosting model
        
        Args:
            n_estimators: Number of boosting stages
            learning_rate: Learning rate
            
        Returns:
            Dictionary with model performance metrics
        """
        X_train_scaled, X_test_scaled = self.preprocess_data()
        
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(X_train_scaled, self.y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        # Feature importance
        feature_importance = dict(zip(self.X.columns, model.feature_importances_))
        metrics['feature_importance'] = feature_importance
        
        self.models['gradient_boosting'] = model
        self.model_report['gradient_boosting'] = metrics
        
        logger.info(f"Gradient Boosting trained - Accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba) -> Dict:
        """Calculate classification metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
    
    def select_best_model(self, metric: str = 'f1_score') -> str:
        """
        Select the best model based on a metric
        
        Args:
            metric: Metric to use for selection
            
        Returns:
            Name of the best model
        """
        if not self.model_report:
            logger.warning("No models trained yet")
            return None
        
        best_model_name = max(
            self.model_report.keys(),
            key=lambda x: self.model_report[x].get(metric, 0)
        )
        
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        logger.info(f"Best model selected: {best_model_name}")
        return best_model_name
    
    def predict(self, X_new: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the best model
        
        Args:
            X_new: New features to predict
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.best_model is None:
            logger.warning("No best model selected")
            return None, None
        
        X_new_scaled = self.scaler.transform(X_new.fillna(X_new.median()))
        predictions = self.best_model.predict(X_new_scaled)
        probabilities = self.best_model.predict_proba(X_new_scaled)[:, 1]
        
        return predictions, probabilities
    
    def save_model(self, model_name: str, filepath: str):
        """Save a trained model"""
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found")
            return
        
        joblib.dump(self.models[model_name], filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model = joblib.load(filepath)
        self.best_model = model
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_report(self) -> Dict:
        """Get the model training report"""
        return self.model_report
    
    def print_report(self):
        """Print a formatted model report"""
        report = self.model_report
        
        print("=" * 80)
        print("HEATWAVE PREDICTION MODEL REPORT")
        print("=" * 80)
        
        for model_name, metrics in report.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
            
            if 'feature_importance' in metrics:
                print("  Top 5 Feature Importances:")
                for feat, imp in sorted(metrics['feature_importance'].items(),
                                        key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    {feat}: {imp:.4f}")
        
        if self.best_model_name:
            print(f"\nBest Model: {self.best_model_name}")
        
        print("\n" + "=" * 80)


def main():
    """Test the heatwave predictor"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'temperature': np.random.normal(35, 8, n_samples),
        'humidity': np.random.normal(50, 20, n_samples),
        'pressure': np.random.normal(1010, 15, n_samples),
        'uv_index': np.random.uniform(0, 12, n_samples),
        'heat_index': np.random.normal(38, 10, n_samples)
    }
    X = pd.DataFrame(data)
    
    # Create target (heatwave based on temperature > 35°C)
    y = (X['temperature'] > 35).astype(int)
    
    predictor = HeatwavePredictor(X, y)
    
    # Train models
    predictor.train_logistic_regression()
    predictor.train_random_forest()
    predictor.train_gradient_boosting()
    
    # Select best model
    predictor.select_best_model()
    
    predictor.print_report()


if __name__ == "__main__":
    main()

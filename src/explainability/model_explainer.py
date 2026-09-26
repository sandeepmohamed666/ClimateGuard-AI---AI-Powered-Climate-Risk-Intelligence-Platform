"""
Explainable AI Module using sklearn native features
Provides model explanations for climate predictions without SHAP
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from typing import Dict, List, Optional, Tuple
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExplainer:
    """Explain ML model predictions using sklearn native methods"""
    
    def __init__(self, model, X_train: pd.DataFrame, feature_names: List[str] = None):
        """
        Initialize the model explainer
        
        Args:
            model: Trained ML model
            X_train: Training data for explanations
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train.copy()
        self.feature_names = feature_names or X_train.columns.tolist()
        self.scaler = StandardScaler()
        self.X_train_scaled = None
        self.feature_importance = {}
        self.permutation_importance = {}
        self.explanation_report = {}
    
    def preprocess_data(self):
        """Scale the training data"""
        numerical_cols = self.X_train.select_dtypes(include=[np.number]).columns
        # store numerical columns used for scaling so we can reuse the exact order later
        self.numerical_cols = list(numerical_cols)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train[self.numerical_cols].fillna(0))
        logger.info("Data preprocessed and scaled")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the model
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            importance = dict(zip(self.feature_names, np.abs(self.model.coef_[0])))
        else:
            # Use permutation importance as fallback
            self.calculate_permutation_importance()
            importance = self.permutation_importance
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        self.feature_importance = importance
        self.explanation_report['feature_importance'] = importance
        return importance

    def save(self, filepath: str):
        """Save this sklearn explainer instance to a pickle file."""
        joblib.dump(self, filepath)
        logger.info(f"Sklearn explainer saved to {filepath}")
        return filepath
    
    def calculate_permutation_importance(self, n_repeats: int = 5, random_state: int = 42):
        """
        Calculate permutation importance
        
        Args:
            n_repeats: Number of times to permute each feature
            random_state: Random seed
        """
        if self.X_train_scaled is None:
            self.preprocess_data()
        
        # Create target if needed
        if hasattr(self.model, 'predict_proba'):
            scoring = 'accuracy'
        else:
            scoring = 'r2'
        
        result = permutation_importance(
            self.model, self.X_train_scaled, 
            np.random.randint(0, 2, len(self.X_train_scaled)) if scoring == 'accuracy' else np.random.randn(len(self.X_train_scaled)),
            n_repeats=n_repeats, random_state=random_state, scoring=scoring
        )
        
        importance = dict(zip(self.feature_names, result.importances_mean))
        self.permutation_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        logger.info(f"Permutation importance calculated for {len(self.feature_names)} features")
    
    def plot_feature_importance(self, save_path: str = None, top_n: int = 15):
        """
        Plot feature importance inline
        
        Args:
            save_path: Optional path to save the plot
            top_n: Number of top features to show
        """
        if not self.feature_importance:
            self.get_feature_importance()

        plt.figure(figsize=(12, 8))
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, scores = zip(*sorted_features)
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info("Feature importance plot displayed")
    
    def plot_permutation_importance(self, save_path: str = None, top_n: int = 15):
        """
        Plot permutation importance inline
        
        Args:
            save_path: Optional path to save the plot
            top_n: Number of top features to show
        """
        if not self.permutation_importance:
            self.calculate_permutation_importance()

        plt.figure(figsize=(12, 8))
        sorted_features = sorted(self.permutation_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, scores = zip(*sorted_features)
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Permutation Importance')
        plt.title(f'Top {top_n} Permutation Importances')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info("Permutation importance plot displayed")
    
    def plot_partial_dependence(self, feature_idx: int = 0, save_path: str = None):
        """
        Plot partial dependence for a single feature inline
        
        Args:
            feature_idx: Index of the feature to plot
            save_path: Optional path to save the plot
        """
        if self.X_train_scaled is None:
            self.preprocess_data()

        fig, ax = plt.subplots(figsize=(10, 6))
        PartialDependenceDisplay.from_estimator(
            self.model, self.X_train_scaled, [feature_idx],
            feature_names=self.feature_names, ax=ax
        )
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info("Partial dependence plot displayed")
    
    def explain_instance(self, X_instance: pd.DataFrame) -> Dict:
        """
        Explain a single instance using feature contributions
        
        Args:
            X_instance: Instance to explain
            
        Returns:
            Dictionary with explanation details
        """
        if self.X_train_scaled is None:
            self.preprocess_data()
        
        # Use the same numerical columns/order used during preprocessing
        if hasattr(self, 'numerical_cols') and self.numerical_cols is not None:
            cols = self.numerical_cols
        else:
            cols = X_instance.select_dtypes(include=[np.number]).columns.tolist()

        # Reindex to ensure all expected columns are present (missing -> 0)
        X_for_scaling = X_instance.reindex(columns=cols, fill_value=0)
        X_scaled = self.scaler.transform(X_for_scaling.fillna(0))
        
        # Get prediction
        if hasattr(self.model, 'predict_proba'):
            prediction = self.model.predict_proba(X_scaled)[0]
            pred_class = np.argmax(prediction)
            pred_value = float(prediction[pred_class])
        else:
            prediction = self.model.predict(X_scaled)[0]
            pred_value = float(prediction)
        
        # Calculate feature contributions using SHAP-like approximation
        # Use feature importance as a proxy
        if not self.feature_importance:
            self.get_feature_importance()
        
        contributions = {}
        for i, feat in enumerate(self.feature_names):
            importance = self.feature_importance.get(feat, 0)
            # Normalize importance
            if importance > 0:
                contributions[feat] = {
                    'importance': float(importance),
                    'impact': 'high' if importance > np.mean(list(self.feature_importance.values())) else 'low'
                }
            else:
                contributions[feat] = {
                    'importance': 0.0,
                    'impact': 'none'
                }
        
        explanation = {
            'prediction': pred_value,
            'feature_contributions': contributions,
            'top_features': sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        
        return explanation

    def generate_explanation_report(self) -> Dict:
        """
        Generate a comprehensive explanation report
        
        Returns:
            Dictionary with all explanation results
        """
        self.get_feature_importance()
        self.calculate_permutation_importance()
        
        self.explanation_report = {
            'feature_importance': self.feature_importance,
            'permutation_importance': self.permutation_importance,
            'n_features': len(self.feature_names),
            'n_samples': len(self.X_train)
        }
        
        return self.explanation_report
    
    def print_report(self):
        """Print a formatted explanation report"""
        report = self.explanation_report
        
        print("=" * 80)
        print("EXPLAINABLE AI REPORT (sklearn native)")
        print("=" * 80)
        
        print(f"\nNumber of features: {report['n_features']}")
        print(f"Number of samples: {report['n_samples']}")
        
        print("\n" + "-" * 80)
        print("FEATURE IMPORTANCE (Model-based)")
        print("-" * 80)
        for i, (feat, imp) in enumerate(list(report['feature_importance'].items())[:15]):
            print(f"{i+1}. {feat}: {imp:.4f}")
        
        print("\n" + "-" * 80)
        print("PERMUTATION IMPORTANCE")
        print("-" * 80)
        for i, (feat, imp) in enumerate(list(report['permutation_importance'].items())[:15]):
            print(f"{i+1}. {feat}: {imp:.4f}")
        
        print("\n" + "=" * 80)


def main():
    """Test the model explainer"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'temperature': np.random.normal(28, 5, n_samples),
        'humidity': np.random.normal(65, 15, n_samples),
        'pressure': np.random.normal(1013, 10, n_samples),
        'wind_speed': np.random.normal(10, 5, n_samples),
        'precipitation': np.random.exponential(2, n_samples)
    }
    X = pd.DataFrame(data)
    y = (X['humidity'] > 70).astype(int)
    
    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Create explainer
    explainer = ModelExplainer(model, X)
    
    # Generate explanations
    explainer.generate_explanation_report()
    explainer.print_report()
    
    # Plot visualizations
    explainer.plot_feature_importance()
    explainer.plot_permutation_importance()


if __name__ == "__main__":
    main()

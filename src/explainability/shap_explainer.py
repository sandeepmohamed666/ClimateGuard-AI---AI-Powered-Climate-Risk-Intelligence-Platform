"""
Explainable AI Module using SHAP
Provides model explanations for climate predictions
"""

import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import logging
import joblib
import matplotlib.pyplot as plt
from IPython.display import display, HTML

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExplainer:
    """Explain ML model predictions using SHAP"""
    
    def __init__(self, model, X_train: pd.DataFrame, feature_names: List[str] = None):
        """
        Initialize the model explainer
        
        Args:
            model: Trained ML model
            X_train: Training data for SHAP values
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train.copy()
        self.feature_names = feature_names or X_train.columns.tolist()
        self.scaler = StandardScaler()
        self.X_train_scaled = None
        self.explainer = None
        self.shap_values = None
        self.explanation_report = {}
    
    def preprocess_data(self):
        """Scale the training data"""
        numerical_cols = self.X_train.select_dtypes(include=[np.number]).columns
        self.X_train_scaled = self.scaler.fit_transform(self.X_train[numerical_cols].fillna(0))
        logger.info("Data preprocessed and scaled")
    
    def create_tree_explainer(self):
        """Create SHAP explainer for tree-based models"""
        if self.X_train_scaled is None:
            self.preprocess_data()
        
        self.explainer = shap.TreeExplainer(self.model)
        logger.info("Tree explainer created")
    
    def create_kernel_explainer(self):
        """Create SHAP kernel explainer (model-agnostic)"""
        if self.X_train_scaled is None:
            self.preprocess_data()
        
        def model_predict(x):
            return self.model.predict_proba(x) if hasattr(self.model, 'predict_proba') else self.model.predict(x)
        
        self.explainer = shap.KernelExplainer(model_predict, self.X_train_scaled[:100])
        logger.info("Kernel explainer created")
    
    def calculate_shap_values(self, X_test: pd.DataFrame = None, max_samples: int = 100):
        """
        Calculate SHAP values for test data
        
        Args:
            X_test: Test data (default: use training data)
            max_samples: Maximum number of samples to explain
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            self.create_tree_explainer()
        
        if X_test is None:
            X_to_explain = self.X_train_scaled[:max_samples]
        else:
            numerical_cols = X_test.select_dtypes(include=[np.number]).columns
            X_to_explain = self.scaler.transform(X_test[numerical_cols].fillna(0))
            X_to_explain = X_to_explain[:max_samples]
        
        self.shap_values = self.explainer.shap_values(X_to_explain)
        
        logger.info(f"SHAP values calculated for {len(X_to_explain)} samples")
        return self.shap_values

    def save(self, filepath: str):
        """Save this SHAP explainer instance to a pickle file."""
        joblib.dump(self, filepath)
        logger.info(f"SHAP explainer saved to {filepath}")
        return filepath
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get global feature importance from SHAP values
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.shap_values is None:
            self.calculate_shap_values()
        
        # Calculate mean absolute SHAP values
        if isinstance(self.shap_values, list):
            # For classification (multi-class)
            shap_abs = np.abs(self.shap_values[0])
        else:
            shap_abs = np.abs(self.shap_values)
        
        mean_shap = np.mean(shap_abs, axis=0)
        importance = dict(zip(self.feature_names, mean_shap))
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        self.explanation_report['feature_importance'] = importance
        return importance
    
    def plot_summary_plot(self):
        """
        Plot SHAP summary plot inline
        """
        if self.shap_values is None:
            self.calculate_shap_values()

        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, self.X_train_scaled[:100],
                          feature_names=self.feature_names, show=False)
        plt.tight_layout()
        plt.show()
        logger.info("SHAP summary plot displayed")
    
    def plot_bar_plot(self):
        """
        Plot SHAP bar plot (feature importance) inline
        """
        if self.shap_values is None:
            self.calculate_shap_values()

        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, self.X_train_scaled[:100],
                          feature_names=self.feature_names, plot_type='bar', show=False)
        plt.tight_layout()
        plt.show()
        logger.info("SHAP bar plot displayed")
    
    def plot_waterfall_plot(self, instance_idx: int = 0):
        """
        Plot SHAP waterfall plot for a single instance inline
        """
        if self.shap_values is None:
            self.calculate_shap_values()

        if isinstance(self.shap_values, list):
            shap_val = self.shap_values[0][instance_idx]
            expected_val = self.explainer.expected_value[0]
        else:
            shap_val = self.shap_values[instance_idx]
            expected_val = self.explainer.expected_value

        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap.Explanation(values=shap_val,
                                             base_values=expected_val,
                                             feature_names=self.feature_names), show=False)
        plt.tight_layout()
        plt.show()
        logger.info("SHAP waterfall plot displayed")
    
    def plot_force_plot(self, instance_idx: int = 0, save_path: str = None):
        """
        Plot SHAP force plot for a single instance inline or save if save_path is provided
        """
        if self.shap_values is None:
            self.calculate_shap_values()

        if isinstance(self.shap_values, list):
            shap_val = self.shap_values[0][instance_idx]
            expected_val = self.explainer.expected_value[0]
        else:
            shap_val = self.shap_values[instance_idx]
            expected_val = self.explainer.expected_value

        force = shap.force_plot(expected_val, shap_val, self.X_train_scaled[instance_idx],
                               feature_names=self.feature_names, matplotlib=False, show=False)
        if save_path is not None:
            try:
                if hasattr(force, 'save'):
                    force.save(save_path)
                else:
                    html = force.html() if hasattr(force, 'html') else str(force)
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(html)
            except Exception:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(str(force))
            logger.info(f"SHAP force plot saved to {save_path}")

        try:
            html = force.html() if hasattr(force, 'html') else str(force)
            display(HTML(html))
        except Exception:
            logger.exception('Failed to display force plot HTML inline')
    
    def plot_decision_plot(self, instance_idx: int = 0):
        """
        Plot SHAP decision plot for a single instance inline
        """
        if self.shap_values is None:
            self.calculate_shap_values()

        if isinstance(self.shap_values, list):
            shap_val = self.shap_values[0][instance_idx]
            expected_val = self.explainer.expected_value[0]
        else:
            shap_val = self.shap_values[instance_idx]
            expected_val = self.explainer.expected_value

        plt.figure(figsize=(10, 8))
        shap.decision_plot(expected_val, shap_val, self.X_train_scaled[instance_idx],
                           feature_names=self.feature_names, show=False)
        plt.tight_layout()
        plt.show()
        logger.info("SHAP decision plot displayed")
    
    def explain_instance(self, X_instance: pd.DataFrame) -> Dict:
        """
        Explain a single instance
        
        Args:
            X_instance: Instance to explain
            
        Returns:
            Dictionary with explanation details
        """
        if self.explainer is None:
            self.create_tree_explainer()
        
        numerical_cols = X_instance.select_dtypes(include=[np.number]).columns
        X_scaled = self.scaler.transform(X_instance[numerical_cols].fillna(0))
        
        shap_vals = self.explainer.shap_values(X_scaled)
        
        if isinstance(shap_vals, list):
            shap_val = shap_vals[0][0]
            expected_val = self.explainer.expected_value[0]
        else:
            shap_val = shap_vals[0]
            expected_val = self.explainer.expected_value
        
        # Create feature contribution dictionary
        contributions = {}
        for i, feat in enumerate(self.feature_names):
            contributions[feat] = {
                'shap_value': float(shap_val[i]),
                'impact': 'increases' if shap_val[i] > 0 else 'decreases'
            }
        
        explanation = {
            'base_value': float(expected_val),
            'prediction': float(expected_val + shap_val.sum()),
            'feature_contributions': contributions
        }
        
        return explanation
    
    def generate_explanation_report(self, X_test: pd.DataFrame = None) -> Dict:
        """
        Generate a comprehensive explanation report
        
        Args:
            X_test: Test data (optional)
            
        Returns:
            Dictionary with all explanation results
        """
        self.calculate_shap_values(X_test)
        feature_importance = self.get_feature_importance()
        
        self.explanation_report = {
            'feature_importance': feature_importance,
            'n_features': len(self.feature_names),
            'n_samples_explained': len(self.shap_values[0]) if isinstance(self.shap_values, list) else len(self.shap_values)
        }
        
        return self.explanation_report
    
    def get_explanation_report(self) -> Dict:
        """Get the explanation report"""
        return self.explanation_report
    
    def print_report(self):
        """Print a formatted explanation report"""
        report = self.explanation_report
        
        print("=" * 80)
        print("EXPLAINABLE AI REPORT (SHAP)")
        print("=" * 80)
        
        print(f"\nNumber of features: {report['n_features']}")
        print(f"Number of samples explained: {report['n_samples_explained']}")
        
        print("\n" + "-" * 80)
        print("FEATURE IMPORTANCE (Global)")
        print("-" * 80)
        for i, (feat, imp) in enumerate(report['feature_importance'].items()):
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
    explainer.plot_summary_plot()
    explainer.plot_bar_plot()
    explainer.plot_waterfall_plot()


if __name__ == "__main__":
    main()

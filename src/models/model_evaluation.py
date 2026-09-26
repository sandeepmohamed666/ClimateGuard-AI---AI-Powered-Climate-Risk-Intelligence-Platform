"""
Model Evaluation Module
Comprehensive evaluation metrics for ML models
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, mean_squared_error,
    mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Optional, Tuple
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, y_true, y_pred, y_pred_proba=None):
        """
        Initialize the model evaluator
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_pred_proba = np.array(y_pred_proba) if y_pred_proba is not None else None
        self.evaluation_report = {}
    
    def calculate_classification_metrics(self) -> Dict:
        """
        Calculate classification metrics
        
        Returns:
            Dictionary with classification metrics
        """
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            'recall': recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        }
        
        # Binary classification metrics
        if len(np.unique(self.y_true)) == 2:
            metrics.update({
                'precision_binary': precision_score(self.y_true, self.y_pred, average='binary', zero_division=0),
                'recall_binary': recall_score(self.y_true, self.y_pred, average='binary', zero_division=0),
                'f1_binary': f1_score(self.y_true, self.y_pred, average='binary', zero_division=0)
            })
        
        # ROC AUC if probabilities available
        if self.y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_pred_proba)
            except:
                metrics['roc_auc'] = None
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(self.y_true, self.y_pred).tolist()
        
        self.evaluation_report['classification'] = metrics
        return metrics
    
    def calculate_regression_metrics(self) -> Dict:
        """
        Calculate regression metrics
        
        Returns:
            Dictionary with regression metrics
        """
        metrics = {
            'mse': mean_squared_error(self.y_true, self.y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_true, self.y_pred)),
            'mae': mean_absolute_error(self.y_true, self.y_pred),
            'r2_score': r2_score(self.y_true, self.y_pred)
        }
        
        self.evaluation_report['regression'] = metrics
        return metrics
    
    def plot_confusion_matrix(self, save_path: str = None):
        """
        Plot confusion matrix
        
        Args:
            save_path: Path to save the plot (deprecated, plots are displayed inline)
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        logger.info("Confusion matrix displayed")
    
    def plot_roc_curve(self, save_path: str = None):
        """
        Plot ROC curve
        
        Args:
            save_path: Path to save the plot (deprecated, plots are displayed inline)
        """
        if self.y_pred_proba is None:
            logger.warning("Probabilities not available for ROC curve")
            return
        
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)
        auc = roc_auc_score(self.y_true, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        logger.info("ROC curve displayed")
    
    def plot_precision_recall_curve(self, save_path: str = None):
        """
        Plot Precision-Recall curve
        
        Args:
            save_path: Path to save the plot (deprecated, plots are displayed inline)
        """
        if self.y_pred_proba is None:
            logger.warning("Probabilities not available for PR curve")
            return
        
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        logger.info("PR curve displayed")
    
    def cross_validate(self, model, X, y, cv: int = 5, scoring: str = 'f1_weighted') -> Dict:
        """
        Perform cross-validation
        
        Args:
            model: ML model to evaluate
            X: Feature matrix
            y: Target vector
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with cross-validation results
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        cv_results = {
            'scores': scores.tolist(),
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max())
        }
        
        self.evaluation_report['cross_validation'] = cv_results
        logger.info(f"Cross-validation completed - Mean {scoring}: {cv_results['mean']:.4f}")
        return cv_results
    
    def generate_evaluation_report(self) -> Dict:
        """
        Generate comprehensive evaluation report
        
        Returns:
            Dictionary with all evaluation results
        """
        # Determine if classification or regression
        if len(np.unique(self.y_true)) <= 10:
            self.calculate_classification_metrics()
        else:
            self.calculate_regression_metrics()
        
        return self.evaluation_report
    
    def get_evaluation_report(self) -> Dict:
        """Get the evaluation report"""
        return self.evaluation_report
    
    def print_report(self):
        """Print a formatted evaluation report"""
        report = self.evaluation_report
        
        print("=" * 80)
        print("MODEL EVALUATION REPORT")
        print("=" * 80)
        
        if 'classification' in report:
            metrics = report['classification']
            print("\nCLASSIFICATION METRICS:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision (weighted): {metrics['precision']:.4f}")
            print(f"  Recall (weighted): {metrics['recall']:.4f}")
            print(f"  F1 Score (weighted): {metrics['f1_score']:.4f}")
            
            if 'precision_binary' in metrics:
                print(f"  Precision (binary): {metrics['precision_binary']:.4f}")
                print(f"  Recall (binary): {metrics['recall_binary']:.4f}")
                print(f"  F1 Score (binary): {metrics['f1_binary']:.4f}")
            
            if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
                print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
            
            print("\nCONFUSION MATRIX:")
            for row in metrics['confusion_matrix']:
                print(f"  {row}")
        
        if 'regression' in report:
            metrics = report['regression']
            print("\nREGRESSION METRICS:")
            print(f"  MSE: {metrics['mse']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  R² Score: {metrics['r2_score']:.4f}")
        
        if 'cross_validation' in report:
            cv = report['cross_validation']
            print("\nCROSS-VALIDATION:")
            print(f"  Mean Score: {cv['mean']:.4f}")
            print(f"  Std Score: {cv['std']:.4f}")
            print(f"  Min Score: {cv['min']:.4f}")
            print(f"  Max Score: {cv['max']:.4f}")
        
        print("\n" + "=" * 80)


class ClusteringEvaluator:
    """Evaluator for clustering models"""
    
    def __init__(self, X, labels):
        """
        Initialize the clustering evaluator
        
        Args:
            X: Feature matrix
            labels: Cluster labels
        """
        self.X = np.array(X)
        self.labels = np.array(labels)
        self.evaluation_report = {}
    
    def calculate_clustering_metrics(self) -> Dict:
        """
        Calculate clustering metrics
        
        Returns:
            Dictionary with clustering metrics
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        
        metrics = {
            'silhouette_score': silhouette_score(self.X, self.labels),
            'davies_bouldin_score': davies_bouldin_score(self.X, self.labels),
            'calinski_harabasz_score': calinski_harabasz_score(self.X, self.labels)
        }
        
        self.evaluation_report['clustering'] = metrics
        return metrics
    
    def generate_evaluation_report(self) -> Dict:
        """Generate comprehensive clustering evaluation report"""
        self.calculate_clustering_metrics()
        return self.evaluation_report
    
    def print_report(self):
        """Print a formatted clustering evaluation report"""
        report = self.evaluation_report
        
        print("=" * 80)
        print("CLUSTERING EVALUATION REPORT")
        print("=" * 80)
        
        if 'clustering' in report:
            metrics = report['clustering']
            print("\nCLUSTERING METRICS:")
            print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
            print(f"  Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")
            print(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
        
        print("\n" + "=" * 80)


def main():
    """Test the model evaluator"""
    # Classification test
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred = np.random.randint(0, 2, 1000)
    y_pred_proba = np.random.random(1000)
    
    evaluator = ModelEvaluator(y_true, y_pred, y_pred_proba)
    evaluator.generate_evaluation_report()
    evaluator.print_report()
    
    # Clustering test
    X = np.random.randn(1000, 5)
    labels = np.random.randint(0, 3, 1000)
    
    cluster_eval = ClusteringEvaluator(X, labels)
    cluster_eval.generate_evaluation_report()
    cluster_eval.print_report()


if __name__ == "__main__":
    main()

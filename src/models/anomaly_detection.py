"""
Climate Anomaly Detection Module
Detects weather anomalies using various algorithms
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClimateAnomalyDetector:
    """Detect climate anomalies using various algorithms"""
    
    def __init__(self, X: pd.DataFrame):
        """
        Initialize the anomaly detector
        
        Args:
            X: Feature DataFrame for anomaly detection
        """
        self.X = X.copy()
        self.scaler = StandardScaler()
        self.X_scaled = None
        self.anomaly_labels = None
        self.anomaly_scores = None
        self.anomaly_report = {}
    
    def preprocess_data(self):
        """Scale the data for anomaly detection"""
        numerical_cols = self.X.select_dtypes(include=[np.number]).columns
        self.X_scaled = self.scaler.fit_transform(self.X[numerical_cols].fillna(0))
        logger.info("Data preprocessed and scaled")
    
    def isolation_forest_detection(
        self, 
        contamination: float = 0.1, 
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Isolation Forest
        
        Args:
            contamination: Expected proportion of outliers
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (anomaly labels, anomaly scores)
        """
        if self.X_scaled is None:
            self.preprocess_data()
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.anomaly_labels = iso_forest.fit_predict(self.X_scaled)
        self.anomaly_scores = iso_forest.score_samples(self.X_scaled)
        
        n_anomalies = (self.anomaly_labels == -1).sum()
        
        self.anomaly_report['isolation_forest'] = {
            'contamination': contamination,
            'n_anomalies': int(n_anomalies),
            'anomaly_percentage': float(n_anomalies / len(self.X) * 100)
        }
        
        logger.info(f"Isolation Forest detected {n_anomalies} anomalies")
        return self.anomaly_labels, self.anomaly_scores
    
    def one_class_svm_detection(
        self, 
        nu: float = 0.1, 
        kernel: str = 'rbf'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using One-Class SVM
        
        Args:
            nu: Expected proportion of outliers
            kernel: Kernel type ('rbf', 'linear', 'poly')
            
        Returns:
            Tuple of (anomaly labels, anomaly scores)
        """
        if self.X_scaled is None:
            self.preprocess_data()
        
        svm = OneClassSVM(nu=nu, kernel=kernel)
        
        self.anomaly_labels = svm.fit_predict(self.X_scaled)
        self.anomaly_scores = svm.score_samples(self.X_scaled)
        
        n_anomalies = (self.anomaly_labels == -1).sum()
        
        self.anomaly_report['one_class_svm'] = {
            'nu': nu,
            'kernel': kernel,
            'n_anomalies': int(n_anomalies),
            'anomaly_percentage': float(n_anomalies / len(self.X) * 100)
        }
        
        logger.info(f"One-Class SVM detected {n_anomalies} anomalies")
        return self.anomaly_labels, self.anomaly_scores
    
    def local_outlier_factor_detection(
        self, 
        n_neighbors: int = 20,
        contamination: float = 0.1
    ) -> np.ndarray:
        """
        Detect anomalies using Local Outlier Factor
        
        Args:
            n_neighbors: Number of neighbors to use
            contamination: Expected proportion of outliers
            
        Returns:
            Anomaly labels
        """
        if self.X_scaled is None:
            self.preprocess_data()
        
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            n_jobs=-1
        )
        
        self.anomaly_labels = lof.fit_predict(self.X_scaled)
        self.anomaly_scores = lof.negative_outlier_factor_
        
        n_anomalies = (self.anomaly_labels == -1).sum()
        
        self.anomaly_report['local_outlier_factor'] = {
            'n_neighbors': n_neighbors,
            'contamination': contamination,
            'n_anomalies': int(n_anomalies),
            'anomaly_percentage': float(n_anomalies / len(self.X) * 100)
        }
        
        logger.info(f"LOF detected {n_anomalies} anomalies")
        return self.anomaly_labels
    
    def elliptic_envelope_detection(
        self, 
        contamination: float = 0.1,
        support_fraction: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Elliptic Envelope
        
        Args:
            contamination: Expected proportion of outliers
            support_fraction: Support fraction for robust covariance
            
        Returns:
            Tuple of (anomaly labels, anomaly scores)
        """
        if self.X_scaled is None:
            self.preprocess_data()
        
        envelope = EllipticEnvelope(
            contamination=contamination,
            support_fraction=support_fraction,
            random_state=42
        )
        
        self.anomaly_labels = envelope.fit_predict(self.X_scaled)
        self.anomaly_scores = envelope.score_samples(self.X_scaled)
        
        n_anomalies = (self.anomaly_labels == -1).sum()
        
        self.anomaly_report['elliptic_envelope'] = {
            'contamination': contamination,
            'support_fraction': support_fraction,
            'n_anomalies': int(n_anomalies),
            'anomaly_percentage': float(n_anomalies / len(self.X) * 100)
        }
        
        logger.info(f"Elliptic Envelope detected {n_anomalies} anomalies")
        return self.anomaly_labels, self.anomaly_scores
    
    def ensemble_detection(
        self, 
        methods: List[str] = None,
        voting: str = 'majority'
    ) -> np.ndarray:
        """
        Ensemble anomaly detection using multiple methods
        
        Args:
            methods: List of methods to use
            voting: Voting method ('majority', 'unanimous', 'any')
            
        Returns:
            Ensemble anomaly labels
        """
        if methods is None:
            methods = ['isolation_forest', 'one_class_svm', 'elliptic_envelope']
        
        all_predictions = []
        
        for method in methods:
            if method == 'isolation_forest':
                labels, _ = self.isolation_forest_detection()
            elif method == 'one_class_svm':
                labels, _ = self.one_class_svm_detection()
            elif method == 'local_outlier_factor':
                labels = self.local_outlier_factor_detection()
            elif method == 'elliptic_envelope':
                labels, _ = self.elliptic_envelope_detection()
            else:
                continue
            
            all_predictions.append(labels)
        
        # Convert to anomaly indicators (1 for anomaly, 0 for normal)
        anomaly_votes = np.array([(pred == -1).astype(int) for pred in all_predictions])
        
        # Voting
        if voting == 'majority':
            ensemble_labels = np.where(anomaly_votes.sum(axis=0) > len(methods) / 2, -1, 1)
        elif voting == 'unanimous':
            ensemble_labels = np.where(anomaly_votes.sum(axis=0) == len(methods), -1, 1)
        elif voting == 'any':
            ensemble_labels = np.where(anomaly_votes.sum(axis=0) > 0, -1, 1)
        else:
            ensemble_labels = np.where(anomaly_votes.sum(axis=0) > len(methods) / 2, -1, 1)
        
        self.anomaly_labels = ensemble_labels
        n_anomalies = (ensemble_labels == -1).sum()
        
        self.anomaly_report['ensemble'] = {
            'methods': methods,
            'voting': voting,
            'n_anomalies': int(n_anomalies),
            'anomaly_percentage': float(n_anomalies / len(self.X) * 100)
        }
        
        logger.info(f"Ensemble detected {n_anomalies} anomalies")
        return ensemble_labels
    
    def get_anomaly_details(self) -> pd.DataFrame:
        """
        Get detailed information about detected anomalies
        
        Returns:
            DataFrame with anomaly details
        """
        if self.anomaly_labels is None:
            logger.warning("No anomaly detection performed yet")
            return pd.DataFrame()
        
        df_with_anomalies = self.X.copy()
        df_with_anomalies['is_anomaly'] = (self.anomaly_labels == -1).astype(int)
        
        if self.anomaly_scores is not None:
            df_with_anomalies['anomaly_score'] = self.anomaly_scores
        
        return df_with_anomalies
    
    def get_anomaly_report(self) -> Dict:
        """Get the anomaly detection report"""
        return self.anomaly_report
    
    def print_report(self):
        """Print a formatted anomaly detection report"""
        report = self.anomaly_report
        
        print("=" * 80)
        print("CLIMATE ANOMALY DETECTION REPORT")
        print("=" * 80)
        
        for method, results in report.items():
            print(f"\n{method.upper()}:")
            print(f"  Number of anomalies: {results['n_anomalies']}")
            print(f"  Anomaly percentage: {results['anomaly_percentage']:.2f}%")
            
            if 'contamination' in results:
                print(f"  Contamination parameter: {results['contamination']}")
            if 'nu' in results:
                print(f"  Nu parameter: {results['nu']}")
        
        print("\n" + "=" * 80)


def main():
    """Test the anomaly detector"""
    # Create sample data with some anomalies
    np.random.seed(42)
    n_samples = 1000
    
    # Normal data
    normal_data = np.random.normal(0, 1, (n_samples - 50, 5))
    
    # Anomalous data
    anomaly_data = np.random.normal(5, 1, (50, 5))
    
    data = np.vstack([normal_data, anomaly_data])
    
    X = pd.DataFrame(data, columns=['temp', 'humidity', 'pressure', 'wind', 'rain'])
    
    detector = ClimateAnomalyDetector(X)
    
    # Test different methods
    detector.isolation_forest_detection()
    detector.one_class_svm_detection()
    detector.elliptic_envelope_detection()
    
    # Ensemble
    detector.ensemble_detection()
    
    detector.print_report()
    
    # Get anomaly details
    anomaly_details = detector.get_anomaly_details()
    print(f"\nAnomalies detected: {anomaly_details['is_anomaly'].sum()}")


if __name__ == "__main__":
    main()

"""
Climate Profiling Module
Clustering algorithms for climate zone classification
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClimateProfiler:
    """Climate profiling using clustering algorithms"""
    
    def __init__(self, X: pd.DataFrame):
        """
        Initialize the climate profiler
        
        Args:
            X: Feature DataFrame for clustering
        """
        self.X = X.copy()
        self.scaler = StandardScaler()
        self.X_scaled = None
        self.labels = None
        self.cluster_report = {}
    
    def preprocess_data(self):
        """Scale the data for clustering"""
        numerical_cols = self.X.select_dtypes(include=[np.number]).columns
        self.X_scaled = self.scaler.fit_transform(self.X[numerical_cols].fillna(0))
        logger.info("Data preprocessed and scaled")
    
    def kmeans_clustering(self, n_clusters: int = 5, random_state: int = 42) -> np.ndarray:
        """
        Perform K-Means clustering
        
        Args:
            n_clusters: Number of clusters
            random_state: Random state for reproducibility
            
        Returns:
            Cluster labels
        """
        if self.X_scaled is None:
            self.preprocess_data()
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.labels = kmeans.fit_predict(self.X_scaled)
        
        # Calculate metrics
        silhouette = silhouette_score(self.X_scaled, self.labels)
        db_score = davies_bouldin_score(self.X_scaled, self.labels)
        ch_score = calinski_harabasz_score(self.X_scaled, self.labels)
        
        self.cluster_report['kmeans'] = {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'davies_bouldin_score': db_score,
            'calinski_harabasz_score': ch_score,
            'cluster_centers': kmeans.cluster_centers_.tolist()
        }
        
        logger.info(f"K-Means clustering completed with {n_clusters} clusters")
        return self.labels
    
    def dbscan_clustering(self, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        """
        Perform DBSCAN clustering
        
        Args:
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood
            
        Returns:
            Cluster labels
        """
        if self.X_scaled is None:
            self.preprocess_data()
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels = dbscan.fit_predict(self.X_scaled)
        
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        
        # Calculate metrics (only if more than 1 cluster)
        if n_clusters > 1:
            mask = self.labels != -1
            if mask.sum() > 0:
                silhouette = silhouette_score(self.X_scaled[mask], self.labels[mask])
                db_score = davies_bouldin_score(self.X_scaled[mask], self.labels[mask])
                ch_score = calinski_harabasz_score(self.X_scaled[mask], self.labels[mask])
            else:
                silhouette = db_score = ch_score = None
        else:
            silhouette = db_score = ch_score = None
        
        self.cluster_report['dbscan'] = {
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': list(self.labels).count(-1),
            'silhouette_score': silhouette,
            'davies_bouldin_score': db_score,
            'calinski_harabasz_score': ch_score
        }
        
        logger.info(f"DBSCAN clustering completed with {n_clusters} clusters")
        return self.labels
    
    def gaussian_mixture_clustering(self, n_components: int = 5, random_state: int = 42) -> np.ndarray:
        """
        Perform Gaussian Mixture clustering
        
        Args:
            n_components: Number of mixture components
            random_state: Random state for reproducibility
            
        Returns:
            Cluster labels
        """
        if self.X_scaled is None:
            self.preprocess_data()
        
        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        self.labels = gmm.fit_predict(self.X_scaled)
        
        # Calculate metrics
        silhouette = silhouette_score(self.X_scaled, self.labels)
        db_score = davies_bouldin_score(self.X_scaled, self.labels)
        ch_score = calinski_harabasz_score(self.X_scaled, self.labels)
        
        self.cluster_report['gaussian_mixture'] = {
            'n_components': n_components,
            'silhouette_score': silhouette,
            'davies_bouldin_score': db_score,
            'calinski_harabasz_score': ch_score,
            'aic': gmm.aic_,
            'bic': gmm.bic_
        }
        
        logger.info(f"Gaussian Mixture clustering completed with {n_components} components")
        return self.labels
    
    def agglomerative_clustering(self, n_clusters: int = 5, linkage: str = 'ward') -> np.ndarray:
        """
        Perform Agglomerative clustering
        
        Args:
            n_clusters: Number of clusters
            linkage: Linkage method ('ward', 'complete', 'average', 'single')
            
        Returns:
            Cluster labels
        """
        if self.X_scaled is None:
            self.preprocess_data()
        
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.labels = agg.fit_predict(self.X_scaled)
        
        # Calculate metrics
        silhouette = silhouette_score(self.X_scaled, self.labels)
        db_score = davies_bouldin_score(self.X_scaled, self.labels)
        ch_score = calinski_harabasz_score(self.X_scaled, self.labels)
        
        self.cluster_report['agglomerative'] = {
            'n_clusters': n_clusters,
            'linkage': linkage,
            'silhouette_score': silhouette,
            'davies_bouldin_score': db_score,
            'calinski_harabasz_score': ch_score
        }
        
        logger.info(f"Agglomerative clustering completed with {n_clusters} clusters")
        return self.labels
    
    def find_optimal_clusters(self, max_clusters: int = 10, method: str = 'kmeans', 
                             sample_size: int = 10000, use_sampling: bool = True) -> Dict:
        """
        Find optimal number of clusters using elbow method and silhouette analysis
        
        Args:
            max_clusters: Maximum number of clusters to test
            method: Clustering method to use
            sample_size: Number of samples to use for faster computation (if use_sampling=True)
            use_sampling: Whether to use sampling for faster computation
            
        Returns:
            Dictionary with optimal cluster information
        """
        if self.X_scaled is None:
            self.preprocess_data()
        
        # Use sampling for faster computation on large datasets
        if use_sampling and len(self.X_scaled) > sample_size:
            logger.info(f"Using {sample_size} samples from {len(self.X_scaled)} for faster computation")
            indices = np.random.choice(len(self.X_scaled), sample_size, replace=False)
            X_sample = self.X_scaled[indices]
        else:
            X_sample = self.X_scaled
        
        silhouette_scores = []
        inertias = []
        cluster_range = range(2, max_clusters + 1)
        
        for n_clusters in cluster_range:
            logger.info(f"Testing {n_clusters} clusters...")
            
            if method == 'kmeans':
                # Use fewer n_init and max_iter for faster computation
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5, max_iter=300)
                labels = kmeans.fit_predict(X_sample)
                inertias.append(kmeans.inertia_)
            elif method == 'gaussian_mixture':
                gmm = GaussianMixture(n_components=n_clusters, random_state=42, max_iter=100)
                labels = gmm.fit_predict(X_sample)
            else:
                continue
            
            if len(set(labels)) > 1:
                silhouette = silhouette_score(X_sample, labels)
                silhouette_scores.append(silhouette)
                logger.info(f"  Silhouette score: {silhouette:.4f}")
            else:
                silhouette_scores.append(0)
                logger.info(f"  Silhouette score: 0 (single cluster)")
        
        # Find optimal clusters (max silhouette)
        optimal_n = cluster_range[np.argmax(silhouette_scores)]
        
        self.cluster_report['optimal_clusters'] = {
            'method': method,
            'max_clusters_tested': max_clusters,
            'sample_size_used': len(X_sample),
            'silhouette_scores': dict(zip(cluster_range, silhouette_scores)),
            'inertias': dict(zip(cluster_range, inertias)) if inertias else None,
            'optimal_n_clusters': optimal_n,
            'max_silhouette': max(silhouette_scores)
        }
        
        logger.info(f"Optimal number of clusters: {optimal_n}")
        return self.cluster_report['optimal_clusters']
    
    def get_cluster_characteristics(self) -> pd.DataFrame:
        """
        Get characteristics of each cluster
        
        Returns:
            DataFrame with cluster statistics
        """
        if self.labels is None:
            logger.warning("No clustering results available")
            return pd.DataFrame()
        
        df_with_clusters = self.X.copy()
        df_with_clusters['cluster'] = self.labels
        
        cluster_stats = df_with_clusters.groupby('cluster').agg(['mean', 'std'])
        
        return cluster_stats
    
    def get_cluster_report(self) -> Dict:
        """Get the clustering report"""
        return self.cluster_report
    
    def print_report(self):
        """Print a formatted clustering report"""
        report = self.cluster_report
        
        print("=" * 80)
        print("CLIMATE PROFILING REPORT")
        print("=" * 80)
        
        for method, results in report.items():
            print(f"\n{method.upper()}:")
            if 'n_clusters' in results or 'n_components' in results:
                n = results.get('n_clusters', results.get('n_components', 'N/A'))
                print(f"  Number of clusters: {n}")
            
            if 'silhouette_score' in results:
                print(f"  Silhouette Score: {results['silhouette_score']:.4f}")
            if 'davies_bouldin_score' in results:
                print(f"  Davies-Bouldin Score: {results['davies_bouldin_score']:.4f}")
            if 'calinski_harabasz_score' in results:
                print(f"  Calinski-Harabasz Score: {results['calinski_harabasz_score']:.2f}")
            
            if 'n_noise' in results:
                print(f"  Noise points: {results['n_noise']}")
        
        print("\n" + "=" * 80)


def main():
    """Test the climate profiler"""
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
    
    profiler = ClimateProfiler(X)
    
    # Test K-Means
    print("K-Means Clustering:")
    profiler.kmeans_clustering(n_clusters=5)
    
    # Test DBSCAN
    print("\nDBSCAN Clustering:")
    profiler.dbscan_clustering(eps=0.5, min_samples=5)
    
    # Find optimal clusters
    print("\nFinding Optimal Clusters:")
    profiler.find_optimal_clusters(max_clusters=10)
    
    profiler.print_report()


if __name__ == "__main__":
    main()

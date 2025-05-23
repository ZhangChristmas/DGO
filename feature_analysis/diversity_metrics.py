# ultimate_morph_generator/feature_analysis/diversity_metrics.py
import numpy as np
from sklearn.neighbors import KernelDensity  # For KDE based density estimation
from sklearn.cluster import KMeans  # For cluster-based diversity
from scipy.spatial.distance import pdist, squareform  # For pairwise distances
from typing import List, Optional, Dict, Tuple

from ..config import get_config, FeatureAnalysisConfig
from ..utilities.type_definitions import FeatureVector
from ..utilities.logging_config import setup_logging

logger = setup_logging()


class LibraryDiversityAnalyzer:
    """
    Analyzes the diversity of the morphology library using DGO feature vectors.
    Methods can include:
    - Average pairwise distance in feature space.
    - Coverage of the feature space (e.g., using KDE or clustering).
    - Entropy-based measures.
    """

    def __init__(self, feature_dim: int, diversity_cfg: FeatureAnalysisConfig):
        self.feature_dim = feature_dim
        self.cfg = diversity_cfg
        self.kde_estimator: Optional[KernelDensity] = None

        if self.cfg.diversity_strategy == "explore_low_density" and \
                self.cfg.feature_space_density_estimator == "kde":
            self.kde_estimator = KernelDensity(bandwidth=self.cfg.kde_bandwidth, kernel='gaussian')
            logger.info(
                f"KDE estimator initialized for density-based diversity with bandwidth {self.cfg.kde_bandwidth}.")

    def get_average_pairwise_distance(self, feature_vectors: List[FeatureVector],
                                      metric: str = 'euclidean') -> Optional[float]:
        """
        Computes the average pairwise distance between feature vectors.
        A higher average distance suggests greater diversity.
        Args:
            feature_vectors: A list of 1D NumPy arrays.
            metric: Distance metric (e.g., 'euclidean', 'cosine', 'cityblock').
        """
        if not feature_vectors or len(feature_vectors) < 2:
            return 0.0  # Or None if no diversity can be computed

        try:
            feature_matrix = np.array(feature_vectors)  # (N_samples, feature_dim)
            if feature_matrix.ndim != 2 or feature_matrix.shape[1] != self.feature_dim:
                logger.error(f"Invalid feature matrix shape for pairwise distance: {feature_matrix.shape}")
                return None

            distances = pdist(feature_matrix, metric=metric)  # Condensed distance matrix (1D)
            return float(np.mean(distances)) if distances.size > 0 else 0.0
        except Exception as e:
            logger.error(f"Error computing average pairwise distance: {e}")
            return None

    def estimate_feature_space_density_kde(self, feature_vectors: List[FeatureVector],
                                           query_points: Optional[List[FeatureVector]] = None
                                           ) -> Optional[Union[np.ndarray, float]]:
        """
        Estimates density in the feature space using Kernel Density Estimation.
        If query_points are provided, returns density at these points.
        Otherwise, fits KDE to feature_vectors and could return average log-density or the fitted estimator.
        Higher density regions are well-explored; lower density regions are candidates for exploration.
        """
        if not self.kde_estimator:
            logger.warning("KDE estimator not initialized. Cannot estimate density.")
            return None
        if not feature_vectors:
            logger.warning("No feature vectors provided for KDE.")
            return None

        try:
            feature_matrix = np.array(feature_vectors)
            if feature_matrix.ndim != 2 or feature_matrix.shape[1] != self.feature_dim:
                logger.error(f"Invalid feature matrix shape for KDE: {feature_matrix.shape}")
                return None

            self.kde_estimator.fit(feature_matrix)  # Fit KDE to the current library

            if query_points:
                query_matrix = np.array(query_points)
                if query_matrix.ndim != 2 or query_matrix.shape[1] != self.feature_dim:
                    logger.error(f"Invalid query points shape for KDE: {query_matrix.shape}")
                    return None
                # score_samples returns log-density (log P(x))
                log_densities = self.kde_estimator.score_samples(query_matrix)
                return np.exp(log_densities)  # Convert log-density to density P(x)
            else:
                # If no query points, can return average log-density of the fitted data as a general measure
                avg_log_density = self.kde_estimator.score(feature_matrix)  # Average log-likelihood
                return float(avg_log_density)
        except Exception as e:
            logger.error(f"Error during KDE density estimation: {e}")
            return None

    def get_feature_space_coverage_kmeans(self, feature_vectors: List[FeatureVector],
                                          num_clusters_ratio: float = 0.1,  # Ratio of N_samples to use as K
                                          min_clusters: int = 5, max_clusters: int = 50
                                          ) -> Optional[Tuple[int, float]]:
        """
        Estimates feature space coverage using K-Means clustering.
        Returns:
            - Number of non-empty clusters found.
            - Silhouette score (if computable, measures cluster separation and cohesion).
        More clusters for a given K, or better silhouette score, might indicate better coverage/structure.
        """
        if not feature_vectors or len(feature_vectors) < min_clusters:  # Need enough samples for clustering
            logger.warning(
                f"Not enough samples ({len(feature_vectors)}) for K-Means coverage analysis (min: {min_clusters}).")
            return None

        try:
            feature_matrix = np.array(feature_vectors)
            if feature_matrix.ndim != 2 or feature_matrix.shape[1] != self.feature_dim:
                logger.error(f"Invalid feature matrix shape for K-Means: {feature_matrix.shape}")
                return None

            n_samples = feature_matrix.shape[0]
            k = max(min_clusters, min(max_clusters, int(n_samples * num_clusters_ratio)))

            if n_samples <= k:  # K must be less than N_samples for meaningful silhouette
                k = max(2, n_samples - 1) if n_samples > 1 else 1  # Adjust k
                if k == 1 and n_samples > 1:  # Cannot compute silhouette for k=1
                    logger.warning(f"Adjusted K to {k} for K-Means, silhouette score will not be computed.")

            if k < 2:  # Cannot run K-Means or compute silhouette with k < 2
                logger.warning(f"K-Means requires at least 2 clusters, requested/derived K is {k}. Skipping.")
                return (0, 0.0) if k == 0 else (1, 0.0)  # (num_active_clusters, silhouette_score)

            kmeans = KMeans(n_clusters=k, random_state=get_config().random_seed, n_init='auto')
            cluster_labels = kmeans.fit_predict(feature_matrix)

            num_active_clusters = len(np.unique(cluster_labels))

            silhouette_avg: float = 0.0
            if num_active_clusters >= 2 and num_active_clusters < n_samples:  # Silhouette needs 2 <= n_labels < n_samples
                from sklearn.metrics import silhouette_score  # Import here as it's specific
                try:
                    silhouette_avg = silhouette_score(feature_matrix, cluster_labels)
                except ValueError as sve:  # Can happen if a cluster is too small
                    logger.warning(f"Could not compute silhouette score: {sve}")
                    silhouette_avg = -1.0  # Indicate failure

            return num_active_clusters, float(silhouette_avg)

        except Exception as e:
            logger.error(f"Error during K-Means coverage analysis: {e}")
            return None

    def get_convex_hull_volume_proxy(self, feature_vectors: List[FeatureVector]) -> Optional[float]:
        """
        Approximates the volume of the convex hull of the feature vectors.
        This is computationally expensive in high dimensions.
        Consider PCA to reduce dimensionality first if feature_dim is high.
        Requires scipy.spatial.ConvexHull.
        """
        if not feature_vectors or len(feature_vectors) < self.feature_dim + 1:
            # Need at least D+1 points for a D-dimensional hull
            logger.warning("Not enough unique points to compute convex hull volume.")
            return 0.0

        try:
            from scipy.spatial import ConvexHull  # Import heavy module locally
            feature_matrix = np.array(feature_vectors)
            if feature_matrix.ndim != 2 or feature_matrix.shape[1] != self.feature_dim:
                logger.error(f"Invalid feature matrix shape for Convex Hull: {feature_matrix.shape}")
                return None

            # PCA for dimensionality reduction if feature_dim is too high (e.g., > 10-15)
            # For now, assume feature_dim is manageable.

            # ConvexHull can fail if points are coplanar/collinear (QhullError)
            # Add small noise to potentially mitigate this, or handle QhullError.
            # feature_matrix_perturbed = feature_matrix + 1e-6 * np.random.randn(*feature_matrix.shape)

            hull = ConvexHull(feature_matrix)  # Can throw QhullError
            return float(hull.volume)  # 'volume' for 3D+, 'area' for 2D
        except ImportError:
            logger.warning("SciPy not fully installed or ConvexHull unavailable. Cannot compute hull volume.")
            return None
        except Exception as e:  # Catches QhullError specifically if not caught by scipy
            logger.error(f"Error computing convex hull volume: {e}. Points might be degenerate.")
            return None


if __name__ == "__main__":
    # --- Test LibraryDiversityAnalyzer ---
    from ..config import SystemConfig

    test_feature_dim_div = 32  # Use a smaller dim for some tests like hull

    temp_sys_cfg_data_div = {
        "feature_analysis": {
            "diversity_strategy": "explore_low_density",  # For KDE init
            "feature_space_density_estimator": "kde",
            "kde_bandwidth": 0.5,
            # Other diversity params not directly used by analyzer init
        },
        "logging": {"level": "DEBUG"},
        "random_seed": 123  # For K-Means reproducibility
    }
    from ..config import _config_instance

    _config_instance = SystemConfig.model_validate(temp_sys_cfg_data_div)  # Update global
    cfg_diversity = get_config().feature_analysis

    analyzer = LibraryDiversityAnalyzer(feature_dim=test_feature_dim_div, diversity_cfg=cfg_diversity)

    # Create dummy feature vectors
    num_samples = 50
    # Group 1 (more dense)
    fv_group1 = np.random.randn(num_samples // 2, test_feature_dim_div) * 0.5 + np.array(
        [1, 1] + [0] * (test_feature_dim_div - 2))  # Centered around [1,1,0...]
    # Group 2 (more sparse, further away)
    fv_group2 = np.random.randn(num_samples // 2, test_feature_dim_div) * 1.0 + np.array(
        [-1, -1] + [0] * (test_feature_dim_div - 2))  # Centered around [-1,-1,0...]

    all_fvs: List[FeatureVector] = [FeatureVector(fv) for fv in np.vstack((fv_group1, fv_group2))]

    # Test average pairwise distance
    print("\n--- Testing Average Pairwise Distance ---")
    avg_dist = analyzer.get_average_pairwise_distance(all_fvs)
    print(f"Average pairwise Euclidean distance: {avg_dist:.4f}")
    assert avg_dist is not None and avg_dist > 0

    # Test KDE density estimation
    print("\n--- Testing KDE Density Estimation ---")
    # Query density at a point within group1 (should be higher) and one far away (lower)
    query_pt_dense = FeatureVector(np.array([0.9, 0.9] + [0] * (test_feature_dim_div - 2)))
    query_pt_sparse = FeatureVector(np.array([5.0, 5.0] + [0] * (test_feature_dim_div - 2)))
    densities = analyzer.estimate_feature_space_density_kde(all_fvs, query_points=[query_pt_dense, query_pt_sparse])
    if densities is not None:
        print(f"Densities at query points: P(dense_pt)={densities[0]:.4e}, P(sparse_pt)={densities[1]:.4e}")
        assert densities[0] > densities[1]  # Expect point in cluster to have higher density
    avg_log_density_fit = analyzer.estimate_feature_space_density_kde(all_fvs)  # Fits again
    print(f"Average log-density of the fitted data: {avg_log_density_fit:.4f}")

    # Test K-Means coverage
    print("\n--- Testing K-Means Coverage ---")
    # Use a smaller K for this test size
    coverage_res = analyzer.get_feature_space_coverage_kmeans(all_fvs, num_clusters_ratio=0.05, min_clusters=2,
                                                              max_clusters=5)
    if coverage_res:
        num_clusters, sil_score = coverage_res
        print(f"K-Means: Found {num_clusters} active clusters. Silhouette Score: {sil_score:.4f}")
        assert num_clusters >= 1  # Should find at least 1, ideally 2 or more for these separated groups

    # Test Convex Hull Volume (Proxy) - can be slow or fail in high dims
    # For this test, reduce dimensionality of features for ConvexHull if test_feature_dim_div is high.
    # For dim=32, ConvexHull is likely too slow / memory intensive.
    # Let's test with first 3 dimensions for concept.
    print("\n--- Testing Convex Hull Volume (Proxy on 3D subspace) ---")
    if test_feature_dim_div >= 3:
        fvs_3d = [FeatureVector(fv[:3]) for fv in all_fvs]
        # Create a new analyzer instance for 3D features for this specific test
        analyzer_3d = LibraryDiversityAnalyzer(feature_dim=3, diversity_cfg=cfg_diversity)
        hull_vol_3d = analyzer_3d.get_convex_hull_volume_proxy(fvs_3d)
        if hull_vol_3d is not None:
            print(f"Convex Hull Volume (3D subspace): {hull_vol_3d:.4f}")
            assert hull_vol_3d >= 0  # Volume can be 0 if points are degenerate
        else:
            print("Convex Hull Volume (3D subspace) could not be computed (SciPy missing or QhullError).")
    else:
        print("Skipping Convex Hull test as feature_dim < 3.")

    print("\nLibraryDiversityAnalyzer tests completed.")
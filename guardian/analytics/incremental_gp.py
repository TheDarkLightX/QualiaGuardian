"""
Incremental Gaussian Process Learning for Test Quality Prediction

This module implements an incremental sparse Gaussian Process learner with:
- Core GP functionality with RBF and Matern kernels
- Sparse GP with inducing points for scalability
- Incremental updates without full retraining
- Memory management with compression
- Hyperparameter optimization
- Active learning with acquisition functions
- Integration interfaces for Guardian

Author: DarkLightX/Dana Edwards
"""

import numpy as np
import time
import psutil
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Protocol
from scipy.special import gamma, kv
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.linalg import cho_solve, cho_factor, LinAlgError
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)


# Interfaces (Dependency Inversion Principle)
class IKernel(Protocol):
    """Interface for GP kernels."""
    
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel value between two points."""
        ...
    
    def compute_matrix(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute kernel matrix."""
        ...
    
    def compute_with_gradients(self, x1: np.ndarray, x2: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Compute kernel value and gradients w.r.t hyperparameters."""
        ...
    
    def get_hyperparameters(self) -> Dict[str, float]:
        """Get current hyperparameters."""
        ...
    
    def set_hyperparameters(self, params: Dict[str, float]) -> None:
        """Set hyperparameters."""
        ...


class IGPModel(Protocol):
    """Interface for GP models."""
    
    def add_observation(self, observation: 'TestObservation') -> None:
        """Add a new observation."""
        ...
    
    def fit(self) -> None:
        """Fit the model to observations."""
        ...
    
    def predict(self, test_ids: List[str], features: np.ndarray) -> 'PredictionResult':
        """Make prediction for new point."""
        ...
    
    def incremental_update(self, observation: 'TestObservation') -> None:
        """Incrementally update model with new observation."""
        ...


class IInducingPointStrategy(Protocol):
    """Interface for inducing point selection strategies."""
    
    def select(self, X: np.ndarray, n_inducing: int, **kwargs) -> np.ndarray:
        """Select inducing points from data."""
        ...
    
    def update(self, X: np.ndarray, current_indices: np.ndarray, n_inducing: int) -> np.ndarray:
        """Update inducing points with new data."""
        ...


class IMemoryStrategy(Protocol):
    """Interface for memory management strategies."""
    
    def add(self, observation: 'TestObservation', importance: float = 1.0) -> None:
        """Add observation to memory."""
        ...
    
    def can_add(self, observation: 'TestObservation') -> bool:
        """Check if observation can be added."""
        ...
    
    def compress_old_observations(self, age_threshold: int) -> int:
        """Compress old observations."""
        ...
    
    def get_retained_observations(self) -> List['TestObservation']:
        """Get retained observations."""
        ...


class IHyperparameterStrategy(Protocol):
    """Interface for hyperparameter optimization strategies."""
    
    def optimize(self, model: IGPModel) -> Dict[str, float]:
        """Optimize hyperparameters."""
        ...
    
    def should_optimize(self, model: IGPModel, step: int) -> bool:
        """Check if optimization is needed."""
        ...


# Data structures
@dataclass
class TestObservation:
    """Observation of test execution results."""
    test_ids: List[str]
    features: Optional[np.ndarray]
    quality_score: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def extract_features(self) -> None:
        """Extract features from test IDs if not provided."""
        if self.features is None:
            # Simple feature extraction based on test IDs
            n_tests = len(self.test_ids)
            hash_features = [hash(tid) % 1000 / 1000.0 for tid in self.test_ids[:5]]
            
            # Pad or truncate to fixed size
            while len(hash_features) < 5:
                hash_features.append(0.0)
            
            self.features = np.array(hash_features[:5])


@dataclass
class PredictionResult:
    """Result of GP prediction."""
    mean: float
    std_dev: float
    confidence_interval: Tuple[float, float]
    test_ids: List[str]
    acquisition_value: Optional[float] = None


# Kernel implementations (Single Responsibility Principle)
class GPKernel(ABC):
    """Abstract base class for GP kernels."""
    
    @abstractmethod
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel value between two points."""
        pass
    
    @abstractmethod
    def compute_matrix(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute kernel matrix."""
        pass
    
    @abstractmethod
    def compute_with_gradients(self, x1: np.ndarray, x2: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Compute kernel value and gradients."""
        pass
    
    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, float]:
        """Get current hyperparameters."""
        pass
    
    @abstractmethod
    def set_hyperparameters(self, params: Dict[str, float]) -> None:
        """Set hyperparameters."""
        pass


class RBFKernel(GPKernel):
    """Radial Basis Function (Gaussian) kernel."""
    
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.length_scale = length_scale
        self.variance = variance
    
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute RBF kernel value."""
        dist_sq = np.sum((x1 - x2) ** 2)
        return self.variance * np.exp(-0.5 * dist_sq / (self.length_scale ** 2))
    
    def compute_matrix(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute RBF kernel matrix efficiently."""
        if Y is None:
            Y = X
        
        # Compute pairwise squared distances
        dist_sq = cdist(X, Y, 'sqeuclidean')
        
        # Apply RBF kernel
        K = self.variance * np.exp(-0.5 * dist_sq / (self.length_scale ** 2))
        
        # Add small jitter for numerical stability
        if X is Y or np.array_equal(X, Y):
            K += 1e-6 * np.eye(K.shape[0])
        
        return K
    
    def compute_with_gradients(self, x1: np.ndarray, x2: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Compute kernel value and gradients w.r.t hyperparameters."""
        dist_sq = np.sum((x1 - x2) ** 2)
        k_val = self.variance * np.exp(-0.5 * dist_sq / (self.length_scale ** 2))
        
        gradients = {
            'variance': k_val / self.variance,
            'length_scale': k_val * dist_sq / (self.length_scale ** 3)
        }
        
        return k_val, gradients
    
    def get_hyperparameters(self) -> Dict[str, float]:
        """Get current hyperparameters."""
        return {
            'length_scale': self.length_scale,
            'variance': self.variance
        }
    
    def set_hyperparameters(self, params: Dict[str, float]) -> None:
        """Set hyperparameters."""
        if 'length_scale' in params:
            self.length_scale = params['length_scale']
        if 'variance' in params:
            self.variance = params['variance']


class MaternKernel(GPKernel):
    """Matern kernel with configurable smoothness."""
    
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, nu: float = 2.5):
        self.length_scale = length_scale
        self.variance = variance
        self.nu = nu
    
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Matern kernel value."""
        dist = np.linalg.norm(x1 - x2)
        
        if dist == 0:
            return self.variance
        
        # Matern kernel computation
        sqrt_2nu = np.sqrt(2 * self.nu)
        dist_scaled = sqrt_2nu * dist / self.length_scale
        
        # Use modified Bessel function
        bessel_val = kv(self.nu, dist_scaled)
        
        k_val = self.variance * (2 ** (1 - self.nu) / gamma(self.nu)) * \
                (dist_scaled ** self.nu) * bessel_val
        
        return k_val
    
    def compute_matrix(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute Matern kernel matrix."""
        if Y is None:
            Y = X
        
        n, m = X.shape[0], Y.shape[0]
        K = np.zeros((n, m))
        
        for i in range(n):
            for j in range(m):
                K[i, j] = self.compute(X[i], Y[j])
        
        # Add jitter for stability
        if X is Y or np.array_equal(X, Y):
            K += 1e-6 * np.eye(K.shape[0])
        
        return K
    
    def compute_with_gradients(self, x1: np.ndarray, x2: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Compute kernel value and gradients."""
        k_val = self.compute(x1, x2)
        
        # Simplified gradient computation
        gradients = {
            'variance': k_val / self.variance,
            'length_scale': 0.0  # Placeholder for now
        }
        
        return k_val, gradients
    
    def get_hyperparameters(self) -> Dict[str, float]:
        """Get current hyperparameters."""
        return {
            'length_scale': self.length_scale,
            'variance': self.variance,
            'nu': self.nu
        }
    
    def set_hyperparameters(self, params: Dict[str, float]) -> None:
        """Set hyperparameters."""
        if 'length_scale' in params:
            self.length_scale = params['length_scale']
        if 'variance' in params:
            self.variance = params['variance']
        if 'nu' in params:
            self.nu = params['nu']


# Inducing point selection (Single Responsibility Principle)
class InducingPointSelector:
    """Manages inducing point selection strategies."""
    
    def __init__(self, strategy: str = 'kmeans'):
        self.strategy = strategy
    
    def select(self, X: np.ndarray, n_inducing: int, y: Optional[np.ndarray] = None, 
               kernel: Optional[IKernel] = None, **kwargs) -> np.ndarray:
        """Select inducing points from data."""
        n_data = X.shape[0]
        
        if n_inducing >= n_data:
            return np.arange(n_data)
        
        if self.strategy == 'random':
            return self._random_selection(n_data, n_inducing)
        elif self.strategy == 'kmeans':
            return self._kmeans_selection(X, n_inducing)
        elif self.strategy == 'information_gain':
            return self._information_gain_selection(X, n_inducing, y, kernel)
        elif self.strategy == 'adaptive':
            return self._adaptive_selection(X, n_inducing)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def update(self, X: np.ndarray, current_indices: np.ndarray, n_inducing: int, 
               force_include_last: bool = False) -> np.ndarray:
        """Update inducing points with new data."""
        n_data = X.shape[0]
        
        if n_inducing >= n_data:
            return np.arange(n_data)
        
        # If forced to include last point (e.g., extreme observation)
        if force_include_last:
            # Always include the last observation
            last_idx = n_data - 1
            
            # Keep some old inducing points
            n_keep = min(len(current_indices), (n_inducing - 1) // 2)
            kept_indices = current_indices[:n_keep]
            
            # Select new inducing points from recent data
            recent_start = max(0, n_data - 50)
            recent_indices = np.arange(recent_start, n_data - 1)  # Exclude last since we're including it
            
            n_new = n_inducing - n_keep - 1  # -1 for the last point we're forcing
            if len(recent_indices) > n_new and n_new > 0:
                new_selection = np.random.choice(recent_indices, n_new, replace=False)
            else:
                new_selection = recent_indices[:n_new] if n_new > 0 else np.array([])
            
            # Combine: old + new + last
            all_indices = np.concatenate([kept_indices, new_selection, [last_idx]])
            return np.unique(all_indices)  # Remove duplicates
        
        # Original logic when not forcing last point
        n_keep = min(len(current_indices), n_inducing // 2)
        kept_indices = current_indices[:n_keep]
        
        # Select new inducing points from recent data
        recent_start = max(0, n_data - 50)
        recent_indices = np.arange(recent_start, n_data)
        
        n_new = n_inducing - n_keep
        if len(recent_indices) > n_new:
            new_selection = np.random.choice(recent_indices, n_new, replace=False)
        else:
            new_selection = recent_indices
        
        return np.concatenate([kept_indices, new_selection])
    
    def _random_selection(self, n_data: int, n_inducing: int) -> np.ndarray:
        """Random inducing point selection."""
        return np.random.choice(n_data, n_inducing, replace=False)
    
    def _kmeans_selection(self, X: np.ndarray, n_inducing: int) -> np.ndarray:
        """K-means based inducing point selection."""
        kmeans = KMeans(n_clusters=n_inducing, random_state=42)
        kmeans.fit(X)
        
        # Find closest point to each centroid
        inducing_indices = []
        for centroid in kmeans.cluster_centers_:
            distances = np.sum((X - centroid) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            if closest_idx not in inducing_indices:
                inducing_indices.append(closest_idx)
        
        # Fill remaining with random if needed
        while len(inducing_indices) < n_inducing:
            idx = np.random.randint(X.shape[0])
            if idx not in inducing_indices:
                inducing_indices.append(idx)
        
        return np.array(inducing_indices[:n_inducing])
    
    def _information_gain_selection(self, X: np.ndarray, n_inducing: int, 
                                  y: Optional[np.ndarray], kernel: Optional[IKernel]) -> np.ndarray:
        """Information gain based selection."""
        if y is None or kernel is None:
            # Fall back to k-means if missing requirements
            return self._kmeans_selection(X, n_inducing)
        
        # Greedy forward selection based on information gain
        selected = []
        remaining = list(range(X.shape[0]))
        
        # Start with point having highest variance
        variances = y ** 2
        first_idx = np.argmax(variances)
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # Iteratively add points that maximize information gain
        while len(selected) < n_inducing and remaining:
            best_gain = -np.inf
            best_idx = None
            
            for idx in remaining[:100]:  # Limit search for efficiency
                # Simplified information gain approximation
                X_selected = X[selected]
                x_candidate = X[idx]
                
                # Distance to nearest selected point
                min_dist = np.min([np.linalg.norm(x_candidate - x_sel) 
                                 for x_sel in X_selected])
                
                gain = min_dist * abs(y[idx])
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        return np.array(selected)
    
    def _adaptive_selection(self, X: np.ndarray, n_inducing: int) -> np.ndarray:
        """Adaptive selection strategy."""
        # Start with k-means selection
        return self._kmeans_selection(X, n_inducing)


# Sparse GP Model (Single Responsibility Principle)
class SparseGPModel:
    """Sparse Gaussian Process model with inducing points."""
    
    def __init__(self, kernel_type: str = 'rbf', inducing_points: int = 20, 
                 noise_variance: float = 0.1):
        self.kernel_type = kernel_type
        self.inducing_points = inducing_points
        self.noise_variance = noise_variance
        
        # Initialize kernel with reasonable default length scale
        if kernel_type == 'rbf':
            self.kernel = RBFKernel(length_scale=1.0, variance=1.0)  # Standard length scale
        elif kernel_type == 'matern':
            self.kernel = MaternKernel(length_scale=1.0, variance=1.0)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        # Data storage
        self.observations: List[TestObservation] = []
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        
        # Sparse GP components
        self.Z: Optional[np.ndarray] = None  # Inducing points
        self.inducing_selector = InducingPointSelector(strategy='kmeans')
        
        # Cached matrices
        self.Kuu: Optional[np.ndarray] = None
        self.Kuf: Optional[np.ndarray] = None
        self.Qff_diag: Optional[np.ndarray] = None
        self.Lambda: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        
        # Cholesky factors
        self.L_uu: Optional[np.ndarray] = None
        self.L_lambda: Optional[np.ndarray] = None
    
    def add_observation(self, observation: TestObservation) -> None:
        """Add new observation to the model."""
        if observation.features is None:
            observation.extract_features()
        self.observations.append(observation)
    
    def fit(self) -> None:
        """Fit sparse GP to all observations."""
        if not self.observations:
            return
        
        # Extract features and targets
        self._prepare_data()
        
        # Select inducing points
        n_inducing = min(self.inducing_points, len(self.observations))
        inducing_indices = self.inducing_selector.select(self.X, n_inducing)
        self.Z = self.X[inducing_indices]
        
        # Compute sparse GP matrices
        self._compute_sparse_matrices()
    
    def predict(self, test_ids: List[str], features: np.ndarray) -> PredictionResult:
        """Make prediction for new test configuration."""
        if self.Z is None or self.alpha is None or len(self.observations) < 5:
            # Not fitted yet or too few observations, return prior with higher uncertainty
            prior_std = np.sqrt(self.kernel.variance + self.noise_variance)
            return PredictionResult(
                mean=0.0,
                std_dev=prior_std,
                confidence_interval=(-1.96 * prior_std, 1.96 * prior_std),
                test_ids=test_ids
            )
        
        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Compute predictive mean and variance
        Ksu = self.kernel.compute_matrix(features, self.Z)
        
        # Mean: Ksu @ alpha
        mean = Ksu @ self.alpha
        mean = float(mean[0]) if mean.shape else float(mean)
        
        # Variance computation with numerical stability
        try:
            v = cho_solve((self.L_uu, True), Ksu.T)
        except LinAlgError:
            # If Cholesky solve fails, use default variance
            variance = self.kernel.variance + self.noise_variance
            std_dev = np.sqrt(variance)
            z_score = 1.96
            ci = (mean - z_score * std_dev, mean + z_score * std_dev)
            
            return PredictionResult(
                mean=mean,
                std_dev=std_dev,
                confidence_interval=ci,
                test_ids=test_ids
            )
        
        # Prior variance
        prior_var = self.kernel.variance
        
        # Reduction from inducing points (should be positive)
        var_reduction = np.sum(Ksu * v.T, axis=1)
        var_reduction = np.clip(var_reduction, 0, prior_var)
        
        # Noise contribution
        if self.L_lambda is not None:
            try:
                # Avoid numerical overflow by clipping v before multiplication
                v_clipped = np.clip(v, -1e6, 1e6)
                w = cho_solve((self.L_lambda, True), v_clipped)
                w_clipped = np.clip(w, -1e6, 1e6)
                var_noise = np.sum(v_clipped * w_clipped, axis=0)
                var_noise = np.clip(var_noise, 0, prior_var)
            except (LinAlgError, RuntimeWarning):
                var_noise = 0
        else:
            var_noise = 0
        
        # Total variance with numerical stability
        # For sparse GPs, the variance should be: prior_var - var_reduction + var_noise + noise_var
        # But var_noise can be large, so we need to be careful
        variance = prior_var - var_reduction + var_noise + self.noise_variance
        
        # Handle array/scalar conversion
        if hasattr(variance, 'shape'):
            if variance.shape:
                variance = float(variance[0])
            else:
                variance = float(variance)
        
        # Ensure reasonable bounds on variance
        min_variance = self.noise_variance
        max_variance = prior_var + self.noise_variance + var_noise
        variance = np.clip(variance, min_variance, max_variance)
        
        std_dev = np.sqrt(variance)
        
        # 95% confidence interval
        z_score = 1.96
        ci = (mean - z_score * std_dev, mean + z_score * std_dev)
        
        return PredictionResult(
            mean=mean,
            std_dev=std_dev,
            confidence_interval=ci,
            test_ids=test_ids
        )
    
    def incremental_update(self, observation: TestObservation) -> None:
        """Incrementally update model with new observation."""
        # Add observation
        self.add_observation(observation)
        
        if len(self.observations) < 10:
            # Too few observations, just refit
            self.fit()
            return
        
        # Fast incremental update
        if observation.features is None:
            observation.extract_features()
        
        # Update data arrays
        new_x = observation.features.reshape(1, -1)
        new_y = np.array([observation.quality_score])
        
        if self.X is None:
            self.X = new_x
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_x])
            self.y = np.append(self.y, new_y)
        
        # Update inducing points occasionally or when we get a surprising observation
        should_update_inducing = (len(self.observations) % 20 == 0)
        
        # Also update if this is a highly informative observation
        if not should_update_inducing and self.alpha is not None and self.Z is not None:
            # Check if this observation is surprising
            ksu = self.kernel.compute_matrix(new_x, self.Z).flatten()
            predicted_mean = ksu @ self.alpha
            residual = abs(float(new_y[0] if new_y.shape else new_y) - predicted_mean)
            # Also check if the observation value is extreme
            obs_value = float(new_y[0] if new_y.shape else new_y)
            mean_value = np.mean(self.y[:-1]) if len(self.y) > 1 else 0.0
            std_value = np.std(self.y[:-1]) if len(self.y) > 1 else 1.0
            z_score = abs(obs_value - mean_value) / (std_value + 1e-6)
            
            if residual > 1.0 or z_score > 3.0:  # Update for surprising or extreme observations
                should_update_inducing = True
        
        if should_update_inducing or self.Kuf is None:
            n_inducing = min(self.inducing_points, len(self.observations))
            # Force include the last observation if it's extreme
            force_include = z_score > 3.0 if 'z_score' in locals() else False
            inducing_indices = self.inducing_selector.update(
                self.X, 
                np.arange(len(self.Z)) if self.Z is not None else np.array([]), 
                n_inducing,
                force_include_last=force_include
            )
            self.Z = self.X[inducing_indices]
            self._compute_sparse_matrices()
        else:
            # Quick rank-1 update approximation
            self._rank_one_update(new_x, new_y)
    
    def get_hyperparameters(self) -> Dict[str, float]:
        """Get current hyperparameters."""
        params = self.kernel.get_hyperparameters()
        params['noise_variance'] = self.noise_variance
        return params
    
    def set_hyperparameters(self, params: Dict[str, float]) -> None:
        """Set hyperparameters."""
        if 'noise_variance' in params:
            self.noise_variance = params['noise_variance']
        
        kernel_params = {k: v for k, v in params.items() if k != 'noise_variance'}
        if kernel_params:
            self.kernel.set_hyperparameters(kernel_params)
        
        # Recompute matrices with new hyperparameters
        if self.Z is not None:
            self._compute_sparse_matrices()
    
    def log_marginal_likelihood(self) -> float:
        """Compute log marginal likelihood for hyperparameter optimization."""
        if self.y is None or self.L_lambda is None:
            return -np.inf
        
        n = len(self.y)
        
        # Data fit term: -0.5 * y.T @ inv(K + sigma^2 I) @ y
        data_fit = -0.5 * np.sum(self.y * (self.y - self.Kuf.T @ self.alpha))
        
        # Complexity penalty: -0.5 * log|K + sigma^2 I|
        complexity = -np.sum(np.log(np.diag(self.L_lambda)))
        complexity -= np.sum(np.log(np.diag(self.L_uu)))
        
        # Constant term
        const = -0.5 * n * np.log(2 * np.pi)
        
        return data_fit + complexity + const
    
    def _prepare_data(self) -> None:
        """Prepare data arrays from observations."""
        features_list = []
        targets_list = []
        
        for obs in self.observations:
            if obs.features is None:
                obs.extract_features()
            features_list.append(obs.features)
            targets_list.append(obs.quality_score)
        
        self.X = np.array(features_list)
        self.y = np.array(targets_list)
    
    def _compute_sparse_matrices(self) -> None:
        """Compute sparse GP matrices."""
        if self.Z is None or self.X is None or self.y is None:
            return
        
        # Compute kernel matrices
        self.Kuu = self.kernel.compute_matrix(self.Z)  # m x m
        self.Kuf = self.kernel.compute_matrix(self.Z, self.X)  # m x n
        
        # Compute Qff diagonal
        try:
            L_uu = cho_factor(self.Kuu)
            self.L_uu = L_uu[0]
            
            # Qff = Kfu @ inv(Kuu) @ Kuf
            v = cho_solve(L_uu, self.Kuf)  # inv(Kuu) @ Kuf
            self.Qff_diag = np.sum(self.Kuf * v, axis=0)
            
            # Lambda = diag(Kff - Qff) + sigma^2 I
            # We only need diagonal of Kff
            Kff_diag = self.kernel.variance * np.ones(len(self.X))
            
            # Ensure Qff_diag doesn't exceed Kff_diag (numerical stability)
            self.Qff_diag = np.minimum(self.Qff_diag, Kff_diag - 1e-6)
            
            Lambda_diag = Kff_diag - self.Qff_diag + self.noise_variance
            Lambda_diag = np.maximum(Lambda_diag, self.noise_variance)  # Ensure positive
            
            self.Lambda = np.diag(Lambda_diag)
            
            # Compute alpha = inv(Kuu + Kuf @ inv(Lambda) @ Kfu) @ Kuf @ inv(Lambda) @ y
            # Using Woodbury identity
            A = self.Kuf / Lambda_diag  # Kuf @ inv(Lambda)
            B = self.Kuu + A @ self.Kuf.T  # Kuu + Kuf @ inv(Lambda) @ Kfu
            
            L_b = cho_factor(B)
            self.L_lambda = L_b[0]
            
            # alpha = inv(B) @ A @ y
            self.alpha = cho_solve(L_b, A @ self.y)
            
        except LinAlgError:
            logger.warning("Cholesky decomposition failed, adding more jitter")
            self.Kuu += 1e-4 * np.eye(len(self.Z))
            self._compute_sparse_matrices()  # Retry
    
    def _rank_one_update(self, new_x: np.ndarray, new_y: np.ndarray) -> None:
        """Approximate rank-one update for new observation."""
        if self.alpha is None or self.Z is None or self.Kuf is None:
            self._compute_sparse_matrices()  # Need full computation
            return
        
        # For better incremental updates, periodically recompute the full matrices
        # This ensures uncertainty estimates are properly updated
        if len(self.observations) % 10 == 0:
            self._compute_sparse_matrices()
            return
        
        # The key issue: we need to update Kuf to include the new observation
        # Kuf is m x n, where m is number of inducing points and n is number of observations
        # We need to add a new column for the new observation
        kuf_new = self.kernel.compute_matrix(self.Z, new_x).reshape(-1, 1)
        self.Kuf = np.hstack([self.Kuf, kuf_new])
        
        # Now recompute matrices with the updated Kuf
        # This is necessary because alpha depends on all observations through Kuf
        try:
            # Recompute Qff diagonal with new observation
            L_uu = cho_factor(self.Kuu)
            v_all = cho_solve(L_uu, self.Kuf)  # inv(Kuu) @ Kuf
            self.Qff_diag = np.sum(self.Kuf * v_all, axis=0)
            
            # Update Lambda with new observation
            Kff_diag = self.kernel.variance * np.ones(len(self.X))
            self.Qff_diag = np.minimum(self.Qff_diag, Kff_diag - 1e-6)
            
            Lambda_diag = Kff_diag - self.Qff_diag + self.noise_variance
            Lambda_diag = np.maximum(Lambda_diag, self.noise_variance)
            
            self.Lambda = np.diag(Lambda_diag)
            
            # Recompute alpha with all observations including the new one
            A = self.Kuf / Lambda_diag  # Kuf @ inv(Lambda)
            B = self.Kuu + A @ self.Kuf.T  # Kuu + Kuf @ inv(Lambda) @ Kfu
            
            L_b = cho_factor(B)
            self.L_lambda = L_b[0]
            
            # alpha = inv(B) @ A @ y (where y now includes the new observation)
            self.alpha = cho_solve(L_b, A @ self.y)
            
        except LinAlgError:
            # If anything fails, fall back to full recomputation
            self._compute_sparse_matrices()


# Memory Management (Single Responsibility Principle)
class MemoryManager:
    """Manages memory usage for observations."""
    
    def __init__(self, max_memory_bytes: int, strategy: str = 'fifo', 
                 compression_ratio: float = 0.5):
        self.max_memory_bytes = max_memory_bytes
        self.strategy = strategy
        self.compression_ratio = compression_ratio
        
        self.observations: List[TestObservation] = []
        self.importance_scores: Dict[int, float] = {}
        self.compressed_indices: set = set()
    
    def add(self, observation: TestObservation, importance: float = 1.0) -> None:
        """Add observation to memory."""
        if not self.can_add(observation):
            self._make_space(observation)
        
        self.observations.append(observation)
        obs_idx = len(self.observations) - 1
        self.importance_scores[obs_idx] = importance
    
    def can_add(self, observation: TestObservation) -> bool:
        """Check if observation can be added without exceeding limit."""
        current_usage = self.get_memory_usage()
        obs_size = self._estimate_size(observation)
        return current_usage + obs_size <= self.max_memory_bytes
    
    def is_memory_full(self) -> bool:
        """Check if memory is at capacity."""
        return self.get_memory_usage() >= 0.95 * self.max_memory_bytes
    
    def get_memory_usage(self) -> int:
        """Estimate current memory usage."""
        total_size = 0
        for i, obs in enumerate(self.observations):
            size = self._estimate_size(obs)
            if i in self.compressed_indices:
                size = int(size * self.compression_ratio)
            total_size += size
        return total_size
    
    def compress_old_observations(self, age_threshold: int) -> int:
        """Compress observations older than threshold."""
        current_time = time.time()
        compressed_count = 0
        
        for i, obs in enumerate(self.observations):
            age = current_time - obs.timestamp
            if age > age_threshold and i not in self.compressed_indices:
                # Simulate compression by reducing feature precision
                if obs.features is not None:
                    obs.features = obs.features.astype(np.float16)
                self.compressed_indices.add(i)
                compressed_count += 1
        
        return compressed_count
    
    def get_retained_observations(self) -> List[TestObservation]:
        """Get list of retained observations."""
        return self.observations.copy()
    
    def _estimate_size(self, observation: TestObservation) -> int:
        """Estimate memory size of observation."""
        size = 100  # Base overhead
        
        # Features array
        if observation.features is not None:
            size += observation.features.nbytes
        
        # Test IDs
        size += sum(len(tid) for tid in observation.test_ids)
        
        # Metadata
        size += len(str(observation.metadata))
        
        return size
    
    def _make_space(self, new_observation: TestObservation) -> None:
        """Make space for new observation by removing old ones."""
        if self.strategy == 'fifo':
            # Remove oldest observations
            while not self.can_add(new_observation) and self.observations:
                removed_idx = 0
                self.observations.pop(removed_idx)
                # Update importance scores indices
                new_scores = {}
                for idx, score in self.importance_scores.items():
                    if idx > removed_idx:
                        new_scores[idx - 1] = score
                self.importance_scores = new_scores
        
        elif self.strategy == 'importance':
            # Remove least important observations
            if not self.observations:
                return
            
            # Sort by importance
            sorted_indices = sorted(self.importance_scores.keys(), 
                                  key=lambda i: self.importance_scores.get(i, 0))
            
            # Remove least important
            for idx in sorted_indices:
                if self.can_add(new_observation):
                    break
                if idx < len(self.observations):
                    self.observations.pop(idx)
                    del self.importance_scores[idx]
                    # Update remaining indices
                    new_scores = {}
                    for i, score in self.importance_scores.items():
                        if i > idx:
                            new_scores[i - 1] = score
                        else:
                            new_scores[i] = score
                    self.importance_scores = new_scores
        
        elif self.strategy == 'compression':
            # First try compression
            self.compress_old_observations(3600)  # Compress observations older than 1 hour
            
            # If still not enough space, remove old observations
            if not self.can_add(new_observation):
                self._make_space_fifo(new_observation)
    
    def _make_space_fifo(self, new_observation: TestObservation) -> None:
        """FIFO removal strategy."""
        while not self.can_add(new_observation) and self.observations:
            self.observations.pop(0)


# Hyperparameter Optimization (Single Responsibility Principle)
class HyperparameterOptimizer:
    """Optimizes GP hyperparameters."""
    
    def __init__(self, method: str = 'gradient', schedule: str = 'periodic',
                 bounds: Optional[Dict[str, Tuple[float, float]]] = None):
        self.method = method
        self.schedule = schedule
        self.bounds = bounds or {
            'length_scale': (0.1, 10.0),
            'variance': (0.1, 10.0),
            'noise_variance': (1e-4, 1.0)
        }
        
        self.optimization_history = []
        self.last_optimization_step = 0
    
    def optimize(self, model: SparseGPModel) -> Dict[str, float]:
        """Optimize hyperparameters of the model."""
        if self.method == 'gradient':
            return self._gradient_optimization(model)
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")
    
    def should_optimize(self, model: SparseGPModel, step: int) -> bool:
        """Check if optimization should be performed."""
        if self.schedule == 'periodic':
            # Optimize every 20 steps
            return step % 20 == 0
        
        elif self.schedule == 'adaptive':
            # Optimize more frequently early on
            if step < 10:
                return step % 2 == 0
            elif step < 50:
                return step % 5 == 0
            else:
                return step % 20 == 0
        
        return False
    
    def _gradient_optimization(self, model: SparseGPModel) -> Dict[str, float]:
        """Gradient-based hyperparameter optimization."""
        # Get current parameters
        current_params = model.get_hyperparameters()
        
        # Convert to optimization vector
        param_names = list(self.bounds.keys())
        x0 = np.array([current_params.get(name, 1.0) for name in param_names])
        
        # Transform to log space for better optimization
        x0_log = np.log(x0)
        
        # Define objective function (negative log marginal likelihood)
        def objective(x_log):
            x = np.exp(x_log)
            params = dict(zip(param_names, x))
            
            # Set parameters
            model.set_hyperparameters(params)
            
            # Compute negative log marginal likelihood
            try:
                lml = model.log_marginal_likelihood()
                return -lml
            except Exception:
                return np.inf
        
        # Set bounds in log space
        bounds_list = []
        for name in param_names:
            low, high = self.bounds[name]
            bounds_list.append((np.log(low), np.log(high)))
        
        # Optimize
        result = minimize(objective, x0_log, method='L-BFGS-B', bounds=bounds_list)
        
        # Extract optimized parameters
        x_opt = np.exp(result.x)
        optimized_params = dict(zip(param_names, x_opt))
        
        # Ensure parameters are strictly within bounds (fix floating point precision issues)
        for name in param_names:
            low, high = self.bounds[name]
            optimized_params[name] = np.clip(optimized_params[name], low, high)
        
        # Record optimization
        self.optimization_history.append({
            'step': self.last_optimization_step,
            'params': optimized_params,
            'lml': -result.fun
        })
        
        return optimized_params


# Main Incremental GP Learner (Interface Segregation Principle)
class IncrementalGPLearner:
    """Main class for incremental Gaussian Process learning."""
    
    def __init__(self, kernel_type: str = 'rbf', inducing_points: int = 30,
                 memory_limit_mb: int = 50, optimization_schedule: str = 'adaptive'):
        self.kernel_type = kernel_type
        self.inducing_points = inducing_points
        self.memory_limit_mb = memory_limit_mb
        
        # Initialize components
        self.model = SparseGPModel(
            kernel_type=kernel_type,
            inducing_points=inducing_points
        )
        
        self.memory_manager = MemoryManager(
            max_memory_bytes=memory_limit_mb * 1024 * 1024,
            strategy='importance'
        )
        
        self.hyperparameter_optimizer = HyperparameterOptimizer(
            method='gradient',
            schedule=optimization_schedule
        )
        
        self.observation_count = 0
    
    def add_observation(self, observation: TestObservation) -> None:
        """Add new observation to the learner."""
        # Calculate importance based on quality deviation
        importance = abs(observation.quality_score)
        
        # Add to memory manager
        self.memory_manager.add(observation, importance)
        
        # Add to model
        self.model.add_observation(observation)
        
        self.observation_count += 1
    
    def fit(self) -> None:
        """Fit the GP model to all observations."""
        # Get observations from memory manager
        observations = self.memory_manager.get_retained_observations()
        
        # Update model observations
        self.model.observations = observations
        
        # Fit model
        self.model.fit()
        
        # Check if hyperparameter optimization is needed
        if self.hyperparameter_optimizer.should_optimize(self.model, self.observation_count):
            self.optimize_hyperparameters()
    
    def predict(self, test_ids: List[str], features: np.ndarray) -> PredictionResult:
        """Make prediction for test configuration."""
        return self.model.predict(test_ids, features)
    
    def incremental_update(self, observation: TestObservation) -> None:
        """Incrementally update model with new observation."""
        # Calculate importance for memory manager
        importance = abs(observation.quality_score)
        
        # Add to memory manager only (not to model, as incremental_update will do that)
        self.memory_manager.add(observation, importance)
        
        # Perform incremental update (this adds the observation to the model)
        self.model.incremental_update(observation)
        
        self.observation_count += 1
        
        # Periodic hyperparameter optimization
        if self.hyperparameter_optimizer.should_optimize(self.model, self.observation_count):
            self.optimize_hyperparameters()
    
    def suggest_next_tests(self, candidate_features: np.ndarray, n_suggestions: int = 5,
                          acquisition: str = 'ucb') -> List[int]:
        """Suggest next tests to run based on acquisition function."""
        acquisition_values = []
        
        for i in range(len(candidate_features)):
            features = candidate_features[i]
            pred = self.model.predict([f'candidate_{i}'], features)
            
            if acquisition == 'ucb':
                # Upper Confidence Bound
                beta = 2.0  # Exploration parameter
                acq_value = pred.mean + beta * pred.std_dev
            elif acquisition == 'ei':
                # Expected Improvement
                # Simplified version
                best_value = max([obs.quality_score for obs in self.model.observations] + [0])
                improvement = pred.mean - best_value
                acq_value = improvement + pred.std_dev
            else:
                acq_value = pred.std_dev  # Pure exploration
            
            # Ensure scalar value
            if hasattr(acq_value, 'shape') and acq_value.shape:
                acq_value = acq_value.item()
            acquisition_values.append(float(acq_value))
        
        # Select top candidates
        acquisition_values = np.array(acquisition_values).flatten()  # Ensure 1D array
        
        # Add small random noise to break ties
        acquisition_values = acquisition_values + np.random.normal(0, 1e-6, size=len(acquisition_values))
        
        sorted_indices = np.argsort(acquisition_values)[::-1]
        top_indices = sorted_indices[:n_suggestions]
        return top_indices.tolist()  # Convert to list of integers
    
    def optimize_hyperparameters(self) -> None:
        """Optimize GP hyperparameters."""
        optimized_params = self.hyperparameter_optimizer.optimize(self.model)
        self.model.set_hyperparameters(optimized_params)
    
    def extract_test_features(self, test_ids: List[str]) -> np.ndarray:
        """Extract features from test IDs for Guardian integration."""
        # Feature extraction logic
        features = []
        
        # Number of tests
        features.append(len(test_ids))
        
        # Average test ID length
        avg_length = np.mean([len(tid) for tid in test_ids]) if test_ids else 0
        features.append(avg_length)
        
        # Hash-based features for test combination
        combined_hash = hash(tuple(sorted(test_ids)))
        features.append((combined_hash % 1000) / 1000.0)
        
        # Test diversity (unique prefixes)
        prefixes = set()
        for tid in test_ids:
            if '_' in tid:
                prefixes.add(tid.split('_')[0])
        features.append(len(prefixes))
        
        # Interaction feature
        if len(test_ids) >= 2:
            # Simple interaction based on string similarity
            interaction = sum(1 for i in range(len(test_ids)) 
                            for j in range(i+1, len(test_ids))
                            if test_ids[i][:3] == test_ids[j][:3])
            features.append(interaction)
        else:
            features.append(0)
        
        return np.array(features, dtype=np.float64)
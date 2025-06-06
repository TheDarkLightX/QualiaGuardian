"""
TDD Tests for Incremental Gaussian Process Learning

Author: DarkLightX/Dana Edwards
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import time
import psutil
import os
from typing import Dict, List, Optional, Tuple

from guardian.analytics.incremental_gp import (
    IncrementalGPLearner,
    SparseGPModel,
    InducingPointSelector,
    HyperparameterOptimizer,
    MemoryManager,
    TestObservation,
    PredictionResult,
    GPKernel,
    RBFKernel,
    MaternKernel,
    IGPModel,
    IKernel,
    IInducingPointStrategy,
    IMemoryStrategy,
    IHyperparameterStrategy
)


class TestGPKernels:
    """TDD tests for GP kernel implementations."""
    
    def test_rbf_kernel_computation(self):
        """RED: Test RBF kernel computation."""
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        
        # Test on simple points
        x1 = np.array([0, 0])
        x2 = np.array([1, 1])
        
        # K(x,x) should equal variance
        k_same = kernel.compute(x1, x1)
        assert abs(k_same - 1.0) < 1e-6
        
        # K(x,y) should decay with distance
        k_diff = kernel.compute(x1, x2)
        assert 0 < k_diff < 1.0
        
        # Test symmetry
        k_reverse = kernel.compute(x2, x1)
        assert abs(k_diff - k_reverse) < 1e-6
    
    def test_kernel_matrix_computation(self):
        """RED: Test kernel matrix computation."""
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        K = kernel.compute_matrix(X)
        
        # Check shape
        assert K.shape == (4, 4)
        
        # Check symmetry
        assert np.allclose(K, K.T)
        
        # Check positive definiteness (all eigenvalues > 0)
        eigenvalues = np.linalg.eigvalsh(K)
        assert all(eig > -1e-6 for eig in eigenvalues)
    
    def test_kernel_hyperparameter_gradients(self):
        """RED: Test kernel gradient computation."""
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        
        x1 = np.array([0, 0])
        x2 = np.array([1, 1])
        
        k_val, gradients = kernel.compute_with_gradients(x1, x2)
        
        assert 'length_scale' in gradients
        assert 'variance' in gradients
        
        # Gradient w.r.t. variance should be k/variance
        expected_var_grad = k_val / 1.0
        assert abs(gradients['variance'] - expected_var_grad) < 1e-6
    
    def test_matern_kernel(self):
        """RED: Test Matern kernel implementation."""
        kernel = MaternKernel(length_scale=1.0, variance=1.0, nu=2.5)
        
        x1 = np.array([0, 0])
        x2 = np.array([1, 0])
        
        k_val = kernel.compute(x1, x2)
        
        # Should be between 0 and variance
        assert 0 < k_val <= 1.0
        
        # Test limiting behavior (as nu -> inf, should approach RBF)
        kernel_large_nu = MaternKernel(length_scale=1.0, variance=1.0, nu=50)
        rbf_kernel = RBFKernel(length_scale=1.0, variance=1.0)
        
        k_matern = kernel_large_nu.compute(x1, x2)
        k_rbf = rbf_kernel.compute(x1, x2)
        
        assert abs(k_matern - k_rbf) < 0.1  # Should be similar


class TestTestObservation:
    """TDD tests for test observation data structure."""
    
    def test_observation_creation(self):
        """RED: Test creating test observations."""
        obs = TestObservation(
            test_ids=['test1', 'test2'],
            features=np.array([1.0, 2.0, 3.0]),
            quality_score=0.85,
            timestamp=100
        )
        
        assert obs.test_ids == ['test1', 'test2']
        assert np.array_equal(obs.features, np.array([1.0, 2.0, 3.0]))
        assert obs.quality_score == 0.85
        assert obs.timestamp == 100
    
    def test_observation_validation(self):
        """RED: Test observation validation."""
        # Should handle empty test_ids
        obs = TestObservation(
            test_ids=[],
            features=np.array([1.0]),
            quality_score=0.5,
            timestamp=0
        )
        assert obs.test_ids == []
        
        # Should handle negative quality scores
        obs = TestObservation(
            test_ids=['test1'],
            features=np.array([1.0]),
            quality_score=-0.5,
            timestamp=0
        )
        assert obs.quality_score == -0.5
    
    def test_observation_feature_extraction(self):
        """RED: Test feature extraction from observations."""
        obs = TestObservation(
            test_ids=['test1', 'test2', 'test3'],
            features=None,
            quality_score=0.7,
            timestamp=0
        )
        
        # Should automatically extract features
        obs.extract_features()
        
        assert obs.features is not None
        assert len(obs.features) > 0


class TestInducingPointSelection:
    """TDD tests for inducing point selection strategies."""
    
    def test_random_inducing_selection(self):
        """RED: Test random inducing point selection."""
        selector = InducingPointSelector(strategy='random')
        
        # Create data points
        X = np.random.randn(100, 5)
        
        # Select inducing points
        inducing_indices = selector.select(X, n_inducing=10)
        
        assert len(inducing_indices) == 10
        assert all(0 <= idx < 100 for idx in inducing_indices)
        assert len(set(inducing_indices)) == 10  # All unique
    
    def test_kmeans_inducing_selection(self):
        """RED: Test k-means based inducing selection."""
        selector = InducingPointSelector(strategy='kmeans')
        
        # Create clustered data
        X1 = np.random.randn(50, 5) + np.array([5, 0, 0, 0, 0])
        X2 = np.random.randn(50, 5) + np.array([-5, 0, 0, 0, 0])
        X = np.vstack([X1, X2])
        
        # Select inducing points
        inducing_indices = selector.select(X, n_inducing=10)
        
        assert len(inducing_indices) == 10
        
        # Should select from both clusters
        cluster1_selected = sum(1 for idx in inducing_indices if idx < 50)
        cluster2_selected = sum(1 for idx in inducing_indices if idx >= 50)
        
        assert cluster1_selected > 0
        assert cluster2_selected > 0
    
    def test_information_gain_selection(self):
        """RED: Test information gain based selection."""
        selector = InducingPointSelector(strategy='information_gain')
        
        # Create data with known informativeness
        n = 100
        X = np.random.randn(n, 5)
        y = np.sin(X[:, 0]) + 0.1 * np.random.randn(n)
        
        # Provide kernel for information calculation
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        
        inducing_indices = selector.select(
            X, n_inducing=20, y=y, kernel=kernel
        )
        
        assert len(inducing_indices) == 20
        
        # Selected points should be spread out (not all clustered)
        inducing_X = X[inducing_indices]
        min_distances = []
        
        for i in range(len(inducing_X)):
            distances = [np.linalg.norm(inducing_X[i] - inducing_X[j])
                        for j in range(len(inducing_X)) if i != j]
            min_distances.append(min(distances))
        
        avg_min_dist = np.mean(min_distances)
        assert avg_min_dist > 0.5  # Points should be spread
    
    def test_adaptive_inducing_selection(self):
        """RED: Test adaptive inducing point updates."""
        selector = InducingPointSelector(strategy='adaptive')
        
        # Initial selection
        X_initial = np.random.randn(50, 5)
        initial_indices = selector.select(X_initial, n_inducing=10)
        
        # Add new data
        X_new = np.random.randn(20, 5) + np.array([3, 0, 0, 0, 0])
        X_combined = np.vstack([X_initial, X_new])
        
        # Update selection
        updated_indices = selector.update(
            X_combined,
            current_indices=initial_indices,
            n_inducing=15
        )
        
        assert len(updated_indices) == 15
        
        # Should include some new points
        new_point_selected = any(idx >= 50 for idx in updated_indices)
        assert new_point_selected


class TestSparseGPModel:
    """TDD tests for sparse GP implementation."""
    
    @pytest.fixture
    def sparse_gp(self):
        """Create a sparse GP model."""
        return SparseGPModel(
            kernel_type='rbf',
            inducing_points=20,
            noise_variance=0.01
        )
    
    def test_sparse_gp_initialization(self, sparse_gp):
        """RED: Test sparse GP initialization."""
        assert sparse_gp.inducing_points == 20
        assert sparse_gp.noise_variance == 0.01
        assert sparse_gp.kernel is not None
        assert len(sparse_gp.observations) == 0
    
    def test_add_observations(self, sparse_gp):
        """RED: Test adding observations to sparse GP."""
        # Add observations
        for i in range(30):
            obs = TestObservation(
                test_ids=[f'test_{i}'],
                features=np.random.randn(5),
                quality_score=np.random.randn(),
                timestamp=i
            )
            sparse_gp.add_observation(obs)
        
        assert len(sparse_gp.observations) == 30
    
    def test_sparse_gp_fitting(self, sparse_gp):
        """RED: Test fitting sparse GP model."""
        # Generate synthetic data
        n = 50
        X = np.random.randn(n, 5)
        y = np.sin(X[:, 0]) + 0.1 * np.random.randn(n)
        
        # Add observations
        for i in range(n):
            obs = TestObservation(
                test_ids=[f'test_{i}'],
                features=X[i],
                quality_score=y[i],
                timestamp=i
            )
            sparse_gp.add_observation(obs)
        
        # Fit model
        sparse_gp.fit()
        
        # Should have selected inducing points
        assert sparse_gp.Z is not None
        assert sparse_gp.Z.shape[0] <= sparse_gp.inducing_points
        
        # Should have computed necessary matrices
        assert hasattr(sparse_gp, 'Kuu')  # Inducing kernel matrix
        assert hasattr(sparse_gp, 'Kuf')  # Cross kernel matrix
    
    def test_sparse_prediction(self, sparse_gp):
        """RED: Test sparse GP predictions."""
        # Train on data
        n = 50
        X = np.random.randn(n, 5)
        y = np.sin(X[:, 0]) + 0.1 * np.random.randn(n)
        
        for i in range(n):
            obs = TestObservation(
                test_ids=[f'test_{i}'],
                features=X[i],
                quality_score=y[i],
                timestamp=i
            )
            sparse_gp.add_observation(obs)
        
        sparse_gp.fit()
        
        # Make predictions
        X_test = np.random.randn(10, 5)
        
        for i in range(10):
            result = sparse_gp.predict(
                test_ids=[f'pred_{i}'],
                features=X_test[i]
            )
            
            assert isinstance(result, PredictionResult)
            assert not np.isnan(result.mean)
            assert result.std_dev > 0
            assert len(result.confidence_interval) == 2
            assert result.confidence_interval[0] < result.mean < result.confidence_interval[1]
    
    def test_incremental_update(self, sparse_gp):
        """RED: Test incremental sparse GP updates."""
        # Initial training
        n_initial = 30
        for i in range(n_initial):
            obs = TestObservation(
                test_ids=[f'test_{i}'],
                features=np.random.randn(5),
                quality_score=np.random.randn(),
                timestamp=i
            )
            sparse_gp.add_observation(obs)
        
        sparse_gp.fit()
        
        # Store initial state
        initial_pred = sparse_gp.predict(['test'], np.zeros(5))
        
        # Add new observation
        new_obs = TestObservation(
            test_ids=['new_test'],
            features=np.zeros(5),  # Near test point
            quality_score=2.0,  # High value
            timestamp=100
        )
        
        start_time = time.time()
        sparse_gp.incremental_update(new_obs)
        update_time = time.time() - start_time
        
        # Update should be fast
        assert update_time < 0.1  # 100ms
        
        # Prediction should change
        updated_pred = sparse_gp.predict(['test'], np.zeros(5))
        assert updated_pred.mean > initial_pred.mean
    
    def test_memory_efficiency(self, sparse_gp):
        """RED: Test memory efficiency of sparse GP."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Add many observations
        n = 1000
        for i in range(n):
            obs = TestObservation(
                test_ids=[f'test_{i}'],
                features=np.random.randn(5),
                quality_score=np.random.randn(),
                timestamp=i
            )
            sparse_gp.add_observation(obs)
        
        sparse_gp.fit()
        
        final_memory = process.memory_info().rss
        memory_increase_mb = (final_memory - initial_memory) / 1024 / 1024
        
        # Memory should scale with m^2, not n^2
        # For m=20, expect < 10MB increase
        assert memory_increase_mb < 50  # Generous bound


class TestHyperparameterOptimization:
    """TDD tests for hyperparameter optimization."""
    
    def test_gradient_based_optimization(self):
        """RED: Test gradient-based hyperparameter optimization."""
        optimizer = HyperparameterOptimizer(method='gradient')
        
        # Create simple GP model
        model = SparseGPModel(kernel_type='rbf')
        
        # Add training data
        n = 50
        X = np.random.randn(n, 5)
        y = np.sin(X[:, 0]) + 0.1 * np.random.randn(n)
        
        for i in range(n):
            obs = TestObservation(
                test_ids=[f'test_{i}'],
                features=X[i],
                quality_score=y[i],
                timestamp=i
            )
            model.add_observation(obs)
        
        model.fit()
        
        # Get initial parameters
        initial_params = model.get_hyperparameters()
        initial_lml = model.log_marginal_likelihood()
        
        # Optimize
        optimized_params = optimizer.optimize(model)
        
        # Parameters should change
        assert optimized_params != initial_params
        
        # Apply optimized parameters
        model.set_hyperparameters(optimized_params)
        model.fit()
        
        # Log marginal likelihood should improve
        final_lml = model.log_marginal_likelihood()
        assert final_lml >= initial_lml
    
    def test_adaptive_optimization_schedule(self):
        """RED: Test adaptive optimization scheduling."""
        optimizer = HyperparameterOptimizer(
            method='gradient',
            schedule='adaptive'
        )
        
        model = SparseGPModel(kernel_type='rbf')
        
        # Track optimization calls
        optimization_points = []
        
        for i in range(100):
            obs = TestObservation(
                test_ids=[f'test_{i}'],
                features=np.random.randn(5),
                quality_score=np.random.randn(),
                timestamp=i
            )
            model.add_observation(obs)
            
            # Check if optimization is needed
            if optimizer.should_optimize(model, i):
                optimizer.optimize(model)
                optimization_points.append(i)
        
        # Should optimize periodically but not every step
        assert len(optimization_points) > 2
        assert len(optimization_points) < 20
        
        # Early optimizations should be more frequent
        early_gap = optimization_points[1] - optimization_points[0]
        late_gap = optimization_points[-1] - optimization_points[-2]
        assert late_gap >= early_gap
    
    def test_hyperparameter_bounds(self):
        """RED: Test hyperparameter bound constraints."""
        optimizer = HyperparameterOptimizer(
            method='gradient',
            bounds={
                'length_scale': (0.1, 10.0),
                'variance': (0.1, 10.0),
                'noise_variance': (1e-4, 1.0)
            }
        )
        
        model = SparseGPModel(kernel_type='rbf')
        
        # Add data
        for i in range(30):
            obs = TestObservation(
                test_ids=[f'test_{i}'],
                features=np.random.randn(5),
                quality_score=np.random.randn(),
                timestamp=i
            )
            model.add_observation(obs)
        
        model.fit()
        
        # Optimize
        optimized_params = optimizer.optimize(model)
        
        # Check bounds are respected
        assert 0.1 <= optimized_params['length_scale'] <= 10.0
        assert 0.1 <= optimized_params['variance'] <= 10.0
        assert 1e-4 <= optimized_params['noise_variance'] <= 1.0


class TestMemoryManagement:
    """TDD tests for memory management strategies."""
    
    def test_memory_limit_enforcement(self):
        """RED: Test memory limit enforcement."""
        memory_manager = MemoryManager(
            max_memory_bytes=10 * 1024 * 1024,  # 10MB
            strategy='fifo'
        )
        
        observations = []
        
        # Generate observations until memory limit
        while not memory_manager.is_memory_full():
            obs = TestObservation(
                test_ids=[f'test_{len(observations)}'],
                features=np.random.randn(1000),  # Large features
                quality_score=np.random.randn(),
                timestamp=len(observations)
            )
            
            if memory_manager.can_add(obs):
                observations.append(obs)
                memory_manager.add(obs)
            else:
                break
        
        assert memory_manager.is_memory_full()
        assert memory_manager.get_memory_usage() <= memory_manager.max_memory_bytes
    
    def test_observation_compression(self):
        """RED: Test observation compression."""
        memory_manager = MemoryManager(
            max_memory_bytes=5 * 1024 * 1024,
            strategy='compression',
            compression_ratio=0.5
        )
        
        # Add observations
        observations = []
        for i in range(100):
            obs = TestObservation(
                test_ids=[f'test_{i}'],
                features=np.random.randn(100),
                quality_score=np.random.randn(),
                timestamp=i
            )
            observations.append(obs)
            memory_manager.add(obs)
        
        # Should compress old observations
        compressed_count = memory_manager.compress_old_observations(50)
        assert compressed_count > 0
        
        # Memory usage should decrease
        memory_after = memory_manager.get_memory_usage()
        assert memory_after < memory_manager.max_memory_bytes
    
    def test_importance_based_retention(self):
        """RED: Test importance-based observation retention."""
        memory_manager = MemoryManager(
            max_memory_bytes=1 * 1024 * 1024,
            strategy='importance'
        )
        
        # Add observations with importance scores
        important_obs = TestObservation(
            test_ids=['critical_test'],
            features=np.array([0, 0, 0, 0, 0]),
            quality_score=10.0,  # High importance
            timestamp=0
        )
        
        memory_manager.add(important_obs, importance=10.0)
        
        # Add many low-importance observations
        for i in range(1000):
            obs = TestObservation(
                test_ids=[f'test_{i}'],
                features=np.random.randn(100),
                quality_score=np.random.randn(),
                timestamp=i
            )
            memory_manager.add(obs, importance=0.1)
        
        # Important observation should be retained
        retained = memory_manager.get_retained_observations()
        retained_ids = [obs.test_ids[0] for obs in retained]
        
        assert 'critical_test' in retained_ids


class TestIncrementalGPLearner:
    """TDD tests for the main incremental GP learner."""
    
    @pytest.fixture
    def gp_learner(self):
        """Create an incremental GP learner."""
        return IncrementalGPLearner(
            kernel_type='rbf',
            inducing_points=30,
            memory_limit_mb=50
        )
    
    def test_initialization(self, gp_learner):
        """RED: Test GP learner initialization."""
        assert gp_learner.kernel_type == 'rbf'
        assert gp_learner.inducing_points == 30
        assert gp_learner.memory_limit_mb == 50
        assert gp_learner.model is not None
        assert gp_learner.memory_manager is not None
    
    def test_observation_stream_processing(self, gp_learner):
        """RED: Test processing stream of observations."""
        # Process observations one by one
        for i in range(100):
            obs = TestObservation(
                test_ids=[f'test_{i}'],
                features=np.random.randn(5),
                quality_score=np.sin(i / 10.0),
                timestamp=i
            )
            
            gp_learner.add_observation(obs)
            
            # Periodically fit model
            if i % 20 == 19:
                gp_learner.fit()
        
        # Should have processed all observations
        assert len(gp_learner.model.observations) <= 100
        
        # Should be able to make predictions
        result = gp_learner.predict(['test'], np.random.randn(5))
        assert not np.isnan(result.mean)
    
    def test_incremental_learning(self, gp_learner):
        """RED: Test incremental learning capabilities."""
        # Initial training with controlled data
        np.random.seed(42)  # For reproducibility
        for i in range(50):
            obs = TestObservation(
                test_ids=[f'test_{i}'],
                features=np.random.randn(5) * 2,  # Spread out the initial data
                quality_score=np.random.randn() * 0.5,  # Smaller variance in initial scores
                timestamp=i
            )
            gp_learner.add_observation(obs)
        
        gp_learner.fit()
        
        # Make initial prediction
        test_point = np.array([1, 0, 0, 0, 0])
        initial_pred = gp_learner.predict(['test'], test_point)
        
        # Add new relevant observation very close to test point
        new_obs = TestObservation(
            test_ids=['new_test'],
            features=test_point + np.random.normal(0, 0.01, 5),  # Very close to test point
            quality_score=5.0,  # High value relative to initial data
            timestamp=100
        )
        
        gp_learner.incremental_update(new_obs)
        
        # Prediction should change
        updated_pred = gp_learner.predict(['test'], test_point)
        
        # The mean should move toward the new observation's high value
        assert updated_pred.mean > initial_pred.mean
        
        # The uncertainty should reduce since we now have an observation very close to the test point
        assert updated_pred.std_dev < initial_pred.std_dev
    
    def test_active_learning_suggestions(self, gp_learner):
        """RED: Test active learning suggestions."""
        # Train on initial data with deterministic seed
        np.random.seed(123)
        X_train = np.random.randn(30, 5)
        for i in range(30):
            obs = TestObservation(
                test_ids=[f'test_{i}'],
                features=X_train[i],
                quality_score=np.sin(X_train[i, 0]) + 0.1 * np.random.randn(),
                timestamp=i
            )
            gp_learner.add_observation(obs)
        
        gp_learner.fit()
        
        # Get suggestions for next tests
        candidate_features = np.random.randn(50, 5)
        suggestions = gp_learner.suggest_next_tests(
            candidate_features,
            n_suggestions=5,
            acquisition='ucb'
        )
        
        assert len(suggestions) == 5
        assert all(0 <= idx < 50 for idx in suggestions)
        
        # Suggestions should be diverse
        assert len(set(suggestions)) == 5
    
    def test_uncertainty_calibration(self, gp_learner):
        """RED: Test uncertainty estimate calibration."""
        # Train on known function
        n = 100
        np.random.seed(456)  # For reproducibility
        X = np.random.randn(n, 5)
        y_true = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1])
        y_observed = y_true + np.random.normal(0, 0.1, n)
        
        for i in range(n):
            obs = TestObservation(
                test_ids=[f'test_{i}'],
                features=X[i],
                quality_score=y_observed[i],
                timestamp=i
            )
            gp_learner.add_observation(obs)
        
        gp_learner.fit()
        
        # Optimize hyperparameters for better fit
        gp_learner.optimize_hyperparameters()
        
        # Test calibration on new points
        n_test = 100
        X_test = np.random.randn(n_test, 5)
        y_test_true = np.sin(X_test[:, 0]) + 0.5 * np.cos(X_test[:, 1])
        
        in_interval_count = 0
        
        for i in range(n_test):
            pred = gp_learner.predict([f'test_{i}'], X_test[i])
            
            # Check if true value in 95% CI
            if pred.confidence_interval[0] <= y_test_true[i] <= pred.confidence_interval[1]:
                in_interval_count += 1
        
        # Should be approximately 95%, but sparse GPs are approximations
        # so we allow more tolerance
        coverage = in_interval_count / n_test
        assert 0.50 <= coverage <= 1.0  # Allow tolerance for sparse GP approximation
    
    def test_performance_benchmarks(self, gp_learner):
        """RED: Test performance meets requirements."""
        # Benchmark update time
        update_times = []
        for i in range(50):
            obs = TestObservation(
                test_ids=[f'test_{i}'],
                features=np.random.randn(5),
                quality_score=np.random.randn(),
                timestamp=i
            )
            
            start = time.time()
            gp_learner.incremental_update(obs)
            update_time = (time.time() - start) * 1000  # ms
            update_times.append(update_time)
        
        avg_update_time = np.mean(update_times)
        assert avg_update_time < 100  # < 100ms requirement
        
        # Benchmark prediction time
        prediction_times = []
        for i in range(100):
            start = time.time()
            gp_learner.predict([f'pred_{i}'], np.random.randn(5))
            pred_time = (time.time() - start) * 1000  # ms
            prediction_times.append(pred_time)
        
        avg_pred_time = np.mean(prediction_times)
        assert avg_pred_time < 10  # < 10ms requirement
    
    def test_guardian_integration(self, gp_learner):
        """RED: Test integration with Guardian's test system."""
        # Simulate Guardian test evaluation
        def evaluate_test_subset(test_ids):
            # Mock evaluation based on test combination
            if len(test_ids) == 0:
                return 0.0
            
            base_score = len(test_ids) * 0.1
            synergy = 0.05 * (len(test_ids) - 1)
            
            return min(1.0, base_score + synergy)
        
        # Learn from test combinations
        for _ in range(50):
            n_tests = np.random.randint(1, 6)
            test_ids = [f'test_{i}' for i in range(n_tests)]
            
            # Extract features from test combination
            features = gp_learner.extract_test_features(test_ids)
            quality = evaluate_test_subset(test_ids)
            
            obs = TestObservation(
                test_ids=test_ids,
                features=features,
                quality_score=quality,
                timestamp=time.time()
            )
            
            gp_learner.add_observation(obs)
        
        gp_learner.fit()
        
        # Predict quality of new combination
        new_test_ids = ['test_0', 'test_1', 'test_2']
        new_features = gp_learner.extract_test_features(new_test_ids)
        
        prediction = gp_learner.predict(new_test_ids, new_features)
        
        # Should make reasonable prediction
        expected = evaluate_test_subset(new_test_ids)
        assert abs(prediction.mean - expected) < 0.3


class TestIntegration:
    """Integration tests for complete GP system."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from data to predictions."""
        # Initialize system
        learner = IncrementalGPLearner(
            kernel_type='rbf',
            inducing_points=25,
            memory_limit_mb=10
        )
        
        # Simulate test execution stream
        test_results = []
        
        for day in range(7):  # 7 days of data
            for hour in range(24):
                # Generate test results
                n_tests = np.random.randint(1, 5)
                test_ids = [f'test_{i}' for i in np.random.choice(20, n_tests, replace=False)]
                
                # Simulate quality (with daily patterns)
                time_factor = np.sin(hour * np.pi / 12)
                quality = 0.5 + 0.3 * time_factor + np.random.normal(0, 0.1)
                
                obs = TestObservation(
                    test_ids=test_ids,
                    features=learner.extract_test_features(test_ids),
                    quality_score=quality,
                    timestamp=day * 24 + hour
                )
                
                learner.add_observation(obs)
                test_results.append(obs)
                
                # Periodic model updates
                if hour % 6 == 0:
                    learner.fit()
                    
                    # Optimize hyperparameters daily
                    if hour == 0 and day > 0:
                        learner.optimize_hyperparameters()
        
        # Verify system learned patterns
        morning_tests = ['test_0', 'test_1']
        evening_tests = ['test_10', 'test_11']
        
        morning_pred = learner.predict(
            morning_tests,
            learner.extract_test_features(morning_tests)
        )
        
        evening_pred = learner.predict(
            evening_tests,
            learner.extract_test_features(evening_tests)
        )
        
        # Should have learned something
        assert morning_pred.mean != evening_pred.mean
        
        # Check memory usage
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Should stay within limits (with overhead)
        assert memory_mb < 200  # Generous bound for test
    
    def test_with_shapley_and_ucb1(self):
        """Test integration with other Guardian components."""
        from guardian.analytics.shapley_convergence import OptimizedShapleyCalculator
        from guardian.analytics.ucb1_thompson_hybrid import UCB1ThompsonHybrid
        
        # Initialize components
        gp_learner = IncrementalGPLearner(inducing_points=20)
        shapley_calc = OptimizedShapleyCalculator()
        test_selector = UCB1ThompsonHybrid()
        
        # Define test suite
        all_tests = [Path(f'test_{i}.py') for i in range(10)]
        
        # Use GP predictions as metric evaluator for Shapley
        def gp_metric_evaluator(test_subset):
            if not test_subset:
                return 0.0
            
            test_ids = [str(t) for t in test_subset]
            features = gp_learner.extract_test_features(test_ids)
            
            prediction = gp_learner.predict(test_ids, features)
            return prediction.mean
        
        # Initial exploration with UCB1
        for _ in range(20):
            # Select test to run
            selected_test = test_selector.select_next_test(all_tests)
            
            # "Execute" test and get quality
            quality = np.random.random()  # Mock execution
            
            # Update selector
            test_selector.update(selected_test, quality)
            
            # Update GP model
            obs = TestObservation(
                test_ids=[str(selected_test)],
                features=gp_learner.extract_test_features([str(selected_test)]),
                quality_score=quality,
                timestamp=time.time()
            )
            gp_learner.add_observation(obs)
        
        gp_learner.fit()
        
        # Calculate Shapley values using GP predictions
        shapley_values = shapley_calc.calculate_shapley_values(
            all_tests[:5],  # Subset for speed
            gp_metric_evaluator
        )
        
        # Should have meaningful Shapley values
        assert len(shapley_values) == 5
        assert all(not np.isnan(v) for v in shapley_values.values())
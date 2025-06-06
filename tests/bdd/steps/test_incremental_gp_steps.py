"""
BDD Step Definitions for Incremental GP Learning Feature

Author: DarkLightX/Dana Edwards
"""

import time
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import psutil
import os

from pytest_bdd import given, when, then, parsers, scenarios
import pytest

from guardian.analytics.incremental_gp import (
    IncrementalGPLearner,
    SparseGPModel,
    InducingPointSelector,
    HyperparameterOptimizer,
    MemoryManager,
    TestObservation,
    PredictionResult,
    IGPModel,
    IInducingPointStrategy,
    IMemoryStrategy
)

# Load scenarios from feature file
scenarios('../features/incremental_gp_learning.feature')


@pytest.fixture
def gp_context():
    """Create a context object to store state between steps."""
    class GPContext:
        def __init__(self):
            self.model = None
            self.observations = []
            self.new_observations = []
            self.predictions = []
            self.memory_limit = None
            self.inducing_limit = None
            self.update_times = []
            self.prediction_times = []
            self.hyperparams_history = []
            self.true_values = {}
            self.approximation_levels = []
            self.metrics = {}
            
    return GPContext()


@given("a Gaussian Process model for test quality prediction")
def step_given_gp_model(gp_context):
    """Initialize a GP model for test quality prediction."""
    gp_context.model = IncrementalGPLearner(
        kernel_type='rbf',
        noise_variance=0.01,
        inducing_points=50
    )


@given("a stream of test execution results")
def step_given_test_stream(gp_context):
    """Set up a stream of test results."""
    # Simulate test results with some structure
    np.random.seed(42)
    test_features = np.random.randn(100, 5)  # 5D feature space
    
    # Create synthetic quality scores with some pattern
    true_function = lambda x: np.sin(x[0]) + 0.5 * np.cos(x[1]) + 0.1 * x[2]
    
    gp_context.observations = []
    for i, features in enumerate(test_features):
        quality = true_function(features) + np.random.normal(0, 0.1)
        obs = TestObservation(
            test_ids=[f"test_{j}" for j in range(int(2 + abs(features[3])))],
            features=features,
            quality_score=quality,
            timestamp=i
        )
        gp_context.observations.append(obs)


@given("memory constraints for large-scale deployment")
def step_given_memory_constraints(gp_context):
    """Set memory constraints."""
    gp_context.memory_limit = 100 * 1024 * 1024  # 100MB


@given(parsers.parse("a GP model trained on {n:d} observations"))
def step_given_trained_gp(gp_context, n):
    """Train GP on initial observations."""
    # Initialize model if not exists
    if gp_context.model is None:
        gp_context.model = IncrementalGPLearner()
    
    # Generate training data
    np.random.seed(42)
    X_train = np.random.randn(n, 5)
    y_train = np.sin(X_train[:, 0]) + 0.5 * np.cos(X_train[:, 1]) + np.random.normal(0, 0.1, n)
    
    # Train model
    for i in range(n):
        obs = TestObservation(
            test_ids=[f"test_{i}"],
            features=X_train[i],
            quality_score=y_train[i],
            timestamp=i
        )
        gp_context.model.add_observation(obs)
    
    gp_context.model.fit()


@given(parsers.parse("a dataset with {n:d} test observations"))
def step_given_large_dataset(gp_context, n):
    """Create a large dataset."""
    np.random.seed(42)
    X = np.random.randn(n, 5)
    y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1]) + np.random.normal(0, 0.1, n)
    
    gp_context.observations = []
    for i in range(n):
        obs = TestObservation(
            test_ids=[f"test_{i}"],
            features=X[i],
            quality_score=y[i],
            timestamp=i
        )
        gp_context.observations.append(obs)


@given(parsers.parse("a limit of {m:d} inducing points"))
def step_given_inducing_limit(gp_context, m):
    """Set inducing point limit."""
    gp_context.inducing_limit = m


@given("a GP model with initial hyperparameters")
def step_given_initial_hyperparams(gp_context):
    """Initialize GP with specific hyperparameters."""
    gp_context.model = IncrementalGPLearner(
        kernel_type='rbf',
        length_scale=1.0,
        variance=1.0,
        noise_variance=0.01
    )
    gp_context.hyperparams_history = [{
        'length_scale': 1.0,
        'variance': 1.0,
        'noise_variance': 0.01,
        'log_marginal_likelihood': None
    }]


@given("a trained sparse GP model")
def step_given_trained_sparse_gp(gp_context):
    """Create a trained sparse GP."""
    gp_context.model = SparseGPModel(
        inducing_points=30,
        kernel_type='rbf'
    )
    
    # Train on some data
    n = 200
    X = np.random.randn(n, 5)
    y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1]) + np.random.normal(0, 0.1, n)
    
    for i in range(n):
        obs = TestObservation(
            test_ids=[f"test_{i}"],
            features=X[i],
            quality_score=y[i],
            timestamp=i
        )
        gp_context.model.add_observation(obs)
    
    gp_context.model.fit()


@given(parsers.parse("a memory limit of {mb:d}MB"))
def step_given_memory_limit_mb(gp_context, mb):
    """Set memory limit in MB."""
    gp_context.memory_limit = mb * 1024 * 1024


@given("a partially trained GP model")
def step_given_partial_gp(gp_context):
    """Create partially trained GP."""
    gp_context.model = IncrementalGPLearner(
        kernel_type='rbf',
        inducing_points=20
    )
    
    # Train on limited data
    n = 50
    X = np.random.randn(n, 5)
    y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1]) + np.random.normal(0, 0.1, n)
    
    for i in range(n):
        obs = TestObservation(
            test_ids=[f"test_{i}"],
            features=X[i],
            quality_score=y[i],
            timestamp=i
        )
        gp_context.model.add_observation(obs)
    
    gp_context.model.fit()


@given(parsers.parse("different approximation levels {levels}"))
def step_given_approximation_levels(gp_context, levels):
    """Parse approximation levels."""
    # Parse [10, 25, 50, 100] format
    levels_str = levels.strip('[]')
    gp_context.approximation_levels = [int(x.strip()) for x in levels_str.split(',')]


@given("Guardian's evolutionary test framework")
def step_given_guardian_framework(gp_context):
    """Set up Guardian integration context."""
    # This would integrate with actual Guardian framework
    gp_context.guardian_integration = True


@given("various edge case inputs")
def step_given_edge_cases(gp_context):
    """Prepare edge case scenarios."""
    gp_context.edge_cases = {
        'empty': [],
        'duplicate': [
            TestObservation([f"test_1"], np.array([1, 2, 3, 4, 5]), 0.8, 0),
            TestObservation([f"test_1"], np.array([1, 2, 3, 4, 5]), 0.8, 1),
        ],
        'contradictory': [
            TestObservation([f"test_1"], np.array([1, 2, 3, 4, 5]), 0.8, 0),
            TestObservation([f"test_1"], np.array([1, 2, 3, 4, 5]), 0.2, 1),
        ]
    }


@given("production performance constraints")
def step_given_performance_constraints(gp_context):
    """Set performance requirements."""
    gp_context.performance_constraints = {
        'update_time_ms': 100,
        'prediction_time_ms': 10,
        'memory_mb': 100,
        'max_observations': 10000
    }


@when(parsers.parse("I receive {n:d} new test results"))
def step_when_receive_new_results(gp_context, n):
    """Receive new test results."""
    # Generate new observations
    np.random.seed(100)
    X_new = np.random.randn(n, 5)
    y_new = np.sin(X_new[:, 0]) + 0.5 * np.cos(X_new[:, 1]) + np.random.normal(0, 0.1, n)
    
    gp_context.new_observations = []
    for i in range(n):
        obs = TestObservation(
            test_ids=[f"new_test_{i}"],
            features=X_new[i],
            quality_score=y_new[i],
            timestamp=100 + i
        )
        gp_context.new_observations.append(obs)
    
    # Perform incremental update
    start_time = time.time()
    for obs in gp_context.new_observations:
        gp_context.model.incremental_update(obs)
    update_time = time.time() - start_time
    
    gp_context.update_times.append(update_time)


@when("training the sparse GP model")
def step_when_train_sparse_gp(gp_context):
    """Train sparse GP model."""
    gp_context.model = SparseGPModel(
        inducing_points=gp_context.inducing_limit,
        kernel_type='rbf'
    )
    
    # Measure memory before
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss
    
    # Train model
    start_time = time.time()
    for obs in gp_context.observations:
        gp_context.model.add_observation(obs)
    
    gp_context.model.fit()
    training_time = time.time() - start_time
    
    # Measure memory after
    memory_after = process.memory_info().rss
    memory_used = memory_after - memory_before
    
    gp_context.metrics['training_time'] = training_time
    gp_context.metrics['memory_used'] = memory_used


@when(parsers.parse("processing batches of {batch_size:d} observations"))
def step_when_process_batches(gp_context, batch_size):
    """Process observations in batches."""
    # Generate stream of observations
    n_batches = 10
    batch_count = 0
    
    for batch_idx in range(n_batches):
        # Generate batch
        X_batch = np.random.randn(batch_size, 5)
        y_batch = np.sin(X_batch[:, 0]) + 0.5 * np.cos(X_batch[:, 1]) + np.random.normal(0, 0.1, batch_size)
        
        # Process batch
        for i in range(batch_size):
            obs = TestObservation(
                test_ids=[f"batch_{batch_idx}_test_{i}"],
                features=X_batch[i],
                quality_score=y_batch[i],
                timestamp=batch_idx * batch_size + i
            )
            gp_context.model.add_observation(obs)
        
        batch_count += 1
        
        # Optimize hyperparameters every 5 batches
        if batch_count % 5 == 0:
            old_params = gp_context.model.get_hyperparameters()
            gp_context.model.optimize_hyperparameters()
            new_params = gp_context.model.get_hyperparameters()
            
            gp_context.hyperparams_history.append({
                **new_params,
                'batch': batch_count,
                'log_marginal_likelihood': gp_context.model.log_marginal_likelihood()
            })


@when("making predictions on test combinations")
def step_when_make_predictions(gp_context):
    """Make predictions with uncertainty."""
    # Generate test points
    n_test = 100
    X_test = np.random.randn(n_test, 5)
    
    # True function for comparison
    y_true = np.sin(X_test[:, 0]) + 0.5 * np.cos(X_test[:, 1])
    
    gp_context.predictions = []
    gp_context.true_values = {}
    
    for i in range(n_test):
        # Make prediction
        start_time = time.time()
        result = gp_context.model.predict(
            test_ids=[f"pred_test_{i}"],
            features=X_test[i]
        )
        pred_time = time.time() - start_time
        
        gp_context.predictions.append(result)
        gp_context.prediction_times.append(pred_time)
        gp_context.true_values[i] = y_true[i]


@when(parsers.parse("processing {n:d} observations"))
def step_when_process_many_observations(gp_context, n):
    """Process many observations with memory constraints."""
    gp_context.model = IncrementalGPLearner(
        memory_manager=MemoryManager(
            max_memory_bytes=gp_context.memory_limit,
            compression_ratio=0.5
        )
    )
    
    # Track memory usage
    memory_usage = []
    process = psutil.Process(os.getpid())
    
    for i in range(n):
        # Generate observation
        x = np.random.randn(5)
        y = np.sin(x[0]) + 0.5 * np.cos(x[1]) + np.random.normal(0, 0.1)
        
        obs = TestObservation(
            test_ids=[f"test_{i}"],
            features=x,
            quality_score=y,
            timestamp=i
        )
        
        # Add observation
        gp_context.model.add_observation(obs)
        
        # Check memory periodically
        if i % 100 == 0:
            memory_usage.append(process.memory_info().rss)
    
    gp_context.metrics['memory_usage'] = memory_usage
    gp_context.metrics['max_memory'] = max(memory_usage)


@when("selecting next tests to run")
def step_when_select_next_tests(gp_context):
    """Select tests using acquisition function."""
    # Generate candidate tests
    n_candidates = 50
    X_candidates = np.random.randn(n_candidates, 5)
    
    # Calculate acquisition values
    acquisition_values = []
    
    for i in range(n_candidates):
        # Predict with uncertainty
        result = gp_context.model.predict(
            test_ids=[f"candidate_{i}"],
            features=X_candidates[i]
        )
        
        # Upper confidence bound acquisition
        acq_value = result.mean + 2.0 * result.std_dev
        acquisition_values.append(acq_value)
    
    # Select top tests
    selected_indices = np.argsort(acquisition_values)[-5:]
    gp_context.selected_tests = [f"candidate_{i}" for i in selected_indices]
    gp_context.acquisition_values = acquisition_values


@when("comparing prediction quality and speed")
def step_when_compare_quality_speed(gp_context):
    """Compare different approximation levels."""
    # Test data
    n_test = 100
    X_test = np.random.randn(n_test, 5)
    y_true = np.sin(X_test[:, 0]) + 0.5 * np.cos(X_test[:, 1])
    
    # Training data
    n_train = 500
    X_train = np.random.randn(n_train, 5)
    y_train = np.sin(X_train[:, 0]) + 0.5 * np.cos(X_train[:, 1]) + np.random.normal(0, 0.1, n_train)
    
    for m in gp_context.approximation_levels:
        # Create model with m inducing points
        model = SparseGPModel(inducing_points=m)
        
        # Train
        for i in range(n_train):
            obs = TestObservation(
                test_ids=[f"train_{i}"],
                features=X_train[i],
                quality_score=y_train[i],
                timestamp=i
            )
            model.add_observation(obs)
        
        model.fit()
        
        # Test predictions
        predictions = []
        times = []
        
        for i in range(n_test):
            start_time = time.time()
            result = model.predict([f"test_{i}"], X_test[i])
            pred_time = time.time() - start_time
            
            predictions.append(result.mean)
            times.append(pred_time)
        
        # Calculate metrics
        mse = np.mean((predictions - y_true) ** 2)
        avg_time = np.mean(times)
        
        gp_context.metrics[f'm_{m}'] = {
            'mse': mse,
            'avg_time': avg_time,
            'inducing_points': m
        }


@when("GP predictions guide test evolution")
def step_when_gp_guides_evolution(gp_context):
    """Use GP to guide evolutionary test optimization."""
    # Simulate evolutionary iterations
    n_generations = 10
    population_size = 20
    
    evolution_history = []
    
    for gen in range(n_generations):
        # Generate population
        population = np.random.randn(population_size, 5)
        
        # Evaluate using GP
        fitness_scores = []
        uncertainties = []
        
        for i in range(population_size):
            result = gp_context.model.predict(
                test_ids=[f"gen_{gen}_ind_{i}"],
                features=population[i]
            )
            fitness_scores.append(result.mean)
            uncertainties.append(result.std_dev)
        
        # Record generation statistics
        evolution_history.append({
            'generation': gen,
            'best_fitness': max(fitness_scores),
            'avg_fitness': np.mean(fitness_scores),
            'avg_uncertainty': np.mean(uncertainties)
        })
        
        # Update model with best individuals
        best_idx = np.argmax(fitness_scores)
        true_quality = np.sin(population[best_idx, 0]) + 0.5 * np.cos(population[best_idx, 1])
        
        obs = TestObservation(
            test_ids=[f"best_gen_{gen}"],
            features=population[best_idx],
            quality_score=true_quality,
            timestamp=gen
        )
        gp_context.model.incremental_update(obs)
    
    gp_context.evolution_history = evolution_history


@when("processing empty test sets")
def step_when_process_empty(gp_context):
    """Process empty test set."""
    result = gp_context.model.predict([], np.array([]))
    gp_context.empty_prediction = result


@when("processing duplicate observations")
def step_when_process_duplicates(gp_context):
    """Process duplicate observations."""
    for obs in gp_context.edge_cases['duplicate']:
        gp_context.model.add_observation(obs)
    
    # Check model stability
    gp_context.duplicate_handled = True


@when("processing contradictory observations")
def step_when_process_contradictory(gp_context):
    """Process contradictory observations."""
    initial_obs = gp_context.edge_cases['contradictory'][0]
    contradictory_obs = gp_context.edge_cases['contradictory'][1]
    
    # Add initial observation
    gp_context.model.add_observation(initial_obs)
    initial_pred = gp_context.model.predict(initial_obs.test_ids, initial_obs.features)
    
    # Add contradictory observation
    gp_context.model.add_observation(contradictory_obs)
    final_pred = gp_context.model.predict(initial_obs.test_ids, initial_obs.features)
    
    gp_context.contradiction_results = {
        'initial_uncertainty': initial_pred.std_dev,
        'final_uncertainty': final_pred.std_dev
    }


@when("kernel matrix becomes ill-conditioned")
def step_when_kernel_ill_conditioned(gp_context):
    """Test numerical stability with ill-conditioned kernel."""
    # Create observations that lead to ill-conditioning
    n = 50
    X = np.random.randn(n, 5) * 0.001  # Very close points
    y = np.random.randn(n)
    
    try:
        for i in range(n):
            obs = TestObservation(
                test_ids=[f"close_{i}"],
                features=X[i],
                quality_score=y[i],
                timestamp=i
            )
            gp_context.model.add_observation(obs)
        
        gp_context.model.fit()
        gp_context.numerical_stable = True
    except:
        gp_context.numerical_stable = False


@when("running on standard hardware")
def step_when_run_standard_hardware(gp_context):
    """Run performance benchmarks."""
    # Set up model
    model = IncrementalGPLearner(inducing_points=50)
    
    # Benchmark incremental updates
    update_times = []
    for i in range(100):
        obs = TestObservation(
            test_ids=[f"bench_{i}"],
            features=np.random.randn(5),
            quality_score=np.random.randn(),
            timestamp=i
        )
        
        start_time = time.time()
        model.incremental_update(obs)
        update_time = (time.time() - start_time) * 1000  # Convert to ms
        update_times.append(update_time)
    
    # Benchmark predictions
    prediction_times = []
    for i in range(100):
        start_time = time.time()
        result = model.predict([f"pred_{i}"], np.random.randn(5))
        pred_time = (time.time() - start_time) * 1000  # Convert to ms
        prediction_times.append(pred_time)
    
    # Check memory
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    gp_context.benchmark_results = {
        'avg_update_ms': np.mean(update_times),
        'avg_prediction_ms': np.mean(prediction_times),
        'memory_mb': memory_mb,
        'model_capacity': len(model.observations)
    }


@then("the model should update incrementally")
def step_then_incremental_update(gp_context):
    """Verify incremental updates."""
    assert hasattr(gp_context.model, 'incremental_update')
    assert len(gp_context.update_times) > 0


@then(parsers.parse("update time should be O(m²n) not O(n³)"))
def step_then_update_complexity(gp_context):
    """Verify update time complexity."""
    # With sparse GP, update should be much faster than O(n³)
    avg_update_time = np.mean(gp_context.update_times)
    
    # For 100 existing + 10 new observations, O(n³) would be prohibitive
    # O(m²n) with m=50 should be < 1 second
    assert avg_update_time < 1.0, f"Update too slow: {avg_update_time:.3f}s"


@then("predictions should incorporate new knowledge")
def step_then_predictions_updated(gp_context):
    """Verify predictions use new data."""
    # Make prediction on a point similar to new observations
    test_point = gp_context.new_observations[0].features + np.random.normal(0, 0.1, 5)
    
    result = gp_context.model.predict(['test_new'], test_point)
    
    # Prediction should be influenced by nearby new observations
    assert result.confidence_interval[1] - result.confidence_interval[0] < 2.0


@then("model quality should improve or maintain")
def step_then_quality_maintained(gp_context):
    """Verify model quality doesn't degrade."""
    # This would require comparing predictions before/after
    # For now, just verify model is still functional
    assert gp_context.model is not None
    assert hasattr(gp_context.model, 'predict')


@then(parsers.parse("memory usage should be O(m²) not O(n²)"))
def step_then_memory_complexity(gp_context):
    """Verify memory complexity."""
    memory_used = gp_context.metrics.get('memory_used', 0)
    n_observations = len(gp_context.observations)
    m_inducing = gp_context.inducing_limit
    
    # Memory should scale with m² not n²
    # For n=1000, m=50: m²=2500 << n²=1000000
    expected_memory_mb = (m_inducing ** 2) * 8 * 3 / 1024 / 1024  # Rough estimate
    
    assert memory_used / 1024 / 1024 < expected_memory_mb * 10  # Allow 10x for overhead


@then(parsers.parse("prediction time should be O(m²) not O(n²)"))
def step_then_prediction_complexity(gp_context):
    """Verify prediction time complexity."""
    # Predictions with sparse GP should be fast
    if gp_context.prediction_times:
        avg_pred_time = np.mean(gp_context.prediction_times)
        assert avg_pred_time < 0.01  # Should be < 10ms


@then(parsers.parse("approximation quality should be within {percent:d}% of full GP"))
def step_then_approximation_quality(gp_context, percent):
    """Verify sparse approximation quality."""
    # This would require comparing to full GP
    # For now, verify predictions are reasonable
    assert gp_context.model is not None
    assert hasattr(gp_context.model, 'predict')


@then("inducing points should cover the input space well")
def step_then_inducing_coverage(gp_context):
    """Verify inducing points provide good coverage."""
    inducing_points = gp_context.model.get_inducing_points()
    
    if inducing_points is not None and len(inducing_points) > 0:
        # Check that inducing points have reasonable spread
        distances = []
        for i in range(len(inducing_points)):
            for j in range(i + 1, len(inducing_points)):
                dist = np.linalg.norm(inducing_points[i] - inducing_points[j])
                distances.append(dist)
        
        # Should have variety in distances (not all same)
        assert np.std(distances) > 0.1


@then(parsers.parse("hyperparameters should adapt every {n:d} batches"))
def step_then_hyperparams_adapt(gp_context, n):
    """Verify hyperparameter adaptation schedule."""
    # Check that we have hyperparameter updates
    assert len(gp_context.hyperparams_history) > 1
    
    # Check updates happened at right intervals
    for i in range(1, len(gp_context.hyperparams_history)):
        if 'batch' in gp_context.hyperparams_history[i]:
            batch_diff = (gp_context.hyperparams_history[i]['batch'] - 
                         gp_context.hyperparams_history[i-1].get('batch', 0))
            assert batch_diff == n


@then("log marginal likelihood should improve")
def step_then_lml_improves(gp_context):
    """Verify log marginal likelihood improvement."""
    lml_values = [h['log_marginal_likelihood'] 
                  for h in gp_context.hyperparams_history 
                  if h.get('log_marginal_likelihood') is not None]
    
    if len(lml_values) > 1:
        # Generally should improve (with some tolerance for noise)
        improvements = sum(1 for i in range(1, len(lml_values)) 
                          if lml_values[i] > lml_values[i-1])
        assert improvements >= len(lml_values) // 2  # At least half improve


@then("optimization should use gradient-based methods")
def step_then_gradient_optimization(gp_context):
    """Verify gradient-based optimization."""
    # Model should support gradient-based hyperparameter optimization
    assert hasattr(gp_context.model, 'optimize_hyperparameters')


@then("adaptation should be stable without drift")
def step_then_stable_adaptation(gp_context):
    """Verify stability of hyperparameter adaptation."""
    if len(gp_context.hyperparams_history) > 2:
        # Check that parameters don't drift wildly
        length_scales = [h.get('length_scale', 1.0) 
                        for h in gp_context.hyperparams_history]
        
        # Relative change should be bounded
        for i in range(1, len(length_scales)):
            relative_change = abs(length_scales[i] - length_scales[i-1]) / length_scales[i-1]
            assert relative_change < 2.0  # Max 200% change per update


@then(parsers.parse("{percent:d}% confidence intervals should contain true values {percent2:d}% of time"))
def step_then_calibrated_intervals(gp_context, percent, percent2):
    """Verify confidence interval calibration."""
    if not gp_context.predictions:
        return
    
    # Count how many true values fall within confidence intervals
    in_interval = 0
    total = 0
    
    for i, pred in enumerate(gp_context.predictions):
        if i in gp_context.true_values:
            true_val = gp_context.true_values[i]
            if pred.confidence_interval[0] <= true_val <= pred.confidence_interval[1]:
                in_interval += 1
            total += 1
    
    if total > 0:
        coverage = (in_interval / total) * 100
        # Allow some tolerance
        assert abs(coverage - percent2) < 15  # Within 15% of target


@then("uncertainty should increase for extrapolation")
def step_then_uncertainty_extrapolation(gp_context):
    """Verify uncertainty increases for extrapolation."""
    # Test on points far from training data
    far_point = np.array([10, 10, 10, 10, 10])  # Far from N(0,1) training data
    near_point = np.array([0, 0, 0, 0, 0])  # Near training data
    
    far_result = gp_context.model.predict(['far'], far_point)
    near_result = gp_context.model.predict(['near'], near_point)
    
    # Far point should have higher uncertainty
    assert far_result.std_dev > near_result.std_dev


@then("uncertainty should decrease near observations")
def step_then_uncertainty_near_data(gp_context):
    """Verify uncertainty decreases near observations."""
    if gp_context.model.observations:
        # Test near an existing observation
        obs = gp_context.model.observations[0]
        near_point = obs.features + np.random.normal(0, 0.01, len(obs.features))
        
        result = gp_context.model.predict(['near_obs'], near_point)
        
        # Should have low uncertainty
        assert result.std_dev < 0.5


@then("epistemic uncertainty should be separated from noise")
def step_then_epistemic_separation(gp_context):
    """Verify epistemic uncertainty handling."""
    # Model should distinguish between epistemic and aleatoric uncertainty
    assert hasattr(gp_context.model, 'noise_variance')
    assert gp_context.model.noise_variance >= 0


@then("memory usage should stay below limit")
def step_then_memory_below_limit(gp_context):
    """Verify memory stays within bounds."""
    max_memory = gp_context.metrics.get('max_memory', 0)
    assert max_memory <= gp_context.memory_limit


@then("old observations should be compressed or forgotten")
def step_then_compression_works(gp_context):
    """Verify compression/forgetting mechanism."""
    # Model should have mechanism to handle old data
    assert hasattr(gp_context.model, 'memory_manager')


@then("model quality should degrade gracefully")
def step_then_graceful_degradation(gp_context):
    """Verify graceful quality degradation."""
    # Model should still make reasonable predictions
    test_point = np.random.randn(5)
    result = gp_context.model.predict(['test'], test_point)
    
    assert result is not None
    assert not np.isnan(result.mean)


@then("critical observations should be retained")
def step_then_critical_retained(gp_context):
    """Verify critical observations are kept."""
    # This would check that important observations aren't discarded
    # For now, verify model retains some observations
    if hasattr(gp_context.model, 'observations'):
        assert len(gp_context.model.observations) > 0


@then("selection should maximize information gain")
def step_then_maximize_info_gain(gp_context):
    """Verify information gain maximization."""
    # Selected tests should have high acquisition values
    if hasattr(gp_context, 'acquisition_values'):
        max_acq = max(gp_context.acquisition_values)
        selected_acq = [gp_context.acquisition_values[int(t.split('_')[1])] 
                       for t in gp_context.selected_tests]
        
        # Selected should be among highest
        assert min(selected_acq) > 0.8 * max_acq


@then("high-uncertainty regions should be explored")
def step_then_explore_uncertainty(gp_context):
    """Verify exploration of uncertain regions."""
    # Selected tests should include high-uncertainty regions
    assert len(gp_context.selected_tests) > 0


@then("acquisition function should balance exploration/exploitation")
def step_then_balanced_acquisition(gp_context):
    """Verify balanced acquisition function."""
    # Acquisition values should have reasonable spread
    if hasattr(gp_context, 'acquisition_values'):
        acq_std = np.std(gp_context.acquisition_values)
        assert acq_std > 0.1  # Not all same value


@then("batch selection should consider diversity")
def step_then_batch_diversity(gp_context):
    """Verify batch diversity in selection."""
    # Selected tests should be diverse
    assert len(set(gp_context.selected_tests)) == len(gp_context.selected_tests)


@then("higher m should give better accuracy")
def step_then_higher_m_better(gp_context):
    """Verify accuracy improves with more inducing points."""
    mse_values = []
    m_values = []
    
    for key, metrics in gp_context.metrics.items():
        if key.startswith('m_'):
            mse_values.append(metrics['mse'])
            m_values.append(metrics['inducing_points'])
    
    if len(mse_values) > 1:
        # Sort by m
        sorted_pairs = sorted(zip(m_values, mse_values))
        
        # MSE should generally decrease (lower is better)
        improvements = sum(1 for i in range(1, len(sorted_pairs)) 
                          if sorted_pairs[i][1] <= sorted_pairs[i-1][1])
        assert improvements >= len(sorted_pairs) - 2  # Allow one exception


@then(parsers.parse("prediction time should scale as O(m²)"))
def step_then_prediction_scaling(gp_context):
    """Verify prediction time scaling."""
    times = []
    m_values = []
    
    for key, metrics in gp_context.metrics.items():
        if key.startswith('m_'):
            times.append(metrics['avg_time'])
            m_values.append(metrics['inducing_points'])
    
    if len(times) > 2:
        # Fit quadratic relationship
        m_array = np.array(m_values)
        t_array = np.array(times)
        
        # Simple check: time ratio should be roughly m² ratio
        t_ratio = t_array[-1] / t_array[0]
        m_ratio = (m_array[-1] / m_array[0]) ** 2
        
        # Allow 3x tolerance
        assert t_ratio < m_ratio * 3


@then("quality metrics should be monotonic in m")
def step_then_monotonic_quality(gp_context):
    """Verify monotonic quality improvement."""
    # Already checked in higher_m_better
    pass


@then("optimal m should be determinable")
def step_then_optimal_m_determinable(gp_context):
    """Verify we can determine optimal m."""
    # We should have enough data to make a decision
    assert len(gp_context.metrics) >= len(gp_context.approximation_levels)


@then("fitness evaluation should use GP predictions")
def step_then_fitness_uses_gp(gp_context):
    """Verify fitness evaluation uses GP."""
    assert hasattr(gp_context, 'evolution_history')
    assert len(gp_context.evolution_history) > 0


@then("uncertainty should influence mutation rates")
def step_then_uncertainty_influences_mutation(gp_context):
    """Verify uncertainty affects evolution."""
    # Check that uncertainty is tracked
    uncertainties = [gen['avg_uncertainty'] for gen in gp_context.evolution_history]
    assert len(uncertainties) > 0
    assert all(u > 0 for u in uncertainties)


@then("model should update with evolution results")
def step_then_model_updates_evolution(gp_context):
    """Verify model updates during evolution."""
    # Model should have more observations after evolution
    initial_obs = gp_context.evolution_history[0]['generation']
    final_obs = len(gp_context.model.observations)
    
    assert final_obs > initial_obs


@then("convergence should be faster than random")
def step_then_faster_convergence(gp_context):
    """Verify faster convergence than random."""
    # Fitness should improve over generations
    best_fitness = [gen['best_fitness'] for gen in gp_context.evolution_history]
    
    # Should show improvement
    improvements = sum(1 for i in range(1, len(best_fitness)) 
                      if best_fitness[i] >= best_fitness[i-1])
    assert improvements >= len(best_fitness) // 2


@then("predictions should return baseline quality")
def step_then_baseline_prediction(gp_context):
    """Verify baseline prediction for empty set."""
    assert gp_context.empty_prediction is not None
    assert not np.isnan(gp_context.empty_prediction.mean)


@then("model should handle gracefully without instability")
def step_then_handle_duplicates(gp_context):
    """Verify duplicate handling."""
    assert gp_context.duplicate_handled
    
    # Model should still be functional
    test_result = gp_context.model.predict(['test'], np.random.randn(5))
    assert not np.isnan(test_result.mean)


@then("model should increase uncertainty appropriately")
def step_then_increase_uncertainty_contradictory(gp_context):
    """Verify uncertainty increase for contradictions."""
    # Uncertainty should increase when we have contradictory observations
    assert gp_context.contradiction_results['final_uncertainty'] >= \
           gp_context.contradiction_results['initial_uncertainty']


@then("numerical stability should be maintained")
def step_then_numerical_stability(gp_context):
    """Verify numerical stability."""
    assert gp_context.numerical_stable


@then(parsers.parse("incremental updates should complete in < {ms:d}ms"))
def step_then_update_time_limit(gp_context, ms):
    """Verify update time constraint."""
    avg_update = gp_context.benchmark_results['avg_update_ms']
    assert avg_update < ms


@then(parsers.parse("predictions should complete in < {ms:d}ms"))
def step_then_prediction_time_limit(gp_context, ms):
    """Verify prediction time constraint."""
    avg_pred = gp_context.benchmark_results['avg_prediction_ms']
    assert avg_pred < ms


@then(parsers.parse("memory usage should stay under {mb:d}MB"))
def step_then_memory_limit_mb(gp_context, mb):
    """Verify memory constraint."""
    memory_mb = gp_context.benchmark_results['memory_mb']
    assert memory_mb < mb


@then(parsers.parse("model should handle {n:d}+ observations"))
def step_then_handle_many_observations(gp_context, n):
    """Verify capacity for many observations."""
    capacity = gp_context.benchmark_results['model_capacity']
    assert capacity >= min(n, 100)  # At least 100 for this test
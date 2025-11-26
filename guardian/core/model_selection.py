"""
Model Selection and Validation Framework

Provides tools for:
- Comparing different aggregation methods
- Cross-validation
- AIC/BIC for model comparison
- Hyperparameter optimization
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelSelectionCriterion(Enum):
    """Criteria for model selection."""
    AIC = "aic"  # Akaike Information Criterion
    BIC = "bic"  # Bayesian Information Criterion
    CV = "cv"    # Cross-validation
    MAE = "mae"   # Mean Absolute Error
    MSE = "mse"   # Mean Squared Error
    R_SQUARED = "r_squared"


@dataclass
class ModelComparison:
    """Results of model comparison."""
    model_name: str
    criterion_value: float
    rank: int
    parameters: Dict[str, Any]
    predictions: Optional[np.ndarray] = None
    errors: Optional[np.ndarray] = None


class ModelSelector:
    """
    Selects optimal models and hyperparameters using various criteria.
    """
    
    @staticmethod
    def calculate_aic(
        n_parameters: int,
        log_likelihood: float
    ) -> float:
        """
        Calculate Akaike Information Criterion.
        
        AIC = 2k - 2ln(L)
        Lower is better.
        
        Args:
            n_parameters: Number of parameters in model
            log_likelihood: Log-likelihood of model
            
        Returns:
            AIC value
        """
        return 2 * n_parameters - 2 * log_likelihood
    
    @staticmethod
    def calculate_bic(
        n_parameters: int,
        log_likelihood: float,
        n_samples: int
    ) -> float:
        """
        Calculate Bayesian Information Criterion.
        
        BIC = k*ln(n) - 2ln(L)
        Lower is better.
        
        Args:
            n_parameters: Number of parameters in model
            log_likelihood: Log-likelihood of model
            n_samples: Number of samples
            
        Returns:
            BIC value
        """
        return n_parameters * np.log(n_samples) - 2 * log_likelihood
    
    @staticmethod
    def cross_validate(
        model_function: Callable,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        metric: str = "mse"
    ) -> Tuple[float, float]:
        """
        Perform k-fold cross-validation.
        
        Args:
            model_function: Function that takes (X_train, y_train) and returns predictions
            X: Feature matrix
            y: Target values
            n_folds: Number of folds
            metric: Metric to use ("mse", "mae", "r_squared")
            
        Returns:
            Tuple of (mean_score, std_score)
        """
        if len(X) < n_folds:
            n_folds = len(X)
        
        fold_size = len(X) // n_folds
        scores = []
        
        for fold in range(n_folds):
            # Split data
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < n_folds - 1 else len(X)
            
            test_indices = np.arange(test_start, test_end)
            train_indices = np.concatenate([
                np.arange(0, test_start),
                np.arange(test_end, len(X))
            ])
            
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            
            # Train and predict
            try:
                y_pred = model_function(X_train, y_train, X_test)
                
                # Calculate metric
                if metric == "mse":
                    score = np.mean((y_test - y_pred) ** 2)
                elif metric == "mae":
                    score = np.mean(np.abs(y_test - y_pred))
                elif metric == "r_squared":
                    ss_res = np.sum((y_test - y_pred) ** 2)
                    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                    score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                else:
                    score = np.mean((y_test - y_pred) ** 2)
                
                scores.append(score)
            except Exception as e:
                logger.warning(f"Error in fold {fold}: {e}")
                continue
        
        if not scores:
            return float('inf'), 0.0
        
        return np.mean(scores), np.std(scores)
    
    @staticmethod
    def compare_models(
        models: Dict[str, Callable],
        X: np.ndarray,
        y: np.ndarray,
        criterion: ModelSelectionCriterion = ModelSelectionCriterion.CV,
        n_folds: int = 5
    ) -> List[ModelComparison]:
        """
        Compare multiple models using specified criterion.
        
        Args:
            models: Dictionary mapping model names to model functions
            X: Feature matrix
            y: Target values
            criterion: Selection criterion
            n_folds: Number of folds for CV
            
        Returns:
            List of ModelComparison objects, sorted by rank
        """
        comparisons = []
        
        for model_name, model_func in models.items():
            try:
                if criterion == ModelSelectionCriterion.CV:
                    mean_score, std_score = ModelSelector.cross_validate(
                        model_func, X, y, n_folds, metric="mse"
                    )
                    criterion_value = mean_score
                
                elif criterion == ModelSelectionCriterion.MAE:
                    # Simple train/test split
                    split_idx = len(X) // 2
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    y_pred = model_func(X_train, y_train, X_test)
                    criterion_value = np.mean(np.abs(y_test - y_pred))
                
                elif criterion == ModelSelectionCriterion.MSE:
                    split_idx = len(X) // 2
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    y_pred = model_func(X_train, y_train, X_test)
                    criterion_value = np.mean((y_test - y_pred) ** 2)
                
                else:
                    criterion_value = float('inf')
                
                comparisons.append(ModelComparison(
                    model_name=model_name,
                    criterion_value=criterion_value,
                    rank=0,  # Will be set later
                    parameters={}
                ))
            
            except Exception as e:
                logger.warning(f"Error evaluating model {model_name}: {e}")
                comparisons.append(ModelComparison(
                    model_name=model_name,
                    criterion_value=float('inf'),
                    rank=len(models),
                    parameters={}
                ))
        
        # Rank models
        comparisons.sort(key=lambda x: x.criterion_value)
        for i, comp in enumerate(comparisons):
            comp.rank = i + 1
        
        return comparisons
    
    @staticmethod
    def grid_search(
        model_function: Callable,
        param_grid: Dict[str, List[Any]],
        X: np.ndarray,
        y: np.ndarray,
        scoring: str = "mse",
        cv: int = 5
    ) -> Tuple[Dict[str, Any], float]:
        """
        Grid search for hyperparameter optimization.
        
        Args:
            model_function: Function that takes parameters and returns model
            param_grid: Dictionary mapping parameter names to lists of values
            X: Feature matrix
            y: Target values
            scoring: Scoring metric
            cv: Number of CV folds
            
        Returns:
            Tuple of (best_parameters, best_score)
        """
        from itertools import product
        
        best_params = None
        best_score = float('inf')
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))
            
            try:
                # Create model with these parameters
                model = model_function(**params)
                
                # Evaluate using CV
                score, _ = ModelSelector.cross_validate(
                    model, X, y, n_folds=cv, metric=scoring
                )
                
                if score < best_score:
                    best_score = score
                    best_params = params
            
            except Exception as e:
                logger.warning(f"Error with parameters {params}: {e}")
                continue
        
        return best_params, best_score


def create_aggregation_model_comparison(
    historical_scores: np.ndarray,
    historical_factors: Dict[str, np.ndarray]
) -> List[ModelComparison]:
    """
    Compare different aggregation methods using historical data.
    
    Args:
        historical_scores: Historical quality scores (ground truth)
        historical_factors: Dictionary mapping factor names to historical values
        
    Returns:
        List of ModelComparison objects
    """
    from guardian.core.aggregation_methods import AdvancedAggregator, AggregationMethod
    
    models = {}
    
    # Define model functions for each aggregation method
    def create_aggregator(method: AggregationMethod):
        def model_func(X_train, y_train, X_test):
            # For simplicity, use equal weights
            weights = None
            predictions = []
            for x in X_test:
                if method == AggregationMethod.GEOMETRIC:
                    pred = AdvancedAggregator.weighted_geometric_mean(x, weights)
                elif method == AggregationMethod.HARMONIC:
                    pred = AdvancedAggregator.weighted_harmonic_mean(x, weights)
                elif method == AggregationMethod.ARITHMETIC:
                    pred = AdvancedAggregator.weighted_arithmetic_mean(x, weights)
                else:
                    pred = AdvancedAggregator.weighted_geometric_mean(x, weights)
                predictions.append(pred)
            return np.array(predictions)
        return model_func
    
    # Add models for each aggregation method
    for method in [AggregationMethod.GEOMETRIC, AggregationMethod.HARMONIC, 
                   AggregationMethod.ARITHMETIC]:
        models[method.value] = create_aggregator(method)
    
    # Prepare data
    factor_names = list(historical_factors.keys())
    X = np.column_stack([historical_factors[name] for name in factor_names])
    y = historical_scores
    
    # Compare models
    return ModelSelector.compare_models(
        models, X, y, ModelSelectionCriterion.CV
    )

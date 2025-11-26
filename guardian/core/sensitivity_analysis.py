"""
Sensitivity Analysis for Quality Metrics

Analyzes how sensitive quality scores are to changes in input parameters.
Identifies which factors have the most impact on final scores.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    """Results of sensitivity analysis."""
    factor_name: str
    base_value: float
    sensitivity_coefficient: float  # ∂score/∂factor
    elasticity: float  # (∂score/score) / (∂factor/factor) = percentage change
    rank: int  # Rank by absolute sensitivity
    contribution_percentage: float  # % of total variance explained


class SensitivityAnalyzer:
    """
    Performs sensitivity analysis on quality metrics.
    
    Methods:
    - Local sensitivity (derivatives)
    - Global sensitivity (Sobol indices)
    - One-at-a-time (OAT) analysis
    - Morris screening
    """
    
    @staticmethod
    def local_sensitivity(
        score_function: Callable,
        base_inputs: Dict[str, float],
        perturbation: float = 0.01
    ) -> List[SensitivityResult]:
        """
        Calculate local sensitivity using finite differences.
        
        Sensitivity = ∂score/∂factor ≈ (score(x+Δ) - score(x)) / Δ
        
        Args:
            score_function: Function that takes dict of inputs and returns score
            base_inputs: Base values for all inputs
            perturbation: Relative perturbation size (default: 1%)
            
        Returns:
            List of SensitivityResult objects
        """
        base_score = score_function(base_inputs)
        results = []
        
        for factor_name, base_value in base_inputs.items():
            # Forward difference
            perturbed_inputs = base_inputs.copy()
            delta = base_value * perturbation if base_value != 0 else perturbation
            perturbed_inputs[factor_name] = base_value + delta
            
            perturbed_score = score_function(perturbed_inputs)
            
            # Calculate sensitivity coefficient
            sensitivity = (perturbed_score - base_score) / delta
            
            # Calculate elasticity: (Δscore/score) / (Δfactor/factor)
            if base_score != 0 and base_value != 0:
                elasticity = (sensitivity * base_value) / base_score
            else:
                elasticity = 0.0
            
            results.append(SensitivityResult(
                factor_name=factor_name,
                base_value=base_value,
                sensitivity_coefficient=sensitivity,
                elasticity=elasticity,
                rank=0,  # Will be set later
                contribution_percentage=0.0  # Will be set later
            ))
        
        # Rank by absolute sensitivity
        results.sort(key=lambda x: abs(x.sensitivity_coefficient), reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1
        
        # Calculate contribution percentages
        total_abs_sensitivity = sum(abs(r.sensitivity_coefficient) for r in results)
        if total_abs_sensitivity > 0:
            for result in results:
                result.contribution_percentage = (
                    abs(result.sensitivity_coefficient) / total_abs_sensitivity * 100
                )
        
        return results
    
    @staticmethod
    def global_sensitivity_sobol(
        score_function: Callable,
        input_ranges: Dict[str, Tuple[float, float]],
        n_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Calculate global sensitivity using Sobol indices.
        
        Sobol indices measure the contribution of each input to the output variance.
        - First-order index: Main effect
        - Total-order index: Main effect + interactions
        
        Args:
            score_function: Function that takes dict of inputs and returns score
            input_ranges: Dictionary mapping factor names to (min, max) ranges
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with first-order and total-order Sobol indices
        """
        import numpy as np
        
        # Generate random samples
        factors = list(input_ranges.keys())
        n_factors = len(factors)
        
        # Create two independent sample matrices
        A = np.random.random((n_samples, n_factors))
        B = np.random.random((n_samples, n_factors))
        
        # Scale to input ranges
        for i, factor in enumerate(factors):
            min_val, max_val = input_ranges[factor]
            A[:, i] = A[:, i] * (max_val - min_val) + min_val
            B[:, i] = B[:, i] * (max_val - min_val) + min_val
        
        # Evaluate function
        f_A = np.array([score_function({factors[j]: A[i, j] for j in range(n_factors)})
                        for i in range(n_samples)])
        f_B = np.array([score_function({factors[j]: B[i, j] for j in range(n_factors)})
                        for i in range(n_samples)])
        
        f0_squared = np.mean(f_A) ** 2
        total_variance = np.var(f_A)
        
        if total_variance == 0:
            return {factor: 0.0 for factor in factors}
        
        # Calculate first-order Sobol indices
        sobol_indices = {}
        for i, factor in enumerate(factors):
            # Create C_i: all factors from A except factor i from B
            C_i = A.copy()
            C_i[:, i] = B[:, i]
            
            f_C_i = np.array([score_function({factors[j]: C_i[k, j] for j in range(n_factors)})
                             for k in range(n_samples)])
            
            # First-order index: S_i = Var(E[Y|X_i]) / Var(Y)
            # Approximated as: E[Y * f(C_i)] - f0^2 / Var(Y)
            first_order = (np.mean(f_A * f_C_i) - f0_squared) / total_variance
            sobol_indices[f"{factor}_first_order"] = max(0.0, first_order)
            
            # Total-order index: S_Ti = 1 - Var(E[Y|X_~i]) / Var(Y)
            # Approximated as: 1 - (E[Y * f(C_i)] - f0^2) / Var(Y)
            total_order = 1.0 - (np.mean(f_B * f_C_i) - f0_squared) / total_variance
            sobol_indices[f"{factor}_total_order"] = max(0.0, min(1.0, total_order))
        
        return sobol_indices
    
    @staticmethod
    def morris_screening(
        score_function: Callable,
        input_ranges: Dict[str, Tuple[float, float]],
        n_trajectories: int = 10,
        levels: int = 4
    ) -> Dict[str, float]:
        """
        Morris screening for factor prioritization.
        
        Fast screening method to identify important factors.
        Returns elementary effects (EE) for each factor.
        
        Args:
            score_function: Function that takes dict of inputs and returns score
            input_ranges: Dictionary mapping factor names to (min, max) ranges
            n_trajectories: Number of trajectories to sample
            levels: Number of levels for each factor
            
        Returns:
            Dictionary with mean and std of elementary effects for each factor
        """
        factors = list(input_ranges.keys())
        n_factors = len(factors)
        
        elementary_effects = {factor: [] for factor in factors}
        
        for _ in range(n_trajectories):
            # Generate random starting point
            start = {}
            for factor in factors:
                min_val, max_val = input_ranges[factor]
                start[factor] = np.random.uniform(min_val, max_val)
            
            # Random permutation of factors
            factor_order = np.random.permutation(factors)
            
            # Calculate elementary effects
            current = start.copy()
            current_score = score_function(current)
            
            for factor in factor_order:
                # Perturb this factor
                min_val, max_val = input_ranges[factor]
                delta = (max_val - min_val) / (levels - 1)
                
                # Random direction
                direction = np.random.choice([-1, 1])
                new_value = current[factor] + direction * delta
                new_value = np.clip(new_value, min_val, max_val)
                
                perturbed = current.copy()
                perturbed[factor] = new_value
                perturbed_score = score_function(perturbed)
                
                # Elementary effect
                if abs(new_value - current[factor]) > 1e-10:
                    ee = (perturbed_score - current_score) / (new_value - current[factor])
                    elementary_effects[factor].append(ee)
                
                current = perturbed
                current_score = perturbed_score
        
        # Calculate statistics
        results = {}
        for factor in factors:
            if elementary_effects[factor]:
                results[f"{factor}_mean"] = np.mean(elementary_effects[factor])
                results[f"{factor}_std"] = np.std(elementary_effects[factor])
                results[f"{factor}_abs_mean"] = np.mean(np.abs(elementary_effects[factor]))
            else:
                results[f"{factor}_mean"] = 0.0
                results[f"{factor}_std"] = 0.0
                results[f"{factor}_abs_mean"] = 0.0
        
        return results


def analyze_betes_sensitivity(
    calculator,
    base_inputs: Dict[str, float]
) -> List[SensitivityResult]:
    """
    Analyze sensitivity of bE-TES score to input factors.
    
    Args:
        calculator: BETESCalculator instance
        base_inputs: Dictionary with base input values
        
    Returns:
        List of SensitivityResult objects
    """
    def score_function(inputs):
        return calculator.calculate(
            raw_mutation_score=inputs.get('mutation_score', 0.0),
            raw_emt_gain=inputs.get('emt_gain', 0.0),
            raw_assertion_iq=inputs.get('assertion_iq', 1.0),
            raw_behaviour_coverage=inputs.get('behaviour_coverage', 0.0),
            raw_median_test_time_ms=inputs.get('test_time_ms', 1000.0),
            raw_flakiness_rate=inputs.get('flakiness_rate', 0.0)
        ).betes_score
    
    return SensitivityAnalyzer.local_sensitivity(score_function, base_inputs)

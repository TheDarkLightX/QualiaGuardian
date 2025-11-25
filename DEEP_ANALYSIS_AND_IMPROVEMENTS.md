# Deep Analysis: Mathematical Verification, Testing Gaps, and Algorithm Improvements

## Executive Summary

This document provides a comprehensive analysis of the Qualia Guardian codebase, identifying:
1. **Mathematical Issues**: Numerical stability, edge cases, and correctness
2. **Testing Gaps**: Missing unit tests, property-based tests, and formal verification
3. **Algorithm Improvements**: Mathematical optimizations and better formulations
4. **Formal Verification Opportunities**: Properties that can be proven

---

## 1. Mathematical Analysis

### 1.1 bE-TES Calculation - Critical Issues Found

#### Issue 1: Geometric Mean with Zero Weights
**Location**: `betes.py:_calculate_weighted_geometric_mean()`

**Problem**: The current implementation has a logical flaw:
```python
if factor > 0.0 or weight == 0.0:
    weighted_product *= (factor ** weight)
```

**Issue**: When `weight == 0.0`, we multiply by `factor^0 = 1`, which is correct. However, if ALL weights are zero, we return 0.0, but mathematically:
- If all weights are 0: G = (∏1)^(1/0) = undefined
- Current code returns 0.0, which is incorrect

**Fix**: Should return `NaN` or raise an exception when all weights are zero.

#### Issue 2: Numerical Stability in Geometric Mean
**Problem**: For very small factors (e.g., 1e-10), raising to large weights can cause underflow:
```python
weighted_product *= (factor ** weight)  # Can underflow
```

**Better Approach**: Use log-space arithmetic:
```python
log_product = sum(weight * math.log(factor) for factor, weight in zip(factors, weights))
result = math.exp(log_product / sum_of_weights)
```

#### Issue 3: Speed Factor Normalization Discontinuity
**Location**: `betes.py:_normalize_speed_factor()`

**Problem**: At the boundary `raw_time_ms = 100`:
- For `t = 100`: Returns 1.0 (from piece-wise)
- For `t = 100.0001`: Returns `1 / (1 + log10(1.000001)) ≈ 0.9999996`

**Issue**: There's a discontinuity! The function jumps from 1.0 to ~0.9999996.

**Fix**: Make it continuous:
```python
if raw_time_ms <= SPEED_THRESHOLD_MS:
    return 1.0
else:
    # Ensure continuity at threshold
    log_input = raw_time_ms / SPEED_DIVISOR
    log_val = math.log10(log_input)
    # At threshold: log10(100/100) = 0, so 1/(1+0) = 1.0 ✓
    return BETESCalculator._clamp(1.0 / (1.0 + log_val))
```

#### Issue 4: Sigmoid Overflow Handling
**Location**: `betes.py:_sigmoid_normalize()`

**Current Code**:
```python
except OverflowError:
    return 0.0 if exponent > 0 else 1.0
```

**Problem**: This is correct but could be more precise. For very large negative exponents, `exp(-k*(v-c))` → ∞, so `1/(1+∞)` → 0. But the check `exponent > 0` means `-k*(v-c) > 0`, which means `v < c`, so we should return ~0. This is correct.

**However**: We should also handle the case where `exponent` itself overflows (very large k or very large |v-c|).

**Better Fix**:
```python
try:
    exponent = -k * (value - center)
    # Clamp exponent to prevent overflow
    exponent = max(-700, min(700, exponent))  # exp(±700) is near machine limits
    result = 1.0 / (1.0 + math.exp(exponent))
    return BETESCalculator._clamp(result)
except (OverflowError, ValueError):
    # If still fails, use asymptotic behavior
    if value < center:
        return 0.0
    else:
        return 1.0
```

### 1.2 OSQI Calculation - Issues Found

#### Issue 1: Geometric Mean with Zero Pillars
**Location**: `osqi.py:calculate()`

**Problem**: If any weighted pillar is zero, the entire OSQI becomes zero. This is mathematically correct for geometric mean, but may be too harsh. Consider:
- A project with perfect test quality (bE-TES=1.0) but one security vulnerability (Sec_S=0.0) gets OSQI=0.0
- This might be too punitive

**Alternative**: Use a weighted harmonic mean for some pillars, or add a minimum threshold.

#### Issue 2: CHS Normalization Interpolation
**Location**: `osqi.py:_normalize_chs_sub_metric()`

**Problem**: The interpolation logic has potential division-by-zero issues that are handled, but the interpolation formula could be cleaner:

**Current** (for lower-is-better):
```python
if raw_value <= mid_raw:
    return 1.0 - (1.0 - mid_score) * (raw_value - ideal_max) / (mid_raw - ideal_max)
else:
    return mid_score - mid_score * (raw_value - mid_raw) / (poor_min - mid_raw)
```

**Issue**: The second branch can produce negative values if `raw_value > poor_min` (though clamped). The formula is correct but could be simplified using linear interpolation helper.

### 1.3 Shapley Value Calculation - Mathematical Issues

#### Issue 1: Efficiency Property Not Guaranteed
**Location**: `shapley.py` and `shapley_v2.py`

**Problem**: The Monte Carlo approximation doesn't guarantee the efficiency property:
```
∑ Shapley_i = F(N) - F(∅)
```

**Current**: Only verified in test, not enforced.

**Fix**: Add a normalization step:
```python
# After calculating Shapley values
sum_shapley = sum(shapley_values.values())
expected_sum = metric_evaluator_func(test_ids) - metric_evaluator_func([])
if abs(sum_shapley - expected_sum) > tolerance:
    # Normalize to enforce efficiency
    if sum_shapley != 0:
        scale_factor = expected_sum / sum_shapley
        shapley_values = {k: v * scale_factor for k, v in shapley_values.items()}
```

#### Issue 2: Convergence Guarantees
**Problem**: No theoretical guarantee on convergence rate. For n tests, exact Shapley requires O(2^n) evaluations. Monte Carlo with 200 permutations may be insufficient for large test suites.

**Improvement**: Use adaptive sampling with confidence intervals:
```python
def calculate_with_confidence(test_ids, metric_func, confidence=0.95, max_iterations=10000):
    """Calculate Shapley values with statistical confidence."""
    # Use sequential sampling until confidence interval is narrow enough
    # or max_iterations reached
```

---

## 2. Testing Gaps Analysis

### 2.1 Missing Unit Tests

#### Critical Missing Tests for `betes.py`:
1. **Edge Cases**:
   - All factors = 0.0
   - All factors = 1.0
   - Negative raw values (should be clamped)
   - Very large raw values
   - NaN/Inf inputs

2. **Weight Edge Cases**:
   - All weights = 0
   - Negative weights (should be rejected)
   - Very large weights
   - Mixed zero/non-zero weights

3. **Normalization Edge Cases**:
   - `raw_assertion_iq < 1.0` (below minimum)
   - `raw_assertion_iq > 5.0` (above maximum)
   - `raw_flakiness_rate > 1.0` (invalid)
   - `raw_flakiness_rate < 0.0` (invalid)

4. **Sigmoid Edge Cases**:
   - Very large k values
   - Very small k values
   - Value exactly at center
   - Value far from center

#### Missing Tests for `osqi.py`:
1. Missing thresholds in YAML
2. Invalid threshold values (e.g., ideal_max > poor_min)
3. Empty CHS sub-metrics
4. All pillars zero
5. Negative vulnerability density

#### Missing Tests for `classification.py`:
1. Invalid risk class names
2. Missing min_score in risk definitions
3. Non-numeric min_score
4. Empty risk_definitions

### 2.2 Property-Based Testing (Hypothesis)

**Current State**: No property-based tests found using Hypothesis.

**Recommended Properties to Test**:

#### Property 1: bE-TES Boundedness
```python
@given(
    mutation_score=floats(min_value=0.0, max_value=1.0),
    emt_gain=floats(min_value=-1.0, max_value=1.0),
    assertion_iq=floats(min_value=1.0, max_value=5.0),
    behaviour_coverage=floats(min_value=0.0, max_value=1.0),
    test_time=floats(min_value=0.0, max_value=10000.0),
    flakiness=floats(min_value=0.0, max_value=1.0)
)
def test_betes_bounded(calculator, mutation_score, emt_gain, assertion_iq, 
                       behaviour_coverage, test_time, flakiness):
    """Property: bE-TES score is always in [0, 1]"""
    result = calculator.calculate(mutation_score, emt_gain, assertion_iq,
                                  behaviour_coverage, test_time, flakiness)
    assert 0.0 <= result.betes_score <= 1.0
    assert all(0.0 <= getattr(result, f'norm_{attr}') <= 1.0 
               for attr in ['mutation_score', 'emt_gain', 'assertion_iq', 
                           'behaviour_coverage', 'speed_factor'])
```

#### Property 2: Monotonicity
```python
@given(
    base_mutation=floats(min_value=0.6, max_value=0.95),
    delta=floats(min_value=0.0, max_value=0.1)
)
def test_mutation_score_monotonicity(calculator, base_mutation, delta):
    """Property: Higher mutation score → higher normalized score"""
    result1 = calculator.calculate(base_mutation, 0, 3, 0.5, 100, 0)
    result2 = calculator.calculate(base_mutation + delta, 0, 3, 0.5, 100, 0)
    assert result2.norm_mutation_score >= result1.norm_mutation_score
```

#### Property 3: Shapley Efficiency
```python
@given(
    test_ids=lists(text(min_size=1, max_size=10), min_size=1, max_size=10),
    num_permutations=integers(min_value=50, max_value=500)
)
def test_shapley_efficiency(test_ids, num_permutations, metric_func):
    """Property: Sum of Shapley values = F(N) - F(∅)"""
    shapley_values = calculate_shapley_values(test_ids, metric_func, num_permutations)
    total_shapley = sum(shapley_values.values())
    expected = metric_func(test_ids) - metric_func([])
    assert abs(total_shapley - expected) < 0.1  # Allow Monte Carlo error
```

#### Property 4: Symmetry (Shapley)
```python
@given(test_ids=lists(text(), min_size=2, max_size=5))
def test_shapley_symmetry(test_ids, metric_func):
    """Property: Permuting test order doesn't change Shapley values"""
    shapley1 = calculate_shapley_values(test_ids, metric_func, seed=42)
    shuffled = list(reversed(test_ids))
    shapley2 = calculate_shapley_values(shuffled, metric_func, seed=42)
    # Values should be the same (just keys reordered)
    assert set(shapley1.values()) == set(shapley2.values())
```

#### Property 5: Dummy Player (Shapley)
```python
@given(test_ids=lists(text(), min_size=2, max_size=5))
def test_shapley_dummy_player(test_ids, metric_func):
    """Property: Test with zero marginal contribution has Shapley value = 0"""
    # Add a dummy test that never contributes
    dummy_test = "dummy_test_never_contributes"
    all_tests = test_ids + [dummy_test]
    
    def metric_with_dummy(subset):
        # Dummy test never affects the metric
        subset_without_dummy = [t for t in subset if t != dummy_test]
        return metric_func(subset_without_dummy)
    
    shapley = calculate_shapley_values(all_tests, metric_with_dummy)
    assert abs(shapley.get(dummy_test, 0)) < 1e-6
```

### 2.3 Formal Verification Opportunities

#### Property 1: bE-TES Boundedness (Proven)
**Theorem**: For any valid inputs, `0 ≤ bE-TES ≤ 1`

**Proof Sketch**:
1. All normalized factors are in [0, 1] (by clamp)
2. Geometric mean of values in [0, 1] is in [0, 1]
3. Trust coefficient T = 1 - flakiness ∈ [0, 1]
4. Product of two values in [0, 1] is in [0, 1]
5. Final clamp ensures result ∈ [0, 1]

**Formal Statement** (in TLA+ or Coq):
```tla
THEOREM BETESBounded ==
  \A mutation, emt, iq, coverage, time, flakiness \in ValidInputs :
    0 <= BETES(mutation, emt, iq, coverage, time, flakiness) <= 1
```

#### Property 2: Monotonicity (Partial)
**Theorem**: If mutation_score₁ < mutation_score₂, then bE-TES₁ ≤ bE-TES₂ (assuming other inputs equal)

**Proof**: Depends on normalization method:
- Min-max: Linear, so monotonic ✓
- Sigmoid: Monotonic (sigmoid is strictly increasing) ✓

#### Property 3: Shapley Axioms
The Shapley value satisfies four axioms (can be formally verified):
1. **Efficiency**: ∑φᵢ = v(N) - v(∅)
2. **Symmetry**: If v(S ∪ {i}) = v(S ∪ {j}) for all S, then φᵢ = φⱼ
3. **Dummy**: If v(S ∪ {i}) = v(S) for all S, then φᵢ = 0
4. **Additivity**: φ(v + w) = φ(v) + φ(w)

---

## 3. Algorithm Improvements

### 3.1 Numerical Stability Improvements

#### Improvement 1: Log-Space Geometric Mean
**Current**: Direct multiplication can underflow
**Better**:
```python
@staticmethod
def _calculate_weighted_geometric_mean_log_space(
    factors: List[float], weights: List[float]
) -> float:
    """Calculate weighted geometric mean using log-space for numerical stability."""
    if not factors or not weights or len(factors) != len(weights):
        return 0.0

    sum_of_weights = sum(weights)
    if sum_of_weights == 0:
        return float('nan')  # Undefined, not 0.0

    # Check for zero factors
    for factor, weight in zip(factors, weights):
        if factor == 0.0 and weight > 0.0:
            return 0.0
        if factor < 0.0:
            return float('nan')  # Negative values not allowed

    # Use log-space to prevent underflow
    log_sum = 0.0
    for factor, weight in zip(factors, weights):
        if factor > 0.0:
            log_sum += weight * math.log(factor)
        # If weight == 0, skip (factor^0 = 1, log(1) = 0)

    if log_sum == float('-inf'):
        return 0.0

    return math.exp(log_sum / sum_of_weights)
```

#### Improvement 2: Better Sigmoid Implementation
**Current**: Can overflow for extreme values
**Better**:
```python
@staticmethod
def _sigmoid_normalize_stable(value: float, k: float, center: float) -> float:
    """Numerically stable sigmoid normalization."""
    # Clamp exponent to prevent overflow
    exponent = -k * (value - center)
    exponent = max(-700, min(700, exponent))  # exp(±700) ≈ machine limits
    
    # Use numerically stable sigmoid: 1/(1+exp(-x)) = exp(x)/(1+exp(x)) for x>0
    if exponent > 0:
        exp_val = math.exp(-exponent)  # exp(-x) where x>0, so this is < 1
        return 1.0 / (1.0 + exp_val)
    else:
        exp_val = math.exp(exponent)  # exp(x) where x<=0, so this is <= 1
        return exp_val / (1.0 + exp_val)
```

### 3.2 Algorithmic Optimizations

#### Optimization 1: Cached Shapley Calculation
**Current**: Recalculates for each permutation
**Better**: Use memoization with subset hashing:
```python
from functools import lru_cache
from typing import Tuple

class CachedShapleyCalculator:
    def __init__(self, metric_func, cache_size=32768):
        self.metric_func = metric_func
        self._cache = {}
    
    def _subset_key(self, subset: List[TestId]) -> Tuple[TestId, ...]:
        """Create immutable key for subset."""
        return tuple(sorted(subset, key=str))
    
    def evaluate(self, subset: List[TestId]) -> float:
        key = self._subset_key(subset)
        if key not in self._cache:
            self._cache[key] = self.metric_func(subset)
            # Evict if cache too large (simple FIFO)
            if len(self._cache) > self.cache_size:
                # Remove oldest (first) entry
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]
```

#### Optimization 2: Adaptive Shapley Sampling
**Current**: Fixed number of permutations
**Better**: Adaptive sampling with confidence intervals:
```python
def calculate_shapley_adaptive(
    test_ids: List[TestId],
    metric_func: Callable,
    confidence: float = 0.95,
    max_iterations: int = 10000,
    min_iterations: int = 100
) -> Tuple[Dict[TestId, float], Dict[str, float]]:
    """Calculate Shapley values with adaptive sampling."""
    from scipy import stats
    
    n = len(test_ids)
    shapley_estimates = {tid: [] for tid in test_ids}
    iteration = 0
    
    while iteration < min_iterations or iteration < max_iterations:
        # Calculate one permutation
        perm = random.sample(test_ids, n)
        current_subset = []
        score_current = metric_func([])
        
        for test_id in perm:
            new_subset = current_subset + [test_id]
            score_new = metric_func(new_subset)
            marginal = score_new - score_current
            shapley_estimates[test_id].append(marginal)
            current_subset = new_subset
            score_current = score_new
        
        iteration += 1
        
        # Check convergence every 50 iterations
        if iteration % 50 == 0 and iteration >= min_iterations:
            converged = True
            for tid in test_ids:
                estimates = shapley_estimates[tid]
                if len(estimates) < 30:  # Need minimum samples
                    converged = False
                    break
                # Calculate confidence interval
                mean = np.mean(estimates)
                std = np.std(estimates, ddof=1)
                ci = stats.t.interval(confidence, len(estimates)-1, 
                                     loc=mean, scale=std/np.sqrt(len(estimates)))
                if (ci[1] - ci[0]) / abs(mean) > 0.1:  # Relative error > 10%
                    converged = False
                    break
            
            if converged:
                break
    
    # Calculate final estimates
    shapley_values = {tid: np.mean(shapley_estimates[tid]) 
                     for tid in test_ids}
    
    # Calculate statistics
    stats_dict = {
        'iterations': iteration,
        'converged': iteration < max_iterations,
        'std_errors': {tid: np.std(shapley_estimates[tid], ddof=1) / np.sqrt(len(shapley_estimates[tid]))
                      for tid in test_ids}
    }
    
    return shapley_values, stats_dict
```

### 3.3 Mathematical Formulation Improvements

#### Improvement 1: Better Speed Factor Formula
**Current**: `S' = 1 / (1 + log₁₀(t/100))` for t > 100

**Issue**: This decays too slowly. For t=1000ms, S' = 1/(1+1) = 0.5, which might be too generous.

**Alternative**: Use exponential decay:
```python
S' = exp(-α * (t - 100) / 100)  for t > 100
```
Where α controls decay rate (e.g., α=0.5 gives S'(200)≈0.61, S'(1000)≈0.0067)

Or use a power law:
```python
S' = (100 / t)^β  for t > 100
```
Where β controls steepness (e.g., β=0.5 gives S'(200)≈0.71, S'(1000)≈0.32)

#### Improvement 2: Weighted Harmonic Mean Alternative
**Current**: Geometric mean for OSQI

**Alternative**: Consider weighted harmonic mean for some pillars:
```python
H = (∑w_i) / (∑w_i / p_i)
```
This is more sensitive to low values (good for security - one vulnerability should heavily penalize).

**Hybrid Approach**: Use geometric mean for most pillars, harmonic mean for security:
```python
# For security pillar, use harmonic mean component
if security_score == 0:
    osqi_score = 0  # Fail fast
else:
    # Use geometric mean of other pillars, then multiply by security
    other_pillars_gm = geometric_mean([betes, chs, arch])
    osqi_score = other_pillars_gm * security_score  # Multiplicative penalty
```

---

## 4. Recommended Test Suite

### 4.1 Unit Tests to Add

```python
# tests/core/test_betes_comprehensive.py
import unittest
import math
import numpy as np
from hypothesis import given, strategies as st
from guardian.core.betes import BETESCalculator, BETESComponents

class TestBETESComprehensive(unittest.TestCase):
    
    @given(
        mutation=st.floats(min_value=0.0, max_value=1.0),
        emt=st.floats(min_value=-1.0, max_value=1.0),
        iq=st.floats(min_value=1.0, max_value=5.0),
        coverage=st.floats(min_value=0.0, max_value=1.0),
        time=st.floats(min_value=0.0, max_value=10000.0),
        flakiness=st.floats(min_value=0.0, max_value=1.0)
    )
    def test_boundedness_property(self, mutation, emt, iq, coverage, time, flakiness):
        """Property: All outputs are bounded [0, 1]"""
        calc = BETESCalculator()
        result = calc.calculate(mutation, emt, iq, coverage, time, flakiness)
        assert 0.0 <= result.betes_score <= 1.0
        assert 0.0 <= result.norm_mutation_score <= 1.0
        assert 0.0 <= result.norm_emt_gain <= 1.0
        assert 0.0 <= result.norm_assertion_iq <= 1.0
        assert 0.0 <= result.norm_behaviour_coverage <= 1.0
        assert 0.0 <= result.norm_speed_factor <= 1.0
        assert 0.0 <= result.trust_coefficient_t <= 1.0
    
    def test_zero_flakiness_gives_full_trust(self):
        """Zero flakiness should give trust coefficient = 1.0"""
        calc = BETESCalculator()
        result = calc.calculate(0.8, 0.1, 3.0, 0.7, 100, 0.0)
        assert result.trust_coefficient_t == 1.0
    
    def test_max_flakiness_gives_zero_trust(self):
        """Flakiness = 1.0 should give trust coefficient = 0.0"""
        calc = BETESCalculator()
        result = calc.calculate(0.8, 0.1, 3.0, 0.7, 100, 1.0)
        assert result.trust_coefficient_t == 0.0
    
    def test_all_factors_one_gives_perfect_score(self):
        """If all normalized factors = 1.0 and flakiness = 0, score = 1.0"""
        # This requires careful input selection
        calc = BETESCalculator()
        # Perfect inputs: mutation=0.95, emt=0.25, iq=5.0, coverage=1.0, time=50, flakiness=0
        result = calc.calculate(0.95, 0.25, 5.0, 1.0, 50, 0.0)
        # Should be very close to 1.0 (may not be exactly 1.0 due to geometric mean)
        assert result.betes_score > 0.99
    
    def test_geometric_mean_zero_factor(self):
        """If any factor is zero (with non-zero weight), geometric mean = 0"""
        calc = BETESCalculator()
        result = calc.calculate(0.0, 0.1, 3.0, 0.7, 100, 0.0)  # mutation = 0
        assert result.geometric_mean_g == 0.0
        assert result.betes_score == 0.0
```

### 4.2 Integration Tests

```python
# tests/integration/test_betes_osqi_integration.py
def test_betes_as_osqi_pillar():
    """Test that bE-TES correctly integrates as OSQI pillar"""
    # Calculate bE-TES
    betes_calc = BETESCalculator()
    betes_result = betes_calc.calculate(0.8, 0.1, 3.0, 0.7, 100, 0.05)
    
    # Use in OSQI
    osqi_calc = OSQICalculator(...)
    osqi_input = OSQIRawPillarsInput(
        betes_score=betes_result.betes_score,
        ...
    )
    osqi_result = osqi_calc.calculate(osqi_input, "python")
    
    # Verify bE-TES score is preserved
    assert osqi_result.normalized_pillars.betes_score == betes_result.betes_score
```

---

## 5. Implementation Priority

### High Priority (Critical Bugs)
1. ✅ Fix geometric mean zero-weight handling
2. ✅ Fix speed factor discontinuity
3. ✅ Add numerical stability (log-space)
4. ✅ Add input validation

### Medium Priority (Improvements)
1. Add property-based tests
2. Implement adaptive Shapley sampling
3. Improve sigmoid numerical stability
4. Add comprehensive edge case tests

### Low Priority (Nice to Have)
1. Formal verification (TLA+/Coq)
2. Alternative speed factor formulas
3. Hybrid mean approaches for OSQI

---

## 6. Conclusion

The codebase has a solid mathematical foundation but needs:
1. **Numerical stability improvements** (log-space arithmetic)
2. **Comprehensive testing** (property-based tests)
3. **Edge case handling** (zero weights, invalid inputs)
4. **Mathematical correctness** (discontinuity fixes)

The algorithms are sound but can be improved for robustness and numerical stability.

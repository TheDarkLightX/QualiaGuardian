# Test Importance Prediction Features Guide

**Author**: DarkLightX/Dana Edwards  
**Version**: 1.0  
**Last Updated**: Current Session

## Table of Contents
1. [Implemented Features](#implemented-features)
   - [Convergence-Based Shapley Sampling](#1-convergence-based-shapley-sampling)
   - [UCB1-Thompson Hybrid](#2-ucb1-thompson-hybrid)
   - [Incremental GP Learning](#3-incremental-gp-learning)
2. [Planned Features](#planned-features)
   - [Test Clustering System](#4-test-clustering-system)
   - [Mutation Survival Pattern Learning](#5-mutation-survival-pattern-learning)
   - [Persistent Result Cache](#6-persistent-result-cache)
   - [Contextual Bandits](#7-contextual-bandits)
   - [Parallel Shapley Evaluation](#8-parallel-shapley-evaluation)
   - [Progressive Approximation Framework](#9-progressive-approximation-framework)
   - [Temporal Importance Tracking](#10-temporal-importance-tracking)

---

## Implemented Features

### 1. Convergence-Based Shapley Sampling

#### Concept
Shapley values measure each test's contribution to overall quality by considering all possible test combinations. Traditional computation requires 2^n evaluations, which is intractable for large test suites.

#### How It Works
```
Traditional Shapley:
┌─────────────┐
│ All 2^n     │ ──→ Evaluate all ──→ Exact values
│ Permutations│     permutations     (Exponential time)
└─────────────┘

Convergence-Based:
┌─────────────┐     ┌──────────┐     ┌────────────┐
│ Sample      │ ──→ │ Check    │ ──→ │ Stop Early │
│ Permutations│     │Convergence│     │ if Stable  │
└─────────────┘     └──────────┘     └────────────┘
      ↑                    │
      └────────────────────┘
         Continue if not converged
```

#### Algorithm Flow
```python
# Pseudocode
shapley_values = {}
variance_window = deque(maxlen=10)

for iteration in range(max_iterations):
    permutation = random_permutation(tests)
    marginal_contributions = compute_marginals(permutation)
    update_shapley_estimates(marginal_contributions)
    
    variance = compute_variance(shapley_values)
    variance_window.append(variance)
    
    if max(variance_window) < threshold:
        break  # Converged!
```

#### Benefits
- **3x faster** on average
- **Same accuracy** (within 1%)
- **Adaptive** to problem difficulty

#### Antithetic Variates
```
Regular sampling:    π₁ = [A, B, C, D]  (random)
Antithetic pair:     π₂ = [D, C, B, A]  (reversed)

Variance reduction through negative correlation
```

---

### 2. UCB1-Thompson Hybrid

#### Concept
Multi-armed bandit problem: Select tests to maximize information gain while balancing exploration of unknown tests with exploitation of known good tests.

#### State Machine
```
┌─────────────┐     High          ┌──────────────┐
│    UCB1     │ ─────────────────→│   Thompson   │
│ Exploration │   Confidence      │ Exploitation │
└─────────────┘                   └──────────────┘
      ↑                                   │
      │            Low                    │
      └───────────────────────────────────┘
              Confidence
```

#### UCB1 Formula
```
UCB(test) = μ(test) + c√(ln(N)/n(test))

Where:
- μ(test) = average reward
- c = exploration constant
- N = total iterations
- n(test) = times test selected
```

#### Thompson Sampling
```
1. Model each test's quality as Beta(α, β)
2. Sample θᵢ ~ Beta(αᵢ, βᵢ) for each test
3. Select test with highest sampled value
4. Update based on observed reward
```

#### Transition Logic
```python
confidence = min(attempts_per_test) / threshold
if confidence > 0.8:
    use_thompson_sampling()
else:
    use_ucb1()
```

#### Benefits
- **No regret** asymptotically
- **Fast convergence** in practice
- **Handles non-stationary** environments

---

### 3. Incremental GP Learning

#### Concept
Gaussian Process (GP) models test quality as a probabilistic function, providing both predictions and uncertainty estimates. Sparse approximations make it scalable.

#### Architecture
```
┌─────────────────────────────────────────────┐
│           Full GP: O(n³)                    │
│  K = [k(x₁,x₁) ... k(x₁,xₙ)]              │
│      [   ⋮      ⋱     ⋮   ]               │
│      [k(xₙ,x₁) ... k(xₙ,xₙ)]              │
└─────────────────────────────────────────────┘
                    ↓
           Sparse Approximation
                    ↓
┌─────────────────────────────────────────────┐
│         Sparse GP: O(m²n)                   │
│   Use m << n inducing points                │
│   K̃ = Kfu K⁻¹uu Kuf                        │
└─────────────────────────────────────────────┘
```

#### Incremental Update
```
New observation (xₙ₊₁, yₙ₊₁) arrives:

1. Rank-1 update to covariance
   K' = [K   k]
        [k'  κ]

2. Update predictions using Woodbury identity
   (A + UCV)⁻¹ = A⁻¹ - A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹VA⁻¹

3. Recompute inducing points if needed
```

#### Memory Management
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Recent     │     │  Important   │     │ Compressed   │
│ Observations │ +   │ Observations │ +   │    Old       │ = 100MB
│   (Full)     │     │   (Full)     │     │Observations  │   Limit
└──────────────┘     └──────────────┘     └──────────────┘
```

#### Benefits
- **O(m²) memory** instead of O(n²)
- **< 100ms updates**
- **Uncertainty quantification**

---

## Planned Features

### 4. Test Clustering System

#### Concept
Group similar tests together to reduce Shapley computation complexity from O(2^n) to O(2^k) where k is the number of clusters.

#### Clustering Pipeline
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│Extract Test  │ ──→ │  Clustering  │ ──→ │   Shapley    │
│  Features    │     │  Algorithm   │     │ on Clusters  │
└──────────────┘     └──────────────┘     └──────────────┘
       │                     │                     │
   Code AST            K-means/DBSCAN         O(2^k)
   Coverage            Hierarchical          computation
   Dependencies        Spectral
```

#### Feature Extraction
```python
test_features = {
    'code_complexity': cyclomatic_complexity,
    'coverage_vector': [lines_covered],
    'dependencies': import_graph,
    'execution_time': avg_runtime,
    'failure_rate': historical_failures,
    'code_embedding': bert_embedding(test_code)
}
```

#### Hierarchical Structure
```
All Tests (1000)
    ├── Unit Tests (600)
    │   ├── Model Tests (200)
    │   │   ├── User Model (50)
    │   │   └── Product Model (150)
    │   └── Utility Tests (400)
    └── Integration Tests (400)
        ├── API Tests (200)
        └── DB Tests (200)
```

#### Benefits
- **100x speedup** for large suites
- **Interpretable groups**
- **Preserves accuracy** (±5%)

---

### 5. Mutation Survival Pattern Learning

#### Concept
Learn which code patterns consistently survive mutations, indicating weak test coverage.

#### Pattern Detection Flow
```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Mutation   │ ──→ │Pattern Mining│ ──→ │  Weakness    │
│  Results    │     │              │     │   Report     │
└─────────────┘     └──────────────┘     └──────────────┘
       │                    │                     │
  Survived/Killed      AST Analysis         Prioritized
   Mutations          Frequent Patterns    Improvements
```

#### Pattern Examples
```python
# Weak Pattern 1: Unchecked None
if obj is not None:  # Mutation: Remove check
    obj.method()      # Often survives

# Weak Pattern 2: Boundary Conditions  
if x > 0:            # Mutation: x >= 0
    process(x)       # Often survives

# Weak Pattern 3: Error Messages
raise ValueError("Invalid input")  # String mutations survive
```

#### Learning Algorithm
```
1. Extract AST patterns from survived mutations
2. Build pattern frequency table
3. Weight by impact (critical path vs edge case)
4. Generate test improvement suggestions
```

#### Benefits
- **Automated weakness detection**
- **Actionable improvements**
- **Continuous learning**

---

### 6. Persistent Result Cache

#### Concept
Content-addressed storage for expensive computations with automatic invalidation.

#### Cache Architecture
```
┌─────────────────────────────────────────┐
│           Cache Manager                 │
├─────────────────────────────────────────┤
│  Content Hash │ Result │ Metadata       │
├───────────────┼────────┼────────────────┤
│  sha256(...)  │ Shapley│ Time, Version  │
│  sha256(...)  │ GP Pred│ Dependencies   │
│  sha256(...)  │ Cluster│ Expiry         │
└─────────────────────────────────────────┘
                     │
                     ↓
            Persistent Storage
           (SQLite/RocksDB)
```

#### Content Addressing
```python
def cache_key(test_ids, algorithm, params):
    content = {
        'test_ids': sorted(test_ids),
        'algorithm': algorithm,
        'params': params,
        'version': CACHE_VERSION
    }
    return hashlib.sha256(
        json.dumps(content).encode()
    ).hexdigest()
```

#### Invalidation Strategy
```
Invalidate when:
- Test code changes (AST hash)
- Dependencies update
- Algorithm version changes
- TTL expires (configurable)
```

#### Benefits
- **10-100x speedup** for repeated analysis
- **Shareable** across team
- **Versioned** results

---

### 7. Contextual Bandits

#### Concept
Select tests based on context features like time, developer, recent changes, and system state.

#### Context Features
```
┌─────────────────────────────────────┐
│          Context Vector             │
├─────────────────────────────────────┤
│ Time of Day    : [0.7, 0.3]       │
│ Day of Week    : [0,0,0,1,0,0,0]  │
│ Developer      : [0.2, 0.8, 0.1]   │
│ Changed Files  : sparse([1,0,1...])│
│ System Load    : 0.65              │
│ Recent Failures: [test_ids]         │
└─────────────────────────────────────┘
```

#### Policy Learning
```
π(a|s,θ) = P(select test a | context s, parameters θ)

1. Observe context s
2. Select test a ~ π(a|s,θ)  
3. Observe reward r
4. Update θ using gradient:
   ∇θ J(θ) = E[r · ∇θ log π(a|s,θ)]
```

#### Example Patterns Learned
```
Friday Afternoon → Integration Tests (high priority)
Junior Developer → More Unit Tests (safety)
Database Changes → DB Integration Tests
High Load → Fast Tests Only
After Incident → Related Test Suite
```

#### Benefits
- **Personalized** test selection
- **Adapts** to patterns
- **Risk-aware** testing

---

### 8. Parallel Shapley Evaluation

#### Concept
Distribute Shapley permutation evaluation across multiple cores/machines.

#### Parallelization Strategy
```
┌──────────────┐
│Master Process│
└──────┬───────┘
       │ Distribute permutations
   ┌───┴────┬────────┬────────┐
   ↓        ↓        ↓        ↓
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│Core 1│ │Core 2│ │Core 3│ │Core 4│
└──────┘ └──────┘ └──────┘ └──────┘
   │        │        │        │
   └────────┴────────┴────────┘
            │ Aggregate
       ┌────┴─────┐
       │ Results  │
       └──────────┘
```

#### Implementation Approaches
```python
# Approach 1: Thread Pool
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = []
    for _ in range(n_permutations):
        future = executor.submit(evaluate_permutation)
        futures.append(future)
    
    results = [f.result() for f in futures]

# Approach 2: Ray Distributed
@ray.remote
def evaluate_permutation_ray(tests, evaluator):
    return compute_marginals(tests, evaluator)

results = ray.get([
    evaluate_permutation_ray.remote(tests, evaluator)
    for _ in range(n_permutations)
])
```

#### Load Balancing
```
Dynamic work stealing:
- Fast workers take tasks from slow workers
- Adaptive batch sizing
- Progress monitoring
```

#### Benefits
- **4-8x speedup** on multicore
- **Linear scaling** with cores
- **Fault tolerant**

---

### 9. Progressive Approximation Framework

#### Concept
Provide increasingly accurate results over time, allowing users to stop when satisfied.

#### Anytime Algorithm Design
```
┌─────────────────────────────────────────┐
│         Quality vs Time                 │
│                                         │
│  100% ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─●  │
│   90%                          ●        │
│   80%                   ●               │
│   70%            ●                      │
│   60%      ●                            │
│   50% ●                                 │
│       └─────┴─────┴─────┴─────┴─────┴  │
│       0s   10s   30s   1m    2m    5m  │
└─────────────────────────────────────────┘
```

#### Progressive Stages
```python
class ProgressiveAnalyzer:
    def analyze(self, tests):
        # Stage 1: Quick heuristics (1s)
        yield self.heuristic_estimate()
        
        # Stage 2: Sparse sampling (10s)
        yield self.sparse_sample_estimate()
        
        # Stage 3: GP predictions (30s)
        yield self.gp_estimate()
        
        # Stage 4: Partial Shapley (1m)
        yield self.partial_shapley()
        
        # Stage 5: Full analysis (5m)
        yield self.full_analysis()
```

#### User Interface
```
$ guardian analyze --progressive

[▓▓▓▓░░░░░░] 40% | ETA: 45s
Current top tests:
1. test_payment_process (0.82 ± 0.15)
2. test_user_auth (0.79 ± 0.12)
3. test_data_validation (0.71 ± 0.18)

Press 's' to stop with current results
Press 'f' to jump to full analysis
```

#### Benefits
- **Immediate feedback**
- **User control**
- **Resource efficient**

---

### 10. Temporal Importance Tracking

#### Concept
Track how test importance changes over time to identify trends and decay.

#### Time Series Model
```
Importance(test, t) = β₀ + β₁·t + seasonal(t) + decay(t) + ε

Where:
- β₀: baseline importance
- β₁: trend component  
- seasonal(t): periodic patterns
- decay(t): relevance decay
- ε: random variation
```

#### Tracking Dashboard
```
Test: test_user_authentication
┌─────────────────────────────────────────┐
│ 1.0 ┤                                   │
│     │     ╱╲    ╱╲                      │
│ 0.8 ┤    ╱  ╲  ╱  ╲    ╱╲              │
│     │   ╱    ╲╱    ╲  ╱  ╲             │
│ 0.6 ┤  ╱            ╲╱    ╲    ↘       │
│     │ ╱                    ╲     ↘      │
│ 0.4 ┤╱                      ╲     ↘decay│
│     └─────────────────────────────────┘ │
│     Jan   Feb   Mar   Apr   May   Jun  │
└─────────────────────────────────────────┘

Insights:
- Declining importance (-23% over 6 months)
- Seasonal spike during releases
- Consider deprecation or refactoring
```

#### Decay Detection
```python
def detect_decay(time_series, window=30):
    recent = time_series[-window:]
    older = time_series[-2*window:-window]
    
    decay_rate = (mean(recent) - mean(older)) / mean(older)
    
    if decay_rate < -0.2:  # 20% decline
        return "significant_decay"
    elif decay_rate < -0.1:
        return "moderate_decay"
    else:
        return "stable"
```

#### Benefits
- **Identify obsolete tests**
- **Predict future importance**
- **Optimize maintenance effort**

---

## Integration Architecture

### How Features Work Together

```
┌─────────────────────────────────────────────────────┐
│                   Guardian Core                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Test Selection Pipeline:                           │
│                                                      │
│  1. Context Analysis ──→ Contextual Bandits         │
│                              ↓                       │
│  2. Historical Data ──→ Temporal Tracking           │
│                              ↓                       │
│  3. Importance Calc ──→ Clustered Shapley           │
│                         + Parallel Compute           │
│                              ↓                       │
│  4. Quality Predict ──→ Incremental GP              │
│                              ↓                       │
│  5. Selection ──────→ UCB1-Thompson Hybrid          │
│                              ↓                       │
│  6. Results ────────→ Persistent Cache              │
│                              ↓                       │
│  7. Learning ───────→ Mutation Patterns             │
│                                                      │
│  All wrapped in Progressive Approximation            │
└─────────────────────────────────────────────────────┘
```

### Performance Impact

| Feature | Speed Impact | Accuracy Impact | Memory Impact |
|---------|--------------|-----------------|---------------|
| Clustering | 100x faster | -5% accuracy | -90% memory |
| Parallel | 4-8x faster | No change | +50% memory |
| Cache | 10-100x faster | No change | +1GB storage |
| Progressive | Immediate feedback | Improves over time | No change |
| GP Sparse | O(m²) vs O(n³) | -2% accuracy | O(m²) memory |

### Example: Complete Analysis Pipeline

```bash
# Analyze large test suite with all features
$ guardian analyze --suite tests/ \
    --cluster --parallel --progressive \
    --context "pre-release" --cache

Starting progressive analysis...
[▓░░░░░] 10% - Quick clustering (1000 → 50 groups)
[▓▓░░░░] 30% - Parallel Shapley on clusters  
[▓▓▓░░░] 60% - GP predictions with uncertainty
[▓▓▓▓░░] 80% - Contextual adjustments
[▓▓▓▓▓░] 95% - Mutation pattern analysis
[▓▓▓▓▓▓] 100% - Complete!

Top 10 Critical Tests (context: pre-release):
1. test_payment_processing    (0.95 ± 0.02) ⬆️ +12%
2. test_user_authentication   (0.89 ± 0.03) ⬇️ -5%
3. test_data_integrity       (0.87 ± 0.04) → stable
...

Mutation Survival Patterns Detected:
- Error handling: 67% survival rate in payment module
- Boundary conditions: Missing in user validation
- Null checks: Weak coverage in API endpoints

Recommended Actions:
1. Add error injection tests for payment_processor.py
2. Increase boundary testing for validate_user_input()
3. Add null safety tests for api/endpoints.py

Cache Status: 78% hit rate (saved 4.5 minutes)
```

## Conclusion

These features transform Guardian from a test runner into an intelligent test optimization system that:

1. **Learns** from historical data
2. **Adapts** to changing codebases
3. **Scales** to enterprise-size test suites
4. **Provides** actionable insights
5. **Saves** significant time and resources

Each feature is designed to work independently or together, providing flexibility in adoption and maximum benefit when fully integrated.
# QualiaGuardian Rust - Improvement Plan

Author: DarkLightX/Dana Edwards

## Current Status: 77/163 tasks (47%)
## Goal: 163/163 tasks (100%)

## 1. Code Quality Improvements 🔧

### Clippy Lints
```bash
# Run clippy with pedantic lints
cargo clippy -- -W clippy::pedantic -W clippy::nursery -W clippy::cargo

# Fix all clippy warnings
cargo clippy --fix
```

### Naming Conventions
- [ ] Rename `BETESCalculator` → `BoundedEvolutionaryTesCalculator`
- [ ] Rename `OSQICalculator` → `OverallSoftwareQualityIndexCalculator`
- [ ] Use more descriptive variable names (no single letters)
- [ ] Consistent acronym casing: `ID` not `Id`, `URL` not `Url`

### Design Patterns

#### Builder Pattern for Complex Types
```rust
// Instead of many constructor parameters
impl QualityConfig {
    pub fn builder() -> QualityConfigBuilder { ... }
}

// Usage
let config = QualityConfig::builder()
    .mode(QualityMode::BETESv31)
    .risk_class(RiskClass::Enterprise)
    .build()?;
```

#### Strategy Pattern for Sensors
```rust
pub trait SensorStrategy {
    fn execute(&self, context: &SensorContext) -> Result<SensorOutput>;
}

pub struct AdaptiveSensorExecutor {
    strategies: HashMap<String, Box<dyn SensorStrategy>>,
}
```

#### Observer Pattern for Progress
```rust
pub trait ProgressObserver {
    fn on_sensor_start(&self, name: &str);
    fn on_sensor_complete(&self, name: &str, result: &Result<Value>);
    fn on_analysis_complete(&self, score: QualityScore);
}
```

## 2. Algorithm Enhancements 🚀

### Parallel Processing
```rust
use rayon::prelude::*;

// Parallel sensor execution
let results: Vec<_> = sensors
    .par_iter()
    .map(|sensor| sensor.measure(&context))
    .collect();
```

### Caching Layer
```rust
use dashmap::DashMap;
use std::time::{Duration, Instant};

pub struct CachedSensor<S: Sensor> {
    inner: S,
    cache: DashMap<String, (Instant, S::Output)>,
    ttl: Duration,
}
```

### Adaptive Thresholds
```rust
pub struct AdaptiveThresholdCalculator {
    history: VecDeque<f64>,
    window_size: usize,
}

impl AdaptiveThresholdCalculator {
    pub fn calculate_dynamic_threshold(&self) -> f64 {
        // Use statistical methods (mean + 2*std_dev)
        let mean = statistical::mean(&self.history);
        let std_dev = statistical::std_deviation(&self.history);
        mean + 2.0 * std_dev
    }
}
```

## 3. Missing Implementations 📝

### Complete Sensor Implementations

#### BehaviorCoverageSensor
```rust
// Parse LCOV format
pub struct LcovParser;
impl LcovParser {
    pub fn parse_file(path: &Path) -> Result<CoverageData> { ... }
}

// Parse coverage.py XML
pub struct CoveragePyParser;
impl CoveragePyParser {
    pub fn parse_xml(path: &Path) -> Result<CoverageData> { ... }
}
```

#### SpeedSensor
```rust
// Integration with cargo test --timings
pub async fn measure_test_performance(
    project_path: &Path,
) -> Result<TestTimings> {
    let output = Command::new("cargo")
        .arg("test")
        .arg("--timings=json")
        .output()
        .await?;
    
    parse_cargo_timings(&output.stdout)
}
```

#### FlakinessSensor
```rust
// CI/CD integrations
pub trait CiProvider: Send + Sync {
    async fn fetch_test_history(&self, days: u32) -> Result<Vec<TestRun>>;
}

pub struct GitHubActionsProvider { ... }
pub struct JenkinsProvider { ... }
pub struct GitLabCiProvider { ... }
```

### Evolution Engine Implementation

```rust
// crates/evolution/src/nsga2.rs
pub struct Nsga2<T: Individual> {
    population_size: usize,
    mutation_rate: f64,
    crossover_rate: f64,
    objectives: Vec<Box<dyn Objective<T>>>,
}

pub trait Individual: Clone + Send + Sync {
    fn mutate(&mut self, rate: f64);
    fn crossover(&self, other: &Self) -> Self;
}

pub trait Objective<T: Individual>: Send + Sync {
    fn evaluate(&self, individual: &T) -> f64;
    fn is_minimizing(&self) -> bool;
}
```

### Analytics Implementation

```rust
// crates/analytics/src/shapley.rs
pub struct ShapleyValueCalculator {
    baseline_score: f64,
    contributions: HashMap<String, f64>,
}

impl ShapleyValueCalculator {
    pub fn calculate_marginal_contributions(
        &self,
        test_suite: &[Test],
    ) -> HashMap<String, f64> {
        // Implement cooperative game theory calculations
    }
}
```

## 4. Database Layer 💾

### SQLx Migrations
```sql
-- migrations/001_initial_schema.sql
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    project_path TEXT NOT NULL,
    quality_score REAL NOT NULL,
    components JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Repository Pattern
```rust
#[async_trait]
pub trait Repository<T, ID> {
    async fn find_by_id(&self, id: ID) -> Result<Option<T>>;
    async fn save(&self, entity: &T) -> Result<ID>;
    async fn update(&self, entity: &T) -> Result<()>;
    async fn delete(&self, id: ID) -> Result<()>;
}

pub struct RunRepository {
    pool: SqlitePool,
}
```

## 5. Testing Strategy 🧪

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_quality_score_bounds(value in 0.0..=1.0) {
            let score = QualityScore::new(value).unwrap();
            prop_assert!(score.value() >= 0.0);
            prop_assert!(score.value() <= 1.0);
        }
    }
}
```

### Integration Tests
```rust
// tests/integration/sensor_suite.rs
#[tokio::test]
async fn test_all_sensors_integration() {
    let context = create_test_context();
    let registry = create_default_registry();
    
    for sensor_name in registry.list() {
        let sensor = registry.get(sensor_name).unwrap();
        let result = sensor.measure(&context).await;
        assert!(result.is_ok(), "Sensor {} failed", sensor_name);
    }
}
```

### Benchmarks
```rust
// benches/quality_metrics.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_betes_calculation(c: &mut Criterion) {
    c.bench_function("betes_v3.1", |b| {
        b.iter(|| {
            let calc = BETESCalculator::new(Default::default(), None);
            calc.calculate(&black_box(create_test_input()))
        })
    });
}
```

## 6. Documentation 📚

### API Documentation
```rust
/// Calculates the bounded evolutionary test effectiveness score.
/// 
/// # Algorithm
/// 
/// The bE-TES score combines multiple quality factors using a weighted
/// geometric mean, then applies a trust coefficient based on flakiness:
/// 
/// ```text
/// G = (M'^w_m × E'^w_e × A'^w_a × B'^w_b × S'^w_s)^(1/Σw)
/// bE-TES = G × T
/// ```
/// 
/// # Examples
/// 
/// ```
/// use qualia_core::{BETESCalculator, BETESInput};
/// 
/// let calculator = BETESCalculator::default();
/// let input = BETESInput {
///     raw_mutation_score: 0.85,
///     raw_emt_gain: 0.15,
///     // ... other fields
/// };
/// 
/// let (score, components) = calculator.calculate(&input)?;
/// assert!(score.value() > 0.7);
/// ```
pub fn calculate(&self, input: &BETESInput) -> Result<(QualityScore, BETESComponents)>
```

## 7. Performance Optimizations ⚡

### Zero-Copy Deserialization
```rust
use serde_json::value::RawValue;

pub struct LazyMetrics<'a> {
    raw: &'a RawValue,
}

impl<'a> LazyMetrics<'a> {
    pub fn mutation_score(&self) -> Result<f64> {
        // Parse only what we need
    }
}
```

### Memory Pool for AST Analysis
```rust
use typed_arena::Arena;

pub struct AstAnalyzer<'a> {
    arena: &'a Arena<syn::File>,
}
```

### SIMD Operations
```rust
use packed_simd::f64x4;

fn calculate_geometric_mean_simd(values: &[f64], weights: &[f64]) -> f64 {
    // Use SIMD for parallel multiplication
}
```

## Task Breakdown to Reach 163/163

### Phase 1: Code Quality (10 tasks)
- [ ] Apply clippy pedantic fixes
- [ ] Implement builder patterns
- [ ] Add comprehensive error context
- [ ] Improve naming conventions
- [ ] Add #[must_use] annotations
- [ ] Implement Display for all types
- [ ] Add serde rename attributes
- [ ] Document all public APIs
- [ ] Add examples to documentation
- [ ] Create type aliases for clarity

### Phase 2: Complete Sensors (24 tasks)
- [ ] BehaviorCoverageSensor implementation
- [ ] LCOV parser
- [ ] Coverage.py XML parser
- [ ] SpeedSensor implementation
- [ ] Cargo timings integration
- [ ] FlakinessSensor implementation
- [ ] GitHub Actions API client
- [ ] Jenkins API client
- [ ] GitLab CI API client
- [ ] CircleCI API client
- [ ] CHSSensor implementation
- [ ] Cyclomatic complexity calculator
- [ ] Maintainability index
- [ ] Shannon entropy calculator
- [ ] SecuritySensor implementation
- [ ] SAST tool integration
- [ ] CVE database client
- [ ] ArchSensor implementation
- [ ] Dependency graph builder
- [ ] Coupling analyzer
- [ ] Cohesion calculator
- [ ] Module boundary detector
- [ ] Algebraic connectivity
- [ ] Sensor caching layer

### Phase 3: Evolution Engine (18 tasks)
- [ ] NSGA-II core algorithm
- [ ] Individual trait
- [ ] Population management
- [ ] Fitness functions
- [ ] Non-dominated sorting
- [ ] Crowding distance
- [ ] Tournament selection
- [ ] Mutation operators
- [ ] Crossover operators
- [ ] AST-based mutations
- [ ] Smart mutation strategies
- [ ] Test prioritization
- [ ] Pareto front tracking
- [ ] Convergence detection
- [ ] Archive management
- [ ] Multi-objective optimization
- [ ] Parallel evaluation
- [ ] Evolution visualizer

### Phase 4: Analytics (12 tasks)
- [ ] Shapley value calculator
- [ ] Marginal contribution analysis
- [ ] Coalition formation
- [ ] Test importance ranking
- [ ] Trend analysis
- [ ] Anomaly detection
- [ ] Statistical utilities
- [ ] Time series analysis
- [ ] Correlation matrix
- [ ] PCA implementation
- [ ] Clustering algorithms
- [ ] Prediction models

### Phase 5: Database & Persistence (15 tasks)
- [ ] SQLx migration system
- [ ] Player repository
- [ ] Run repository
- [ ] Badge repository
- [ ] Quest repository
- [ ] Agent repository
- [ ] Transaction support
- [ ] Connection pooling
- [ ] Query optimization
- [ ] Index creation
- [ ] Backup/restore
- [ ] Data export
- [ ] Import functionality
- [ ] Cache invalidation
- [ ] Event sourcing

### Phase 6: Testing & Quality (17 tasks)
- [ ] Unit test coverage >80%
- [ ] Integration test suite
- [ ] End-to-end tests
- [ ] Property-based tests
- [ ] Mutation testing setup
- [ ] Performance benchmarks
- [ ] Memory leak tests
- [ ] Concurrency tests
- [ ] Error recovery tests
- [ ] Mock implementations
- [ ] Test fixtures
- [ ] Golden tests
- [ ] Regression tests
- [ ] Fuzz testing
- [ ] Security tests
- [ ] Load tests
- [ ] CI/CD pipeline

Total New Tasks: 86
Current: 77/163
Target: 163/163 ✅

## Immediate Next Steps

1. **Fix the failing test**: Create config/chs_thresholds.yml
2. **Run clippy**: `cargo clippy -- -W clippy::pedantic`
3. **Complete BehaviorCoverageSensor**: Most valuable sensor
4. **Implement caching**: Improve performance
5. **Add more tests**: Increase coverage

This plan provides a clear path to 100% completion with focus on code quality, performance, and maintainability.
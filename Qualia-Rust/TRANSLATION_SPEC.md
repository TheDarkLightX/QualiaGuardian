# QualiaGuardian Rust Translation Specification

Author: DarkLightX/Dana Edwards

## Project Overview

QualiaGuardian is an autonomous code quality optimization system being translated from Python to Rust. This document outlines the translation approach, architecture decisions, and implementation guidelines.

## Architecture Mapping

### Core Components

#### 1. Quality Metrics System
**Python Module**: `guardian/core/`
**Rust Crate**: `crates/core/`

- **TES (Test Effectiveness Score)**
  - Python: `guardian/core/tes.py`
  - Rust: `crates/core/src/tes.rs`
  - Key trait: `QualityMetric`

- **bE-TES (Bounded Evolutionary TES)**
  - Python: `guardian/core/betes.py`
  - Rust: `crates/core/src/betes.rs`
  - Uses sigmoid bounds and evolutionary scoring

- **OSQI (Overall Software Quality Index)**
  - Python: `guardian/core/osqi.py`
  - Rust: `crates/core/src/osqi.rs`
  - Holistic quality measurement

#### 2. Sensor Architecture
**Python Module**: `guardian/sensors/`
**Rust Crate**: `crates/sensors/`

Common trait interface:
```rust
#[async_trait]
pub trait Sensor: Send + Sync {
    type Output: Serialize + DeserializeOwned;
    type Error: Error + Send + Sync + 'static;
    
    async fn measure(&self, context: &SensorContext) -> Result<Self::Output, Self::Error>;
    fn name(&self) -> &'static str;
}
```

Sensors to implement:
- MutationSensor (mutmut integration)
- AssertionIQSensor (AST analysis)
- BehaviorCoverageSensor (LCOV/coverage.py)
- SpeedSensor (benchmark integration)
- FlakinessSensor (CI/CD integration)
- CHSSensor (Code Health Score)
- SecuritySensor (vulnerability detection)
- ArchSensor (architectural metrics)

#### 3. Evolutionary Engine
**Python Module**: `guardian/evolution/`
**Rust Crate**: `crates/evolution/`

- NSGA-II multi-objective optimization
- Smart mutation operators
- Parallel population evaluation
- Fitness function traits

#### 4. CLI Interface
**Python Module**: `guardian/cli/`
**Rust Crate**: `crates/cli/`

- Use `clap` for argument parsing
- Use `indicatif` for progress bars
- Use `comfy-table` for output formatting
- Support all existing commands

#### 5. Database Layer
**Python**: SQLite via direct queries
**Rust**: SQLx with compile-time checked queries

Schema remains identical:
- players
- runs
- badges
- quests
- agent_runs

## Type System Design

### Core Types

```rust
// Quality scores bounded [0.0, 1.0]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct QualityScore(f64);

impl QualityScore {
    pub fn new(value: f64) -> Result<Self> {
        if (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(QualityError::ScoreOutOfBounds)
        }
    }
}

// Risk classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RiskClass {
    Aerospace,    // >0.95
    Medical,      // >0.90
    Financial,    // >0.80
    Enterprise,   // >0.70
    Standard,     // >0.60
    Prototype,    // >0.40
    Experimental, // >=0.0
}

// Component results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentResult {
    pub name: String,
    pub raw_value: f64,
    pub normalized_value: QualityScore,
    pub weight: f64,
}
```

## Error Handling Strategy

Use `thiserror` for domain errors:

```rust
#[derive(Debug, thiserror::Error)]
pub enum GuardianError {
    #[error("Sensor error: {0}")]
    Sensor(#[from] SensorError),
    
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),
    
    #[error("Evolution error: {0}")]
    Evolution(#[from] EvolutionError),
}
```

## Concurrency Model

1. **Async Runtime**: Tokio for all async operations
2. **Parallel Sensors**: Run independent sensors concurrently
3. **Population Evolution**: Rayon for parallel fitness evaluation
4. **Database Pool**: Connection pooling with SQLx

## External Integrations

### Python Interop (for mutmut)
Use `pyo3` for Python integration:
```rust
use pyo3::prelude::*;

#[pyfunction]
fn run_mutation_testing(path: &str) -> PyResult<MutationResults> {
    // Call mutmut via Python
}
```

### CI/CD APIs
Use `reqwest` with platform-specific auth:
- GitHub Actions API
- Jenkins REST API
- GitLab CI API
- CircleCI API

### Coverage Tools
- Parse LCOV format natively
- Parse coverage.py XML/JSON

## Configuration

Use `serde` with YAML/TOML support:

```rust
#[derive(Debug, Deserialize)]
pub struct GuardianConfig {
    pub quality_mode: QualityMode,
    pub risk_thresholds: HashMap<RiskClass, f64>,
    pub sensor_weights: HashMap<String, f64>,
    pub evolution: EvolutionConfig,
}
```

## Testing Strategy

1. **Unit Tests**: Test each component in isolation
2. **Integration Tests**: Test sensor combinations
3. **Property Tests**: Use `proptest` for invariants
4. **Benchmarks**: Use `criterion` for performance

## Performance Targets

- Sensor execution: <100ms per sensor
- Quality calculation: <10ms
- Database queries: <50ms p99
- CLI response: <200ms for analysis
- Memory usage: <100MB for typical project

## Migration Phases

### Phase 1: Core Infrastructure
1. Set up workspace structure
2. Implement core quality metrics
3. Create sensor trait system
4. Set up database layer

### Phase 2: Sensor Implementation
1. Port all sensor modules
2. Implement parallel execution
3. Add caching layer
4. Integration tests

### Phase 3: Evolution Engine
1. Port NSGA-II algorithm
2. Implement mutation operators
3. Add parallel fitness evaluation
4. Benchmark performance

### Phase 4: CLI and Integration
1. Port all CLI commands
2. Add progress indicators
3. Implement output formatters
4. End-to-end tests

### Phase 5: Advanced Features
1. Agent system with LLM integration
2. Self-improvement capabilities
3. Gamification features
4. Web API

## Code Style Guidelines

1. Use `rustfmt` with default settings
2. Use `clippy` with pedantic lints
3. Document all public APIs
4. Use descriptive variable names
5. Prefer composition over inheritance
6. Use Result<T, E> for fallible operations
7. Avoid unwrap() in production code

## Dependencies Policy

- Prefer well-maintained crates
- Minimize dependency tree
- Audit dependencies regularly
- Pin major versions
- Use cargo-deny for checks

## Security Considerations

1. Validate all external input
2. Use secure random for crypto
3. Sanitize file paths
4. Limit resource consumption
5. Audit dependencies for CVEs
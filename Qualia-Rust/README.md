# QualiaGuardian - Rust Implementation

Author: DarkLightX/Dana Edwards

## Overview

This is the Rust translation of QualiaGuardian, an autonomous code quality optimization system. The project is structured as a Cargo workspace with multiple crates for modularity and reusability.

## Project Structure

```
Qualia-Rust/
├── Cargo.toml                 # Workspace root
├── TRANSLATION_SPEC.md        # Detailed translation specification
├── IMPLEMENTATION_CHECKLIST.md # Progress tracking
├── crates/
│   ├── core/                  # Core quality metrics
│   │   ├── src/
│   │   │   ├── lib.rs        # Main library exports
│   │   │   ├── error.rs      # Error types
│   │   │   ├── types.rs      # Core types (QualityScore, RiskClass, etc.)
│   │   │   ├── config.rs     # Configuration structures
│   │   │   ├── traits.rs     # Core traits
│   │   │   ├── tes.rs        # Main quality score dispatcher
│   │   │   ├── betes.rs      # bE-TES implementation
│   │   │   └── osqi.rs       # OSQI implementation
│   │   └── Cargo.toml
│   ├── sensors/               # Sensor implementations
│   │   ├── src/
│   │   │   ├── lib.rs        # Sensor trait and registry
│   │   │   ├── mutation.rs   # Mutation testing sensor
│   │   │   ├── assertion_iq.rs # Assertion quality sensor
│   │   │   ├── behaviour_coverage.rs
│   │   │   ├── speed.rs
│   │   │   ├── flakiness.rs
│   │   │   ├── chs.rs        # Code Health Score
│   │   │   ├── security.rs
│   │   │   └── arch.rs       # Architecture metrics
│   │   └── Cargo.toml
│   ├── analytics/             # Analytics and metrics
│   │   └── Cargo.toml
│   ├── evolution/             # Evolutionary algorithms
│   │   └── Cargo.toml
│   └── cli/                   # Command-line interface
│       ├── src/
│       │   ├── main.rs       # CLI entry point
│       │   ├── commands/     # Command implementations
│       │   │   ├── mod.rs
│       │   │   ├── analyze.rs
│       │   │   ├── evolve.rs
│       │   │   ├── gamify.rs
│       │   │   ├── history.rs
│       │   │   └── self_improve.rs
│       │   ├── output.rs     # Output formatting
│       │   └── database.rs   # Database layer
│       └── Cargo.toml
```

## Implementation Status

### ✅ Completed
- Core workspace structure
- Quality metric implementations (TES, bE-TES, OSQI)
- Sensor trait architecture
- Basic sensor implementations
- CLI framework with clap
- Output formatting (text, JSON, markdown, HTML)

### 🚧 In Progress
- Full sensor implementations (currently have stubs)
- Database layer with SQLx
- Evolution engine
- Analytics modules

### 📋 TODO
- Complete sensor implementations with actual analysis
- Implement NSGA-II evolutionary algorithm
- Add Shapley value calculations
- Create integration tests
- Add benchmarks
- Documentation

## Building

```bash
# Install Rust if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build the project
cargo build

# Run tests
cargo test

# Run the CLI
cargo run --bin guardian -- --help
```

## Usage Examples

```bash
# Analyze a project with bE-TES v3.1
cargo run --bin guardian -- analyze /path/to/project --run-quality --quality-mode betes_v3.1

# Run specific sensors
cargo run --bin guardian -- analyze /path/to/project --sensors mutation,assertion_iq

# Evaluate against risk class
cargo run --bin guardian -- analyze /path/to/project --run-quality --risk-class standard

# Show gamification status
cargo run --bin guardian -- gamify status

# View analysis history
cargo run --bin guardian -- history --limit 20
```

## Architecture Highlights

### Type Safety
- All quality scores are bounded to [0.0, 1.0] using the `QualityScore` type
- Risk classes and quality modes use enums for compile-time safety
- Comprehensive error handling with custom error types

### Async/Concurrent Design
- Sensors run asynchronously using Tokio
- Parallel sensor execution for performance
- Database operations use async SQLx

### Modularity
- Clear separation between core metrics, sensors, and CLI
- Trait-based sensor system allows easy extension
- Each crate can be used independently

### Performance
- Zero-copy deserialization where possible
- Efficient parallel processing with Rayon
- Careful memory management

## Configuration

The system uses YAML configuration files:
- `config/risk_classes.yml` - Risk class thresholds
- `config/chs_thresholds.yml` - Code health thresholds by language

## Contributing

See IMPLEMENTATION_CHECKLIST.md for current progress and tasks.

## License

MIT License - See LICENSE file for details.
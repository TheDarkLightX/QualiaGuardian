# bE-TES v3.0 Framework and Sensor Integration

This document outlines the architecture and usage of the Bounded Evolutionary Test-Effectiveness Score (bE-TES v3.0) framework within the Guardian AI Tool, including its sensor-based data collection mechanism.

## 1. Overview of bE-TES v3.0

The bE-TES v3.0 is a comprehensive metric designed to provide a bounded, interpretable, and robust measure of test suite quality.

**Core Formula:**
`bE-TES = G * T`

Where:
*   **G (Weighted Geometric Mean):** `(M'^(w_M) * E'^(w_E) * A'^(w_A) * B'^(w_B) * S'^(w_S))^(1 / sum(w))`
    *   `M'`: Normalized Mutation Score (0-1)
    *   `E'`: Normalized EMT Gain (0-1)
    *   `A'`: Normalized Assertion IQ (0-1, from a 1-5 rubric score)
    *   `B'`: Normalized Behaviour Coverage (0-1)
    *   `S'`: Normalized Speed Factor (0-1, from median test time)
    *   `w_X`: Configurable weights for each factor (default to 1.0).
*   **T (Trust Coefficient):** `1 - flakiness_rate` (0-1)

The score is designed to be between 0 and 1, providing a clear gauge of test effectiveness.

## 2. Coexistence with E-TES v2.0

The bE-TES v3.0 framework is designed to coexist with the existing E-TES v2.0. Users can select the desired scoring mode via CLI arguments.

*   **Selection:** Use the `--quality-mode` CLI argument.
    *   `--quality-mode betes_v3` for bE-TES v3.0.
    *   `--quality-mode etes_v2` for E-TES v2.0 (default if `--run-quality` is specified without a mode).

## 3. Core Modules

The implementation is distributed across several core modules:

*   **`guardian.core.betes`**:
    *   `BETESWeights`: Dataclass for bE-TES component weights.
    *   `BETESComponents`: Dataclass to store raw inputs, normalized factors, intermediate calculations, and the final bE-TES score.
    *   `BETESCalculator`: Class responsible for the bE-TES calculation logic using the formulas above.
    *   `classify_betes`: Function to evaluate a bE-TES score against predefined risk thresholds.

*   **`guardian.core.etes`**:
    *   `QualityConfig`: Dataclass (evolved from `ETESConfig`) holding configuration for both E-TES v2.0 and bE-TES v3.0. This includes the scoring `mode`, `betes_weights`, `risk_class` for bE-TES, and paths for sensor data.

*   **`guardian.core.tes`**:
    *   `calculate_quality_score`: The main dispatch function. It reads the `mode` from `QualityConfig` and routes the calculation to either `BETESCalculator` or `ETESCalculator`. It's responsible for passing the correct data (raw metrics for bE-TES, or `test_suite_data`/`codebase_data` for E-TES v2.0).
    *   `get_etes_grade`: Utility function to convert a 0-1 score into a letter grade (A+, A, B, etc.), applicable to both E-TES v2.0 and bE-TES v3.0.

## 4. Configuration

### `QualityConfig`
Located in `guardian.core.etes`, this dataclass centralizes configuration. Key fields for bE-TES include:
*   `mode: Literal["etes_v2", "betes_v3"]`
*   `betes_weights: BETESWeights`
*   `risk_class: Optional[str]`
*   `test_root_path: Optional[str]` (for Assertion IQ sensor)
*   `coverage_file_path: Optional[str]` (for Behaviour Coverage sensor)
*   `critical_behaviors_manifest_path: Optional[str]` (for Behaviour Coverage sensor)
*   `ci_platform: Optional[str]` (for Flakiness sensor)
*   (Implicitly, paths for mutation reports and pytest report logs are also needed and passed via CLI).

### `risk_classes.yml`
*   **Location:** `guardian_ai_tool/config/risk_classes.yml`
*   **Purpose:** Defines bE-TES score thresholds for different project risk classifications (e.g., Standard SaaS, Financial).
*   **Format:** YAML dictionary where keys are risk class names.
    ```yaml
    standard_saas:
      min_score: 0.75
    financial:
      min_score: 0.85
    medical_class_iii:
      min_score: 0.92
    ```

## 5. CLI Usage (`guardian_ai_tool/guardian/cli.py`)

The main CLI script orchestrates the analysis and quality scoring.

*   **To run quality analysis:** Use the `--run-quality` flag.
*   **Key Arguments for bE-TES:**
    *   `--quality-mode betes_v3`: Selects the bE-TES v3.0 calculator.
    *   `--risk-class <class_name>`: (Optional) Specifies a risk class from `risk_classes.yml` for threshold evaluation.
    *   `--betes-w-m <float>`, `--betes-w-e <float>`, etc.: To set custom weights for bE-TES factors.
    *   `--test-root-path <path>`: Path to test files for Assertion IQ.
    *   `--coverage-file-path <path>`: Path to LCOV or similar coverage file.
    *   `--critical-behaviors-manifest-path <path>`: Path to critical behaviors YAML.
    *   `--pytest-reportlog-path <path>`: Path to `pytest-reportlog` JSONL output for speed metrics.
    *   `--ci-platform <name>`: Identifier for the CI platform (e.g., `gh-actions`) for flakiness.
    *   `--project-ci-identifier <id>`: Project ID/slug for CI API.

*   **Example Command:**
    ```bash
    python guardian_ai_tool/guardian/cli.py ./my_project --run-quality --quality-mode betes_v3 --risk-class standard_saas --pytest-reportlog-path ./reports/report.jsonl --coverage-file-path ./coverage.lcov
    ```

## 6. Sensor Architecture (`guardian.sensors`)

The `guardian.sensors` package is responsible for autonomously collecting the raw metrics needed for bE-TES. The CLI (`cli.py`) calls these sensor functions when bE-TES mode is active.

*   **Current (Placeholder) Sensor Modules:**
    *   `mutation.py`: For Mutation Score (M) and EMT Gain (E). (Simulates running a tool and caches previous MS).
    *   `assertion_iq.py`: For Assertion IQ (A). (Simulates static analysis of test files).
    *   `behaviour_coverage.py`: For Behaviour Coverage (B). (Simulates parsing LCOV and a manifest).
    *   `speed.py`: For Speed Factor (S) via Median Test Time. (Simulates parsing `pytest-reportlog`).
    *   `flakiness.py`: For Flake Rate (for Trust coefficient T). (Simulates CI API interaction).

*   **Note:** As of this document, these sensor modules contain placeholder logic. Future work involves implementing them to interact with actual tools and data sources.

## 7. Data Flow for bE-TES Calculation (Conceptual)

1.  User invokes `guardian ... --run-quality --quality-mode betes_v3` with relevant path arguments.
2.  `cli.py` parses arguments and creates `QualityConfig`.
3.  `cli.py` calls functions from `guardian.sensors.*` modules, passing necessary paths and configurations.
4.  Each sensor function (currently placeholder) returns its specific raw metric:
    *   Raw Mutation Score, Raw EMT Gain
    *   Raw Mean Assertion IQ (1-5)
    *   Raw Behaviour Coverage Ratio
    *   Raw Median Test Time (ms)
    *   Raw Flakiness Rate
5.  `cli.py` collects these into a `raw_metrics_betes` dictionary.
6.  This dictionary and `QualityConfig` are passed to `calculate_quality_score` in `tes.py`.
7.  `calculate_quality_score` (seeing mode is `betes_v3`) instantiates `BETESCalculator` (from `betes.py`) with weights from `QualityConfig`.
8.  `BETESCalculator.calculate()` is called with the raw metrics.
    *   It normalizes each metric.
    *   Calculates G (weighted geometric mean) and T (trust).
    *   Computes the final bE-TES score.
    *   Returns a `BETESComponents` object.
9.  If a `--risk-class` was specified, `cli.py` calls `classify_betes` (from `betes.py`) with the score and loaded `risk_classes.yml` data.
10. The final score, components, and classification (if any) are added to the main results and formatted for output.

This framework provides a modular and extensible way to calculate bE-TES v3.0, with clear separation of concerns between CLI, configuration, core calculation logic, and data sensing.
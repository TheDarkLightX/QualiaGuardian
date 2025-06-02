# Guardian AI Tool: Implementation Status for EQRA Self-Improvement

**Last Updated:** 2025-06-02

This document summarizes the current implementation status of the Guardian AI tool, with a focus on the components required for the EQRA-based self-improvement system. It is based on reviews of project documentation and discussions up to the date above.

## I. Core Quality Metrics

### A. OSQI (Overall Software Quality Index)
- **Status:** Design specified in `bE_TES_v3.1_OSQI_v1.0_implementation_plan.md`.
- **Details:**
    - Calculation Logic: Assumed to be planned or partially implemented as per its design document.
    - Sensor Integration: Dependent on individual sensor development (see bE-TES sensors).
    - CLI Integration: Likely planned as part of OSQI rollout.

### B. bE-TES (Bounded Evolutionary Test-Effectiveness Score)
- **Status:** Core calculation logic and CLI integration are implemented. Sensors are placeholders.
- **Details (based on `betes_framework.md`):**
    - Calculation Logic (`BETESCalculator` in `guardian.core.betes`): Implemented.
    - Core Modules (`guardian.core.betes`, `guardian.core.etes`, `guardian.core.tes`): Implemented.
    - Configuration (`QualityConfig`, `risk_classes.yml`): Implemented.
    - CLI Integration (`guardian/cli.py`): Implemented for invoking bE-TES calculation and using its parameters.
    - **Sensor Architecture (`guardian.sensors`):** Defined, but individual sensor modules currently contain **placeholder logic**.
        - Mutation Score & EMT Gain Sensor (`mutation.py`): Placeholder.
        - Assertion IQ Sensor (`assertion_iq.py`): Placeholder.
        - Behaviour Coverage Sensor (`behaviour_coverage.py`): Placeholder.
        - Speed Factor Sensor (`speed.py`): Placeholder.
        - Flakiness Sensor (`flakiness.py`): Placeholder.

## II. EQRA-Based Self-Improvement System

This system is the primary focus for new development.

### A. EQRAScheduler (`guardian/scheduler.py` - To be created)
- **Status:** Planned / Design Phase.
- **Details:**
    - Core Loop Logic (action selection, budget exhaustion): Designed in discussion, not yet implemented.
    - EQRA Calculation Function (incorporating E[ΔQ|a], Var[ΔQ|a], C(a), λ): Designed, not yet implemented.
    - Budget Management: Designed, not yet implemented.
    - Action Selection (choosing highest positive EQRA): Designed, not yet implemented.
    - History Logging (tracking actions, ΔQ, costs, EQRA scores): Planned.
    - Persistence of Action Stats: Planned.

### B. ActionProvider Framework (`guardian/actions.py` or similar - To be created)
- **Status:** Planned / Design Phase.
- **Details:**
    - `ActionProvider` Base Class (abstract methods for `estimate_cost`, `run`, stats management): Designed in discussion, not yet implemented.
    - `ActionStats` Dataclass (to hold `n_a`, `delta_q_bar`, `m2`, priors `mu0`, `sigma0_sq`): Designed, not yet implemented.
    - Posterior Update Mechanism (Welford's algorithm for `delta_q_bar`, `m2`): Logic defined, to be implemented within `ActionProvider` or a helper.
    - Concrete Action Implementations (e.g., `flake_heal`, `auto_test`, `dep_autobump`): Not Implemented. These will be specific fixers/enhancers Guardian can apply.
        - Mock Actions (`MockFixedImprovementAction`, `MockRiskyAction`, `NoOpAction`): Designed for testing the scheduler, not yet implemented.

### C. Cost Estimation (`C(a)`)
- **Status:** Planned / Conceptual.
- **Details:**
    - General Framework: Formula discussed (e.g., $C(a) = \beta_0 + \beta_1\,\text{pop_size}+\beta_2\,\text{generations}$ for some actions, or fixed costs for others).
    - Learning Mechanism for Cost Coefficients (e.g., OLS on past run data): Not Implemented.
    - Integration with `ActionProvider.estimate_cost()`: Planned.

### D. Gamification Hook (XP Rewards)
- **Status:** Planned / Conceptual.
- **Details:**
    - XP Formula (e.g., $\text{XP} = \kappa \times \frac{\Delta Q}{C(a)}\times 100$): Defined in EQRA specification.
    - Integration with `EQRAScheduler` after an action runs: Planned.

## III. Supporting Systems & Other Features

### A. Configuration System
- **Status:** Partially Implemented.
- **Details:**
    - `QualityConfig` for bE-TES: Implemented.
    - `risk_classes.yml` for bE-TES: Implemented.
    - Configuration for EQRA (e.g., risk aversion $\lambda$, default priors for actions, action registry): Not Implemented.

### B. Command Line Interface (CLI - `guardian/cli.py`)
- **Status:** Partially Implemented.
- **Details:**
    - Invocation of bE-TES/OSQI quality score calculations: Implemented.
    - Invocation, configuration, and control of the EQRA Scheduler / Self-Improvement Loop: Not Implemented.

### C. BDD Test Scenarios
- **Status:** High-level scenarios implemented (as per `BDD_IMPLEMENTATION_SUMMARY.md`).
- **Details:**
    - Scenarios like "User enables Guardian self-improvement mode" and "User configures improvement selection modes" exist.
    - These test the *expected high-level behavior* of a self-improvement system. They do not yet test the specific mechanics of the EQRA implementation as it's not built.

### D. Other "Innovation Tracks" (from user-provided document, considered future work relative to EQRA)
- **Status:** Future / Not Implemented.
- **Details:**
    - Test-Shapley Analytics: Not Implemented.
    - MPC Budget Controller: Not Implemented.
    - Quality-Debt Index (QDI): Not Implemented.
    - Reinforcement-Learning Action Scheduler (Thompson Sampling): Not Implemented (EQRA with Welford/Normal-Inverse-Gamma is the current focus).
    - Seasonal Gamification & Marketplace: Not Implemented.

## IV. Summary of What's Immediately Next for EQRA Implementation

1.  **Create `guardian/actions.py` (or similar):**
    - Implement the `ActionStats` dataclass.
    - Implement the `ActionProvider` abstract base class, including the `update_stats` method with Welford's algorithm logic and methods for `get_expected_delta_q` and `get_variance_delta_q`.
2.  **Create Mock Actions (e.g., in `guardian/mock_actions.py`):**
    - Implement `MockFixedImprovementAction`.
    - Implement `MockRiskyAction`.
    - Implement `NoOpAction`.
3.  **Create `guardian/scheduler.py`:**
    - Implement the `EQRAScheduler` class.
    - Implement the `_calculate_eqra` method.
    - Implement the main `run_cycle` loop, including action selection, running actions, budget updates, and calling `update_stats`.
    - Implement basic history logging.
    - Implement placeholder `_get_current_quality_score` and `_verify_action_impact` (or integrate with mock action reporting).
4.  **Create an initial test script** to instantiate the scheduler with mock actions and run a cycle to observe behavior.
5.  **Refine persistence strategy** for action statistics.

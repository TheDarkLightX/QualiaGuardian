# EQRA Implementation Checklist

This checklist tracks the implementation of the EQRA (Expected Quality Return on Action) self-improvement system for Guardian AI.

**Phase 1: Core Framework, BDD Scenarios & Initial Unit Tests**

1.  **Core Action Infrastructure (`guardian_ai_tool/guardian/core/actions.py`)**
    *   [x] Create `ActionStats` dataclass.
    *   [x] Create `ActionProvider` abstract base class.

2.  **Mock Action Implementations (`guardian_ai_tool/guardian/actions/`)**
    *   [x] Create `guardian_ai_tool/guardian/actions/` directory.
    *   [x] Create `guardian_ai_tool/guardian/actions/__init__.py`.
    *   [x] Create `guardian_ai_tool/guardian/actions/mock_actions.py` (with `MockFixedImprovementAction`, `MockRiskyAction`, `NoOpAction`).

3.  **EQRA Scheduler (`guardian_ai_tool/guardian/core/scheduler.py`)**
    *   [x] Ensure `guardian_ai_tool/guardian/__init__.py` exists.
    *   [x] Ensure `guardian_ai_tool/guardian/core/__init__.py` exists.
    *   [ ] **CRITICAL PENDING USER ACTION:** Create `guardian_ai_tool/guardian/core/scheduler.py` with the `EQRAScheduler` class (content previously provided by Cascade). **This is required to run BDD and Unit tests.**
        *   _Includes EQRA calculation, action selection, budget management, history logging, `run_cycle` method._

4.  **BDD Test Setup & Scenarios (`guardian_ai_tool/tests/bdd/`)**
    *   [x] Ensure `guardian_ai_tool/tests/__init__.py` exists.
    *   [x] Create `guardian_ai_tool/tests/bdd/` directory.
    *   [x] Create `guardian_ai_tool/tests/bdd/__init__.py`.
    *   [x] Create `guardian_ai_tool/tests/bdd/features/` directory.
    *   [x] Create `guardian_ai_tool/tests/bdd/features/eqra_self_improvement.feature` (initial scenarios for EQRA loop, budget, risk aversion).
    *   [x] Create `guardian_ai_tool/tests/bdd/steps/` directory.
    *   [x] Create `guardian_ai_tool/tests/bdd/steps/__init__.py`.
    *   [x] Create `guardian_ai_tool/tests/bdd/steps/test_eqra_steps.py` with initial step definitions.
    *   [ ] **PENDING (after `scheduler.py` is created):** Run BDD tests (`pytest`) and ensure scenarios pass. Refine step definitions as needed.

5.  **Unit Testing (`guardian_ai_tool/tests/unit/`)**
    *   [x] Ensure `guardian_ai_tool/tests/unit/__init__.py` exists.
    *   [x] Create `guardian_ai_tool/tests/unit/test_eqra_core.py`.
        *   [x] Test `ActionStats` initialization and single observation update.
        *   [ ] **PENDING (after `scheduler.py` is created & BDDs run):** Add more tests for `ActionStats` (multiple observations, edge cases).
        *   [ ] Test `ActionProvider` mock implementations (cost, run, stats update interaction).
        *   [ ] Test `EQRAScheduler` with mock actions (can be guided by BDD scenario needs):
            *   [ ] Initialization and `add_action`.
            *   [ ] `_calculate_eqra` correctness.
            *   [ ] `select_best_action` logic.
            *   [ ] `run_cycle` behavior.
            *   [ ] Behavior with different `risk_aversion_lambda` values.
            *   [ ] Behavior with empty action list or no eligible actions.

**Phase 2: Integration & Enhancements (Order may adjust based on BDD/Unit test findings)**

6.  **Persistence**
    *   [ ] Design and implement persistence for `ActionStats`.
    *   [ ] Modify `ActionProvider` and `EQRAScheduler` to load/save stats.

7.  **Real Metric Integration**
    *   [ ] Integrate real quality metric sensors (bE-TES/OSQI) for `delta_q_observed`.

8.  **Real Action Providers**
    *   [ ] Implement initial real action providers.

9.  **Cost Model**
    *   [ ] Implement an initial cost model for `C(a)` for real actions.

10. **Gamification**
    *   [ ] Integrate gamification XP rewards.

11. **CLI Integration**
    *   [ ] Enhance `guardian_ai_tool/guardian/cli.py`.

**Phase 3: Advanced Features (Future)**
*   [ ] Thompson Sampling based RL Scheduler.
*   [ ] MPC Budget Controller.
*   [ ] Quality-Debt Index integration.
*   [ ] Seasonal Gamification elements.

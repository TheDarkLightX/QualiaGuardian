# QualiaGuardian: Intelligent Code Quality & Autonomous Optimization

**QualiaGuardian** is an advanced, automated tool designed to empower developers and AI agents to achieve and maintain superior code quality, security, and design. It actively executes tests, provides holistic Test Effectiveness Scores (TES), performs comprehensive codebase analysis, and features an autonomous agent for quality optimization.

## Vision

To create a tool that helps produce high-quality, secure, and well-designed software by providing actionable insights, automated improvements, and intelligent metrics.

## Core Principles

1.  **TES-Driven:** The Test Effectiveness Score (bE-TES, OSQI) is a central metric.
2.  **Evidence-Based:** Utilizes empirically validated metrics and security standards.
3.  **Active Analysis:** Directly executes tests (including mutation testing) and gathers metrics.
4.  **Comprehensive Insight:** Covers code quality, test effectiveness, security, and design.
5.  **AI & Developer Centric:** Easy to use for both humans (CLI, future IDE/WebUI) and AI agents (CLI with structured output, direct API).
6.  **Actionable Feedback & Automation:** Provides clear recommendations and can autonomously apply improvements.
7.  **Gamified Experience:** Encourages engagement with code quality through a rewards system.

## Current Status

QualiaGuardian is under active development. Key milestones achieved include:
*   Core platform for code analysis and metric calculation.
*   Advanced Test Effectiveness Scoring (TES, bE-TES, OSQI).
*   Command-Line Interface (CLI) with gamification features.
*   Integration with `mutmut` for mutation testing.
*   SQLite database for persistent history, metrics, and gamification state.
*   Initial implementation of the Autonomous Quality-Optimizer Agent with LLM integration (OpenAI).

## Key Features

*   **Comprehensive Code Analysis:**
    *   Static analysis (complexity, maintainability via Radon).
    *   Security vulnerability scanning (via Safety).
    *   Mutation testing to assess test suite thoroughness (via `mutmut`).
*   **Advanced Test Effectiveness Scoring:**
    *   **Original Test Effectiveness Score (TES):** A foundational metric for test suite effectiveness.
    *   **Bounded Evolutionary Test-Effectiveness Score (bE-TES):** A more nuanced score defined as:
        *   **Bounded:** Every component is normalised to the 0 – 1 range, so the final score is always between 0 and 1.
        *   **Evolutionary:** This refers to the capability of QualiaGuardian (specifically through its `AdaptiveEMT` engine, currently under development as per Phase 2 of the Enhancement Plan) to perform **search-based optimization of the tests themselves**.
            *   **Oracle:** Standard mutation testing tools (like `mutmut`, which is integrated) act as an "oracle" by generating code mutants and reporting which tests kill them.
            *   **Search Engine:** The `AdaptiveEMT` layer is designed to function as a search engine (e.g., using Genetic Algorithms or Multi-Objective Evolutionary Algorithms). It aims to:
                1.  Represent tests in a way that allows for automated modification (their "genome").
                2.  Apply variation operators (e.g., mutating test assertions, crossover of test logic) to create new candidate tests.
                3.  Evaluate these candidate tests based on a multi-objective fitness function (e.g., maximizing killed mutants, minimizing flakiness and execution time, maximizing behavioral coverage).
                4.  Select the fittest tests to form the next generation, iteratively improving the test suite's effectiveness.
            *   This evolutionary process of refining the tests directly drives improvements in the bE-TES. While the full `AdaptiveEMT` search engine is part of ongoing development, the bE-TES framework is designed to measure the outcome of such evolutionary improvements.
        *   **Test-Effectiveness Score:** It quantifies how well the test suite actually detects real or mutant faults, weighting in stability, behaviour coverage, and speed.
        It incorporates mutation testing (used as the oracle) as follows:
        *   Mutation Score (MS) = (killed mutants) / (non-equivalent mutants)
        *   Normalized MS (M') = minmax(MS, 0.6, 0.95)
        *   bE-TES Uplift (Δ bE-TES) ≈ ((M'+ΔM')B'TS')^(1/5) - (M'B'TS')^(1/5)
            *   (Where B'TS' represents other bounded and normalized components of the score)
        For a more detailed visual explanation of Evolutionary Mutation Testing (EMT) and its benefits, see the [EMT Analysis Report](docs/emt_analysis_report.html).
    *   **Overall Software Quality Index (OSQI):** A holistic measure combining various quality aspects.
*   **Gamified Command-Line Interface:**
    *   Track progress with Experience Points (XP) and Levels.
    *   Earn Badges for achieving quality milestones.
    *   Complete Quests related to improving code quality.
    *   Commands: `guardian gamify status`, `guardian gamify crown` (Shapley-value based test valuation).
*   **Autonomous Quality-Optimizer Agent:**
    *   LLM-powered decision-making to select optimal improvement actions.
    *   Utilizes Expected Quality Return per Action (EQRA) for prioritization.
    *   Capable of proposing and (in future versions) applying patches.
    *   `dry_run` mode for testing agent logic without live LLM calls.
*   **Persistent History Tracking:**
    *   SQLite database stores run history, metrics, gamification progress, and agent actions.
*   **Extensible Action Marketplace (Planned):**
    *   Define custom improvement actions through YAML manifests.

## Getting Started

### Prerequisites

*   Python 3.8+
*   Git
*   (Optional, for Agent) An OpenAI API key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TheDarkLightX/QualiaGuardian.git
    cd QualiaGuardian/guardian_ai_tool
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -e .
    ```

4.  **(Optional) Set OpenAI API Key:**
    For the Autonomous Quality-Optimizer Agent to make real LLM calls, set the `OPENAI_API_KEY` environment variable:
    ```bash
    export OPENAI_API_KEY="your_openai_api_key_here"
    ```
    Without this, the agent can still run in `dry_run` mode.

### Basic Usage

QualiaGuardian provides a rich CLI.

*   **View available commands:**
    ```bash
    guardian --help
    ```

*   **Run legacy Guardian quality analysis (example):**
    This command runs a comprehensive analysis including static checks, mutation testing, and TES calculations.
    ```bash
    guardian quality /path/to/your/project --target-path /path/to/your/project/src --test-path /path/to/your/project/tests
    ```
    *Replace paths with your actual project paths.*

*   **Check your gamification status:**
    ```bash
    guardian gamify status
    ```

*   **Evaluate test importance with Shapley values:**
    This command requires a configured project and mutation testing results.
    ```bash
    guardian gamify crown /path/to/your/project --target-path /path/to/your/project/src --test-path /path/to/your/project/tests
    ```

*   **Run the Autonomous Quality-Optimizer Agent (in dry_run mode for testing):**
    The agent will simulate decision-making without actual LLM calls or file modifications.
    ```bash
    python guardian/agent/optimizer_agent.py
    ```
    To run with real LLM calls (ensure API key is set), you would modify `optimizer_agent.py` to instantiate `LLMWrapper` with `dry_run=False`.

## Project Structure

*   `guardian_ai_tool/guardian/`: Core library code.
    *   `agent/`: Autonomous agent components (LLM wrapper, optimizer).
    *   `analysis/`: Static and security analysis modules.
    *   `analytics/`: Metrics and scoring logic (TES, bE-TES, OSQI, Shapley).
    *   `cli/`: Command-Line Interface implementation using Typer.
    *   `core/`: Central logic for Guardian's operations.
    *   `evolution/`: Evolutionary computation components (SmartMutator, AdaptiveEMT).
    *   `sensors/`: Modules for gathering specific metrics (mutation, churn, etc.).
    *   `history.py`: Database interaction layer (`HistoryManager`).
*   `guardian_ai_tool/tests/`: Unit and integration tests.
*   `guardian_ai_tool/config/`: Default configuration files.
*   `guardian_ai_tool/docs/`: Detailed design documents and plans.

## Autonomous Quality-Optimizer Agent

The agent aims to autonomously improve code quality. It operates in a loop:
1.  Fetches current project metrics.
2.  Gets available improvement actions and their Expected Quality Return per Action (EQRA).
3.  Uses an LLM (via `LLMWrapper`) to pick the best action.
4.  (Future) Invokes an implementation agent to generate a patch.
5.  (Future) Applies and verifies the patch.
6.  Records the outcome and updates posteriors.

Key design documents:
*   [Autonomous Quality Optimizer Plan](docs/autonomous_quality_optimizer_plan.md)
*   [LLM Integration Plan](docs/llm_integration_plan.md) *(Assuming this file will be moved to docs/)*

## Gamification System

To make improving code quality more engaging, QualiaGuardian includes:
*   **Experience Points (XP):** Earned by running analyses, applying fixes, and completing quests.
*   **Levels:** Gained by accumulating XP.
*   **Badges:** Awarded for significant achievements (e.g., "Mutation Conqueror," "Flake Slayer").
*   **Quests:** Short-term goals to guide quality improvement efforts (e.g., "Improve mutation score by 5%").

## Roadmap & Future Work

*   Full implementation of patch application and verification by the agent.
*   Expansion of the Action Marketplace with more diverse improvement actions.
*   Development of an A* planner for long-term quality optimization strategies.
*   Enhanced UI/Web Interface for visualization and interaction.
*   Deeper integration with IDEs.

## Contributing

Contributions are welcome! Please follow these general steps:
1.  Fork the repository.
2.  Create a new branch for your feature or bug fix (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Write or update tests for your changes.
5.  Ensure all tests pass (`python -m pytest`).
6.  Commit your changes (`git commit -am 'Add some feature'`).
7.  Push to the branch (`git push origin feature/your-feature-name`).
8.  Create a new Pull Request.

Please ensure your code adheres to existing styling and quality standards.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (assuming a LICENSE file will be added).
Currently, the license is specified in `pyproject.toml`.
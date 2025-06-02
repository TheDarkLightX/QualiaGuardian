import random
from guardian.core.actions import ActionProvider, ActionStats

class MockFixedImprovementAction(ActionProvider):
    """
    A mock action that always provides a fixed improvement at a fixed cost.
    Useful for testing reliable, predictable actions.
    """
    def __init__(self, name: str = "FixedImprovement", description: str = "A mock action that always provides a fixed improvement.", 
                 cost: float = 1.0, improvement: float = 0.05, initial_stats: ActionStats = None):
        super().__init__(name, description, initial_stats=initial_stats)
        self._cost = cost
        self._improvement = improvement

    def estimate_cost(self) -> float:
        return self._cost

    def run(self, current_budget_for_action: float, project_context: any) -> dict:
        """Simulates running the action and returning its predefined effect."""
        print(f"Running {self.name} with budget {current_budget_for_action}...")
        actual_cost = self._cost
        actual_improvement = self._improvement
        patch_info = f"{self.name} applied successfully."
        return {
            "delta_q_observed": actual_improvement,
            "cost_incurred": actual_cost,
            "patch_info": patch_info
        }

class MockRiskyAction(ActionProvider):
    """
    A mock action with variable outcomes, simulating risk.
    It has a chance of success with positive improvement or a chance of a minor regression/no effect.
    """
    def __init__(self, name: str = "RiskyAction", description: str = "A mock action with a chance of success or minor regression.",
                 cost: float = 3.0, avg_improvement_on_success: float = 0.1, 
                 success_rate: float = 0.6, regression_on_failure_max: float = 0.03,
                 initial_stats: ActionStats = None):
        expected_outcome = (avg_improvement_on_success * success_rate) - \
                           (regression_on_failure_max * 0.5 * (1 - success_rate))
        default_stats = ActionStats(mu0=expected_outcome, sigma0_sq=0.0025)
        current_initial_stats = initial_stats if initial_stats is not None else default_stats
        super().__init__(name, description, initial_stats=current_initial_stats)
        self._cost = cost
        self._avg_improvement_on_success = avg_improvement_on_success
        self._success_rate = success_rate
        self._regression_on_failure_max = regression_on_failure_max

    def estimate_cost(self) -> float:
        return self._cost

    def run(self, current_budget_for_action: float, project_context: any) -> dict:
        """Simulates running the risky action."""
        print(f"Running {self.name} with budget {current_budget_for_action}...")
        cost_incurred = self._cost
        if random.random() < self._success_rate:
            delta_q_observed = random.uniform(
                self._avg_improvement_on_success * 0.7, 
                self._avg_improvement_on_success * 1.3
            )
            patch_info = f"{self.name} succeeded with improvement."
        else:
            delta_q_observed = random.uniform(-self._regression_on_failure_max, 0.0)
            patch_info = f"{self.name} had limited or negative effect."
        return {
            "delta_q_observed": delta_q_observed,
            "cost_incurred": cost_incurred,
            "patch_info": patch_info
        }

class NoOpAction(ActionProvider):
    """
    A mock action that represents doing nothing or performing a very minor analysis.
    It has a small cost and zero expected quality improvement.
    """
    def __init__(self, name: str = "NoOperation", description: str = "A mock action that does nothing but incurs a small analysis cost.",
                 cost: float = 0.1, initial_stats: ActionStats = None):
        default_stats = ActionStats(mu0=0.0, sigma0_sq=0.00001)
        current_initial_stats = initial_stats if initial_stats is not None else default_stats
        super().__init__(name, description, initial_stats=current_initial_stats)
        self._cost = cost

    def estimate_cost(self) -> float:
        return self._cost

    def run(self, current_budget_for_action: float, project_context: any) -> dict:
        """Simulates performing a no-operation or minimal analysis."""
        print(f"Running {self.name} with budget {current_budget_for_action}...")
        return {
            "delta_q_observed": 0.0,
            "cost_incurred": self._cost,
            "patch_info": f"{self.name} performed analysis, no changes made."
        }

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class ActionStats:
    """
    Stores the statistical information for an action provider,
    including priors and evolving posterior statistics.
    """
    n_a: int = 0  # Number of times this action has been run
    # delta_q_bar is initialized with mu0 via __post_init__
    m2: float = 0.0  # Sum of squares of differences from the mean for delta_q variance calculation

    # Priors
    mu0: float = 0.02  # Prior mean for expected delta Q
    sigma0_sq: float = 0.0004  # Prior variance for expected delta Q

    delta_q_bar: float = field(init=False)  # Posterior mean E[ΔQ|a]

    def __post_init__(self):
        """Initialize delta_q_bar with the prior mean mu0."""
        self.delta_q_bar = self.mu0

    def get_expected_delta_q(self) -> float:
        """Returns the current posterior mean expected change in quality (E[ΔQ|a])."""
        return self.delta_q_bar

    def get_variance_of_delta_q_mean(self) -> float:
        """
        Returns the variance of the posterior mean of delta Q (Var[E[ΔQ|a]]).
        This is var_hat / n_a as per the EQRA specification.
        """
        if self.n_a == 0:
            # If no runs, variance of the mean is the prior variance of the mean (which is sigma0_sq itself if mu0 is seen as a fixed point)
            # Or, if we interpret sigma0_sq as the variance of a single observation, then Var(mean) for n=0 is undefined or effectively infinite.
            # Given EQRA penalizes variance, a high value for n_a=0 is safer. Let's stick to sigma0_sq as per discussion.
            return self.sigma0_sq

        # var_hat is the sample variance of the observed delta_q's
        if self.n_a == 1:
            # As per EQRA spec: "var_hat = m2 / (n_a - 1) if n_a > 1 else sigma0**2"
            # So, if n_a = 1, var_hat is sigma0_sq.
            var_hat = self.sigma0_sq
        else:  # n_a > 1
            var_hat = self.m2 / (self.n_a - 1)
        
        # Variance of the mean E[ΔQ|a] is var_hat / n_a
        return var_hat / self.n_a

class ActionProvider(ABC):
    """
    Abstract base class for all action providers in Guardian.
    An action provider represents a specific operation Guardian can perform
    to potentially improve code quality.
    """
    def __init__(self, name: str, description: str, initial_stats: ActionStats = None):
        self.name = name
        self.description = description
        # Ensure each instance gets its own mutable ActionStats object if not provided
        self.stats = initial_stats if initial_stats is not None else ActionStats()

    def get_id(self) -> str:
        """Returns a unique identifier for the action, typically its name."""
        return self.name

    @abstractmethod
    def estimate_cost(self) -> float:
        """
        Estimates the cost C(a) of running this action.
        Cost can be in CPU-minutes or other composite budget units.
        """
        pass

    @abstractmethod
    def run(self, current_budget_for_action: float, project_context: any) -> dict:
        """
        Executes the action.

        Args:
            current_budget_for_action: The budget allocated by the scheduler for this specific run.
                                       The action should attempt to operate within this.
            project_context: Contextual information about the project (e.g., paths, configs).

        Returns:
            A dictionary containing at least:
            - 'delta_q_observed': The actual observed change in the quality metric (e.g., OSQI).
            - 'cost_incurred': The actual cost spent by the action.
            - 'patch_info': (Optional) Descriptive information about what the action did.
        """
        pass

    def update_stats(self, delta_q_observed: float):
        """
        Updates the action's statistics using Welford's online algorithm
        after an action run.

        Args:
            delta_q_observed: The actual change in quality metric observed from the run.
        """
        
        # Determine the mean before this observation was added
        if self.stats.n_a == 0:
            prev_mean = self.stats.mu0 # Before any observations, the 'previous mean' is the prior mean
        else:
            prev_mean = self.stats.delta_q_bar # This is the mean of the first n_a (now old n_a) observations

        self.stats.n_a += 1
        
        # Update delta_q_bar (the running mean)
        self.stats.delta_q_bar = prev_mean + (delta_q_observed - prev_mean) / self.stats.n_a
        
        # Update m2 (sum of squared differences from the *updated* mean)
        # Welford's M_k = M_{k-1} + (x_k - M_{k-1})(x_k - M_k)
        if self.stats.n_a == 1:
            # For the first point, m2 (S_k in Welford's) is 0 if starting from scratch.
            # If we consider the prior, the first update is special.
            # The formula Var[ΔQ|a] = var_hat/n_a and var_hat = sigma0_sq for n_a=1 handles this.
            # So, m2 should be such that var_hat / n_a for n_a=1 is sigma0_sq.
            # If var_hat = m2/(n_a-1) for n_a > 1, and var_hat = sigma0_sq for n_a=1, then m2 for n_a=1 is not directly used by that var_hat formula.
            # Let's keep m2 as the sum of squared diffs from the *current* mean. For n=1, m2 is 0.
            self.stats.m2 = 0.0
        else:
            self.stats.m2 += (delta_q_observed - prev_mean) * (delta_q_observed - self.stats.delta_q_bar)

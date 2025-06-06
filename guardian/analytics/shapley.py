import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Union  # Added Union

# Assuming metric_evaluator_stub is in a sibling module metric_stubs.py
# For actual use, this might be passed in or imported differently.
from .metric_stubs import metric_evaluator_stub

logger = logging.getLogger(__name__)

# Type alias for a test identifier, can be Path or str
TestId = Union[Path, str]  # Path for file/node, str for just name


def calculate_shapley_values(
    test_ids: List[TestId],
    metric_evaluator_func: Callable[[List[TestId]], float],
    num_permutations: int = 200,
    use_progress_bar: bool = False,  # Placeholder for rich progress
) -> Dict[TestId, float]:
    """
    Calculates approximate Shapley values for a list of tests (or features).

    Shapley values quantify the marginal contribution of each test to an
    overall metric (e.g., bE-TES score) calculated by metric_evaluator_func.
    This implementation uses Monte Carlo sampling of permutations for approximation.

    Args:
        test_ids: A list of unique identifiers for the tests.
                  These identifiers must be usable by metric_evaluator_func.
        metric_evaluator_func: A function that takes a list of test_ids (a subset)
                               and returns a single float score for that subset.
        num_permutations: The number of random permutations to sample for the
                          Monte Carlo approximation. Defaults to 200.
        use_progress_bar: If True, attempts to show a progress bar (not implemented in this basic version).

    Returns:
        A dictionary mapping each test_id to its approximate Shapley value.
    """
    n = len(test_ids)
    if n == 0:
        return {}

    shapley_values: Dict[TestId, float] = defaultdict(float)

    # Initial score of empty set (F(emptyset))
    # Some metric evaluators might require a non-empty list.
    # The stub handles empty list and returns 0.
    score_empty_set = metric_evaluator_func([])

    # TODO: Integrate rich.progress.Progress if use_progress_bar is True
    # from rich.progress import Progress
    # with Progress() as progress:
    #   task = progress.add_task("[cyan]Calculating Shapley values...", total=num_permutations)
    #   for _ in range(num_permutations):
    #       progress.update(task, advance=1)
    #       ... (rest of the loop) ...

    for i in range(num_permutations):
        if (
            use_progress_bar and i % (num_permutations // 20) == 0
        ):  # Basic progress print
            logger.debug(f"Shapley permutation {i+1}/{num_permutations}")

        shuffled_test_ids = random.sample(
            test_ids, n
        )  # More robust than shuffle in-place if test_ids is used elsewhere

        current_subset: List[TestId] = []
        # Score of the current subset *before* adding the next test from permutation
        score_of_current_subset = score_empty_set

        for test_id in shuffled_test_ids:
            # Score of the subset *with* the current test_id added
            score_with_test = metric_evaluator_func(current_subset + [test_id])

            marginal_contribution = score_with_test - score_of_current_subset
            shapley_values[test_id] += marginal_contribution

            # Update for the next iteration: the current test becomes part of the subset
            current_subset.append(test_id)
            score_of_current_subset = score_with_test

    # Average the contributions over all permutations
    for test_id in test_ids:
        shapley_values[test_id] /= num_permutations

    logger.info(
        f"Calculated Shapley values for {n} tests using {num_permutations} permutations."
    )
    return dict(shapley_values)  # Convert defaultdict to dict for cleaner output


if __name__ == "__main__":
    # Example usage:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Use the stub from metric_stubs.py
    # The keys in TEST_CACHE are Path objects.
    # Let's define our test_identifiers as those Path objects.
    from .metric_stubs import TEST_CACHE

    # Convert TEST_CACHE keys to a list of TestId (which are Path objects here)
    test_identifiers: List[TestId] = list(TEST_CACHE.keys())

    # Ensure we only use a subset for quick testing if TEST_CACHE is large
    test_identifiers_subset = (
        test_identifiers[:5] if len(test_identifiers) > 5 else test_identifiers
    )
    if not test_identifiers_subset:
        logger.warning(
            "TEST_CACHE in metric_stubs.py is empty or test_identifiers_subset is empty. Cannot run example."
        )
    else:
        logger.info(
            f"Running Shapley calculation for {len(test_identifiers_subset)} example tests..."
        )

        # The metric_evaluator_stub expects List[Path]
        # Ensure our test_identifiers_subset matches this type if TestId is Union[Path, str]
        # In this case, they are already Path objects.

        calculated_values = calculate_shapley_values(
            test_ids=test_identifiers_subset,
            metric_evaluator_func=metric_evaluator_stub,  # Using the imported stub
            num_permutations=500,  # Increase for more stable example, but 200 is typical
        )

        logger.info("Approximate Shapley Values:")
        # Sort by value for readability
        for test_id, value in sorted(
            calculated_values.items(), key=lambda item: item[1], reverse=True
        ):
            logger.info(f"  {str(test_id):<60}: {value:.4f}")

        # Verify sum of Shapley values (should be close to F(N) - F(emptyset))
        # F(N) is score of all tests in the subset
        # F(emptyset) is score of empty set (0 for our stub)
        score_of_full_subset = metric_evaluator_stub(test_identifiers_subset)
        sum_of_shapley_values = sum(calculated_values.values())

        logger.info(f"\nSum of Shapley values: {sum_of_shapley_values:.4f}")
        logger.info(f"Score of the full subset (F(N)): {score_of_full_subset:.4f}")
        logger.info(
            f"Score of empty set (F(emptyset)): {metric_evaluator_stub([]):.4f}"
        )

        # The sum of Shapley values should equal F(N) - F(emptyset)
        # For our stub, F(emptyset) = 0, so sum should be close to F(N)
        if (
            abs(sum_of_shapley_values - score_of_full_subset) < 1e-3
        ):  # Allow small tolerance for float arithmetic
            logger.info(
                "Efficiency property verified (Sum of Shapley values ≈ F(N) - F(Ø))."
            )
        else:
            logger.warning(
                "Efficiency property NOT verified. Sum of Shapley values differs significantly from F(N) - F(Ø)."
            )
            logger.warning(
                f"Difference: {abs(sum_of_shapley_values - score_of_full_subset):.4f}"
            )

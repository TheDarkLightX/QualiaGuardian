Feature: Fast Shapley Value Computation with Convergence Detection
  As a developer analyzing test importance
  I want Shapley values to converge automatically with variance reduction
  So that I don't waste compute on stable values

  Background:
    Given a test suite with multiple tests
    And a metric evaluator that can score test subsets

  Scenario: Early convergence detection saves computation time
    Given a test suite with 50 tests
    And a convergence threshold of 0.01
    When I compute Shapley values with convergence detection
    Then results should be available in less than 30 seconds
    And accuracy should be greater than 95% vs full computation
    And convergence should be detected automatically
    And computation should stop early when values stabilize

  Scenario: Antithetic variance reduction improves efficiency
    Given the same random seed for reproducibility
    And a test suite with 25 tests
    When I use antithetic variates vs standard sampling
    Then variance should reduce by more than 50%
    And convergence should be 2x faster
    And standard error should be measurably smaller
    And fewer permutations should be needed for same accuracy

  Scenario: Progressive approximation quality with confidence intervals
    Given a test suite with 30 tests
    And iteration counts of [50, 100, 200, 500]
    When monitoring approximation quality over iterations
    Then accuracy should improve monotonically
    And confidence intervals should narrow appropriately
    And 95% confidence interval should contain true values
    And computational cost should scale linearly with iterations

  Scenario: Handles edge cases gracefully
    Given a test suite with only 1 test
    When computing Shapley values
    Then the single test should have value equal to metric difference
    And no convergence detection should be needed
    And computation should complete immediately

  Scenario: Memory efficiency for large test suites
    Given a test suite with 200 tests
    When computing Shapley values with convergence detection
    Then memory usage should remain below 100MB
    And intermediate results should be garbage collected
    And no memory leaks should occur during long computations
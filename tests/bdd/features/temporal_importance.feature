Feature: Temporal Importance Tracking
  As a quality engineer
  I want to track how test importance changes over time
  So that I can make informed decisions about test maintenance and optimization

  Background:
    Given I have a temporal importance tracker initialized
    And I have a test suite with historical importance data

  Scenario: Recording test importance over time
    Given I have a test named "test_critical_feature"
    When I record importance values:
      | timestamp        | importance |
      | 2024-01-01 10:00 | 0.95      |
      | 2024-01-02 10:00 | 0.93      |
      | 2024-01-03 10:00 | 0.91      |
      | 2024-01-04 10:00 | 0.89      |
    Then the importance history should be persisted
    And I should be able to retrieve the full time series

  Scenario: Decomposing time series into components
    Given I have a test "test_seasonal_behavior" with 30 days of data
    And the data shows weekly seasonality
    When I decompose the time series
    Then I should get trend component
    And I should get seasonal component with period 7
    And I should get residual component
    And the components should sum to the original series

  Scenario: Detecting change points in importance
    Given I have a test "test_refactored_module" with importance history:
      | day | importance |
      | 1-7 | 0.8-0.85  |
      | 8   | 0.6       |
      | 9-15| 0.58-0.62 |
    When I run change point detection
    Then I should detect a change point at day 8
    And the change magnitude should be approximately 0.2
    And the direction should be "decrease"

  Scenario: Modeling exponential decay
    Given I have a test "test_legacy_feature" showing decay pattern
    When I model the decay as exponential
    Then I should get decay parameters (a, b, c)
    And the R-squared should be greater than 0.8
    And I should get a half-life estimate
    And the current decay rate should be calculated

  Scenario: Modeling linear decay
    Given I have a test "test_deprecated_api" with linear decline
    When I model the decay as linear
    Then I should get linear parameters (slope, intercept)
    And the model should predict when importance reaches zero
    And the fit quality should be assessed

  Scenario: Forecasting future importance
    Given I have a test "test_core_functionality" with 30 days of history
    When I forecast importance for the next 7 days
    Then I should get predicted values for each day
    And I should get confidence intervals
    And the forecast should respect bounds [0, 1]
    And trend should be incorporated into forecast

  Scenario: Detecting weekly patterns
    Given I have a test "test_weekend_job" with data showing:
      | day_of_week | avg_importance |
      | Monday      | 0.9           |
      | Tuesday     | 0.85          |
      | Wednesday   | 0.8           |
      | Thursday    | 0.75          |
      | Friday      | 0.7           |
      | Saturday    | 0.3           |
      | Sunday      | 0.3           |
    When I analyze temporal patterns
    Then I should detect a weekly pattern
    And the pattern strength should be greater than 0.6
    And the pattern period should be 7 days

  Scenario: Generating sudden change alerts
    Given I have a test "test_broken_integration" with stable importance 0.8
    When the importance suddenly drops to 0.2
    Then an alert should be generated
    And the alert type should be "sudden_decrease"
    And the severity should be high (>0.8)
    And the recommendation should suggest investigating recent changes

  Scenario: Alerting on importance decay threshold
    Given I have a test "test_obsolete_feature"
    When the importance decays below 0.1
    Then a decay threshold alert should be generated
    And the recommendation should suggest test removal or update

  Scenario: Handling anomalies in time series
    Given I have a test "test_flaky_component" with importance:
      | day | importance |
      | 1-6 | 0.7-0.75  |
      | 7   | 0.2       |
      | 8   | 0.72      |
    When I analyze the time series
    Then day 7 should be flagged as an anomaly
    And the anomaly should not affect trend calculation
    And an anomaly alert should be generated

  Scenario: Visualizing temporal importance
    Given I have a test "test_evolving_feature" with 60 days of data
    When I request visualization data
    Then I should get time series data points
    And I should get decomposition components if available
    And I should get marked change points
    And I should get forecast with confidence bands
    And I should get summary statistics

  Scenario: Comparing decay models
    Given I have a test "test_gradual_obsolescence" with decay pattern
    When I fit multiple decay models:
      | model        |
      | exponential  |
      | linear       |
      | logarithmic  |
      | power        |
    Then I should get goodness of fit for each model
    And the best fitting model should be identified
    And model parameters should be interpretable

  Scenario: Step function decay detection
    Given I have a test "test_feature_toggle" with importance:
      | days  | importance |
      | 1-30  | 0.9       |
      | 31-60 | 0.1       |
    When I model decay as step function
    Then the step point should be detected around day 30
    And before/after values should be 0.9 and 0.1

  Scenario: Forecasting with Holt-Winters
    Given I have a test with trend and seasonality
    And I have at least 28 days of data
    When I forecast using Holt-Winters method
    Then the forecast should capture trend
    And the forecast should capture seasonality
    And confidence intervals should widen over time

  Scenario: Retrieving historical alerts
    Given I have multiple tests with various alerts
    When I query alerts for "test_problematic"
    Then I should get alerts in reverse chronological order
    And I should be able to filter by acknowledged status
    And each alert should have severity and recommendations

  Scenario: Detecting daily patterns
    Given I have a test "test_business_hours" with hourly data
    And importance peaks during 9-5 business hours
    When I analyze temporal patterns
    Then I should detect a daily pattern
    And the pattern period should be 1 day
    And the phase should indicate peak hours

  Scenario: Handling sparse data
    Given I have a test "test_rare_scenario" with irregular recordings:
      | days_between_recordings |
      | 1, 3, 7, 2, 14, 5     |
    When I analyze the sparse time series
    Then interpolation should be applied appropriately
    And pattern detection should handle gaps
    And forecasting should account for irregularity

  Scenario: Real-time importance updates
    Given I have a test with existing history
    When I record a new importance value
    Then the cache should be invalidated
    And alerts should be checked immediately
    And patterns should be updated if significant

  Scenario: Persistence and recovery
    Given I have recorded importance data for multiple tests
    When I restart the tracker
    Then all historical data should be available
    And detected patterns should be preserved
    And unacknowledged alerts should remain active
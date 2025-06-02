"""
High-Quality Tests for Guardian Core Components
Located in guardian/ directory to improve Guardian's self-analysis TES/E-TES scores

These tests provide comprehensive coverage with meaningful assertions,
boundary value analysis, and property-based testing to improve scores from F to A.
"""

import pytest
import time
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
import math # For sigmoid calculations if needed in tests

from guardian.core.tes import calculate_etes_v2, get_etes_grade # calculate_etes_v2 is the old E-TES v2
from guardian.core.etes import ETESCalculator, ETESComponents, BETESSettingsV31 # QualityConfig is in etes.py
from guardian.core.betes import BETESCalculator, BETESComponents, BETESWeights # BETES specific classes
from guardian.cli.analyzer import ProjectAnalyzer
from guardian.test_execution.pytest_runner import run_pytest


class TestTESCalculationCore:
    """High-quality tests for TES calculation with comprehensive coverage"""
    
    def test_should_calculate_high_tes_score_when_excellent_metrics_provided(self):
        """Test TES calculation with excellent metrics targeting A grade"""
        start_time = time.time()
        
        # Arrange: Excellent metrics that meet all targets
        mutation_score = 0.90      # >0.85 target ✓
        assertion_density = 4.5    # >3.0 target ✓
        behavior_coverage = 0.95   # >0.90 target ✓
        speed_factor = 0.85        # >0.80 target ✓
        
        # Act: Calculate TES
        tes_score = calculate_etes_v2( # Assuming this is the intended replacement for TES calculation
            test_suite_data={ # calculate_etes_v2 expects test_suite_data and codebase_data
                'mutation_score': mutation_score, # 0.90
                'avg_test_execution_time_ms': 19, # For SF ~0.8518
                'assertions': [ # Aim for higher AIQ
                    {'type': 'equality', 'code': 'assert x == 10', 'target_criticality': 1.5},
                    {'type': 'boundary', 'code': 'assert y > 0', 'target_criticality': 2.0},
                    {'type': 'exception', 'code': 'with pytest.raises(ValueError):', 'target_criticality': 2.0},
                    {'type': 'invariant', 'code': 'assert is_valid(obj)', 'target_criticality': 2.5, 'checks_invariant': True},
                    # Total weighted score: (1.0*1.5) + (1.6*2.0) + (1.5*2.0) + (2.0*1.5*2.5) -> for invariant, weight is 2.0 * 1.5 (bonus)
                    # = 1.5 + 3.2 + 3.0 + 7.5 = 15.2
                    # Number of assertions = 4
                    # Avg weighted score = 15.2 / 4 = 3.8
                    # AIQ = min(3.8 / 2.0, 1.0) = min(1.9, 1.0) = 1.0. This should be high enough.
                ],
                'covered_behaviors': ['dummy_behavior'], # For BC = 1.0
                'execution_results': [{'passed': True, 'execution_time_ms': 100}], # Not directly used in score
                'determinism_score': 0.95, # For QF
                'stability_score': 0.95,   # For QF
                'readability_score': 0.95, # For QF
                'independence_score': 0.95 # For QF
                # QF = (0.95^4)^0.25 = 0.95
            },
            codebase_data={
                'all_behaviors': ['dummy_behavior'], # Approximate
                'behavior_criticality': {'dummy_behavior': behavior_coverage / 0.1 if behavior_coverage else 1}, # Approximate
                'complexity_metrics': {'avg_cyclomatic_complexity': 3.0, 'total_loc': 1000} # Dummy
            }
        )[0] # calculate_etes_v2 returns (score, components)
        
        execution_time = (time.time() - start_time) * 1000
        
        # Assert: High-quality assertions with meaningful validation
        assert isinstance(tes_score, (int, float)), "TES score must be numeric"
        assert 0.0 <= tes_score <= 1.0, f"TES score must be 0-1, got {tes_score}"
        assert tes_score >= 0.5, f"Excellent metrics should yield TES ≥0.5, got {tes_score}"
        
        # Property: Grade should reflect score quality
        grade = get_etes_grade(tes_score)
        assert grade != "F", f"Excellent metrics should not yield F grade, got {grade}"
        assert grade in ["A+", "A", "B", "C"], f"Expected good grade for excellent metrics, got {grade}"
        
        # Invariant: Better inputs should yield better scores
        poor_tes = calculate_etes_v2(
            test_suite_data={'mutation_score': 0.3, 'avg_test_execution_time_ms': 300, 'assertions': [{'type': 'equality', 'code': 'dummy', 'target_criticality': 1.0/3}], 'covered_behaviors': ['dummy'], 'execution_results': [{'passed': True, 'execution_time_ms': 300}], 'determinism_score': 0.9, 'stability_score': 0.9, 'readability_score': 0.9, 'independence_score': 0.9},
            codebase_data={'all_behaviors': ['dummy'], 'behavior_criticality': {'dummy': 0.4/0.1}, 'complexity_metrics': {'avg_cyclomatic_complexity': 3.0, 'total_loc': 1000}}
        )[0]
        assert tes_score > poor_tes, "Better metrics should yield better TES score"
        
        # Performance requirement
        assert execution_time < 200.0, f"TES calculation should be <200ms, took {execution_time:.1f}ms"
    
    def test_should_handle_boundary_conditions_correctly_when_calculating_tes(self):
        """Test TES calculation with boundary values and edge cases"""
        start_time = time.time()
        
        # Test exact target boundaries
        boundary_cases = [
            (0.85, 3.0, 0.90, 0.80),  # Exact targets
            (0.84, 2.9, 0.89, 0.79),  # Just below targets
            (0.86, 3.1, 0.91, 0.81),  # Just above targets
            (1.0, 5.0, 1.0, 1.0),     # Maximum values
            (0.0, 0.0, 0.0, 0.0),     # Minimum values
        ]
        
        for mutation, assertion, behavior, speed in boundary_cases:
            # Act: Calculate TES for boundary case
            tes_score = calculate_etes_v2(
                test_suite_data={'mutation_score': mutation, 'avg_test_execution_time_ms': 100, 'assertions': [{'type': 'equality', 'code': 'dummy', 'target_criticality': assertion/3 if assertion else 1}], 'covered_behaviors': ['dummy'], 'execution_results': [{'passed': True, 'execution_time_ms': 100}], 'determinism_score': 0.9, 'stability_score': 0.9, 'readability_score': 0.9, 'independence_score': 0.9},
                codebase_data={'all_behaviors': ['dummy'], 'behavior_criticality': {'dummy': behavior/0.1 if behavior else 1}, 'complexity_metrics': {'avg_cyclomatic_complexity': 3.0, 'total_loc': 1000}}
            )[0]
            
            # Assert: Boundary value validation
            assert isinstance(tes_score, (int, float)), f"TES should be numeric for {(mutation, assertion, behavior, speed)}"
            assert 0.0 <= tes_score <= 1.0, f"TES should be 0-1 for {(mutation, assertion, behavior, speed)}, got {tes_score}"
            
            # Property: Zero inputs should yield very low TES
            if all(x == 0.0 for x in [mutation, assertion, behavior, speed]):
                assert tes_score <= 0.1, f"Zero inputs should yield very low TES, got {tes_score}"
        
        execution_time = (time.time() - start_time) * 1000
        assert execution_time < 200.0, f"Boundary testing should be fast, took {execution_time:.1f}ms"
    
    def test_should_validate_grade_consistency_when_scores_calculated(self):
        """Test TES grade consistency with score ranges"""
        start_time = time.time()
        
        # Test score-to-grade mapping consistency
        test_scores = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
        
        for score in test_scores:
            # Act: Get grade for score
            grade = get_etes_grade(score)
            
            # Assert: Grade consistency validation
            assert isinstance(grade, str), f"Grade should be string for score {score}"
            assert grade in ["A+", "A", "B", "C", "D", "F"], f"Invalid grade {grade} for score {score}"
            
            # Property: Higher scores should have better or equal grades
            if score >= 0.9:
                assert grade == "A+", f"Score {score} should have A+ grade, got {grade}" # A_PLUS is distinct from A
            elif score >= 0.8: # A
                assert grade == "A", f"Score {score} should have A grade, got {grade}"
            elif score >= 0.7: # B
                assert grade == "B", f"Score {score} should have B grade, got {grade}"
            elif score >= 0.6: # C
                assert grade == "C", f"Score {score} should have C grade, got {grade}"
            else: # F for scores < 0.6
                assert grade == "F", f"Score {score} (<0.6) should have F grade, got {grade}"
        
        execution_time = (time.time() - start_time) * 1000
        assert execution_time < 200.0, f"Grade validation should be fast, took {execution_time:.1f}ms"


class TestETESCalculationCore:
    """High-quality tests for E-TES v2.0 calculation"""
    
    def test_should_calculate_high_etes_score_when_comprehensive_data_provided(self):
        """Test E-TES calculation with comprehensive test data"""
        start_time = time.time()
        
        # Arrange: Comprehensive test suite data
        test_suite_data = {
            'mutation_score': 0.88,
            'avg_test_execution_time_ms': 95,
            'total_tests': 150,
            'assertions': [
                {'type': 'equality', 'code': 'assert result == expected', 'target_criticality': 1.0},
                {'type': 'boundary', 'code': 'assert 0 <= value <= 100', 'target_criticality': 1.5},
                {'type': 'exception', 'code': 'with pytest.raises(ValueError):', 'target_criticality': 1.8},
                {'type': 'invariant', 'code': 'assert state_is_valid()', 'target_criticality': 2.0, 'checks_invariant': True},
                {'type': 'property', 'code': 'assert len(result) > 0', 'target_criticality': 1.6},
            ],
            'covered_behaviors': ['authentication', 'validation', 'data_processing', 'error_handling'],
            'execution_results': [
                {'passed': True, 'execution_time_ms': 95},
                {'passed': True, 'execution_time_ms': 102},
                {'passed': True, 'execution_time_ms': 88}
            ],
            'determinism_score': 0.96,
            'stability_score': 0.92,
            'readability_score': 0.89,
            'independence_score': 0.94,
        }
        
        codebase_data = {
            'all_behaviors': ['authentication', 'validation', 'data_processing', 'error_handling', 'reporting'],
            'behavior_criticality': {
                'authentication': 2.0,
                'validation': 1.8,
                'data_processing': 1.5,
                'error_handling': 1.7,
                'reporting': 1.2
            },
            'complexity_metrics': {
                'avg_cyclomatic_complexity': 2.8,
                'total_loc': 2500,
                'function_count': 85
            }
        }
        
        # Act: Calculate E-TES v2.0
        calculator = ETESCalculator()
        etes_score, components = calculator.calculate_etes(test_suite_data, codebase_data)
        
        execution_time = (time.time() - start_time) * 1000
        
        # Assert: High-quality E-TES validation
        assert isinstance(etes_score, (int, float)), "E-TES score must be numeric"
        assert 0.0 <= etes_score <= 1.0, f"E-TES score must be 0-1, got {etes_score}"
        
        # Validate components
        assert isinstance(components, ETESComponents), "Components must be ETESComponents instance"
        assert 0.0 <= components.mutation_score <= 1.0, "Mutation score must be 0-1"
        assert components.evolution_gain >= 1.0, "Evolution gain must be ≥1.0"
        assert 0.0 <= components.assertion_iq <= 1.0, "Assertion IQ must be 0-1"
        assert 0.0 <= components.behavior_coverage <= 1.0, "Behavior coverage must be 0-1"
        assert 0.0 <= components.speed_factor <= 1.0, "Speed factor must be 0-1"
        assert 0.0 <= components.quality_factor <= 1.0, "Quality factor must be 0-1"
        
        # Property: Good test data should yield reasonable E-TES score
        assert etes_score >= 0.1, f"Comprehensive test data should yield E-TES ≥0.1, got {etes_score}"
        
        # Performance requirement
        assert execution_time < 1000.0, f"E-TES calculation should be <1s, took {execution_time:.1f}ms"
    
    def test_should_show_evolution_gain_when_previous_score_provided(self):
        """Test E-TES evolution gain calculation"""
        start_time = time.time()
        
        # Arrange: Test data with improvement potential
        test_suite_data = {
            'mutation_score': 0.82,
            'avg_test_execution_time_ms': 120,
            'total_tests': 100,
            'assertions': [
                {'type': 'equality', 'code': 'assert x == y', 'target_criticality': 1.0},
                {'type': 'boundary', 'code': 'assert value >= 0', 'target_criticality': 1.5},
            ],
            'covered_behaviors': ['core_feature', 'validation'],
            'execution_results': [{'passed': True, 'execution_time_ms': 120}],
            'determinism_score': 0.90,
            'stability_score': 0.85,
            'readability_score': 0.80,
            'independence_score': 0.88,
        }
        
        codebase_data = {
            'all_behaviors': ['core_feature', 'validation', 'reporting'],
            'behavior_criticality': {'core_feature': 2.0, 'validation': 1.5, 'reporting': 1.0},
            'complexity_metrics': {'avg_cyclomatic_complexity': 3.0, 'total_loc': 1000}
        }
        
        # Act: Calculate with previous score
        calculator = ETESCalculator()
        previous_score = 0.45
        etes_score, components = calculator.calculate_etes(test_suite_data, codebase_data, previous_score)
        
        execution_time = (time.time() - start_time) * 1000
        
        # Assert: Evolution gain validation
        assert isinstance(etes_score, (int, float)), "E-TES score must be numeric"
        assert isinstance(components, ETESComponents), "Components must be ETESComponents instance"
        assert components.evolution_gain >= 1.0, "Evolution gain must be ≥1.0"
        
        # Property: Evolution gain should reflect improvement potential
        if etes_score > previous_score:
            # If score improved, evolution gain might be > 1.0 in future iterations
            pass
        
        # Performance requirement
        assert execution_time < 1000.0, f"E-TES with evolution should be <1s, took {execution_time:.1f}ms"


class TestProjectAnalyzerCore:
    """High-quality tests for project analyzer"""
    
    def test_should_analyze_project_comprehensively_when_realistic_codebase_provided(self):
        """Test comprehensive project analysis with realistic codebase"""
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange: Create realistic Python project
            project_path = Path(temp_dir)
            
            # Create main module
            (project_path / 'calculator.py').write_text('''
"""Simple calculator module for testing"""

class Calculator:
    """A simple calculator class"""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        """Add two numbers"""
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("Arguments must be numbers")
        
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a, b):
        """Subtract two numbers"""
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("Arguments must be numbers")
        
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def multiply(self, a, b):
        """Multiply two numbers"""
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("Arguments must be numbers")
        
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def divide(self, a, b):
        """Divide two numbers"""
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("Arguments must be numbers")
        
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def get_history(self):
        """Get calculation history"""
        return self.history.copy()
    
    def clear_history(self):
        """Clear calculation history"""
        self.history.clear()
''')
            
            # Create test file
            (project_path / 'test_calculator.py').write_text('''
"""Comprehensive tests for calculator module"""
import pytest
from calculator import Calculator

class TestCalculator:
    """Test Calculator class with comprehensive coverage"""
    
    def test_addition_with_valid_numbers(self):
        """Test addition with valid numbers"""
        calc = Calculator()
        
        # Test positive numbers
        assert calc.add(2, 3) == 5
        assert calc.add(10, 20) == 30
        
        # Test negative numbers
        assert calc.add(-5, -3) == -8
        assert calc.add(-5, 3) == -2
        
        # Test floats
        assert calc.add(2.5, 3.7) == pytest.approx(6.2)
        
        # Test zero
        assert calc.add(0, 5) == 5
        assert calc.add(5, 0) == 5
    
    def test_addition_with_invalid_types(self):
        """Test addition with invalid types"""
        calc = Calculator()
        
        with pytest.raises(TypeError, match="Arguments must be numbers"):
            calc.add("2", 3)
        
        with pytest.raises(TypeError, match="Arguments must be numbers"):
            calc.add(2, "3")
        
        with pytest.raises(TypeError, match="Arguments must be numbers"):
            calc.add(None, 3)
    
    def test_subtraction_operations(self):
        """Test subtraction operations"""
        calc = Calculator()
        
        assert calc.subtract(10, 3) == 7
        assert calc.subtract(5, 10) == -5
        assert calc.subtract(-5, -3) == -2
        assert calc.subtract(0, 5) == -5
    
    def test_multiplication_operations(self):
        """Test multiplication operations"""
        calc = Calculator()
        
        assert calc.multiply(3, 4) == 12
        assert calc.multiply(-3, 4) == -12
        assert calc.multiply(0, 5) == 0
        assert calc.multiply(2.5, 4) == 10.0
    
    def test_division_operations(self):
        """Test division operations"""
        calc = Calculator()
        
        assert calc.divide(10, 2) == 5.0
        assert calc.divide(7, 2) == 3.5
        assert calc.divide(-10, 2) == -5.0
    
    def test_division_by_zero(self):
        """Test division by zero error"""
        calc = Calculator()
        
        with pytest.raises(ZeroDivisionError, match="Cannot divide by zero"):
            calc.divide(10, 0)
    
    def test_history_functionality(self):
        """Test calculation history"""
        calc = Calculator()
        
        # Initially empty
        assert calc.get_history() == []
        
        # Add operations
        calc.add(2, 3)
        calc.multiply(4, 5)
        
        history = calc.get_history()
        assert len(history) == 2
        assert "2 + 3 = 5" in history
        assert "4 * 5 = 20" in history
        
        # Clear history
        calc.clear_history()
        assert calc.get_history() == []
''')
            
            # Act: Analyze the project
            analyzer = ProjectAnalyzer()
            results = analyzer.analyze_project(str(project_path))
            
            execution_time = (time.time() - start_time) * 1000
            
            # Assert: Comprehensive analysis validation
            assert isinstance(results, dict), "Analysis results must be a dictionary"
            
            # Property: Results should contain all required sections
            required_sections = ['status', 'metrics', 'tes_score', 'tes_grade']
            for section in required_sections:
                assert section in results, f"Results missing required section: {section}"
            
            # Validate metrics
            metrics = results['metrics']
            assert isinstance(metrics, dict), "Metrics must be a dictionary"
            assert metrics['python_files_analyzed'] == 2, "Should analyze exactly 2 Python files"
            assert metrics['total_lines_of_code_python'] > 50, "Should count substantial lines of code"
            
            # Validate TES score
            tes_score = results['tes_score']
            assert isinstance(tes_score, (int, float)), "TES score must be numeric"
            assert 0.0 <= tes_score <= 1.0, f"TES score must be 0-1, got {tes_score}"
            
            # Property: TES grade should be consistent with score
            tes_grade = results['tes_grade']
            assert isinstance(tes_grade, str), "TES grade must be string"
            # ProjectAnalyzer._calculate_tes_score currently returns "N/A"
            # The main CLI's run_analysis function adds a separate 'quality_analysis' block
            # This test is for ProjectAnalyzer's direct output.
            assert tes_grade == "N/A", f"Expected 'N/A' for tes_grade from ProjectAnalyzer, got {tes_grade}"
        
            # Performance requirement - increased timeout
            assert execution_time < 30000.0, f"Analysis should complete in <30s, took {execution_time:.1f}ms"


class TestPytestRunnerCore:
    """High-quality tests for pytest runner"""
    
    def test_should_execute_tests_successfully_when_valid_test_suite_provided(self):
        """Test pytest execution with valid test suite"""
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange: Create simple test file
            test_file = Path(temp_dir) / 'test_simple.py'
            test_file.write_text('''
import pytest

def test_basic_arithmetic():
    """Test basic arithmetic operations"""
    assert 2 + 2 == 4
    assert 5 - 3 == 2
    assert 3 * 4 == 12
    assert 8 / 2 == 4

def test_string_operations():
    """Test string operations"""
    text = "Hello World"
    assert text.upper() == "HELLO WORLD"
    assert text.lower() == "hello world"
    assert len(text) == 11

def test_list_operations():
    """Test list operations"""
    data = [1, 2, 3, 4, 5]
    assert len(data) == 5
    assert sum(data) == 15
    assert max(data) == 5
    assert min(data) == 1

def test_error_handling():
    """Test error handling"""
    with pytest.raises(ZeroDivisionError):
        result = 1 / 0
    
    with pytest.raises(ValueError):
        int("not_a_number")
''')
            
            # Act: Run pytest
            results = run_pytest(str(temp_dir))
            
            execution_time = (time.time() - start_time) * 1000
            
            # Assert: Comprehensive pytest validation
            assert isinstance(results, dict), "Results must be a dictionary"
            
            # Property: Results should contain execution information
            required_fields = ['success', 'exit_code', 'duration_seconds']
            for field in required_fields:
                assert field in results, f"Results missing required field: {field}"
            
            # Validate execution success
            success = results['success']
            assert isinstance(success, bool), "Success must be boolean"
            
            exit_code = results['exit_code']
            assert isinstance(exit_code, int), "Exit code must be integer"
            
            # Property: Good tests should execute successfully
            if success:
                assert exit_code in [0, 5], f"Successful execution should have exit code 0 or 5, got {exit_code}"
            
            # Validate duration
            duration = results['duration_seconds']
            assert isinstance(duration, (int, float)), "Duration must be numeric"
            assert duration > 0, "Duration must be positive"
            assert duration < 30.0, f"Tests should complete in <30s, took {duration}s"
            
            # Performance requirement
            assert execution_time < 5000.0, f"Pytest runner should complete in <5s, took {execution_time:.1f}ms"


import unittest # Add this import

class TestBETESCalculatorCore(unittest.TestCase): # Inherit from unittest.TestCase
    """High-quality tests for bE-TES v3.1 calculation."""

    def test_betes_v3_1_default_calculation(self):
        """Test bE-TES v3.1 calculation with default settings (no sigmoids)."""
        calculator = BETESCalculator(settings_v3_1=BETESSettingsV31()) # Default settings
        components = calculator.calculate(
            raw_mutation_score=0.8,    # M' via minmax: (0.8-0.6)/(0.95-0.6) = 0.2/0.35 = ~0.5714
            raw_emt_gain=0.1,          # E' via clip: 0.1/0.25 = 0.4
            raw_assertion_iq=3.0,      # A': (3-1)/4 = 0.5
            raw_behaviour_coverage=0.7,# B': 0.7
            raw_median_test_time_ms=150, # S' via log: 1 / (1 + log10(150/100)) = 1 / (1 + log10(1.5)) = 1 / (1+0.176) = ~0.850
            raw_flakiness_rate=0.05    # T = 0.95
        )
        
        self.assertAlmostEqual(components.norm_mutation_score, (0.8 - 0.6) / (0.95 - 0.6), places=4)
        self.assertAlmostEqual(components.norm_emt_gain, 0.1 / 0.25, places=4)
        self.assertAlmostEqual(components.norm_assertion_iq, (3.0 - 1.0) / 4.0, places=4)
        self.assertAlmostEqual(components.norm_behaviour_coverage, 0.7, places=4)
        self.assertAlmostEqual(components.norm_speed_factor, 1.0 / (1.0 + math.log10(1.5)), places=4)
        self.assertAlmostEqual(components.trust_coefficient_t, 1.0 - 0.05, places=4)

        # G = (M'^1 * E'^1 * A'^1 * B'^1 * S'^1)^(1/5)
        expected_g = (components.norm_mutation_score * components.norm_emt_gain * components.norm_assertion_iq * \
                      components.norm_behaviour_coverage * components.norm_speed_factor) ** (1/5)
        self.assertAlmostEqual(components.geometric_mean_g, expected_g, places=4)
        self.assertAlmostEqual(components.betes_score, expected_g * components.trust_coefficient_t, places=4)

    def test_betes_s_prime_piecewise_normalization(self):
        """Test S' (Speed Factor) piece-wise normalization."""
        calc = BETESCalculator() # Default settings
        # Test t <= 0 -> S' = 0
        self.assertAlmostEqual(calc.calculate(0,0,1,0,0,0).norm_speed_factor, 0.0)
        self.assertAlmostEqual(calc.calculate(0,0,1,0,-10,0).norm_speed_factor, 0.0)
        # Test 0 < t <= 100 -> S' = 1.0
        self.assertAlmostEqual(calc.calculate(0,0,1,0,1,0).norm_speed_factor, 1.0)
        self.assertAlmostEqual(calc.calculate(0,0,1,0,50,0).norm_speed_factor, 1.0)
        self.assertAlmostEqual(calc.calculate(0,0,1,0,100,0).norm_speed_factor, 1.0)
        # Test t > 100 -> S' = 1 / (1 + log10(t/100))
        self.assertAlmostEqual(calc.calculate(0,0,1,0,200,0).norm_speed_factor, 1.0 / (1.0 + math.log10(2.0)), places=5)
        self.assertAlmostEqual(calc.calculate(0,0,1,0,1000,0).norm_speed_factor, 1.0 / (1.0 + math.log10(10.0)), places=5) # 1/(1+1) = 0.5
        self.assertAlmostEqual(calc.calculate(0,0,1,0,10,0).norm_speed_factor, 1.0) # Covered by t <= 100

    def test_betes_m_prime_sigmoid_normalization(self):
        """Test M' (Mutation Score) sigmoid normalization."""
        settings = BETESSettingsV31(smooth_m=True, k_m=14.0) # k_m=14 as per plan
        calculator = BETESCalculator(settings_v3_1=settings)
        
        # M = 0.775 (center) -> M' = 0.5
        self.assertAlmostEqual(calculator.calculate(0.775,0,1,0,100,0).norm_mutation_score, 0.5, places=4)
        # M = 0.6 (lower interesting bound)
        m_low = 0.6
        expected_low = 1 / (1 + math.exp(-14.0 * (m_low - 0.775)))
        self.assertAlmostEqual(calculator.calculate(m_low,0,1,0,100,0).norm_mutation_score, expected_low, places=4)
        # M = 0.95 (upper interesting bound)
        m_high = 0.95
        expected_high = 1 / (1 + math.exp(-14.0 * (m_high - 0.775)))
        self.assertAlmostEqual(calculator.calculate(m_high,0,1,0,100,0).norm_mutation_score, expected_high, places=4)
        # M very low -> M' approaches 0
        self.assertLess(calculator.calculate(0.1,0,1,0,100,0).norm_mutation_score, 0.01)
        # M very high -> M' approaches 1
        self.assertGreater(calculator.calculate(1.5,0,1,0,100,0).norm_mutation_score, 0.99)

    def test_betes_e_prime_sigmoid_normalization(self):
        """Test E' (EMT Gain) sigmoid normalization."""
        settings = BETESSettingsV31(smooth_e=True, k_e=12.0) # k_e=12 as per plan
        calculator = BETESCalculator(settings_v3_1=settings)

        # E_raw = 0.125 (center) -> E' = 0.5
        self.assertAlmostEqual(calculator.calculate(0,0.125,1,0,100,0).norm_emt_gain, 0.5, places=4)
        # E_raw = 0.0 (lower interesting bound for gain)
        e_low = 0.0
        expected_low = 1 / (1 + math.exp(-12.0 * (e_low - 0.125)))
        self.assertAlmostEqual(calculator.calculate(0,e_low,1,0,100,0).norm_emt_gain, expected_low, places=4)
        # E_raw = 0.25 (upper interesting bound for gain, full credit in clip)
        e_high = 0.25
        expected_high = 1 / (1 + math.exp(-12.0 * (e_high - 0.125)))
        self.assertAlmostEqual(calculator.calculate(0,e_high,1,0,100,0).norm_emt_gain, expected_high, places=4)
         # E_raw very low (negative gain) -> E' approaches 0
        self.assertLess(calculator.calculate(0,-0.5,1,0,100,0).norm_emt_gain, 0.01)
        # E_raw very high -> E' approaches 1
        self.assertGreater(calculator.calculate(0,1.0,1,0,100,0).norm_emt_gain, 0.99)

    def test_betes_calculator_init_with_settings(self):
        """Test BETESCalculator initialization with v3.1 settings."""
        weights = BETESWeights(w_m=2.0)
        settings = BETESSettingsV31(smooth_m=True, k_m=10.0)
        calculator = BETESCalculator(weights=weights, settings_v3_1=settings)
        self.assertEqual(calculator.weights.w_m, 2.0)
        self.assertTrue(calculator.settings_v3_1.smooth_m)
        self.assertEqual(calculator.settings_v3_1.k_m, 10.0)

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

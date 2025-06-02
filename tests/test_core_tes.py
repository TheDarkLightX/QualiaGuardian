"""
Tests for Guardian Core TES functionality

Testing the core Test Effectiveness Score calculation and components.
"""

import pytest
import sys
import os

# Add guardian to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'guardian'))

from guardian.core.tes import get_etes_grade, calculate_etes_v2, compare_tes_vs_etes
from guardian.core.etes import ETESCalculator, QualityConfig, ETESComponents


class TestTESCalculation:
    """Test TES calculation functionality"""
    
    def test_calculate_tes_basic(self):
        """Test basic TES calculation"""
        # Test with known values
        # Test with known values
        # calculate_tes was removed, using calculate_etes_v2 as a stand-in.
        # This requires providing more comprehensive data.
        tes_score = calculate_etes_v2(
            test_suite_data={
                'mutation_score': 0.8,
                'avg_test_execution_time_ms': 100, # Dummy, adjust if speed_factor is used differently
                'assertions': [{'type': 'equality', 'code': 'dummy', 'target_criticality': 3.5/3}], # Approximate
                'covered_behaviors': ['dummy_behavior'], # Approximate
                'execution_results': [{'passed': True, 'execution_time_ms': 100}],
                'determinism_score': 0.9, # Dummy
                'stability_score': 0.9, # Dummy
                'readability_score': 0.9, # Dummy
                'independence_score': 0.9, # Dummy
                # speed_factor (0.85) is now calculated internally by ETESCalculator
            },
            codebase_data={
                'all_behaviors': ['dummy_behavior'], # Approximate
                'behavior_criticality': {'dummy_behavior': 0.9/0.1}, # Approximate
                'complexity_metrics': {'avg_cyclomatic_complexity': 3.0, 'total_loc': 1000} # Dummy
            }
        )[0] # calculate_etes_v2 returns (score, components)
        
        # TES should be a float between 0 and 1
        assert isinstance(tes_score, float)
        assert 0.0 <= tes_score <= 1.0
        
        # With good values, should get a reasonable score
        assert tes_score > 0.5
    
    def test_calculate_tes_edge_cases(self):
        """Test TES calculation with edge cases"""
        # Test with zero values
        tes_zero = calculate_etes_v2(
            test_suite_data={'mutation_score': 0.0, 'avg_test_execution_time_ms': 100, 'assertions': [], 'covered_behaviors': [], 'execution_results': [], 'determinism_score': 0.0, 'stability_score': 0.0, 'readability_score': 0.0, 'independence_score': 0.0},
            codebase_data={'all_behaviors': [], 'behavior_criticality': {}, 'complexity_metrics': {}}
        )[0]
        assert tes_zero == 0.0
        
        # Test with maximum values
        tes_max = calculate_etes_v2(
            test_suite_data={'mutation_score': 1.0, 'avg_test_execution_time_ms': 10, 'assertions': [{'type': 'equality', 'code': 'dummy', 'target_criticality': 10.0/3}], 'covered_behaviors': ['dummy'], 'execution_results': [{'passed': True, 'execution_time_ms': 10}], 'determinism_score': 1.0, 'stability_score': 1.0, 'readability_score': 1.0, 'independence_score': 1.0},
            codebase_data={'all_behaviors': ['dummy'], 'behavior_criticality': {'dummy': 1.0/0.1}, 'complexity_metrics': {}}
        )[0]
        assert isinstance(tes_max, float)
        assert tes_max > 0.8  # Should be high with max values
    
    def test_get_tes_grade(self):
        """Test TES grade assignment"""
        assert get_etes_grade(0.95) == "A+"
        assert get_etes_grade(0.85) == "A"
        assert get_etes_grade(0.75) == "B"
        assert get_etes_grade(0.65) == "C"
        assert get_etes_grade(0.45) == "F"
        assert get_etes_grade(0.0) == "F"
    
    def test_tes_consistency(self):
        """Test TES calculation consistency"""
        # Same inputs should give same outputs
        params = {
            'mutation_score': 0.75,
            'assertion_density': 4.2,
            'behavior_coverage': 0.82,
            'speed_factor': 0.78
        }
        
        # calculate_tes was removed, using calculate_etes_v2 as a stand-in.
        # This requires providing more comprehensive data.
        test_suite_data_params = {
            'mutation_score': params['mutation_score'],
            'avg_test_execution_time_ms': 100, # Dummy, adjust if speed_factor is used differently
            'assertions': [{'type': 'equality', 'code': 'dummy', 'target_criticality': params['assertion_density']/3 if params['assertion_density'] else 1}],
            'covered_behaviors': ['dummy_behavior'],
            'execution_results': [{'passed': True, 'execution_time_ms': 100}],
            'determinism_score': 0.9, 'stability_score': 0.9, 'readability_score': 0.9, 'independence_score': 0.9
            # speed_factor (params['speed_factor']) is now calculated internally
        }
        codebase_data_params = {
            'all_behaviors': ['dummy_behavior'],
            'behavior_criticality': {'dummy_behavior': params['behavior_coverage']/0.1 if params['behavior_coverage'] else 1},
            'complexity_metrics': {'avg_cyclomatic_complexity': 3.0, 'total_loc': 1000}
        }
        score1 = calculate_etes_v2(test_suite_data_params, codebase_data_params)[0]
        score2 = calculate_etes_v2(test_suite_data_params, codebase_data_params)[0]
        
        assert score1 == score2


class TestETESCalculation:
    """Test E-TES v2.0 calculation functionality"""
    
    def test_etes_calculator_initialization(self):
        """Test E-TES calculator initialization"""
        calculator = ETESCalculator()
        assert calculator.config is not None
        assert calculator.history == []
        assert calculator.baseline_score is None
    
    def test_etes_calculator_with_config(self):
        """Test E-TES calculator with custom config"""
        config = QualityConfig(
            max_generations=5,
            population_size=25,
            min_mutation_score=0.7
        )
        calculator = ETESCalculator(config)
        assert calculator.config.max_generations == 5
        assert calculator.config.population_size == 25
        assert calculator.config.min_mutation_score == 0.7
    
    def test_calculate_etes_v2_basic(self):
        """Test basic E-TES v2.0 calculation"""
        test_suite_data = {
            'mutation_score': 0.8,
            'avg_test_execution_time_ms': 150,
            'assertions': [
                {'type': 'equality', 'code': 'assert x == y', 'target_criticality': 1.0},
                {'type': 'boundary', 'code': 'assert len(data) > 0', 'target_criticality': 1.5}
            ],
            'covered_behaviors': ['login', 'validation'],
            'execution_results': [
                {'passed': True, 'execution_time_ms': 150}
            ],
            'determinism_score': 0.95,
            'stability_score': 0.90,
            'readability_score': 0.85,
            'independence_score': 0.88
        }
        
        codebase_data = {
            'all_behaviors': ['login', 'validation', 'logout'],
            'behavior_criticality': {'login': 2.0, 'validation': 1.5, 'logout': 1.0},
            'complexity_metrics': {'avg_cyclomatic_complexity': 3.0, 'total_loc': 1000}
        }
        
        etes_score, components = calculate_etes_v2(test_suite_data, codebase_data)
        
        # Validate results
        assert isinstance(etes_score, float)
        assert 0.0 <= etes_score <= 1.0
        assert isinstance(components, ETESComponents)
        
        # Check components are reasonable
        assert 0.0 <= components.mutation_score <= 1.0
        assert components.evolution_gain >= 1.0
        assert 0.0 <= components.assertion_iq <= 1.0
        assert 0.0 <= components.behavior_coverage <= 1.0
        assert 0.0 <= components.speed_factor <= 1.0
        assert 0.0 <= components.quality_factor <= 1.0
    
    def test_etes_components_validation(self):
        """Test E-TES components are properly calculated"""
        test_data = {
            'mutation_score': 0.9,
            'avg_test_execution_time_ms': 50,  # Fast execution
            'assertions': [
                {'type': 'invariant', 'code': 'assert invariant()', 'target_criticality': 2.0, 'checks_invariant': True},
                {'type': 'property', 'code': 'assert property()', 'target_criticality': 1.8}
            ],
            'covered_behaviors': ['critical_feature'],
            'execution_results': [{'passed': True, 'execution_time_ms': 50}],
            'determinism_score': 0.98,
            'stability_score': 0.95,
            'readability_score': 0.92,
            'independence_score': 0.96
        }
        
        codebase_data = {
            'all_behaviors': ['critical_feature'],
            'behavior_criticality': {'critical_feature': 3.0},
            'complexity_metrics': {'avg_cyclomatic_complexity': 2.0, 'total_loc': 500}
        }
        
        etes_score, components = calculate_etes_v2(test_data, codebase_data)
        
        # With high-quality inputs, should get high scores
        assert etes_score > 0.7
        assert components.mutation_score > 0.8
        assert components.assertion_iq > 0.8  # High due to invariant assertions
        assert components.speed_factor > 0.8  # Fast execution
        assert components.quality_factor > 0.9  # High quality scores
    
    def test_compare_tes_vs_etes(self):
        """Test TES vs E-TES comparison functionality"""
        # Calculate legacy TES
        # calculate_tes was removed, using calculate_etes_v2 as a stand-in.
        # This requires providing more comprehensive data.
        legacy_tes = calculate_etes_v2(
            test_suite_data={
                'mutation_score': 0.7,
                'avg_test_execution_time_ms': 100, # Dummy, adjust if speed_factor is used differently
                'assertions': [{'type': 'equality', 'code': 'dummy', 'target_criticality': 3.0/3}],
                'covered_behaviors': ['dummy_behavior'],
                'execution_results': [{'passed': True, 'execution_time_ms': 100}],
                'determinism_score': 0.9, 'stability_score': 0.9, 'readability_score': 0.9, 'independence_score': 0.9
                 # speed_factor (0.75) is now calculated internally
            },
            codebase_data={
                'all_behaviors': ['dummy_behavior'],
                'behavior_criticality': {'dummy_behavior': 0.8/0.1},
                'complexity_metrics': {'avg_cyclomatic_complexity': 3.0, 'total_loc': 1000}
            }
        )[0]
        
        # Calculate E-TES
        test_data = {
            'mutation_score': 0.75,
            'avg_test_execution_time_ms': 120,
            'assertions': [
                {'type': 'equality', 'code': 'assert x == y', 'target_criticality': 1.0}
            ],
            'covered_behaviors': ['feature'],
            'execution_results': [{'passed': True, 'execution_time_ms': 120}],
            'determinism_score': 0.9,
            'stability_score': 0.85,
            'readability_score': 0.8,
            'independence_score': 0.87
        }
        
        codebase_data = {
            'all_behaviors': ['feature'],
            'behavior_criticality': {'feature': 2.0},
            'complexity_metrics': {'avg_cyclomatic_complexity': 3.5, 'total_loc': 1200}
        }
        
        etes_score, etes_components = calculate_etes_v2(test_data, codebase_data)
        
        # Compare
        comparison = compare_tes_vs_etes(legacy_tes, etes_score, etes_components)
        
        # Validate comparison structure
        assert 'legacy_tes' in comparison
        assert 'etes_v2' in comparison
        assert 'improvement' in comparison
        assert 'recommendations' in comparison
        
        assert comparison['legacy_tes'] == legacy_tes
        assert comparison['etes_v2'] == etes_score
        assert comparison['improvement'] == etes_score - legacy_tes
        assert isinstance(comparison['recommendations'], list)


class TestETESConfig:
    """Test E-TES configuration functionality"""
    
    def test_default_config(self):
        """Test default E-TES configuration"""
        config = QualityConfig()
        
        # Check default values
        assert config.max_generations == 10
        assert config.population_size == 100
        assert config.min_mutation_score == 0.8
        assert config.min_behavior_coverage == 0.9
        assert config.max_test_runtime_ms == 200.0
        
        # Check weights sum to reasonable value
        weight_sum = sum(config.weights.values())
        assert 0.9 <= weight_sum <= 1.1  # Should be close to 1.0
    
    def test_custom_config(self):
        """Test custom E-TES configuration"""
        custom_weights = {
            'mutation_score': 0.4,
            'evolution_gain': 0.1,
            'assertion_iq': 0.2,
            'behavior_coverage': 0.2,
            'speed_factor': 0.05,
            'quality_factor': 0.05
        }
        
        config = QualityConfig(
            max_generations=20,
            population_size=150,
            min_mutation_score=0.85,
            weights=custom_weights
        )
        
        assert config.max_generations == 20
        assert config.population_size == 150
        assert config.min_mutation_score == 0.85
        assert config.weights == custom_weights


class TestETESIntegration:
    """Integration tests for E-TES system"""
    
    def test_etes_evolution_tracking(self):
        """Test E-TES evolution tracking over multiple calculations"""
        calculator = ETESCalculator()
        
        test_data = {
            'mutation_score': 0.6,
            'avg_test_execution_time_ms': 200,
            'assertions': [{'type': 'equality', 'code': 'assert True', 'target_criticality': 1.0}],
            'covered_behaviors': ['basic'],
            'execution_results': [{'passed': True, 'execution_time_ms': 200}],
            'determinism_score': 0.8,
            'stability_score': 0.75,
            'readability_score': 0.7,
            'independence_score': 0.8
        }
        
        codebase_data = {
            'all_behaviors': ['basic', 'advanced'],
            'behavior_criticality': {'basic': 1.0, 'advanced': 2.0},
            'complexity_metrics': {'avg_cyclomatic_complexity': 4.0, 'total_loc': 1500}
        }
        
        # First calculation
        score1, components1 = calculator.calculate_etes(test_data, codebase_data)
        
        # Improve test data
        test_data['mutation_score'] = 0.8
        test_data['avg_test_execution_time_ms'] = 150
        test_data['covered_behaviors'] = ['basic', 'advanced']
        
        # Second calculation
        score2, components2 = calculator.calculate_etes(test_data, codebase_data, score1)
        
        # Should show evolution gain
        assert components2.evolution_gain > 1.0
        assert score2 > score1
        
        # History should be tracked
        assert len(calculator.history) == 2
    
    def test_etes_insights_generation(self):
        """Test E-TES insights generation"""
        test_data = {
            'mutation_score': 0.4,  # Low
            'avg_test_execution_time_ms': 500,  # Slow
            'assertions': [{'type': 'equality', 'code': 'assert True', 'target_criticality': 1.0}],
            'covered_behaviors': ['basic'],
            'execution_results': [{'passed': True, 'execution_time_ms': 500}],
            'determinism_score': 0.6,  # Low
            'stability_score': 0.5,   # Low
            'readability_score': 0.7,
            'independence_score': 0.8
        }
        
        codebase_data = {
            'all_behaviors': ['basic', 'advanced', 'critical'],
            'behavior_criticality': {'basic': 1.0, 'advanced': 2.0, 'critical': 3.0},
            'complexity_metrics': {'avg_cyclomatic_complexity': 6.0, 'total_loc': 3000}
        }
        
        etes_score, components = calculate_etes_v2(test_data, codebase_data)
        
        # Should generate insights for low scores
        assert len(components.insights) > 0
        
        # Check for specific insights about low scores
        insights_text = ' '.join(components.insights).lower()
        assert 'mutation' in insights_text or 'coverage' in insights_text or 'speed' in insights_text


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

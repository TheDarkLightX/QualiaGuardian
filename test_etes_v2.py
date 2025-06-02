#!/usr/bin/env python3
"""
Test script for E-TES v2.0 implementation

This script demonstrates and tests the E-TES v2.0 functionality
including evolutionary mutation testing and multi-objective optimization.
"""

import sys
import os
import time
import json
from typing import Dict, Any

# Add guardian to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'guardian'))

from guardian.core.etes import ETESCalculator, QualityConfig, ETESComponents
from guardian.core.tes import calculate_etes_v2, compare_tes_vs_etes
from guardian.evolution.adaptive_emt import AdaptiveEMT
from guardian.evolution.smart_mutator import SmartMutator
from guardian.metrics.quality_factor import QualityFactorCalculator
from guardian.metrics.evolution_history import EvolutionHistoryTracker


def create_sample_test_suite_data() -> Dict[str, Any]:
    """Create sample test suite data for E-TES calculation"""
    return {
        'mutation_score': 0.75,
        'avg_test_execution_time_ms': 150.0,
        'assertions': [
            {'type': 'equality', 'code': 'assert result == expected', 'target_criticality': 1.0},
            {'type': 'type_check', 'code': 'assert isinstance(obj, MyClass)', 'target_criticality': 1.2},
            {'type': 'exception', 'code': 'with pytest.raises(ValueError):', 'target_criticality': 1.5},
            {'type': 'boundary', 'code': 'assert len(data) > 0', 'target_criticality': 1.6},
            {'type': 'invariant', 'code': 'assert invariant_holds(state)', 'target_criticality': 2.0, 'checks_invariant': True},
        ],
        'covered_behaviors': ['user_login', 'data_validation', 'error_handling'],
        'execution_results': [
            {'passed': True, 'execution_time_ms': 145},
            {'passed': True, 'execution_time_ms': 152},
            {'passed': True, 'execution_time_ms': 148},
            {'passed': True, 'execution_time_ms': 155},
            {'passed': True, 'execution_time_ms': 150},
        ],
        'determinism_score': 0.95,
        'stability_score': 0.90,
        'readability_score': 0.85,
        'independence_score': 0.88,
        'mutants': [
            {'killed': True, 'severity_weight': 2.0, 'mutation_type': 'boundary'},
            {'killed': True, 'severity_weight': 1.5, 'mutation_type': 'arithmetic'},
            {'killed': False, 'severity_weight': 3.0, 'mutation_type': 'logical'},
            {'killed': True, 'severity_weight': 1.0, 'mutation_type': 'relational'},
        ]
    }


def create_sample_codebase_data() -> Dict[str, Any]:
    """Create sample codebase data for E-TES calculation"""
    return {
        'all_behaviors': ['user_login', 'data_validation', 'error_handling', 'user_logout', 'data_processing'],
        'behavior_criticality': {
            'user_login': 3.0,
            'data_validation': 2.5,
            'error_handling': 2.0,
            'user_logout': 1.5,
            'data_processing': 2.0
        },
        'complexity_metrics': {
            'avg_cyclomatic_complexity': 4.2,
            'total_loc': 1500
        }
    }


def test_etes_calculator():
    """Test the core E-TES calculator"""
    print("🧪 Testing E-TES Calculator...")
    
    config = QualityConfig(
        max_generations=5,
        population_size=50,
        min_mutation_score=0.70,
        min_behavior_coverage=0.80
    )
    
    calculator = ETESCalculator(config)
    test_data = create_sample_test_suite_data()
    codebase_data = create_sample_codebase_data()
    
    # Calculate E-TES
    start_time = time.time()
    etes_score, components = calculator.calculate_etes(test_data, codebase_data)
    calculation_time = time.time() - start_time
    
    print(f"✅ E-TES Score: {etes_score:.3f}")
    print(f"⏱️  Calculation Time: {calculation_time:.2f}s")
    print(f"📊 Components:")
    print(f"   • Mutation Score: {components.mutation_score:.3f}")
    print(f"   • Evolution Gain: {components.evolution_gain:.3f}")
    print(f"   • Assertion IQ: {components.assertion_iq:.3f}")
    print(f"   • Behavior Coverage: {components.behavior_coverage:.3f}")
    print(f"   • Speed Factor: {components.speed_factor:.3f}")
    print(f"   • Quality Factor: {components.quality_factor:.3f}")
    
    if components.insights:
        print(f"💡 Insights:")
        for insight in components.insights:
            print(f"   • {insight}")
    
    return etes_score, components


def test_smart_mutator():
    """Test the smart mutation generator"""
    print("\n🧬 Testing Smart Mutator...")
    
    # Create a simple test file
    test_code = """
def calculate_sum(a, b):
    if a > 0 and b > 0:
        return a + b
    elif a < 0 or b < 0:
        raise ValueError("Negative values not allowed")
    else:
        return 0

def process_list(items):
    result = []
    for i in range(len(items)):
        if items[i] is not None:
            result.append(items[i] * 2)
    return result
"""
    
    with open('temp_test_file.py', 'w') as f:
        f.write(test_code)
    
    try:
        mutator = SmartMutator('.', mutation_budget=20)
        mutants = mutator.generate_smart_mutants('temp_test_file.py')
        
        print(f"✅ Generated {len(mutants)} smart mutants")
        
        # Show top 5 mutants
        for i, mutant in enumerate(mutants[:5]):
            print(f"   {i+1}. {mutant.description}")
            print(f"      Impact: {mutant.impact_score:.2f}, Likelihood: {mutant.likelihood:.2f}")
            print(f"      Type: {mutant.mutation_type.value}")
        
        return mutants
        
    finally:
        # Cleanup
        if os.path.exists('temp_test_file.py'):
            os.remove('temp_test_file.py')


def test_quality_factor():
    """Test the quality factor calculator"""
    print("\n📏 Testing Quality Factor Calculator...")
    
    calculator = QualityFactorCalculator(sample_runs=5)
    
    test_data = {
        'execution_results': [
            {'passed': True, 'execution_time_ms': 100},
            {'passed': True, 'execution_time_ms': 105},
            {'passed': True, 'execution_time_ms': 98},
            {'passed': True, 'execution_time_ms': 102},
            {'passed': True, 'execution_time_ms': 101},
        ],
        'environment_failures': 0,
        'timing_issues': 0,
        'resource_issues': 0,
        'modification_frequency': 0.1,
        'total_runs': 5
    }
    
    test_code = """
def test_user_login():
    # Arrange
    user = create_test_user()
    credentials = get_valid_credentials()
    
    # Act
    result = login_service.authenticate(user, credentials)
    
    # Assert
    assert result.success is True
    assert result.user_id == user.id
    assert result.session_token is not None
"""
    
    metrics = calculator.calculate_quality_factor(test_data, test_code)
    
    print(f"✅ Quality Factor: {metrics.calculate_quality_factor():.3f}")
    print(f"📊 Components:")
    print(f"   • Determinism: {metrics.determinism:.3f}")
    print(f"   • Stability: {metrics.stability:.3f}")
    print(f"   • Clarity: {metrics.clarity:.3f}")
    print(f"   • Independence: {metrics.independence:.3f}")
    print(f"   • Flakiness Score: {metrics.flakiness_score:.3f}")
    print(f"   • Readability Score: {metrics.readability_score:.3f}")
    
    return metrics


def test_evolution_history():
    """Test the evolution history tracker"""
    print("\n📈 Testing Evolution History Tracker...")
    
    tracker = EvolutionHistoryTracker(db_path="test_evolution.db")
    
    # Simulate evolution progress
    for gen in range(10):
        snapshot = tracker.EvolutionSnapshot(
            timestamp=time.time(),
            generation=gen,
            etes_score=0.5 + (gen * 0.04) + (0.01 * (gen % 3)),  # Simulated improvement
            mutation_score=0.6 + (gen * 0.03),
            assertion_iq=0.7 + (gen * 0.02),
            behavior_coverage=0.8 + (gen * 0.015),
            speed_factor=0.9 - (gen * 0.005),
            quality_factor=0.85 + (gen * 0.01),
            population_size=100,
            best_individual_id=f"individual_{gen}_best",
            diversity_score=0.8 - (gen * 0.02),
            mutation_rate=0.1 + (gen * 0.005),
            convergence_indicator=gen * 0.1,
            evaluation_time_ms=1000 + (gen * 50),
            memory_usage_mb=500 + (gen * 20)
        )
        tracker.record_snapshot(snapshot)
    
    # Analyze trends
    trends = tracker.analyze_trends()
    summary = tracker.get_performance_summary()
    insights = tracker.get_insights()
    
    print(f"✅ Evolution Analysis Complete")
    print(f"📊 Performance Summary:")
    print(f"   • Best Score: {summary.get('best_score', 0):.3f}")
    print(f"   • Final Score: {summary.get('final_score', 0):.3f}")
    print(f"   • Total Improvement: {summary.get('improvement_total', 0):.3f}")
    print(f"   • Convergence Generation: {summary.get('convergence_generation', 'N/A')}")
    
    print(f"📈 Trend Analysis:")
    print(f"   • Improvement Rate: {trends.improvement_rate:.4f}")
    print(f"   • Trend Direction: {trends.trend_direction}")
    print(f"   • Plateau Detected: {trends.plateau_detected}")
    print(f"   • Confidence: {trends.confidence:.3f}")
    
    if insights:
        print(f"💡 Evolution Insights:")
        for insight in insights:
            print(f"   • {insight}")
    
    # Cleanup
    if os.path.exists("test_evolution.db"):
        os.remove("test_evolution.db")
    
    return trends, summary


def test_tes_vs_etes_comparison():
    """Test comparison between legacy TES and E-TES v2.0"""
    print("\n⚖️  Testing TES vs E-TES Comparison...")
    
    # Calculate legacy TES
    legacy_tes = calculate_tes(
        mutation_score=0.75,
        assertion_density=3.5,
        behavior_coverage=0.80,
        speed_factor=0.85
    )
    
    # Calculate E-TES v2.0
    test_data = create_sample_test_suite_data()
    codebase_data = create_sample_codebase_data()
    etes_score, etes_components = calculate_etes_v2(test_data, codebase_data)
    
    # Compare
    comparison = compare_tes_vs_etes(legacy_tes, etes_score, etes_components)
    
    print(f"✅ Comparison Complete")
    print(f"📊 Scores:")
    print(f"   • Legacy TES: {comparison['legacy_tes']:.3f} (Grade: {comparison['legacy_grade']})")
    print(f"   • E-TES v2.0: {comparison['etes_v2']:.3f} (Grade: {comparison['etes_grade']})")
    print(f"   • Improvement: {comparison['improvement']:.3f}")
    
    if comparison['recommendations']:
        print(f"💡 Recommendations:")
        for rec in comparison['recommendations']:
            print(f"   • {rec}")
    
    return comparison


def run_comprehensive_test():
    """Run comprehensive E-TES v2.0 test suite"""
    print("🚀 Starting E-TES v2.0 Comprehensive Test Suite")
    print("=" * 60)
    
    try:
        # Test individual components
        etes_score, components = test_etes_calculator()
        mutants = test_smart_mutator()
        quality_metrics = test_quality_factor()
        trends, summary = test_evolution_history()
        comparison = test_tes_vs_etes_comparison()
        
        print("\n" + "=" * 60)
        print("🎉 All Tests Completed Successfully!")
        print(f"📊 Final E-TES Score: {etes_score:.3f}")
        print(f"🧬 Generated {len(mutants)} smart mutants")
        print(f"📏 Quality Factor: {quality_metrics.calculate_quality_factor():.3f}")
        print(f"📈 Evolution Improvement Rate: {trends.improvement_rate:.4f}")
        print(f"⚖️  TES vs E-TES Improvement: {comparison['improvement']:.3f}")
        
        # Export results
        results = {
            'etes_score': etes_score,
            'etes_components': {
                'mutation_score': components.mutation_score,
                'evolution_gain': components.evolution_gain,
                'assertion_iq': components.assertion_iq,
                'behavior_coverage': components.behavior_coverage,
                'speed_factor': components.speed_factor,
                'quality_factor': components.quality_factor,
                'insights': components.insights
            },
            'mutants_generated': len(mutants),
            'quality_metrics': {
                'overall_quality': quality_metrics.calculate_quality_factor(),
                'determinism': quality_metrics.determinism,
                'stability': quality_metrics.stability,
                'clarity': quality_metrics.clarity,
                'independence': quality_metrics.independence
            },
            'evolution_analysis': {
                'improvement_rate': trends.improvement_rate,
                'trend_direction': trends.trend_direction,
                'convergence_detected': trends.convergence_generation is not None
            },
            'comparison': comparison
        }
        
        with open('etes_v2_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Results exported to: etes_v2_test_results.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)

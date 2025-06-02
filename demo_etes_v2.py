#!/usr/bin/env python3
"""
E-TES v2.0 Demonstration Script

This script showcases the key features and capabilities of the 
E-TES v2.0 evolutionary test effectiveness scoring system.
"""

import sys
import os
import time
import json

# Add guardian to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'guardian'))

from guardian.core.etes import ETESCalculator, ETESConfig
from guardian.core.tes import calculate_etes_v2, compare_tes_vs_etes, calculate_tes


def demo_basic_etes():
    """Demonstrate basic E-TES v2.0 calculation"""
    print("🎯 E-TES v2.0 Basic Demonstration")
    print("=" * 50)
    
    # Sample test suite with good coverage
    good_test_data = {
        'mutation_score': 0.87,
        'avg_test_execution_time_ms': 95,
        'assertions': [
            {'type': 'equality', 'code': 'assert result == expected', 'target_criticality': 1.0},
            {'type': 'type_check', 'code': 'assert isinstance(obj, MyClass)', 'target_criticality': 1.2},
            {'type': 'exception', 'code': 'with pytest.raises(ValueError):', 'target_criticality': 1.5},
            {'type': 'boundary', 'code': 'assert len(data) > 0', 'target_criticality': 1.6},
            {'type': 'invariant', 'code': 'assert invariant_holds()', 'target_criticality': 2.0, 'checks_invariant': True},
            {'type': 'property', 'code': 'assert property_maintained()', 'target_criticality': 1.8},
        ],
        'covered_behaviors': ['user_auth', 'data_validation', 'error_handling', 'business_logic'],
        'execution_results': [
            {'passed': True, 'execution_time_ms': 92},
            {'passed': True, 'execution_time_ms': 98},
            {'passed': True, 'execution_time_ms': 95},
        ],
        'determinism_score': 0.98,
        'stability_score': 0.92,
        'readability_score': 0.89,
        'independence_score': 0.91,
    }
    
    # Sample codebase with defined behaviors
    codebase_data = {
        'all_behaviors': ['user_auth', 'data_validation', 'error_handling', 'business_logic', 'reporting'],
        'behavior_criticality': {
            'user_auth': 3.0,
            'data_validation': 2.5,
            'error_handling': 2.0,
            'business_logic': 2.8,
            'reporting': 1.5
        },
        'complexity_metrics': {
            'avg_cyclomatic_complexity': 3.8,
            'total_loc': 2500
        }
    }
    
    # Calculate E-TES
    print("📊 Calculating E-TES v2.0...")
    start_time = time.time()
    etes_score, components = calculate_etes_v2(good_test_data, codebase_data)
    calc_time = time.time() - start_time
    
    print(f"✅ E-TES Score: {etes_score:.3f}")
    print(f"⏱️  Calculation Time: {calc_time:.3f}s")
    print()
    
    print("📈 Component Breakdown:")
    print(f"   • Mutation Score: {components.mutation_score:.3f}")
    print(f"   • Evolution Gain: {components.evolution_gain:.3f}")
    print(f"   • Assertion IQ: {components.assertion_iq:.3f}")
    print(f"   • Behavior Coverage: {components.behavior_coverage:.3f}")
    print(f"   • Speed Factor: {components.speed_factor:.3f}")
    print(f"   • Quality Factor: {components.quality_factor:.3f}")
    print()
    
    if components.insights:
        print("💡 E-TES Insights:")
        for insight in components.insights:
            print(f"   • {insight}")
        print()
    
    return etes_score, components


def demo_comparison():
    """Demonstrate TES vs E-TES comparison"""
    print("⚖️  TES vs E-TES v2.0 Comparison")
    print("=" * 50)
    
    # Calculate legacy TES
    legacy_tes = calculate_tes(
        mutation_score=0.75,
        assertion_density=4.2,
        behavior_coverage=0.80,
        speed_factor=0.88
    )
    
    # Sample data for E-TES
    test_data = {
        'mutation_score': 0.82,
        'avg_test_execution_time_ms': 110,
        'assertions': [
            {'type': 'equality', 'code': 'assert x == y', 'target_criticality': 1.0},
            {'type': 'boundary', 'code': 'assert len(data) >= 0', 'target_criticality': 1.6},
            {'type': 'exception', 'code': 'with pytest.raises(Error):', 'target_criticality': 1.5},
            {'type': 'invariant', 'code': 'assert state_valid()', 'target_criticality': 2.0, 'checks_invariant': True},
        ],
        'covered_behaviors': ['core_feature', 'validation'],
        'execution_results': [{'passed': True, 'execution_time_ms': 110}],
        'determinism_score': 0.94,
        'stability_score': 0.89,
        'readability_score': 0.87,
        'independence_score': 0.92,
    }
    
    codebase_data = {
        'all_behaviors': ['core_feature', 'validation', 'reporting'],
        'behavior_criticality': {'core_feature': 3.0, 'validation': 2.5, 'reporting': 1.0},
        'complexity_metrics': {'avg_cyclomatic_complexity': 4.0, 'total_loc': 1800}
    }
    
    # Calculate E-TES
    etes_score, etes_components = calculate_etes_v2(test_data, codebase_data)
    
    # Compare
    comparison = compare_tes_vs_etes(legacy_tes, etes_score, etes_components)
    
    print(f"📊 Legacy TES: {comparison['legacy_tes']:.3f} (Grade: {comparison['legacy_grade']})")
    print(f"🧬 E-TES v2.0: {comparison['etes_v2']:.3f} (Grade: {comparison['etes_grade']})")
    print(f"📈 Improvement: {comparison['improvement']:+.3f}")
    print()
    
    if comparison['recommendations']:
        print("🎯 Recommendations:")
        for rec in comparison['recommendations']:
            print(f"   • {rec}")
        print()
    
    return comparison


def demo_evolution_scenarios():
    """Demonstrate different evolution scenarios"""
    print("🧬 Evolution Scenarios")
    print("=" * 50)
    
    scenarios = [
        {
            'name': 'Poor Test Suite',
            'mutation_score': 0.45,
            'assertion_types': ['equality', 'equality'],
            'behavior_coverage': 0.30,
            'speed_ms': 250,
            'quality_scores': [0.70, 0.65, 0.60, 0.75]
        },
        {
            'name': 'Average Test Suite',
            'mutation_score': 0.72,
            'assertion_types': ['equality', 'type_check', 'boundary'],
            'behavior_coverage': 0.65,
            'speed_ms': 150,
            'quality_scores': [0.85, 0.80, 0.82, 0.88]
        },
        {
            'name': 'Excellent Test Suite',
            'mutation_score': 0.91,
            'assertion_types': ['equality', 'type_check', 'exception', 'boundary', 'invariant', 'property'],
            'behavior_coverage': 0.95,
            'speed_ms': 80,
            'quality_scores': [0.96, 0.94, 0.92, 0.95]
        }
    ]
    
    for scenario in scenarios:
        print(f"📋 {scenario['name']}:")
        
        # Build test data
        assertions = []
        for i, atype in enumerate(scenario['assertion_types']):
            assertions.append({
                'type': atype,
                'code': f'assert {atype}_test()',
                'target_criticality': 2.0 if atype == 'invariant' else 1.5 if atype in ['exception', 'boundary'] else 1.0,
                'checks_invariant': atype == 'invariant'
            })
        
        test_data = {
            'mutation_score': scenario['mutation_score'],
            'avg_test_execution_time_ms': scenario['speed_ms'],
            'assertions': assertions,
            'covered_behaviors': ['feature_a', 'feature_b'][:int(scenario['behavior_coverage'] * 3)],
            'execution_results': [{'passed': True, 'execution_time_ms': scenario['speed_ms']}],
            'determinism_score': scenario['quality_scores'][0],
            'stability_score': scenario['quality_scores'][1],
            'readability_score': scenario['quality_scores'][2],
            'independence_score': scenario['quality_scores'][3],
        }
        
        codebase_data = {
            'all_behaviors': ['feature_a', 'feature_b', 'feature_c'],
            'behavior_criticality': {'feature_a': 2.0, 'feature_b': 2.5, 'feature_c': 1.5},
            'complexity_metrics': {'avg_cyclomatic_complexity': 3.5, 'total_loc': 2000}
        }
        
        # Calculate E-TES
        etes_score, components = calculate_etes_v2(test_data, codebase_data)
        
        print(f"   Score: {etes_score:.3f}")
        print(f"   Key Strengths: ", end="")
        strengths = []
        if components.mutation_score > 0.8:
            strengths.append("High mutation killing")
        if components.assertion_iq > 0.7:
            strengths.append("Intelligent assertions")
        if components.behavior_coverage > 0.8:
            strengths.append("Good behavior coverage")
        if components.speed_factor > 0.8:
            strengths.append("Fast execution")
        if components.quality_factor > 0.9:
            strengths.append("High quality")
        
        print(", ".join(strengths) if strengths else "Needs improvement")
        
        if components.insights:
            print(f"   Top Insight: {components.insights[0]}")
        print()


def demo_configuration():
    """Demonstrate E-TES configuration options"""
    print("⚙️  E-TES Configuration Demo")
    print("=" * 50)
    
    # Custom configuration
    config = ETESConfig(
        max_generations=15,
        population_size=150,
        min_mutation_score=0.85,
        min_behavior_coverage=0.95,
        max_test_runtime_ms=100.0,
        weights={
            'mutation_score': 0.30,
            'evolution_gain': 0.10,
            'assertion_iq': 0.25,
            'behavior_coverage': 0.25,
            'speed_factor': 0.05,
            'quality_factor': 0.05
        }
    )
    
    print("🔧 Custom Configuration:")
    print(f"   • Max Generations: {config.max_generations}")
    print(f"   • Population Size: {config.population_size}")
    print(f"   • Min Mutation Score: {config.min_mutation_score}")
    print(f"   • Min Behavior Coverage: {config.min_behavior_coverage}")
    print(f"   • Max Test Runtime: {config.max_test_runtime_ms}ms")
    print("   • Component Weights:")
    for component, weight in config.weights.items():
        print(f"     - {component}: {weight:.2f}")
    print()
    
    # Test with custom config
    calculator = ETESCalculator(config)
    
    test_data = {
        'mutation_score': 0.88,
        'avg_test_execution_time_ms': 85,
        'assertions': [
            {'type': 'invariant', 'code': 'assert invariant()', 'target_criticality': 2.0, 'checks_invariant': True},
            {'type': 'property', 'code': 'assert property()', 'target_criticality': 1.8},
        ],
        'covered_behaviors': ['critical_feature'],
        'execution_results': [{'passed': True, 'execution_time_ms': 85}],
        'determinism_score': 0.97,
        'stability_score': 0.95,
        'readability_score': 0.93,
        'independence_score': 0.96,
    }
    
    codebase_data = {
        'all_behaviors': ['critical_feature'],
        'behavior_criticality': {'critical_feature': 3.0},
        'complexity_metrics': {'avg_cyclomatic_complexity': 2.5, 'total_loc': 1200}
    }
    
    score, components = calculator.calculate_etes(test_data, codebase_data)
    
    print(f"📊 Result with Custom Config:")
    print(f"   • E-TES Score: {score:.3f}")
    print(f"   • Meets Mutation Threshold: {'✅' if components.mutation_score >= config.min_mutation_score else '❌'}")
    print(f"   • Meets Behavior Threshold: {'✅' if components.behavior_coverage >= config.min_behavior_coverage else '❌'}")
    print()


def main():
    """Run the complete E-TES v2.0 demonstration"""
    print("🚀 E-TES v2.0: Evolutionary Test Effectiveness Score")
    print("🧬 Advanced Test Quality Assessment with Evolutionary Intelligence")
    print("=" * 70)
    print()
    
    try:
        # Run demonstrations
        etes_score, components = demo_basic_etes()
        print()
        
        comparison = demo_comparison()
        print()
        
        demo_evolution_scenarios()
        print()
        
        demo_configuration()
        print()
        
        # Summary
        print("🎉 E-TES v2.0 Demonstration Complete!")
        print("=" * 70)
        print("✨ Key Features Demonstrated:")
        print("   • Multi-dimensional quality assessment")
        print("   • Evolutionary improvement tracking")
        print("   • Intelligent assertion analysis")
        print("   • Behavior coverage optimization")
        print("   • Quality factor evaluation")
        print("   • Configurable scoring weights")
        print("   • Actionable insights generation")
        print()
        print("🔬 Ready for production use!")
        print("📚 See ETES_V2_README.md for detailed documentation")
        
        return True
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

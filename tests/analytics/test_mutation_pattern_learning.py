"""
TDD Tests for Mutation Survival Pattern Learning.

This module tests the MutationPatternLearner class which analyzes
survived mutations to identify patterns and suggest test improvements.

Author: DarkLightX/Dana Edwards
"""

import ast
import json
import pytest
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from unittest.mock import Mock, patch, MagicMock

# Import the module we're testing
from guardian.analytics.mutation_pattern_learning import (
    MutationPatternLearner,
    MutationPattern,
    PatternCategory,
    TestImprovement,
    ImprovementTemplate,
    ImprovementSuggestion,
    PatternReport
)


class TestMutationPatternExtraction:
    """Test pattern extraction from survived mutations."""
    
    def test_extract_ast_pattern_from_survived_mutation(self):
        """Test extracting AST pattern from a survived mutation."""
        # Arrange
        learner = MutationPatternLearner()
        survived_mutation = {
            "id": "mut_001",
            "file_path": "/path/to/file.py",
            "line_number": 42,
            "original_code": "if x > 0:",
            "mutated_code": "if x >= 0:",
            "mutation_type": "boundary",
            "survived": True,
            "context": {
                "function_name": "validate_input",
                "class_name": "InputValidator",
                "complexity": 5
            }
        }
        
        # Act
        pattern = learner.extract_pattern(survived_mutation)
        
        # Assert
        assert pattern is not None
        assert pattern.ast_type == "Compare"
        assert pattern.operator_type == "Gt"
        assert pattern.context_type == "boundary_check"
        assert pattern.frequency == 1
        assert pattern.survival_rate == 1.0
    
    def test_extract_pattern_with_complex_ast_structure(self):
        """Test pattern extraction from complex AST structures."""
        # Arrange
        learner = MutationPatternLearner()
        survived_mutation = {
            "id": "mut_002",
            "original_code": "result = value if value is not None else default",
            "mutated_code": "result = value if value is None else default",
            "mutation_type": "null_check",
            "survived": True
        }
        
        # Act
        pattern = learner.extract_pattern(survived_mutation)
        
        # Assert
        assert pattern.ast_type == "IfExp"
        assert pattern.includes_null_check is True
        assert pattern.pattern_signature == "IfExp:Compare:IsNot:None"
    
    def test_extract_pattern_from_exception_handling(self):
        """Test pattern extraction from exception handling mutations."""
        # Arrange
        learner = MutationPatternLearner()
        survived_mutation = {
            "id": "mut_003",
            "original_code": "except ValueError:",
            "mutated_code": "except Exception:",
            "mutation_type": "exception",
            "survived": True,
            "context": {
                "in_try_except": True,
                "exception_types": ["ValueError"]
            }
        }
        
        # Act
        pattern = learner.extract_pattern(survived_mutation)
        
        # Assert
        assert pattern.ast_type == "ExceptHandler"
        assert pattern.context_type == "exception_handling"
        assert "ValueError" in pattern.metadata["original_exceptions"]


class TestPatternFrequencyAnalysis:
    """Test frequency analysis of survival patterns."""
    
    def test_analyze_pattern_frequency_single_pattern(self):
        """Test frequency analysis with a single pattern type."""
        # Arrange
        learner = MutationPatternLearner()
        mutations = [
            {
                "id": f"mut_{i}",
                "original_code": f"if x > {i}:",
                "mutated_code": f"if x >= {i}:",
                "mutation_type": "boundary",
                "survived": True
            }
            for i in range(5)
        ]
        
        # Act
        frequency_map = learner.analyze_frequencies(mutations)
        
        # Assert
        assert len(frequency_map) == 1
        boundary_pattern = next(iter(frequency_map.values()))
        assert boundary_pattern.frequency == 5
        assert boundary_pattern.pattern_type == "boundary_comparison"
    
    def test_analyze_pattern_frequency_multiple_patterns(self):
        """Test frequency analysis with multiple pattern types."""
        # Arrange
        learner = MutationPatternLearner()
        mutations = [
            # Boundary mutations
            {"id": "mut_1", "original_code": "if x > 0:", "mutated_code": "if x >= 0:", "survived": True},
            {"id": "mut_2", "original_code": "if y < 10:", "mutated_code": "if y <= 10:", "survived": True},
            # Null check mutations
            {"id": "mut_3", "original_code": "if obj is None:", "mutated_code": "if obj is not None:", "survived": True},
            {"id": "mut_4", "original_code": "if val is not None:", "mutated_code": "if val is None:", "survived": True},
            # Arithmetic mutations
            {"id": "mut_5", "original_code": "x + y", "mutated_code": "x - y", "survived": False}
        ]
        
        # Act
        frequency_map = learner.analyze_frequencies(mutations, include_killed=True)
        
        # Assert
        assert len(frequency_map) >= 2  # At least boundary and null check patterns
        assert any(p.pattern_type == "boundary_comparison" for p in frequency_map.values())
        assert any(p.pattern_type == "null_check" for p in frequency_map.values())
    
    def test_calculate_survival_rate_by_pattern(self):
        """Test calculation of survival rates for different patterns."""
        # Arrange
        learner = MutationPatternLearner()
        mutations = [
            # Boundary mutations: 3 survived, 1 killed
            {"id": "mut_1", "pattern": "boundary", "survived": True},
            {"id": "mut_2", "pattern": "boundary", "survived": True},
            {"id": "mut_3", "pattern": "boundary", "survived": True},
            {"id": "mut_4", "pattern": "boundary", "survived": False},
            # Null checks: 1 survived, 2 killed
            {"id": "mut_5", "pattern": "null_check", "survived": True},
            {"id": "mut_6", "pattern": "null_check", "survived": False},
            {"id": "mut_7", "pattern": "null_check", "survived": False},
        ]
        
        # Act
        survival_rates = learner.calculate_survival_rates(mutations)
        
        # Assert
        assert survival_rates["boundary"] == 0.75  # 3/4
        assert survival_rates["null_check"] == 0.33  # 1/3 (approximately)


class TestPatternClassification:
    """Test classification of mutation patterns."""
    
    def test_classify_boundary_pattern(self):
        """Test classification of boundary condition patterns."""
        # Arrange
        learner = MutationPatternLearner()
        pattern = MutationPattern(
            ast_type="Compare",
            operator_type="Gt",
            pattern_signature="Compare:Gt:Num",
            frequency=10
        )
        
        # Act
        category = learner.classify_pattern(pattern)
        
        # Assert
        assert category == PatternCategory.BOUNDARY
        assert pattern.risk_level == "high"
    
    def test_classify_error_handling_pattern(self):
        """Test classification of error handling patterns."""
        # Arrange
        learner = MutationPatternLearner()
        pattern = MutationPattern(
            ast_type="ExceptHandler",
            context_type="exception_handling",
            pattern_signature="Try:ExceptHandler:ValueError"
        )
        
        # Act
        category = learner.classify_pattern(pattern)
        
        # Assert
        assert category == PatternCategory.ERROR_HANDLING
        assert pattern.importance == "critical"
    
    def test_classify_null_check_pattern(self):
        """Test classification of null/None check patterns."""
        # Arrange
        learner = MutationPatternLearner()
        pattern = MutationPattern(
            ast_type="Compare",
            operator_type="Is",
            right_operand="None",
            pattern_signature="Compare:Is:None"
        )
        
        # Act
        category = learner.classify_pattern(pattern)
        
        # Assert
        assert category == PatternCategory.NULL_CHECK
    
    def test_classify_complex_patterns(self):
        """Test classification of complex multi-condition patterns."""
        # Arrange
        learner = MutationPatternLearner()
        pattern = MutationPattern(
            ast_type="BoolOp",
            operator_type="And",
            sub_patterns=["Compare:Gt", "Compare:Lt"],
            pattern_signature="BoolOp:And:[Compare:Gt,Compare:Lt]"
        )
        
        # Act
        category = learner.classify_pattern(pattern)
        
        # Assert
        assert category == PatternCategory.COMPLEX_CONDITION


class TestImpactWeighting:
    """Test impact weighting based on code criticality."""
    
    def test_weight_pattern_by_code_location(self):
        """Test weighting patterns based on their location in code."""
        # Arrange
        learner = MutationPatternLearner()
        pattern = MutationPattern(
            pattern_signature="Compare:Gt:Num",
            locations=[
                {"file": "auth.py", "function": "validate_token", "complexity": 8},
                {"file": "utils.py", "function": "format_string", "complexity": 2},
                {"file": "security.py", "function": "check_permissions", "complexity": 10}
            ]
        )
        
        # Act
        impact_score = learner.calculate_impact_weight(pattern)
        
        # Assert
        assert impact_score > 0.7  # High impact due to security-critical locations
        assert pattern.critical_locations == ["security.py", "auth.py"]
    
    def test_weight_pattern_by_frequency_and_survival_rate(self):
        """Test weighting based on frequency and survival rate."""
        # Arrange
        learner = MutationPatternLearner()
        pattern = MutationPattern(
            frequency=50,
            survival_rate=0.9,
            total_mutations=55
        )
        
        # Act
        impact_score = learner.calculate_impact_weight(pattern)
        
        # Assert
        assert impact_score > 0.8  # High frequency + high survival = high impact
    
    def test_weight_pattern_by_complexity(self):
        """Test weighting based on code complexity metrics."""
        # Arrange
        learner = MutationPatternLearner()
        pattern = MutationPattern(
            average_complexity=12,  # High cyclomatic complexity
            max_nesting_depth=5,
            involves_loops=True
        )
        
        # Act
        impact_score = learner.calculate_impact_weight(pattern)
        
        # Assert
        assert impact_score > 0.6  # Complex code patterns have higher impact


class TestImprovementSuggestionGeneration:
    """Test generation of test improvement suggestions."""
    
    def test_generate_boundary_test_suggestion(self):
        """Test generating suggestions for boundary condition testing."""
        # Arrange
        learner = MutationPatternLearner()
        pattern = MutationPattern(
            category=PatternCategory.BOUNDARY,
            ast_type="Compare",
            operator_type="Gt",
            example_mutations=[
                {"original": "x > 0", "mutated": "x >= 0"},
                {"original": "y < 100", "mutated": "y <= 100"}
            ]
        )
        
        # Act
        suggestions = learner.generate_improvement_suggestions(pattern)
        
        # Assert
        assert len(suggestions) > 0
        assert any("boundary" in s.description.lower() for s in suggestions)
        assert any("edge case" in s.description.lower() for s in suggestions)
        assert suggestions[0].test_template is not None
        assert suggestions[0].assertion_patterns is not None
    
    def test_generate_null_check_test_suggestion(self):
        """Test generating suggestions for null/None check testing."""
        # Arrange
        learner = MutationPatternLearner()
        pattern = MutationPattern(
            category=PatternCategory.NULL_CHECK,
            example_mutations=[
                {"original": "if obj is None:", "mutated": "if obj is not None:"},
                {"original": "val or default", "mutated": "val and default"}
            ]
        )
        
        # Act
        suggestions = learner.generate_improvement_suggestions(pattern)
        
        # Assert
        assert any("None" in s.description for s in suggestions)
        assert any("null" in s.description.lower() for s in suggestions)
        assert any(s.priority == "high" for s in suggestions)
    
    def test_generate_exception_handling_suggestion(self):
        """Test generating suggestions for exception handling."""
        # Arrange
        learner = MutationPatternLearner()
        pattern = MutationPattern(
            category=PatternCategory.ERROR_HANDLING,
            example_mutations=[
                {"original": "except ValueError:", "mutated": "except Exception:"},
                {"original": "raise ValueError(msg)", "mutated": "pass"}
            ]
        )
        
        # Act
        suggestions = learner.generate_improvement_suggestions(pattern)
        
        # Assert
        assert any("exception" in s.description.lower() for s in suggestions)
        assert any("error" in s.description.lower() for s in suggestions)
        assert any(s.test_template.includes_exception_testing for s in suggestions)


class TestHistoricalDataLearning:
    """Test learning from historical mutation data."""
    
    def test_learn_from_historical_mutations(self):
        """Test learning patterns from historical mutation results."""
        # Arrange
        learner = MutationPatternLearner()
        historical_data = {
            "project_id": "test_project",
            "mutations": [
                {"id": f"mut_{i}", "survived": i % 3 != 0, "pattern": "boundary"}
                for i in range(100)
            ],
            "test_runs": 10
        }
        
        # Act
        learned_patterns = learner.learn_from_history(historical_data)
        
        # Assert
        assert len(learned_patterns) > 0
        assert learned_patterns[0].historical_confidence > 0.5
        assert learned_patterns[0].trend_direction in ["increasing", "decreasing", "stable"]
    
    def test_update_pattern_knowledge_base(self):
        """Test updating the pattern knowledge base with new data."""
        # Arrange
        learner = MutationPatternLearner()
        existing_pattern = MutationPattern(
            pattern_signature="Compare:Gt:Num",
            frequency=10,
            survival_rate=0.7
        )
        new_mutations = [
            {"pattern_signature": "Compare:Gt:Num", "survived": True},
            {"pattern_signature": "Compare:Gt:Num", "survived": False},
            {"pattern_signature": "Compare:Gt:Num", "survived": True}
        ]
        
        # Act
        updated_pattern = learner.update_pattern(existing_pattern, new_mutations)
        
        # Assert
        assert updated_pattern.frequency == 13  # 10 + 3
        assert updated_pattern.survival_rate == 0.69  # (7 + 2) / (10 + 3) = 9/13 = 0.69
    
    def test_identify_emerging_patterns(self):
        """Test identification of emerging mutation patterns."""
        # Arrange
        learner = MutationPatternLearner()
        recent_mutations = [
            {"pattern": "async_await", "survived": True, "timestamp": "2024-01-01"},
            {"pattern": "async_await", "survived": True, "timestamp": "2024-01-02"},
            {"pattern": "async_await", "survived": True, "timestamp": "2024-01-03"},
        ]
        
        # Act
        emerging_patterns = learner.identify_emerging_patterns(recent_mutations)
        
        # Assert
        assert len(emerging_patterns) > 0
        assert emerging_patterns[0].pattern_type == "async_await"
        assert emerging_patterns[0].emergence_confidence > 0.7


class TestPatternConfidenceScoring:
    """Test pattern confidence scoring mechanisms."""
    
    def test_calculate_confidence_score_high_confidence(self):
        """Test confidence scoring for well-established patterns."""
        # Arrange
        learner = MutationPatternLearner()
        pattern = MutationPattern(
            frequency=100,
            survival_rate=0.85,
            occurrence_consistency=0.9,
            sample_size=100
        )
        
        # Act
        confidence_score = learner.calculate_confidence_score(pattern)
        
        # Assert
        assert confidence_score > 0.8
        assert pattern.confidence_level == "high"
    
    def test_calculate_confidence_score_low_sample_size(self):
        """Test confidence scoring with low sample size."""
        # Arrange
        learner = MutationPatternLearner()
        pattern = MutationPattern(
            frequency=3,
            survival_rate=1.0,  # Perfect but low sample
            sample_size=3
        )
        
        # Act
        confidence_score = learner.calculate_confidence_score(pattern)
        
        # Assert
        assert confidence_score < 0.5  # Low confidence due to small sample
        assert pattern.confidence_level == "low"
    
    def test_adjust_confidence_for_code_volatility(self):
        """Test confidence adjustment based on code volatility."""
        # Arrange
        learner = MutationPatternLearner()
        pattern = MutationPattern(
            base_confidence=0.8,
            code_locations=[
                {"file": "stable_module.py", "last_modified": "2023-01-01"},
                {"file": "volatile_module.py", "last_modified": "2024-01-15"}
            ]
        )
        
        # Act
        adjusted_confidence = learner.adjust_confidence_for_volatility(pattern)
        
        # Assert
        assert adjusted_confidence < pattern.base_confidence
        assert pattern.volatility_factor > 0


class TestPatternVisualization:
    """Test pattern visualization and reporting."""
    
    def test_generate_pattern_report(self):
        """Test generating a comprehensive pattern report."""
        # Arrange
        learner = MutationPatternLearner()
        patterns = [
            MutationPattern(
                pattern_type="boundary",
                frequency=50,
                survival_rate=0.8,
                impact_score=0.9
            ),
            MutationPattern(
                pattern_type="null_check",
                frequency=30,
                survival_rate=0.6,
                impact_score=0.7
            )
        ]
        
        # Act
        report = learner.generate_pattern_report(patterns)
        
        # Assert
        assert report.total_patterns == 2
        assert report.high_impact_patterns == 1
        assert report.recommendations is not None
        assert len(report.pattern_summaries) == 2
    
    def test_export_patterns_to_json(self):
        """Test exporting patterns to JSON format."""
        # Arrange
        learner = MutationPatternLearner()
        patterns = [
            MutationPattern(
                pattern_signature="Compare:Gt:Num",
                frequency=10,
                category=PatternCategory.BOUNDARY
            )
        ]
        
        # Act
        json_output = learner.export_patterns_to_json(patterns)
        
        # Assert
        data = json.loads(json_output)
        assert data["patterns"][0]["pattern_signature"] == "Compare:Gt:Num"
        assert data["patterns"][0]["frequency"] == 10
        assert data["metadata"]["version"] is not None


class TestIntegrationWithMutationTesting:
    """Test integration with mutation testing frameworks."""
    
    @patch('guardian.sensors.mutation.get_mutation_score_data')
    def test_analyze_mutmut_results(self, mock_mutmut):
        """Test analyzing results from mutmut."""
        # Arrange
        learner = MutationPatternLearner()
        mock_mutmut.return_value = (75.0, 100, 75)  # score, total, killed
        
        mutation_results = {
            "survived_mutants": [
                {
                    "id": "1",
                    "location": "file.py:10",
                    "mutation": "< to <=",
                    "status": "survived"
                }
            ]
        }
        
        # Act
        patterns = learner.analyze_mutation_results(mutation_results)
        
        # Assert
        assert len(patterns) > 0
        assert patterns[0].source == "mutmut"
    
    def test_prioritize_tests_for_patterns(self):
        """Test prioritizing tests based on pattern analysis."""
        # Arrange
        learner = MutationPatternLearner()
        patterns = [
            MutationPattern(
                pattern_type="boundary",
                impact_score=0.9,
                affected_files=["auth.py", "payment.py"]
            ),
            MutationPattern(
                pattern_type="null_check",
                impact_score=0.5,
                affected_files=["utils.py"]
            )
        ]
        
        # Act
        test_priorities = learner.prioritize_tests_for_patterns(patterns)
        
        # Assert
        assert test_priorities[0].pattern_type == "boundary"
        assert test_priorities[0].suggested_tests[0].priority == "critical"
        assert len(test_priorities[0].affected_modules) == 2


# Test classes are now imported from the actual implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Evolutionary Operators for Test Suite Optimization

Crossover and mutation operators for evolving test suites with
context-aware genetic operations.
"""

import random
import ast
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import re

from .types import TestIndividual

logger = logging.getLogger(__name__)


class CrossoverOperator:
    """
    Smart crossover operator that combines complementary test strengths
    """
    
    def __init__(self):
        self.crossover_strategies = [
            self._assertion_based_crossover,
            self._coverage_based_crossover,
            self._semantic_crossover
        ]
    
    def crossover(self, parent1: TestIndividual, parent2: TestIndividual) -> Tuple[TestIndividual, TestIndividual]:
        """
        Perform intelligent crossover between two test individuals
        
        Args:
            parent1: First parent test
            parent2: Second parent test
            
        Returns:
            Tuple of two offspring tests
        """
        try:
            # Choose crossover strategy based on parent characteristics
            strategy = self._select_crossover_strategy(parent1, parent2)
            return strategy(parent1, parent2)
            
        except Exception as e:
            logger.warning(f"Error in crossover: {e}")
            # Fallback to simple copy
            return self._simple_copy_crossover(parent1, parent2)
    
    def _select_crossover_strategy(self, parent1: TestIndividual, parent2: TestIndividual) -> callable:
        """Select appropriate crossover strategy based on parent characteristics"""
        
        # If parents have different assertion types, use assertion-based crossover
        p1_assertion_types = set(a.get('type', 'equality') for a in parent1.assertions)
        p2_assertion_types = set(a.get('type', 'equality') for a in parent2.assertions)
        
        if p1_assertion_types != p2_assertion_types:
            return self._assertion_based_crossover
        
        # If parents have different coverage strengths, use coverage-based
        p1_coverage = self._estimate_coverage_strength(parent1)
        p2_coverage = self._estimate_coverage_strength(parent2)
        
        if abs(p1_coverage - p2_coverage) > 0.3:
            return self._coverage_based_crossover
        
        # Default to semantic crossover
        return self._semantic_crossover
    
    def _assertion_based_crossover(self, parent1: TestIndividual, parent2: TestIndividual) -> Tuple[TestIndividual, TestIndividual]:
        """Crossover based on assertion complementarity"""
        
        # Identify complementary assertions
        p1_strengths = self._get_assertion_strengths(parent1)
        p2_strengths = self._get_assertion_strengths(parent2)
        
        # Create children by combining non-overlapping strengths
        child1_assertions = []
        child2_assertions = []
        
        # Child 1: P1's unique strengths + P2's complementary assertions
        child1_assertions.extend([a for a in parent1.assertions if a.get('type') in p1_strengths])
        child1_assertions.extend([a for a in parent2.assertions if a.get('type') not in p1_strengths])
        
        # Child 2: P2's unique strengths + P1's complementary assertions  
        child2_assertions.extend([a for a in parent2.assertions if a.get('type') in p2_strengths])
        child2_assertions.extend([a for a in parent1.assertions if a.get('type') not in p2_strengths])
        
        # Combine test code intelligently
        child1_code = self._merge_test_code(parent1.test_code, parent2.test_code, 0.7)
        child2_code = self._merge_test_code(parent2.test_code, parent1.test_code, 0.7)
        
        # Create offspring
        child1 = TestIndividual(
            test_code=child1_code,
            assertions=child1_assertions,
            setup_code=self._merge_setup(parent1.setup_code, parent2.setup_code),
            teardown_code=parent1.teardown_code,  # Inherit from parent1
            parent_ids=[parent1.id, parent2.id]
        )
        
        child2 = TestIndividual(
            test_code=child2_code,
            assertions=child2_assertions,
            setup_code=self._merge_setup(parent2.setup_code, parent1.setup_code),
            teardown_code=parent2.teardown_code,  # Inherit from parent2
            parent_ids=[parent1.id, parent2.id]
        )
        
        return child1, child2
    
    def _coverage_based_crossover(self, parent1: TestIndividual, parent2: TestIndividual) -> Tuple[TestIndividual, TestIndividual]:
        """Crossover based on coverage complementarity"""
        
        # Analyze coverage patterns (simplified)
        p1_patterns = self._extract_coverage_patterns(parent1)
        p2_patterns = self._extract_coverage_patterns(parent2)
        
        # Create hybrid test code
        child1_code = self._create_hybrid_test(parent1, parent2, p1_patterns, p2_patterns)
        child2_code = self._create_hybrid_test(parent2, parent1, p2_patterns, p1_patterns)
        
        # Merge assertions intelligently
        child1_assertions = self._merge_assertions_by_coverage(parent1.assertions, parent2.assertions)
        child2_assertions = self._merge_assertions_by_coverage(parent2.assertions, parent1.assertions)
        
        child1 = TestIndividual(
            test_code=child1_code,
            assertions=child1_assertions,
            setup_code=parent1.setup_code,
            teardown_code=parent1.teardown_code,
            parent_ids=[parent1.id, parent2.id]
        )
        
        child2 = TestIndividual(
            test_code=child2_code,
            assertions=child2_assertions,
            setup_code=parent2.setup_code,
            teardown_code=parent2.teardown_code,
            parent_ids=[parent1.id, parent2.id]
        )
        
        return child1, child2
    
    def _semantic_crossover(self, parent1: TestIndividual, parent2: TestIndividual) -> Tuple[TestIndividual, TestIndividual]:
        """Semantic-aware crossover preserving test logic"""
        
        # Parse test structures
        p1_structure = self._parse_test_structure(parent1.test_code)
        p2_structure = self._parse_test_structure(parent2.test_code)
        
        # Create semantically valid combinations
        child1_structure = self._combine_structures(p1_structure, p2_structure, 0.6)
        child2_structure = self._combine_structures(p2_structure, p1_structure, 0.6)
        
        # Generate code from structures
        child1_code = self._generate_code_from_structure(child1_structure)
        child2_code = self._generate_code_from_structure(child2_structure)
        
        # Combine assertions semantically
        child1_assertions = self._semantic_assertion_merge(parent1.assertions, parent2.assertions)
        child2_assertions = self._semantic_assertion_merge(parent2.assertions, parent1.assertions)
        
        child1 = TestIndividual(
            test_code=child1_code,
            assertions=child1_assertions,
            setup_code=self._optimize_setup(parent1.setup_code, parent2.setup_code),
            teardown_code=parent1.teardown_code,
            parent_ids=[parent1.id, parent2.id]
        )
        
        child2 = TestIndividual(
            test_code=child2_code,
            assertions=child2_assertions,
            setup_code=self._optimize_setup(parent2.setup_code, parent1.setup_code),
            teardown_code=parent2.teardown_code,
            parent_ids=[parent1.id, parent2.id]
        )
        
        return child1, child2
    
    def _simple_copy_crossover(self, parent1: TestIndividual, parent2: TestIndividual) -> Tuple[TestIndividual, TestIndividual]:
        """Fallback simple crossover"""
        # Simple single-point crossover on test code
        p1_lines = parent1.test_code.split('\n')
        p2_lines = parent2.test_code.split('\n')
        
        if len(p1_lines) > 1 and len(p2_lines) > 1:
            crossover_point = random.randint(1, min(len(p1_lines), len(p2_lines)) - 1)
            
            child1_code = '\n'.join(p1_lines[:crossover_point] + p2_lines[crossover_point:])
            child2_code = '\n'.join(p2_lines[:crossover_point] + p1_lines[crossover_point:])
        else:
            child1_code = parent1.test_code
            child2_code = parent2.test_code
        
        child1 = TestIndividual(
            test_code=child1_code,
            assertions=parent1.assertions[:],
            setup_code=parent1.setup_code,
            teardown_code=parent1.teardown_code,
            parent_ids=[parent1.id, parent2.id]
        )
        
        child2 = TestIndividual(
            test_code=child2_code,
            assertions=parent2.assertions[:],
            setup_code=parent2.setup_code,
            teardown_code=parent2.teardown_code,
            parent_ids=[parent1.id, parent2.id]
        )
        
        return child1, child2
    
    def _get_assertion_strengths(self, individual: TestIndividual) -> set:
        """Identify assertion type strengths of an individual"""
        assertion_types = [a.get('type', 'equality') for a in individual.assertions]
        
        # Count assertion types
        type_counts = {}
        for atype in assertion_types:
            type_counts[atype] = type_counts.get(atype, 0) + 1
        
        # Return types that are well-represented (>= 2 instances)
        return {atype for atype, count in type_counts.items() if count >= 2}
    
    def _estimate_coverage_strength(self, individual: TestIndividual) -> float:
        """Estimate coverage strength of an individual"""
        # Simple heuristic based on test complexity
        code_complexity = len(individual.test_code.split('\n'))
        assertion_diversity = len(set(a.get('type', 'equality') for a in individual.assertions))
        
        return min((code_complexity * assertion_diversity) / 20.0, 1.0)
    
    def _merge_test_code(self, primary_code: str, secondary_code: str, primary_weight: float) -> str:
        """Merge test code from two parents"""
        # Simple line-based merging (in practice, would use AST)
        primary_lines = primary_code.split('\n')
        secondary_lines = secondary_code.split('\n')
        
        merged_lines = []
        max_lines = max(len(primary_lines), len(secondary_lines))
        
        for i in range(max_lines):
            if random.random() < primary_weight:
                if i < len(primary_lines):
                    merged_lines.append(primary_lines[i])
            else:
                if i < len(secondary_lines):
                    merged_lines.append(secondary_lines[i])
        
        return '\n'.join(merged_lines)
    
    def _merge_setup(self, setup1: str, setup2: str) -> str:
        """Merge setup code from two parents"""
        if not setup1:
            return setup2
        if not setup2:
            return setup1
        
        # Combine unique setup statements
        setup1_lines = set(line.strip() for line in setup1.split('\n') if line.strip())
        setup2_lines = set(line.strip() for line in setup2.split('\n') if line.strip())
        
        combined_lines = setup1_lines.union(setup2_lines)
        return '\n'.join(sorted(combined_lines))
    
    def _extract_coverage_patterns(self, individual: TestIndividual) -> List[str]:
        """Extract coverage patterns from test code"""
        # Simple pattern extraction (placeholder)
        patterns = []
        
        if 'for ' in individual.test_code:
            patterns.append('loop_coverage')
        if 'if ' in individual.test_code:
            patterns.append('branch_coverage')
        if 'try:' in individual.test_code:
            patterns.append('exception_coverage')
        if any(a.get('type') == 'boundary' for a in individual.assertions):
            patterns.append('boundary_coverage')
        
        return patterns
    
    def _create_hybrid_test(self, primary: TestIndividual, secondary: TestIndividual,
                          primary_patterns: List[str], secondary_patterns: List[str]) -> str:
        """Create hybrid test combining coverage patterns"""
        # Start with primary test
        hybrid_code = primary.test_code
        
        # Add missing patterns from secondary
        missing_patterns = set(secondary_patterns) - set(primary_patterns)
        
        for pattern in missing_patterns:
            if pattern == 'exception_coverage' and 'try:' not in hybrid_code:
                # Add exception handling
                hybrid_code += "\n    try:\n        # Exception test\n        pass\n    except Exception:\n        pass"
            elif pattern == 'boundary_coverage':
                # Add boundary test
                hybrid_code += "\n    # Boundary test\n    assert len(data) >= 0"
        
        return hybrid_code
    
    def _merge_assertions_by_coverage(self, primary_assertions: List[Dict], 
                                    secondary_assertions: List[Dict]) -> List[Dict]:
        """Merge assertions based on coverage complementarity"""
        merged = primary_assertions[:]
        
        # Add unique assertion types from secondary
        primary_types = set(a.get('type', 'equality') for a in primary_assertions)
        
        for assertion in secondary_assertions:
            if assertion.get('type', 'equality') not in primary_types:
                merged.append(assertion)
                primary_types.add(assertion.get('type', 'equality'))
        
        return merged
    
    def _parse_test_structure(self, test_code: str) -> Dict[str, Any]:
        """Parse test code into structural components"""
        structure = {
            'setup': [],
            'actions': [],
            'assertions': [],
            'cleanup': []
        }
        
        lines = test_code.split('\n')
        current_section = 'setup'
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if 'assert' in line:
                current_section = 'assertions'
                structure[current_section].append(line)
            elif any(keyword in line for keyword in ['def ', 'class ', 'import ']):
                structure['setup'].append(line)
            else:
                if current_section == 'assertions':
                    current_section = 'cleanup'
                elif current_section == 'setup':
                    current_section = 'actions'
                structure[current_section].append(line)
        
        return structure
    
    def _combine_structures(self, primary: Dict[str, Any], secondary: Dict[str, Any], 
                          primary_weight: float) -> Dict[str, Any]:
        """Combine test structures intelligently"""
        combined = {
            'setup': [],
            'actions': [],
            'assertions': [],
            'cleanup': []
        }
        
        for section in combined.keys():
            # Combine sections with weighted selection
            primary_items = primary.get(section, [])
            secondary_items = secondary.get(section, [])
            
            # Take all setup items (they're usually necessary)
            if section == 'setup':
                combined[section] = list(set(primary_items + secondary_items))
            else:
                # Weighted selection for other sections
                all_items = primary_items + secondary_items
                for item in all_items:
                    if item in primary_items and random.random() < primary_weight:
                        combined[section].append(item)
                    elif item in secondary_items and random.random() < (1 - primary_weight):
                        combined[section].append(item)
        
        return combined
    
    def _generate_code_from_structure(self, structure: Dict[str, Any]) -> str:
        """Generate test code from structural components"""
        code_lines = []
        
        # Add sections in order
        for section in ['setup', 'actions', 'assertions', 'cleanup']:
            if structure.get(section):
                code_lines.extend(structure[section])
        
        return '\n'.join(code_lines)
    
    def _semantic_assertion_merge(self, primary_assertions: List[Dict], 
                                secondary_assertions: List[Dict]) -> List[Dict]:
        """Merge assertions while preserving semantic validity"""
        merged = []
        
        # Group assertions by target/subject
        primary_targets = {}
        for assertion in primary_assertions:
            target = assertion.get('target', 'default')
            if target not in primary_targets:
                primary_targets[target] = []
            primary_targets[target].append(assertion)
        
        # Add primary assertions
        merged.extend(primary_assertions)
        
        # Add complementary secondary assertions
        for assertion in secondary_assertions:
            target = assertion.get('target', 'default')
            assertion_type = assertion.get('type', 'equality')
            
            # Check if this type of assertion already exists for this target
            existing_types = set(a.get('type', 'equality') for a in primary_targets.get(target, []))
            
            if assertion_type not in existing_types:
                merged.append(assertion)
        
        return merged
    
    def _optimize_setup(self, setup1: str, setup2: str) -> str:
        """Optimize combined setup code"""
        if not setup1 and not setup2:
            return ""
        
        # Combine and deduplicate setup statements
        all_setup = (setup1 + '\n' + setup2).split('\n')
        unique_setup = []
        seen = set()
        
        for line in all_setup:
            line = line.strip()
            if line and line not in seen:
                unique_setup.append(line)
                seen.add(line)
        
        return '\n'.join(unique_setup)


class MutationOperator:
    """
    Guided mutation operator for test enhancement
    """
    
    def __init__(self):
        self.mutation_strategies = [
            self._add_assertion_mutation,
            self._modify_assertion_mutation,
            self._add_boundary_test_mutation,
            self._add_exception_test_mutation,
            self._enhance_coverage_mutation
        ]
    
    def mutate(self, individual: TestIndividual, mutation_rate: float) -> TestIndividual:
        """
        Apply guided mutations to enhance test effectiveness
        
        Args:
            individual: Test individual to mutate
            mutation_rate: Probability of applying mutations
            
        Returns:
            Mutated test individual
        """
        if random.random() > mutation_rate:
            return individual  # No mutation
        
        try:
            # Choose mutation strategy based on test weaknesses
            strategy = self._select_mutation_strategy(individual)
            return strategy(individual)
            
        except Exception as e:
            logger.warning(f"Error in mutation: {e}")
            return individual  # Return original if mutation fails
    
    def _select_mutation_strategy(self, individual: TestIndividual) -> callable:
        """Select mutation strategy based on test analysis"""
        
        # Analyze test weaknesses
        assertion_types = set(a.get('type', 'equality') for a in individual.assertions)
        
        # If lacking boundary tests, add them
        if 'boundary' not in assertion_types:
            return self._add_boundary_test_mutation
        
        # If lacking exception tests, add them
        if 'exception' not in assertion_types and 'try:' not in individual.test_code:
            return self._add_exception_test_mutation
        
        # If few assertions, add more
        if len(individual.assertions) < 3:
            return self._add_assertion_mutation
        
        # Default to coverage enhancement
        return self._enhance_coverage_mutation
    
    def _add_assertion_mutation(self, individual: TestIndividual) -> TestIndividual:
        """Add new assertion to test"""
        new_assertion = {
            'type': random.choice(['equality', 'type_check', 'property']),
            'code': 'assert True  # Generated assertion',
            'target_criticality': 1.0
        }
        
        new_assertions = individual.assertions + [new_assertion]
        new_code = individual.test_code + '\n    assert True  # Generated assertion'
        
        mutated = TestIndividual(
            test_code=new_code,
            assertions=new_assertions,
            setup_code=individual.setup_code,
            teardown_code=individual.teardown_code,
            parent_ids=[individual.id]
        )
        mutated.mutation_history = individual.mutation_history + ['add_assertion']
        
        return mutated
    
    def _modify_assertion_mutation(self, individual: TestIndividual) -> TestIndividual:
        """Modify existing assertion"""
        if not individual.assertions:
            return individual
        
        # Select random assertion to modify
        assertion_idx = random.randint(0, len(individual.assertions) - 1)
        new_assertions = individual.assertions[:]
        
        # Modify assertion type
        assertion_types = ['equality', 'inequality', 'type_check', 'exception', 'boundary']
        new_type = random.choice(assertion_types)
        new_assertions[assertion_idx] = {
            **new_assertions[assertion_idx],
            'type': new_type
        }
        
        mutated = TestIndividual(
            test_code=individual.test_code,
            assertions=new_assertions,
            setup_code=individual.setup_code,
            teardown_code=individual.teardown_code,
            parent_ids=[individual.id]
        )
        mutated.mutation_history = individual.mutation_history + ['modify_assertion']
        
        return mutated
    
    def _add_boundary_test_mutation(self, individual: TestIndividual) -> TestIndividual:
        """Add boundary value test"""
        boundary_assertion = {
            'type': 'boundary',
            'code': 'assert len(data) >= 0  # Boundary test',
            'target_criticality': 1.5
        }
        
        new_assertions = individual.assertions + [boundary_assertion]
        new_code = individual.test_code + '\n    # Boundary test\n    assert len(data) >= 0'
        
        mutated = TestIndividual(
            test_code=new_code,
            assertions=new_assertions,
            setup_code=individual.setup_code,
            teardown_code=individual.teardown_code,
            parent_ids=[individual.id]
        )
        mutated.mutation_history = individual.mutation_history + ['add_boundary_test']
        
        return mutated
    
    def _add_exception_test_mutation(self, individual: TestIndividual) -> TestIndividual:
        """Add exception handling test"""
        exception_assertion = {
            'type': 'exception',
            'code': 'with pytest.raises(ValueError):',
            'target_criticality': 2.0
        }
        
        new_assertions = individual.assertions + [exception_assertion]
        new_code = individual.test_code + '\n    # Exception test\n    with pytest.raises(ValueError):\n        invalid_operation()'
        
        mutated = TestIndividual(
            test_code=new_code,
            assertions=new_assertions,
            setup_code=individual.setup_code,
            teardown_code=individual.teardown_code,
            parent_ids=[individual.id]
        )
        mutated.mutation_history = individual.mutation_history + ['add_exception_test']
        
        return mutated
    
    def _enhance_coverage_mutation(self, individual: TestIndividual) -> TestIndividual:
        """Enhance test coverage"""
        # Add loop or conditional coverage
        coverage_enhancements = [
            '\n    # Enhanced coverage\n    for i in range(3):\n        assert process_item(i) is not None',
            '\n    # Conditional coverage\n    if data:\n        assert validate(data)\n    else:\n        assert True'
        ]
        
        enhancement = random.choice(coverage_enhancements)
        new_code = individual.test_code + enhancement
        
        mutated = TestIndividual(
            test_code=new_code,
            assertions=individual.assertions,
            setup_code=individual.setup_code,
            teardown_code=individual.teardown_code,
            parent_ids=[individual.id]
        )
        mutated.mutation_history = individual.mutation_history + ['enhance_coverage']
        
        return mutated

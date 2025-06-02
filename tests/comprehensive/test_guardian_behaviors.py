"""
Behavior-Driven Tests for Guardian AI Tool
Designed to improve behavior coverage from F to A grade

This test suite implements comprehensive behavior coverage for:
- User story validation
- Critical path testing
- End-to-end workflows
- Error scenario handling
- Performance requirements
"""

import pytest
import time
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add guardian to path
guardian_path = os.path.join(os.path.dirname(__file__), '..', '..', 'guardian_ai_tool', 'guardian')
sys.path.insert(0, guardian_path)

from guardian.cli.analyzer import ProjectAnalyzer
from guardian.core.tes import calculate_etes_v2 # calculate_tes is removed


class TestGuardianUserStories:
    """Test Guardian behaviors from user story perspective"""
    
    def test_should_analyze_project_quality_when_developer_runs_basic_analysis(self):
        """
        User Story: As a developer, I want to analyze my project's code quality
        so that I can identify areas for improvement.
        
        Behavior: Basic project analysis workflow
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange: Create sample project
            project_path = Path(temp_dir)
            
            # Create main module
            main_file = project_path / 'main.py'
            main_file.write_text('''
def calculate_total(items):
    """Calculate total of numeric items"""
    total = 0
    for item in items:
        if isinstance(item, (int, float)):
            total += item
    return total

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def add_item(self, item):
        self.data.append(item)
    
    def process(self):
        return calculate_total(self.data)
''')
            
            # Create test file
            test_file = project_path / 'test_main.py'
            test_file.write_text('''
import pytest
from main import calculate_total, DataProcessor

def test_calculate_total_with_numbers():
    assert calculate_total([1, 2, 3]) == 6

def test_calculate_total_with_mixed_types():
    assert calculate_total([1, "hello", 2.5]) == 3.5

def test_data_processor():
    processor = DataProcessor()
    processor.add_item(10)
    processor.add_item(20)
    assert processor.process() == 30
''')
            
            # Act: Analyze project
            analyzer = ProjectAnalyzer()
            results = analyzer.analyze_project(str(project_path))
            
            # Assert: Should provide comprehensive analysis
            assert isinstance(results, dict)
            assert 'status' in results
            assert 'metrics' in results
            assert 'tes_score' in results
            
            # Should analyze Python files
            metrics = results['metrics']
            assert metrics['python_files_analyzed'] >= 1
            assert metrics['total_lines_of_code_python'] > 0
            
            # Should calculate quality scores
            tes_score = results['tes_score']
            assert isinstance(tes_score, float)
            assert 0.0 <= tes_score <= 1.0
    
    def test_should_detect_security_issues_when_security_analyst_runs_security_scan(self):
        """
        User Story: As a security analyst, I want to detect potential security vulnerabilities
        so that I can ensure the codebase is secure.
        
        Behavior: Security analysis workflow
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange: Create project with security issues
            project_path = Path(temp_dir)
            
            # Create file with security issues
            security_file = project_path / 'security_issues.py'
            security_file.write_text('''
import os

# Hardcoded secret (security issue)
API_KEY = "sk-1234567890abcdef"
DATABASE_PASSWORD = "admin123"

def execute_command(user_input):
    """Dangerous function using eval"""
    # Security issue: eval usage
    result = eval(user_input)
    return result

def get_user_data(user_id):
    """Function with potential SQL injection"""
    # Security issue: string formatting in SQL
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return query

class ConfigManager:
    def __init__(self):
        # Security issue: hardcoded credentials
        self.admin_token = "admin_secret_token_123"
    
    def authenticate(self, token):
        return token == self.admin_token
''')
            
            # Act: Analyze for security issues
            analyzer = ProjectAnalyzer({'include_security_analysis': True})
            results = analyzer.analyze_project(str(project_path))
            
            # Assert: Should detect security issues
            assert isinstance(results, dict)
            
            security_analysis = results.get('security_analysis', {})
            assert isinstance(security_analysis, dict)
            
            # Should detect eval usage
            eval_count = security_analysis.get('eval_usage_count', 0)
            assert eval_count > 0, "Should detect eval() usage"
            
            # Should detect hardcoded secrets
            secrets_count = security_analysis.get('hardcoded_secrets_count', 0)
            assert secrets_count > 0, "Should detect hardcoded secrets"
    
    def test_should_provide_actionable_recommendations_when_team_lead_reviews_quality_report(self):
        """
        User Story: As a team lead, I want actionable recommendations for code improvement
        so that I can guide my team's development efforts.
        
        Behavior: Quality reporting and recommendations
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange: Create project with various quality issues
            project_path = Path(temp_dir)
            
            # Create file with quality issues
            quality_file = project_path / 'quality_issues.py'
            quality_file.write_text('''
import unused_module
import json
import os

class LargeClass:
    """A class with too many methods"""
    
    def method1(self): pass
    def method2(self): pass
    def method3(self): pass
    def method4(self): pass
    def method5(self): pass
    def method6(self): pass
    def method7(self): pass
    def method8(self): pass
    def method9(self): pass
    def method10(self): pass
    def method11(self): pass
    def method12(self): pass

def very_long_function():
    """A function that is too long"""
    line1 = 1
    line2 = 2
    line3 = 3
    line4 = 4
    line5 = 5
    line6 = 6
    line7 = 7
    line8 = 8
    line9 = 9
    line10 = 10
    line11 = 11
    line12 = 12
    line13 = 13
    line14 = 14
    line15 = 15
    line16 = 16
    line17 = 17
    line18 = 18
    line19 = 19
    line20 = 20
    line21 = 21
    line22 = 22
    line23 = 23
    line24 = 24
    line25 = 25
    return line25

def complex_function(a, b, c, d, e):
    """A function with high cyclomatic complexity"""
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        return a + b + c + d + e
                    else:
                        return a + b + c + d
                else:
                    return a + b + c
            else:
                return a + b
        else:
            return a
    else:
        return 0
''')
            
            # Act: Analyze for quality issues
            analyzer = ProjectAnalyzer()
            results = analyzer.analyze_project(str(project_path))
            
            # Assert: Should provide actionable insights
            assert isinstance(results, dict)
            
            metrics = results['metrics']
            
            # Should detect long functions
            long_functions = metrics.get('long_functions_count', 0)
            assert long_functions > 0, "Should detect long functions"
            
            # Should detect large classes
            large_classes = metrics.get('large_classes_count', 0)
            assert large_classes > 0, "Should detect large classes"
            
            # Should detect unused imports
            unused_imports = metrics.get('unused_imports_count', 0)
            assert unused_imports > 0, "Should detect unused imports"
            
            # Should calculate complexity
            complexity = metrics.get('average_cyclomatic_complexity', 0)
            assert complexity > 0, "Should calculate cyclomatic complexity"


class TestGuardianCriticalPaths:
    """Test Guardian critical path behaviors"""
    
    def test_should_handle_empty_project_gracefully_when_analyzing_empty_directory(self):
        """
        Critical Path: Empty project analysis
        Behavior: Graceful handling of edge cases
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange: Empty directory
            empty_project = Path(temp_dir) / 'empty_project'
            empty_project.mkdir()
            
            # Act: Analyze empty project
            analyzer = ProjectAnalyzer()
            results = analyzer.analyze_project(str(empty_project))
            
            # Assert: Should handle gracefully
            assert isinstance(results, dict)
            assert 'status' in results
            assert 'metrics' in results
            
            # Should report zero files
            metrics = results['metrics']
            assert metrics['python_files_analyzed'] == 0
            assert metrics['total_lines_of_code_python'] == 0
    
    def test_should_handle_large_project_efficiently_when_analyzing_many_files(self):
        """
        Critical Path: Large project analysis
        Behavior: Performance and scalability
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange: Create project with many files
            project_path = Path(temp_dir)
            
            # Create multiple Python files
            for i in range(20):
                file_path = project_path / f'module_{i}.py'
                file_path.write_text(f'''
def function_{i}():
    """Function in module {i}"""
    return {i}

class Class_{i}:
    """Class in module {i}"""
    
    def method_{i}(self):
        return function_{i}()
''')
            
            # Act: Analyze large project
            start_time = time.time()
            analyzer = ProjectAnalyzer()
            results = analyzer.analyze_project(str(project_path))
            analysis_time = time.time() - start_time
            
            # Assert: Should handle efficiently
            assert isinstance(results, dict)
            
            # Should analyze all files
            metrics = results['metrics']
            assert metrics['python_files_analyzed'] == 20
            
            # Should complete in reasonable time (performance requirement)
            assert analysis_time < 30.0, f"Analysis took too long: {analysis_time:.2f}s"
    
    def test_should_provide_consistent_results_when_analyzing_same_project_multiple_times(self):
        """
        Critical Path: Analysis consistency
        Behavior: Deterministic results
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange: Create consistent test project
            project_path = Path(temp_dir)
            
            test_file = project_path / 'consistent_module.py'
            test_file.write_text('''
def stable_function(x, y):
    """A stable function for consistency testing"""
    if x > 0 and y > 0:
        return x + y
    elif x > 0:
        return x
    elif y > 0:
        return y
    else:
        return 0

class StableClass:
    """A stable class for consistency testing"""
    
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value
    
    def set_value(self, new_value):
        self.value = new_value
''')
            
            # Act: Analyze multiple times
            analyzer = ProjectAnalyzer()
            
            results1 = analyzer.analyze_project(str(project_path))
            results2 = analyzer.analyze_project(str(project_path))
            results3 = analyzer.analyze_project(str(project_path))
            
            # Assert: Should provide consistent results
            assert isinstance(results1, dict)
            assert isinstance(results2, dict)
            assert isinstance(results3, dict)
            
            # Key metrics should be consistent
            metrics1 = results1['metrics']
            metrics2 = results2['metrics']
            metrics3 = results3['metrics']
            
            assert metrics1['python_files_analyzed'] == metrics2['python_files_analyzed'] == metrics3['python_files_analyzed']
            assert metrics1['total_lines_of_code_python'] == metrics2['total_lines_of_code_python'] == metrics3['total_lines_of_code_python']
            
            # TES scores should be consistent (within small tolerance for floating point)
            tes1 = results1['tes_score']
            tes2 = results2['tes_score']
            tes3 = results3['tes_score']
            
            assert abs(tes1 - tes2) < 0.01, "TES scores should be consistent"
            assert abs(tes2 - tes3) < 0.01, "TES scores should be consistent"


class TestGuardianPerformanceRequirements:
    """Test Guardian performance behavior requirements"""
    
    def test_should_complete_analysis_within_time_limit_when_analyzing_medium_project(self):
        """
        Performance Requirement: Analysis speed
        Behavior: Sub-30 second analysis for medium projects
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange: Create medium-sized project
            project_path = Path(temp_dir)
            
            # Create 10 files with moderate complexity
            for i in range(10):
                file_path = project_path / f'module_{i}.py'
                file_path.write_text(f'''
import json
import os
from typing import List, Dict, Optional

class DataProcessor_{i}:
    """Data processor class {i}"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data = []
        self.processed = False
    
    def load_data(self, source: str) -> bool:
        """Load data from source"""
        try:
            if os.path.exists(source):
                with open(source, 'r') as f:
                    self.data = json.load(f)
                return True
        except Exception as e:
            print(f"Error loading data: {{e}}")
        return False
    
    def process_data(self) -> List[Dict[str, Any]]:
        """Process loaded data"""
        if not self.data:
            return []
        
        results = []
        for item in self.data:
            if isinstance(item, dict):
                processed_item = self._process_item(item)
                if processed_item:
                    results.append(processed_item)
        
        self.processed = True
        return results
    
    def _process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process individual item"""
        if 'id' not in item:
            return None
        
        processed = {{
            'id': item['id'],
            'processed_at': time.time(),
            'processor_id': {i}
        }}
        
        if 'value' in item:
            processed['value'] = item['value'] * 2
        
        return processed

def utility_function_{i}(data: List[Any]) -> Dict[str, Any]:
    """Utility function {i}"""
    if not data:
        return {{'count': 0, 'sum': 0, 'avg': 0}}
    
    numeric_data = [x for x in data if isinstance(x, (int, float))]
    
    if not numeric_data:
        return {{'count': len(data), 'sum': 0, 'avg': 0}}
    
    total = sum(numeric_data)
    return {{
        'count': len(numeric_data),
        'sum': total,
        'avg': total / len(numeric_data)
    }}
''')
            
            # Act: Measure analysis time
            start_time = time.time()
            analyzer = ProjectAnalyzer({'use_etes_v2': True})
            results = analyzer.analyze_project(str(project_path))
            analysis_time = time.time() - start_time
            
            # Assert: Should meet performance requirements
            assert isinstance(results, dict)
            assert analysis_time < 30.0, f"Analysis exceeded time limit: {analysis_time:.2f}s"
            
            # Should still provide quality results
            metrics = results['metrics']
            assert metrics['python_files_analyzed'] == 10
            assert metrics['total_lines_of_code_python'] > 500


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

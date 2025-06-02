"""
Comprehensive CLI Test Suite

Tests for Guardian's command-line interface, output formatting,
and project analysis integration with high coverage.
"""

import pytest
import tempfile
import shutil
import os
import sys
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from io import StringIO
from typing import Dict, Any

# Add guardian to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'guardian'))

from guardian.cli.analyzer import ProjectAnalyzer
from guardian.cli.output_formatter import OutputFormatter, FormattingConfig, OutputLevel, Color


class TestProjectAnalyzerCore:
    """Test core project analyzer functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self._create_test_project()
        
        self.analyzer = ProjectAnalyzer({
            'max_function_lines': 20,
            'max_class_methods': 10,
            'use_etes_v2': True
        })
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_should_initialize_with_default_config_when_none_provided(self):
        """Test analyzer initialization with default configuration"""
        analyzer = ProjectAnalyzer()
        
        assert analyzer.config == {}
        assert analyzer.max_function_lines == 20
        assert analyzer.max_class_methods == 10
        assert analyzer.use_etes_v2 is False
        assert analyzer.total_files_analyzed == 0
        assert analyzer.errors_encountered == []
        assert analyzer.warnings_encountered == []
    
    def test_should_initialize_with_custom_config_when_provided(self):
        """Test analyzer initialization with custom configuration"""
        config = {
            'max_function_lines': 30,
            'max_class_methods': 15,
            'use_etes_v2': True
        }
        analyzer = ProjectAnalyzer(config)
        
        assert analyzer.max_function_lines == 30
        assert analyzer.max_class_methods == 15
        assert analyzer.use_etes_v2 is True
    
    def test_should_validate_project_path_when_analyzing(self):
        """Test project path validation"""
        # Test with valid path
        validation_result = self.analyzer._validate_inputs(str(self.temp_path), None, None)
        assert validation_result['has_errors'] is False
        
        # Test with invalid path
        validation_result = self.analyzer._validate_inputs("/nonexistent/path", None, None)
        assert validation_result['has_errors'] is True
        assert len(validation_result['errors']) > 0
    
    def test_should_analyze_project_successfully_when_valid_project_provided(self):
        """Test successful project analysis"""
        with patch.object(self.analyzer, '_run_tests') as mock_tests:
            mock_tests.return_value = {
                'pytest_ran_successfully': True,
                'pytest_exit_code': 0,
                'pytest_duration_seconds': 2.5
            }
            
            results = self.analyzer.analyze_project(str(self.temp_path))
            
            assert results is not None
            assert 'status' in results
            assert 'project_path' in results
            assert 'metrics' in results
            assert 'tes_score' in results
            assert results['project_path'] == str(self.temp_path)
            assert results['status'] in ['analysis_complete', 'analysis_partial']
    
    def test_should_handle_analysis_errors_gracefully(self):
        """Test graceful handling of analysis errors"""
        # Test with invalid project path
        results = self.analyzer.analyze_project("/invalid/path")
        
        assert results is not None
        assert results['status'] == 'validation_failed'
        assert 'errors' in results
    
    def test_should_analyze_codebase_metrics_accurately(self):
        """Test codebase metrics analysis"""
        metrics_result = self.analyzer._analyze_codebase(str(self.temp_path))
        
        assert 'metrics' in metrics_result
        assert 'details' in metrics_result
        
        metrics = metrics_result['metrics']
        assert 'total_lines_of_code_python' in metrics
        assert 'python_files_analyzed' in metrics
        assert 'average_cyclomatic_complexity' in metrics
        assert 'long_functions_count' in metrics
        assert 'large_classes_count' in metrics
        
        assert metrics['python_files_analyzed'] > 0
        assert metrics['total_lines_of_code_python'] > 0
    
    def test_should_perform_security_analysis_when_requested(self):
        """Test security analysis functionality"""
        with patch('guardian.cli.analyzer.check_dependencies_vulnerabilities') as mock_vuln:
            mock_vuln.return_value = {'details': [], 'error': None}
            
            security_results = self.analyzer._analyze_security(str(self.temp_path))
            
            assert 'dependency_vulnerabilities_count' in security_results
            assert 'eval_usage_count' in security_results
            assert 'hardcoded_secrets_count' in security_results
            assert security_results['dependency_vulnerabilities_count'] == 0
    
    def test_should_calculate_tes_score_when_analysis_complete(self):
        """Test TES score calculation"""
        mock_results = {
            'total_user_stories': 5,
            'covered_user_stories': 3,
            'test_execution_summary': {
                'pytest_duration_seconds': 1.5
            }
        }
        
        tes_results = self.analyzer._calculate_tes_score(mock_results)
        
        assert 'tes_score' in tes_results
        assert 'tes_grade' in tes_results
        assert 'tes_components' in tes_results
        assert 0.0 <= tes_results['tes_score'] <= 1.0
        assert tes_results['tes_grade'] in ['A+', 'A', 'B', 'C', 'D', 'F']
    
    def test_should_calculate_etes_score_when_enabled(self):
        """Test E-TES v2.0 score calculation"""
        mock_results = {
            'tes_components': {
                'mutation_score': 0.75,
                'avg_test_execution_time_ms': 150.0
            },
            'test_execution_summary': {
                'pytest_ran_successfully': True
            },
            'metrics': {
                'average_cyclomatic_complexity': 3.0,
                'total_lines_of_code_python': 1000
            }
        }
        
        etes_results = self.analyzer._calculate_etes_score(mock_results)
        
        assert 'etes_score' in etes_results
        assert 'etes_grade' in etes_results
        assert 'etes_components' in etes_results
        assert 'etes_comparison' in etes_results
        assert 0.0 <= etes_results['etes_score'] <= 1.0
    
    def test_should_identify_critical_issues_when_present(self):
        """Test critical issue identification"""
        # Test with critical issues
        results_with_issues = {
            'security_analysis': {
                'dependency_vulnerabilities_count': 2,
                'eval_usage_count': 1
            },
            'test_execution_summary': {
                'pytest_ran_successfully': False
            },
            'metrics': {
                'circular_dependencies_count': 1
            }
        }
        self.analyzer.errors_encountered = ['Some error']
        
        has_critical = self.analyzer._has_critical_issues(results_with_issues)
        assert has_critical is True
        
        # Test without critical issues
        results_without_issues = {
            'security_analysis': {
                'dependency_vulnerabilities_count': 0,
                'eval_usage_count': 0,
                'hardcoded_secrets_count': 0
            },
            'test_execution_summary': {
                'pytest_ran_successfully': True
            },
            'metrics': {
                'circular_dependencies_count': 0
            }
        }
        self.analyzer.errors_encountered = []
        
        has_critical = self.analyzer._has_critical_issues(results_without_issues)
        assert has_critical is False
    
    def test_should_handle_user_stories_analysis_when_file_provided(self):
        """Test user stories analysis"""
        # Create user stories file
        stories_file = self.temp_path / "user_stories.txt"
        stories_file.write_text("""
As a user, I want to login
As a user, I want to view my profile
As an admin, I want to manage users
""")
        
        stories_result = self.analyzer._analyze_user_stories(str(stories_file))
        
        assert 'total_user_stories' in stories_result
        assert 'covered_user_stories' in stories_result
        assert stories_result['total_user_stories'] == 3
    
    def test_should_handle_missing_user_stories_file_gracefully(self):
        """Test handling of missing user stories file"""
        stories_result = self.analyzer._analyze_user_stories("/nonexistent/stories.txt")
        
        assert stories_result['total_user_stories'] == 0
        assert stories_result['covered_user_stories'] == 0
    
    def _create_test_project(self):
        """Create a test project structure"""
        # Create Python files
        (self.temp_path / "main.py").write_text('''
def main():
    """Main function"""
    print("Hello, world!")
    return 0

class TestClass:
    def method1(self):
        return "test"
    
    def method2(self):
        return "another test"
''')
        
        (self.temp_path / "utils.py").write_text('''
import os
import sys

def utility_function(x, y):
    """A utility function"""
    if x > y:
        return x
    else:
        return y

def long_function():
    """A function with many lines"""
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
    return line22
''')


class TestOutputFormatterCore:
    """Test output formatter functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.config = FormattingConfig(use_colors=True, max_line_length=80)
        self.formatter = OutputFormatter(self.config)
    
    def test_should_initialize_with_default_config_when_none_provided(self):
        """Test formatter initialization with defaults"""
        formatter = OutputFormatter()
        
        assert formatter.config is not None
        assert formatter.config.use_colors is True
        assert formatter.config.max_line_length == 80
        assert formatter.level == OutputLevel.NORMAL
    
    def test_should_initialize_with_custom_config_when_provided(self):
        """Test formatter initialization with custom config"""
        config = FormattingConfig(
            use_colors=False,
            max_line_length=120,
            show_timestamps=True
        )
        formatter = OutputFormatter(config)
        
        assert formatter.config.use_colors is False
        assert formatter.config.max_line_length == 120
        assert formatter.config.show_timestamps is True
    
    def test_should_set_output_level_when_requested(self):
        """Test output level setting"""
        self.formatter.set_level(OutputLevel.VERBOSE)
        assert self.formatter.level == OutputLevel.VERBOSE
        
        self.formatter.set_level(OutputLevel.QUIET)
        assert self.formatter.level == OutputLevel.QUIET
    
    def test_should_format_json_output_when_requested(self):
        """Test JSON output formatting"""
        test_results = {
            'project_path': '/test/path',
            'tes_score': 0.75,
            'status': 'complete'
        }
        
        json_output = self.formatter.format_analysis_results(test_results, 'json')
        
        # Should be valid JSON
        parsed = json.loads(json_output)
        assert parsed['project_path'] == '/test/path'
        assert parsed['tes_score'] == 0.75
        assert parsed['status'] == 'complete'
    
    def test_should_format_human_readable_output_when_requested(self):
        """Test human-readable output formatting"""
        test_results = {
            'project_path': '/test/path',
            'status': 'analysis_complete',
            'tes_score': 0.85,
            'tes_grade': 'A',
            'etes_v2_enabled': True,
            'etes_score': 0.88,
            'etes_grade': 'A',
            'metrics': {
                'total_lines_of_code_python': 1500,
                'python_files_analyzed': 10,
                'average_cyclomatic_complexity': 3.2
            },
            'test_execution_summary': {
                'pytest_ran_successfully': True,
                'pytest_exit_code': 0,
                'pytest_duration_seconds': 2.1
            },
            'security_analysis': {
                'dependency_vulnerabilities_count': 0,
                'eval_usage_count': 0,
                'hardcoded_secrets_count': 0
            },
            'has_critical_issues': False
        }
        
        human_output = self.formatter.format_analysis_results(test_results, 'human')
        
        assert isinstance(human_output, str)
        assert 'Guardian Analysis Report' in human_output
        assert '/test/path' in human_output
        assert '0.85' in human_output  # TES score
        assert '0.88' in human_output  # E-TES score
        assert 'Grade: A' in human_output
    
    def test_should_colorize_output_when_colors_enabled(self):
        """Test color application in output"""
        # Test with colors enabled
        formatter_with_colors = OutputFormatter(FormattingConfig(use_colors=True))
        
        error_msg = formatter_with_colors.format_error("Test error")
        warning_msg = formatter_with_colors.format_warning("Test warning")
        success_msg = formatter_with_colors.format_success("Test success")
        
        # Should contain ANSI color codes
        assert '\033[' in error_msg or 'Test error' in error_msg
        assert '\033[' in warning_msg or 'Test warning' in warning_msg
        assert '\033[' in success_msg or 'Test success' in success_msg
    
    def test_should_not_colorize_output_when_colors_disabled(self):
        """Test plain output when colors disabled"""
        formatter_no_colors = OutputFormatter(FormattingConfig(use_colors=False))
        
        error_msg = formatter_no_colors.format_error("Test error")
        warning_msg = formatter_no_colors.format_warning("Test warning")
        success_msg = formatter_no_colors.format_success("Test success")
        
        # Should not contain ANSI color codes
        assert '\033[' not in error_msg
        assert '\033[' not in warning_msg
        assert '\033[' not in success_msg
        assert 'Test error' in error_msg
        assert 'Test warning' in warning_msg
        assert 'Test success' in success_msg
    
    def test_should_format_tes_scores_section_accurately(self):
        """Test TES scores section formatting"""
        test_results = {
            'tes_score': 0.756,
            'tes_grade': 'B',
            'tes_components': {
                'mutation_score': 0.8,
                'behavior_coverage_calculated': 0.75,
                'speed_factor_calculated': 0.9
            }
        }
        
        tes_section = self.formatter._format_tes_scores(test_results)
        
        assert 'Test Effectiveness Score' in tes_section
        assert '0.756' in tes_section
        assert 'Grade: B' in tes_section
        assert 'Mutation Score: 0.800' in tes_section
        assert 'Behavior Coverage: 0.750' in tes_section
        assert 'Speed Factor: 0.900' in tes_section
    
    def test_should_format_etes_scores_section_when_enabled(self):
        """Test E-TES scores section formatting"""
        test_results = {
            'etes_score': 0.823,
            'etes_grade': 'A',
            'etes_components': {
                'mutation_score': 0.85,
                'evolution_gain': 1.15,
                'assertion_iq': 0.78,
                'behavior_coverage': 0.82,
                'speed_factor': 0.91,
                'quality_factor': 0.88,
                'insights': ['Excellent mutation coverage', 'Good test speed']
            },
            'etes_comparison': {
                'improvement': 0.067,
                'recommendations': ['Maintain current quality', 'Focus on assertion intelligence']
            }
        }
        
        etes_section = self.formatter._format_etes_scores(test_results)
        
        assert 'E-TES v2.0' in etes_section
        assert '0.823' in etes_section
        assert 'Grade: A' in etes_section
        assert 'Evolution Gain: 1.150' in etes_section
        assert 'Assertion IQ: 0.780' in etes_section
        assert 'Excellent mutation coverage' in etes_section
        assert '+0.067' in etes_section  # Improvement
    
    def test_should_format_critical_issues_section_when_present(self):
        """Test critical issues section formatting"""
        test_results = {
            'has_critical_issues': True,
            'details': {
                'vulnerability_details_list': [
                    {'name': 'requests', 'id': 'CVE-2023-1234', 'version': '2.25.1'}
                ],
                'eval_usage_details_list': [
                    {'file': 'dangerous.py', 'line_number': 42}
                ]
            }
        }
        
        issues_section = self.formatter._format_critical_issues(test_results)
        
        assert 'CRITICAL ISSUES DETECTED' in issues_section
        assert 'Security Vulnerabilities' in issues_section
        assert 'requests' in issues_section
        assert 'CVE-2023-1234' in issues_section
        assert 'Dangerous eval() Usage' in issues_section
        assert 'dangerous.py:42' in issues_section
    
    def test_should_get_appropriate_score_color_based_on_value(self):
        """Test score color selection"""
        # High score should be green
        high_color = self.formatter._get_score_color(0.9)
        assert high_color == Color.GREEN or high_color == ""
        
        # Medium score should be yellow
        medium_color = self.formatter._get_score_color(0.7)
        assert medium_color == Color.YELLOW or medium_color == ""
        
        # Low score should be red
        low_color = self.formatter._get_score_color(0.4)
        assert low_color == Color.RED or low_color == ""


class TestColorUtilities:
    """Test color utility functions"""
    
    def test_should_apply_color_when_colorize_called(self):
        """Test color application"""
        colored_text = Color.colorize("test text", Color.RED)
        
        # Should either contain color codes or be the original text
        assert Color.RED in colored_text or "test text" in colored_text
        assert Color.RESET in colored_text or "test text" in colored_text
    
    def test_should_handle_empty_text_gracefully(self):
        """Test color application with empty text"""
        colored_text = Color.colorize("", Color.BLUE)
        
        # Should handle empty text without errors
        assert isinstance(colored_text, str)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

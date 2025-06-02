"""
Comprehensive Static Analysis Test Suite

Tests for Guardian's static analysis capabilities including complexity metrics,
code smell detection, and security analysis with high mutation score coverage.
"""

import pytest
import tempfile
import shutil
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from typing import Dict, Any, List

# Add guardian to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'guardian'))

from guardian.analysis.static import (
    analyze_file, calculate_cyclomatic_complexity, 
    detect_code_smells, get_function_metrics,
    count_lines_of_code, find_long_elements,
    find_large_classes, analyze_imports,
    find_unused_imports, build_import_graph,
    find_circular_dependencies
)
from guardian.analysis.security import (
    check_dependencies_vulnerabilities,
    check_for_eval_usage,
    check_for_hardcoded_secrets
)


class TestStaticAnalysisCore:
    """Test core static analysis functionality with comprehensive edge cases"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_should_analyze_simple_file_when_valid_python_provided(self):
        """Test basic file analysis with simple Python code"""
        test_file = self.temp_path / "simple.py"
        test_file.write_text('''
def simple_function(x):
    """A simple function"""
    return x + 1

class SimpleClass:
    def method(self):
        return "hello"
''')
        
        result = analyze_file(str(test_file))
        
        assert result is not None
        assert 'file_path' in result
        assert 'total_lines' in result
        assert 'functions' in result
        assert 'classes' in result
        assert result['total_lines'] > 0
        assert len(result['functions']) >= 1
        assert len(result['classes']) >= 1
    
    def test_should_handle_nonexistent_file_gracefully(self):
        """Test handling of non-existent files"""
        result = analyze_file("nonexistent_file.py")
        
        assert result is None or 'error' in result
    
    def test_should_handle_invalid_python_syntax_gracefully(self):
        """Test handling of files with syntax errors"""
        test_file = self.temp_path / "invalid.py"
        test_file.write_text("def invalid_syntax(:\n    pass")
        
        result = analyze_file(str(test_file))
        
        # Should handle gracefully without crashing
        assert result is None or 'error' in result or 'syntax_error' in result
    
    def test_should_handle_empty_file_gracefully(self):
        """Test handling of empty files"""
        test_file = self.temp_path / "empty.py"
        test_file.write_text("")
        
        result = analyze_file(str(test_file))
        
        assert result is not None
        assert result['total_lines'] == 0
        assert len(result['functions']) == 0
        assert len(result['classes']) == 0
    
    def test_should_handle_file_with_only_comments_gracefully(self):
        """Test handling of files with only comments"""
        test_file = self.temp_path / "comments_only.py"
        test_file.write_text('''
# This is a comment
# Another comment
"""
This is a docstring
"""
''')
        
        result = analyze_file(str(test_file))
        
        assert result is not None
        assert result['total_lines'] > 0
        assert len(result['functions']) == 0
        assert len(result['classes']) == 0
    
    def test_should_calculate_cyclomatic_complexity_accurately_for_simple_function(self):
        """Test cyclomatic complexity calculation for simple functions"""
        simple_code = '''
def simple(x):
    return x + 1
'''
        complexity = calculate_cyclomatic_complexity(simple_code)
        assert complexity == 1  # No branches = complexity 1
    
    def test_should_calculate_cyclomatic_complexity_accurately_for_complex_function(self):
        """Test cyclomatic complexity calculation for complex functions"""
        complex_code = '''
def complex(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return x + y + z
            else:
                return x + y
        else:
            return x
    else:
        return 0
'''
        complexity = calculate_cyclomatic_complexity(complex_code)
        assert complexity > 3  # Multiple nested branches
    
    def test_should_calculate_cyclomatic_complexity_for_loops_and_exceptions(self):
        """Test cyclomatic complexity with loops and exception handling"""
        loop_exception_code = '''
def complex_function(items):
    try:
        for item in items:
            if item > 0:
                continue
            elif item < 0:
                break
            else:
                pass
        return True
    except ValueError:
        return False
    except TypeError:
        return None
    finally:
        cleanup()
'''
        complexity = calculate_cyclomatic_complexity(loop_exception_code)
        assert complexity > 5  # for + if + elif + try + except + except
    
    def test_should_count_lines_of_code_excluding_comments_and_blanks(self):
        """Test line counting accuracy"""
        code_with_comments = '''
# This is a comment
def function():
    """Docstring"""
    # Another comment
    
    x = 1  # Inline comment
    return x

# Final comment
'''
        loc = count_lines_of_code(code_with_comments)
        assert loc == 3  # Only actual code lines: def, x = 1, return x
    
    def test_should_detect_long_functions_when_threshold_exceeded(self):
        """Test detection of long functions"""
        # Create a function with many lines
        lines = ["    line{} = {}".format(i, i) for i in range(1, 26)]
        long_function_code = f'''
def very_long_function():
{chr(10).join(lines)}
    return line25
'''
        
        long_elements = find_long_elements(long_function_code, max_lines=20)
        
        assert len(long_elements) > 0
        assert long_elements[0]['name'] == 'very_long_function'
        assert long_elements[0]['lines'] > 20
    
    def test_should_not_detect_short_functions_as_long(self):
        """Test that short functions are not flagged as long"""
        short_function_code = '''
def short_function():
    return 42
'''
        
        long_elements = find_long_elements(short_function_code, max_lines=20)
        
        assert len(long_elements) == 0
    
    def test_should_detect_large_classes_when_method_threshold_exceeded(self):
        """Test detection of large classes"""
        # Create a class with many methods
        methods = [f"    def method{i}(self): pass" for i in range(1, 16)]
        large_class_code = f'''
class VeryLargeClass:
{chr(10).join(methods)}
'''
        
        large_classes = find_large_classes(large_class_code, max_methods=10)
        
        assert len(large_classes) > 0
        assert large_classes[0]['name'] == 'VeryLargeClass'
        assert large_classes[0]['method_count'] > 10
    
    def test_should_not_detect_small_classes_as_large(self):
        """Test that small classes are not flagged as large"""
        small_class_code = '''
class SmallClass:
    def method1(self):
        pass
    
    def method2(self):
        pass
'''
        
        large_classes = find_large_classes(small_class_code, max_methods=10)
        
        assert len(large_classes) == 0
    
    def test_should_analyze_imports_accurately_when_various_import_styles_used(self):
        """Test import analysis with different import styles"""
        import_code = '''
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from collections import defaultdict, Counter
'''
        
        imports = analyze_imports(import_code)
        
        assert len(imports) > 0
        
        # Check for specific imports
        import_modules = [imp['module'] for imp in imports]
        assert 'os' in import_modules
        assert 'sys' in import_modules
        assert 'pathlib' in import_modules
        assert 'typing' in import_modules
        assert 'numpy' in import_modules
        assert 'collections' in import_modules
    
    def test_should_detect_unused_imports_when_imports_not_referenced(self):
        """Test detection of unused imports"""
        unused_import_code = '''
import os
import sys
import unused_module
from pathlib import Path
from typing import Dict

def function():
    path = Path(".")
    return str(path)
'''
        
        unused_imports = find_unused_imports(unused_import_code)
        
        assert len(unused_imports) > 0
        
        # Should detect unused imports
        unused_names = [imp['name'] for imp in unused_imports]
        assert 'os' in unused_names
        assert 'sys' in unused_names
        assert 'unused_module' in unused_names
        assert 'Dict' in unused_names
        
        # Should not flag used imports
        assert 'Path' not in unused_names
    
    def test_should_not_flag_used_imports_as_unused(self):
        """Test that used imports are not flagged as unused"""
        used_import_code = '''
import os
from pathlib import Path

def function():
    current_dir = os.getcwd()
    path = Path(current_dir)
    return path.exists()
'''
        
        unused_imports = find_unused_imports(used_import_code)
        
        # Should not flag os or Path as unused
        unused_names = [imp['name'] for imp in unused_imports]
        assert 'os' not in unused_names
        assert 'Path' not in unused_names
    
    def test_should_build_import_graph_when_multiple_files_analyzed(self):
        """Test import graph building"""
        # Create multiple files with dependencies
        file1 = self.temp_path / "module1.py"
        file1.write_text("from module2 import function2")
        
        file2 = self.temp_path / "module2.py"
        file2.write_text("from module3 import function3")
        
        file3 = self.temp_path / "module3.py"
        file3.write_text("def function3(): pass")
        
        import_graph = build_import_graph(str(self.temp_path))
        
        assert isinstance(import_graph, dict)
        assert len(import_graph) > 0
    
    def test_should_detect_circular_dependencies_when_present(self):
        """Test circular dependency detection"""
        # Create files with circular dependencies
        file1 = self.temp_path / "circular1.py"
        file1.write_text("from circular2 import func2")
        
        file2 = self.temp_path / "circular2.py"
        file2.write_text("from circular1 import func1")
        
        import_graph = build_import_graph(str(self.temp_path))
        circular_deps = find_circular_dependencies(import_graph)
        
        # Should detect the circular dependency
        assert len(circular_deps) > 0
    
    def test_should_not_detect_circular_dependencies_when_none_present(self):
        """Test that linear dependencies are not flagged as circular"""
        # Create files with linear dependencies
        file1 = self.temp_path / "linear1.py"
        file1.write_text("from linear2 import func2")
        
        file2 = self.temp_path / "linear2.py"
        file2.write_text("def func2(): pass")
        
        import_graph = build_import_graph(str(self.temp_path))
        circular_deps = find_circular_dependencies(import_graph)
        
        # Should not detect circular dependencies
        assert len(circular_deps) == 0


class TestCodeSmellDetection:
    """Test code smell detection algorithms"""
    
    def test_should_detect_all_smell_types_when_problematic_code_analyzed(self):
        """Test comprehensive code smell detection"""
        test_file = Path(tempfile.mktemp(suffix='.py'))
        
        # Create code with multiple smells
        problematic_code = '''
class VeryLargeClass:
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

def very_complex_function(a, b, c, d, e, f, g, h):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        if f > 0:
                            if g > 0:
                                if h > 0:
                                    return "all positive"
                                else:
                                    return "h not positive"
                            else:
                                return "g not positive"
                        else:
                            return "f not positive"
                    else:
                        return "e not positive"
                else:
                    return "d not positive"
            else:
                return "c not positive"
        else:
            return "b not positive"
    else:
        return "a not positive"
'''
        
        test_file.write_text(problematic_code)
        
        try:
            result = analyze_file(str(test_file))
            smells = detect_code_smells(result)
            
            # Should detect multiple types of smells
            smell_types = [smell['type'] for smell in smells]
            assert 'large_class' in smell_types
            assert 'long_function' in smell_types
            assert 'complex_function' in smell_types
            assert 'long_parameter_list' in smell_types
            
            # Should provide suggestions
            for smell in smells:
                assert 'suggestion' in smell
                assert len(smell['suggestion']) > 0
        
        finally:
            test_file.unlink()
    
    def test_should_not_detect_smells_in_clean_code(self):
        """Test that clean code doesn't trigger smell detection"""
        test_file = Path(tempfile.mktemp(suffix='.py'))
        
        clean_code = '''
class CleanClass:
    def method1(self):
        return "clean"
    
    def method2(self):
        return "also clean"

def clean_function(x, y):
    if x > y:
        return x
    else:
        return y
'''
        
        test_file.write_text(clean_code)
        
        try:
            result = analyze_file(str(test_file))
            smells = detect_code_smells(result)
            
            # Should not detect any smells
            assert len(smells) == 0
        
        finally:
            test_file.unlink()


class TestSecurityAnalysis:
    """Test security analysis functionality"""
    
    def test_should_detect_eval_usage_when_present(self):
        """Test detection of eval() usage"""
        code_with_eval = '''
def dangerous_function(user_input):
    result = eval(user_input)  # Dangerous!
    return result

def another_function():
    x = eval("2 + 2")  # Also dangerous
    return x
'''
        
        eval_findings = check_for_eval_usage(code_with_eval)
        
        assert len(eval_findings) == 2
        assert all('eval' in finding['line_content'] for finding in eval_findings)
        assert all(finding['line_number'] > 0 for finding in eval_findings)
    
    def test_should_not_detect_eval_in_comments_or_strings(self):
        """Test that eval in comments/strings is not flagged"""
        code_without_real_eval = '''
def safe_function():
    # This mentions eval() but doesn't use it
    message = "Don't use eval() in production"
    return message
'''
        
        eval_findings = check_for_eval_usage(code_without_real_eval)
        
        # Should not detect eval in comments or strings
        assert len(eval_findings) == 0
    
    def test_should_detect_hardcoded_secrets_when_present(self):
        """Test detection of hardcoded secrets"""
        code_with_secrets = '''
API_KEY = "sk-1234567890abcdef"
PASSWORD = "super_secret_password"
TOKEN = "ghp_1234567890abcdef"

def connect_to_api():
    key = "another-secret-key"
    return key
'''
        
        secret_findings = check_for_hardcoded_secrets(code_with_secrets)
        
        assert len(secret_findings) > 0
        
        # Should detect various types of secrets
        patterns_found = [finding['pattern_name'] for finding in secret_findings]
        assert len(patterns_found) > 0
    
    def test_should_not_detect_false_positive_secrets(self):
        """Test that legitimate strings are not flagged as secrets"""
        code_without_secrets = '''
def safe_function():
    message = "Hello, world!"
    number = 12345
    return f"{message} {number}"
'''
        
        secret_findings = check_for_hardcoded_secrets(code_without_secrets)
        
        # Should not detect false positives
        assert len(secret_findings) == 0
    
    @patch('subprocess.run')
    def test_should_handle_dependency_check_success(self, mock_subprocess):
        """Test successful dependency vulnerability check"""
        # Mock successful safety check
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "No known security vulnerabilities found."
        mock_subprocess.return_value.stderr = ""
        
        result = check_dependencies_vulnerabilities()
        
        assert 'error' not in result
        assert result.get('details', []) == []  # No vulnerabilities
    
    @patch('subprocess.run')
    def test_should_handle_dependency_check_with_vulnerabilities(self, mock_subprocess):
        """Test dependency check with vulnerabilities found"""
        # Mock safety check with vulnerabilities
        mock_subprocess.return_value.returncode = 64  # Safety exit code for vulnerabilities
        mock_subprocess.return_value.stdout = '''
{
  "vulnerabilities": [
    {
      "package": "requests",
      "version": "2.25.1",
      "id": "12345",
      "advisory": "Security vulnerability in requests"
    }
  ]
}
'''
        mock_subprocess.return_value.stderr = ""
        
        result = check_dependencies_vulnerabilities()
        
        assert 'details' in result
        assert len(result['details']) > 0
    
    @patch('subprocess.run')
    def test_should_handle_dependency_check_failure(self, mock_subprocess):
        """Test handling of dependency check tool failure"""
        # Mock tool failure
        mock_subprocess.side_effect = FileNotFoundError("safety command not found")
        
        result = check_dependencies_vulnerabilities()
        
        assert 'error' in result
        assert 'not found' in result['error'].lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

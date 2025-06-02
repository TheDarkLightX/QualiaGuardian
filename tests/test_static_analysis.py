"""
Tests for Guardian Static Analysis functionality

Testing the static analysis capabilities including complexity metrics,
code smell detection, and security analysis.
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path

# Add guardian to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'guardian'))

from guardian.analysis.static import (
    analyze_file, calculate_cyclomatic_complexity, 
    detect_code_smells, get_function_metrics
)


class TestStaticAnalysis:
    """Test static analysis functionality"""
    
    def setup_method(self):
        """Set up test files"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test Python file
        self.test_file = self.temp_path / "test_code.py"
        self.test_file.write_text('''
def simple_function(x):
    """A simple function"""
    return x + 1

def complex_function(a, b, c, d, e):
    """A complex function with many branches"""
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

class LargeClass:
    """A class with many methods"""
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
    line23 = 23
    line24 = 24
    line25 = 25
    return line25

def duplicate_code_1():
    """First duplicate"""
    x = 1
    y = 2
    z = x + y
    return z * 2

def duplicate_code_2():
    """Second duplicate"""
    x = 1
    y = 2
    z = x + y
    return z * 2
''')
    
    def teardown_method(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_analyze_file_basic(self):
        """Test basic file analysis"""
        result = analyze_file(str(self.test_file))
        
        # Should return analysis results
        assert isinstance(result, dict)
        assert 'file_path' in result
        assert 'total_lines' in result
        assert 'functions' in result
        assert 'classes' in result
        
        # Check basic metrics
        assert result['total_lines'] > 0
        assert len(result['functions']) >= 5  # We defined 5+ functions
        assert len(result['classes']) >= 1   # We defined 1 class
    
    def test_calculate_cyclomatic_complexity(self):
        """Test cyclomatic complexity calculation"""
        # Test simple function (should have low complexity)
        simple_code = '''
def simple(x):
    return x + 1
'''
        complexity = calculate_cyclomatic_complexity(simple_code)
        assert complexity == 1  # No branches = complexity 1
        
        # Test complex function (should have high complexity)
        complex_code = '''
def complex(x):
    if x > 0:
        if x > 10:
            return "high"
        else:
            return "medium"
    else:
        return "low"
'''
        complexity = calculate_cyclomatic_complexity(complex_code)
        assert complexity > 1  # Multiple branches = higher complexity
    
    def test_detect_code_smells(self):
        """Test code smell detection"""
        result = analyze_file(str(self.test_file))
        smells = detect_code_smells(result)
        
        assert isinstance(smells, list)
        
        # Should detect some smells in our test file
        smell_types = [smell['type'] for smell in smells]
        
        # Should detect large class (LargeClass has 12 methods)
        assert 'large_class' in smell_types
        
        # Should detect long function
        assert 'long_function' in smell_types
        
        # Should detect complex function
        assert 'complex_function' in smell_types
    
    def test_get_function_metrics(self):
        """Test function metrics extraction"""
        result = analyze_file(str(self.test_file))
        
        # Check that we get metrics for functions
        assert 'functions' in result
        functions = result['functions']
        
        # Find the complex function
        complex_func = None
        for func in functions:
            if func['name'] == 'complex_function':
                complex_func = func
                break
        
        assert complex_func is not None
        assert 'complexity' in complex_func
        assert 'lines' in complex_func
        assert 'parameters' in complex_func
        
        # Complex function should have high complexity
        assert complex_func['complexity'] > 5
        
        # Should have 5 parameters
        assert complex_func['parameters'] == 5
    
    def test_analyze_nonexistent_file(self):
        """Test analysis of non-existent file"""
        result = analyze_file("nonexistent_file.py")
        
        # Should handle gracefully
        assert result is None or 'error' in result
    
    def test_analyze_invalid_python(self):
        """Test analysis of invalid Python code"""
        invalid_file = self.temp_path / "invalid.py"
        invalid_file.write_text("def invalid_syntax(:\n    pass")
        
        result = analyze_file(str(invalid_file))
        
        # Should handle syntax errors gracefully
        assert result is None or 'error' in result or 'syntax_error' in result


class TestComplexityMetrics:
    """Test complexity metric calculations"""
    
    def test_cyclomatic_complexity_edge_cases(self):
        """Test cyclomatic complexity edge cases"""
        # Empty function
        empty_code = "def empty(): pass"
        assert calculate_cyclomatic_complexity(empty_code) == 1
        
        # Function with loops
        loop_code = '''
def with_loops(items):
    for item in items:
        if item > 0:
            continue
        else:
            break
    return items
'''
        complexity = calculate_cyclomatic_complexity(loop_code)
        assert complexity > 2  # for + if + else = at least 3
        
        # Function with try/except
        exception_code = '''
def with_exception():
    try:
        risky_operation()
    except ValueError:
        handle_value_error()
    except TypeError:
        handle_type_error()
    finally:
        cleanup()
'''
        complexity = calculate_cyclomatic_complexity(exception_code)
        assert complexity > 1  # Multiple exception handlers increase complexity
    
    def test_function_length_calculation(self):
        """Test function length calculation"""
        # Create a function with known line count
        long_function_code = '''
def long_function():
    line1 = 1
    line2 = 2
    line3 = 3
    line4 = 4
    line5 = 5
    return line5
'''
        
        # Write to temp file and analyze
        temp_file = Path(tempfile.mktemp(suffix='.py'))
        temp_file.write_text(long_function_code)
        
        try:
            result = analyze_file(str(temp_file))
            functions = result['functions']
            
            long_func = functions[0]  # Should be the only function
            assert long_func['name'] == 'long_function'
            assert long_func['lines'] >= 6  # Should count the lines
            
        finally:
            temp_file.unlink()
    
    def test_parameter_count_calculation(self):
        """Test parameter count calculation"""
        multi_param_code = '''
def many_params(a, b, c, d, e, f, g, h, i, j):
    return a + b + c + d + e + f + g + h + i + j

def few_params(x, y):
    return x + y

def no_params():
    return 42
'''
        
        temp_file = Path(tempfile.mktemp(suffix='.py'))
        temp_file.write_text(multi_param_code)
        
        try:
            result = analyze_file(str(temp_file))
            functions = result['functions']
            
            # Check parameter counts
            param_counts = {func['name']: func['parameters'] for func in functions}
            
            assert param_counts['many_params'] == 10
            assert param_counts['few_params'] == 2
            assert param_counts['no_params'] == 0
            
        finally:
            temp_file.unlink()


class TestCodeSmellDetection:
    """Test code smell detection algorithms"""
    
    def test_large_class_detection(self):
        """Test detection of large classes"""
        large_class_code = '''
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
    def method13(self): pass
    def method14(self): pass
    def method15(self): pass

class SmallClass:
    def method1(self): pass
    def method2(self): pass
'''
        
        temp_file = Path(tempfile.mktemp(suffix='.py'))
        temp_file.write_text(large_class_code)
        
        try:
            result = analyze_file(str(temp_file))
            smells = detect_code_smells(result)
            
            # Should detect large class smell
            large_class_smells = [s for s in smells if s['type'] == 'large_class']
            assert len(large_class_smells) > 0
            
            # Should identify the correct class
            assert any('VeryLargeClass' in smell['description'] for smell in large_class_smells)
            
        finally:
            temp_file.unlink()
    
    def test_long_function_detection(self):
        """Test detection of long functions"""
        # Create a function with many lines
        lines = ["    line{} = {}".format(i, i) for i in range(1, 26)]
        long_function_code = f'''
def very_long_function():
{chr(10).join(lines)}
    return line25

def short_function():
    return 42
'''
        
        temp_file = Path(tempfile.mktemp(suffix='.py'))
        temp_file.write_text(long_function_code)
        
        try:
            result = analyze_file(str(temp_file))
            smells = detect_code_smells(result)
            
            # Should detect long function smell
            long_function_smells = [s for s in smells if s['type'] == 'long_function']
            assert len(long_function_smells) > 0
            
            # Should identify the correct function
            assert any('very_long_function' in smell['description'] for smell in long_function_smells)
            
        finally:
            temp_file.unlink()
    
    def test_complex_function_detection(self):
        """Test detection of complex functions"""
        complex_code = '''
def very_complex_function(a, b, c, d, e, f, g, h, i, j):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        if f > 0:
                            if g > 0:
                                if h > 0:
                                    if i > 0:
                                        if j > 0:
                                            return "all positive"
                                        else:
                                            return "j not positive"
                                    else:
                                        return "i not positive"
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

def simple_function(x):
    return x + 1
'''
        
        temp_file = Path(tempfile.mktemp(suffix='.py'))
        temp_file.write_text(complex_code)
        
        try:
            result = analyze_file(str(temp_file))
            smells = detect_code_smells(result)
            
            # Should detect complex function smell
            complex_smells = [s for s in smells if s['type'] == 'complex_function']
            assert len(complex_smells) > 0
            
            # Should identify the correct function
            assert any('very_complex_function' in smell['description'] for smell in complex_smells)
            
        finally:
            temp_file.unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

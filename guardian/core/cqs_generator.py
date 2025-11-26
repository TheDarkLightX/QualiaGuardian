"""
Automated Code Generation for Quality Improvement

Generates high-quality code variants using:
1. Template-based generation
2. Pattern-based transformation
3. Quality-guided search
4. Semantic preservation
"""

import ast
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GeneratedCode:
    """Generated code variant."""
    code: str
    quality_score: float
    improvements: List[str]
    transformation_applied: str


class QualityCodeGenerator:
    """
    Generates high-quality code variants.
    
    Uses templates and patterns learned from high-quality code
    to generate improved versions.
    """
    
    # Quality templates
    FUNCTION_TEMPLATE = """def {name}({params}) -> {return_type}:
    \"\"\"{docstring}\"\"\"
    {body}
"""
    
    EARLY_RETURN_TEMPLATE = """    if not {condition}:
        return {default_value}
    
    {rest_of_code}
"""
    
    def generate_improved_code(
        self,
        original_code: str,
        target_quality: float = 0.9
    ) -> List[GeneratedCode]:
        """
        Generate improved code variants.
        
        Args:
            original_code: Original code
            target_quality: Target quality score
            
        Returns:
            List of generated code variants
        """
        variants = []
        
        try:
            tree = ast.parse(original_code)
            
            # Generate variants through transformations
            variants.extend(self._generate_early_return_variants(original_code, tree))
            variants.extend(self._generate_extracted_method_variants(original_code, tree))
            variants.extend(self._generate_type_hint_variants(original_code, tree))
            variants.extend(self._generate_docstring_variants(original_code, tree))
            variants.extend(self._generate_naming_variants(original_code, tree))
            
            # Score and sort
            from guardian.core.cqs_enhanced import EnhancedCQSCalculator
            calc = EnhancedCQSCalculator()
            
            for variant in variants:
                enhanced = calc.calculate_enhanced(variant.code)
                variant.quality_score = enhanced.enhanced_cqs_score
            
            variants.sort(key=lambda v: v.quality_score, reverse=True)
            
        except SyntaxError as e:
            logger.warning(f"Syntax error: {e}")
        
        return variants
    
    def _generate_early_return_variants(
        self,
        code: str,
        tree: ast.AST
    ) -> List[GeneratedCode]:
        """Generate variants with early returns."""
        variants = []
        lines = code.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and node.orelse:
                # Convert if-else to early return
                try:
                    if_code = '\n'.join(lines[node.lineno - 1:node.end_lineno if hasattr(node, 'end_lineno') else node.lineno + 5])
                    
                    # Generate early return variant
                    new_code = self._apply_early_return_pattern(code, node, lines)
                    if new_code and new_code != code:
                        variants.append(GeneratedCode(
                            code=new_code,
                            quality_score=0.0,  # Will be calculated later
                            improvements=["Converted nested if-else to early return pattern"],
                            transformation_applied="early_return"
                        ))
                except:
                    pass
        
        return variants
    
    def _generate_extracted_method_variants(
        self,
        code: str,
        tree: ast.AST
    ) -> List[GeneratedCode]:
        """Generate variants with extracted methods."""
        variants = []
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        for func in functions:
            func_lines = func.end_lineno - func.lineno if hasattr(func, 'end_lineno') else 20
            if func_lines > 30:
                # Suggest method extraction
                new_code = self._extract_method_suggestion(code, func)
                if new_code:
                    variants.append(GeneratedCode(
                        code=new_code,
                        quality_score=0.0,
                        improvements=[f"Extracted method from {func.name} to reduce complexity"],
                        transformation_applied="extract_method"
                    ))
        
        return variants
    
    def _generate_type_hint_variants(
        self,
        code: str,
        tree: ast.AST
    ) -> List[GeneratedCode]:
        """Generate variants with type hints."""
        variants = []
        
        if not self._has_type_hints(code):
            new_code = self._add_type_hints(code, tree)
            if new_code != code:
                variants.append(GeneratedCode(
                    code=new_code,
                    quality_score=0.0,
                    improvements=["Added type hints for better clarity"],
                    transformation_applied="add_type_hints"
                ))
        
        return variants
    
    def _generate_docstring_variants(
        self,
        code: str,
        tree: ast.AST
    ) -> List[GeneratedCode]:
        """Generate variants with docstrings."""
        variants = []
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        for func in functions:
            if not ast.get_docstring(func):
                new_code = self._add_docstring(code, func)
                if new_code != code:
                    variants.append(GeneratedCode(
                        code=new_code,
                        quality_score=0.0,
                        improvements=[f"Added docstring to {func.name}"],
                        transformation_applied="add_docstring"
                    ))
        
        return variants
    
    def _generate_naming_variants(
        self,
        code: str,
        tree: ast.AST
    ) -> List[GeneratedCode]:
        """Generate variants with improved naming."""
        variants = []
        
        # Find poorly named functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.name) < 5 or not self._is_descriptive(node.name):
                    new_code = self._improve_naming(code, node)
                    if new_code != code:
                        variants.append(GeneratedCode(
                            code=new_code,
                            quality_score=0.0,
                            improvements=[f"Improved function name: {node.name}"],
                            transformation_applied="improve_naming"
                        ))
        
        return variants
    
    def _apply_early_return_pattern(
        self,
        code: str,
        if_node: ast.If,
        lines: List[str]
    ) -> Optional[str]:
        """Apply early return pattern."""
        # Simplified - would need full AST manipulation
        return code
    
    def _extract_method_suggestion(
        self,
        code: str,
        func: ast.FunctionDef
    ) -> Optional[str]:
        """Suggest method extraction."""
        # Simplified - would need sophisticated analysis
        return None
    
    def _has_type_hints(self, code: str) -> bool:
        """Check if code has type hints."""
        return '->' in code or (': ' in code and any(t in code for t in ['int', 'str', 'float', 'bool', 'List', 'Dict']))
    
    def _add_type_hints(self, code: str, tree: ast.AST) -> str:
        """Add type hints to code."""
        # Simplified - would need full AST manipulation
        return code
    
    def _add_docstring(self, code: str, func: ast.FunctionDef) -> str:
        """Add docstring to function."""
        lines = code.split('\n')
        func_line = func.lineno - 1
        
        # Generate docstring from function name
        docstring = f'    """{func.name.replace("_", " ").title()}."""'
        
        if func_line + 1 < len(lines):
            # Find insertion point (after function definition)
            insert_line = func_line + 1
            # Skip if already has docstring
            if insert_line < len(lines) and '"""' not in lines[insert_line]:
                lines.insert(insert_line, docstring)
        
        return '\n'.join(lines)
    
    def _is_descriptive(self, name: str) -> bool:
        """Check if name is descriptive."""
        return len(name) >= 5 and ('_' in name or name[0].isupper())
    
    def _improve_naming(self, code: str, func: ast.FunctionDef) -> str:
        """Improve function naming."""
        # Simple improvements
        old_name = func.name
        new_name = self._generate_better_name(old_name)
        
        if new_name != old_name:
            # Replace function name
            code = code.replace(f'def {old_name}(', f'def {new_name}(')
            # Replace calls (simplified)
            code = re.sub(rf'\b{old_name}\s*\(', f'{new_name}(', code)
        
        return code
    
    def _generate_better_name(self, name: str) -> str:
        """Generate a better name."""
        # Common improvements
        improvements = {
            'calc': 'calculate',
            'proc': 'process',
            'get': 'retrieve',
            'set': 'assign',
            'chk': 'check',
            'val': 'validate',
        }
        
        for short, long in improvements.items():
            if name.startswith(short):
                return name.replace(short, long, 1)
        
        # Add descriptive suffix if too short
        if len(name) < 5:
            return name + '_value'
        
        return name

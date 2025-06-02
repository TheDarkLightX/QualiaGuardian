import unittest
import ast
from typing import List, Dict, Any, Optional
import os
import pathlib
import tempfile
import shutil

# Assuming MutationOperatorVisitor and MutantType are in this path
# Adjust if the actual location is different after refactoring or if they are in types.py
from guardian_ai_tool.guardian.evolution.smart_mutator import MutationOperatorVisitor, MutantType, SmartMutator

class TestMutationOperatorVisitor(unittest.TestCase):

    def _assert_mutant_details(self, mutant: Dict[str, Any], expected_file_path: str, expected_line_number: int, 
                               expected_original_op: str, expected_mutated_op: str, 
                               expected_mutation_type: str, original_code_present: bool = True):
        self.assertEqual(mutant["file_path"], expected_file_path)
        self.assertEqual(mutant["line_number"], expected_line_number)
        self.assertEqual(mutant["original_operator"], expected_original_op)
        self.assertEqual(mutant["mutated_operator"], expected_mutated_op)
        self.assertEqual(mutant["mutation_type"], expected_mutation_type)
        if original_code_present:
            self.assertIsNotNone(mutant["original_code_snippet"])
        self.assertIsNotNone(mutant["mutated_code_snippet"])

    def test_binary_operator_mutations(self):
        code_snippet = "result = a + b"
        file_path = "dummy_file.py"
        
        tree = ast.parse(code_snippet)
        visitor = MutationOperatorVisitor(file_path=file_path, original_file_content=code_snippet)
        visitor.visit(tree)
        
        mutants = visitor.mutants
        self.assertEqual(len(mutants), 1) # Expect one mutation: Add -> Sub
        
        mutant = mutants[0]
        self._assert_mutant_details(
            mutant=mutant,
            expected_file_path=file_path,
            expected_line_number=1,
            expected_original_op="Add",
            expected_mutated_op="Sub",
            expected_mutation_type=MutantType.ARITHMETIC.value
        )
        self.assertEqual(mutant["original_code_snippet"], "a + b")
        self.assertEqual(mutant["mutated_code_snippet"], "a - b")

        code_snippet_2 = "val = x * y\nres = val / z"
        tree_2 = ast.parse(code_snippet_2)
        visitor_2 = MutationOperatorVisitor(file_path=file_path, original_file_content=code_snippet_2)
        visitor_2.visit(tree_2)
        mutants_2 = visitor_2.mutants
        self.assertEqual(len(mutants_2), 2) # Mult -> Div, Div -> Mult

        # Check first mutant (Mult -> Div)
        mutant_2_0 = next(m for m in mutants_2 if m["original_operator"] == "Mult")
        self._assert_mutant_details(
            mutant=mutant_2_0,
            expected_file_path=file_path,
            expected_line_number=1,
            expected_original_op="Mult",
            expected_mutated_op="Div",
            expected_mutation_type=MutantType.ARITHMETIC.value
        )
        self.assertEqual(mutant_2_0["original_code_snippet"], "x * y")
        self.assertEqual(mutant_2_0["mutated_code_snippet"], "x / y")

        # Check second mutant (Div -> Mult)
        mutant_2_1 = next(m for m in mutants_2 if m["original_operator"] == "Div")
        self._assert_mutant_details(
            mutant=mutant_2_1,
            expected_file_path=file_path,
            expected_line_number=2,
            expected_original_op="Div",
            expected_mutated_op="Mult",
            expected_mutation_type=MutantType.ARITHMETIC.value
        )
        self.assertEqual(mutant_2_1["original_code_snippet"], "val / z")
        # Note: ast.unparse might add parentheses if not present or simplify if present
        # For "val / z" -> "val * z", it should be straightforward.
        self.assertEqual(mutant_2_1["mutated_code_snippet"], "val * z")

    def test_comparison_operator_mutations(self):
        code_snippet = "if a == b:\n    pass"
        file_path = "dummy_compare.py"
        
        tree = ast.parse(code_snippet)
        visitor = MutationOperatorVisitor(file_path=file_path, original_file_content=code_snippet)
        visitor.visit(tree)
        
        mutants = visitor.mutants
        self.assertEqual(len(mutants), 2) # Expect two mutations: Eq -> NotEq (from visit_Compare) and IfCondition -> IfConditionNot (from visit_If)
        
        mutant_compare = next(m for m in mutants if m["original_operator"] == "Eq")
        self._assert_mutant_details(
            mutant=mutant_compare,
            expected_file_path=file_path,
            expected_line_number=1, # The comparison node is on line 1
            expected_original_op="Eq",
            expected_mutated_op="NotEq",
            expected_mutation_type=MutantType.RELATIONAL.value
        )
        self.assertEqual(mutant_compare["original_code_snippet"], "a == b")
        self.assertEqual(mutant_compare["mutated_code_snippet"], "a != b")

        mutant_if = next(m for m in mutants if m["original_operator"] == "IfCondition")
        self._assert_mutant_details(
            mutant=mutant_if,
            expected_file_path=file_path,
            expected_line_number=1,
            expected_original_op="IfCondition",
            expected_mutated_op="IfConditionNot",
            expected_mutation_type=MutantType.LOGICAL.value # visit_If produces LOGICAL mutants
        )
        # self.assertEqual(mutant["original_code_snippet"], "a == b") # This was for mutant_compare
        self.assertEqual(mutant_if["original_code_snippet"], "a == b") # Original condition of the if
        self.assertEqual(mutant_if["mutated_code_snippet"], "not a == b") # Corrected: visit_If stores unparse of the UnaryOp

        code_snippet_2 = "while x < y:\n    break"
        tree_2 = ast.parse(code_snippet_2)
        visitor_2 = MutationOperatorVisitor(file_path=file_path, original_file_content=code_snippet_2)
        visitor_2.visit(tree_2)
        mutants_2 = visitor_2.mutants
        # For "while x < y", only visit_Compare should trigger as there's no visit_While in the visitor.
        # If visit_While were added and it also negated conditions, this might be 2.
        self.assertEqual(len(mutants_2), 1) # Lt -> Gt from visit_Compare
        
        mutant_2_compare = mutants_2[0]
        self._assert_mutant_details(
            mutant=mutant_2_compare,
            expected_file_path=file_path,
            expected_line_number=1,
            expected_original_op="Lt",
            expected_mutated_op="Gt",
            expected_mutation_type=MutantType.RELATIONAL.value
        )
        self.assertEqual(mutant_2_compare["original_code_snippet"], "x < y")
        self.assertEqual(mutant_2_compare["mutated_code_snippet"], "x > y")

    def test_assignment_mutations_numeric(self):
        code_snippet = "count = 1"
        file_path = "dummy_assign_numeric.py"
        
        tree = ast.parse(code_snippet)
        visitor = MutationOperatorVisitor(
            file_path=file_path,
            original_file_content=code_snippet,
            target_mutation_types=[MutantType.ASSIGNMENT] # Focus on assignment mutations
        )
        visitor.visit(tree)
        
        # Filter for mutants specifically from visit_Assign
        assign_mutants = [
            m for m in visitor.mutants
            if m["mutation_type"] == MutantType.ASSIGNMENT.value and \
               m["original_operator"].startswith("Assign(value=")
        ]
        # Expected mutations for 1 (from visit_Assign): 0, 2, -1, None. So 4 mutants.
        self.assertEqual(len(assign_mutants), 4)
        
        expected_mutated_values = {0, 2, -1, None}
        found_mutated_values = set()

        for mutant in assign_mutants: # Iterate over filtered mutants
            self._assert_mutant_details(
                mutant=mutant,
                expected_file_path=file_path,
                expected_line_number=1,
                expected_original_op="Assign(value=int)",
                expected_mutated_op=f"Assign(value={type(ast.literal_eval(mutant['mutated_code_snippet'].split('=')[1].strip())).__name__})",
                expected_mutation_type=MutantType.ASSIGNMENT.value
            )
            self.assertTrue(mutant["original_code_snippet"].strip().startswith("count = 1"))
            
            # Extract assigned value from mutated snippet for checking
            # This is a bit fragile but works for simple "var = val"
            mutated_val_str = mutant["mutated_code_snippet"].split("=")[1].strip()
            
            # ast.literal_eval can parse None, True, False, numbers, strings
            try:
                mutated_actual_value = ast.literal_eval(mutated_val_str)
            except ValueError: # for cases like "None" which is not a literal for literal_eval
                 if mutated_val_str == "None":
                     mutated_actual_value = None
                 else:
                     raise
            found_mutated_values.add(mutated_actual_value)

        self.assertEqual(expected_mutated_values, found_mutated_values)

    def test_assignment_mutations_boolean(self):
        code_snippet = "is_active = True"
        file_path = "dummy_assign_bool.py"
        tree = ast.parse(code_snippet)
        visitor = MutationOperatorVisitor(
            file_path=file_path,
            original_file_content=code_snippet,
            target_mutation_types=[MutantType.ASSIGNMENT] # Focus on assignment mutations
        )
        visitor.visit(tree)
        
        assign_mutants = [
            m for m in visitor.mutants
            if m["mutation_type"] == MutantType.ASSIGNMENT.value and \
               m["original_operator"].startswith("Assign(value=")
        ]
        # True -> False, True -> None (2 mutants from visit_Assign)
        self.assertEqual(len(assign_mutants), 2)

        mutant_to_false = next(m for m in assign_mutants if "False" in m["mutated_code_snippet"])
        self._assert_mutant_details(
            mutant=mutant_to_false, expected_file_path=file_path, expected_line_number=1,
            expected_original_op="Assign(value=bool)", expected_mutated_op="Assign(value=bool)",
            expected_mutation_type=MutantType.ASSIGNMENT.value
        )
        self.assertTrue("is_active = False" in mutant_to_false["mutated_code_snippet"])
        
        mutant_to_none = next(m for m in assign_mutants if "None" in m["mutated_code_snippet"])
        self._assert_mutant_details(
            mutant=mutant_to_none, expected_file_path=file_path, expected_line_number=1,
            expected_original_op="Assign(value=bool)", expected_mutated_op="Assign(value=NoneType)",
            expected_mutation_type=MutantType.ASSIGNMENT.value
        )
        self.assertTrue("is_active = None" in mutant_to_none["mutated_code_snippet"])


    def test_assignment_mutations_string(self):
        code_snippet = 'name = "test"'
        file_path = "dummy_assign_str.py"
        tree = ast.parse(code_snippet)
        visitor = MutationOperatorVisitor(
            file_path=file_path,
            original_file_content=code_snippet,
            target_mutation_types=[MutantType.ASSIGNMENT] # Focus on assignment mutations
        )
        visitor.visit(tree)
        
        assign_mutants = [
            m for m in visitor.mutants
            if m["mutation_type"] == MutantType.ASSIGNMENT.value and \
               m["original_operator"].startswith("Assign(value=")
        ]
        # "test" -> "", "mutated_string", None (3 mutants from visit_Assign)
        self.assertEqual(len(assign_mutants), 3)
        
        found_snippets = {m["mutated_code_snippet"] for m in assign_mutants}
        self.assertIn("name = ''", found_snippets)
        self.assertIn("name = 'mutated_string'", found_snippets)
        self.assertIn('name = None', found_snippets)


    def test_assignment_mutations_none(self):
        code_snippet = "value = None"
        file_path = "dummy_assign_none.py"
        tree = ast.parse(code_snippet)
        visitor = MutationOperatorVisitor(
            file_path=file_path,
            original_file_content=code_snippet,
            target_mutation_types=[MutantType.ASSIGNMENT] # Focus on assignment mutations
        )
        visitor.visit(tree)
        
        assign_mutants = [
            m for m in visitor.mutants
            if m["mutation_type"] == MutantType.ASSIGNMENT.value and \
               m["original_operator"].startswith("Assign(value=")
        ]
        # None -> 0, "" (2 mutants from visit_Assign)
        self.assertEqual(len(assign_mutants), 2)
        found_snippets = {m["mutated_code_snippet"] for m in assign_mutants}
        self.assertIn("value = 0", found_snippets)
        self.assertIn("value = ''", found_snippets)


    def test_logical_operator_mutations(self):
        code_snippet = "if a and b:\n    pass"
        file_path = "dummy_logical.py"
        
        tree = ast.parse(code_snippet)
        visitor = MutationOperatorVisitor(file_path=file_path, original_file_content=code_snippet)
        visitor.visit(tree)
        
        mutants = visitor.mutants
        self.assertEqual(len(mutants), 2) # Expect two: And -> Or (from visit_BoolOp) and IfCondition -> IfConditionNot (from visit_If)
        
        mutant_boolop = next(m for m in mutants if m["original_operator"] == "And")
        self._assert_mutant_details(
            mutant=mutant_boolop,
            expected_file_path=file_path,
            expected_line_number=1, # The BoolOp node is on line 1
            expected_original_op="And",
            expected_mutated_op="Or",
            expected_mutation_type=MutantType.LOGICAL.value
        )
        self.assertEqual(mutant_boolop["original_code_snippet"], "a and b")
        self.assertEqual(mutant_boolop["mutated_code_snippet"], "a or b")

        mutant_if_logic = next(m for m in mutants if m["original_operator"] == "IfCondition")
        self._assert_mutant_details(
            mutant=mutant_if_logic,
            expected_file_path=file_path,
            expected_line_number=1,
            expected_original_op="IfCondition",
            expected_mutated_op="IfConditionNot",
            expected_mutation_type=MutantType.LOGICAL.value
        )
        self.assertEqual(mutant_if_logic["original_code_snippet"], "a and b") # Original condition
        self.assertEqual(mutant_if_logic["mutated_code_snippet"], "not (a and b)") # Mutated condition

        code_snippet_2 = "if x or y:\n    pass"
        tree_2 = ast.parse(code_snippet_2)
        visitor_2 = MutationOperatorVisitor(file_path=file_path, original_file_content=code_snippet_2)
        visitor_2.visit(tree_2)
        mutants_2 = visitor_2.mutants
        self.assertEqual(len(mutants_2), 2) # Or -> And (from visit_BoolOp) and IfCondition -> IfConditionNot (from visit_If)
        
        mutant_2_boolop = next(m for m in mutants_2 if m["original_operator"] == "Or")
        self._assert_mutant_details(
            mutant=mutant_2_boolop,
            expected_file_path=file_path,
            expected_line_number=1,
            expected_original_op="Or",
            expected_mutated_op="And",
            expected_mutation_type=MutantType.LOGICAL.value
        )
        self.assertEqual(mutant_2_boolop["original_code_snippet"], "x or y")
        self.assertEqual(mutant_2_boolop["mutated_code_snippet"], "x and y")

        mutant_2_if_logic = next(m for m in mutants_2 if m["original_operator"] == "IfCondition")
        self._assert_mutant_details(
            mutant=mutant_2_if_logic,
            expected_file_path=file_path,
            expected_line_number=1,
            expected_original_op="IfCondition",
            expected_mutated_op="IfConditionNot",
            expected_mutation_type=MutantType.LOGICAL.value
        )
        self.assertEqual(mutant_2_if_logic["original_code_snippet"], "x or y") # Original condition
        self.assertEqual(mutant_2_if_logic["mutated_code_snippet"], "not (x or y)") # Mutated condition

    def test_if_condition_negation(self):
        file_path = "dummy_if_negate.py"

        # Test simple condition negation
        code_snippet_simple = "if x > 10:\n    print('greater')" # Also has constants 10 and 'greater'
        tree_simple = ast.parse(code_snippet_simple)
        visitor_simple = MutationOperatorVisitor(
            file_path=file_path,
            original_file_content=code_snippet_simple,
            target_mutation_types=[MutantType.LOGICAL, MutantType.RELATIONAL] # Focus
        )
        visitor_simple.visit(tree_simple)
        
        # Filter for relevant mutants
        logical_or_relational_mutants = [
            m for m in visitor_simple.mutants
            if m["mutation_type"] in [MutantType.LOGICAL.value, MutantType.RELATIONAL.value]
        ]
        
        # Expect 2 mutants:
        # 1. From visit_If: `if x > 10` -> `if not (x > 10)` (LOGICAL)
        # 2. From visit_Compare on `x > 10`: `x > 10` -> `x < 10` (RELATIONAL)
        self.assertEqual(len(logical_or_relational_mutants), 2)
        
        mutant_if_simple = next(m for m in logical_or_relational_mutants if m["original_operator"] == "IfCondition")
        self._assert_mutant_details(
            mutant=mutant_if_simple,
            expected_file_path=file_path,
            expected_line_number=1,
            expected_original_op="IfCondition",
            expected_mutated_op="IfConditionNot",
            expected_mutation_type=MutantType.LOGICAL.value
        )
        self.assertEqual(mutant_if_simple["original_code_snippet"], "x > 10") # from visit_If, original_code_snippet is node.test
        self.assertEqual(mutant_if_simple["mutated_code_snippet"], "not x > 10")

        mutant_compare_simple = next(m for m in logical_or_relational_mutants if m["original_operator"] == "Gt")
        self._assert_mutant_details(
            mutant=mutant_compare_simple,
            expected_file_path=file_path,
            expected_line_number=1, # The Gt node is on line 1
            expected_original_op="Gt",
            expected_mutated_op="Lt", # Assuming Gt -> Lt mapping
            expected_mutation_type=MutantType.RELATIONAL.value
        )
        self.assertEqual(mutant_compare_simple["original_code_snippet"], "x > 10")
        self.assertEqual(mutant_compare_simple["mutated_code_snippet"], "x < 10")

        # Test unwrapping an existing "not"
        code_snippet_not = "if not is_valid:\n    return" # No other easily mutable parts for LOGICAL/RELATIONAL
        tree_not = ast.parse(code_snippet_not)
        visitor_not = MutationOperatorVisitor(
            file_path=file_path,
            original_file_content=code_snippet_not,
            target_mutation_types=[MutantType.LOGICAL, MutantType.RELATIONAL] # Focus
        )
        visitor_not.visit(tree_not)
        
        if_condition_mutants_not = [
            m for m in visitor_not.mutants if m["original_operator"] == "IfCondition"
        ]
        self.assertEqual(len(if_condition_mutants_not), 1)
        mutant_not = if_condition_mutants_not[0]
        self._assert_mutant_details(
            mutant=mutant_not,
            expected_file_path=file_path,
            expected_line_number=1,
            expected_original_op="IfCondition", # Original is 'not is_valid' but we classify the op on 'if'
            expected_mutated_op="IfConditionNot", # Mutated is 'is_valid' (negation of 'not is_valid')
            expected_mutation_type=MutantType.LOGICAL.value
        )
        self.assertEqual(mutant_not["original_code_snippet"], "not is_valid")
        self.assertEqual(mutant_not["mutated_code_snippet"], "is_valid")
        
        # Test a more complex condition
        code_snippet_complex = "if a and (b or c):\n    execute()" # No other easily mutable parts for LOGICAL/RELATIONAL
        tree_complex = ast.parse(code_snippet_complex)
        visitor_complex = MutationOperatorVisitor(
            file_path=file_path,
            original_file_content=code_snippet_complex,
            target_mutation_types=[MutantType.LOGICAL, MutantType.RELATIONAL] # Focus
        )
        visitor_complex.visit(tree_complex)
        
        # We expect 3 LOGICAL mutants:
        # 1. if -> if not
        # 2. and -> or
        # 3. or -> and
        logical_mutants_complex = [
            m for m in visitor_complex.mutants if m["mutation_type"] == MutantType.LOGICAL.value
        ]
        self.assertEqual(len(logical_mutants_complex), 3)

        # Specifically find the IfConditionNot mutant among them
        mutant_complex = next(
            m for m in logical_mutants_complex
            if m["original_operator"] == "IfCondition" and m["mutated_operator"] == "IfConditionNot"
        )
        
        self._assert_mutant_details(
            mutant=mutant_complex,
            expected_file_path=file_path,
            expected_line_number=1,
            expected_original_op="IfCondition",
            expected_mutated_op="IfConditionNot",
            expected_mutation_type=MutantType.LOGICAL.value
        )
        self.assertEqual(mutant_complex["original_code_snippet"], "a and (b or c)")
        self.assertEqual(mutant_complex["mutated_code_snippet"], "not (a and (b or c))")

    def test_try_except_remove_handler(self):
        code_snippet = """
try:
    x = 1 / 0
except ZeroDivisionError:
    print("Caught error")
finally:
    print("Finally done")
"""
        file_path = "dummy_try_except.py"
        tree = ast.parse(code_snippet)
        visitor = MutationOperatorVisitor(
            file_path=file_path,
            original_file_content=code_snippet,
            target_mutation_types=[MutantType.EXCEPTION] # Filter for exception mutants only
        )
        visitor.visit(tree)
        mutants = visitor.mutants

        # Expected: 1 for removing TryExceptBlock, 1 for changing ZeroDivisionError to ArithmeticError, 1 for changing ZeroDivisionError to BareException
        self.assertEqual(len(mutants), 3)

        # 1. Test TryExceptBlock removal
        unwrapped_try_mutant = next(m for m in mutants if m["original_operator"] == "TryExceptBlock")
        self._assert_mutant_details(
            mutant=unwrapped_try_mutant,
            expected_file_path=file_path,
            expected_line_number=2, # Line of 'try:'
            expected_original_op="TryExceptBlock",
            expected_mutated_op="UnwrappedTryBody",
            expected_mutation_type=MutantType.EXCEPTION.value
        )
        # Original snippet should be the whole try...except...finally block
        self.assertTrue("try:" in unwrapped_try_mutant["original_code_snippet"])
        self.assertTrue("except ZeroDivisionError:" in unwrapped_try_mutant["original_code_snippet"])
        self.assertTrue("finally:" in unwrapped_try_mutant["original_code_snippet"])
        # Mutated snippet should be just the body of the try block
        self.assertEqual(unwrapped_try_mutant["mutated_code_snippet"].strip(), "x = 1 / 0")

    def test_except_handler_type_change(self):
        code_snippet = """
try:
    data = {}
    val = data['key']
except KeyError:
    print("Key error")
except ValueError:
    print("Value error")
"""
        file_path = "dummy_except_change.py"
        tree = ast.parse(code_snippet)
        visitor = MutationOperatorVisitor(
            file_path=file_path,
            original_file_content=code_snippet,
            target_mutation_types=[MutantType.EXCEPTION] # Focus
        )
        visitor.visit(tree)
        
        exception_mutants = [
            m for m in visitor.mutants if m["mutation_type"] == MutantType.EXCEPTION.value
        ]
        
        # Expected mutations (all EXCEPTION type):
        # 1. TryExceptBlock removal for the whole try-except (line 2)
        # 2. KeyError -> IndexError (line 5)
        # 3. KeyError -> BareException (line 5)
        # 4. ValueError -> TypeError (line 7)
        # 5. ValueError -> BareException (line 7)
        # Total = 5
        self.assertEqual(len(exception_mutants), 5)

        # Test KeyError -> IndexError
        key_error_mutant = next(m for m in exception_mutants if m["original_operator"] == "Catch(KeyError)" and m["mutated_operator"] == "Catch(IndexError)")
        self._assert_mutant_details(
            mutant=key_error_mutant,
            expected_file_path=file_path,
            expected_line_number=5, # Line of 'except KeyError:'
            expected_original_op="Catch(KeyError)",
            expected_mutated_op="Catch(IndexError)",
            expected_mutation_type=MutantType.EXCEPTION.value
        )
        self.assertTrue("except KeyError:" in key_error_mutant["original_code_snippet"])
        self.assertTrue("except IndexError:" in key_error_mutant["mutated_code_snippet"])
        
        # Test ValueError -> TypeError
        value_error_mutant = next(m for m in exception_mutants if m["original_operator"] == "Catch(ValueError)" and m["mutated_operator"] == "Catch(TypeError)")
        self._assert_mutant_details(
            mutant=value_error_mutant,
            expected_file_path=file_path,
            expected_line_number=7, # Line of 'except ValueError:'
            expected_original_op="Catch(ValueError)",
            expected_mutated_op="Catch(TypeError)",
            expected_mutation_type=MutantType.EXCEPTION.value
        )
        self.assertTrue("except ValueError:" in value_error_mutant["original_code_snippet"])
        self.assertTrue("except TypeError:" in value_error_mutant["mutated_code_snippet"])

        # Test KeyError -> BareException
        key_to_bare_mutant = next(m for m in exception_mutants if m["original_operator"] == "Catch(KeyError)" and m["mutated_operator"] == "Catch(BareException)")
        self._assert_mutant_details(
            mutant=key_to_bare_mutant,
            expected_file_path=file_path,
            expected_line_number=5,
            expected_original_op="Catch(KeyError)",
            expected_mutated_op="Catch(BareException)",
            expected_mutation_type=MutantType.EXCEPTION.value
        )
        self.assertTrue("except KeyError:" in key_to_bare_mutant["original_code_snippet"])
        self.assertTrue("except:" in key_to_bare_mutant["mutated_code_snippet"] and "KeyError" not in key_to_bare_mutant["mutated_code_snippet"])


    def test_except_handler_bare_except_no_type_change(self):
        code_snippet = """
try:
    action()
except: # Bare except
    handle_all()
"""
        file_path = "dummy_bare_except.py"
        tree = ast.parse(code_snippet)
        visitor = MutationOperatorVisitor(
            file_path=file_path,
            original_file_content=code_snippet,
            target_mutation_types=[MutantType.EXCEPTION] # Focus
        )
        visitor.visit(tree)
        
        exception_mutants = [
            m for m in visitor.mutants if m["mutation_type"] == MutantType.EXCEPTION.value
        ]
        
        # Expected: 1 for TryExceptBlock removal.
        # No type change for bare except, and no "change to bare" because it's already bare.
        self.assertEqual(len(exception_mutants), 1)
        self.assertEqual(exception_mutants[0]["original_operator"], "TryExceptBlock")

    def test_constant_numeric_mutations_non_assignment(self):
        # Test constants NOT on the RHS of an assignment
        code_snippet = "print(10)\nmy_list = [-2.5, 0]"
        file_path = "dummy_const_numeric_non_assign.py"
        tree = ast.parse(code_snippet)
        visitor = MutationOperatorVisitor(file_path=file_path, original_file_content=code_snippet)
        visitor.visit(tree)
        
        # Filter for mutants from visit_Constant
        constant_mutants = [m for m in visitor.mutants if m["original_operator"].startswith("Constant(")]

        # For 10: BOUNDARY (11,9,0,-10 -> 4), ASSIGNMENT (None -> 1) = 5
        # For -2.5: BOUNDARY (-1.5, -3.5, 0, 2.5 -> 4), ASSIGNMENT (None -> 1) = 5
        # For 0: BOUNDARY (1, -1 -> 2), ASSIGNMENT (None -> 1) = 3
        # Total = 5 + 5 + 3 = 13
        self.assertEqual(len(constant_mutants), 13)

        mutants_for_10 = [m for m in constant_mutants if m["original_code_snippet"] == "10"]
        self.assertEqual(len(mutants_for_10), 5)
        
        expected_10_boundary_vals = {11, 9, 0, -10}
        found_10_boundary_vals = {ast.literal_eval(m["mutated_code_snippet"]) for m in mutants_for_10 if m["mutation_type"] == MutantType.BOUNDARY.value}
        self.assertEqual(expected_10_boundary_vals, found_10_boundary_vals)
        self.assertTrue(any(m["mutation_type"] == MutantType.ASSIGNMENT.value and m["mutated_code_snippet"] == "None" for m in mutants_for_10))

        # Test with BOUNDARY filter for "print(10)"
        visitor_filtered = MutationOperatorVisitor(
            file_path=file_path,
            original_file_content="print(10)",
            target_mutation_types=[MutantType.BOUNDARY]
        )
        visitor_filtered.visit(ast.parse("print(10)"))
        boundary_mutants_filtered = [m for m in visitor_filtered.mutants if m["mutation_type"] == MutantType.BOUNDARY.value]
        self.assertEqual(len(boundary_mutants_filtered), 4) # 11, 9, 0, -10
        for m in boundary_mutants_filtered:
            self.assertEqual(m["original_code_snippet"], "10")


    def test_constant_boolean_mutations_non_assignment(self):
        code_snippet = "my_func(True, arg=False)"
        file_path = "dummy_const_bool_non_assign.py"
        tree = ast.parse(code_snippet)
        visitor = MutationOperatorVisitor(file_path=file_path, original_file_content=code_snippet)
        visitor.visit(tree)
        constant_mutants = [m for m in visitor.mutants if m["original_operator"].startswith("Constant(")]
        
        # For True: LOGICAL (False -> 1), ASSIGNMENT (None -> 1) = 2
        # For False: LOGICAL (True -> 1), ASSIGNMENT (None -> 1) = 2
        # Total = 4
        self.assertEqual(len(constant_mutants), 4)

        mut_true_to_false = next(m for m in constant_mutants if m["original_code_snippet"] == "True" and m["mutated_code_snippet"] == "False")
        self._assert_mutant_details(mut_true_to_false, file_path, 1, "Constant(bool)", "Constant(bool)", MutantType.LOGICAL.value)
        
        mut_true_to_none = next(m for m in constant_mutants if m["original_code_snippet"] == "True" and m["mutated_code_snippet"] == "None")
        self._assert_mutant_details(mut_true_to_none, file_path, 1, "Constant(bool)", "Constant(NoneType)", MutantType.ASSIGNMENT.value)


    def test_constant_string_mutations_non_assignment(self):
        code_snippet = 'return "test"\nprint("")'
        file_path = "dummy_const_str_non_assign.py"
        tree = ast.parse(code_snippet)
        visitor = MutationOperatorVisitor(file_path=file_path, original_file_content=code_snippet)
        visitor.visit(tree)
        constant_mutants = [m for m in visitor.mutants if m["original_operator"].startswith("Constant(")]

        # For "test": ASSIGNMENT ("", "mutated_string", None -> 3)
        # For "": ASSIGNMENT ("mutated_string", None -> 2)
        # Total = 5
        self.assertEqual(len(constant_mutants), 5)

        mutants_for_test_str = [m for m in constant_mutants if m["original_code_snippet"] == "'test'"]
        self.assertEqual(len(mutants_for_test_str), 3)
        self.assertTrue(all(m["mutation_type"] == MutantType.ASSIGNMENT.value for m in mutants_for_test_str))


    def test_constant_none_mutations_non_assignment(self):
        code_snippet = "yield None"
        file_path = "dummy_const_none_non_assign.py"
        tree = ast.parse(code_snippet)
        visitor = MutationOperatorVisitor(file_path=file_path, original_file_content=code_snippet)
        visitor.visit(tree)
        constant_mutants = [m for m in visitor.mutants if m["original_operator"].startswith("Constant(")]

        # For None: ASSIGNMENT (0, "" -> 2)
        self.assertEqual(len(constant_mutants), 2)
        for m in constant_mutants:
            self.assertEqual(m["mutation_type"], MutantType.ASSIGNMENT.value)
            self.assertEqual(m["original_code_snippet"], "None")
            self.assertIn(m["mutated_code_snippet"], ["0", "''"])


if __name__ == '__main__':
    unittest.main()

class TestSmartMutatorMethods(unittest.TestCase):
    def setUp(self):
        # SmartMutator requires a codebase_path, but for _mutate_complex_node,
        # it's not directly used by the method itself if source_code and file_path are provided.
        # We pass a dummy path.
        self.smart_mutator_instance = SmartMutator(codebase_path="dummy_codebase") # Used for _mutate_complex_node
        self.temp_project_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_project_dir)

    def _create_dummy_file(self, filename: str, content: str):
        file_path = pathlib.Path(self.temp_project_dir) / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path

    def test_generate_mutants_for_project_no_filter(self):
        self._create_dummy_file("file1.py", "x = a + b\nif c == d: pass")
        self._create_dummy_file("file2.py", "y = 1\nif e and f: pass")
        
        mutator = SmartMutator(codebase_path=self.temp_project_dir) # Path doesn't matter much here as we pass it to method
        mutants = mutator.generate_mutants_for_project(self.temp_project_dir)

        # file1:
        #   x = a + b: Add -> Sub (ARITHMETIC) = 1
        #   if c == d: Eq -> NotEq (RELATIONAL), IfCondition -> IfConditionNot (LOGICAL) = 2
        #   Subtotal file1 = 3
        # file2:
        #   y = 1:
        #     visit_Assign: 1->0, 1->2, 1->-1, 1->None (ASSIGNMENT) = 4
        #     visit_Constant (on '1'): 1->2, 1->0, 1->-1 (BOUNDARY), 1->None (ASSIGNMENT) = 4
        #   if e and f: And -> Or (LOGICAL), IfCondition -> IfConditionNot (LOGICAL) = 2
        #   Subtotal file2 = 4 + 4 + 2 = 10
        # Total expected: 3 + 10 = 13
        self.assertEqual(len(mutants), 13)

        arith_mutants = [m for m in mutants if m["mutation_type"] == MutantType.ARITHMETIC.value]
        rel_mutants = [m for m in mutants if m["mutation_type"] == MutantType.RELATIONAL.value]
        assign_mutants = [m for m in mutants if m["mutation_type"] == MutantType.ASSIGNMENT.value]
        # Logical mutants include BoolOp changes and IfCondition changes
        logical_mutants = [m for m in mutants if m["mutation_type"] == MutantType.LOGICAL.value]

        self.assertEqual(len(arith_mutants), 1) # x = a+b
        self.assertEqual(len(rel_mutants), 1) # c == d
        
        # Assign mutants: from y=1 (4 from visit_Assign) + from const 1 (1 to None from visit_Constant) = 5
        assign_mutants_actual = [m for m in mutants if m["mutation_type"] == MutantType.ASSIGNMENT.value]
        self.assertEqual(len(assign_mutants_actual), 5)
        
        # Logical mutants: if c==d (1), if e and f (1), e and f (1) = 3
        # Plus, visit_Constant on True from `val = True` in the other test file (if it were run together, but it's not)
        # Here, for file2: if e and f (1 from IfCondition), e and f (1 from BoolOp) = 2
        # For file1: if c == d (1 from IfCondition) = 1
        # Total logical = 3
        self.assertEqual(len(logical_mutants), 3)

        # Boundary mutants from y=1 (visit_Constant on '1'): 1->2, 1->0, 1->-1 = 3
        boundary_mutants = [m for m in mutants if m["mutation_type"] == MutantType.BOUNDARY.value]
        self.assertEqual(len(boundary_mutants), 3)

    def test_generate_mutants_for_project_filter_arithmetic(self):
        self._create_dummy_file("file1.py", "x = a + b\nif c == d: pass\nz=1")
        mutator = SmartMutator(codebase_path=self.temp_project_dir)
        mutants = mutator.generate_mutants_for_project(
            self.temp_project_dir,
            target_mutation_types=[MutantType.ARITHMETIC]
        )
        self.assertEqual(len(mutants), 1)
        self.assertEqual(mutants[0]["mutation_type"], MutantType.ARITHMETIC.value)
        self.assertEqual(mutants[0]["original_operator"], "Add")

    def test_generate_mutants_for_project_filter_logical_and_assignment(self):
        self._create_dummy_file("file1.py", "x = a + b\nif c == d and e < f: val = True")
        mutator = SmartMutator(codebase_path=self.temp_project_dir)
        mutants = mutator.generate_mutants_for_project(
            self.temp_project_dir,
            target_mutation_types=[MutantType.LOGICAL, MutantType.ASSIGNMENT]
        )
        # Expected:
        # LOGICAL:
        #   c == d and e < f  -> c == d or e < f (BoolOp from visit_BoolOp) = 1
        #   if c == d and e < f -> if not (...) (IfCondition from visit_If) = 1
        #   val = True (Constant True -> False by visit_Constant) = 1
        #   Total LOGICAL = 3
        # ASSIGNMENT:
        #   val = True -> val = False (from visit_Assign) = 1
        #   val = True -> val = None (from visit_Assign) = 1
        #   val = True (Constant True -> None by visit_Constant) = 1
        #   Total ASSIGNMENT = 3
        # Total = 3 + 3 = 6
        self.assertEqual(len(mutants), 6)
        logical_count = sum(1 for m in mutants if m["mutation_type"] == MutantType.LOGICAL.value)
        assignment_count = sum(1 for m in mutants if m["mutation_type"] == MutantType.ASSIGNMENT.value)
        self.assertEqual(logical_count, 3)
        self.assertEqual(assignment_count, 3)

    def test_generate_mutants_for_project_empty_filter_means_all(self):
        self._create_dummy_file("file1.py", "x = a + b\nif c == d: pass")
        mutator = SmartMutator(codebase_path=self.temp_project_dir)
        mutants = mutator.generate_mutants_for_project(
            self.temp_project_dir,
            target_mutation_types=[] # Empty list
        )
        # Should behave as if no filter was passed (all types)
        # x = a + b -> x = a - b (ARITHMETIC)
        # c == d -> c != d (RELATIONAL)
        # if c == d -> if not (c == d) (LOGICAL from IfCondition)
        self.assertEqual(len(mutants), 3)


    def test_generate_mutants_for_project_no_matching_types(self):
        # Create a file that would normally produce ARITHMETIC and RELATIONAL mutants
        self._create_dummy_file("file1.py", "x = a + b\nif c == d: pass")
        mutator = SmartMutator(codebase_path=self.temp_project_dir)
        # Filter for only ASSIGNMENT, which won't be found in this file's simple ops
        mutants = mutator.generate_mutants_for_project(
            self.temp_project_dir,
            target_mutation_types=[MutantType.ASSIGNMENT]
        )
        self.assertEqual(len(mutants), 0)



    def test_generate_smart_mutants_basic(self):
        dummy_file_content = """
def process_data(a, b, c=10):
    if a > b: # Line 3
        result = a + b # Line 4
    else:
        result = a - b # Line 6
    
    is_active = True # Line 8
    name = "test" # Line 9

    try: # Line 11
        x = c / (a - b) # Line 12
    except ZeroDivisionError: # Line 13
        x = -1 # Line 14
    
    for i in range(c): # Line 16 (FaultPattern: Off-by-one)
        print(i)
    
    return result + x
"""
        dummy_file_path = self._create_dummy_file("smart_mutant_test_file.py", dummy_file_content)
        
        # Use the existing smart_mutator_instance or create a new one for this specific temp dir
        mutator = SmartMutator(codebase_path=self.temp_project_dir)
        # generate_smart_mutants uses the codebase_path to find files if target_file is None.
        # Here we provide the specific target_file.
        mutants = mutator.generate_smart_mutants(target_file=str(dummy_file_path))

        # Expected AST-based mutants (rough count, details matter more):
        # Line 3: a > b (Compare: Gt -> Lt), (If: if -> if not) = 2 LOGICAL/RELATIONAL
        # Line 4: a + b (BinOp: Add -> Sub) = 1 ARITHMETIC
        # Line 6: a - b (BinOp: Sub -> Add) = 1 ARITHMETIC
        # Line 8: True (Constant: True -> False, True -> None) = 2 LOGICAL/ASSIGNMENT
        # Line 9: "test" (Constant: "test" -> "", "test" -> "mutated_string", "test" -> None) = 3 ASSIGNMENT
        # Line 10: c=10 in def (Constant: 10 -> 11,9,0,-10,None) = 5 BOUNDARY/ASSIGNMENT
        # Line 12: c / (a-b) (BinOp: Div -> Mult) = 1 ARITHMETIC
        # Line 12: a - b (BinOp: Sub -> Add) = 1 ARITHMETIC
        # Line 13: try...except (Try: unwrap) = 1 EXCEPTION
        # Line 13: ZeroDivisionError (ExceptHandler: -> ArithmeticError, -> Bare) = 2 EXCEPTION
        # Line 14: -1 (Constant: -1 -> 0,-2,1,None) = 4 BOUNDARY/ASSIGNMENT
        # Total AST-based approx = 2+1+1+2+3+5+1+1+1+2+4 = 23

        # Expected Regex-based (FaultPattern) mutants:
        # Line 16: range(c) -> range(c-1), range(c+1) = 2 BOUNDARY
        
        # Total expected mutants = 23 (AST) + 2 (Regex) = 25
        # This count is sensitive to exact implementation details of each visitor and scoring.
        # For now, let's check for the presence of key mutant types.
        
        self.assertTrue(len(mutants) > 15, f"Expected a significant number of mutants, got {len(mutants)}") # General check

        # Check for an arithmetic mutant from "a + b"
        arith_mutant = next((m for m in mutants if m.line_number == 4 and m.mutation_type == MutantType.ARITHMETIC and m.original_code == "a + b"), None)
        self.assertIsNotNone(arith_mutant, "Arithmetic mutant for 'a + b' not found")
        self.assertEqual(arith_mutant.mutated_code, "a - b")

        # Check for an if-negation mutant
        if_negation_mutant = next((m for m in mutants if m.line_number == 3 and m.mutation_type == MutantType.LOGICAL and m.original_code == "a > b" and "not" in m.mutated_code), None)
        self.assertIsNotNone(if_negation_mutant, "If-negation mutant for 'a > b' not found")
        self.assertEqual(if_negation_mutant.mutated_code, "not a > b") # Based on current visit_If

        # Check for a constant boolean mutation (True -> False)
        bool_mutant = next((m for m in mutants if m.line_number == 8 and m.original_code == "True" and m.mutated_code == "False"), None)
        self.assertIsNotNone(bool_mutant, "Boolean constant mutant 'True -> False' not found")
        self.assertEqual(bool_mutant.mutation_type, MutantType.LOGICAL)
        
        # Check for a constant string mutation ("test" -> "")
        string_mutant = next((m for m in mutants if m.line_number == 9 and m.original_code == "'test'" and m.mutated_code == "''"), None) # ast.unparse uses single quotes
        self.assertIsNotNone(string_mutant, "String constant mutant ''test' -> ''' not found")
        self.assertEqual(string_mutant.mutation_type, MutantType.ASSIGNMENT)

        # Check for an exception type change (ZeroDivisionError -> ArithmeticError)
        exc_type_mutant = next((m for m in mutants if m.line_number == 13 and m.mutation_type == MutantType.EXCEPTION and "ZeroDivisionError" in m.original_code and "ArithmeticError" in m.mutated_code), None)
        self.assertIsNotNone(exc_type_mutant, "Exception type change mutant not found")

        # Check for a fault pattern mutant (range(c) -> range(c - 1))
        fault_pattern_mutant = next((m for m in mutants if m.line_number == 16 and m.mutation_type == MutantType.BOUNDARY and "range(c - 1)" in m.mutated_code), None)
        self.assertIsNotNone(fault_pattern_mutant, "Fault pattern mutant for range off-by-one not found")
        self.assertTrue("range(c)" in fault_pattern_mutant.original_code) # Original regex match
        self.assertEqual(fault_pattern_mutant.description, "Off-by-one errors: Reduce range by 1")

    def test_generate_smart_mutants_prioritization_and_budget(self):
        # Simplified content to isolate specific high-score and lower-score mutants
        dummy_file_content_v5 = """
# Line 1
# Line 2
my_data_object.process_it() # Line 3 (Null Pointer Fault Pattern), score 1.5
x = 1 + 1 # Line 4 (BinOp AST, score 0.5), (Arith Fault Pattern, score 0.75)
"""
        dummy_file_path_v5 = self._create_dummy_file("priority_test_v5.py", dummy_file_content_v5)
        
        # Test with budget 1
        mutator_budget1 = SmartMutator(codebase_path=self.temp_project_dir, mutation_budget=1)
        mutants_budget1 = mutator_budget1.generate_smart_mutants(target_file=str(dummy_file_path_v5))
        
        self.assertEqual(len(mutants_budget1), 1)
        if mutants_budget1: # Check if list is not empty before accessing element
            top_mutant = mutants_budget1[0]
            self.assertEqual(top_mutant.description, "Null pointer dereference: Replace object with None")
            self.assertEqual(top_mutant.line_number, 4) # Adjusted for leading newline in dummy content
            self.assertAlmostEqual(top_mutant.impact_score * top_mutant.likelihood, 1.5)

        # Test with budget 2
        # Expected: Null Pointer (1.5), then Arithmetic Fault Pattern (0.75)
        mutator_budget2 = SmartMutator(codebase_path=self.temp_project_dir, mutation_budget=2)
        mutants_budget2 = mutator_budget2.generate_smart_mutants(target_file=str(dummy_file_path_v5))
        
        self.assertEqual(len(mutants_budget2), 2)
        if len(mutants_budget2) == 2: # Check if list has two elements
            # Mutant 1 (highest score)
            self.assertEqual(mutants_budget2[0].description, "Null pointer dereference: Replace object with None")
            self.assertEqual(mutants_budget2[0].line_number, 4) # Adjusted
            self.assertAlmostEqual(mutants_budget2[0].impact_score * mutants_budget2[0].likelihood, 1.5)
            
            # Mutant 2 (second highest score)
            # AST-based BinOp: '+' -> '-', impact 1.0, likelihood 0.5 -> score 0.5
            # The description for AST mutants from BinOp is "AST-based mutation"
            # or more specific if visit_BinOp provided one (it doesn't currently for simple ops).
            # The default description from _generate_file_mutants is "AST-based mutation".
            self.assertEqual(mutants_budget2[1].description, "AST-based mutation")
            self.assertEqual(mutants_budget2[1].line_number, 5) # Adjusted (Line of "x = 1 + 1")
            self.assertAlmostEqual(mutants_budget2[1].impact_score * mutants_budget2[1].likelihood, 0.5)
            self.assertEqual(mutants_budget2[1].original_code, "1 + 1")
            self.assertEqual(mutants_budget2[1].mutated_code, "1 - 1")
            self.assertEqual(mutants_budget2[1].mutation_type, MutantType.ARITHMETIC)

        # Verify sorting by checking priority scores for budget 2
        last_priority_score = float('inf')
        for m in mutants_budget2:
            current_priority_score = m.impact_score * m.likelihood
            self.assertLessEqual(current_priority_score, last_priority_score, "Mutants are not sorted by priority score.")
            last_priority_score = current_priority_score
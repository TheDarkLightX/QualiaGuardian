# Placeholder for static analysis logic.
# This module will include functions to calculate metrics like
# Cyclomatic Complexity, Lines of Code, etc.
# Cognitive Complexity will be revisited.
from radon.complexity import cc_rank, cc_visit
# import ast # No longer needed for CognitiveComplexityVisitor here

# def calculate_cognitive_complexity(code_content):
#     """
#     TEMPORARILY REMOVED - Placeholder for calculating Cognitive Complexity.
#     Args:
#         code_content (str): The source code as a string.
#     Returns:
#         int: The calculated Cognitive Complexity score.
#     """
#     print(f"Cognitive Complexity calculation temporarily disabled.")
#     return 0

def calculate_cyclomatic_complexity(code_content):
    """
    Calculates average Cyclomatic Complexity using radon.
    Args:
        code_content (str): The source code as a string.
    Returns:
        float: The average Cyclomatic Complexity score, or 0.0 if error/no functions.
    """
    try:
        results = cc_visit(code_content)
        # print(f"DEBUG: cc_visit results for content '{code_content[:30]}...': {results}") 
        total_cc = 0
        count = 0
        for item in results:
            # print(f"DEBUG: item name: {getattr(item, 'name', 'N/A')}, complexity: {getattr(item, 'complexity', 'N/A')}, is_method: {getattr(item, 'is_method', 'N/A')}, classname: {getattr(item, 'classname', 'N/A')}")
            is_function_or_method = False
            if hasattr(item, 'is_method'): 
                if item.is_method: 
                    is_function_or_method = True
                elif item.classname is None: 
                    is_function_or_method = True
            
            if is_function_or_method and hasattr(item, 'complexity'):
                total_cc += item.complexity
                count += 1
        # print(f"DEBUG: total_cc: {total_cc}, count: {count}")
        return total_cc / count if count > 0 else 0.0
    except Exception as e:
        print(f"Error calculating cyclomatic complexity: {e}")
        return 0.0

def find_long_elements(code_content, max_lines):
    """
    Finds functions or methods longer than max_lines.
    Args:
        code_content (str): The source code as a string.
        max_lines (int): The maximum allowed lines for a function/method.
    Returns:
        list: A list of dictionaries, each representing a long element.
    """
    long_elements_found = []
    try:
        blocks = cc_visit(code_content) 
        for block in blocks:
            block_type_str = None
            if hasattr(block, 'is_method'): 
                if block.is_method:
                    block_type_str = "method"
                elif block.classname is None: 
                    block_type_str = "function"

            if block_type_str and hasattr(block, 'lineno') and hasattr(block, 'endline') and hasattr(block, 'name'):
                block_line_count = block.endline - block.lineno + 1
                if block_line_count > max_lines:
                    long_elements_found.append({
                        "name": block.name,
                        "type": block_type_str,
                        "lines": block_line_count,
                        "lineno": block.lineno,
                        "endline": block.endline
                    })
    except Exception as e:
        print(f"Error finding long elements: {e}")
    return long_elements_found

def find_large_classes(code_content, max_methods):
    """
    Finds classes with more than max_methods.
    Args:
        code_content (str): The source code as a string.
        max_methods (int): The maximum allowed methods for a class.
    Returns:
        list: A list of dictionaries, each representing a large class.
    """
    large_classes_found = []
    try:
        blocks = cc_visit(code_content)
        
        class_method_counts = {} 
        class_info = {} 

        # First pass: identify all classes and initialize their info
        # print(f"DEBUG find_large_classes: Starting 1st pass. blocks: {blocks}") # DEBUG REMOVED
        from radon.visitors import Class as RadonClass # Import for isinstance check
        for block in blocks:
            # print(f"DEBUG find_large_classes: 1st pass - block name: {getattr(block, 'name', 'N/A')}, type: {type(block)}, classname: {getattr(block, 'classname', 'N/A')}, is_method: {getattr(block, 'is_method', 'N/A')}") # DEBUG REMOVED
            if isinstance(block, RadonClass): # Check if it's a Class object
                if block.name not in class_method_counts:
                    class_method_counts[block.name] = 0
                    if hasattr(block, 'lineno') and hasattr(block, 'endline'):
                        class_info[block.name] = {"lineno": block.lineno, "endline": block.endline, "name": block.name}
                    else:
                        class_info[block.name] = {"lineno": 0, "endline": 0, "name": block.name}
        
        # print(f"DEBUG find_large_classes: After 1st pass, class_info: {class_info}") # DEBUG REMOVED
        # print(f"DEBUG find_large_classes: After 1st pass, class_method_counts: {class_method_counts}") # DEBUG REMOVED

        # Second pass: count methods for each identified class
        for block in blocks:
            if hasattr(block, 'is_method') and block.is_method:
                if block.classname and block.classname in class_method_counts:
                    class_method_counts[block.classname] += 1

        # print(f"DEBUG find_large_classes: After 2nd pass, class_method_counts: {class_method_counts}") # DEBUG REMOVED

        for class_name_key in class_method_counts:
            method_count = class_method_counts[class_name_key]
            if method_count > max_methods:
                if class_name_key in class_info:
                    info = class_info[class_name_key]
                    large_classes_found.append({
                        "name": info["name"],
                        "method_count": method_count,
                        "lineno": info["lineno"],
                        "endline": info["endline"]
                    })
        
        # print(f"DEBUG find_large_classes: large_classes_found: {large_classes_found}") # DEBUG REMOVED
                
    except Exception as e:
        print(f"Error finding large classes: {e}")
    return large_classes_found

def analyze_imports(code_content, file_path_str="<unknown>"):
    """
    Analyzes import statements in the given Python code content using AST.
    Args:
        code_content (str): The source code as a string.
        file_path_str (str): The path of the file being analyzed (for context in errors).
    Returns:
        list: A list of dictionaries, each representing an import.
              e.g., {"module": "os", "names": [], "lineno": 1, "type": "Import"}
              e.g., {"module": "sys", "names": ["argv", "exit"], "lineno": 2, "type": "ImportFrom"}
    """
    import_details = []
    import ast
    try:
        tree = ast.parse(code_content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_details.append({
                        "module": alias.name,
                        "asname": alias.asname, # Could be None
                        "names": [], # No specific names for ast.Import
                        "lineno": node.lineno,
                        "type": "Import"
                    })
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module if node.module else "." # Handle relative imports like 'from . import foo'
                if node.level > 0: # Relative import like 'from ..foo import bar'
                    module_name = "." * node.level + (module_name if module_name != "." else "")

                imported_names = []
                for alias in node.names:
                    imported_names.append({
                        "name": alias.name,
                        "asname": alias.asname # Could be None
                    })
                import_details.append({
                    "module": module_name,
                    "names": imported_names,
                    "lineno": node.lineno,
                    "type": "ImportFrom",
                    "level": node.level # For relative imports
                })
    except SyntaxError as e:
        print(f"SyntaxError analyzing imports in {file_path_str}: {e}")
    except Exception as e:
        print(f"Error analyzing imports in {file_path_str}: {e}")
    return import_details

def get_module_name_from_path(base_path, file_path):
    """Converts a file path to a Python module name relative to base_path."""
    import os
    rel_path = os.path.relpath(file_path, base_path)
    module_path, _ = os.path.splitext(rel_path)
    return module_path.replace(os.sep, '.')

def build_import_graph(project_path):
    """
    Builds an import graph for modules within the project.
    Args:
        project_path (str): The root path of the project.
    Returns:
        dict: A dictionary where keys are module names (e.g., 'package.module')
              and values are sets of module names they import.
    """
    import os
    import_graph = {} # module_name -> set_of_imported_module_names

    project_files = []
    for root, _, files in os.walk(project_path):
        if ".venv" in root.split(os.sep) or ".git" in root.split(os.sep):
            continue
        for file in files:
            if file.endswith(".py"):
                project_files.append(os.path.join(root, file))

    # Create a mapping of all potential internal modules
    internal_modules = set()
    for py_file_path in project_files:
        module_name = get_module_name_from_path(project_path, py_file_path)
        internal_modules.add(module_name)
        # Ensure __init__.py files also register their package name
        if os.path.basename(py_file_path) == "__init__.py":
            package_name = get_module_name_from_path(project_path, os.path.dirname(py_file_path))
            if package_name: # Avoid adding '.' for project root __init__.py if project_path is '.'
                 internal_modules.add(package_name)


    for py_file_path in project_files:
        current_module_name = get_module_name_from_path(project_path, py_file_path)
        import_graph.setdefault(current_module_name, set())

        try:
            with open(py_file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            imports_data = analyze_imports(content, file_path_str=py_file_path)
            
            for imp_info in imports_data:
                targets_to_add = set()
                if imp_info["type"] == "Import": # e.g. import foo.bar or import foo.bar as fb
                    targets_to_add.add(imp_info["module"])
                elif imp_info["type"] == "ImportFrom":
                    base_imported_module = imp_info["module"] # e.g., 'collections', '.sibling', '..package.sibling'
                    
                    if imp_info["level"] > 0: # Relative import
                        current_pkg_path_parts = current_module_name.split('.')
                        # If current file is a module (not __init__.py), its package is one level up in terms of path parts
                        if os.path.basename(py_file_path) != "__init__.py":
                            current_pkg_path_parts = current_pkg_path_parts[:-1]

                        # Resolve the base part of the relative import
                        # e.g. for 'from ..foo import bar' in 'pkg.sub.mod', level=2, module='foo'
                        # current_pkg_path_parts = ['pkg']
                        # resolved_base_parts = path_parts_up_to_level(current_pkg_path_parts, imp_info["level"]-1) -> e.g. ['pkg'] if level is 2 from sub.mod
                        
                        num_dots = imp_info["level"]
                        resolved_base_parts = []
                        if num_dots <= len(current_pkg_path_parts):
                            resolved_base_parts = current_pkg_path_parts[:len(current_pkg_path_parts) - (num_dots -1)]
                        
                        if not base_imported_module or base_imported_module == '.':
                            # Handles 'from . import X' or 'from .. import X'
                            # X (or Xs) are in imp_info["names"]
                            for name_data in imp_info["names"]:
                                if resolved_base_parts: # if not an attempt to go above project root
                                    targets_to_add.add(".".join(resolved_base_parts) + "." + name_data["name"])
                                else: # from . import X at top level (e.g. project_path is '.')
                                     if num_dots == 1: # from . import X
                                         targets_to_add.add(name_data["name"])
                                     # else: from .. import X at top level is an error, skip
                            
                        elif base_imported_module.startswith('.'): # Should have been caught by level > 0, but defensive
                             # This case is complex, e.g. from ..foo import bar
                             # The previous logic for this was:
                             # path_parts = current_module_name.split('.')
                             # effective_level = num_dots
                             # if os.path.basename(py_file_path) == "__init__.py": effective_level = num_dots -1
                             # if effective_level < len(path_parts): base_parts = path_parts[:len(path_parts)-effective_level] ...
                             # This needs to be robustly reimplemented if we encounter complex relative imports.
                             # For now, the above handles "from . import X" and "from .sibling import X" (if module is '.sibling')
                             # Let's assume the simple case for now: from .mod or from ..mod
                            if resolved_base_parts:
                                targets_to_add.add(".".join(resolved_base_parts + [base_imported_module.lstrip('.')]))


                        else: # from .sibling import X (module is 'sibling', level is 1)
                            if resolved_base_parts:
                                targets_to_add.add(".".join(resolved_base_parts) + "." + base_imported_module)
                            else: # from .sibling at top level
                                if num_dots == 1:
                                    targets_to_add.add(base_imported_module)


                    else: # Absolute import from a package
                        targets_to_add.add(base_imported_module)

                for target_module_str in targets_to_add:
                    if target_module_str and target_module_str in internal_modules:
                        import_graph[current_module_name].add(target_module_str)
                    # else:
                        # print(f"DEBUG: Skipping external/unresolved import: {target_module_str} from {current_module_name}")
        except Exception as e:
            print(f"Error building import graph for file {py_file_path}: {e}")
            
    # print(f"DEBUG: Final import_graph: {import_graph}")
    return import_graph

def find_circular_dependencies(import_graph):
    """
    Finds circular dependencies in the import graph.
    Args:
        import_graph (dict): A dictionary representing the import graph.
    Returns:
        list: A list of cycles (each cycle is a list of module names).
    """
    # Algorithm to find all elementary cycles in a directed graph
    # Uses a modified DFS approach
    
    cycles = []
    
    # Adjacency list representation from import_graph
    adj = import_graph
    
    # Path: current path in DFS
    # Visited: nodes currently in the recursion stack (gray set)
    # FullyExplored: nodes for which all paths have been explored (black set)
    
    path = []
    visited_in_path = set() # Nodes currently in the recursion stack for this DFS path
    
    # We need to iterate over all nodes to find all cycles, as graph might be disconnected
    all_nodes_in_graph = set(adj.keys())
    for neighbors in adj.values():
        all_nodes_in_graph.update(neighbors)

    # To avoid redundant cycle reporting if multiple DFS start points hit same cycle:
    # Store cycles as frozensets of tuples to make them hashable and comparable irrespective of start node
    found_cycles_set = set()

    def find_cycles_recursive(u):
        path.append(u)
        visited_in_path.add(u)

        if u in adj: # Check if u has outgoing edges
            for v in sorted(list(adj[u])): # Sort for deterministic cycle reporting
                if v in visited_in_path: # Cycle detected
                    try:
                        cycle_start_index = path.index(v)
                        current_cycle = tuple(path[cycle_start_index:])
                        # Store sorted tuple to uniquely identify cycle regardless of starting point in the cycle
                        sorted_cycle_tuple = tuple(sorted(current_cycle))
                        if sorted_cycle_tuple not in found_cycles_set:
                            cycles.append(list(current_cycle)) # Report the actual path of the cycle
                            found_cycles_set.add(sorted_cycle_tuple)
                    except ValueError: # Should not happen if v is in visited_in_path
                        pass
                else:
                    # Only recurse if v is a node that can lead to further paths
                    # (i.e., it's in the graph or could be an endpoint of another cycle)
                    # This check might be redundant if all_nodes_in_graph covers all relevant nodes.
                    # For now, assume adj[u] only contains valid graph nodes.
                    find_cycles_recursive(v)
        
        path.pop()
        visited_in_path.remove(u)

    # Start DFS from each node to find all cycles
    # We only need to start DFS from nodes that are part of the graph keys,
    # as cycles must involve nodes that import others.
    for node in sorted(list(adj.keys())): # Sort for deterministic behavior
        find_cycles_recursive(node)
        # After a full DFS from a node, clear path and visited_in_path for the next starting node
        # This is handled by the pop() in find_cycles_recursive for backtracking.
        # The main loop ensures we try all potential starting points for cycles.
        # However, the above DFS will find all cycles reachable from `node`.
        # To find *all* elementary cycles, a more complex algorithm like Tarjan's or Johnson's is typically used.
        # The current DFS approach might report non-elementary cycles or miss some if not careful.
        # For now, this is a simpler approach that should find *some* cycles.
        # Let's refine: the recursive DFS needs to be called for each unvisited node globally.

    # A more standard DFS for cycle detection:
    # 0: white (unvisited), 1: gray (visiting), 2: black (visited)
    color = {node: 0 for node in all_nodes_in_graph}
    parent = {node: None for node in all_nodes_in_graph}
    
    # Clear previous cycle finding attempts
    cycles.clear()
    found_cycles_set.clear()

    def dfs_visit(u_node):
        color[u_node] = 1 # Mark as gray
        
        # Ensure u_node is in adj before trying to iterate its neighbors
        if u_node in adj:
            for v_node in sorted(list(adj[u_node])): # Sort for deterministic cycle reporting
                if v_node not in color: # If v_node is not in color map, it's an external/unknown module
                    continue
                if color[v_node] == 1: # Gray to Gray edge -> cycle found
                    # Backtrack from u_node to v_node to find the cycle path
                    # print(f"Cycle detected: {v_node} -> {u_node}")
                    curr = u_node
                    cycle_path = [curr]
                    while curr != v_node and parent[curr] is not None : # Check parent[curr] is not None
                        curr = parent[curr]
                        cycle_path.append(curr)
                        if len(cycle_path) > len(all_nodes_in_graph): # Safety break for malformed parent chain
                            # print("Error: Cycle path reconstruction seems to be in a loop")
                            break
                    cycle_path.reverse()
                    
                    # Ensure v_node is actually in the reconstructed path if loop didn't break early
                    if v_node in cycle_path:
                        # Normalize cycle representation (e.g., start with smallest node, sorted tuple)
                        # For now, just add the path as found.
                        # To avoid duplicates of the same cycle starting at different nodes:
                        sorted_cycle_tuple = tuple(sorted(cycle_path))
                        if sorted_cycle_tuple not in found_cycles_set:
                            cycles.append(list(cycle_path))
                            found_cycles_set.add(sorted_cycle_tuple)

                elif color[v_node] == 0: # White node
                    parent[v_node] = u_node
                    dfs_visit(v_node)
        color[u_node] = 2 # Mark as black

    for node_key in sorted(list(all_nodes_in_graph)): # Iterate over all nodes
        if color.get(node_key, 0) == 0: # If node is white (unvisited)
            dfs_visit(node_key)
            
    return cycles

def find_unused_imports(code_content, file_path_str="<unknown>"):
    """
    Finds unused import statements in the given Python code content.
    Args:
        code_content (str): The source code as a string.
        file_path_str (str): The path of the file being analyzed (for context).
    Returns:
        list: A list of dictionaries, each representing an unused import.
              e.g., {"module": "os", "name": "path", "alias": "op", "lineno": 1}
    """
    import ast
    unused_imports_found = []
    
    declared_imports_info = analyze_imports(code_content, file_path_str)
    if not declared_imports_info:
        return []

    imported_names_map = {}
    for imp_info in declared_imports_info:
        if imp_info["type"] == "Import":
            name_to_check = imp_info["asname"] if imp_info["asname"] else imp_info["module"]
            imported_names_map[name_to_check] = {
                "module": imp_info["module"],
                "original_name": imp_info["module"],
                "lineno": imp_info["lineno"]
            }
        elif imp_info["type"] == "ImportFrom":
            for name_data in imp_info["names"]:
                name_to_check = name_data["asname"] if name_data["asname"] else name_data["name"]
                imported_names_map[name_to_check] = {
                    "module": imp_info["module"],
                    "original_name": name_data["name"],
                    "lineno": imp_info["lineno"]
                }
    
    if not imported_names_map:
        return []

    used_names = set()
    try:
        tree = ast.parse(code_content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                 used_names.add(node.value.id)
    except SyntaxError as e:
        print(f"SyntaxError finding unused imports in {file_path_str}: {e}")
        return []
    except Exception as e:
        print(f"Error finding unused imports in {file_path_str}: {e}")
        return []

    for imported_name, info in imported_names_map.items():
        if imported_name not in used_names:
            if info["original_name"] == "*":
                continue
            unused_imports_found.append({
                "module": info["module"],
                "name": info["original_name"],
                "alias": imported_name if imported_name != info["original_name"] else None,
                "lineno": info["lineno"]
            })
            
    return unused_imports_found

def analyze_file(file_path):
    """
    Analyze a Python file and return comprehensive metrics

    Args:
        file_path (str): Path to the Python file to analyze

    Returns:
        dict: Analysis results including functions, classes, and metrics
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()

        # Basic file metrics
        lines = code_content.split('\n')
        total_lines = len(lines)

        # Get complexity blocks
        blocks = cc_visit(code_content)

        # Extract functions
        functions = []
        classes = []

        for block in blocks:
            if hasattr(block, 'is_method') and hasattr(block, 'name'):
                if block.is_method:
                    # This is a method
                    functions.append({
                        'name': block.name,
                        'type': 'method',
                        'complexity': getattr(block, 'complexity', 1),
                        'lines': getattr(block, 'endline', 0) - getattr(block, 'lineno', 0) + 1,
                        'parameters': len(getattr(block, 'args', [])) if hasattr(block, 'args') else 0,
                        'lineno': getattr(block, 'lineno', 0),
                        'endline': getattr(block, 'endline', 0)
                    })
                elif block.classname is None:
                    # This is a function
                    functions.append({
                        'name': block.name,
                        'type': 'function',
                        'complexity': getattr(block, 'complexity', 1),
                        'lines': getattr(block, 'endline', 0) - getattr(block, 'lineno', 0) + 1,
                        'parameters': len(getattr(block, 'args', [])) if hasattr(block, 'args') else 0,
                        'lineno': getattr(block, 'lineno', 0),
                        'endline': getattr(block, 'endline', 0)
                    })

        # Extract classes using radon
        from radon.visitors import Class as RadonClass
        for block in blocks:
            if isinstance(block, RadonClass):
                classes.append({
                    'name': block.name,
                    'methods': 0,  # Will be counted separately
                    'lineno': getattr(block, 'lineno', 0),
                    'endline': getattr(block, 'endline', 0)
                })

        # Count methods per class
        class_method_counts = {}
        for func in functions:
            if func['type'] == 'method':
                # Find which class this method belongs to
                for block in blocks:
                    if (hasattr(block, 'is_method') and block.is_method and
                        hasattr(block, 'name') and block.name == func['name'] and
                        hasattr(block, 'classname') and block.classname):
                        class_name = block.classname
                        class_method_counts[class_name] = class_method_counts.get(class_name, 0) + 1

        # Update class method counts
        for cls in classes:
            cls['methods'] = class_method_counts.get(cls['name'], 0)

        return {
            'file_path': file_path,
            'total_lines': total_lines,
            'functions': functions,
            'classes': classes,
            'avg_complexity': calculate_cyclomatic_complexity(code_content)
        }

    except FileNotFoundError:
        return None
    except Exception as e:
        return {'error': str(e)}


def detect_code_smells(analysis_result):
    """
    Detect code smells from analysis results

    Args:
        analysis_result (dict): Result from analyze_file()

    Returns:
        list: List of detected code smells
    """
    smells = []

    if not analysis_result or 'error' in analysis_result:
        return smells

    # Check for large classes (>10 methods)
    for cls in analysis_result.get('classes', []):
        if cls['methods'] > 10:
            smells.append({
                'type': 'large_class',
                'severity': 'medium',
                'description': f"Class '{cls['name']}' has {cls['methods']} methods (>10)",
                'line': cls['lineno'],
                'suggestion': 'Consider breaking this class into smaller, more focused classes'
            })

    # Check for long functions (>20 lines)
    for func in analysis_result.get('functions', []):
        if func['lines'] > 20:
            smells.append({
                'type': 'long_function',
                'severity': 'medium',
                'description': f"Function '{func['name']}' has {func['lines']} lines (>20)",
                'line': func['lineno'],
                'suggestion': 'Consider breaking this function into smaller functions'
            })

        # Check for complex functions (complexity >10)
        if func['complexity'] > 10:
            smells.append({
                'type': 'complex_function',
                'severity': 'high',
                'description': f"Function '{func['name']}' has complexity {func['complexity']} (>10)",
                'line': func['lineno'],
                'suggestion': 'Consider simplifying this function or breaking it into smaller functions'
            })

        # Check for functions with too many parameters (>5)
        if func['parameters'] > 5:
            smells.append({
                'type': 'long_parameter_list',
                'severity': 'medium',
                'description': f"Function '{func['name']}' has {func['parameters']} parameters (>5)",
                'line': func['lineno'],
                'suggestion': 'Consider using a configuration object or reducing parameters'
            })

    return smells


def get_function_metrics(file_path):
    """
    Get detailed metrics for all functions in a file

    Args:
        file_path (str): Path to the Python file

    Returns:
        list: List of function metrics
    """
    analysis = analyze_file(file_path)
    if not analysis or 'error' in analysis:
        return []

    return analysis.get('functions', [])


def count_lines_of_code(code_content):
    """
    Counts non-empty, non-comment lines of code.
    This is a simplified version.
    Args:
        code_content (str): The source code as a string.
    Returns:
        int: The number of lines of code.
    """
    lines = code_content.splitlines()
    loc_count = 0
    for line in lines:
        stripped_line = line.strip()
        if stripped_line and not stripped_line.startswith("#"): 
            loc_count += 1
    return loc_count

if __name__ == '__main__':
    # Test find_long_elements and find_large_classes
    test_code_for_smells = """
def short_function(): # 3 lines
    pass

def very_long_function(a, b, c, d, e, f, g, h, i, j): # 20 lines from def to return
    print(a); print(b); print(c); print(d); print(e); # 5
    print(f); print(g); print(h); print(i); print(j); # 10
    print(a); print(b); print(c); print(d); print(e); # 15
    print(f); print(g); print(h); print(i); # 19
    return j # 20

class NormalClass: # 2 methods
    def method1(self): pass
    def method2(self): pass

class LargeClassWithManyMethods: # 11 methods
    def m1(self): pass
    def m2(self): pass
    def m3(self): pass
    def m4(self): pass
    def m5(self): pass
    def m6(self): pass
    def m7(self): pass
    def m8(self): pass
    def m9(self): pass
    def m10(self): pass
    def m11(self): pass # Exceeds 10 methods

class ClassWithLongMethod:
    def short_method(self): #3 lines
        pass

    def very_long_method(self, x): # 20 lines from def to return
        y = x * 1; y = x * 2; y = x * 3; y = x * 4; y = x * 5;
        y = x * 6; y = x * 7; y = x * 8; y = x * 9; y = x * 10;
        y = x * 11; y = x * 12; y = x * 13; y = x * 14; y = x * 15;
        y = x * 16; y = x * 17; y = x * 18; y = x * 19;
        return y # L20
"""
    print("\n--- Testing find_long_elements (threshold 19) ---")
    long_elements = find_long_elements(test_code_for_smells, 19)
    for el in long_elements:
        print(f"  Found Long: {el['type']} {el['name']} - Lines: {el['lines']} (Lines {el['lineno']}-{el['endline']})")

    print("\n--- Testing find_large_classes (threshold 10) ---")
    large_classes = find_large_classes(test_code_for_smells, 10)
    for lc in large_classes:
        print(f"  Found Large Class: {lc['name']} - Methods: {lc['method_count']} (Lines {lc['lineno']}-{lc['endline']})")

    print("\n--- Testing find_large_classes (threshold 2) ---")
    large_classes_low_thresh = find_large_classes(test_code_for_smells, 2)
    for lc in large_classes_low_thresh:
        print(f"  Found Large Class: {lc['name']} - Methods: {lc['method_count']} (Lines {lc['lineno']}-{lc['endline']})")

    print("\n--- Testing analyze_imports ---")
    import_test_code = """
import os
import sys as system
from collections import defaultdict, Counter as MyCounter
from . import local_module
from ..parent_module import specific_thing
import math.submodule # This is not standard, but ast parses 'math.submodule' as module name
"""
    imports = analyze_imports(import_test_code, "import_test.py")
    print(f"Found {len(imports)} import statements:")
    for imp in imports:
        print(f"  - Line {imp['lineno']}: Type: {imp['type']}, Module: {imp['module']}")
        if imp["type"] == "Import":
            if imp["asname"]:
                print(f"    As: {imp['asname']}")
        elif imp["type"] == "ImportFrom":
            print(f"    Level: {imp['level']}")
            for name_info in imp["names"]:
                if name_info["asname"]:
                    print(f"    Name: {name_info['name']} as {name_info['asname']}")
                else:
                    print(f"    Name: {name_info['name']}")

    print("\n--- Testing find_unused_imports ---")
    unused_import_test_code_1 = """
import os # unused
import sys # used
from collections import Counter # unused
from datetime import date as d # used
import math as m # unused
import json as my_json # used

print(sys.argv)
today = d.today()
data = my_json.dumps({"key": "value"})
# print(m.pi) # This would make 'm' (math) used
"""
    unused1 = find_unused_imports(unused_import_test_code_1, "unused_test1.py")
    print(f"Unused imports in test 1 ({len(unused1)}):")
    for u_imp in unused1:
        print(f"  - Line {u_imp['lineno']}: Module: {u_imp['module']}, Name: {u_imp['name']}, Alias: {u_imp['alias']}")

    unused_import_test_code_2 = """
import re
from os import path, environ # path used, environ unused
import shutil # used

print(path.join('.', 'file'))
shutil.copyfile('a', 'b')
"""
    unused2 = find_unused_imports(unused_import_test_code_2, "unused_test2.py")
    print(f"Unused imports in test 2 ({len(unused2)}):")
    for u_imp in unused2:
        print(f"  - Line {u_imp['lineno']}: Module: {u_imp['module']}, Name: {u_imp['name']}, Alias: {u_imp['alias']}")

    unused_import_test_code_3 = "import this_is_used\nprint(this_is_used.VERSION)"
    unused3 = find_unused_imports(unused_import_test_code_3, "unused_test3.py")
    print(f"Unused imports in test 3 ({len(unused3)}):") # Expect 0
    for u_imp in unused3:
        print(f"  - Line {u_imp['lineno']}: Module: {u_imp['module']}, Name: {u_imp['name']}, Alias: {u_imp['alias']}")
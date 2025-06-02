"""
Sensor for calculating Algebraic Connectivity (lambda_2) of a project's
dependency graph, as a measure of modularity and coupling.
"""
import ast
import logging
import networkx as nx
import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import eigh # For dense matrices, if graph is small
# from scipy.sparse.linalg import eigsh # For sparse matrices, preferred for larger graphs
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple

logger = logging.getLogger(__name__)

class AlgebraicConnectivitySensor:
    """
    Calculates the algebraic connectivity (lambda_2, the second smallest eigenvalue
    of the graph Laplacian) for a project's module dependency graph.
    A higher lambda_2 generally suggests a "less splittable" or more interconnected graph.
    For modularity, a lower lambda_2 among loosely coupled components might be preferred,
    but this raw metric needs careful interpretation in context.
    """

    def __init__(self):
        pass

    def _resolve_relative_import(self, importing_module_name: str, level: int, relative_module_name: Optional[str]) -> Optional[str]:
        """
        Resolves a relative import to an absolute module name.
        Example: importing_module_name='pkg.sub.mod', level=1, relative_module_name='sibling' -> 'pkg.sub.sibling'
                 importing_module_name='pkg.sub.mod', level=2, relative_module_name='parent_mod' -> 'pkg.parent_mod'
                 importing_module_name='pkg.sub.mod', level=1, relative_module_name=None -> 'pkg.sub' (imports from __init__.py of current)
        """
        if level == 0: # Should not happen for relative imports
            return relative_module_name

        parts = importing_module_name.split('.')
        # For `from . import X`, level is 1. We go up `level-1` parts from the *package* containing the module.
        # If importing_module_name is 'pkg.sub.mod', its package is 'pkg.sub'.
        # `from .sibling` (level 1) means 'pkg.sub.sibling'.
        # `from ..common` (level 2) means 'pkg.common'.
        
        # Base for relative import is the package containing the current module.
        # If importing_module_name is 'a.b.c', base parts are ['a', 'b']
        # If level is 1, we want to go up 0 levels from this base.
        # If level is 2, we want to go up 1 level from this base.
        base_module_parts = parts[:-1] # Package containing the current module

        if level > len(base_module_parts) +1 : # Cannot go further up than project root
             # e.g. from ..foo in a top-level module.
             # This case might indicate an invalid import structure or one that's hard to resolve statically here.
            logger.debug(f"Relative import level {level} too high for module '{importing_module_name}'.")
            return None


        # Effective levels to go up from the *importer's package*
        # `from .mod` -> level 1. Base is current package. `base_module_parts` are correct.
        # `from ..mod` -> level 2. Base is parent package. `base_module_parts[:-1]`
        # `from ...mod` -> level 3. Base is grandparent package. `base_module_parts[:-2]`
        
        # Number of levels to ascend from the *importer's own package path*
        # if importing_module_name = 'pkgA.pkgB.modC'
        # from . D (level 1) -> pkgA.pkgB.D
        # from .. E (level 2) -> pkgA.E
        # from ... F (level 3) -> F (if pkgA is top level)

        dot_count = level
        
        # Start with the parts of the importing module itself
        current_module_path_parts = importing_module_name.split('.')
        
        # For relative imports, we first navigate up from the *current module's directory*.
        # Each dot in `from .mod` or `from ..mod` means going up one level.
        # `from .mod` (level 1) means `mod` is a sibling in the same package.
        # `from ..mod` (level 2) means `mod` is in the parent package.
        
        # Path parts of the importing module
        importer_parts = importing_module_name.split('.')
        
        # The "anchor" for relative import is the package of the importing module.
        # For `from .m import X` in `a.b.c`, anchor is `a.b`. `level` is 1.
        # For `from ..m import X` in `a.b.c`, anchor is `a`. `level` is 2.
        
        if level > len(importer_parts): # Trying to go above top-level package
            return None # Invalid relative import

        # Base path for the import after going up `level` directories from the module's location.
        # If importing_module_name is 'pkg.sub.mod', and level is 1 (from .), base is 'pkg.sub'
        # If importing_module_name is 'pkg.sub.mod', and level is 2 (from ..), base is 'pkg'
        base_parts = importer_parts[:-(level)] # Go up 'level' directories

        if relative_module_name:
            return ".".join(base_parts + [relative_module_name])
        else: # from . import X (relative_module_name is None)
            # This typically means importing from __init__.py of the current package,
            # or a sibling module if `X` is specified in `names`.
            # If `relative_module_name` is None, it means `from . import name1, name2`.
            # The `name1`, `name2` are siblings.
            # The base_parts already represent the package.
            # This function is about resolving the *module path* part of `from module_path import ...`
            # So if relative_module_name is None, it means the import refers to names within the package
            # identified by base_parts. For graph construction, we care about module dependencies.
            # `from . import foo` means `current_package.foo`.
            # `from .foo import bar` means `current_package.foo`.
            # This function is called with `node.module` which is `foo` in the second case.
            # If `node.module` is None (first case), it means we are importing names from the current package's __init__.py
            # or sibling modules. This function should return the module being imported.
            # This case (relative_module_name is None) is tricky for this helper.
            # The caller `_extract_imports` handles `node.names` for `from . import name1`.
            # This helper is for `from .module_part import ...`.
            # If relative_module_name is None here, it means `from . import ...` where the module is implied.
            # This typically means importing from the current package's __init__.py.
            # For graph purposes, if `from . import foo` is used, `foo` is the module.
            # This helper is for the `node.module` part.
            # If `node.module` is None and `level > 0`, it means `from . import name`.
            # The `name` itself is the sibling module.
            # This helper is for `from .sibling_module import X`. Here `relative_module_name` would be `sibling_module`.
            # So, if `relative_module_name` is None, it's likely an import from current package's `__init__.py`.
            # We can return ".".join(base_parts) if base_parts else None.
            # However, _extract_imports passes node.module as relative_module_name.
            # If node.module is None (e.g. from . import foo), _extract_imports iterates node.names.
            # So this helper should always receive a non-None relative_module_name if it's from `node.module`.
            # The case `from . import foo` is handled by `_extract_imports` taking `foo` from `node.names`.
            # This function is for `from .foo import Bar`. `relative_module_name` will be `foo`.
            return ".".join(base_parts) if base_parts else None # Should not happen if relative_module_name is always passed

    def _extract_imports(self, file_path: Path, current_module_fqn: str) -> Set[str]:
        """
        Extracts imported module names from a Python file, attempting to resolve them.
        Args:
            file_path: Path to the Python source file.
            current_module_fqn: Fully qualified name of the module being parsed.
        """
        imports: Set[str] = set()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content, filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import): # e.g. import foo, import foo.bar
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom): # e.g. from foo import bar, from . import bar
                    if node.level == 0:  # Absolute import: from package import item or from package.module import item
                        if node.module: # node.module is "package" or "package.module"
                            imports.add(node.module) # Add the base module/package
                            # Also consider items in node.names if they form part of a FQN with node.module
                            # e.g. from package import submodule -> add "package.submodule"
                            # This helps if "submodule" itself is a module file.
                            for alias in node.names:
                                # Check if alias.name could be a submodule of node.module
                                # This is heuristic; graph building will confirm if "node.module.alias.name" is a known module.
                                potential_submodule_fqn = f"{node.module}.{alias.name}"
                                imports.add(potential_submodule_fqn) # Add this potential FQN
                        # else: node.module is None, which is invalid for absolute from-import
                    else:  # Relative import: node.level > 0
                        if node.module:
                            # e.g., from .sibling_module import X
                            # node.module is "sibling_module"
                            resolved_import = self._resolve_relative_import(current_module_fqn, node.level, node.module)
                            if resolved_import:
                                imports.add(resolved_import)
                        else:
                            # e.g., from . import X, from .. import Y
                            # node.module is None, X/Y are in node.names
                            for alias in node.names:
                                resolved_import = self._resolve_relative_import(current_module_fqn, node.level, alias.name)
                                if resolved_import:
                                    imports.add(resolved_import)
        except Exception as e:
            logger.warning(f"AlgebraicConnectivitySensor: Failed to parse {file_path} (module: {current_module_fqn}): {e}")
        return imports

    def build_dependency_graph(self, project_root: Path, project_modules: Optional[List[Path]] = None) -> nx.Graph:
        """
        Builds an undirected dependency graph for the project.
        Nodes are modules, edges represent an import relationship.

        Args:
            project_root: The root path of the project.
            project_modules: A list of specific module paths to consider. 
                             If None, scans for *.py files in project_root.
        
        Returns:
            An undirected NetworkX graph.
        """
        graph = nx.Graph()
        module_map: Dict[str, Path] = {} # Maps module name to its file path

        if project_modules:
            py_files = [p for p in project_modules if p.is_file() and p.suffix == '.py']
        else:
            py_files = list(project_root.rglob("*.py"))
            # Exclude files in common virtual environment directories
            py_files = [
                p for p in py_files 
                if not any(part in ['.venv', 'venv', 'env', 'ENV', '.tox', '.nox', '__pycache__'] for part in p.parts)
            ]


        for py_file in py_files:
            try:
                # Determine module name from file path relative to project_root
                relative_path = py_file.relative_to(project_root)
                module_name_parts = list(relative_path.parts)
                if module_name_parts[-1] == "__init__.py":
                    module_name_parts.pop() # Directory is the module
                elif module_name_parts[-1].endswith(".py"):
                    module_name_parts[-1] = module_name_parts[-1][:-3] # Remove .py
                
                if not module_name_parts:
                    # This case implies __init__.py at project_root, which is unusual for a module name.
                    # Or a file directly at project_root.
                    # If project_root is 'myproj' and file is 'myproj/mod.py', parts=['mod']
                    # If project_root is 'myproj' and file is 'myproj/__init__.py', parts=[]
                    # Let's use the file stem if parts is empty and it's not an __init__.py
                    if py_file.stem != "__init__":
                         module_name = py_file.stem
                    else: # __init__.py at root, module name is effectively the directory name
                         module_name = project_root.name
                else:
                    module_name = ".".join(module_name_parts)
                
                if not module_name:
                    logger.debug(f"Could not determine module name for {py_file} relative to {project_root}, skipping.")
                    continue

                graph.add_node(module_name)
                module_map[module_name] = py_file
            except ValueError:
                 logger.warning(f"File {py_file} could not be made relative to project_root {project_root}. Skipping.")
                 continue
            except Exception as e:
                 logger.error(f"Error processing file {py_file} for module name: {e}")
                 continue
        
        # Second pass to add edges
        for importer_fqn, py_file in module_map.items():
            # Pass current module's FQN to _extract_imports for relative import resolution
            extracted_import_names = self._extract_imports(py_file, importer_fqn)
            for imported_name in extracted_import_names:
                # `imported_name` could be 'os' or 'my_project.utils.helpers' or 'sibling_module' (if resolved from relative)
                # We only add edges to modules that are part of our scanned project (i.e., in module_map keys)
                
                # Attempt to match the full imported_name first
                logger.debug(f"Attempting edge for importer '{importer_fqn}': considering import '{imported_name}'")
                if imported_name in module_map:
                    if importer_fqn != imported_name:
                        graph.add_edge(importer_fqn, imported_name)
                        logger.info(f"ADDED Edge: {importer_fqn} -> {imported_name}") # Changed to INFO for visibility
                    else:
                        logger.debug(f"Skipping self-import: {importer_fqn} -> {imported_name}")
                else:
                    logger.debug(f"Import '{imported_name}' not in module_map. Keys: {list(module_map.keys())}")
                    # If not a direct match, it might be an import of a top-level package
                    # that contains modules we've mapped. E.g. import my_package, and we have my_package.mod1
                    # This part is complex. For now, we rely on _extract_imports resolving to FQNs
                    # that should match keys in module_map if they are internal.
                    # If `imported_name` is like 'os', it won't be in module_map, correctly ignored.
                    # If `imported_name` was resolved by `_extract_imports` to `project.actual_module`
                    # and `project.actual_module` is a key in `module_map`, it will be added.
                    pass # External or unresolvable imports are ignored for graph edges

        logger.info(f"Built dependency graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        return graph

    def calculate_algebraic_connectivity(self, project_root_str: str) -> Optional[float]:
        """
        Calculates the algebraic connectivity (lambda_2) for the project.

        Args:
            project_root_str: String path to the project's root directory.

        Returns:
            The algebraic connectivity as a float, or None if an error occurs
            or the graph is too small/disconnected in a way that lambda_2 is not meaningful.
        """
        project_root = Path(project_root_str)
        if not project_root.is_dir():
            logger.error(f"AlgebraicConnectivitySensor: Project root {project_root_str} is not a valid directory.")
            return None

        graph = self.build_dependency_graph(project_root)

        logger.debug(f"Graph for {project_root_str}: Nodes: {list(graph.nodes())}, Edges: {list(graph.edges())}") # Reverted to DEBUG

        if graph.number_of_nodes() < 2:
            logger.info("AlgebraicConnectivitySensor: Graph has fewer than 2 nodes. Lambda_2 is undefined or 0.")
            return 0.0 # Or None, typically lambda_2 is 0 for a graph with 1 node or an empty graph.

        # Ensure the graph is treated as a single component for Laplacian calculation.
        # If the graph is disconnected, algebraic connectivity is 0.
        # We can calculate for the largest connected component if desired,
        # but standard lambda_2 for the whole graph will be 0 if disconnected.
        if not nx.is_connected(graph):
            logger.info(f"AlgebraicConnectivitySensor: Graph for {project_root_str} is not connected. Analyzing largest connected component.")
            connected_components = list(nx.connected_components(graph))
            if not connected_components: # Should not happen if graph has nodes
                logger.warning("AlgebraicConnectivitySensor: No connected components found in a non-empty graph. Returning 0.0")
                return 0.0
            
            largest_cc_nodes = max(connected_components, key=len)
            sub_graph = graph.subgraph(largest_cc_nodes)
            logger.info(f"Largest CC for {project_root_str}: Nodes: {list(sub_graph.nodes())}, Edges: {list(sub_graph.edges())}")
            
            if sub_graph.number_of_nodes() < 2:
                logger.info("AlgebraicConnectivitySensor: Largest connected component has fewer than 2 nodes. Lambda_2 is 0.")
                return 0.0
            # Replace the graph with its largest connected component for lambda_2 calculation
            graph_to_analyze = sub_graph
        else:
            logger.info(f"AlgebraicConnectivitySensor: Graph for {project_root_str} is connected.")
            graph_to_analyze = graph

        logger.debug(f"Graph to analyze for {project_root_str} (is_connected: {nx.is_connected(graph_to_analyze)}): Nodes: {list(graph_to_analyze.nodes())}, Edges: {list(graph_to_analyze.edges())}")

        try:
            # Using networkx's built-in method which handles sparse/dense appropriately
            # and sorts eigenvalues. It's generally robust.
            # laplacian = nx.laplacian_matrix(graph).asfptype()
            # eigenvalues = np.sort(eigh(laplacian.toarray())[0]) # eigh for symmetric, [0] gets eigenvalues

            # NetworkX has a direct method for algebraic connectivity
            # However, it might raise an exception for disconnected graphs if not handled.
            # Since we checked for nx.is_connected, this should be safer.
            # For very large graphs, computing all eigenvalues can be slow.
            # nx.algebraic_connectivity uses scipy.sparse.linalg.eigsh by default for connected graphs.
            
            lambda_2 = nx.algebraic_connectivity(graph_to_analyze, method='lanczos')
            # Valid methods include 'tracemin_lu', 'tracemin_mg', 'lanczos', 'lobpcg'.
            # 'lanczos' is often good for sparse graphs.

        except nx.NetworkXError as e: # Handles cases like disconnected graph if check missed, or other issues
            logger.error(f"AlgebraicConnectivitySensor: NetworkX error calculating algebraic connectivity: {e}")
            # Fallback: try to compute eigenvalues manually if direct method fails
            try:
                laplacian_matrix = nx.laplacian_matrix(graph_to_analyze).astype(float) # Ensure float for eigh
                if laplacian_matrix.shape[0] < 2: return 0.0 # Should be caught by earlier checks on graph_to_analyze
                
                # For sparse matrices, eigsh is better:
                # if graph.number_of_nodes() > 100: # Heuristic for "large"
                #    eigenvalues = eigsh(laplacian_matrix, k=min(2, graph.number_of_nodes()-1), which='SM', return_eigenvectors=False)
                # else: # For smaller graphs, dense eigh is fine
                eigenvalues = eigh(laplacian_matrix.toarray(), eigvals_only=True)

                eigenvalues = np.sort(eigenvalues)
                if len(eigenvalues) < 2:
                    lambda_2 = 0.0 # Graph is too small or disconnected
                else:
                    # lambda_2 is the second smallest eigenvalue. Smallest is always 0 for connected.
                    lambda_2 = eigenvalues[1] 
            except Exception as e_manual:
                logger.error(f"AlgebraicConnectivitySensor: Manual eigenvalue calculation failed after NetworkXError: {e_manual}")
                return None
        except Exception as e:
            logger.error(f"AlgebraicConnectivitySensor: Unexpected error calculating algebraic connectivity: {e}")
            return None
        
        # lambda_2 can be very close to zero due to floating point issues.
        if np.isclose(lambda_2, 0.0):
            lambda_2 = 0.0

        logger.info(f"Calculated Algebraic Connectivity (lambda_2): {lambda_2:.6f}")
        return float(lambda_2)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    sensor = AlgebraicConnectivitySensor()

    # Create a dummy project structure for testing
    temp_dir_obj = tempfile.TemporaryDirectory()
    project_root = Path(temp_dir_obj.name) / "dummy_project_arch"
    project_root.mkdir(parents=True, exist_ok=True)

    (project_root / "module_a.py").write_text("import module_b\nimport module_c")
    (project_root / "module_b.py").write_text("import module_c\nimport external_lib")
    (project_root / "module_c.py").write_text("import os")
    (project_root / "module_d.py").write_text("print('isolated')") # Isolated module
    
    # Subpackage
    sub_pkg_path = project_root / "subpackage"
    sub_pkg_path.mkdir(exist_ok=True)
    (sub_pkg_path / "__init__.py").write_text("# subpackage init")
    (sub_pkg_path / "module_e.py").write_text("from ..module_a import something\nimport os")


    lambda_2_val = sensor.calculate_algebraic_connectivity(str(project_root))
    if lambda_2_val is not None:
        logger.info(f"Lambda_2 for dummy project: {lambda_2_val:.6f}")
        # Expected: Graph of A, B, C, subpackage.module_e. D is isolated.
        # A -> B, C
        # B -> C
        # E -> A (relative import)
        # C, D are effectively sinks or isolated in terms of project deps.
        # The graph for lambda_2 calculation will likely be (A,B,C, subpackage.module_e)
        # If D is included and isolated, lambda_2 for whole graph is 0.
        # If only largest connected component is considered, it would be non-zero.
        # Current implementation returns 0 if graph is disconnected.

    # Test with a non-existent path
    sensor.calculate_algebraic_connectivity("./non_existent_project_arch")

    temp_dir_obj.cleanup()
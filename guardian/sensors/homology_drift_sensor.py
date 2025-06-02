"""
Sensor for calculating Persistent Homology Drift (Wasserstein distance)
between two versions of a codebase, indicating structural shifts.
"""
import logging
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance
import numpy as np
from typing import Optional, List, Any, Dict, Set, Tuple # Added Set and Tuple
# For graph representation if we use graphs as input to Gudhi
import networkx as nx
# from .algebraic_connectivity_sensor import AlgebraicConnectivitySensor # If we reuse graph building

logger = logging.getLogger(__name__)

class HomologyDriftSensor:
    """
    Calculates the Wasserstein distance between persistence diagrams of two
    representations of a codebase (e.g., dependency graphs from two versions).
    This distance quantifies structural "drift" between versions.
    """

    def __init__(self, max_dimension: int = 2, max_edge_length: float = 10.0):
        """
        Initializes the sensor.

        Args:
            max_dimension: Maximum homology dimension to compute (e.g., 2 means H0, H1).
                           Gudhi typically computes up to dimension `max_dimension - 1`.
                           So, for H0 and H1, this should be 2 or 3 depending on Gudhi's API for Rips.
                           For RipsComplex, `max_dimension` is the limit.
            max_edge_length: Maximum edge length for the Rips complex construction.
                             This is a critical parameter that affects the persistence diagram.
        """
        self.max_dimension = max_dimension # Gudhi's RipsComplex computes up to this dimension
        self.max_edge_length = max_edge_length
        # Check if Gudhi is available
        if gd.__version__ is None: # Basic check, proper import error handled by Python
            logger.error("Gudhi library does not seem to be installed correctly.")
            # Potentially raise an error or set a non-functional flag

    def _create_point_cloud_from_graph(self, graph: nx.Graph) -> Optional[np.ndarray]:
        """
        Creates a point cloud representation from a graph.
        Placeholder: A simple way is to use node degrees or layout positions.
        A more sophisticated approach might use graph embeddings.
        For now, let's use a simple degree-based representation if no layout is available.
        """
        if not graph or graph.number_of_nodes() == 0:
            return None
        
        # Option 1: Use spring layout (can be non-deterministic without seed)
        # pos = nx.spring_layout(graph, seed=42) 
        # points = np.array([list(pos[node]) for node in graph.nodes()])
        
        # Option 2: Use node degrees as a simple 1D feature, then expand.
        # This is a very naive point cloud for demonstration.
        # A better approach would involve more meaningful node features or embeddings.
        if graph.number_of_nodes() == 1:
            return np.array([[0.0]]) # Single point for a single node graph

        try:
            # Use a spectral layout for potentially more stable embeddings
            # It requires a connected graph. If not connected, use largest CC.
            if not nx.is_connected(graph):
                largest_cc_nodes = max(nx.connected_components(graph), key=len)
                sub_graph = graph.subgraph(largest_cc_nodes)
                if sub_graph.number_of_nodes() < 2: # If largest CC is too small
                     # Fallback to a simple representation for small/disconnected graphs
                    points = np.array([[i, graph.degree(node)] for i, node in enumerate(graph.nodes())])
                    return points
                graph_to_layout = sub_graph
            else:
                graph_to_layout = graph

            if graph_to_layout.number_of_nodes() >= 2 : # Spectral layout needs at least 2 nodes
                pos = nx.spectral_layout(graph_to_layout)
                # Ensure all original graph nodes get a position, even if 0,0 for disconnected ones
                # For now, this focuses on the largest component.
                points = np.array([pos.get(node, [0.0, 0.0]) for node in graph_to_layout.nodes()])

            else: # Fallback for very small graphs
                 points = np.array([[i, graph.degree(node)] for i, node in enumerate(graph.nodes())])


        except Exception as e:
            logger.warning(f"HomologyDriftSensor: Graph layout failed ({e}). Falling back to degree-based points.")
            # Fallback to simple degree based points if layout fails
            points = np.array([[i, graph.degree(node)] for i, node in enumerate(graph.nodes())])

        if points.ndim == 1: # Ensure it's 2D
            points = points.reshape(-1, 1)
        
        # Normalize points to prevent huge edge lengths in Rips complex if using layout
        if points.shape[0] > 0 and points.shape[1] > 0 : # Check if points is not empty
            points = (points - np.mean(points, axis=0)) / (np.std(points, axis=0) + 1e-6) # Add epsilon to avoid div by zero
        
        return points


    def calculate_persistence_diagram(self, data_representation: Any) -> List[Tuple[int, Tuple[float, float]]]:
        """
        Calculates the persistence diagram for a given data representation.
        Currently supports point clouds (numpy arrays).
        Future: Could support graphs directly if Gudhi offers such simplicial complexes easily.

        Args:
            data_representation: A numpy array representing a point cloud,
                                 or potentially other structures in the future.
        Returns:
            A list of persistence points (dimension, birth, death).
            Returns empty list if diagram cannot be computed.
        """
        if not isinstance(data_representation, np.ndarray) or data_representation.size == 0:
            logger.warning("HomologyDriftSensor: Invalid or empty data representation for persistence diagram.")
            return []

        try:
            # Using Rips complex from point cloud
            rips_complex = gd.RipsComplex(points=data_representation, max_edge_length=self.max_edge_length)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dimension)
            
            # Compute persistence
            diag = simplex_tree.persistence()
            
            # Gudhi's persistence() returns a list of tuples (dimension, (birth, death))
            # Filter out infinite death times for practical use if necessary, though Wasserstein distance handles them.
            # For Wasserstein, (d, (b, inf)) is fine.
            
            # Example: plot_persistence_barcode(diag) or plot_persistence_diagram(diag) for visualization
            logger.debug(f"Computed persistence diagram with {len(diag)} points.")
            return diag
        except Exception as e:
            logger.error(f"HomologyDriftSensor: Error computing persistence diagram: {e}")
            return []

    def calculate_drift(self, representation_v1: Any, representation_v2: Any, homology_dimension: int = 1) -> Optional[float]:
        """
        Calculates the Wasserstein distance between persistence diagrams of two
        codebase representations for a specific homology dimension.

        Args:
            representation_v1: Data representation of the first version (e.g., point cloud).
            representation_v2: Data representation of the second version.
            homology_dimension: The homology dimension (0 for H0, 1 for H1, etc.) to compare.

        Returns:
            The Wasserstein-1 distance, or None if an error occurs.
        """
        diag1 = self.calculate_persistence_diagram(representation_v1)
        diag2 = self.calculate_persistence_diagram(representation_v2)

        if not diag1 and not diag2: # Both empty, no drift or error already logged
            logger.debug(f"Both diagrams empty for H{homology_dimension}. Drift is 0.")
            return 0.0
        # Allow Gudhi's wasserstein_distance to handle cases where one diagram might be empty
        # after filtering by dimension. It should correctly calculate the sum of persistences
        # of the non-empty diagram's points to the diagonal.
        # if not diag1 or not diag2:
        #     logger.warning("HomologyDriftSensor: One of the persistence diagrams (pre-filter) is empty.")
            # No longer returning None here.

        # Filter diagrams for the specified homology dimension
        # Gudhi persistence points are (dim, (birth, death))
        diag1_dim = np.array([p[1] for p in diag1 if p[0] == homology_dimension and p[1][1] != p[1][0]]) # Filter out (b,b) points
        diag2_dim = np.array([p[1] for p in diag2 if p[0] == homology_dimension and p[1][1] != p[1][0]])
        
        # Handle cases where one or both filtered diagrams are empty for that dimension
        if diag1_dim.shape[0] == 0 and diag2_dim.shape[0] == 0:
            logger.debug(f"Both diagrams empty for H{homology_dimension}. Drift is 0.")
            return 0.0
        # If one is empty and other is not, distance is sum of persistence of non-empty one (or other convention)
        # Wasserstein distance function in Gudhi might handle this.
        # Let's ensure they are 2D arrays of shape (n_points, 2)
        if diag1_dim.ndim == 1 and diag1_dim.shape[0] > 0 : diag1_dim = diag1_dim.reshape(-1,2)
        if diag2_dim.ndim == 1 and diag2_dim.shape[0] > 0 : diag2_dim = diag2_dim.reshape(-1,2)
        if diag1_dim.shape[0] == 0: diag1_dim = np.empty((0,2))
        if diag2_dim.shape[0] == 0: diag2_dim = np.empty((0,2))


        try:
            # Use order=1 for W1 distance. Default is 2.
            # Default delta=0.1 (approximation parameter for Wasserstein)
            distance = wasserstein_distance(diag1_dim, diag2_dim, order=1.0, internal_p=2.0) 
            logger.info(f"Calculated Wasserstein-1 distance for H{homology_dimension}: {distance:.6f}")
            return distance
        except Exception as e:
            logger.error(f"HomologyDriftSensor: Error calculating Wasserstein distance: {e}")
            return None

# Example usage:
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    sensor = HomologyDriftSensor(max_edge_length=5.0) # Smaller max_edge_length for faster example

    # Create two simple graphs (representing two versions of a module structure)
    g1 = nx.Graph()
    g1.add_edges_from([(0,1), (1,2), (2,3), (3,0)]) # A square

    g2 = nx.Graph()
    g2.add_edges_from([(0,1), (1,2), (2,3), (3,0), (0,2)]) # A square with a diagonal

    # Convert graphs to point clouds (very naive method for example)
    # In a real scenario, these representations would be more meaningful (e.g. from ASTs, embeddings)
    # For now, using the sensor's internal _create_point_cloud_from_graph
    
    # Need an AlgebraicConnectivitySensor instance if we were to use its graph builder
    # For this standalone test, let's just create simple point clouds directly or use internal helper
    
    pc1 = sensor._create_point_cloud_from_graph(g1)
    pc2 = sensor._create_point_cloud_from_graph(g2)

    if pc1 is not None and pc2 is not None:
        logger.info(f"Point cloud 1 shape: {pc1.shape}")
        logger.info(f"Point cloud 2 shape: {pc2.shape}")

        # Calculate drift for H0 (connected components)
        drift_h0 = sensor.calculate_drift(pc1, pc2, homology_dimension=0)
        if drift_h0 is not None:
            logger.info(f"H0 Drift (Wasserstein-1 distance): {drift_h0:.4f}")

        # Calculate drift for H1 (cycles/holes)
        drift_h1 = sensor.calculate_drift(pc1, pc2, homology_dimension=1)
        if drift_h1 is not None:
            logger.info(f"H1 Drift (Wasserstein-1 distance): {drift_h1:.4f}")
    else:
        logger.error("Failed to create point clouds for testing.")

    # Example with slightly more complex point clouds
    points1 = np.array([[0,0], [1,0], [0,1], [1,1], [0.5, 0.5]]) # Square with center
    points2 = np.array([[0,0], [1,0], [0,1], [1,1], [2,2]])    # Square and an outlier
    
    drift_h1_complex = sensor.calculate_drift(points1, points2, homology_dimension=1)
    if drift_h1_complex is not None:
        logger.info(f"H1 Drift for complex point clouds: {drift_h1_complex:.4f}")
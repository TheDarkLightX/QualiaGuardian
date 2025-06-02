import unittest
import numpy as np
import networkx as nx
from unittest.mock import patch, MagicMock
import math # Added import

# Conditional import for gudhi to allow tests to run if gudhi is not installed
try:
    import gudhi as gd
    from gudhi.wasserstein import wasserstein_distance
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False

from guardian.sensors.homology_drift_sensor import HomologyDriftSensor

@unittest.skipIf(not GUDHI_AVAILABLE, "Gudhi library not installed, skipping HomologyDriftSensor tests.")
class TestHomologyDriftSensor(unittest.TestCase):

    def setUp(self):
        self.sensor = HomologyDriftSensor(max_edge_length=2.0, max_dimension=2) # Use smaller params for tests

    def test_create_point_cloud_from_graph_empty(self):
        graph = nx.Graph()
        pc = self.sensor._create_point_cloud_from_graph(graph)
        self.assertIsNone(pc, "Point cloud for empty graph should be None")

    def test_create_point_cloud_from_graph_single_node(self):
        graph = nx.Graph()
        graph.add_node(0)
        pc = self.sensor._create_point_cloud_from_graph(graph)
        self.assertIsNotNone(pc)
        if pc is not None:
            self.assertEqual(pc.shape, (1,1), "Point cloud for single node graph should have shape (1,1)")

    def test_create_point_cloud_from_graph_simple(self):
        graph = nx.path_graph(3) # 0-1-2
        pc = self.sensor._create_point_cloud_from_graph(graph)
        self.assertIsNotNone(pc)
        if pc is not None:
            self.assertEqual(pc.shape[0], 3, "Point cloud should have 3 points for a 3-node graph")
            self.assertTrue(pc.shape[1] > 0, "Point cloud should have at least one dimension")

    def test_calculate_persistence_diagram_empty_input(self):
        diag = self.sensor.calculate_persistence_diagram(np.array([]))
        self.assertEqual(diag, [], "Persistence diagram for empty point cloud should be empty list")

    def test_calculate_persistence_diagram_simple_points(self):
        # Three points forming a small triangle, plus one outlier
        points = np.array([[0,0], [1,0], [0,1], [10,10]])
        diag = self.sensor.calculate_persistence_diagram(points)
        self.assertIsNotNone(diag)
        # Basic check: H0 should have one infinite bar if points are somewhat connected by Rips
        # H1 might capture the triangle depending on max_edge_length
        h0_points = [p for p in diag if p[0] == 0]
        self.assertTrue(any(p[1][1] == float('inf') for p in h0_points), "H0 should have at least one infinite persistence bar")

    @patch('gudhi.RipsComplex')
    def test_calculate_persistence_diagram_gudhi_error(self, MockRipsComplex):
        # Simulate an error during Gudhi's processing
        mock_simplex_tree = MagicMock()
        mock_simplex_tree.persistence.side_effect = Exception("Gudhi internal error")
        
        mock_rips_instance = MagicMock()
        mock_rips_instance.create_simplex_tree.return_value = mock_simplex_tree
        MockRipsComplex.return_value = mock_rips_instance
        
        points = np.array([[0,0], [1,1]])
        diag = self.sensor.calculate_persistence_diagram(points)
        self.assertEqual(diag, [], "Should return empty list on Gudhi error")

    def test_calculate_drift_identical_diagrams(self):
        # Create a simple diagram (dim, birth, death)
        # Note: sensor's calculate_persistence_diagram returns list of (dim, (birth, death))
        # The calculate_drift method filters this by homology_dimension and converts to Nx2 array for gudhi.wasserstein
        points = np.array([[0,0], [1,0], [0.5, 0.866]]) # Equilateral triangle
        
        # We need to mock calculate_persistence_diagram to return controlled diagrams
        # or ensure that identical point clouds produce identical diagrams for this test.
        # For simplicity, let's test calculate_drift with pre-made diagrams.
        
        # Mocking the internal call to self.calculate_persistence_diagram
        with patch.object(self.sensor, 'calculate_persistence_diagram') as mock_calc_diag:
            # Diagram for H1 (a cycle)
            example_diag_h1 = [(1, (0.5, 1.0))] # One H1 feature
            # Diagram for H0 (components)
            example_diag_h0 = [(0, (0.0, 1.0)), (0, (0.0, float('inf')))]


            mock_calc_diag.return_value = example_diag_h1 + example_diag_h0 # Combined diagram

            drift_h1 = self.sensor.calculate_drift(points, points, homology_dimension=1)
            self.assertIsNotNone(drift_h1)
            self.assertAlmostEqual(drift_h1, 0.0, places=6, msg="Drift between identical H1 diagrams should be 0")
            
            drift_h0 = self.sensor.calculate_drift(points, points, homology_dimension=0)
            self.assertIsNotNone(drift_h0)
            self.assertAlmostEqual(drift_h0, 0.0, places=6, msg="Drift between identical H0 diagrams should be 0")


    def test_calculate_drift_different_diagrams_h0(self):
        # Diagram 1: Two components, one infinite
        diag1_raw = [(0, (0.0, 0.5)), (0, (0.0, float('inf')))]
        # Diagram 2: One component, infinite
        diag2_raw = [(0, (0.0, float('inf')))]

        with patch.object(self.sensor, 'calculate_persistence_diagram') as mock_calc_diag:
            mock_calc_diag.side_effect = [diag1_raw, diag2_raw] # First call gets diag1, second gets diag2
            
            # Dummy point clouds, as calculate_persistence_diagram is mocked
            dummy_pc = np.array([[0,0]]) 
            drift = self.sensor.calculate_drift(dummy_pc, dummy_pc, homology_dimension=0)
            
            self.assertIsNotNone(drift)
            self.assertGreater(drift, 0.0, "Drift for different H0 diagrams should be positive")
            # Expected W1 distance for H0: point (0.0, 0.5) vs its projection on diagonal.
            # Distance = (death - birth) / sqrt(2) for L2 projection.
            expected_distance = (0.5 - 0.0) / math.sqrt(2)
            self.assertAlmostEqual(drift, expected_distance, places=5)


    def test_calculate_drift_one_diagram_empty(self):
        diag1_raw = [(1, (0.5, 1.0))] # One H1 cycle
        diag2_raw = [] # Empty diagram (no H1 cycles)

        with patch.object(self.sensor, 'calculate_persistence_diagram') as mock_calc_diag:
            mock_calc_diag.side_effect = [diag1_raw, diag2_raw]
            dummy_pc = np.array([[0,0]])
            drift = self.sensor.calculate_drift(dummy_pc, dummy_pc, homology_dimension=1)
            self.assertIsNotNone(drift)
            expected_distance = (1.0 - 0.5) / math.sqrt(2)
            self.assertAlmostEqual(drift, expected_distance, places=5)

            # Test other way around
            mock_calc_diag.side_effect = [diag2_raw, diag1_raw]
            drift_rev = self.sensor.calculate_drift(dummy_pc, dummy_pc, homology_dimension=1)
            self.assertIsNotNone(drift_rev)
            self.assertAlmostEqual(drift_rev, expected_distance, places=5)


    def test_calculate_drift_no_diagrams_for_dimension(self):
        # Both diagrams have H0 features, but no H1 features
        diag_common = [(0, (0.0, float('inf')))]
        with patch.object(self.sensor, 'calculate_persistence_diagram') as mock_calc_diag:
            mock_calc_diag.return_value = diag_common
            dummy_pc = np.array([[0,0]])
            drift_h1 = self.sensor.calculate_drift(dummy_pc, dummy_pc, homology_dimension=1)
            self.assertIsNotNone(drift_h1)
            self.assertEqual(drift_h1, 0.0, "Drift should be 0 if no features in specified dimension for both")

if __name__ == '__main__':
    unittest.main()
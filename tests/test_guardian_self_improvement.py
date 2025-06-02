"""
Guardian Self-Improvement Test Suite

Tests for Guardian's ability to improve itself using E-TES v2.0 metrics.
Critical behavior coverage for the self-improvement optimizer.
"""

import pytest
import tempfile
import shutil
import os
import sys
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any, List

# Add guardian to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'guardian'))

from guardian.self_improvement.guardian_optimizer import GuardianOptimizer, SelectionMode, ImprovementTarget
from guardian.self_improvement.gamified_monitor import GamifiedMonitor, AchievementSystem
from guardian.self_improvement.console_interface import BeautifulConsole, ProgressTracker
from guardian.core.etes import ETESCalculator, ETESComponents


class TestGuardianOptimizerCore:
    """Test core Guardian self-improvement functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create mock Guardian codebase structure
        self._create_mock_guardian_structure()
        
        self.optimizer = GuardianOptimizer(
            guardian_root=str(self.temp_path),
            selection_mode=SelectionMode.GUIDED
        )
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_should_initialize_with_guided_mode_when_specified(self):
        """Test optimizer initialization with guided selection mode"""
        optimizer = GuardianOptimizer(
            guardian_root=str(self.temp_path),
            selection_mode=SelectionMode.GUIDED
        )
        
        assert optimizer.guardian_root == str(self.temp_path)
        assert optimizer.selection_mode == SelectionMode.GUIDED
        assert optimizer.current_iteration == 0
        assert len(optimizer.improvement_history) == 0
        assert optimizer.baseline_etes_score is None
    
    def test_should_initialize_with_random_mode_when_specified(self):
        """Test optimizer initialization with random selection mode"""
        optimizer = GuardianOptimizer(
            guardian_root=str(self.temp_path),
            selection_mode=SelectionMode.RANDOM
        )
        
        assert optimizer.selection_mode == SelectionMode.RANDOM
    
    def test_should_initialize_with_hybrid_mode_when_specified(self):
        """Test optimizer initialization with hybrid selection mode"""
        optimizer = GuardianOptimizer(
            guardian_root=str(self.temp_path),
            selection_mode=SelectionMode.HYBRID
        )
        
        assert optimizer.selection_mode == SelectionMode.HYBRID
    
    def test_should_identify_improvement_targets_when_analyzing_codebase(self):
        """Test identification of improvement targets"""
        targets = self.optimizer.identify_improvement_targets()
        
        assert isinstance(targets, list)
        assert len(targets) > 0
        assert all(isinstance(target, ImprovementTarget) for target in targets)
        
        # Should have core targets
        target_names = [t.name for t in targets]
        assert 'Core E-TES Engine' in target_names
        assert 'Evolution Algorithms' in target_names
        assert 'Static Analysis' in target_names
    
    def test_should_prioritize_targets_by_impact_when_guided_mode_used(self):
        """Test target prioritization in guided mode"""
        targets = self.optimizer.identify_improvement_targets()
        
        # In guided mode, should prioritize by impact and priority
        assert targets[0].priority <= targets[-1].priority  # Higher priority first
        assert targets[0].estimated_impact >= targets[-1].estimated_impact  # Higher impact first
    
    def test_should_select_random_targets_when_random_mode_used(self):
        """Test random target selection"""
        optimizer = GuardianOptimizer(
            guardian_root=str(self.temp_path),
            selection_mode=SelectionMode.RANDOM
        )
        
        targets = optimizer.identify_improvement_targets()
        selected = optimizer.select_improvement_target(targets)
        
        assert selected in targets
        # In random mode, any target could be selected
    
    def test_should_mix_guided_and_random_when_hybrid_mode_used(self):
        """Test hybrid selection mode behavior"""
        optimizer = GuardianOptimizer(
            guardian_root=str(self.temp_path),
            selection_mode=SelectionMode.HYBRID
        )
        
        targets = optimizer.identify_improvement_targets()
        
        # Run multiple selections to test hybrid behavior
        selections = []
        for _ in range(10):
            selected = optimizer.select_improvement_target(targets)
            selections.append(selected)
        
        # Should have some variety (not always the same target)
        unique_selections = set(s.name for s in selections)
        assert len(unique_selections) > 1  # Should select different targets
    
    @patch('guardian.self_improvement.guardian_optimizer.ETESCalculator')
    def test_should_calculate_baseline_etes_when_first_run(self, mock_etes_calc):
        """Test baseline E-TES calculation on first run"""
        # Mock E-TES calculator
        mock_calc_instance = Mock()
        mock_calc_instance.calculate_etes.return_value = (0.65, self._create_mock_components())
        mock_etes_calc.return_value = mock_calc_instance
        
        baseline_score = self.optimizer.calculate_current_etes_score()
        
        assert baseline_score == 0.65
        assert self.optimizer.baseline_etes_score == 0.65
        mock_calc_instance.calculate_etes.assert_called_once()
    
    @patch('guardian.self_improvement.guardian_optimizer.ETESCalculator')
    def test_should_apply_improvement_when_target_selected(self, mock_etes_calc):
        """Test improvement application to selected target"""
        # Mock E-TES calculator
        mock_calc_instance = Mock()
        mock_calc_instance.calculate_etes.return_value = (0.75, self._create_mock_components())
        mock_etes_calc.return_value = mock_calc_instance
        
        targets = self.optimizer.identify_improvement_targets()
        target = targets[0]  # Select first target
        
        result = self.optimizer.apply_improvement(target)
        
        assert result['success'] is True
        assert 'changes_made' in result
        assert 'new_etes_score' in result
        assert result['new_etes_score'] > 0
    
    def test_should_track_improvement_history_when_iterations_performed(self):
        """Test improvement history tracking"""
        with patch.object(self.optimizer, 'calculate_current_etes_score', return_value=0.7):
            with patch.object(self.optimizer, 'apply_improvement') as mock_apply:
                mock_apply.return_value = {
                    'success': True,
                    'changes_made': ['Improved function X'],
                    'new_etes_score': 0.75
                }
                
                targets = self.optimizer.identify_improvement_targets()
                target = targets[0]
                
                self.optimizer.apply_improvement(target)
                
                assert len(self.optimizer.improvement_history) == 1
                history_entry = self.optimizer.improvement_history[0]
                assert history_entry['target_name'] == target.name
                assert history_entry['success'] is True
                assert history_entry['new_score'] == 0.75
    
    def test_should_run_complete_improvement_cycle_when_requested(self):
        """Test complete improvement cycle execution"""
        with patch.object(self.optimizer, 'calculate_current_etes_score') as mock_calc:
            mock_calc.side_effect = [0.6, 0.65, 0.7]  # Progressive improvement
            
            with patch.object(self.optimizer, 'apply_improvement') as mock_apply:
                mock_apply.return_value = {
                    'success': True,
                    'changes_made': ['Improvement applied'],
                    'new_etes_score': 0.7
                }
                
                results = self.optimizer.run_improvement_cycle(max_iterations=2)
                
                assert isinstance(results, dict)
                assert 'iterations_completed' in results
                assert 'final_score' in results
                assert 'improvements_made' in results
                assert results['iterations_completed'] == 2
    
    def test_should_stop_early_when_target_score_reached(self):
        """Test early stopping when target score is achieved"""
        with patch.object(self.optimizer, 'calculate_current_etes_score') as mock_calc:
            mock_calc.side_effect = [0.6, 0.85]  # Reaches target quickly
            
            with patch.object(self.optimizer, 'apply_improvement') as mock_apply:
                mock_apply.return_value = {
                    'success': True,
                    'changes_made': ['Major improvement'],
                    'new_etes_score': 0.85
                }
                
                results = self.optimizer.run_improvement_cycle(
                    max_iterations=10,
                    target_score=0.8
                )
                
                assert results['iterations_completed'] < 10  # Should stop early
                assert results['final_score'] >= 0.8
    
    def test_should_handle_improvement_failure_gracefully(self):
        """Test handling of improvement failures"""
        with patch.object(self.optimizer, 'apply_improvement') as mock_apply:
            mock_apply.return_value = {
                'success': False,
                'error': 'Improvement failed',
                'new_etes_score': 0.6
            }
            
            targets = self.optimizer.identify_improvement_targets()
            target = targets[0]
            
            result = self.optimizer.apply_improvement(target)
            
            assert result['success'] is False
            assert 'error' in result
            # Should not crash or raise exceptions
    
    def test_should_generate_improvement_summary_when_cycle_complete(self):
        """Test improvement summary generation"""
        # Add some mock history
        self.optimizer.improvement_history = [
            {
                'iteration': 1,
                'target_name': 'Core E-TES Engine',
                'success': True,
                'old_score': 0.6,
                'new_score': 0.65,
                'improvement': 0.05,
                'changes_made': ['Optimized calculation']
            },
            {
                'iteration': 2,
                'target_name': 'Evolution Algorithms',
                'success': True,
                'old_score': 0.65,
                'new_score': 0.72,
                'improvement': 0.07,
                'changes_made': ['Improved convergence']
            }
        ]
        self.optimizer.baseline_etes_score = 0.6
        
        summary = self.optimizer.get_improvement_summary()
        
        assert summary is not None
        assert 'total_iterations' in summary
        assert 'success_rate' in summary
        assert 'overall_improvement' in summary
        assert 'final_score' in summary
        assert 'targets_achieved' in summary
        
        assert summary['total_iterations'] == 2
        assert summary['success_rate'] == 1.0  # 100% success
        assert summary['overall_improvement'] == 0.12  # 0.72 - 0.6
        assert summary['final_score'] == 0.72
    
    def test_should_complete_within_performance_threshold(self):
        """Test performance of improvement cycle"""
        with patch.object(self.optimizer, 'calculate_current_etes_score', return_value=0.7):
            with patch.object(self.optimizer, 'apply_improvement') as mock_apply:
                mock_apply.return_value = {
                    'success': True,
                    'changes_made': ['Quick improvement'],
                    'new_etes_score': 0.75
                }
                
                start_time = time.time()
                self.optimizer.run_improvement_cycle(max_iterations=1)
                end_time = time.time()
                
                execution_time = (end_time - start_time) * 1000  # Convert to ms
                assert execution_time < 5000  # Should complete in <5 seconds
    
    def _create_mock_guardian_structure(self):
        """Create mock Guardian codebase structure"""
        # Create core directories
        (self.temp_path / 'guardian' / 'core').mkdir(parents=True)
        (self.temp_path / 'guardian' / 'analysis').mkdir(parents=True)
        (self.temp_path / 'guardian' / 'evolution').mkdir(parents=True)
        (self.temp_path / 'guardian' / 'self_improvement').mkdir(parents=True)
        
        # Create mock Python files
        files_to_create = [
            'guardian/core/etes.py',
            'guardian/core/tes.py',
            'guardian/analysis/static.py',
            'guardian/analysis/security.py',
            'guardian/evolution/adaptive_emt.py',
            'guardian/evolution/smart_mutator.py',
            'guardian/self_improvement/guardian_optimizer.py',
            'guardian/cli.py'
        ]
        
        for file_path in files_to_create:
            full_path = self.temp_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(f'# Mock {file_path}\nclass MockClass:\n    pass\n')
    
    def _create_mock_components(self) -> ETESComponents:
        """Create mock E-TES components"""
        return ETESComponents(
            mutation_score=0.75,
            evolution_gain=1.1,
            assertion_iq=0.8,
            behavior_coverage=0.85,
            speed_factor=0.9,
            quality_factor=0.88,
            insights=['Mock insight'],
            calculation_time=50.0
        )


class TestGamifiedMonitor:
    """Test gamified monitoring and achievement system"""
    
    def setup_method(self):
        """Set up test environment"""
        self.achievement_system = AchievementSystem()
        self.monitor = GamifiedMonitor()
    
    def test_should_initialize_achievement_system_when_created(self):
        """Test achievement system initialization"""
        assert self.achievement_system.current_level == 1
        assert self.achievement_system.total_experience == 0
        assert len(self.achievement_system.unlocked_achievements) == 0
        assert len(self.achievement_system.available_achievements) > 0
    
    def test_should_unlock_first_steps_achievement_when_score_improves(self):
        """Test First Steps achievement unlock"""
        components = ETESComponents(
            mutation_score=0.6,
            evolution_gain=1.0,
            assertion_iq=0.5,
            behavior_coverage=0.7,
            speed_factor=0.8,
            quality_factor=0.75,
            insights=[],
            calculation_time=100.0
        )
        
        newly_unlocked = self.achievement_system.update_progress(
            etes_score=0.6,
            components=components,
            optimization_time=30.0,
            mutations_killed=5
        )
        
        achievement_names = [ach['name'] for ach in newly_unlocked]
        assert 'First Steps' in achievement_names
    
    def test_should_unlock_speed_demon_when_fast_optimization(self):
        """Test Speed Demon achievement unlock"""
        components = ETESComponents(
            mutation_score=0.8,
            evolution_gain=1.2,
            assertion_iq=0.8,
            behavior_coverage=0.85,
            speed_factor=0.95,  # High speed factor
            quality_factor=0.9,
            insights=[],
            calculation_time=15.0  # Fast calculation
        )
        
        newly_unlocked = self.achievement_system.update_progress(
            etes_score=0.85,
            components=components,
            optimization_time=8.0,  # Very fast optimization
            mutations_killed=12
        )
        
        achievement_names = [ach['name'] for ach in newly_unlocked]
        assert 'Speed Demon' in achievement_names
    
    def test_should_level_up_when_sufficient_experience_gained(self):
        """Test level progression"""
        initial_level = self.achievement_system.current_level
        
        # Simulate multiple improvements to gain experience
        for i in range(5):
            components = ETESComponents(
                mutation_score=0.7 + i * 0.05,
                evolution_gain=1.1 + i * 0.1,
                assertion_iq=0.8,
                behavior_coverage=0.85,
                speed_factor=0.9,
                quality_factor=0.88,
                insights=[],
                calculation_time=50.0
            )
            
            self.achievement_system.update_progress(
                etes_score=0.7 + i * 0.05,
                components=components,
                optimization_time=25.0,
                mutations_killed=8 + i
            )
        
        assert self.achievement_system.current_level > initial_level
        assert self.achievement_system.total_experience > 0
    
    def test_should_display_dashboard_without_errors(self):
        """Test dashboard display functionality"""
        components = ETESComponents(
            mutation_score=0.8,
            evolution_gain=1.15,
            assertion_iq=0.85,
            behavior_coverage=0.9,
            speed_factor=0.88,
            quality_factor=0.92,
            insights=['Great progress!'],
            calculation_time=45.0
        )
        
        # Should not raise any exceptions
        try:
            self.achievement_system.display_dashboard(0.85, components)
            dashboard_displayed = True
        except Exception:
            dashboard_displayed = False
        
        assert dashboard_displayed is True


class TestBeautifulConsole:
    """Test beautiful console interface"""
    
    def setup_method(self):
        """Set up test environment"""
        self.console = BeautifulConsole()
    
    def test_should_create_colored_output_when_colors_enabled(self):
        """Test colored output generation"""
        header = self.console.print_header("Test Header")
        
        # Should contain ANSI color codes when colors are enabled
        assert isinstance(header, str)
        # Color codes should be present (or gracefully handled)
    
    def test_should_create_progress_tracker_when_requested(self):
        """Test progress tracker creation"""
        tracker = ProgressTracker(10, "Test Progress")
        
        assert tracker.total_steps == 10
        assert tracker.description == "Test Progress"
        assert tracker.current_step == 0
    
    def test_should_update_progress_tracker_when_step_completed(self):
        """Test progress tracker updates"""
        tracker = ProgressTracker(5, "Test Progress")
        
        tracker.update(message="Step 1")
        assert tracker.current_step == 1
        
        tracker.update(message="Step 2")
        assert tracker.current_step == 2
    
    def test_should_complete_progress_tracker_when_all_steps_done(self):
        """Test progress tracker completion"""
        tracker = ProgressTracker(3, "Test Progress")
        
        for i in range(3):
            tracker.update(message=f"Step {i+1}")
        
        assert tracker.current_step == 3
        # Should handle completion gracefully
    
    def test_should_create_boxes_for_content_display(self):
        """Test box creation for content display"""
        content = "Test content\nMultiple lines\nMore content"
        
        box = self.console.Box.create(
            content,
            width=50,
            title="Test Box",
            color=self.console.Color.BLUE
        )
        
        assert isinstance(box, str)
        assert len(box) > 0
        assert "Test Box" in box or "Test content" in box


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

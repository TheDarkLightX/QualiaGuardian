"""
Evolution History Tracker

Tracks and analyzes the evolution of test suites over time,
providing insights into improvement trends and optimization patterns.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import sqlite3
import os

logger = logging.getLogger(__name__)


@dataclass
class EvolutionSnapshot:
    """Snapshot of evolution state at a point in time"""
    timestamp: float
    generation: int
    etes_score: float
    mutation_score: float
    assertion_iq: float
    behavior_coverage: float
    speed_factor: float
    quality_factor: float
    population_size: int
    best_individual_id: str
    diversity_score: float
    mutation_rate: float
    convergence_indicator: float
    
    # Performance metrics
    evaluation_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Metadata
    notes: str = ""
    experiment_id: str = ""


@dataclass
class TrendAnalysis:
    """Analysis of evolution trends"""
    improvement_rate: float
    convergence_generation: Optional[int]
    plateau_detected: bool
    trend_direction: str  # 'improving', 'declining', 'stable'
    confidence: float
    
    # Component-specific trends
    component_trends: Dict[str, float]
    
    # Predictions
    predicted_final_score: float
    estimated_generations_to_convergence: Optional[int]


class EvolutionHistoryTracker:
    """
    Comprehensive evolution history tracking and analysis system
    """
    
    def __init__(self, db_path: str = "evolution_history.db", 
                 max_snapshots: int = 1000):
        self.db_path = db_path
        self.max_snapshots = max_snapshots
        self.snapshots: List[EvolutionSnapshot] = []
        
        # Initialize database
        self._init_database()
        
        # Load existing history
        self._load_history()
    
    def record_snapshot(self, snapshot: EvolutionSnapshot) -> None:
        """Record a new evolution snapshot"""
        try:
            # Add to memory
            self.snapshots.append(snapshot)
            
            # Maintain size limit
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots = self.snapshots[-self.max_snapshots:]
            
            # Persist to database
            self._save_snapshot_to_db(snapshot)
            
            logger.debug(f"Recorded evolution snapshot: gen {snapshot.generation}, score {snapshot.etes_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error recording snapshot: {e}")
    
    def analyze_trends(self, window_size: int = 10) -> TrendAnalysis:
        """
        Analyze evolution trends and patterns
        
        Args:
            window_size: Number of recent snapshots to analyze
            
        Returns:
            Comprehensive trend analysis
        """
        try:
            if len(self.snapshots) < 3:
                return TrendAnalysis(
                    improvement_rate=0.0,
                    convergence_generation=None,
                    plateau_detected=False,
                    trend_direction='stable',
                    confidence=0.0,
                    component_trends={},
                    predicted_final_score=0.0,
                    estimated_generations_to_convergence=None
                )
            
            recent_snapshots = self.snapshots[-window_size:]
            
            # Calculate improvement rate
            improvement_rate = self._calculate_improvement_rate(recent_snapshots)
            
            # Detect convergence
            convergence_gen = self._detect_convergence()
            
            # Detect plateau
            plateau_detected = self._detect_plateau(recent_snapshots)
            
            # Determine trend direction
            trend_direction = self._determine_trend_direction(recent_snapshots)
            
            # Calculate confidence
            confidence = self._calculate_trend_confidence(recent_snapshots)
            
            # Analyze component trends
            component_trends = self._analyze_component_trends(recent_snapshots)
            
            # Make predictions
            predicted_score = self._predict_final_score(recent_snapshots)
            estimated_gens = self._estimate_convergence_time(recent_snapshots)
            
            return TrendAnalysis(
                improvement_rate=improvement_rate,
                convergence_generation=convergence_gen,
                plateau_detected=plateau_detected,
                trend_direction=trend_direction,
                confidence=confidence,
                component_trends=component_trends,
                predicted_final_score=predicted_score,
                estimated_generations_to_convergence=estimated_gens
            )
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return TrendAnalysis(
                improvement_rate=0.0,
                convergence_generation=None,
                plateau_detected=False,
                trend_direction='stable',
                confidence=0.0,
                component_trends={},
                predicted_final_score=0.0,
                estimated_generations_to_convergence=None
            )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.snapshots:
            return {}
        
        scores = [s.etes_score for s in self.snapshots]
        generations = [s.generation for s in self.snapshots]
        
        return {
            'total_generations': len(self.snapshots),
            'best_score': max(scores),
            'worst_score': min(scores),
            'average_score': np.mean(scores),
            'final_score': scores[-1],
            'improvement_total': scores[-1] - scores[0] if len(scores) > 1 else 0.0,
            'best_generation': generations[np.argmax(scores)],
            'convergence_generation': self._detect_convergence(),
            'total_evolution_time': self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
            'average_generation_time': np.mean([s.evaluation_time_ms for s in self.snapshots]),
            'peak_memory_usage': max([s.memory_usage_mb for s in self.snapshots]),
        }
    
    def export_history(self, format: str = 'json') -> str:
        """Export evolution history in specified format"""
        try:
            if format.lower() == 'json':
                return json.dumps([asdict(snapshot) for snapshot in self.snapshots], indent=2)
            elif format.lower() == 'csv':
                return self._export_csv()
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting history: {e}")
            return ""
    
    def get_insights(self) -> List[str]:
        """Generate actionable insights from evolution history"""
        insights = []
        
        if len(self.snapshots) < 3:
            insights.append("Insufficient evolution history for detailed insights")
            return insights
        
        trend_analysis = self.analyze_trends()
        
        # Performance insights
        if trend_analysis.improvement_rate > 0.05:
            insights.append("Strong improvement trend detected - evolution is working well")
        elif trend_analysis.improvement_rate < -0.02:
            insights.append("Declining performance detected - consider adjusting parameters")
        elif trend_analysis.plateau_detected:
            insights.append("Performance plateau reached - consider increasing mutation rate or population diversity")
        
        # Convergence insights
        if trend_analysis.convergence_generation:
            insights.append(f"Convergence detected at generation {trend_analysis.convergence_generation}")
        elif trend_analysis.estimated_generations_to_convergence:
            insights.append(f"Estimated {trend_analysis.estimated_generations_to_convergence} generations to convergence")
        
        # Component-specific insights
        for component, trend in trend_analysis.component_trends.items():
            if trend < -0.1:
                insights.append(f"{component} is declining - focus optimization efforts here")
            elif trend > 0.1:
                insights.append(f"{component} is improving well")
        
        # Resource insights
        recent_snapshots = self.snapshots[-5:]
        avg_eval_time = np.mean([s.evaluation_time_ms for s in recent_snapshots])
        if avg_eval_time > 5000:  # 5 seconds
            insights.append("Evolution evaluation is slow - consider optimizing fitness functions")
        
        avg_memory = np.mean([s.memory_usage_mb for s in recent_snapshots])
        if avg_memory > 1000:  # 1GB
            insights.append("High memory usage detected - consider reducing population size")
        
        return insights
    
    def _calculate_improvement_rate(self, snapshots: List[EvolutionSnapshot]) -> float:
        """Calculate rate of improvement over snapshots"""
        if len(snapshots) < 2:
            return 0.0
        
        scores = [s.etes_score for s in snapshots]
        generations = [s.generation for s in snapshots]
        
        # Linear regression to find improvement rate
        if len(scores) >= 2:
            coeffs = np.polyfit(generations, scores, 1)
            return coeffs[0]  # Slope = improvement rate per generation
        
        return 0.0
    
    def _detect_convergence(self, threshold: float = 0.01, window: int = 5) -> Optional[int]:
        """Detect convergence generation"""
        if len(self.snapshots) < window + 2:
            return None
        
        for i in range(window, len(self.snapshots)):
            recent_scores = [s.etes_score for s in self.snapshots[i-window:i]]
            score_variance = np.var(recent_scores)
            
            if score_variance < threshold:
                return self.snapshots[i-window].generation
        
        return None
    
    def _detect_plateau(self, snapshots: List[EvolutionSnapshot], 
                       threshold: float = 0.005) -> bool:
        """Detect if evolution has plateaued"""
        if len(snapshots) < 5:
            return False
        
        recent_scores = [s.etes_score for s in snapshots[-5:]]
        score_range = max(recent_scores) - min(recent_scores)
        
        return score_range < threshold
    
    def _determine_trend_direction(self, snapshots: List[EvolutionSnapshot]) -> str:
        """Determine overall trend direction"""
        if len(snapshots) < 3:
            return 'stable'
        
        improvement_rate = self._calculate_improvement_rate(snapshots)
        
        if improvement_rate > 0.01:
            return 'improving'
        elif improvement_rate < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_trend_confidence(self, snapshots: List[EvolutionSnapshot]) -> float:
        """Calculate confidence in trend analysis"""
        if len(snapshots) < 3:
            return 0.0
        
        scores = [s.etes_score for s in snapshots]
        
        # Higher confidence with more data points and consistent trends
        data_confidence = min(len(snapshots) / 10.0, 1.0)
        
        # Calculate trend consistency
        if len(scores) >= 3:
            differences = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
            sign_changes = sum(1 for i in range(len(differences)-1) 
                             if differences[i] * differences[i+1] < 0)
            consistency = 1.0 - (sign_changes / len(differences))
        else:
            consistency = 0.5
        
        return (data_confidence + consistency) / 2.0
    
    def _analyze_component_trends(self, snapshots: List[EvolutionSnapshot]) -> Dict[str, float]:
        """Analyze trends for individual E-TES components"""
        if len(snapshots) < 2:
            return {}
        
        components = {
            'mutation_score': [s.mutation_score for s in snapshots],
            'assertion_iq': [s.assertion_iq for s in snapshots],
            'behavior_coverage': [s.behavior_coverage for s in snapshots],
            'speed_factor': [s.speed_factor for s in snapshots],
            'quality_factor': [s.quality_factor for s in snapshots]
        }
        
        trends = {}
        generations = [s.generation for s in snapshots]
        
        for component, values in components.items():
            if len(values) >= 2:
                coeffs = np.polyfit(generations, values, 1)
                trends[component] = coeffs[0]  # Slope
            else:
                trends[component] = 0.0
        
        return trends
    
    def _predict_final_score(self, snapshots: List[EvolutionSnapshot]) -> float:
        """Predict final E-TES score based on current trends"""
        if len(snapshots) < 3:
            return snapshots[-1].etes_score if snapshots else 0.0
        
        scores = [s.etes_score for s in snapshots]
        generations = [s.generation for s in snapshots]
        
        # Fit exponential decay model for convergence prediction
        try:
            # Simple linear extrapolation for now
            coeffs = np.polyfit(generations, scores, 1)
            
            # Predict score at generation + 20
            future_gen = generations[-1] + 20
            predicted = coeffs[0] * future_gen + coeffs[1]
            
            # Clamp to reasonable bounds
            return max(0.0, min(predicted, 1.0))
            
        except Exception:
            return scores[-1]
    
    def _estimate_convergence_time(self, snapshots: List[EvolutionSnapshot]) -> Optional[int]:
        """Estimate generations until convergence"""
        if len(snapshots) < 3:
            return None
        
        improvement_rate = self._calculate_improvement_rate(snapshots)
        
        if abs(improvement_rate) < 0.001:
            return 0  # Already converged
        
        current_score = snapshots[-1].etes_score
        target_score = min(current_score + 0.1, 1.0)  # Target 10% improvement or max
        
        if improvement_rate > 0:
            generations_needed = (target_score - current_score) / improvement_rate
            return max(1, int(generations_needed))
        
        return None
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS evolution_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        generation INTEGER,
                        etes_score REAL,
                        mutation_score REAL,
                        assertion_iq REAL,
                        behavior_coverage REAL,
                        speed_factor REAL,
                        quality_factor REAL,
                        population_size INTEGER,
                        best_individual_id TEXT,
                        diversity_score REAL,
                        mutation_rate REAL,
                        convergence_indicator REAL,
                        evaluation_time_ms REAL,
                        memory_usage_mb REAL,
                        notes TEXT,
                        experiment_id TEXT
                    )
                ''')
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _save_snapshot_to_db(self, snapshot: EvolutionSnapshot):
        """Save snapshot to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO evolution_snapshots (
                        timestamp, generation, etes_score, mutation_score,
                        assertion_iq, behavior_coverage, speed_factor, quality_factor,
                        population_size, best_individual_id, diversity_score,
                        mutation_rate, convergence_indicator, evaluation_time_ms,
                        memory_usage_mb, notes, experiment_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot.timestamp, snapshot.generation, snapshot.etes_score,
                    snapshot.mutation_score, snapshot.assertion_iq, snapshot.behavior_coverage,
                    snapshot.speed_factor, snapshot.quality_factor, snapshot.population_size,
                    snapshot.best_individual_id, snapshot.diversity_score, snapshot.mutation_rate,
                    snapshot.convergence_indicator, snapshot.evaluation_time_ms,
                    snapshot.memory_usage_mb, snapshot.notes, snapshot.experiment_id
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving snapshot to database: {e}")
    
    def _load_history(self):
        """Load evolution history from database"""
        try:
            if not os.path.exists(self.db_path):
                return
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT * FROM evolution_snapshots 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (self.max_snapshots,))
                
                rows = cursor.fetchall()
                
                for row in rows:
                    snapshot = EvolutionSnapshot(
                        timestamp=row[1],
                        generation=row[2],
                        etes_score=row[3],
                        mutation_score=row[4],
                        assertion_iq=row[5],
                        behavior_coverage=row[6],
                        speed_factor=row[7],
                        quality_factor=row[8],
                        population_size=row[9],
                        best_individual_id=row[10],
                        diversity_score=row[11],
                        mutation_rate=row[12],
                        convergence_indicator=row[13],
                        evaluation_time_ms=row[14],
                        memory_usage_mb=row[15],
                        notes=row[16] or "",
                        experiment_id=row[17] or ""
                    )
                    self.snapshots.append(snapshot)
                
                # Reverse to get chronological order
                self.snapshots.reverse()
                
        except Exception as e:
            logger.error(f"Error loading history from database: {e}")
    
    def _export_csv(self) -> str:
        """Export history as CSV"""
        if not self.snapshots:
            return ""
        
        headers = [
            'timestamp', 'generation', 'etes_score', 'mutation_score',
            'assertion_iq', 'behavior_coverage', 'speed_factor', 'quality_factor',
            'population_size', 'diversity_score', 'mutation_rate'
        ]
        
        lines = [','.join(headers)]
        
        for snapshot in self.snapshots:
            values = [
                str(snapshot.timestamp), str(snapshot.generation), str(snapshot.etes_score),
                str(snapshot.mutation_score), str(snapshot.assertion_iq), str(snapshot.behavior_coverage),
                str(snapshot.speed_factor), str(snapshot.quality_factor), str(snapshot.population_size),
                str(snapshot.diversity_score), str(snapshot.mutation_rate)
            ]
            lines.append(','.join(values))
        
        return '\n'.join(lines)

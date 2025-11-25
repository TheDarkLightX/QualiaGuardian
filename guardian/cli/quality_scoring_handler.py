"""
Quality Scoring Handler

Handles quality score calculation logic, following Single Responsibility Principle.
Extracted from cli.py to reduce complexity and improve maintainability.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from guardian.core.etes import QualityConfig
from guardian.core.betes import BETESWeights, classify_betes
from guardian.core.tes import calculate_quality_score, get_etes_grade

# Import sensor functions
from guardian.sensors import mutation as mutation_sensor
from guardian.sensors import assertion_iq as assertion_iq_sensor
from guardian.sensors import behaviour_coverage as behaviour_coverage_sensor
from guardian.sensors import speed as speed_sensor
from guardian.sensors import flakiness as flakiness_sensor

logger = logging.getLogger(__name__)


class QualityScoringHandler:
    """
    Handles quality scoring configuration and calculation.
    
    Follows Single Responsibility Principle by isolating quality scoring logic.
    """
    
    def __init__(self, project_path: str):
        """
        Initialize quality scoring handler.
        
        Args:
            project_path: Path to the project being analyzed
        """
        self.project_path = project_path
        self._config_dir = None
    
    @property
    def config_dir(self) -> Path:
        """Get the configuration directory path."""
        if self._config_dir is None:
            script_dir = Path(__file__).parent.parent.parent
            self._config_dir = script_dir / "config"
        return self._config_dir
    
    def create_quality_config(self, args: Any) -> QualityConfig:
        """
        Create and configure QualityConfig from CLI arguments.
        
        Args:
            args: Parsed CLI arguments
            
        Returns:
            Configured QualityConfig instance
        """
        betes_weights = BETESWeights(
            w_m=args.betes_w_m if args.betes_w_m is not None else 1.0,
            w_e=args.betes_w_e if args.betes_w_e is not None else 1.0,
            w_a=args.betes_w_a if args.betes_w_a is not None else 1.0,
            w_b=args.betes_w_b if args.betes_w_b is not None else 1.0,
            w_s=args.betes_w_s if args.betes_w_s is not None else 1.0,
        )
        
        # Normalize mode (betes_v3 -> betes_v3.1)
        mode = args.quality_mode
        if mode == "betes_v3":
            logger.info("Mode 'betes_v3' selected, defaulting to 'betes_v3.1' for calculations.")
            mode = "betes_v3.1"
        
        quality_cfg = QualityConfig(
            mode=mode,
            betes_weights=betes_weights,
            risk_class=args.risk_class,
            test_root_path=args.test_root_path or "tests/",
            coverage_file_path=args.coverage_file_path or "coverage.info",
            critical_behaviors_manifest_path=args.critical_behaviors_manifest_path,
            ci_platform=args.ci_platform
        )
        
        # Apply sigmoid settings for betes_v3.1
        if quality_cfg.mode == "betes_v3.1" and args.smooth_sigmoid:
            self._apply_sigmoid_settings(quality_cfg, args.smooth_sigmoid)
        
        # Apply E-TES v2 specific settings
        if quality_cfg.mode == "etes_v2":
            quality_cfg.max_generations = getattr(
                args, 'max_generations_etes', quality_cfg.max_generations
            )
            quality_cfg.min_mutation_score = getattr(
                args, 'min_mutation_score_etes', quality_cfg.min_mutation_score
            )
            quality_cfg.min_behavior_coverage = getattr(
                args, 'min_behavior_coverage_etes', quality_cfg.min_behavior_coverage
            )
        
        return quality_cfg
    
    def _apply_sigmoid_settings(self, quality_cfg: QualityConfig, smooth_sigmoid: str) -> None:
        """
        Apply sigmoid normalization settings.
        
        Args:
            quality_cfg: QualityConfig to modify
            smooth_sigmoid: Comma-separated string of options ('m', 'e', 'all')
        """
        smooth_options = [opt.strip().lower() for opt in smooth_sigmoid.split(',')]
        
        if "all" in smooth_options:
            quality_cfg.betes_v3_1_settings.smooth_m = True
            quality_cfg.betes_v3_1_settings.smooth_e = True
        else:
            if "m" in smooth_options:
                quality_cfg.betes_v3_1_settings.smooth_m = True
            if "e" in smooth_options:
                quality_cfg.betes_v3_1_settings.smooth_e = True
        
        logger.info(
            f"bE-TES v3.1 sigmoid settings: "
            f"smooth_m={quality_cfg.betes_v3_1_settings.smooth_m}, "
            f"smooth_e={quality_cfg.betes_v3_1_settings.smooth_e}"
        )
    
    def load_mutation_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """
        Load mutation sensor configuration from YAML file or use defaults.
        
        Args:
            config_file: Optional path to configuration file
            
        Returns:
            Dictionary with mutation sensor configuration
        """
        default_config = {
            "mutmut_paths_to_mutate": ["src"],
            "mutmut_runner_args": "pytest"
        }
        
        if not config_file:
            logger.info("No config file specified. Using default mutation settings.")
            return default_config
        
        try:
            config_path = Path(config_file)
            if not config_path.is_file():
                logger.warning(
                    f"Config file specified but not found: {config_file}. "
                    "Using default mutation settings."
                )
                return default_config
            
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
            
            if not loaded_config or not isinstance(loaded_config, dict):
                return default_config
            
            sensor_configs = loaded_config.get("sensors", {})
            mutation_config = sensor_configs.get("mutation", {})
            
            # Override defaults with config file values
            if "mutmut_paths_to_mutate" in mutation_config:
                paths = mutation_config["mutmut_paths_to_mutate"]
                default_config["mutmut_paths_to_mutate"] = (
                    [paths] if isinstance(paths, str) else paths
                )
            
            if "mutmut_runner_args" in mutation_config:
                default_config["mutmut_runner_args"] = mutation_config["mutmut_runner_args"]
            
            # Pass through other mutmut settings
            for key, value in mutation_config.items():
                if key not in ["mutmut_paths_to_mutate", "mutmut_runner_args"]:
                    default_config[key] = value
            
            logger.info(f"Loaded mutation sensor configuration from {config_file}")
            return default_config
            
        except ImportError:
            logger.error(
                "PyYAML is not installed. Please install it to use YAML config files "
                "(`pip install PyYAML`). Using default mutation settings."
            )
            return default_config
        except yaml.YAMLError as e:
            logger.error(
                f"Error parsing YAML config file {config_file}: {e}. "
                "Using default mutation settings."
            )
            return default_config
        except Exception as e:
            logger.error(
                f"Unexpected error loading config file {config_file}: {e}. "
                "Using default mutation settings."
            )
            return default_config
    
    def collect_betes_metrics(
        self,
        quality_cfg: QualityConfig,
        mutation_config: Dict[str, Any],
        test_targets: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Collect raw metrics for bE-TES calculation.
        
        Args:
            quality_cfg: Quality configuration
            mutation_config: Mutation sensor configuration
            test_targets: Optional list of specific test targets
            
        Returns:
            Dictionary containing raw metrics
        """
        logger.info(
            f"Collecting raw metrics for {quality_cfg.mode} using sensors..."
        )
        
        # Collect metrics from sensors
        raw_ms_percentage, _, _ = mutation_sensor.get_mutation_score_data(
            config=mutation_config,
            project_path=self.project_path,
            test_targets=test_targets
        )
        
        raw_emt_gain = mutation_sensor.get_emt_gain(
            current_mutation_score=raw_ms_percentage,
            project_path=self.project_path,
            config={}
        )
        
        raw_assertion_iq = assertion_iq_sensor.get_mean_assertion_iq(
            test_root_path=quality_cfg.test_root_path,
            config={}
        )
        
        raw_behaviour_coverage = behaviour_coverage_sensor.get_behaviour_coverage_ratio(
            coverage_file_path=quality_cfg.coverage_file_path,
            critical_behaviors_manifest_path=quality_cfg.critical_behaviors_manifest_path,
            config={}
        )
        
        raw_median_test_time_ms = speed_sensor.get_median_test_time_ms(
            reportlog_path=None,  # Could be passed from args if needed
            config={}
        )
        
        raw_flakiness_rate = flakiness_sensor.get_suite_flakiness_rate(
            project_path=self.project_path,
            ci_platform=quality_cfg.ci_platform,
            project_identifier=self.project_path,
            config={}
        )
        
        return {
            "raw_mutation_score": raw_ms_percentage,
            "raw_emt_gain": raw_emt_gain,
            "raw_assertion_iq": raw_assertion_iq,
            "raw_behaviour_coverage": raw_behaviour_coverage,
            "raw_median_test_time_ms": raw_median_test_time_ms,
            "raw_flakiness_rate": raw_flakiness_rate
        }
    
    def prepare_etes_v2_data(self, results: Dict[str, Any]) -> tuple:
        """
        Prepare data structures for E-TES v2.0 calculation.
        
        Args:
            results: Analysis results from ProjectAnalyzer
            
        Returns:
            Tuple of (test_suite_data, codebase_data)
        """
        logger.info("Preparing data for E-TES v2.0...")
        
        tes_components = results.get("tes_components", {})
        etes_v2_data = results.get("etes_components", {})
        
        test_suite_data = {
            'mutation_score': tes_components.get('mutation_score', 0.0),
            'avg_test_execution_time_ms': tes_components.get('avg_test_execution_time_ms', 100.0),
            'assertions': etes_v2_data.get('assertions_INTERNAL', []),
            'covered_behaviors': etes_v2_data.get('covered_behaviors_INTERNAL', []),
            'determinism_score': etes_v2_data.get('determinism_score_INTERNAL', 1.0),
            'stability_score': etes_v2_data.get('stability_score_INTERNAL', 1.0),
            'readability_score': etes_v2_data.get('readability_score_INTERNAL', 1.0),
            'independence_score': etes_v2_data.get('independence_score_INTERNAL', 1.0),
        }
        
        codebase_data = {
            'all_behaviors': [],
            'behavior_criticality': {},
            'complexity_metrics': results.get("metrics", {})
        }
        
        return test_suite_data, codebase_data
    
    def calculate_quality_score(
        self,
        quality_cfg: QualityConfig,
        raw_metrics_betes: Optional[Dict[str, Any]],
        test_suite_data: Optional[Dict[str, Any]],
        codebase_data: Optional[Dict[str, Any]],
        previous_score: Optional[float],
        project_language: str = "python"
    ) -> tuple:
        """
        Calculate quality score using the unified calculator.
        
        Args:
            quality_cfg: Quality configuration
            raw_metrics_betes: Raw metrics for bE-TES (if applicable)
            test_suite_data: Test suite data for E-TES v2 (if applicable)
            codebase_data: Codebase data for E-TES v2 (if applicable)
            previous_score: Previous score for evolution gain (if applicable)
            project_language: Project language for OSQI
            
        Returns:
            Tuple of (quality_score, quality_components)
        """
        return calculate_quality_score(
            config=quality_cfg,
            raw_metrics_betes=raw_metrics_betes,
            test_suite_data=test_suite_data,
            codebase_data=codebase_data,
            previous_score=previous_score,
            project_path=self.project_path if quality_cfg.mode == "osqi_v1" else None,
            project_language=project_language
        )
    
    def apply_risk_classification(
        self,
        quality_cfg: QualityConfig,
        quality_score: float,
        results: Dict[str, Any]
    ) -> None:
        """
        Apply risk classification if configured.
        
        Args:
            quality_cfg: Quality configuration
            quality_score: Calculated quality score
            results: Results dictionary to update
        """
        if quality_cfg.mode not in ["betes_v3.1", "osqi_v1"] or not quality_cfg.risk_class:
            return
        
        classification_metric_name = (
            "OSQI" if quality_cfg.mode == "osqi_v1" else "bE-TES"
        )
        
        risk_definitions = self._load_risk_definitions()
        
        if not risk_definitions:
            results["quality_analysis"]["classification_error"] = (
                "Risk definitions were not loaded for classification."
            )
            return
        
        try:
            classification = classify_betes(
                score=quality_score,
                risk_class_name=quality_cfg.risk_class,
                risk_definitions=risk_definitions,
                metric_name=classification_metric_name
            )
            results["quality_analysis"]["classification"] = classification
        except Exception as e:
            logger.warning(f"Error during risk classification: {e}")
            results["quality_analysis"]["classification_error"] = str(e)
    
    def _load_risk_definitions(self) -> Dict[str, Any]:
        """
        Load risk class definitions from YAML file.
        
        Returns:
            Dictionary of risk definitions, or empty dict if not found
        """
        risk_file_path = self.config_dir / "risk_classes.yml"
        
        if not risk_file_path.exists():
            logger.warning(
                f"risk_classes.yml not found at {risk_file_path}. "
                "Risk classification will be skipped."
            )
            return {}
        
        try:
            with open(risk_file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            logger.error(
                "PyYAML is not installed. Install it with `pip install PyYAML` "
                "to use risk classification."
            )
            return {}
        except Exception as e:
            logger.error(f"Error loading risk_classes.yml: {e}")
            return {}

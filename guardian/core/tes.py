# Enhanced Test Effectiveness Score (TES) calculation logic with E-TES v2.0 integration.

import logging
from typing import Dict, Any, Optional, Tuple, Union
from .etes import ETESCalculator, QualityConfig, ETESComponents, ETESGrade, BETESSettingsV31 # Added BETESSettingsV31
from .betes import BETESCalculator, BETESComponents as BETESComponentsV3, classify_betes
from .osqi import OSQICalculator, OSQIResult, OSQIRawPillarsInput, classify_osqi

# Import sensor functions for OSQI
from guardian.sensors import chs as chs_sensor
from guardian.analysis import security as security_sensor # security.py contains get_raw_weighted_vulnerability_density
from guardian.sensors import arch as arch_sensor

# Import sensor functions for bE-TES (already used by CLI, but good to have if this module orchestrates)
from guardian.sensors import mutation as mutation_sensor
from guardian.sensors import assertion_iq as assertion_iq_sensor
from guardian.sensors import behaviour_coverage as behaviour_coverage_sensor
from guardian.sensors import speed as speed_sensor
from guardian.sensors import flakiness as flakiness_sensor


logger = logging.getLogger(__name__)


# Removed placeholder _get_raw_metrics_for_betes as CLI now calls sensors.

def calculate_quality_score(
    config: QualityConfig,
    # For bE-TES modes, raw_metrics_betes is expected to be populated by the CLI
    raw_metrics_betes: Optional[Dict[str, Any]] = None,
    # For E-TES v2 mode
    test_suite_data: Optional[Dict[str, Any]] = None,
    codebase_data: Optional[Dict[str, Any]] = None,
    previous_score: Optional[float] = None,
    # For OSQI mode, project_path and project_language are needed for sensors
    project_path: Optional[str] = None,
    project_language: Optional[str] = "python" # Default language for CHS
) -> Tuple[float, Union[ETESComponents, BETESComponentsV3, OSQIResult, Dict]]: # Added OSQIResult and Dict for error
    """
    Calculates the quality score based on the mode specified in the config.
    Dispatches to E-TES v2.0, bE-TES v3.0/v3.1, or OSQI v1.0 calculator.

    Args:
        config: QualityConfig object with mode and specific calculator settings.
        raw_metrics_betes: Dictionary of raw metrics if mode is "betes_v3" or "betes_v3.1".
        test_suite_data: Comprehensive test suite metrics if mode is "etes_v2".
        codebase_data: Codebase analysis data if mode is "etes_v2".
        previous_score: Previous quality score, primarily for E-TES v2's evolution gain.
        project_path: Filesystem path to the project, required for OSQI sensor data collection.
        project_language: Primary language of the project, for OSQI's CHS normalization.

    Returns:
        A tuple containing the calculated score (float) and the components object
        (ETESComponents, BETESComponentsV3, OSQIResult, or an error dict).
    """
    if config.mode == "betes_v3" or config.mode == "betes_v3.1":
        logger.info(f"Calculating bE-TES {config.mode}...")
        if not raw_metrics_betes:
            logger.error(f"Raw metrics not provided for bE-TES {config.mode} calculation.")
            return 0.0, BETESComponentsV3()

        # For betes_v3.1, pass the specific settings
        betes_settings_v3_1 = config.betes_v3_1_settings if config.mode == "betes_v3.1" else None
        
        calculator = BETESCalculator(
            weights=config.betes_weights,
            settings_v3_1=betes_settings_v3_1
        )
        components_v3 = calculator.calculate(
            raw_mutation_score=raw_metrics_betes.get("raw_mutation_score", 0.0),
            raw_emt_gain=raw_metrics_betes.get("raw_emt_gain", 0.0),
            raw_assertion_iq=raw_metrics_betes.get("raw_assertion_iq", 1.0),
            raw_behaviour_coverage=raw_metrics_betes.get("raw_behaviour_coverage", 0.0),
            raw_median_test_time_ms=raw_metrics_betes.get("raw_median_test_time_ms", 1000.0),
            raw_flakiness_rate=raw_metrics_betes.get("raw_flakiness_rate", 0.0)
        )
        logger.info(f"bE-TES {config.mode} calculated: {components_v3.betes_score:.3f}")
        
        # Perform classification if risk_class is set
        classification_result = None
        if config.risk_class:
            # This part is tricky as risk_definitions are loaded in CLI.
            # For now, assume CLI handles classification display.
            # If this function needs to return it, risk_definitions must be passed in.
            # The CLI currently does this after calling this function.
            pass # Classification handled by CLI based on the returned score and components.

        return components_v3.betes_score, components_v3
        
    elif config.mode == "osqi_v1":
        logger.info("Calculating OSQI v1.0...")
        if not project_path:
            logger.error("Project path not provided for OSQI v1.0 calculation.")
            return 0.0, OSQIResult(insights=["Error: Project path is required for OSQI."])

        # 1. Calculate bE-TES v3.1 score first (as it's a pillar)
        # The CLI should have already collected raw_metrics_betes if --run-quality was used.
        # If this function is called for OSQI, it implies bE-TES sensors ran.
        if not raw_metrics_betes:
            logger.error("Raw metrics for bE-TES (a pillar of OSQI) not provided.")
            # Attempt to collect them now if not provided (this duplicates CLI logic, ideally CLI provides them)
            # For simplicity here, we'll assume CLI provides them or we error out.
            # If we were to collect them here:
            # logger.info("Attempting to collect bE-TES metrics for OSQI...")
            # raw_metrics_betes = _collect_betes_metrics_for_osqi(config, project_path) # New helper needed
            return 0.0, OSQIResult(insights=["Error: bE-TES metrics (OSQI pillar) not available."])

        betes_calculator_for_osqi = BETESCalculator(
            weights=config.betes_weights,
            settings_v3_1=config.betes_v3_1_settings # Use v3.1 settings for the bE-TES pillar
        )
        betes_components_for_osqi = betes_calculator_for_osqi.calculate(
            raw_mutation_score=raw_metrics_betes.get("raw_mutation_score", 0.0),
            raw_emt_gain=raw_metrics_betes.get("raw_emt_gain", 0.0),
            raw_assertion_iq=raw_metrics_betes.get("raw_assertion_iq", 1.0),
            raw_behaviour_coverage=raw_metrics_betes.get("raw_behaviour_coverage", 0.0),
            raw_median_test_time_ms=raw_metrics_betes.get("raw_median_test_time_ms", 1000.0),
            raw_flakiness_rate=raw_metrics_betes.get("raw_flakiness_rate", 0.0)
        )
        current_betes_score = betes_components_for_osqi.betes_score
        logger.info(f"OSQI Pillar: bE-TES v3.1 score = {current_betes_score:.3f}")

        # 2. Collect other raw pillar inputs using sensors
        # Sensor configs can be passed via QualityConfig if needed, or use defaults.
        # For now, passing empty dicts for sensor configs.
        raw_chs = chs_sensor.get_raw_chs_sub_metrics(project_path, project_language, config={})
        raw_vuln_density = security_sensor.get_raw_weighted_vulnerability_density(project_path, config={})
        raw_arch_violation = arch_sensor.get_raw_architectural_violation_score(project_path, config={})

        osqi_inputs = OSQIRawPillarsInput(
            betes_score=current_betes_score,
            raw_code_health_sub_metrics=raw_chs,
            raw_weighted_vulnerability_density=raw_vuln_density,
            raw_architectural_violation_score=raw_arch_violation
        )

        # 3. Calculate OSQI
        osqi_calculator = OSQICalculator(
            osqi_weights=config.osqi_weights, # From QualityConfig
            chs_thresholds_path=config.chs_thresholds_path if hasattr(config, 'chs_thresholds_path') else "guardian_ai_tool/config/chs_thresholds.yml"
        )
        osqi_result_obj = osqi_calculator.calculate(osqi_inputs, project_language)
        
        logger.info(f"OSQI v1.0 calculated: {osqi_result_obj.osqi_score:.3f}")
        
        # Classification for OSQI (if risk_class is set)
        # The CLI will handle loading risk_definitions and calling classify_osqi
        # This function returns the OSQIResult, CLI uses its .osqi_score for classification.
        # If this function were to return classification, it would need risk_definitions.
        # For consistency, let CLI do the final classification step.
        # However, we can store the classification in the OSQIResult if needed by adding a field.
        # For now, the CLI will call classify_osqi using the score from this result.

        return osqi_result_obj.osqi_score, osqi_result_obj

    elif config.mode == "etes_v2":
        if test_suite_data is None or codebase_data is None:
            logger.error("Test suite or codebase data not provided for E-TES v2.0 calculation.")
            return 0.0, ETESComponents()
        logger.info("Calculating E-TES v2.0...")
        # ETESCalculator expects the full QualityConfig, it will use its relevant parts.
        calculator = ETESCalculator(config=config)
        score_v2, components_v2 = calculator.calculate_etes(
            test_suite_data, codebase_data, previous_score
        )
        logger.info(f"E-TES v2.0 calculated: {score_v2:.3f}")
        return score_v2, components_v2
    else:
        logger.error(f"Unknown quality score mode: {config.mode}")
        # Return a default/error state. ETESComponents is used as a generic fallback here.
        return 0.0, ETESComponents()


def calculate_etes_v2(test_suite_data: Dict[str, Any],
                     codebase_data: Dict[str, Any],
                     config: Optional[QualityConfig] = None, # Changed ETESConfig to QualityConfig
                     previous_score: Optional[float] = None) -> Tuple[float, ETESComponents]:
    """
    Calculate E-TES v2.0 score with evolutionary capabilities

    Args:
        test_suite_data: Comprehensive test suite metrics
        codebase_data: Codebase analysis data
        config: Optional E-TES configuration
        previous_score: Previous E-TES score for evolution tracking

    Returns:
        Tuple of (etes_score, detailed_components)
    """
    try:
        calculator = ETESCalculator(config)
        return calculator.calculate_etes(test_suite_data, codebase_data, previous_score)
    except Exception as e:
        logger.error(f"Error calculating E-TES v2.0: {e}")
        return 0.0, ETESComponents()


def get_etes_grade(etes_score: float) -> str: # This function can be used for bE-TES too.
    """Get letter grade for E-TES score (also applicable to bE-TES)."""
    grade = ETESGrade.F # Default to F
    if etes_score >= 0.9:
        grade = ETESGrade.A_PLUS
    elif etes_score >= 0.8:
        grade = ETESGrade.A
    elif etes_score >= 0.7:
        grade = ETESGrade.B
    elif etes_score >= 0.6:
        grade = ETESGrade.C
    return grade.value


def compare_tes_vs_etes(legacy_tes: float, etes_score: float, # This function might need an update if comparing bE-TES
                       components: Union[ETESComponents, BETESComponentsV3]) -> Dict[str, Any]:
    """
    Compare legacy TES with E-TES v2.0 for migration insights

    Args:
        legacy_tes: Original TES score
        etes_score: E-TES v2.0 score
        components: E-TES components

    Returns:
        Comparison analysis and migration recommendations
    """
    comparison = {
        'legacy_tes': legacy_tes,
        'etes_v2': etes_score,
        'improvement': etes_score - legacy_tes,
        'legacy_grade': get_tes_grade(legacy_tes),
        'etes_grade': get_etes_grade(etes_score),
        'components': {
            'mutation_score': components.mutation_score,
            'evolution_gain': components.evolution_gain,
            'assertion_iq': components.assertion_iq,
            'behavior_coverage': components.behavior_coverage,
            'speed_factor': components.speed_factor,
            'quality_factor': components.quality_factor
        },
        'insights': components.insights,
        'recommendations': []
    }

    # Generate migration recommendations
    if etes_score > legacy_tes + 0.1:
        comparison['recommendations'].append("E-TES v2.0 shows significant improvement - recommend migration")
    elif etes_score < legacy_tes - 0.1:
        comparison['recommendations'].append("E-TES v2.0 shows lower score - investigate component issues")
    else:
        comparison['recommendations'].append("Scores are comparable - E-TES v2.0 provides additional insights")

    # Component-specific recommendations
    if components.evolution_gain > 1.2:
        comparison['recommendations'].append("Strong evolution gain detected - test suite is improving")

    if components.assertion_iq < 0.7:
        comparison['recommendations'].append("Low assertion IQ - focus on intelligent assertion patterns")

    if components.quality_factor < 0.8:
        comparison['recommendations'].append("Quality factor needs improvement - focus on test reliability")

    return comparison
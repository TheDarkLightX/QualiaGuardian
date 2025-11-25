"""
Shared classification utilities for quality metrics.

Provides reusable classification logic for bE-TES, OSQI, and other metrics.
"""

from typing import Dict, Any, Optional


def classify_metric_score(
    score: float,
    risk_class_name: Optional[str],
    risk_definitions: Dict[str, Dict[str, Any]],
    metric_name: str = "Quality Score"
) -> Dict[str, Any]:
    """
    Evaluates a given score against a defined risk class and its threshold.

    Args:
        score: The calculated score (0.0 to 1.0).
        risk_class_name: The name of the risk class to evaluate against (e.g., "standard_saas").
                         If None, no specific classification is performed, only the score is returned.
        risk_definitions: A dictionary loaded from risk_classes.yml, where keys are
                          risk class names and values are their definitions (e.g., {"min_score": 0.75}).
        metric_name: The name of the metric being classified (e.g., "bE-TES", "OSQI").

    Returns:
        A dictionary containing the score, and if a risk_class_name is provided,
        it includes the risk class, its threshold, a pass/fail verdict, and a message.
        Example:
            {
                "score": 0.82,
                "risk_class": "standard_saas",
                "threshold": 0.75,
                "verdict": "PASS",
                "message": "bE-TES score meets the threshold for Standard SaaS."
            }
        If risk_class_name is None:
            {"score": 0.82}
    """
    result: Dict[str, Any] = {"score": round(score, 3)}

    if not risk_class_name:
        return result

    if not risk_definitions:
        return _create_error_result(result, risk_class_name, "Risk definitions not provided.")

    risk_class_info = risk_definitions.get(risk_class_name)
    if not risk_class_info:
        return _create_error_result(
            result, risk_class_name,
            f"Risk class '{risk_class_name}' not found in definitions."
        )

    min_score_threshold = risk_class_info.get("min_score")
    if min_score_threshold is None:
        return _create_error_result(
            result, risk_class_name,
            f"No 'min_score' threshold defined for risk class '{risk_class_name}'."
        )

    try:
        min_score_threshold = float(min_score_threshold)
    except ValueError:
        return _create_error_result(
            result, risk_class_name,
            f"'min_score' for risk class '{risk_class_name}' is not a valid number."
        )

    result["risk_class"] = risk_class_name
    result["threshold"] = round(min_score_threshold, 3)

    if score >= min_score_threshold:
        result["verdict"] = "PASS"
        result["message"] = (
            f"{metric_name} score {result['score']:.3f} meets or exceeds "
            f"the threshold of {result['threshold']:.3f} for risk class '{risk_class_name}'."
        )
    else:
        result["verdict"] = "FAIL"
        result["message"] = (
            f"{metric_name} score {result['score']:.3f} is below "
            f"the threshold of {result['threshold']:.3f} for risk class '{risk_class_name}'."
        )

    return result


def _create_error_result(
    base_result: Dict[str, Any],
    risk_class_name: str,
    error_message: str
) -> Dict[str, Any]:
    """Create an error result dictionary."""
    result = base_result.copy()
    result["error"] = error_message
    result["risk_class"] = risk_class_name
    result["verdict"] = "UNKNOWN"
    return result

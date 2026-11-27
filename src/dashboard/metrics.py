"""Pure dashboard threat scoring helpers.

Threat level is based on alert events, never routine object detections.
Values represent a display heuristic, not a validated safety assessment.
"""


def compute_threat_level(critical: int, high: int, medium: int, low: int = 0) -> tuple:
    """Return (label, color-class, gauge-percent) for recent alert counts."""
    if critical > 0:
        return ("CRITICAL", "critical", 100)
    if high > 0:
        return ("HIGH", "high", 75)
    if medium > 0:
        return ("MODERATE", "medium", 50)
    if low > 0:
        return ("LOW", "low", 20)
    return ("CLEAR", "low", 0)

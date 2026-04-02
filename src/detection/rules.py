"""Detection rules engine."""
from dataclasses import dataclass
from typing import List, Dict, Any, Callable
from enum import Enum
import time


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DetectionRule:
    """Rule for triggering alerts based on detections."""
    name: str
    conditions: List[Dict[str, Any]]
    severity: Severity
    cooldown_sec: int = 60
    last_triggered: float = 0.0

    def can_trigger(self) -> bool:
        """Check if rule is off cooldown."""
        return time.time() - self.last_triggered >= self.cooldown_sec

    def trigger(self):
        """Mark rule as triggered."""
        self.last_triggered = time.time()


class RuleEngine:
    """
    Engine for evaluating detection rules.
    """

    def __init__(self):
        self.rules: List[DetectionRule] = []
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup predefined detection rules."""

        # Weapon detected - CRITICAL
        self.rules.append(DetectionRule(
            name="WEAPON_DETECTED",
            conditions=[{"class": ["knife", "gun"]}],
            severity=Severity.CRITICAL,
            cooldown_sec=30
        ))

        # Fight detected - HIGH
        self.rules.append(DetectionRule(
            name="FIGHT_DETECTED",
            conditions=[{"behavior": ["fighting", "raised_arms"]}],
            severity=Severity.HIGH,
            cooldown_sec=60
        ))

        # Loitering - MEDIUM
        self.rules.append(DetectionRule(
            name="LOITERING",
            conditions=[{"type": "loitering", "duration": 60}],
            severity=Severity.MEDIUM,
            cooldown_sec=120
        ))

        # Crowd formation - MEDIUM
        self.rules.append(DetectionRule(
            name="CROWD_FORMATION",
            conditions=[{"type": "crowd", "min_people": 10}],
            severity=Severity.MEDIUM,
            cooldown_sec=60
        ))

    def evaluate(self, detections: List[Dict],
                 behaviors: List[Dict],
                 tracks: List[Dict]) -> List[DetectionRule]:
        """
        Evaluate all rules against current state.

        Returns:
            List of triggered rules
        """
        triggered = []

        for rule in self.rules:
            if not rule.can_trigger():
                continue

            if self._check_rule(rule, detections, behaviors, tracks):
                rule.trigger()
                triggered.append(rule)

        return triggered

    def _check_rule(self, rule: DetectionRule,
                    detections: List[Dict],
                    behaviors: List[Dict],
                    tracks: List[Dict]) -> bool:
        """Check if a rule's conditions are met."""
        for condition in rule.conditions:
            if "class" in condition:
                # Check for specific object classes
                classes = condition["class"]
                for det in detections:
                    if det.get("class") in classes:
                        return True

            if "behavior" in condition:
                # Check for specific behaviors
                behavior_names = condition["behavior"]
                for beh in behaviors:
                    if beh.get("behavior") in behavior_names:
                        return True

        return False

    def add_rule(self, rule: DetectionRule):
        """Add a custom rule."""
        self.rules.append(rule)

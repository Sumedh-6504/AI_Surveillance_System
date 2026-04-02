"""Alert engine for managing and dispatching alerts."""
import time
from typing import List, Dict, Any, Optional
from collections import deque
from .channels import EmailChannel, SMSChannel, WebhookChannel
from ..detection.rules import Severity, DetectionRule


class Alert:
    """Represents a single alert."""

    def __init__(self, rule: DetectionRule, detections: List[Dict],
                 frame_id: int = 0, camera_id: str = "cam1"):
        self.rule = rule
        self.detections = detections
        self.frame_id = frame_id
        self.camera_id = camera_id
        self.timestamp = time.time()
        self.dispatched = False
        self.id = f"{rule.name}_{int(self.timestamp)}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "rule_name": self.rule.name,
            "severity": self.rule.severity.value,
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "detection_count": len(self.detections),
            "detections": self.detections
        }


class AlertEngine:
    """
    Central engine for managing and dispatching alerts.
    Handles deduplication, rate limiting, and multi-channel delivery.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alerts: deque = deque(maxlen=1000)
        self.pending_alerts: List[Alert] = []
        self.dispatched_ids: set = set()

        # Initialize channels
        self.channels = []
        self._setup_channels(config)

        # Rate limiting
        self.alert_counts: Dict[str, List[float]] = {}
        self.rate_limits = {
            Severity.LOW: {"count": 10, "window_sec": 60},
            Severity.MEDIUM: {"count": 20, "window_sec": 60},
            Severity.HIGH: {"count": 30, "window_sec": 60},
            Severity.CRITICAL: {"count": 100, "window_sec": 60},  # No limit for critical
        }

        # Statistics
        self.stats = {
            "total_alerts": 0,
            "dispatched": 0,
            "suppressed": 0,
            "by_severity": {s.value: 0 for s in Severity}
        }

    def _setup_channels(self, config: Dict[str, Any]):
        """Initialize notification channels from config."""
        # Email channel
        email_config = config.get("email", {})
        if email_config.get("enabled", False):
            self.channels.append(EmailChannel(email_config))

        # SMS channel
        sms_config = config.get("sms", {})
        if sms_config.get("enabled", False):
            self.channels.append(SMSChannel(sms_config))

        # Webhook channel (for custom integrations)
        webhook_config = config.get("webhook", {})
        if webhook_config.get("enabled", False):
            self.channels.append(WebhookChannel(webhook_config))

    def add_alert(self, rule: DetectionRule, detections: List[Dict],
                  frame_id: int = 0, camera_id: str = "cam1") -> Optional[Alert]:
        """
        Add a new alert to the queue.

        Args:
            rule: Triggered detection rule
            detections: Related detections
            frame_id: Current frame number
            camera_id: Camera identifier

        Returns:
            Alert object if created, None if suppressed
        """
        alert = Alert(rule, detections, frame_id, camera_id)

        # Check rate limiting
        if not self._check_rate_limit(alert):
            self.stats["suppressed"] += 1
            print(f"Alert suppressed (rate limit): {rule.name}")
            return None

        self.pending_alerts.append(alert)
        self.alerts.append(alert)
        self.stats["total_alerts"] += 1
        self.stats["by_severity"][rule.severity.value] += 1

        print(f"Alert created: {rule.name} ({rule.severity.value})")
        return alert

    def _check_rate_limit(self, alert: Alert) -> bool:
        """Check if alert should be rate limited."""
        severity = alert.rule.severity
        limits = self.rate_limits.get(severity, {})

        max_count = limits.get("count", 10)
        window = limits.get("window_sec", 60)

        # Critical alerts bypass rate limiting
        if severity == Severity.CRITICAL:
            return True

        now = time.time()
        key = alert.rule.name

        if key not in self.alert_counts:
            self.alert_counts[key] = []

        # Clean old entries
        self.alert_counts[key] = [
            t for t in self.alert_counts[key]
            if now - t < window
        ]

        # Check limit
        if len(self.alert_counts[key]) >= max_count:
            return False

        # Add current alert
        self.alert_counts[key].append(now)
        return True

    def dispatch(self) -> int:
        """
        Dispatch all pending alerts through configured channels.

        Returns:
            Number of alerts dispatched
        """
        dispatched = 0

        for alert in self.pending_alerts:
            if alert.dispatched or alert.id in self.dispatched_ids:
                continue

            success = self._dispatch_alert(alert)
            if success:
                alert.dispatched = True
                self.dispatched_ids.add(alert.id)
                dispatched += 1
                self.stats["dispatched"] += 1

        # Clear dispatched alerts
        self.pending_alerts = [
            a for a in self.pending_alerts if not a.dispatched
        ]

        return dispatched

    def _dispatch_alert(self, alert: Alert) -> bool:
        """Dispatch a single alert through all channels."""
        subject = self._format_subject(alert)
        body = self._format_body(alert)

        all_success = True

        for channel in self.channels:
            try:
                if isinstance(channel, (EmailChannel, SMSChannel)):
                    success = channel.send(subject, body)
                elif isinstance(channel, WebhookChannel):
                    success = channel.send(subject, body, alert.to_dict())
                else:
                    success = False

                if not success:
                    all_success = False

            except Exception as e:
                print(f"Channel {channel.__class__.__name__} failed: {e}")
                all_success = False

        return all_success

    def _format_subject(self, alert: Alert) -> str:
        """Format alert subject."""
        severity = alert.rule.severity.value.upper()
        return f"[{severity}] {alert.rule.name} - Camera {alert.camera_id}"

    def _format_body(self, alert: Alert) -> str:
        """Format alert body."""
        lines = [
            f"Alert: {alert.rule.name}",
            f"Severity: {alert.rule.severity.value.upper()}",
            f"Camera: {alert.camera_id}",
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}",
            f"Detections: {len(alert.detections)}",
            "",
            "Details:"
        ]

        for det in alert.detections[:5]:  # Limit details
            cls = det.get("class", "unknown")
            conf = det.get("confidence", 0)
            lines.append(f"  - {cls}: {conf:.2f}")

        if len(alert.detections) > 5:
            lines.append(f"  ... and {len(alert.detections) - 5} more")

        return "\n".join(lines)

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        recent = list(self.alerts)[-limit:]
        return [a.to_dict() for a in reversed(recent)]

    def get_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        return dict(self.stats)

    def clear_old_alerts(self, max_age_sec: float = 3600):
        """Clear alerts older than max_age."""
        cutoff = time.time() - max_age_sec
        self.alerts = deque(
            [a for a in self.alerts if a.timestamp > cutoff],
            maxlen=1000
        )
        self.dispatched_ids.clear()

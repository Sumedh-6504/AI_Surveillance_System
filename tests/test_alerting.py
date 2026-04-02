"""Unit tests for alerting modules."""
import pytest
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alerting.engine import AlertEngine, Alert
from src.alerting.channels import EmailChannel, SMSChannel, WebhookChannel
from src.detection.rules import DetectionRule, Severity


class TestAlert:
    """Tests for Alert class."""

    def test_alert_creation(self):
        """Test alert creation."""
        rule = DetectionRule(
            name="TEST_RULE",
            conditions=[{"class": ["person"]}],
            severity=Severity.HIGH
        )

        detections = [{"class": "person", "confidence": 0.9}]
        alert = Alert(rule, detections, frame_id=100, camera_id="cam1")

        assert alert.rule.name == "TEST_RULE"
        assert alert.frame_id == 100
        assert alert.camera_id == "cam1"
        assert alert.dispatched is False
        assert "TEST_RULE" in alert.id

    def test_alert_to_dict(self):
        """Test alert serialization."""
        rule = DetectionRule(
            name="TEST_RULE",
            conditions=[{"class": ["person"]}],
            severity=Severity.MEDIUM
        )

        alert = Alert(rule, [{"class": "person"}])
        data = alert.to_dict()

        assert data["rule_name"] == "TEST_RULE"
        assert data["severity"] == "medium"
        assert data["detection_count"] == 1
        assert "timestamp" in data


class TestAlertEngine:
    """Tests for AlertEngine."""

    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        config = {
            "email": {"enabled": False},
            "sms": {"enabled": False}
        }
        engine = AlertEngine(config)

        assert len(engine.channels) == 0
        assert engine.stats["total_alerts"] == 0

    def test_add_alert(self):
        """Test adding alerts."""
        config = {
            "email": {"enabled": False},
            "sms": {"enabled": False}
        }
        engine = AlertEngine(config)

        rule = DetectionRule(
            name="WEAPON_TEST",
            conditions=[{"class": ["gun"]}],
            severity=Severity.CRITICAL
        )

        alert = engine.add_alert(rule, [{"class": "gun"}])

        assert alert is not None
        assert engine.stats["total_alerts"] == 1
        assert len(engine.pending_alerts) == 1

    def test_rate_limiting(self):
        """Test rate limiting suppresses excessive alerts."""
        config = {
            "email": {"enabled": False},
            "sms": {"enabled": False}
        }
        engine = AlertEngine(config)

        rule = DetectionRule(
            name="LOW_PRIORITY",
            conditions=[{"class": ["person"]}],
            severity=Severity.LOW,
            cooldown_sec=1
        )

        # Add many alerts quickly
        alerts_added = 0
        for _ in range(15):
            alert = engine.add_alert(rule, [{"class": "person"}])
            if alert:
                alerts_added += 1

        # Should be rate limited (max 10 per 60 sec for LOW)
        assert alerts_added <= 10
        assert engine.stats["suppressed"] > 0

    def test_critical_bypasses_rate_limit(self):
        """Test critical alerts bypass rate limiting."""
        config = {
            "email": {"enabled": False},
            "sms": {"enabled": False}
        }
        engine = AlertEngine(config)

        rule = DetectionRule(
            name="CRITICAL_TEST",
            conditions=[{"class": ["gun"]}],
            severity=Severity.CRITICAL
        )

        # Add many critical alerts
        for _ in range(50):
            engine.add_alert(rule, [{"class": "gun"}])

        # All should be added (no rate limiting for critical)
        assert engine.stats["total_alerts"] == 50
        assert engine.stats["suppressed"] == 0

    def test_dispatch(self):
        """Test alert dispatch."""
        config = {
            "email": {"enabled": False},
            "sms": {"enabled": False}
        }
        engine = AlertEngine(config)

        rule = DetectionRule(
            name="TEST_DISPATCH",
            conditions=[{"class": ["person"]}],
            severity=Severity.MEDIUM
        )

        engine.add_alert(rule, [{"class": "person"}])
        dispatched = engine.dispatch()

        # No channels configured, but dispatch should complete
        assert dispatched == 0
        assert len(engine.pending_alerts) == 0  # Cleared after dispatch

    def test_get_recent_alerts(self):
        """Test retrieving recent alerts."""
        config = {
            "email": {"enabled": False},
            "sms": {"enabled": False}
        }
        engine = AlertEngine(config)

        rule = DetectionRule(
            name="TEST_RECENT",
            conditions=[{"class": ["person"]}],
            severity=Severity.LOW
        )

        # Add 5 alerts
        for i in range(5):
            engine.add_alert(rule, [{"class": "person"}])

        recent = engine.get_recent_alerts(limit=3)
        assert len(recent) == 3

    def test_get_stats(self):
        """Test statistics retrieval."""
        config = {
            "email": {"enabled": False},
            "sms": {"enabled": False}
        }
        engine = AlertEngine(config)

        rule = DetectionRule(
            name="STATS_TEST",
            conditions=[{"class": ["person"]}],
            severity=Severity.MEDIUM
        )

        engine.add_alert(rule, [{"class": "person"}])
        stats = engine.get_stats()

        assert stats["total_alerts"] == 1
        assert stats["by_severity"]["medium"] == 1

    def test_clear_old_alerts(self):
        """Test clearing old alerts."""
        config = {
            "email": {"enabled": False},
            "sms": {"enabled": False}
        }
        engine = AlertEngine(config)

        rule = DetectionRule(
            name="OLD_TEST",
            conditions=[{"class": ["person"]}],
            severity=Severity.LOW
        )

        engine.add_alert(rule, [{"class": "person"}])
        assert len(engine.alerts) == 1

        # Clear with 0 age (clears everything)
        engine.clear_old_alerts(max_age_sec=0)
        assert len(engine.alerts) == 0


class TestEmailChannel:
    """Tests for EmailChannel."""

    def test_channel_disabled(self):
        """Test disabled channel doesn't send."""
        config = {"enabled": False}
        channel = EmailChannel(config)

        result = channel.send("Test", "Body")
        assert result is False

    def test_channel_no_recipients(self):
        """Test channel fails without recipients."""
        config = {"enabled": True, "recipients": []}
        channel = EmailChannel(config)

        result = channel.send("Test", "Body")
        assert result is False

    def test_channel_config_defaults(self):
        """Test channel uses default config values."""
        config = {"enabled": True}
        channel = EmailChannel(config)

        assert channel.smtp_server == "smtp.gmail.com"
        assert channel.smtp_port == 587
        assert channel.sender == "alerts@example.com"


class TestSMSChannel:
    """Tests for SMSChannel."""

    def test_channel_disabled(self):
        """Test disabled SMS channel."""
        config = {"enabled": False}
        channel = SMSChannel(config)

        result = channel.send("Test", "Body")
        assert result is False

    def test_channel_no_numbers(self):
        """Test channel fails without phone numbers."""
        config = {"enabled": True, "to_numbers": []}
        channel = SMSChannel(config)

        result = channel.send("Test", "Body")
        assert result is False

    def test_channel_config_defaults(self):
        """Test SMS channel default config."""
        config = {"enabled": True}
        channel = SMSChannel(config)

        assert channel.account_sid == ""
        assert channel.auth_token == ""
        assert channel.from_number == ""


class TestWebhookChannel:
    """Tests for WebhookChannel."""

    def test_channel_disabled(self):
        """Test disabled webhook channel."""
        config = {"enabled": False}
        channel = WebhookChannel(config)

        result = channel.send("Test", "Body")
        assert result is False

    def test_channel_no_url(self):
        """Test channel fails without URL."""
        config = {"enabled": True, "url": ""}
        channel = WebhookChannel(config)

        result = channel.send("Test", "Body")
        assert result is False

    def test_channel_default_headers(self):
        """Test webhook uses default headers."""
        config = {"enabled": True, "url": "http://test.com"}
        channel = WebhookChannel(config)

        assert channel.headers == {"Content-Type": "application/json"}
        assert channel.timeout == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

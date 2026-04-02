"""Alert notification channels."""
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any
from pathlib import Path


class EmailChannel:
    """Email notification channel using SMTP."""

    def __init__(self, config: Dict[str, Any]):
        self.smtp_server = config.get("smtp_server", "smtp.gmail.com")
        self.smtp_port = config.get("smtp_port", 587)
        self.sender = config.get("sender", "alerts@example.com")
        self.password = config.get("password", "")
        self.recipients = config.get("recipients", [])
        self.enabled = config.get("enabled", False)

    def send(self, subject: str, body: str,
             image_path: Optional[str] = None) -> bool:
        """
        Send email alert.

        Args:
            subject: Email subject
            body: Email body text
            image_path: Optional path to attachment

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False

        if not self.recipients:
            print("Email: No recipients configured")
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = self.sender
            msg["To"] = ", ".join(self.recipients)
            msg["Subject"] = f"[ALERT] {subject}"

            msg.attach(MIMEText(body, "plain"))

            # Attach image if provided
            if image_path and Path(image_path).exists():
                from email.mime.image import MIMEImage
                with open(image_path, "rb") as f:
                    img = MIMEImage(f.read())
                    img.add_header("Content-Disposition", "attachment",
                                   filename=Path(image_path).name)
                    msg.attach(img)

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()

            if self.password:
                server.login(self.sender, self.password)

            server.send_message(msg)
            server.quit()

            print(f"Email sent: {subject}")
            return True

        except Exception as e:
            print(f"Email failed: {e}")
            return False


class SMSChannel:
    """SMS notification channel using Twilio."""

    def __init__(self, config: Dict[str, Any]):
        self.account_sid = config.get("twilio_account_sid", "")
        self.auth_token = config.get("twilio_auth_token", "")
        self.from_number = config.get("from_number", "")
        self.to_numbers = config.get("to_numbers", [])
        self.enabled = config.get("enabled", False)

        # Lazy import to avoid dependency if not used
        self._client = None

    @property
    def client(self):
        """Lazy Twilio client initialization."""
        if self._client is None:
            try:
                from twilio.rest import Client
                self._client = Client(self.account_sid, self.auth_token)
            except ImportError:
                print("Twilio not installed: pip install twilio")
                return None
        return self._client

    def send(self, subject: str, body: str,
             image_path: Optional[str] = None) -> bool:
        """
        Send SMS alert.

        Args:
            subject: Alert subject (prepended to body)
            body: Alert message
            image_path: Ignored for SMS

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False

        if not self.to_numbers:
            print("SMS: No recipients configured")
            return False

        if self.client is None:
            return False

        # Truncate message for SMS (160 char limit)
        message_body = f"{subject}: {body}"[:160]

        try:
            for number in self.to_numbers:
                self.client.messages.create(
                    body=message_body,
                    from_=self.from_number,
                    to=number
                )
            print(f"SMS sent to {len(self.to_numbers)} recipients")
            return True

        except Exception as e:
            print(f"SMS failed: {e}")
            return False


class WebhookChannel:
    """Webhook notification channel for custom integrations."""

    def __init__(self, config: Dict[str, Any]):
        self.webhook_url = config.get("url", "")
        self.enabled = config.get("enabled", False)
        self.headers = config.get("headers", {"Content-Type": "application/json"})
        self.timeout = config.get("timeout", 10)

    def send(self, subject: str, body: str,
             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send webhook notification.

        Args:
            subject: Alert subject
            body: Alert body
            metadata: Additional data to include

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False

        if not self.webhook_url:
            return False

        payload = {
            "subject": subject,
            "body": body,
            **(metadata or {})
        }

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            print(f"Webhook sent: {subject}")
            return True

        except Exception as e:
            print(f"Webhook failed: {e}")
            return False

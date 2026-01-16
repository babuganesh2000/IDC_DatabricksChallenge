"""
Multi-channel alerting system for model monitoring.

Supports multiple notification channels including Slack, Email, and PagerDuty.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertManager:
    """
    Manage multi-channel alerting for model monitoring.
    """
    
    def __init__(
        self,
        slack_webhook_url: Optional[str] = None,
        pagerduty_api_key: Optional[str] = None,
        pagerduty_routing_key: Optional[str] = None,
        email_config: Optional[Dict[str, str]] = None,
        default_channels: Optional[List[str]] = None
    ):
        """
        Initialize alert manager.
        
        Args:
            slack_webhook_url: Slack webhook URL for notifications
            pagerduty_api_key: PagerDuty API key
            pagerduty_routing_key: PagerDuty routing key
            email_config: Email configuration dict with keys:
                         smtp_server, smtp_port, sender, password, recipients
            default_channels: Default channels to use ['slack', 'email', 'pagerduty']
        """
        self.slack_webhook_url = slack_webhook_url
        self.pagerduty_api_key = pagerduty_api_key
        self.pagerduty_routing_key = pagerduty_routing_key
        self.email_config = email_config or {}
        self.default_channels = default_channels or []
        
        # Alert history
        self.alert_history: List[Dict[str, Any]] = []
        
        # Validate configurations
        self._validate_config()
        
        logger.info(
            f"Initialized AlertManager with channels: "
            f"{', '.join(self.default_channels)}"
        )
    
    def _validate_config(self) -> None:
        """Validate alert configuration."""
        if 'slack' in self.default_channels and not self.slack_webhook_url:
            logger.warning("Slack enabled but webhook URL not provided")
        
        if 'pagerduty' in self.default_channels:
            if not self.pagerduty_api_key or not self.pagerduty_routing_key:
                logger.warning(
                    "PagerDuty enabled but API key or routing key not provided"
                )
        
        if 'email' in self.default_channels:
            required_keys = ['smtp_server', 'sender', 'recipients']
            missing = [k for k in required_keys if k not in self.email_config]
            if missing:
                logger.warning(
                    f"Email enabled but missing config: {missing}"
                )
    
    def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        channels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        alert_tags: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Send alert through configured channels.
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity level
            channels: Channels to send to (None = use defaults)
            metadata: Additional alert metadata
            alert_tags: Tags for categorizing alerts
        
        Returns:
            Dictionary of channel: success status
        """
        if channels is None:
            channels = self.default_channels
        
        timestamp = datetime.now()
        
        # Store alert in history
        alert_record = {
            'timestamp': timestamp,
            'title': title,
            'message': message,
            'severity': severity.value,
            'channels': channels,
            'metadata': metadata or {},
            'tags': alert_tags or []
        }
        self.alert_history.append(alert_record)
        
        # Send through each channel
        results = {}
        
        if 'slack' in channels:
            results['slack'] = self.send_slack_alert(
                title, message, severity, metadata
            )
        
        if 'email' in channels:
            results['email'] = self.send_email_alert(
                title, message, severity, metadata
            )
        
        if 'pagerduty' in channels:
            results['pagerduty'] = self.send_pagerduty_alert(
                title, message, severity, metadata
            )
        
        # Log results
        success_channels = [c for c, s in results.items() if s]
        failed_channels = [c for c, s in results.items() if not s]
        
        if success_channels:
            logger.info(
                f"Alert sent successfully: {title} -> {', '.join(success_channels)}"
            )
        if failed_channels:
            logger.error(
                f"Alert failed to send: {title} -> {', '.join(failed_channels)}"
            )
        
        return results
    
    def send_slack_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send alert to Slack.
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity
            metadata: Additional metadata
        
        Returns:
            Success status
        """
        if not self.slack_webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False
        
        try:
            import requests
            
            # Color coding by severity
            colors = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9900",
                AlertSeverity.ERROR: "#ff0000",
                AlertSeverity.CRITICAL: "#8b0000"
            }
            
            # Build Slack message
            payload = {
                "attachments": [{
                    "color": colors.get(severity, "#808080"),
                    "title": f"{severity.value.upper()}: {title}",
                    "text": message,
                    "footer": "ML Monitoring Alert",
                    "ts": int(datetime.now().timestamp()),
                    "fields": []
                }]
            }
            
            # Add metadata as fields
            if metadata:
                for key, value in metadata.items():
                    payload["attachments"][0]["fields"].append({
                        "title": key,
                        "value": str(value),
                        "short": True
                    })
            
            response = requests.post(
                self.slack_webhook_url,
                json=payload,
                timeout=10
            )
            
            return response.status_code == 200
        
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def send_email_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send alert via email.
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity
            metadata: Additional metadata
        
        Returns:
            Success status
        """
        if not self.email_config.get('smtp_server'):
            logger.warning("Email SMTP server not configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{severity.value.upper()}] {title}"
            msg['From'] = self.email_config['sender']
            msg['To'] = ', '.join(self.email_config.get('recipients', []))
            
            # Build email body
            html_body = f"""
            <html>
            <head>
                <style>
                    .severity-{severity.value.lower()} {{
                        color: {'green' if severity == AlertSeverity.INFO 
                               else 'orange' if severity == AlertSeverity.WARNING
                               else 'red'};
                        font-weight: bold;
                    }}
                </style>
            </head>
            <body>
                <h2 class="severity-{severity.value.lower()}">
                    {severity.value.upper()} Alert
                </h2>
                <h3>{title}</h3>
                <p>{message}</p>
            """
            
            # Add metadata
            if metadata:
                html_body += "<h4>Details:</h4><ul>"
                for key, value in metadata.items():
                    html_body += f"<li><strong>{key}:</strong> {value}</li>"
                html_body += "</ul>"
            
            html_body += f"""
                <hr>
                <p style="color: gray; font-size: 12px;">
                    Sent at {datetime.now().isoformat()}
                </p>
            </body>
            </html>
            """
            
            # Attach HTML body
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            smtp_port = self.email_config.get('smtp_port', 587)
            with smtplib.SMTP(
                self.email_config['smtp_server'], smtp_port
            ) as server:
                server.starttls()
                
                if 'password' in self.email_config:
                    server.login(
                        self.email_config['sender'],
                        self.email_config['password']
                    )
                
                server.send_message(msg)
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def send_pagerduty_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send alert to PagerDuty.
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity
            metadata: Additional metadata
        
        Returns:
            Success status
        """
        if not self.pagerduty_routing_key:
            logger.warning("PagerDuty routing key not configured")
            return False
        
        try:
            import requests
            
            # Map severity to PagerDuty severity
            pd_severity_map = {
                AlertSeverity.INFO: "info",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "error",
                AlertSeverity.CRITICAL: "critical"
            }
            
            # Build PagerDuty event
            payload = {
                "routing_key": self.pagerduty_routing_key,
                "event_action": "trigger",
                "payload": {
                    "summary": title,
                    "severity": pd_severity_map[severity],
                    "source": "ml_monitoring",
                    "timestamp": datetime.now().isoformat(),
                    "custom_details": {
                        "message": message,
                        **(metadata or {})
                    }
                }
            }
            
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            return response.status_code == 202
        
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            return False
    
    def get_alert_summary(
        self,
        severity_filter: Optional[AlertSeverity] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get summary of recent alerts.
        
        Args:
            severity_filter: Filter by severity level
            limit: Maximum number of alerts to return
        
        Returns:
            Alert summary
        """
        alerts = self.alert_history
        
        # Filter by severity
        if severity_filter:
            alerts = [
                a for a in alerts
                if a['severity'] == severity_filter.value
            ]
        
        # Limit results
        alerts = alerts[-limit:]
        
        # Calculate statistics
        severity_counts = {}
        for alert in self.alert_history:
            sev = alert['severity']
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        return {
            'total_alerts': len(self.alert_history),
            'filtered_alerts': len(alerts),
            'severity_distribution': severity_counts,
            'recent_alerts': [
                {
                    'timestamp': a['timestamp'].isoformat(),
                    'title': a['title'],
                    'severity': a['severity'],
                    'channels': a['channels']
                }
                for a in alerts
            ]
        }
    
    def clear_alert_history(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear alert history.
        
        Args:
            older_than_days: Clear alerts older than N days (None = all)
        
        Returns:
            Number of alerts cleared
        """
        if older_than_days is None:
            count = len(self.alert_history)
            self.alert_history = []
            return count
        
        cutoff = datetime.now() - timedelta(days=older_than_days)
        original_count = len(self.alert_history)
        
        self.alert_history = [
            a for a in self.alert_history
            if a['timestamp'] >= cutoff
        ]
        
        cleared = original_count - len(self.alert_history)
        logger.info(f"Cleared {cleared} alerts older than {older_than_days} days")
        
        return cleared

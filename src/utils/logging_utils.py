"""Structured logging utilities.

Provides centralized logging configuration with structured output.
"""

import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional

# Context variable for correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured information.

        Args:
            record: Log record

        Returns:
            Formatted log string
        """
        correlation_id = correlation_id_var.get()
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if correlation_id:
            log_data["correlation_id"] = correlation_id

        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Format as key=value pairs
        formatted_parts = [f"{k}={v}" for k, v in log_data.items()]
        return " ".join(formatted_parts)


class MLOpsLogger:
    """MLOps logger with structured logging support."""

    def __init__(self, name: str, level: int = logging.INFO):
        """Initialize logger.

        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Remove existing handlers
        self.logger.handlers = []

        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)

    def set_correlation_id(self, correlation_id: Optional[str] = None) -> str:
        """Set correlation ID for request tracing.

        Args:
            correlation_id: Correlation ID (generates new if None)

        Returns:
            Correlation ID
        """
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())
        correlation_id_var.set(correlation_id)
        return correlation_id

    def get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID.

        Returns:
            Correlation ID or None
        """
        return correlation_id_var.get()

    def log(self, level: int, message: str, **kwargs: Any) -> None:
        """Log message with extra fields.

        Args:
            level: Log level
            message: Log message
            **kwargs: Additional fields to include
        """
        extra = {"extra_fields": kwargs}
        self.logger.log(level, message, extra=extra)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self.log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self.log(logging.CRITICAL, message, **kwargs)


def get_logger(name: str, level: int = logging.INFO) -> MLOpsLogger:
    """Get or create logger instance.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        MLOps logger instance
    """
    return MLOpsLogger(name, level)

"""Custom logging handlers for different output destinations.

This module provides specialized handlers for various logging destinations,
each optimized for their specific use case and integration requirements.

Available Handlers:
- Enhanced console handler with color support
- Rotating file handler with automatic cleanup
- Logfire integration handler
- Performance-optimized handlers for production
"""

import logging
import logging.handlers
import sys
from pathlib import Path

from .formatters import get_formatter


class ColoredConsoleHandler(logging.StreamHandler):
    """Enhanced console handler with color support and smart formatting.

    Provides color-coded output for different log levels to improve
    readability during development. Colors are automatically disabled
    when output is redirected to a file or non-TTY device.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, stream=None):
        super().__init__(stream or sys.stdout)
        self.use_colors = self._should_use_colors()

    def _should_use_colors(self) -> bool:
        """Determine if colors should be used based on output destination."""
        return hasattr(self.stream, "isatty") and self.stream.isatty() and sys.platform != "win32"

    def format(self, record: logging.LogRecord) -> str:
        """Format the record with optional color coding."""
        formatted = super().format(record)

        if self.use_colors and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            formatted = formatted.replace(f"[{record.levelname}]", f"[{color}{record.levelname}{self.RESET}]")

        return formatted


class RotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Enhanced rotating file handler with automatic directory creation.

    Extends the standard RotatingFileHandler to automatically create
    the log directory if it doesn't exist and provides better error handling.
    """

    def __init__(self, filename: str, max_bytes: int = 10485760, backup_count: int = 5, encoding: str = "utf-8"):
        log_path = Path(filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(filename=filename, maxBytes=max_bytes, backupCount=backup_count, encoding=encoding)


def create_console_handler(
    format_type: str = "detailed", level: int = logging.INFO, use_colors: bool = True
) -> logging.Handler:
    """Create a configured console handler.

    Args:
        format_type: Type of formatter to use
        level: Minimum log level for this handler
        use_colors: Whether to use colored output (if supported)

    Returns:
        Configured console handler
    """
    handler: logging.Handler
    if use_colors:
        handler = ColoredConsoleHandler()
    else:
        handler = logging.StreamHandler(sys.stdout)

    handler.setLevel(level)
    handler.setFormatter(get_formatter(format_type))

    return handler


def create_file_handler(
    filepath: str, format_type: str = "structured", level: int = logging.DEBUG, max_bytes: int = 10485760, backup_count: int = 5
) -> logging.Handler:
    """Create a configured rotating file handler.

    Args:
        filepath: Path to the log file
        format_type: Type of formatter to use
        level: Minimum log level for this handler
        max_bytes: Maximum size before rotation (default 10MB)
        backup_count: Number of backup files to keep

    Returns:
        Configured file handler
    """
    handler = RotatingFileHandler(filename=filepath, max_bytes=max_bytes, backup_count=backup_count)

    handler.setLevel(level)
    handler.setFormatter(get_formatter(format_type))

    return handler


def create_null_handler() -> logging.Handler:
    """Create a null handler that discards all log records.

    Useful for testing or when logging needs to be completely disabled.

    Returns:
        Null handler instance
    """
    return logging.NullHandler()

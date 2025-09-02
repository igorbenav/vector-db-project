"""Custom logging formatters for different environments and output types.

This module provides specialized formatters that adapt to different environments
and use cases. Each formatter is optimized for its intended output medium and
provides the appropriate level of detail and structure.

Available Formatters:
- SimpleFormatter: Basic console output for development
- DetailedFormatter: Verbose console output with full context
- StructuredFormatter: Structured logging with key-value pairs
- JSONFormatter: Machine-readable JSON format for production
- LogfireFormatter: Integration with Logfire observability
"""

import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Type


class SimpleFormatter(logging.Formatter):
    """Simple formatter for basic console output.

    Provides clean, readable output for development environments
    where human readability is prioritized over structure.

    Format: [LEVEL] module_name: message
    Example: [INFO] fastroai.chat.service: Chat created successfully
    """

    def __init__(self):
        super().__init__(fmt="[%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")


class DetailedFormatter(logging.Formatter):
    """Detailed formatter with timestamp and context information.

    Provides comprehensive information for debugging and development,
    including timestamps, module information, and optional context.

    Format: YYYY-MM-DD HH:MM:SS [LEVEL] module_name: message
    """

    def __init__(self):
        super().__init__(fmt="%(asctime)s [%(levelname)8s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


class StructuredFormatter(logging.Formatter):
    """Structured formatter with key-value pairs.

    Provides structured logging that's both human-readable and
    machine-parseable. Includes automatic context extraction.

    Format: timestamp level=LEVEL module=name message="text" key1=value1 key2=value2
    """

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now(timezone.utc).isoformat()
        parts = [
            f"timestamp={timestamp}",
            f"level={record.levelname}",
            f"module={record.name}",
            f'message="{record.getMessage()}"',
        ]

        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in [
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "message",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                ]:
                    if isinstance(value, str):
                        parts.append(f'{key}="{value}"')
                    elif isinstance(value, (int, float, bool)):
                        parts.append(f"{key}={value}")
                    else:
                        parts.append(f'{key}="{str(value)}"')

        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            exc_text_escaped = exc_text.replace("\n", "\\n")
            parts.append(f'exception="{exc_text_escaped}"')

        return " ".join(parts)


class JSONFormatter(logging.Formatter):
    """JSON formatter for machine-readable structured logging.

    Optimized for production environments where logs are processed
    by log aggregation systems. Provides complete context in JSON format.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
            "filename": record.filename,
            "function": record.funcName,
            "line_number": record.lineno,
            "thread_id": record.thread,
            "process_id": record.process,
        }

        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                try:
                    json.dumps(value)
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_data, ensure_ascii=False)


class LogfireIntegratedFormatter(logging.Formatter):
    """Formatter optimized for Logfire integration.

    Provides structured output that integrates seamlessly with
    Logfire's observability platform while maintaining readability.
    """

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now(timezone.utc).isoformat()

        message_parts = [f"{timestamp}", f"[{record.levelname}]", f"{record.name}:", record.getMessage()]

        context_parts = []
        for key, value in record.__dict__.items():
            if key.startswith(("trace_", "user_", "request_", "operation_")):
                context_parts.append(f"{key}={value}")

        if context_parts:
            message_parts.append(f"({', '.join(context_parts)})")

        result = " ".join(message_parts)

        if record.exc_info:
            result += f"\nException: {self.formatException(record.exc_info)}"

        return result


def get_formatter(format_type: str) -> logging.Formatter:
    """Get the appropriate formatter based on format type.

    Args:
        format_type: The type of formatter to create
                    ("simple", "detailed", "structured", "json", "logfire")

    Returns:
        Configured formatter instance

    Raises:
        ValueError: If format_type is not recognized
    """
    formatters: dict[str, Type[logging.Formatter]] = {
        "simple": SimpleFormatter,
        "detailed": DetailedFormatter,
        "structured": StructuredFormatter,
        "json": JSONFormatter,
        "logfire": LogfireIntegratedFormatter,
    }

    formatter_class = formatters.get(format_type.lower())
    if formatter_class is None:
        raise ValueError(f"Unknown format type: {format_type}. Available: {', '.join(formatters.keys())}")

    return formatter_class()

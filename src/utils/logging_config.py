"""
LectureLens - Logging Configuration
Configures structured, privacy-safe logging across the application.
"""

import logging
import re
from typing import Optional
from utils.config import get_config


class SensitiveDataFilter(logging.Filter):
    """Filter that masks API keys and sensitive tokens in log output."""

    def __init__(self, patterns: Optional[list[str]] = None):
        super().__init__()
        self.patterns = patterns or [
            r"AIza[0-9A-Za-z-_]{35}",  # Google API key pattern
            r"(api[_-]?key[\"']?\s*[:=]\s*[\"']?)([A-Za-z0-9-_]{20,})",
        ]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pattern in self.patterns:
                record.msg = re.sub(pattern, "[MASKED_KEY]", record.msg)
        return True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with secret masking."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        config = get_config()
        level = getattr(logging, config.log_level.upper(), logging.INFO)
        logger.setLevel(level)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.addFilter(SensitiveDataFilter())
        logger.addHandler(handler)

    return logger

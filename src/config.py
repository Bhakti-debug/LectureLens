"""
LectureLens - Config compatibility bridge.
Re-exports configuration variables from utils.config for backward compatibility.
"""

import sys
from pathlib import Path

# Ensure src is in sys.path if invoked directly
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    AppConfig,
    get_config,
    validate_config,
    BASE_DIR,
    DATA_DIR,
    ANALYSIS_PATH,
    NOTES_PATH,
    NOTES_MD_PATH,
    VISUAL_MANIFEST_PATH,
)

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Check your .env file."
    )

__all__ = [
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "AppConfig",
    "get_config",
    "validate_config",
    "BASE_DIR",
    "DATA_DIR",
    "ANALYSIS_PATH",
    "NOTES_PATH",
    "NOTES_MD_PATH",
    "VISUAL_MANIFEST_PATH",
]
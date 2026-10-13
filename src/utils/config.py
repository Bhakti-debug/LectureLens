"""
LectureLens - Central Configuration Module
Provides typed, validated configuration parameters and standard paths.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Root directory of the LectureLens project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Standard data directories
DATA_DIR = BASE_DIR / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
NOTES_DIR = DATA_DIR / "notes"
VISUALS_DIR = DATA_DIR / "visuals"
VISUAL_DIR = VISUALS_DIR
INDEXES_DIR = DATA_DIR / "indexes"
EVALUATIONS_DIR = DATA_DIR / "evaluations"
LECTURES_DIR = DATA_DIR / "lectures"

# Default single-lecture legacy paths for backward compatibility
ANALYSIS_PATH = ANALYSIS_DIR / "video_analysis.json"
NOTES_PATH = NOTES_DIR / "study_notes.json"
NOTES_MD_PATH = NOTES_DIR / "study_notes.md"
VISUAL_MANIFEST_PATH = VISUALS_DIR / "visual_manifest.json"



@dataclass(frozen=True)
class AppConfig:
    """Application configuration parameters."""
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "400")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "60")))
    retrieval_top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "4")))
    relevance_similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("RELEVANCE_SIMILARITY_THRESHOLD", "0.35"))
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Paths
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    analysis_dir: Path = ANALYSIS_DIR
    notes_dir: Path = NOTES_DIR
    visuals_dir: Path = VISUALS_DIR
    visual_dir: Path = VISUALS_DIR
    indexes_dir: Path = INDEXES_DIR

    evaluations_dir: Path = EVALUATIONS_DIR
    lectures_dir: Path = LECTURES_DIR

    analysis_path: Path = ANALYSIS_PATH
    notes_path: Path = NOTES_PATH
    notes_md_path: Path = NOTES_MD_PATH
    visual_manifest_path: Path = VISUAL_MANIFEST_PATH


    def ensure_directories(self) -> None:
        """Create required data directories if they do not exist."""
        for directory in [
            self.data_dir,
            self.analysis_dir,
            self.notes_dir,
            self.visuals_dir,
            self.indexes_dir,
            self.evaluations_dir,
            self.lectures_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


def get_config() -> AppConfig:
    """Return the global application configuration."""
    cfg = AppConfig()
    cfg.ensure_directories()
    return cfg


def validate_config(require_api_key: bool = True) -> AppConfig:
    """Validate that mandatory settings are present."""
    cfg = get_config()
    if require_api_key and not cfg.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Please add it to your .env file."
        )
    return cfg


# Backward compatibility exports
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

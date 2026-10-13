"""
Tests for URL validation, video ID extraction, and timestamp integrity normalization.
"""

import sys
from pathlib import Path

# Ensure src is in sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.validation import (
    extract_youtube_video_id,
    validate_youtube_url,
    validate_and_normalize_timestamps,
    format_timestamp,
)


def test_extract_youtube_video_id():
    """Verify robust extraction of 11-character video IDs."""
    assert extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_youtube_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_youtube_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_youtube_video_id("https://youtube.com/watch?v=dQw4w9WgXcQ&t=42s") == "dQw4w9WgXcQ"
    assert extract_youtube_video_id("not_a_url") is None
    assert extract_youtube_video_id("") is None


def test_validate_youtube_url():
    """Verify validation and normalization of YouTube URLs."""
    valid, msg, canonical = validate_youtube_url("https://youtu.be/dQw4w9WgXcQ?si=test")
    assert valid is True
    assert canonical == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    invalid, err_msg, _ = validate_youtube_url("https://vimeo.com/123456")
    assert invalid is False
    assert "Invalid YouTube URL" in err_msg

    empty_valid, empty_msg, _ = validate_youtube_url("")
    assert empty_valid is False


def test_format_timestamp():
    """Verify format_timestamp outputs MM:SS or HH:MM:SS."""
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(45) == "00:45"
    assert format_timestamp(125) == "02:05"
    assert format_timestamp(3665) == "01:01:05"
    assert format_timestamp(-10) == "00:00"


def test_validate_and_normalize_timestamps():
    """Verify section clamping, start < end enforcement, and visual validation."""
    raw_analysis = {
        "video_summary": "Test lecture",
        "sections": [
            {
                "title": "Section 1",
                "start_time": -5,  # Negative
                "end_time": 50,
                "visual_content": [
                    {"timestamp": 20, "description": "Good visual"},
                    {"timestamp": -3, "description": "Negative visual"},
                    {"timestamp": 120, "description": "Overshoot visual"},
                ],
            },
            {
                "title": "Section 2",
                "start_time": 80,
                "end_time": 70,  # start > end
                "visual_content": [],
            },
            {
                "title": "Section 3",
                "start_time": 95,
                "end_time": 150,  # Exceeds video duration 100s
                "visual_content": [],
            },
        ],
    }

    normalized = validate_and_normalize_timestamps(raw_analysis, actual_duration=100.0)
    sections = normalized["sections"]

    # Section 1 checks
    assert sections[0]["start_time"] == 0.0  # Clamped negative
    assert sections[0]["end_time"] == 50.0
    # Negative visual filtered out
    assert len(sections[0]["visual_content"]) == 2
    assert sections[0]["visual_content"][0]["timestamp"] == 20.0
    # Overshoot visual clamped to <= 100
    assert sections[0]["visual_content"][1]["timestamp"] <= 100.0

    # Section 2 checks: start_time < end_time enforced
    assert sections[1]["start_time"] < sections[1]["end_time"]

    # Section 3 checks: end_time clamped to verified duration
    assert sections[2]["end_time"] == 100.0

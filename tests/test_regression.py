"""
Regression tests verifying existing core functions and schemas remain fully backward-compatible.
"""

import json
import sys
import tempfile
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from note_generator import (
    clean_json_response,
    attach_visuals_to_notes,
    load_analysis,
    load_visual_manifest,
)
from video_analyzer import extract_json
from visual_extractor import make_safe_filename


def test_extract_json_markdown_fences():
    """Verify extract_json strips markdown code fences correctly."""
    raw_with_fences = """```json
    {
        "video_summary": "Test summary",
        "sections": []
    }
    ```"""
    parsed = extract_json(raw_with_fences)
    assert parsed["video_summary"] == "Test summary"
    assert parsed["sections"] == []


def test_clean_json_response():
    """Verify clean_json_response cleans leading and trailing fences."""
    text = "```json\n{\"title\": \"Node\"}\n```"
    cleaned = clean_json_response(text)
    assert json.loads(cleaned)["title"] == "Node"


def test_make_safe_filename():
    """Verify safe filename conversion avoids illegal filesystem characters."""
    raw_title = 'What is "Node.js"? / A Complete Guide: Part 1*'
    safe = make_safe_filename(raw_title)
    assert "/" not in safe
    assert '"' not in safe
    assert "?" not in safe
    assert ":" not in safe
    assert "*" not in safe
    assert len(safe) <= 60


def test_attach_visuals_to_notes():
    """Verify visual items from manifest are attached to note sections."""
    notes = {
        "title": "Sample",
        "sections": [
            {
                "section_number": 1,
                "title": "Intro",
                "start_time": 0,
                "end_time": 60,
            }
        ],
    }

    manifest = {
        "visual_count": 1,
        "visuals": [
            {
                "section_number": 1,
                "timestamp": 25.0,
                "type": "diagram",
                "description": "Intro diagram",
                "image": "img1.jpg",
            }
        ],
    }

    merged = attach_visuals_to_notes(notes, manifest)
    section_visuals = merged["sections"][0].get("visuals", [])
    assert len(section_visuals) == 1
    assert section_visuals[0]["timestamp"] == 25.0
    assert section_visuals[0]["image"] == "img1.jpg"


def test_load_visual_manifest_missing_fallback():
    """Verify load_visual_manifest gracefully returns empty structure if file does not exist."""
    missing_path = Path("non_existent_manifest.json")
    manifest = load_visual_manifest(missing_path)
    assert manifest["visual_count"] == 0
    assert manifest["visuals"] == []

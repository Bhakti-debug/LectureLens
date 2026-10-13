"""
Tests for Markdown and PDF document export services.
"""

import sys
import tempfile
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from services.export_service import ExportService


def test_export_markdown_and_pdf():
    """Verify Markdown and PDF exports generate valid files on disk."""
    service = ExportService()

    sample_notes = {
        "title": "Test Lecture: Python Basics",
        "overview": "A comprehensive introductory overview of Python programming.",
        "duration_seconds": 360,
        "sections": [
            {
                "section_number": 1,
                "title": "Syntax & Semantics",
                "start_time": 0.0,
                "end_time": 180.0,
                "explanation": "Python uses indentation rather than braces to define code blocks.",
                "key_concepts": ["Indentation", "Readability", "PEP 8"],
                "examples": ["def hello():\n    print('Hello')"],
                "key_takeaway": "Readable code is maintainable code.",
                "visuals": [
                    {
                        "type": "code",
                        "timestamp": 45.0,
                        "description": "Code editor showing indentation error",
                        "image": "non_existent_image.jpg",  # Verify missing image does not crash export
                    }
                ],
            }
        ],
        "final_takeaways": ["Write readable code", "Follow PEP 8 guidelines"],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Markdown Export
        md_file = tmp_path / "notes.md"
        exported_md = service.export_markdown(sample_notes, md_file)
        assert exported_md.exists()
        content = exported_md.read_text(encoding="utf-8")
        assert "# Test Lecture: Python Basics" in content
        assert "Syntax & Semantics" in content
        assert "Readable code is maintainable code." in content

        # 2. PDF Export
        pdf_file = tmp_path / "notes.pdf"
        exported_pdf = service.export_pdf(sample_notes, pdf_file)
        assert exported_pdf.exists()
        # Verify non-trivial PDF size (> 2KB)
        assert exported_pdf.stat().st_size > 2000

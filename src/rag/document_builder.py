"""
LectureLens - Document Builder
Constructs normalized, metadata-rich lecture documents from structured multimodal analysis.
"""

from typing import Any, Dict, List, Optional
from models.schemas import LectureDocument
from utils.validation import format_timestamp


def build_lecture_documents(
    analysis_data: Dict[str, Any],
    lecture_id: str,
    notes_data: Optional[Dict[str, Any]] = None,
) -> List[LectureDocument]:
    """
    Transform multimodal analysis and optional notes into standardized LectureDocument objects.

    Each document captures:
    - Section title and timestamp range
    - Spoken lecture content
    - Visual content descriptions and visible text
    - Audio-visual relationships
    - Key conceptual definitions
    """
    documents: List[LectureDocument] = []
    sections = analysis_data.get("sections", [])

    if not isinstance(sections, list):
        return documents

    # Index notes by section number if provided
    notes_by_section: Dict[int, Dict[str, Any]] = {}
    if notes_data and isinstance(notes_data.get("sections"), list):
        for sec in notes_data["sections"]:
            sec_num = sec.get("section_number")
            if sec_num is not None:
                notes_by_section[int(sec_num)] = sec

    for idx, section in enumerate(sections, start=1):
        section_number = idx
        title = section.get("title", f"Section {section_number}")
        start_time = float(section.get("start_time", 0.0))
        end_time = float(section.get("end_time", 0.0))

        content_parts: List[str] = []

        # 1. Section Header & Spoken Content
        content_parts.append(f"Section {section_number}: {title} [{format_timestamp(start_time)} - {format_timestamp(end_time)}]")

        spoken_points = section.get("spoken_content", [])
        if isinstance(spoken_points, list) and spoken_points:
            content_parts.append("Spoken Lecture Content:")
            for pt in spoken_points:
                if pt and str(pt).strip():
                    content_parts.append(f"- {str(pt).strip()}")

        # 2. Visual Content & On-screen Text
        visual_items = section.get("visual_content", [])
        if isinstance(visual_items, list) and visual_items:
            content_parts.append("Visual Material & Slides:")
            for v in visual_items:
                if not isinstance(v, dict):
                    continue
                v_type = v.get("type", "visual")
                v_desc = v.get("description", "").strip()
                v_text = v.get("visible_text", "").strip()
                v_ts = v.get("timestamp")
                ts_str = f" at {format_timestamp(v_ts)}" if v_ts is not None else ""

                entry = f"- [{v_type}{ts_str}]"
                if v_desc:
                    entry += f" {v_desc}"
                if v_text:
                    entry += f" | Text on screen: \"{v_text}\""
                content_parts.append(entry)

        # 3. Audio-Visual Relationship
        av_rel = section.get("audio_visual_relationship", [])
        if isinstance(av_rel, list) and av_rel:
            content_parts.append("Audio-Visual Synthesis:")
            for rel in av_rel:
                if rel and str(rel).strip():
                    content_parts.append(f"- {str(rel).strip()}")

        # 4. Key Concepts
        concepts = section.get("key_concepts", [])
        if isinstance(concepts, list) and concepts:
            content_parts.append(f"Key Concepts: {', '.join(str(c) for c in concepts)}")

        # 5. Complementary Note Explanation (if present and distinct)
        note_sec = notes_by_section.get(section_number)
        if note_sec:
            takeaway = note_sec.get("key_takeaway")
            if takeaway:
                content_parts.append(f"Key Takeaway: {takeaway}")

        full_text = "\n".join(content_parts)

        doc = LectureDocument(
            doc_id=f"{lecture_id}_sec_{section_number:02d}",
            lecture_id=lecture_id,
            section_number=section_number,
            section_title=title,
            start_time=start_time,
            end_time=end_time,
            text=full_text,
            source_type="analyzed_section",
            metadata={
                "video_summary": analysis_data.get("video_summary", ""),
                "key_concepts": concepts,
            },
        )
        documents.append(doc)

    return documents

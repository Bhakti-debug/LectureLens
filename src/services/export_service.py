"""
LectureLens - Export Service
Generates publication-quality study materials in Markdown and PDF formats.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from utils.logging_config import get_logger
from utils.validation import format_timestamp

logger = get_logger("export_service")


class ExportService:
    """Exports structured lecture notes to Markdown and PDF."""

    def __init__(self):
        pass

    def export_markdown(
        self,
        notes_data: Dict[str, Any],
        output_path: Path,
    ) -> Path:
        """Export notes to a clean Markdown document."""
        from note_generator import create_markdown

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        md_content = create_markdown(notes_data, output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Exported Markdown notes to {output_path}")
        return output_path

    def export_pdf(
        self,
        notes_data: Dict[str, Any],
        output_path: Path,
        base_dir: Optional[Path] = None,
    ) -> Path:
        """
        Generate a formatted, publication-quality PDF from study notes using ReportLab.
        Includes cover header, overview, topic sections, visuals with captions, and takeaways.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        styles = getSampleStyleSheet()

        # Custom typography styles
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#1E293B"),
            spaceAfter=6,
        )

        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#64748B"),
            spaceAfter=15,
        )

        section_heading_style = ParagraphStyle(
            "SecHeading",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#0F172A"),
            spaceBefore=14,
            spaceAfter=6,
        )

        subheading_style = ParagraphStyle(
            "SubHeading",
            parent=styles["Heading3"],
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#2563EB"),
            spaceBefore=8,
            spaceAfter=4,
        )

        body_style = ParagraphStyle(
            "DocBody",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#334155"),
            spaceAfter=6,
        )

        bullet_style = ParagraphStyle(
            "DocBullet",
            parent=styles["Normal"],
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#334155"),
            leftIndent=15,
            spaceAfter=3,
        )

        callout_style = ParagraphStyle(
            "CalloutText",
            parent=styles["Normal"],
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#1E3A8A"),
        )

        caption_style = ParagraphStyle(
            "ImgCaption",
            parent=styles["Italic"],
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#64748B"),
            alignment=1,  # Center
            spaceAfter=8,
        )

        story = []

        # 1. Header / Title
        lecture_title = notes_data.get("title", "Lecture Study Notes")
        story.append(Paragraph(lecture_title, title_style))

        duration_sec = notes_data.get("duration_seconds", 0)
        dur_str = f"Duration: {format_timestamp(duration_sec)}" if duration_sec else ""
        date_str = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        meta_line = f"{dur_str}  •  {date_str}  •  LectureLens Study Notes"
        story.append(Paragraph(meta_line, subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0"), spaceAfter=12))

        # 2. Overview
        overview = notes_data.get("overview")
        if overview:
            story.append(Paragraph("Lecture Overview", section_heading_style))
            story.append(Paragraph(overview, body_style))
            story.append(Spacer(1, 10))

        # 3. Topic Sections
        sections = notes_data.get("sections", [])
        for sec in sections:
            sec_num = sec.get("section_number", "")
            sec_title = sec.get("title", "Untitled Section")
            start = format_timestamp(sec.get("start_time", 0))
            end = format_timestamp(sec.get("end_time", 0))

            sec_header = f"{sec_num}. {sec_title} <font color='#64748B' size='9'>({start} - {end})</font>"
            story.append(Paragraph(sec_header, section_heading_style))

            explanation = sec.get("explanation")
            if explanation:
                story.append(Paragraph("Explanation", subheading_style))
                story.append(Paragraph(explanation, body_style))

            concepts = sec.get("key_concepts", [])
            if concepts:
                story.append(Paragraph("Key Concepts", subheading_style))
                for c in concepts:
                    story.append(Paragraph(f"• <b>{c}</b>", bullet_style))

            examples = sec.get("examples", [])
            if examples:
                story.append(Paragraph("Examples", subheading_style))
                for ex in examples:
                    story.append(Paragraph(f"• {ex}", bullet_style))

            # Visuals
            visuals = sec.get("visuals", [])
            if visuals:
                story.append(Paragraph("Visual Assets", subheading_style))
                for v in visuals:
                    v_type = v.get("type", "visual").title()
                    v_ts = format_timestamp(v.get("timestamp", 0))
                    v_desc = v.get("description", "")
                    img_path = v.get("image")

                    if img_path:
                        resolved_path = Path(img_path)
                        if not resolved_path.exists() and base_dir:
                            # Try relative to base_dir or project root
                            norm = str(img_path).replace("\\", "/")
                            candidate = base_dir / norm
                            if candidate.exists():
                                resolved_path = candidate
                            else:
                                candidate2 = base_dir / "data" / "visuals" / Path(img_path).name
                                if candidate2.exists():
                                    resolved_path = candidate2

                        if resolved_path.exists():
                            try:
                                # Scale image preserving aspect ratio to fit width (max 5.5 inches)
                                img = RLImage(str(resolved_path), width=5.5 * inch, height=3.0 * inch)
                                story.append(img)
                            except Exception as e:
                                logger.warning(f"Could not render image {resolved_path}: {e}")

                    caption_text = f"<b>{v_type} at {v_ts}</b>: {v_desc}"
                    story.append(Paragraph(caption_text, caption_style))

            takeaway = sec.get("key_takeaway")
            if takeaway:
                callout_data = [[Paragraph(f"<b>Key Takeaway:</b> {takeaway}", callout_style)]]
                callout_table = Table(callout_data, colWidths=[7.0 * inch])
                callout_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
                            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#BFDBFE")),
                            ("PADDING", (0, 0), (-1, -1), 8),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ]
                    )
                )
                story.append(Spacer(1, 4))
                story.append(callout_table)
                story.append(Spacer(1, 8))

        # 4. Final Takeaways
        final_takeaways = notes_data.get("final_takeaways", [])
        if final_takeaways:
            story.append(Spacer(1, 12))
            story.append(Paragraph("Final Takeaways & Summary", section_heading_style))
            for ft in final_takeaways:
                story.append(Paragraph(f"✓ {ft}", bullet_style))

        # Build document
        doc.build(story)
        logger.info(f"Successfully generated PDF: {output_path}")
        return output_path

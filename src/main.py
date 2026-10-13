import json
import shutil
from pathlib import Path

import streamlit as st

from video_analyzer import analyze_youtube_video, save_analysis
from visual_extractor import extract_visuals
from note_generator import generate_notes_from_file


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

ANALYSIS_PATH = (
    BASE_DIR
    / "data"
    / "analysis"
    / "video_analysis.json"
)

NOTES_PATH = (
    BASE_DIR
    / "data"
    / "notes"
    / "study_notes.json"
)

VISUAL_DIR = (
    BASE_DIR
    / "data"
    / "visuals"
)

VISUAL_MANIFEST_PATH = (
    VISUAL_DIR
    / "visual_manifest.json"
)


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="LectureLens",
    page_icon="🎓",
    layout="wide",
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_json(path):
    """Load a JSON file."""

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_timestamp(seconds):
    """Convert seconds to MM:SS or HH:MM:SS."""

    seconds = int(seconds or 0)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def get_visuals_for_section(section_number, visual_manifest):
    """Return visuals belonging to a particular section."""

    if not visual_manifest:
        return []

    return [
        visual
        for visual in visual_manifest.get("visuals", [])
        if visual.get("section_number") == section_number
    ]


def resolve_image_path(image_path):
    """
    Resolve an image path stored in the manifest.

    Handles Windows paths and project-relative paths.
    """

    if not image_path:
        return None

    path = Path(image_path)

    if path.exists():
        return path

    # Normalize Windows separators.
    normalized = str(image_path).replace("\\", "/")

    relative_path = BASE_DIR / normalized

    if relative_path.exists():
        return relative_path

    return None


def clear_previous_run():
    """
    Remove generated artifacts from the previous lecture.

    This is important because the application currently uses
    fixed output locations.
    """

    if ANALYSIS_PATH.exists():
        ANALYSIS_PATH.unlink()

    if NOTES_PATH.exists():
        NOTES_PATH.unlink()

    if VISUAL_DIR.exists():

        for item in VISUAL_DIR.iterdir():

            if item.is_file() or item.is_symlink():
                item.unlink()

            elif item.is_dir():
                shutil.rmtree(item)


def run_pipeline(youtube_url):
    """
    Run the complete LectureLens pipeline for one YouTube URL.

    Pipeline:

        YouTube URL
            ↓
        Gemini multimodal analysis
            ↓
        video_analysis.json
            ↓
        visual extraction
            ↓
        visual_manifest.json + images
            ↓
        study note generation
            ↓
        study_notes.json
    """

    # -----------------------------------------------------
    # 1. Remove previous lecture artifacts
    # -----------------------------------------------------

    clear_previous_run()

    # -----------------------------------------------------
    # 2. Gemini multimodal analysis
    # -----------------------------------------------------

    with st.status(
        "Analyzing the lecture with Gemini...",
        expanded=True
    ) as status:

        st.write(
            "Sending the YouTube video directly to Gemini."
        )

        analysis = analyze_youtube_video(
            youtube_url
        )

        save_analysis(
            analysis,
            ANALYSIS_PATH
        )

        st.write(
            f"Detected "
            f"{len(analysis.get('sections', []))} "
            f"lecture sections."
        )

        # -------------------------------------------------
        # 3. Extract meaningful visuals
        # -------------------------------------------------

        st.write(
            "Preparing lecture visuals..."
        )

        visual_manifest = extract_visuals(
            youtube_url=youtube_url,
            analysis_path=ANALYSIS_PATH,
            output_dir=VISUAL_DIR,
        )

        st.write(
            f"Prepared "
            f"{visual_manifest.get('visual_count', 0)} "
            f"meaningful visuals."
        )

        # -------------------------------------------------
        # 4. Generate study notes
        # -------------------------------------------------

        st.write(
            "Generating structured study notes..."
        )

        notes = generate_notes_from_file(
            input_path=ANALYSIS_PATH,
            json_output=NOTES_PATH,
            markdown_output=(
                BASE_DIR
                / "data"
                / "notes"
                / "study_notes.md"
            ),
        )

        status.update(
            label="Lecture processing complete.",
            state="complete",
            expanded=False,
        )

    return notes, visual_manifest


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🎓 LectureLens")

st.caption(
    "Turn educational YouTube lectures into structured, "
    "multimodal study notes."
)


# ---------------------------------------------------------
# YouTube input
# ---------------------------------------------------------

st.subheader("Create Study Notes")

youtube_url = st.text_input(
    "YouTube lecture URL",
    placeholder="https://www.youtube.com/watch?v=...",
    help="Paste a public YouTube lecture URL.",
)


generate = st.button(
    "🚀 Generate Study Notes",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------

if generate:

    youtube_url = youtube_url.strip()

    if not youtube_url:

        st.error(
            "Please enter a YouTube URL."
        )

        st.stop()

    try:

        with st.spinner(
            "Processing lecture..."
        ):

            notes, visual_manifest = run_pipeline(
                youtube_url
            )

        # Store the generated data in session state.
        st.session_state["notes"] = notes
        st.session_state["visual_manifest"] = (
            visual_manifest
        )
        st.session_state["youtube_url"] = (
            youtube_url
        )

        st.success(
            "Study notes generated successfully."
        )

    except Exception as exc:

        st.error(
            f"Lecture processing failed:\n\n{exc}"
        )

        st.stop()


# ---------------------------------------------------------
# Load current data
# ---------------------------------------------------------

if "notes" in st.session_state:

    notes = st.session_state["notes"]

else:

    notes = load_json(
        NOTES_PATH
    )


if "visual_manifest" in st.session_state:

    visual_manifest = (
        st.session_state["visual_manifest"]
    )

else:

    visual_manifest = load_json(
        VISUAL_MANIFEST_PATH
    )


# ---------------------------------------------------------
# Empty state
# ---------------------------------------------------------

if notes is None:

    st.info(
        "Paste a YouTube lecture URL above and click "
        "**Generate Study Notes** to begin."
    )

    st.stop()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("Lecture")

    st.markdown(
        f"**{notes.get('title', 'Lecture Notes')}**"
    )

    duration = notes.get(
        "duration_seconds"
    )

    if duration:

        st.caption(
            f"Duration: {format_timestamp(duration)}"
        )

    st.divider()

    st.subheader("Sections")

    sections = notes.get(
        "sections",
        []
    )

    for section in sections:

        number = section.get(
            "section_number",
            ""
        )

        title = section.get(
            "title",
            "Untitled Section"
        )

        start = format_timestamp(
            section.get(
                "start_time",
                0
            )
        )

        st.markdown(
            f"**{number}. {title}**  \n"
            f"`{start}`"
        )


# ---------------------------------------------------------
# Overview
# ---------------------------------------------------------

st.header(
    notes.get(
        "title",
        "Lecture Notes"
    )
)

overview = notes.get(
    "overview"
)

if overview:

    st.markdown(
        overview
    )


# ---------------------------------------------------------
# Main sections
# ---------------------------------------------------------

for section in notes.get(
    "sections",
    []
):

    section_number = section.get(
        "section_number",
        ""
    )

    title = section.get(
        "title",
        "Untitled Section"
    )

    start = format_timestamp(
        section.get(
            "start_time",
            0
        )
    )

    end = format_timestamp(
        section.get(
            "end_time",
            0
        )
    )

    st.divider()

    st.header(
        f"{section_number}. {title}"
    )

    st.caption(
        f"⏱ {start} — {end}"
    )

    # -----------------------------------------------------
    # Explanation
    # -----------------------------------------------------

    explanation = section.get(
        "explanation"
    )

    if explanation:

        st.subheader(
            "Explanation"
        )

        st.write(
            explanation
        )

    # -----------------------------------------------------
    # Key concepts
    # -----------------------------------------------------

    concepts = section.get(
        "key_concepts",
        []
    )

    if concepts:

        st.subheader(
            "Key Concepts"
        )

        for concept in concepts:

            st.markdown(
                f"- {concept}"
            )

    # -----------------------------------------------------
    # Examples
    # -----------------------------------------------------

    examples = section.get(
        "examples",
        []
    )

    if examples:

        st.subheader(
            "Examples"
        )

        for example in examples:

            st.markdown(
                f"- {example}"
            )

    # -----------------------------------------------------
    # Visual explanation
    # -----------------------------------------------------

    visual_explanation = section.get(
        "visual_explanation"
    )

    if visual_explanation:

        st.subheader(
            "Visual Explanation"
        )

        st.info(
            visual_explanation
        )

    # -----------------------------------------------------
    # Important visual text
    # -----------------------------------------------------

    visual_text = section.get(
        "important_visual_text",
        []
    )

    if visual_text:

        st.subheader(
            "Important Text From Visuals"
        )

        for text in visual_text:

            st.markdown(
                f"- `{text}`"
            )

    # -----------------------------------------------------
    # Actual visual images
    # -----------------------------------------------------

    section_visuals = get_visuals_for_section(
        section_number,
        visual_manifest
    )

    if section_visuals:

        st.subheader(
            "Visuals"
        )

        for visual in section_visuals:

            timestamp = visual.get(
                "timestamp",
                0
            )

            visual_type = visual.get(
                "type",
                "other"
            )

            description = visual.get(
                "description",
                ""
            )

            image_path = resolve_image_path(
                visual.get("image")
            )

            st.markdown(
                f"**{visual_type.title()} — "
                f"{format_timestamp(timestamp)}**"
            )

            if image_path:

                st.image(
                    str(image_path),
                    width="stretch"
                )

            else:

                st.warning(
                    "Visual image could not be found."
                )

            if description:

                st.caption(
                    description
                )

            visible_text = visual.get(
                "visible_text"
            )

            if visible_text:

                with st.expander(
                    "View visible text"
                ):

                    st.code(
                        visible_text
                    )

    # -----------------------------------------------------
    # Key takeaway
    # -----------------------------------------------------

    takeaway = section.get(
        "key_takeaway"
    )

    if takeaway:

        st.subheader(
            "Key Takeaway"
        )

        st.success(
            takeaway
        )


# ---------------------------------------------------------
# Final takeaways
# ---------------------------------------------------------

final_takeaways = notes.get(
    "final_takeaways",
    []
)

if final_takeaways:

    st.divider()

    st.header(
        "Final Takeaways"
    )

    for takeaway in final_takeaways:

        st.markdown(
            f"- {takeaway}"
        )


# ---------------------------------------------------------
# Debug information
# ---------------------------------------------------------

with st.sidebar:

    st.divider()

    visual_count = (
        visual_manifest.get(
            "visual_count",
            0
        )
        if visual_manifest
        else 0
    )

    st.caption(
        f"Visuals: {visual_count}"
    )
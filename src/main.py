"""
LectureLens - Streamlit Main Application
Multimodal Educational Video Understanding, Study Note Generation,
Lecture-Grounded RAG Assistant, and Revision Tools.
"""

import json
import shutil
import sys
from pathlib import Path
import streamlit as st

# Ensure src is in sys.path
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from video_analyzer import analyze_youtube_video, save_analysis
from visual_extractor import extract_visuals
from note_generator import generate_notes_from_file
from rag.pipeline import RAGPipeline
from services.lecture_service import get_lecture_service
from services.quiz_service import QuizService
from services.export_service import ExportService
from utils.config import (
    BASE_DIR,
    ANALYSIS_PATH,
    NOTES_PATH,
    NOTES_MD_PATH,
    VISUAL_DIR,
    VISUAL_MANIFEST_PATH,
    get_config,
)
from utils.validation import format_timestamp, validate_youtube_url


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="LectureLens - Multimodal AI Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-caption {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-time {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .source-box {
        border-left: 3px solid #3B82F6;
        padding-left: 10px;
        margin-top: 8px;
        margin-bottom: 8px;
        background-color: #F8FAFC;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def load_json(path):
    """Load a JSON file safely."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_visuals_for_section(section_number, visual_manifest):
    """Return visuals belonging to a particular section."""
    if not visual_manifest:
        return []
    return [
        visual
        for visual in visual_manifest.get("visuals", [])
        if visual.get("section_number") == section_number
    ]


def resolve_image_path(image_path, lecture_id=None):
    """
    Resolve an image path reliably across absolute, relative,
    or relocated directories.
    """
    if not image_path:
        return None

    path = Path(image_path)
    if path.exists():
        return path

    normalized = str(image_path).replace("\\", "/")
    relative_path = BASE_DIR / normalized
    if relative_path.exists():
        return relative_path

    # Fallback by filename in active visuals directory
    filename = Path(image_path).name
    cand1 = VISUAL_DIR / filename
    if cand1.exists():
        return cand1

    if lecture_id:
        cand2 = BASE_DIR / "data" / "lectures" / lecture_id / "visuals" / filename
        if cand2.exists():
            return cand2

    return None


def run_pipeline(youtube_url: str):
    """
    Execute the complete LectureLens pipeline for a YouTube URL:
    1. Gemini Multimodal Video Understanding with timestamp integrity
    2. Frame extraction and visual manifest creation
    3. Structured study note synthesis
    4. Lecture-isolated storage and RAG vector indexing
    """
    service = get_lecture_service()
    lecture_id = service.get_lecture_id_from_url(youtube_url)

    with st.status("Processing educational lecture...", expanded=True) as status:
        st.write("🔍 Step 1: Multimodal analysis with Gemini...")
        analysis = analyze_youtube_video(youtube_url)
        save_analysis(analysis, ANALYSIS_PATH)
        st.write(f"✓ Detected {len(analysis.get('sections', []))} timestamped lecture sections.")

        st.write("📸 Step 2: Extracting educational visual frames with OpenCV...")
        visual_manifest = extract_visuals(
            youtube_url=youtube_url,
            analysis_path=ANALYSIS_PATH,
            output_dir=VISUAL_DIR,
        )
        st.write(f"✓ Extracted {visual_manifest.get('visual_count', 0)} key visual frames.")

        st.write("📝 Step 3: Generating comprehensive study notes...")
        notes = generate_notes_from_file(
            input_path=ANALYSIS_PATH,
            json_output=NOTES_PATH,
            markdown_output=NOTES_MD_PATH,
            visual_manifest_path=VISUAL_MANIFEST_PATH,
        )
        st.write("✓ Study notes synthesized successfully.")

        st.write("⚡ Step 4: Indexing lecture into FAISS vector store for Q&A...")
        pipeline = RAGPipeline(
            lecture_id=lecture_id,
            lecture_title=notes.get("title", "Lecture"),
        )
        pipeline.index_lecture(analysis_data=analysis, notes_data=notes, force_reindex=True)
        st.write("✓ Vector index built and cached.")

        # Save to lecture-isolated store
        service.save_lecture(
            lecture_id=lecture_id,
            analysis_data=analysis,
            notes_data=notes,
            visual_manifest=visual_manifest,
            sync_legacy=True,
        )

        status.update(label="Lecture processing complete!", state="complete", expanded=False)

    return lecture_id, notes, visual_manifest, analysis


# ---------------------------------------------------------
# Sidebar: Lecture Selection & Lecture Overview
# ---------------------------------------------------------
lecture_service = get_lecture_service()
available_lectures = lecture_service.list_lectures()

with st.sidebar:
    st.header("📚 Lecture Library")

    lecture_options = {l["lecture_id"]: l.get("title", l["lecture_id"]) for l in available_lectures}
    if not lecture_options:
        lecture_options["sample_node_lecture"] = "Introduction to Node.js and Core Concepts"

    selected_lec_id = st.selectbox(
        "Active Lecture",
        options=list(lecture_options.keys()),
        format_func=lambda x: lecture_options.get(x, x),
        help="Select a previously analyzed lecture to study or ask questions about.",
    )

    # Load active lecture data
    active_lecture_data = lecture_service.load_lecture(selected_lec_id)
    notes = active_lecture_data.get("notes") or load_json(NOTES_PATH)
    visual_manifest = active_lecture_data.get("visual_manifest") or load_json(VISUAL_MANIFEST_PATH)
    analysis = active_lecture_data.get("analysis") or load_json(ANALYSIS_PATH)

    st.divider()

    if notes:
        st.subheader("Lecture Info")
        st.markdown(f"**{notes.get('title', 'Lecture Notes')}**")
        duration = notes.get("duration_seconds") or (analysis.get("verified_duration_seconds") if analysis else 0)
        if duration:
            st.caption(f"⏱ Duration: {format_timestamp(duration)}")

        st.divider()
        st.subheader("Sections")
        for s in notes.get("sections", []):
            num = s.get("section_number", "")
            title = s.get("title", "Section")
            start = format_timestamp(s.get("start_time", 0))
            st.markdown(f"**{num}. {title}**  \n`{start}`")

    st.divider()
    visual_count = visual_manifest.get("visual_count", 0) if visual_manifest else 0
    st.caption(f"Visual Assets Extracted: {visual_count}")


# ---------------------------------------------------------
# Header & New Lecture Input
# ---------------------------------------------------------
st.markdown('<div class="main-header">🎓 LectureLens</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-caption">An Intelligent Multimodal System for Study Note Generation & Question Answering from Educational Video Lectures</div>',
    unsafe_allow_html=True,
)

with st.expander("➕ Analyze a New YouTube Lecture", expanded=False):
    youtube_url_input = st.text_input(
        "YouTube Lecture URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste a public educational YouTube video URL.",
    )
    col1, col2 = st.columns([1, 4])
    with col1:
        start_btn = st.button("🚀 Analyze Lecture", type="primary", use_container_width=True)

    if start_btn:
        url_clean = youtube_url_input.strip()
        is_valid, msg, canonical_url = validate_youtube_url(url_clean)
        if not is_valid:
            st.error(msg)
        else:
            try:
                new_lec_id, notes, visual_manifest, analysis = run_pipeline(canonical_url)
                st.session_state["active_lecture_id"] = new_lec_id
                st.success("Lecture analyzed successfully! Refreshing active view...")
                st.rerun()
            except Exception as e:
                st.error(f"Lecture processing failed: {e}")


# ---------------------------------------------------------
# Primary View Tabs
# ---------------------------------------------------------
if not notes:
    st.info("Paste a public educational YouTube lecture URL above to begin, or select a lecture from the sidebar.")
    st.stop()

tab_notes, tab_rag, tab_flashcards, tab_quiz, tab_export = st.tabs(
    [
        "📖 Study Notes & Visuals",
        "💬 Lecture Q&A Assistant",
        "🗂️ Revision Flashcards",
        "📝 Self-Assessment Quiz",
        "📥 Export Study Material",
    ]
)


# =========================================================
# TAB 1: STUDY NOTES & VISUALS
# =========================================================
with tab_notes:
    st.header(notes.get("title", "Lecture Notes"))

    overview = notes.get("overview")
    if overview:
        st.markdown(f"*{overview}*")
        st.write("")

    for section in notes.get("sections", []):
        section_number = section.get("section_number", "")
        title = section.get("title", "Untitled Section")
        start = format_timestamp(section.get("start_time", 0))
        end = format_timestamp(section.get("end_time", 0))

        st.divider()
        st.subheader(f"{section_number}. {title}")
        st.markdown(f'<span class="badge-time">⏱ {start} — {end}</span>', unsafe_allow_html=True)
        st.write("")

        # Explanation
        explanation = section.get("explanation")
        if explanation:
            st.markdown("#### Explanation")
            st.write(explanation)

        # Key concepts
        concepts = section.get("key_concepts", [])
        if concepts:
            st.markdown("#### Key Concepts")
            for c in concepts:
                st.markdown(f"- **{c}**")

        # Examples
        examples = section.get("examples", [])
        if examples:
            st.markdown("#### Examples")
            for ex in examples:
                st.markdown(f"- {ex}")

        # Visual explanation
        visual_explanation = section.get("visual_explanation")
        if visual_explanation:
            st.info(f"**Visual Explanation:** {visual_explanation}")

        # Important visual text
        visual_text = section.get("important_visual_text", [])
        if visual_text:
            st.markdown("#### Important Text From Slides / Code")
            for t in visual_text:
                st.code(t)

        # Visual assets
        section_visuals = get_visuals_for_section(section_number, visual_manifest)
        if section_visuals:
            st.markdown("#### Extracted Visuals")
            cols = st.columns(min(len(section_visuals), 2))
            for v_idx, visual in enumerate(section_visuals):
                with cols[v_idx % len(cols)]:
                    v_ts = format_timestamp(visual.get("timestamp", 0))
                    v_type = visual.get("type", "visual").title()
                    v_desc = visual.get("description", "")
                    img_path = resolve_image_path(visual.get("image"), selected_lec_id)

                    st.markdown(f"**{v_type} at {v_ts}**")
                    if img_path:
                        st.image(str(img_path), use_container_width=True)
                    else:
                        st.warning("Visual frame file not found on disk.")
                    if v_desc:
                        st.caption(v_desc)

        # Key takeaway
        takeaway = section.get("key_takeaway")
        if takeaway:
            st.success(f"**Key Takeaway:** {takeaway}")

    # Final takeaways
    final_takeaways = notes.get("final_takeaways", [])
    if final_takeaways:
        st.divider()
        st.header("🎯 Final Takeaways")
        for ft in final_takeaways:
            st.markdown(f"- {ft}")


# =========================================================
# TAB 2: LECTURE Q&A ASSISTANT (RAG)
# =========================================================
with tab_rag:
    st.header("💬 Lecture-Grounded Q&A Assistant")
    st.caption(
        f"Ask questions about **{notes.get('title', 'the current lecture')}**. "
        "Answers are strictly grounded in retrieved evidence from the video lecture."
    )

    # Initialize RAG Pipeline for active lecture
    rag_pipeline = RAGPipeline(
        lecture_id=selected_lec_id,
        lecture_title=notes.get("title", "Active Lecture"),
    )

    # Ensure index exists
    if rag_pipeline.retriever is None and analysis:
        with st.spinner("Building semantic index for this lecture..."):
            rag_pipeline.index_lecture(analysis_data=analysis, notes_data=notes)

    # Chat history session state key unique to active lecture
    chat_key = f"chat_messages_{selected_lec_id}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    # Display conversation history
    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("🔍 View Supporting Lecture Evidence & Timestamps"):
                    for src in msg["sources"]:
                        st.markdown(
                            f"**Section {src['section_number']}: {src['section_title']}** "
                            f"`⏱ {src['timestamp_formatted']}` *(Similarity: {src['score']:.2f})*"
                        )
                        st.markdown(f"> *{src['snippet']}*")

    # Clear chat button
    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("🗑️ Clear Chat History", key=f"clear_{selected_lec_id}"):
            st.session_state[chat_key] = []
            st.rerun()

    # User question input
    user_query = st.chat_input("Ask a question about this lecture...")
    if user_query:
        # Append user message
        st.session_state[chat_key].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate grounded RAG response
        with st.chat_message("assistant"):
            with st.spinner("Searching lecture evidence and generating answer..."):
                rag_resp = rag_pipeline.ask(
                    query=user_query,
                    chat_history=st.session_state[chat_key],
                )

                st.markdown(rag_resp.answer)

                sources_data = []
                if rag_resp.sources:
                    with st.expander("🔍 View Supporting Lecture Evidence & Timestamps", expanded=True):
                        for src in rag_resp.sources:
                            st.markdown(
                                f"**Section {src.section_number}: {src.section_title}** "
                                f"`⏱ {src.timestamp_formatted}` *(Similarity: {src.score:.2f})*"
                            )
                            st.markdown(f"> *{src.snippet}*")
                            sources_data.append(src.model_dump())

                # Record in chat history
                st.session_state[chat_key].append(
                    {
                        "role": "assistant",
                        "content": rag_resp.answer,
                        "sources": sources_data,
                    }
                )


# =========================================================
# TAB 3: REVISION FLASHCARDS
# =========================================================
with tab_flashcards:
    st.header("🗂️ Revision Flashcards")
    st.caption("Active recall cards generated from key definitions and concepts in this lecture.")

    quiz_service = QuizService()
    flashcards = quiz_service.generate_flashcards(analysis_data=analysis, notes_data=notes)

    if not flashcards:
        st.info("No flashcard concepts available for this lecture.")
    else:
        st.write(f"Total Flashcards: **{len(flashcards)}**")
        for idx, card in enumerate(flashcards, start=1):
            with st.container():
                st.markdown(f"### Card {idx} — *{card.section_title}*")
                st.markdown(f"**Concept / Question:** {card.question}")
                show_ans = st.toggle("👁️ Reveal Answer", key=f"fc_toggle_{selected_lec_id}_{idx}")
                if show_ans:
                    st.info(f"**Definition / Explanation:**\n\n{card.answer}")
                st.divider()


# =========================================================
# TAB 4: SELF-ASSESSMENT QUIZ
# =========================================================
with tab_quiz:
    st.header("📝 Self-Assessment Quiz")
    st.caption("Test your comprehension of the concepts covered in this video lecture.")

    quiz_key = f"quiz_questions_{selected_lec_id}"
    if quiz_key not in st.session_state:
        with st.spinner("Preparing quiz questions..."):
            st.session_state[quiz_key] = quiz_service.generate_quiz(
                analysis_data=analysis,
                notes_data=notes,
                num_questions=4,
            )

    quiz_questions = st.session_state[quiz_key]
    user_selections = {}

    with st.form(key=f"quiz_form_{selected_lec_id}"):
        for q_idx, q in enumerate(quiz_questions, start=1):
            st.markdown(f"**Question {q_idx}:** {q.question}")
            st.caption(f"Topic: {q.section_title}")
            choice = st.radio(
                f"Select an option for Question {q_idx}:",
                options=list(range(len(q.options))),
                format_func=lambda i, opts=q.options: opts[i],
                key=f"q_radio_{selected_lec_id}_{q.id}",
                index=None,
            )
            user_selections[q.id] = choice
            st.write("")

        submitted = st.form_submit_button("Submit Quiz", type="primary")

    if submitted:
        grade_results = quiz_service.grade_quiz(user_selections, quiz_questions)
        st.subheader("Quiz Results")
        score = grade_results["score"]
        total = grade_results["total"]
        pct = grade_results["percentage"]

        if pct >= 75:
            st.success(f"🎉 Excellent! You scored **{score}/{total}** ({pct}%).")
        elif pct >= 50:
            st.warning(f"👍 Good effort! You scored **{score}/{total}** ({pct}%). Review the key concepts below.")
        else:
            st.error(f"📚 You scored **{score}/{total}** ({pct}%). Consider reviewing the study notes before retrying.")

        st.divider()
        st.markdown("### Question Breakdown & Explanations")
        for d in grade_results["details"]:
            is_c = d["is_correct"]
            icon = "✅" if is_c else "❌"
            st.markdown(f"**{icon} {d['question']}**")
            if not is_c:
                user_opt = (
                    quiz_questions[0].options[d["user_choice"]]
                    if d["user_choice"] is not None
                    else "None"
                )
                st.markdown(f"- *Your Answer:* `{user_opt}`")
            st.markdown(f"- *Explanation:* {d['explanation']}")
            st.write("")


# =========================================================
# TAB 5: EXPORT STUDY MATERIAL
# =========================================================
with tab_export:
    st.header("📥 Export Lecture Notes")
    st.caption("Download your comprehensive notes as Markdown or publication-quality PDF.")

    export_service = ExportService()

    col_md, col_pdf = st.columns(2)

    with col_md:
        st.subheader("Markdown Export")
        st.write("Clean Markdown with full section explanations, takeaways, and relative visual links.")
        md_file = BASE_DIR / "data" / "notes" / f"{selected_lec_id}_notes.md"
        export_service.export_markdown(notes, md_file)
        if md_file.exists():
            with open(md_file, "r", encoding="utf-8") as f:
                md_bytes = f.read().encode("utf-8")
            st.download_button(
                label="📄 Download Markdown (.md)",
                data=md_bytes,
                file_name=f"{selected_lec_id}_notes.md",
                mime="text/markdown",
                type="primary",
            )

    with col_pdf:
        st.subheader("PDF Export")
        st.write("Formatted PDF with cover typography, timestamp badges, embedded diagrams, and callout boxes.")
        pdf_file = BASE_DIR / "data" / "notes" / f"{selected_lec_id}_notes.pdf"
        try:
            export_service.export_pdf(notes, pdf_file, base_dir=BASE_DIR)
            if pdf_file.exists():
                with open(pdf_file, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="📕 Download PDF Document (.pdf)",
                    data=pdf_bytes,
                    file_name=f"{selected_lec_id}_notes.pdf",
                    mime="application/pdf",
                    type="primary",
                )
        except Exception as e:
            st.warning(f"Could not generate PDF: {e}")
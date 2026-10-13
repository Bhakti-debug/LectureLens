# 🎓 LectureLens
### An Intelligent Multimodal System for Study Note Generation & Question Answering from Educational Video Lectures

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/VectorStore-FAISS-green.svg)](https://github.com/facebookresearch/faiss)

LectureLens transforms educational YouTube lectures into structured, multimodal study materials and provides a lecture-grounded Retrieval-Augmented Generation (RAG) assistant. Rather than an unconstrained chatbot, LectureLens establishes a knowledge boundary around the active lecture, verifying claims against spoken explanations and presentation slides.

---

## 📌 Table of Contents
1. [Key Features](#-key-features)
2. [Architecture Overview](#-architecture-overview)
3. [System Requirements](#-system-requirements)
4. [Installation & Setup](#-installation--setup)
5. [Configuration (.env)](#-configuration-env)
6. [How to Run](#-how-to-run)
7. [Running Automated Tests](#-running-automated-tests)
8. [Research Evaluation Framework](#-research-evaluation-framework)
9. [Data Storage & Lecture Isolation](#-data-storage--lecture-isolation)
10. [Troubleshooting](#-troubleshooting)
11. [Limitations & Future Work](#-limitations--future-work)

---

## 🚀 Key Features

- **Multimodal Video Understanding:** Direct Gemini multimodal processing of educational YouTube videos, extracting spoken explanations, slides, code, diagrams, and audio-visual relationships.
- **Timestamp Integrity:** Pre-analysis duration probing via `yt-dlp`, prompt duration bounds, and post-analysis clamping ensure timestamps never exceed the real video length.
- **Physical Visual Extraction:** Automated temporary video download and OpenCV extraction of key slides, diagrams, and terminal demonstrations into high-resolution JPEG assets.
- **Lecture-Grounded RAG System:**
  - Topic-aware deterministic chunking preserving section titles and timestamps.
  - Local dense vector embeddings via `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional, normalized for cosine similarity).
  - Lecture-isolated FAISS (`faiss-cpu`) vector indexes with synchronized chunk metadata.
  - Semantic retrieval with passage deduplication and timestamped citations (`[MM:SS - MM:SS]`).
- **Relevance Guard & Boundary Protection:** Dual-phase evaluation distinguishing in-scope questions, insufficient evidence queries, and out-of-scope inquiries (e.g. cricket scores, geography).
- **Revision & Assessment Tools:**
  - Active-recall revision flashcards with interactive front/back reveal.
  - Multiple-choice quizzes with automated grading, scoring percentages, and explanations.
- **Publication-Quality Export:** Export notes to Markdown or styled PDF (using ReportLab) with cover headers, callout boxes, and embedded diagrams.

---

## 🏗️ Architecture Overview

LectureLens separates presentation, retrieval, model execution, and persistence:

```
LectureLens/
├── src/
│   ├── main.py                     # Streamlit web application & multi-tab UI
│   ├── video_analyzer.py           # Gemini multimodal analyzer & duration constraints
│   ├── visual_extractor.py         # OpenCV frame extraction & manifest generator
│   ├── note_generator.py           # Study note synthesizer & markdown converter
│   ├── gemini_client.py            # Gemini client factory
│   ├── rag/
│   │   ├── document_builder.py     # Section-to-document transformation
│   │   ├── chunker.py              # Sentence-preserving deterministic chunker
│   │   ├── embeddings.py           # Local SentenceTransformers (all-MiniLM-L6-v2)
│   │   ├── vector_store.py         # FAISS IndexFlatIP inner-product store
│   │   ├── retriever.py            # Semantic retrieval & source citation
│   │   ├── relevance.py            # Relevance guard & refusal handler
│   │   ├── answer_generator.py     # Grounded Gemini answer generator
│   │   └── pipeline.py             # End-to-end RAG coordinator
│   ├── services/
│   │   ├── lecture_service.py      # Multi-lecture lifecycle & legacy path sync
│   │   ├── quiz_service.py         # Flashcards and interactive quiz assessment
│   │   └── export_service.py       # Markdown and ReportLab PDF exporters
│   ├── models/
│   │   └── schemas.py              # Pydantic contracts & data models
│   ├── utils/
│   │   ├── config.py               # Centralized configuration & standard paths
│   │   ├── logging_config.py       # Secret-masked structured logging
│   │   └── validation.py           # URL parsing & timestamp normalization
│   └── evaluations/
│       └── evaluator.py            # Benchmark evaluation runner
├── data/
│   ├── analysis/                   # Active lecture analysis JSON
│   ├── notes/                      # Active study notes (JSON, MD, PDF)
│   ├── visuals/                    # Extracted slide images & visual_manifest.json
│   ├── indexes/                    # FAISS vector indexes (<lecture_id>/index.faiss)
│   ├── lectures/                   # Lecture archives (<lecture_id>/...)
│   └── evaluations/                # Benchmark JSON and CSV metrics
├── tests/                          # 23 automated tests
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
└── AGENTS.md                       # Persistent engineering rules
```

For more details, see [`docs/architecture.md`](docs/architecture.md).

---

## 💻 System Requirements

- **Operating System:** Windows 10/11, macOS, or Linux.
- **Python:** Python 3.10, 3.11, 3.12, or 3.13.
- **Memory (RAM):** 4 GB minimum (8 GB recommended for local embedding model).
- **Disk Space:** ~500 MB for Python packages and local embedding model cache.
- **Network:** Internet access for initial YouTube analysis, Gemini API calls, and downloading public video frames.

---

## 📦 Installation & Setup

### 1. Clone or Open the Repository
```bash
git clone https://github.com/your-username/LectureLens.git
cd LectureLens
```

### 2. Create and Activate a Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ Configuration (.env)

Copy `.env.example` to create your local `.env`:
```bash
cp .env.example .env
```

Open `.env` and configure your API key and preferences:
```env
# Gemini API Configuration
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash

# RAG & Local Embedding Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=400
CHUNK_OVERLAP=60
RETRIEVAL_TOP_K=4
RELEVANCE_SIMILARITY_THRESHOLD=0.35

# Application & Logging
LOG_LEVEL=INFO
```

> **Note:** Never commit `.env` to Git. The `.gitignore` file is pre-configured to ignore sensitive credentials.

---

## 🚀 How to Run

### Run the Streamlit Application
```bash
streamlit run src/main.py
```
Streamlit will launch locally (typically at `http://localhost:8501`).

### What You Can Do in the UI:
1. **Analyze a New Lecture:** Paste any public YouTube educational link in the expander and click **🚀 Analyze Lecture**.
2. **Browse Study Notes:** Read the lecture overview, chronological sections with timestamps, key concepts, code snippets, and extracted visual diagrams.
3. **Ask Questions (RAG):** Navigate to the **💬 Lecture Q&A Assistant** tab. Ask conceptual or specific questions; inspect supporting evidence passages and timestamps.
4. **Study with Flashcards:** Navigate to **🗂️ Revision Flashcards** to test memory recall with interactive question/answer reveal.
5. **Take a Quiz:** Navigate to **📝 Self-Assessment Quiz** to answer multiple-choice questions and receive score feedback with explanations.
6. **Export Notes:** Navigate to **📥 Export Study Material** to download clean `.md` or publication-ready `.pdf` documents.
7. **Switch Lectures:** Use the sidebar dropdown to switch between previously processed lectures without mixing indexes or chat history.

---

## 🧪 Running Automated Tests

LectureLens includes a comprehensive 23-test automated test suite covering URL parsing, timestamp integrity, chunking, embeddings, FAISS indexing, relevance guard boundaries, study tools, and PDF export:

```bash
pytest -v
```

**Test Breakdown:**
- `tests/test_validation.py`: URL validation, video ID extraction, timestamp clamping, negative timestamp removal.
- `tests/test_chunker.py`: Document building, sentence preservation, chunk overlap, determinism, unicode safety.
- `tests/test_embeddings_vectorstore.py`: Vector dimensions (384), L2 normalization, FAISS indexing, top-k ranking, lecture isolation.
- `tests/test_retriever_rag.py`: Semantic retrieval, citation timestamps, passage deduplication.
- `tests/test_relevance.py`: Out-of-scope fast-path refusal, in-scope acceptance.
- `tests/test_study_tools.py`: Flashcard generation, quiz scoring engine, percentage calculations.
- `tests/test_export.py`: Markdown and ReportLab PDF file generation without errors.
- `tests/test_regression.py`: JSON extraction, markdown fence cleaning, safe filename formatting, visual attachment compatibility.

---

## 📊 Research Evaluation Framework

Run the automated evaluation benchmark on the sample lecture:
```bash
python src/evaluations/evaluator.py
```

### Benchmark Metrics Produced:
- **Retrieval HitRate@4:** 1.0 (100% of target sections retrieved in top 4)
- **Retrieval MRR (Mean Reciprocal Rank):** 1.0 (Correct section ranked #1)
- **Retrieval Latency:** ~12.7 ms per query
- **Relevance Guard Accuracy:** 1.0 (Correctly classifies in-scope, insufficient-evidence, and out-of-scope queries)
- **Relevance Precision, Recall & F1:** 1.0

Reports are automatically saved to:
- `data/evaluations/evaluation_report.json`
- `data/evaluations/retrieval_metrics.csv`

---

## 🔒 Data Storage & Lecture Isolation

All artifacts are persisted in lecture-scoped directories:
- **Analysis:** `data/lectures/<lecture_id>/video_analysis.json`
- **Visuals:** `data/lectures/<lecture_id>/visuals/` and `visual_manifest.json`
- **Notes:** `data/lectures/<lecture_id>/study_notes.json`
- **Vector Indexes:** `data/indexes/<lecture_id>/index.faiss` and `metadata.json`

For backward compatibility with existing single-lecture scripts, the active lecture is automatically synchronized with `data/analysis/video_analysis.json`, `data/notes/study_notes.json`, and `data/visuals/visual_manifest.json`.

---

## 🛠️ Troubleshooting

1. **`ModuleNotFoundError: No module named 'faiss'`**
   - Ensure you installed packages using `.venv`:
     `python -m pip install -r requirements.txt`
2. **`Gemini API Error: 503 UNAVAILABLE or 429 RESOURCE_EXHAUSTED`**
   - Free-tier Gemini accounts have a limit of 5 requests/minute. The relevance guard automatically catches quota spikes and falls back to semantic threshold heuristics without crashing.
   - Set `GEMINI_MODEL=gemini-3.8-flash` in `.env` for the latest stable API endpoint.
3. **Missing Visual Frames**
   - If OpenCV reports missing images after moving directories, `resolve_image_path()` automatically searches the active visual directory and matches by filename.

---

## 🔮 Limitations & Future Work

- **Video Downloads:** YouTube frame extraction temporarily downloads video stream fragments to a secure temporary directory (`tempfile.TemporaryDirectory`) which is automatically deleted. Lectures requiring login or premium access cannot be downloaded.
- **Multimodal Vector Search:** Current retrieval uses dense text embeddings of transcribed speech, slide text, and visual descriptions. True multimodal CLIP/SigLIP image embedding search can be added as a future extension using the modular `EmbeddingService`.

# LectureLens Architecture Documentation

## 1. System Overview

**LectureLens** is a research-quality multimodal educational video understanding and study assistant. It takes a public educational YouTube lecture URL and produces:
1. Structured multimodal lecture understanding (spoken content, visual content, audio-visual relationships, and verified timestamps).
2. Physical visual assets extracted at verified frame timestamps.
3. Formatted study notes with explanations, key concepts, examples, and visual references.
4. An isolated lecture-grounded Retrieval-Augmented Generation (RAG) vector database.
5. Interactive Q&A with evidence citation and strict lecture boundary enforcement.
6. Revision tools including active-recall flashcards, interactive quizzes, and Markdown/PDF document export.

---

## 2. Component Architecture

```
LectureLens/
├── src/
│   ├── main.py                     # Streamlit multimodal UI & view controllers
│   ├── video_analyzer.py           # Gemini multimodal video analyzer & duration constraint
│   ├── visual_extractor.py         # OpenCV frame extraction & visual manifest generator
│   ├── note_generator.py           # Study note synthesizer & markdown converter
│   ├── gemini_client.py            # Gemini client factory & model selector
│   ├── rag/
│   │   ├── document_builder.py     # Normalized lecture document construction
│   │   ├── chunker.py              # Topic-aware deterministic sentence-preserving chunker
│   │   ├── embeddings.py           # Local Sentence Transformers (all-MiniLM-L6-v2) singleton
│   │   ├── vector_store.py         # FAISS IndexFlatIP inner-product store with chunk metadata
│   │   ├── retriever.py            # Semantic retrieval, passage deduplication, citation formatting
│   │   ├── relevance.py            # Dual-phase relevance guard for lecture boundary enforcement
│   │   ├── answer_generator.py     # Grounded response synthesis with evidence constraints
│   │   └── pipeline.py             # High-level RAG orchestrator
│   ├── services/
│   │   ├── lecture_service.py      # Multi-lecture lifecycle, stable IDs, legacy sync
│   │   ├── quiz_service.py         # Flashcards and interactive multiple-choice quiz generator
│   │   └── export_service.py       # Publication-quality Markdown and ReportLab PDF exporters
│   ├── models/
│   │   └── schemas.py              # Pydantic data models and contracts
│   ├── utils/
│   │   ├── config.py               # Typed centralized configuration and path resolution
│   │   ├── logging_config.py       # Privacy-safe structured logging with secret masking
│   │   └── validation.py           # URL parsing and duration/timestamp normalization
│   └── evaluations/
│       └── evaluator.py            # Benchmark evaluation runner (HitRate@k, MRR, Precision, Recall, F1)
├── data/
│   ├── analysis/                   # Single-lecture active analysis mirror
│   ├── visuals/                    # Extracted visual frames and visual_manifest.json
│   ├── notes/                      # Single-lecture active notes and exported documents
│   ├── indexes/                    # FAISS vector indexes (<lecture_id>/index.faiss)
│   ├── lectures/                   # Isolated multi-lecture archives (<lecture_id>/...)
│   └── evaluations/                # Benchmark reports and retrieval metrics
├── tests/                          # 23 automated unit, integration, and regression tests
├── docs/                           # Architectural specifications
├── requirements.txt                # Complete dependency manifest
├── README.md                       # Installation and operational documentation
└── AGENTS.md                       # Persistent engineering principles
```

---

## 3. Data Flow Diagram

```
YouTube Video URL
       │
       ▼
[yt-dlp Metadata Probe] ──────► Verified Duration (seconds)
       │
       ▼
[Gemini Multimodal Understanding] ───► Raw Analysis JSON
       │
       ▼
[Timestamp Integrity Normalizer] ───► Clamped & Normalized video_analysis.json
       │
       ├──────────────────────────────────────────────┐
       ▼                                              ▼
[OpenCV Frame Extractor]                      [Document Builder]
 (Temporary Video Download in isolated tempdir)       │
       │                                              ▼
       ▼                                      [Topic-Aware Chunker]
 visual_manifest.json + JPEG frames                   │
       │                                              ▼
       ▼                                      [SentenceTransformer Embeddings]
[Note Generator & Visual Attachment]                  │
       │                                              ▼
       ▼                                      [FAISS Vector Store]
 study_notes.json + study_notes.md             (index.faiss + metadata.json)
       │
       ├───────────────────────────────┬───────────────────────────────┐
       ▼                               ▼                               ▼
[Interactive UI Viewer]       [Study & Quiz Tools]            [RAG Q&A Assistant]
 - Sections                    - Concept Flashcards            - Relevance Guard Check
 - Slide Visuals               - 4-Option Quizzes              - Semantic Retrieval
 - Key Concepts                - Scoring Engine                - Grounded Gemini Answer
 - PDF/MD Export                                               - Section & Timestamp Citations
```

---

## 4. Key Design Decisions

### 4.1 Local Embeddings vs. Paid APIs
- **Decision:** Employed `sentence-transformers/all-MiniLM-L6-v2` locally via CPU/GPU.
- **Rationale:** 
  1. Completely eliminates query-time API costs and quota exhaustion for vector retrieval.
  2. Sub-15ms retrieval latency on standard consumer hardware.
  3. 384-dimensional dense vectors provide excellent semantic match for technical lecture queries.
  4. L2 normalization ensures inner product equals cosine similarity for `faiss.IndexFlatIP`.

### 4.2 Lecture Isolation
- **Decision:** Separate folder per lecture under `data/lectures/<lecture_id>/` and separate FAISS index under `data/indexes/<lecture_id>/`.
- **Rationale:** 
  1. Prevents cross-contamination between lectures (e.g. asking about Node.js will never retrieve Python chunks).
  2. Switching lectures in the UI seamlessly switches active vector indexes and clears conversation history.
  3. Legacy single-lecture paths (`data/analysis/video_analysis.json`, `data/notes/study_notes.json`) are maintained via atomic synchronization.

### 4.3 Timestamp Integrity & Boundary Protection
- **Decision:** Dual-layer duration verification.
- **Rationale:**
  1. `yt-dlp` extracts the true video duration before Gemini analysis.
  2. The prompt explicitly informs Gemini of the verified duration.
  3. Post-analysis verification normalizes all timestamps: clamping any overshoots, enforcing `0 <= start_time < end_time <= duration`, and discarding corrupt visual frames without failing the pipeline.
  4. OpenCV capture releases resources inside a `try ... finally` block.

### 4.4 Relevance Guard Architecture
- **Decision:** Dual-phase evaluation (semantic similarity distance check + lightweight LLM category verification).
- **Rationale:**
  1. Distinguishes three states: `IN_SCOPE`, `INSUFFICIENT_EVIDENCE`, and `OUT_OF_SCOPE`.
  2. Clearly refuses off-topic queries (cricket, general geography, recipes) with polite explanations.
  3. If rate-limits or network interruptions occur, safely falls back to threshold-based topic heuristics to maintain system availability.

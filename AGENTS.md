# AGENTS.md — Persistent Engineering Rules for LectureLens

## 1. Core Principles
- **Preserve Existing Functionality:** LectureLens has an established video-to-notes processing foundation. Never replace working code with untested stubs. Preserve existing public function signatures and JSON contracts.
- **Verification First:** Never claim that code works without executing automated tests or verifying behavior.
- **Research Integrity:** Never fabricate evaluation metrics, citation timestamps, or test outputs.

## 2. Architecture & Modular Boundaries
- Keep presentation logic strictly in `src/main.py` (Streamlit).
- Keep retrieval and vector operations in `src/rag/`.
- Keep business services (lectures, quizzes, exports) in `src/services/`.
- Keep configuration in `src/utils/config.py` and data contracts in `src/models/schemas.py`.
- Keep vector indexes logically isolated per `lecture_id` under `data/indexes/<lecture_id>/`.
- Maintain legacy single-lecture compatibility under `data/analysis/`, `data/notes/`, and `data/visuals/`.

## 3. Security & Secret Handling
- Never hardcode or log API keys or secrets.
- Always use `utils.logging_config.get_logger`, which masks keys using `SensitiveDataFilter`.
- Keep `.env` strictly ignored by Git. All new settings must be reflected with placeholder values in `.env.example`.

## 4. Dependencies & Runtime
- Use `.venv\Scripts\python.exe` on Windows for all package management and script execution.
- Maintain `requirements.txt` with compatible versions.
- Do not introduce paid embedding APIs; local Sentence Transformers (`all-MiniLM-L6-v2`) and FAISS (`faiss-cpu`) are the architectural standard.

## 5. Testing & Evaluation Mandate
- Every new feature must be accompanied by unit or integration tests in `tests/`.
- Run `pytest` before finalizing any change. All tests must pass.
- Use `python src/evaluations/evaluator.py` to assess retrieval HitRate@k, MRR, and relevance classification metrics against the benchmark dataset.

"""
LectureLens - Revision & Self-Assessment Service
Generates structured revision flashcards and grounded multiple-choice quizzes
from lecture analysis and notes.
"""

import json
import re
from typing import Any, Dict, List, Optional

from gemini_client import (
    create_gemini_client,
    get_model_name,
    generate_content_with_retry,
)
from models.schemas import Flashcard, QuizQuestion
from utils.logging_config import get_logger

logger = get_logger("quiz_service")

QUIZ_PROMPT_TEMPLATE = """
You are an expert educational assessment designer for LectureLens.
Create {num_questions} high-quality, conceptual multiple-choice quiz questions based EXCLUSIVELY on the provided lecture study material.

Lecture Title: "{lecture_title}"
Lecture Material:
{lecture_content}

REQUIREMENTS:
1. Every question must test a meaningful concept, architectural detail, or definition from the lecture.
2. Provide exactly 4 plausible options for each question.
3. Indicate the zero-based index of the correct answer (0, 1, 2, or 3).
4. Provide a clear, educational explanation grounded in the lecture.
5. Identify the relevant section and timestamp if mentioned.

Return ONLY valid JSON matching this structure:
{{
  "questions": [
    {{
      "id": "q1",
      "question": "What is ...?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_index": 0,
      "explanation": "Why this is correct based on the lecture.",
      "section_title": "Section Title",
      "timestamp": 120.0
    }}
  ]
}}
"""


class QuizService:
    """Generates revision flashcards and self-assessment quizzes."""

    def __init__(self):
        pass

    def generate_flashcards(
        self,
        analysis_data: Dict[str, Any],
        notes_data: Optional[Dict[str, Any]] = None,
    ) -> List[Flashcard]:
        """
        Generate revision flashcards deterministically from key concepts, definitions,
        and takeaways in the lecture.
        """
        flashcards: List[Flashcard] = []
        seen_concepts = set()
        card_id = 1

        sections = analysis_data.get("sections", [])
        notes_sections = (notes_data.get("sections") if notes_data else None) or []
        notes_by_sec = {s.get("section_number"): s for s in notes_sections}

        for sec_num, sec in enumerate(sections, start=1):
            title = sec.get("title", f"Section {sec_num}")
            start_time = float(sec.get("start_time", 0.0))
            key_concepts = sec.get("key_concepts", [])
            spoken_content = sec.get("spoken_content", [])
            note_sec = notes_by_sec.get(sec_num, {})

            # 1. Concept cards
            for concept in key_concepts:
                c_str = str(concept).strip()
                c_norm = c_str.lower()
                if not c_str or c_norm in seen_concepts:
                    continue
                seen_concepts.add(c_norm)

                # Match relevant explanation snippet
                answer = ""
                for pt in spoken_content:
                    if c_str.lower() in str(pt).lower():
                        answer = str(pt).strip()
                        break
                if not answer and note_sec.get("explanation"):
                    answer = note_sec.get("explanation")[:250] + "..."
                if not answer:
                    answer = f"Core concept introduced in {title}."

                flashcards.append(
                    Flashcard(
                        id=f"fc_{card_id:02d}",
                        question=f"What is the role or significance of **{c_str}** in this lecture?",
                        answer=answer,
                        section_title=title,
                        timestamp=start_time,
                    )
                )
                card_id += 1

            # 2. Key takeaway card
            takeaway = note_sec.get("key_takeaway")
            if takeaway:
                flashcards.append(
                    Flashcard(
                        id=f"fc_{card_id:02d}",
                        question=f"What is the main takeaway of '{title}'?",
                        answer=takeaway,
                        section_title=title,
                        timestamp=start_time,
                    )
                )
                card_id += 1

        return flashcards

    def generate_quiz(
        self,
        analysis_data: Dict[str, Any],
        notes_data: Optional[Dict[str, Any]] = None,
        num_questions: int = 5,
    ) -> List[QuizQuestion]:
        """
        Generate multiple-choice quiz questions using Gemini grounded in the lecture.
        Falls back to rule-based questions if the API is unavailable.
        """
        title = (notes_data.get("title") if notes_data else None) or "Lecture Assessment"

        # Build concise lecture context for the prompt
        content_lines = []
        sections = analysis_data.get("sections", [])
        for s in sections[:6]:
            content_lines.append(f"Section: {s.get('title')} ({s.get('start_time')}s)")
            content_lines.append("Spoken: " + " ".join(s.get("spoken_content", [])[:3]))
            if s.get("key_concepts"):
                content_lines.append("Key Concepts: " + ", ".join(s.get("key_concepts", [])))
            content_lines.append("")

        lecture_content = "\n".join(content_lines)

        try:
            client = create_gemini_client()
            model = get_model_name()
            prompt = QUIZ_PROMPT_TEMPLATE.format(
                num_questions=num_questions,
                lecture_title=title,
                lecture_content=lecture_content,
            )

            response = generate_content_with_retry(
                client=client,
                preferred_model=model,
                contents=prompt,
            )

            resp_text = response.text or ""
            cleaned = re.sub(r"^```json\s*", "", resp_text.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r"^```\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()

            parsed = json.loads(cleaned)
            raw_questions = parsed.get("questions", [])

            questions: List[QuizQuestion] = []
            for q in raw_questions:
                if not isinstance(q, dict):
                    continue
                opts = q.get("options", [])
                c_idx = int(q.get("correct_index", 0))
                if len(opts) < 2 or c_idx >= len(opts):
                    continue

                questions.append(
                    QuizQuestion(
                        id=q.get("id", f"q_{len(questions)+1}"),
                        question=q.get("question", ""),
                        options=opts,
                        correct_index=c_idx,
                        explanation=q.get("explanation", ""),
                        section_title=q.get("section_title", "Lecture Topic"),
                        timestamp=float(q.get("timestamp", 0.0)),
                    )
                )

            if questions:
                return questions

        except Exception as e:
            logger.warning(f"Gemini quiz generation failed: {e}. Generating fallback quiz.")

        # Fallback rule-based quiz generator
        return self._generate_fallback_quiz(analysis_data, num_questions)

    def _generate_fallback_quiz(
        self,
        analysis_data: Dict[str, Any],
        num_questions: int = 4,
    ) -> List[QuizQuestion]:
        """Generate reliable deterministic quiz questions from concepts."""
        sections = analysis_data.get("sections", [])
        questions: List[QuizQuestion] = []
        q_id = 1

        for sec in sections:
            title = sec.get("title", "Lecture Topic")
            concepts = sec.get("key_concepts", [])
            start_time = float(sec.get("start_time", 0.0))

            if len(concepts) >= 2:
                target_c = concepts[0]
                distractor1 = concepts[1]
                questions.append(
                    QuizQuestion(
                        id=f"q_{q_id}",
                        question=f"Which core concept is central to the topic '{title}'?",
                        options=[
                            target_c,
                            f"Not related to {target_c}",
                            distractor1,
                            "None of the above",
                        ],
                        correct_index=0,
                        explanation=f"'{target_c}' is explicitly highlighted in the lecture under '{title}'.",
                        section_title=title,
                        timestamp=start_time,
                    )
                )
                q_id += 1
                if len(questions) >= num_questions:
                    break

        return questions

    def grade_quiz(
        self,
        user_answers: Dict[str, int],
        quiz_questions: List[QuizQuestion],
    ) -> Dict[str, Any]:
        """
        Grade submitted answers and return detailed performance metrics.
        """
        total = len(quiz_questions)
        if total == 0:
            return {"score": 0, "total": 0, "percentage": 0.0, "details": []}

        correct_count = 0
        details = []

        for q in quiz_questions:
            user_choice = user_answers.get(q.id)
            is_correct = user_choice is not None and user_choice == q.correct_index
            if is_correct:
                correct_count += 1

            details.append(
                {
                    "question_id": q.id,
                    "question": q.question,
                    "user_choice": user_choice,
                    "correct_index": q.correct_index,
                    "is_correct": is_correct,
                    "explanation": q.explanation,
                    "section_title": q.section_title,
                }
            )

        percentage = round((correct_count / total) * 100, 1)

        return {
            "score": correct_count,
            "total": total,
            "percentage": percentage,
            "details": details,
        }

"""
Tests for revision flashcards and interactive quiz evaluation.
"""

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from models.schemas import QuizQuestion
from services.quiz_service import QuizService


def test_flashcard_generation():
    """Verify flashcards are created with questions, answers, and timestamps without duplication."""
    service = QuizService()

    sample_analysis = {
        "video_summary": "Lecture on asynchronous JavaScript.",
        "sections": [
            {
                "title": "Asynchronous Programming",
                "start_time": 45.0,
                "end_time": 120.0,
                "spoken_content": [
                    "Event Loop handles non-blocking operations in the background.",
                ],
                "key_concepts": ["Event Loop", "Non-blocking I/O", "Event Loop"],  # Duplicate concept
            }
        ],
    }

    cards = service.generate_flashcards(sample_analysis)
    assert len(cards) >= 2
    # Ensure no duplicates for 'Event Loop'
    concept_names = [c.question for c in cards]
    assert len(concept_names) == len(set(concept_names))

    for card in cards:
        assert card.question
        assert card.answer
        assert card.section_title == "Asynchronous Programming"
        assert card.timestamp == 45.0


def test_quiz_grading_evaluation():
    """Verify grading calculation, percentages, and feedback."""
    service = QuizService()

    questions = [
        QuizQuestion(
            id="q1",
            question="What is V8?",
            options=["A car", "A JS Engine", "A framework", "A database"],
            correct_index=1,
            explanation="V8 is Google's JS engine.",
            section_title="Engines",
            timestamp=20.0,
        ),
        QuizQuestion(
            id="q2",
            question="Who created Node.js?",
            options=["Guido van Rossum", "Brendan Eich", "Ryan Dahl", "Dennis Ritchie"],
            correct_index=2,
            explanation="Ryan Dahl created Node.js.",
            section_title="History",
            timestamp=50.0,
        ),
    ]

    # User answers: q1 correct (1), q2 incorrect (0)
    user_answers = {"q1": 1, "q2": 0}
    result = service.grade_quiz(user_answers, questions)

    assert result["total"] == 2
    assert result["score"] == 1
    assert result["percentage"] == 50.0

    details = result["details"]
    assert details[0]["is_correct"] is True
    assert details[1]["is_correct"] is False
    assert details[1]["correct_index"] == 2

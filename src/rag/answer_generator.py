"""
LectureLens - Grounded Answer Generator
Synthesizes evidence-grounded answers using retrieved lecture passages,
strictly adhering to lecture boundaries and honest evidence attribution.
"""

from typing import Any, Dict, List, Optional

from gemini_client import (
    create_gemini_client,
    get_model_name,
    generate_content_with_retry,
)
from models.schemas import RetrievalResult
from utils.logging_config import get_logger
from utils.validation import format_timestamp

logger = get_logger("answer_generator")

ANSWER_PROMPT_TEMPLATE = """
You are the LectureLens AI Study Assistant for the lecture: "{lecture_title}".

A student is asking a question about this lecture.
Your task is to provide an accurate, clear, and educational answer GROUNDED in the provided lecture evidence passages.

STUDENT QUESTION:
"{question}"

{conversation_context}

RETRIEVED LECTURE EVIDENCE PASSAGES:
{evidence_text}

STRICT GROUNDING GUIDELINES:
1. Prioritize the retrieved lecture passages as your primary source of truth.
2. Ground your explanations in what the instructor presented and demonstrated.
3. Explicitly cite section names or timestamps when referencing specific parts (e.g., "[Understanding JavaScript & JS Engines, 00:35]").
4. If the retrieved evidence is INSUFFICIENT to answer the question, clearly state:
   "Based on the provided lecture material, this specific question is not covered in detail."
   Then, if appropriate, you may provide a brief general explanation, but clearly label it as:
   "*(Note: The following is supplementary conceptual context not explicitly covered in this lecture.)*"
5. Do NOT fabricate instructor statements, timestamps, or quotes.
6. Write in an encouraging, clear, educational tone suitable for a university student.
"""


class AnswerGenerator:
    """Generates grounded answers from retrieved lecture context."""

    def __init__(self):
        self.client = create_gemini_client()
        self.model = get_model_name()

    def generate_answer(
        self,
        question: str,
        lecture_title: str,
        retrieved_results: List[RetrievalResult],
        chat_history: Optional[List[Dict[str, str]]] = None,
        is_insufficient_evidence: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate grounded answer text with citations.

        Returns dict:
            {
                "answer": str,
                "is_sufficient": bool,
            }
        """
        if not retrieved_results and not is_insufficient_evidence:
            return {
                "answer": (
                    f"I could not find any relevant sections in the lecture **{lecture_title}** "
                    f"to answer this question."
                ),
                "is_sufficient": False,
            }

        # Build evidence text
        evidence_blocks = []
        for i, res in enumerate(retrieved_results, start=1):
            chunk = res.chunk
            ts_str = f"{format_timestamp(chunk.start_time)} - {format_timestamp(chunk.end_time)}"
            block = (
                f"--- EVIDENCE {i} ---\n"
                f"Section: {chunk.section_title} [Time: {ts_str}]\n"
                f"Content:\n{chunk.text}\n"
            )
            evidence_blocks.append(block)

        evidence_text = "\n".join(evidence_blocks) if evidence_blocks else "No direct evidence passages found."

        # Format conversation context
        conv_context = ""
        if chat_history and len(chat_history) > 0:
            conv_lines = ["RECENT CONVERSATION HISTORY:"]
            for msg in chat_history[-4:]:
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                conv_lines.append(f"{role}: {content}")
            conv_context = "\n".join(conv_lines) + "\n"

        prompt = ANSWER_PROMPT_TEMPLATE.format(
            lecture_title=lecture_title,
            question=question,
            conversation_context=conv_context,
            evidence_text=evidence_text,
        )

        if is_insufficient_evidence:
            prompt += (
                "\nADDITIONAL INSTRUCTION: The user's question relates to the general field "
                "but is NOT thoroughly addressed in the retrieved lecture evidence. "
                "Acknowledge this clearly at the beginning of your response."
            )

        try:
            response = generate_content_with_retry(
                client=self.client,
                preferred_model=self.model,
                contents=prompt,
            )
            answer_text = response.text or ""
            return {
                "answer": answer_text.strip(),
                "is_sufficient": not is_insufficient_evidence,
            }
        except Exception as e:
            logger.error(f"Error during Gemini answer generation: {e}")
            raise RuntimeError(f"Failed to generate answer from Gemini: {e}") from e

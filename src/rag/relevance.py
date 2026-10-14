"""
LectureLens - Relevance Guard
Evaluates student inquiries against the active lecture scope, distinguishing
in-scope questions, conceptual follow-ups, insufficient evidence, and out-of-scope queries.
"""

import json
import re
from typing import Any, Dict, List, Optional
from google.genai import types

from gemini_client import (
    create_gemini_client,
    get_model_name,
    generate_content_with_retry,
)
from models.schemas import RetrievalResult
from utils.config import get_config
from utils.logging_config import get_logger

logger = get_logger("relevance")

RELEVANCE_PROMPT = """
You are the Lecture Boundary and Relevance Guard for LectureLens.
Your job is to determine whether a student's question is relevant to the current educational lecture.

Active Lecture: "{lecture_title}"
Lecture Overview: "{lecture_overview}"
Key Topics Covered: {key_topics}

Top Retrieved Passages from Lecture:
{retrieved_context}

Question: "{question}"

Instructions:
1. IN_SCOPE: The question asks about concepts, code, diagrams, explanations, or reasonable clarifications/examples directly related to the lecture topic.
2. INSUFFICIENT_EVIDENCE: The question is related to the subject matter of the lecture, but the specific lecture content/passages provided do not contain enough detail to fully answer it.
3. OUT_OF_SCOPE: The question is about completely unrelated subjects (e.g., sports scores, politics, cooking, celebrity gossip, world capitals, unrelated academic fields).

Return ONLY valid JSON in this format:
{{
  "is_relevant": true,
  "category": "in_scope",
  "reason": "Brief explanation of the decision."
}}
Allowed categories: "in_scope", "insufficient_evidence", "out_of_scope".
If category is "out_of_scope", "is_relevant" must be false.
"""


class RelevanceGuard:
    """Detects lecture relevance and handles unrelated queries."""

    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
    ):
        cfg = get_config()
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else cfg.relevance_similarity_threshold
        )

    def evaluate_query(
        self,
        question: str,
        lecture_title: str,
        lecture_overview: str,
        key_topics: List[str],
        retrieved_results: List[RetrievalResult],
    ) -> Dict[str, Any]:
        """
        Evaluate if the query is in-scope, insufficient evidence, or out-of-scope.

        Returns dict:
            {
                "is_relevant": bool,
                "category": str ("in_scope" | "insufficient_evidence" | "out_of_scope"),
                "reason": str,
                "refusal_message": Optional[str],
            }
        """
        clean_q = question.strip()
        if not clean_q:
            return {
                "is_relevant": False,
                "category": "out_of_scope",
                "reason": "Empty question.",
                "refusal_message": "Please enter a valid question about the lecture.",
            }

        max_score = (
            max([r.score for r in retrieved_results])
            if retrieved_results
            else 0.0
        )

        # Extreme threshold shortcut: if score is very low and topic keywords are absent, fast-path
        if max_score < 0.12 and not any(
            topic.lower() in clean_q.lower() for topic in key_topics if len(topic) > 3
        ):
            return {
                "is_relevant": False,
                "category": "out_of_scope",
                "reason": f"Semantic similarity ({max_score:.3f}) is below minimal relevance boundary.",
                "refusal_message": (
                    f"This question appears to be outside the scope of the current lecture ('{lecture_title}'). "
                    f"Please ask a question related to this lecture."
                ),
            }

        # Format context for LLM decision
        context_snippets = []
        for r in retrieved_results[:3]:
            context_snippets.append(f"[{r.chunk.section_title}]: {r.chunk.text[:200]}...")
        context_str = "\n".join(context_snippets) or "No relevant passages found."

        prompt = RELEVANCE_PROMPT.format(
            lecture_title=lecture_title,
            lecture_overview=lecture_overview[:400],
            key_topics=", ".join(key_topics[:10]),
            retrieved_context=context_str,
            question=clean_q,
        )

        try:
            client = create_gemini_client()
            model = get_model_name()
            response = generate_content_with_retry(
                client=client,
                preferred_model=model,
                contents=prompt,
            )

            resp_text = response.text or ""
            # Clean JSON fences
            cleaned = re.sub(r"^```json\s*", "", resp_text.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r"^```\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()

            decision = json.loads(cleaned)
            category = decision.get("category", "in_scope").lower()
            is_relevant = decision.get("is_relevant", True)
            reason = decision.get("reason", "")

            if category == "out_of_scope" or not is_relevant:
                return {
                    "is_relevant": False,
                    "category": "out_of_scope",
                    "reason": reason or "Query is not related to the active lecture.",
                    "refusal_message": (
                        f"I can only answer questions related to the active lecture: **{lecture_title}**. "
                        f"Your question appears to be about an unrelated topic. "
                        f"Please feel free to ask about any concepts covered in this video."
                    ),
                }

            if category == "insufficient_evidence":
                return {
                    "is_relevant": True,
                    "category": "insufficient_evidence",
                    "reason": reason or "Question is topic-relevant, but not covered in the lecture.",
                    "refusal_message": None,
                }

            return {
                "is_relevant": True,
                "category": "in_scope",
                "reason": reason or "Question is directly covered by the lecture.",
                "refusal_message": None,
            }

        except Exception as e:
            logger.warning(f"LLM relevance check encountered exception: {e}. Falling back to threshold heuristic.")
            # Graceful fallback: check if max_score >= threshold or key topic overlap
            is_score_ok = max_score >= self.similarity_threshold
            has_topic_match = any(
                topic.lower() in clean_q.lower() for topic in key_topics if len(topic) > 3
            )
            is_rel = is_score_ok or has_topic_match

            if not is_rel:
                return {
                    "is_relevant": False,
                    "category": "out_of_scope",
                    "reason": f"Fallback: semantic score ({max_score:.2f}) < threshold ({self.similarity_threshold}).",
                    "refusal_message": (
                        f"This question appears to be outside the scope of **{lecture_title}**. "
                        f"Please ask questions related to this lecture."
                    ),
                }

            return {
                "is_relevant": True,
                "category": "in_scope" if is_score_ok else "insufficient_evidence",
                "reason": f"Fallback heuristic accepted (score: {max_score:.2f}).",
                "refusal_message": None,
            }

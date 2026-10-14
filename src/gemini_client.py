"""
LectureLens - Gemini Client & Resilient Model Execution

Centralized access to the Google Gemini GenAI SDK client, with automatic
retry and graceful fallback across available models to protect against
quota exhaustion (429 RESOURCE_EXHAUSTED) and temporary service spikes (503).
"""

import os
import time
from typing import Any, List, Optional
from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL
from utils.logging_config import get_logger

logger = get_logger("gemini_client")

# Candidate models in preference order for failover (text generation only)
DEFAULT_FALLBACK_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
]


def create_gemini_client() -> genai.Client:
    """Create and return an authenticated GenAI client."""
    return genai.Client(api_key=GEMINI_API_KEY)


def get_model_name() -> str:
    """Return the configured primary Gemini model name."""
    return os.getenv("GEMINI_MODEL", GEMINI_MODEL or "gemini-3-flash-preview")


def generate_content_with_retry(
    client: genai.Client,
    contents: Any,
    preferred_model: Optional[str] = None,
    candidate_models: Optional[List[str]] = None,
    max_retries: int = 2,
    **kwargs,
) -> Any:
    """
    Generate content with automatic backoff and model failover.
    Protects against 429 RESOURCE_EXHAUSTED and 503 SERVICE_UNAVAILABLE.
    """
    primary = preferred_model or get_model_name()
    if candidate_models is None:
        chain = [primary] + [m for m in DEFAULT_FALLBACK_MODELS if m != primary]
    else:
        chain = [primary] + [m for m in candidate_models if m != primary]

    last_exc = None
    for model_name in chain:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    **kwargs,
                )
                if model_name != primary:
                    logger.info(f"Successfully generated content using failover model: {model_name}")
                return response
            except Exception as exc:
                last_exc = exc
                err_msg = str(exc)
                is_quota = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg
                is_transient = "503" in err_msg or "UNAVAILABLE" in err_msg or "overloaded" in err_msg.lower()
                is_not_found = "404" in err_msg or "NOT_FOUND" in err_msg

                if is_quota:
                    logger.warning(
                        f"Model {model_name} quota exhausted (429). Switching to next available candidate..."
                    )
                    break  # Break retry loop for this model; move immediately to next candidate
                elif is_not_found:
                    logger.warning(
                        f"Model {model_name} not found (404). Switching to next available candidate..."
                    )
                    break  # Skip unavailable model
                elif is_transient:
                    logger.warning(
                        f"Model {model_name} transient error (attempt {attempt + 1}/{max_retries}): {exc}. Retrying..."
                    )
                    time.sleep(1.5 * (attempt + 1))
                else:
                    # Non-retriable exception (e.g. invalid arguments, auth error)
                    raise exc

    if last_exc:
        raise last_exc
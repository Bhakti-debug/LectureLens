"""
LectureLens - Validation and Timestamp Integrity Utility
Provides URL validation, metadata extraction without full video download,
and rigorous timestamp verification against verified lecture duration.
"""

import re
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from utils.logging_config import get_logger

logger = get_logger("validation")

YOUTUBE_URL_PATTERNS = [
    re.compile(r"^(https?://)?(www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})"),
    re.compile(r"^(https?://)?(www\.)?youtu\.be/([a-zA-Z0-9_-]{11})"),
    re.compile(r"^(https?://)?(www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})"),
    re.compile(r"^(https?://)?(www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})"),
    re.compile(r"^(https?://)?(www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})"),
]


def extract_youtube_video_id(url: str) -> Optional[str]:
    """
    Extract the canonical 11-character YouTube video ID from various URL formats.
    """
    if not url:
        return None

    cleaned_url = url.strip()

    for pattern in YOUTUBE_URL_PATTERNS:
        match = pattern.match(cleaned_url)
        if match:
            return match.group(3)

    try:
        parsed = urlparse(cleaned_url)
        if parsed.hostname in ("www.youtube.com", "youtube.com"):
            qs = parse_qs(parsed.query)
            if "v" in qs and qs["v"]:
                candidate = qs["v"][0]
                if len(candidate) == 11:
                    return candidate
        elif parsed.hostname == "youtu.be":
            candidate = parsed.path.lstrip("/").split("?")[0]
            if len(candidate) == 11:
                return candidate
    except Exception:
        pass

    return None


def validate_youtube_url(url: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate a YouTube URL and return a normalized canonical URL and video ID.

    Returns:
        (is_valid, message, normalized_url)
    """
    if not url or not url.strip():
        return False, "YouTube URL cannot be empty.", None

    url = url.strip()
    video_id = extract_youtube_video_id(url)
    if not video_id:
        return (
            False,
            "Invalid YouTube URL. Please provide a standard YouTube video link (e.g., https://www.youtube.com/watch?v=...)",
            None,
        )

    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    return True, "Valid YouTube URL", canonical_url


def fetch_video_metadata(url: str) -> Dict[str, Any]:
    """
    Extract video metadata (duration, title, channel) using yt-dlp without downloading the video.
    """
    import yt_dlp

    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "socket_timeout": 15,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                return {"error": "Could not extract video info"}

            return {
                "id": info.get("id"),
                "title": info.get("title", "Educational Lecture"),
                "duration": float(info.get("duration", 0.0) or 0.0),
                "channel": info.get("uploader", ""),
                "description": info.get("description", ""),
            }
    except Exception as e:
        err_msg = str(e)
        logger.warning(f"Could not extract online video metadata: {err_msg}")
        return {"error": err_msg}


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS."""
    try:
        secs = int(round(float(seconds or 0)))
    except (ValueError, TypeError):
        return "00:00"

    secs = max(0, secs)
    hours = secs // 3600
    minutes = (secs % 3600) // 60
    rem_secs = secs % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{rem_secs:02d}"
    return f"{minutes:02d}:{rem_secs:02d}"


def validate_and_normalize_timestamps(
    analysis_data: Dict[str, Any],
    actual_duration: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Validate and clamp timestamps in Gemini analysis data against the verified video duration.

    - Ensures 0 <= start_time < end_time <= actual_duration.
    - Clamps overshoot timestamps.
    - Filters out negative or invalid visual timestamps.
    - Logs diagnostic information for any detected anomalies.
    """
    sections = analysis_data.get("sections", [])
    if not isinstance(sections, list):
        return analysis_data

    # If duration is not supplied, infer conservative max from section end_times
    max_observed_time = 0.0
    for section in sections:
        try:
            sec_end = float(section.get("end_time", 0.0))
            if sec_end > max_observed_time:
                max_observed_time = sec_end
        except (ValueError, TypeError):
            pass

    reference_duration = actual_duration if actual_duration and actual_duration > 0 else max_observed_time

    normalized_sections = []

    for i, section in enumerate(sections, start=1):
        try:
            start_time = float(section.get("start_time", 0.0))
        except (ValueError, TypeError):
            logger.warning(f"Section {i} has invalid start_time '{section.get('start_time')}'. Defaulting to 0.0.")
            start_time = 0.0

        try:
            end_time = float(section.get("end_time", start_time + 60.0))
        except (ValueError, TypeError):
            logger.warning(f"Section {i} has invalid end_time '{section.get('end_time')}'. Estimating start+60.")
            end_time = start_time + 60.0

        # Enforce non-negative
        start_time = max(0.0, start_time)
        end_time = max(0.0, end_time)

        # Clamp against actual video duration if available
        if reference_duration > 0:
            if start_time >= reference_duration:
                logger.warning(
                    f"Section {i} start_time ({start_time}s) exceeds video duration ({reference_duration}s). Clamping."
                )
                start_time = max(0.0, reference_duration - 1.0)
            if end_time > reference_duration:
                logger.warning(
                    f"Section {i} end_time ({end_time}s) exceeds video duration ({reference_duration}s). Clamping to duration."
                )
                end_time = reference_duration

        # Ensure start_time < end_time
        if start_time >= end_time:
            logger.warning(
                f"Section {i} start_time ({start_time}s) >= end_time ({end_time}s). Adjusting end_time to start+1."
            )
            end_time = start_time + 1.0

        # Validate visual content
        valid_visuals = []
        raw_visuals = section.get("visual_content", [])
        if isinstance(raw_visuals, list):
            for v_idx, visual in enumerate(raw_visuals, start=1):
                if not isinstance(visual, dict):
                    continue

                ts_raw = visual.get("timestamp")
                if ts_raw is None:
                    logger.warning(f"Section {i} Visual {v_idx} missing timestamp. Skipping.")
                    continue

                try:
                    ts = float(ts_raw)
                except (ValueError, TypeError):
                    logger.warning(f"Section {i} Visual {v_idx} invalid timestamp '{ts_raw}'. Skipping.")
                    continue

                if ts < 0:
                    logger.warning(f"Section {i} Visual {v_idx} negative timestamp '{ts}'. Skipping.")
                    continue

                if reference_duration > 0 and ts > reference_duration:
                    logger.warning(
                        f"Section {i} Visual {v_idx} timestamp ({ts}s) exceeds duration ({reference_duration}s). Clamping to {reference_duration - 0.5}s."
                    )
                    ts = max(0.0, reference_duration - 0.5)

                visual_copy = dict(visual)
                visual_copy["timestamp"] = round(ts, 2)
                valid_visuals.append(visual_copy)

        section_copy = dict(section)
        section_copy["start_time"] = round(start_time, 2)
        section_copy["end_time"] = round(end_time, 2)
        section_copy["visual_content"] = valid_visuals
        normalized_sections.append(section_copy)

    analysis_data["sections"] = normalized_sections
    if reference_duration > 0:
        analysis_data["verified_duration_seconds"] = round(reference_duration, 2)

    return analysis_data

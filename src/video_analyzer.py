"""
LectureLens - YouTube Video Analyzer

Phase 1 / Step 3:
Accept a public YouTube URL and send the video directly to Gemini's
multimodal video-understanding capability.

Gemini analyzes:
- spoken lecture content
- presentation slides
- diagrams
- charts
- tables
- code
- screenshots
- formulas
- relationships between narration and visuals

No local video download.
No OpenCV.
No OCR.
No manual frame extraction.
No separate audio/video pipeline.

Input:
    Public YouTube URL

Output:
    Structured JSON containing timestamped lecture sections.
"""

import json
import re
from pathlib import Path

from google.genai import types

from gemini_client import (
    create_gemini_client,
    get_model_name,
    generate_content_with_retry,
)

ANALYSIS_PROMPT = """
You are the multimodal analysis engine for LectureLens, an application that
turns educational YouTube videos into high-quality study notes.

Analyze the ENTIRE video using both its spoken audio and visual content.

The goal is NOT simply to summarize the narration. You must understand how
the lecturer's speech and the visual material work together.

Analyze the following:

1. SPOKEN CONTENT

Identify important:
- explanations
- definitions
- concepts
- examples
- arguments
- procedures
- step-by-step reasoning
- technical terminology


2. VISUAL CONTENT

Identify meaningful educational visuals such as:
- presentation slides
- diagrams
- charts and graphs
- tables
- source code
- screenshots
- equations and formulas
- illustrations
- demonstrations
- other educational visual material


3. AUDIO-VISUAL RELATIONSHIP

When the lecturer explains something shown on screen, connect the two.

For example:
- If the lecturer explains a diagram, explain what the diagram represents.
- If the lecturer discusses code shown on screen, describe what the code does.
- If a slide contains an important definition, preserve the definition.
- If a chart is discussed, connect the spoken explanation with the chart.


TIMESTAMP ACCURACY AND VIDEO DURATION:

* Determine the actual duration of the supplied video before assigning timestamps.
* All timestamps must be expressed in seconds from the beginning of the video.
* Every timestamp MUST be between 0 and the actual video duration.
* Every section must satisfy: 0 <= start_time < end_time <= video_duration.
* Every visual timestamp must satisfy: 0 <= timestamp < video_duration.
* Never extend the final section beyond the actual end of the video.
* Do not estimate timestamps from the amount of content or the number of sections.
* If the exact timestamp of a visual is uncertain, choose a conservative timestamp within the verified duration.
* Before returning the JSON, verify that no section or visual timestamp exceeds the video duration.

5. IMPORTANT VISUAL TEXT

If readable text appears on a slide, code editor, diagram, table, formula,
or screenshot, preserve the important text.

Do NOT reproduce obvious OCR-like garbage or unreadable fragments.

For code:
- preserve important readable code
- preserve programming language when identifiable
- do not invent missing code

For diagrams:
- preserve important labels and relationships

For tables:
- preserve important headers and values when readable

For formulas:
- preserve the formula when it is clearly readable


6. LOW-VALUE CONTENT

Do not create meaningful sections or visual entries for:
- greetings
- introductions with no educational content
- repeated filler
- long silence
- subscribe/follow requests
- advertisements
- promotional messages
- visually unimportant presenter-only footage


7. ACCURACY

- Do not invent information.
- Do not infer text that cannot reasonably be read.
- Do not invent timestamps.
- Do not invent visual content.
- Preserve technical terminology accurately.
- Prefer fewer meaningful sections over excessive fragmentation.
- Only report a visual when it is actually present in the video.
- Visual timestamps must correspond to the approximate moment where the
  described visual is visible.
- Do not assume that a visual remains unchanged throughout an entire section.


8. STUDY VALUE

Prioritize information that would help a student study the lecture later.

The resulting representation should allow another component to reconstruct
the important lecture content and attach representative screenshots of
important visual material.

Therefore:
- preserve useful educational details
- preserve important visual information
- preserve meaningful timestamps
- connect speech with visuals
- avoid unnecessary repetition
- avoid raw visual noise
- avoid invented information


Return ONLY valid JSON.

Use exactly this structure:

{
  "video_summary": "A concise summary of the complete lecture.",

  "sections": [
    {
      "start_time": 0,
      "end_time": 30,
      "title": "Meaningful section title",

      "spoken_content": [
        "Important explanation from the lecture."
      ],

      "visual_content": [
        {
          "timestamp": 12,
          "type": "slide",
          "description": "Description of the educational visual.",
          "visible_text": "Important readable text appearing on screen."
        }
      ],

      "audio_visual_relationship": [
        "Explanation of how the spoken content relates to the visual."
      ],

      "key_concepts": [
        "Important concept"
      ]
    }
  ]
}


Allowed visual types:

- slide
- diagram
- chart
- table
- code
- screenshot
- formula
- demonstration
- other


IMPORTANT:

The final output will be used by another component to generate study notes
and extract representative screenshots from the original YouTube video.

Therefore:

- Every visual_content item MUST have a timestamp.
- The timestamp must correspond to that specific visual.
- Do not use the section start_time as a substitute for the visual timestamp.
- Multiple meaningful visuals in one section should have separate timestamps.
- The same unchanged visual should normally have only one representative
  timestamp.
- Do not create screenshots for insignificant visual changes.
- Do not create visual entries for advertisements or promotional material.
- Preserve useful educational details.
- Preserve important visual information.
- Preserve meaningful timestamps.
- Connect speech with visuals.
- Avoid unnecessary repetition.
- Avoid raw visual noise.
- Avoid invented information.
"""



def extract_json(text: str):
    """
    Extract and validate JSON from Gemini's response.

    Gemini is instructed to return JSON only, but this function also handles
    accidental Markdown code fences defensively.
    """

    cleaned = text.strip()

    # Remove Markdown code fences if present.
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Gemini returned invalid JSON.\n\n"
            f"Response received:\n{cleaned}"
        ) from exc


def analyze_youtube_video(youtube_url: str):
    """
    Analyze a public YouTube video directly through Gemini.

    The video is NOT downloaded locally.

    Parameters
    ----------
    youtube_url:
        Public YouTube video URL.

    Returns
    -------
    dict
        Structured multimodal lecture analysis with normalized timestamps.
    """
    from utils.validation import (
        validate_youtube_url,
        fetch_video_metadata,
        validate_and_normalize_timestamps,
        format_timestamp,
    )

    is_valid, msg, canonical_url = validate_youtube_url(youtube_url)
    if not is_valid or not canonical_url:
        raise ValueError(msg)

    client = create_gemini_client()
    model = get_model_name()

    print(f"Analyzing YouTube video with Gemini...")
    print(f"Model: {model}")
    print(f"URL: {canonical_url}")

    # Discover real video duration without full download
    meta = fetch_video_metadata(canonical_url)
    verified_duration = meta.get("duration", 0.0)
    if verified_duration > 0:
        print(f"Verified video duration: {format_timestamp(verified_duration)} ({verified_duration:.1f}s)")
        effective_prompt = (
            ANALYSIS_PROMPT
            + f"\n\nVERIFIED VIDEO DURATION:\n"
            f"The verified duration of this video is exactly {verified_duration:.1f} seconds "
            f"({format_timestamp(verified_duration)}). "
            f"All section end_times and visual timestamps MUST be <= {verified_duration:.1f} seconds.\n"
        )
    else:
        effective_prompt = ANALYSIS_PROMPT

    print()
    print("Gemini is processing the video's audio and visual content...")
    print("No local video download is required.")

    try:
        response = generate_content_with_retry(
            client=client,
            preferred_model=model,
            contents=types.Content(
                parts=[
                    types.Part(
                        file_data=types.FileData(
                            file_uri=canonical_url
                        )
                    ),
                    types.Part(
                        text=effective_prompt
                    ),
                ]
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Gemini API request failed: {exc}"
        ) from exc

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    raw_analysis = extract_json(response.text)

    # Post-analysis timestamp integrity validation and clamping
    normalized_analysis = validate_and_normalize_timestamps(
        raw_analysis,
        actual_duration=verified_duration if verified_duration > 0 else None,
    )

    return normalized_analysis



def save_analysis(analysis_data, output_path):
    """
    Save structured multimodal analysis to JSON.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            analysis_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print(f"Analysis saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "  python src/video_analyzer.py <youtube_url>"
        )
        sys.exit(1)

    youtube_url = sys.argv[1]

    analysis = analyze_youtube_video(youtube_url)

    output_path = save_analysis(
        analysis,
        "data/analysis/video_analysis.json"
    )

    print()
    print("----------------------------------------")
    print("LectureLens Video Analysis Complete")
    print("----------------------------------------")
    print(f"Sections detected: {len(analysis.get('sections', []))}")
    print(f"Output: {output_path}")


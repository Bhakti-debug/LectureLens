import json
import sys
import tempfile
from pathlib import Path

import cv2
import yt_dlp


def make_safe_filename(text):
    """Convert text into a filesystem-safe filename."""

    safe = "".join(
        c if c.isalnum() or c in (" ", "-", "_")
        else "_"
        for c in text
    ).strip()

    return safe[:60] or "visual"


def download_video(url, output_path):
    """
    Temporarily download the YouTube video so OpenCV can
    extract the frames identified by Gemini.

    The downloaded video exists only inside a temporary
    directory and is automatically deleted afterward.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

  
    ydl_opts = {
        "format": "bestvideo*+bestaudio/best",
        "outtmpl": str(output_path),
        "quiet": False,
        "noplaylist": True,
        "merge_output_format": "mp4",
    }



    print(
        "Temporarily downloading video "
        "for visual extraction..."
    )

    print(f"Source URL: {url}")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output_path


def extract_frame(video_path, timestamp, output_path):
    """
    Extract the frame closest to the timestamp supplied by Gemini.
    Guarantees VideoCapture resource release on all paths.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open temporary video: {video_path}"
        )

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = total_frames / fps if fps > 0 else 0

        if fps <= 0:
            raise RuntimeError(
                "Could not determine video FPS."
            )

        if timestamp < 0:
            raise ValueError(
                f"Negative timestamp {timestamp}s is invalid."
            )

        # Graceful tolerance: if timestamp slightly exceeds duration by <= 2s, clamp to last valid frame
        if timestamp >= duration:
            if duration > 0 and (timestamp - duration) <= 2.0:
                timestamp = max(0.0, duration - 0.2)
            else:
                raise ValueError(
                    f"Timestamp {timestamp}s is outside the video duration "
                    f"({duration:.1f}s)."
                )

        target_frame = int(
            round(timestamp * fps)
        )

        print(
            f"  Extracting frame at "
            f"{timestamp:.1f}s "
            f"(frame {target_frame})"
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            target_frame
        )

        success, frame = cap.read()

        if not success:
            raise RuntimeError(
                f"Could not extract frame at "
                f"{timestamp}s."
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        saved = cv2.imwrite(
            str(output_path),
            frame
        )

        if not saved:
            raise RuntimeError(
                f"Could not save visual image: "
                f"{output_path}"
            )

        return output_path

    finally:
        cap.release()



def extract_visuals(
    youtube_url,
    analysis_path="data/analysis/video_analysis.json",
    output_dir="data/visuals",
):
    """
    Extract actual visual frames identified by Gemini.

    Gemini is responsible for:
        - understanding the video
        - identifying meaningful visuals
        - selecting timestamps
        - describing the visuals
        - identifying visual types
        - extracting important visible text

    This component is responsible for:
        - temporarily obtaining the SAME YouTube video
        - extracting frames at Gemini's timestamps
        - saving those frames as image assets
        - creating the visual manifest

    The temporary video is deleted after extraction.
    """

    analysis_path = Path(analysis_path)
    output_dir = Path(output_dir)

    if not youtube_url:
        raise ValueError(
            "YouTube URL is required for visual extraction."
        )

    if not analysis_path.exists():
        raise FileNotFoundError(
            f"Analysis file not found: {analysis_path}"
        )

    with open(
        analysis_path,
        "r",
        encoding="utf-8"
    ) as f:
        analysis = json.load(f)

    sections = analysis.get(
        "sections",
        []
    )

    visual_items = []

    # -----------------------------------------------------
    # Read the visual information identified by Gemini
    # -----------------------------------------------------

    for section_number, section in enumerate(
        sections,
        start=1
    ):
        visual_content = section.get(
            "visual_content",
            []
        )

        if not isinstance(
            visual_content,
            list
        ):
            continue

        for visual_index, visual in enumerate(
            visual_content,
            start=1
        ):
            if not isinstance(
                visual,
                dict
            ):
                continue

            timestamp = visual.get(
                "timestamp"
            )

            if timestamp is None:
                print(
                    f"Warning: visual "
                    f"{section_number}.{visual_index} "
                    "has no timestamp. Skipping."
                )
                continue

            try:
                timestamp = float(timestamp)

            except (
                TypeError,
                ValueError
            ):
                print(
                    f"Warning: invalid timestamp "
                    f"'{timestamp}' in visual "
                    f"{section_number}."
                    f"{visual_index}. "
                    "Skipping."
                )
                continue

            if timestamp < 0:
                print(
                    f"Warning: negative timestamp "
                    f"{timestamp} in visual "
                    f"{section_number}."
                    f"{visual_index}. "
                    "Skipping."
                )
                continue

            visual_items.append(
                {
                    "section_number": section_number,

                    "section_title": section.get(
                        "title",
                        f"Section {section_number}"
                    ),

                    "visual_index": visual_index,

                    "timestamp": timestamp,

                    "type": visual.get(
                        "type",
                        "other"
                    ),

                    "description": visual.get(
                        "description",
                        ""
                    ),

                    "visible_text": visual.get(
                        "visible_text",
                        ""
                    ),
                }
            )

    print(
        f"Found {len(visual_items)} "
        "meaningful visual elements."
    )

    if not visual_items:

        print(
            "No timestamped visual content found."
        )

        manifest = {
            "video_summary": analysis.get(
                "video_summary",
                ""
            ),
            "visual_count": 0,
            "visuals": [],
        }

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        manifest_path = (
            output_dir
            / "visual_manifest.json"
        )

        with open(
            manifest_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                manifest,
                f,
                ensure_ascii=False,
                indent=2
            )

        return manifest

    # -----------------------------------------------------
    # Create permanent image directory
    # -----------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    # -----------------------------------------------------
    # Temporary video
    #
    # IMPORTANT:
    # The video is NOT saved permanently.
    # It exists only inside this temporary directory.
    # -----------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="lecturelens_video_"
    ) as temp_dir:

        temp_video_path = (
            Path(temp_dir)
            / "source.mp4"
        )

        # Download the EXACT URL supplied by the user.
        download_video(
            youtube_url,
            temp_video_path
        )

        # -------------------------------------------------
        # Extract every Gemini-selected visual
        # -------------------------------------------------

        
        # -------------------------------------------------
        # Extract Gemini-selected visuals safely
        # -------------------------------------------------

        for item in visual_items:

            timestamp = item["timestamp"]
            section_number = item["section_number"]
            visual_index = item["visual_index"]
            section_title = item["section_title"]
            visual_type = item["type"]

            safe_title = make_safe_filename(section_title)

            filename = (
                f"section_{section_number:02d}_"
                f"visual_{visual_index:02d}_"
                f"{int(timestamp)}s_"
                f"{safe_title}.jpg"
            )

            output_path = output_dir / filename

            print(
                f"Visual {section_number}.{visual_index} "
                f"@ {timestamp:.1f}s ({visual_type})"
            )

            try:
                extract_frame(
                    temp_video_path,
                    timestamp,
                    output_path
                )

            except ValueError as e:
                print(
                    f"Warning: skipping visual "
                    f"{section_number}.{visual_index}: {e}"
                )
                continue

            except RuntimeError as e:
                print(
                    f"Warning: could not extract visual "
                    f"{section_number}.{visual_index}: {e}"
                )
                continue

            results.append(
                {
                    "section_number": section_number,
                    "section_title": section_title,
                    "visual_index": visual_index,
                    "timestamp": timestamp,
                    "type": visual_type,
                    "description": item["description"],
                    "visible_text": item["visible_text"],
                    "image": str(output_path),
                }
            )
    # -----------------------------------------------------
    # Create visual manifest
    # -----------------------------------------------------

    manifest = {
        "video_summary": analysis.get(
            "video_summary",
            ""
        ),

        "visual_count": len(results),

        "visuals": results,
    }

    manifest_path = (
        output_dir
        / "visual_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()

    print(
        f"Extracted {len(results)} "
        "visual images."
    )

    print(
        f"Visual images saved to: "
        f"{output_dir}"
    )

    print(
        f"Visual manifest saved to: "
        f"{manifest_path}"
    )

    return manifest


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "python src/visual_extractor.py "
            "<youtube_url>"
        )

        sys.exit(1)

    youtube_url = sys.argv[1]

    extract_visuals(
        youtube_url=youtube_url
    )
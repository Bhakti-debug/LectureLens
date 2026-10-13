"""
LectureLens - Lecture Service
Manages lecture lifecycle, stable identifiers, multi-lecture isolation,
persistence under data/lectures/<lecture_id>, and backward-compatible legacy mirrors.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.config import get_config
from utils.logging_config import get_logger
from utils.validation import extract_youtube_video_id

logger = get_logger("lecture_service")


class LectureService:
    """Coordinates storage, switching, and lifecycle for multiple lectures."""

    def __init__(self):
        self.config = get_config()
        self.lectures_dir = self.config.lectures_dir
        self.lectures_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_sample_or_legacy_indexed()

    def get_lecture_id_from_url(self, url: str) -> str:
        """Derive a stable lecture identifier from a YouTube URL."""
        video_id = extract_youtube_video_id(url)
        if video_id:
            return f"yt_{video_id}"
        # Fallback to sanitized slug
        clean = "".join(c if c.isalnum() else "_" for c in url.strip()).strip("_")
        return f"lec_{clean[:20]}" or "lec_default"

    def get_lecture_dir(self, lecture_id: str) -> Path:
        """Return the directory path for a specific lecture."""
        lec_dir = self.lectures_dir / lecture_id
        lec_dir.mkdir(parents=True, exist_ok=True)
        return lec_dir

    def save_lecture(
        self,
        lecture_id: str,
        analysis_data: Dict[str, Any],
        notes_data: Optional[Dict[str, Any]] = None,
        visual_manifest: Optional[Dict[str, Any]] = None,
        sync_legacy: bool = True,
    ) -> Path:
        """
        Persist all artifacts for a lecture in its isolated directory and update legacy paths.
        """
        lec_dir = self.get_lecture_dir(lecture_id)

        # 1. Save video analysis
        analysis_file = lec_dir / "video_analysis.json"
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)

        # 2. Save study notes if provided
        if notes_data:
            notes_file = lec_dir / "study_notes.json"
            with open(notes_file, "w", encoding="utf-8") as f:
                json.dump(notes_data, f, ensure_ascii=False, indent=2)

        # 3. Save visual manifest if provided
        if visual_manifest:
            manifest_file = lec_dir / "visual_manifest.json"
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(visual_manifest, f, ensure_ascii=False, indent=2)

        # 4. Save metadata summary
        summary = {
            "lecture_id": lecture_id,
            "title": (notes_data.get("title") if notes_data else None)
            or analysis_data.get("video_summary", "")[:50]
            or "Lecture",
            "section_count": len(analysis_data.get("sections", [])),
            "duration": analysis_data.get("verified_duration_seconds", 0),
        }
        with open(lec_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if sync_legacy:
            self.sync_to_legacy_paths(lecture_id)

        logger.info(f"Lecture data saved for {lecture_id} in {lec_dir}")
        return lec_dir

    def load_lecture(self, lecture_id: str) -> Dict[str, Any]:
        """
        Load all stored artifacts for a given lecture.
        """
        lec_dir = self.get_lecture_dir(lecture_id)
        data: Dict[str, Any] = {
            "lecture_id": lecture_id,
            "analysis": None,
            "notes": None,
            "visual_manifest": None,
        }

        analysis_file = lec_dir / "video_analysis.json"
        if analysis_file.exists():
            with open(analysis_file, "r", encoding="utf-8") as f:
                data["analysis"] = json.load(f)

        notes_file = lec_dir / "study_notes.json"
        if notes_file.exists():
            with open(notes_file, "r", encoding="utf-8") as f:
                data["notes"] = json.load(f)

        manifest_file = lec_dir / "visual_manifest.json"
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                data["visual_manifest"] = json.load(f)

        return data

    def list_lectures(self) -> List[Dict[str, Any]]:
        """List all available stored lectures."""
        results = []
        for d in self.lectures_dir.iterdir():
            if d.is_dir():
                meta_file = d / "meta.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                            results.append(meta)
                            continue
                    except Exception:
                        pass
                # Infer from analysis file
                af = d / "video_analysis.json"
                if af.exists():
                    results.append({"lecture_id": d.name, "title": d.name})
        return results

    def sync_to_legacy_paths(self, lecture_id: str) -> None:
        """
        Copy lecture artifacts to legacy data/analysis, data/notes, data/visuals
        so single-lecture code and original UI components function without regression.
        """
        lec_dir = self.get_lecture_dir(lecture_id)

        # Sync analysis
        source_af = lec_dir / "video_analysis.json"
        if source_af.exists():
            self.config.analysis_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_af, self.config.analysis_path)

        # Sync notes
        source_nf = lec_dir / "study_notes.json"
        if source_nf.exists():
            self.config.notes_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_nf, self.config.notes_path)

        # Sync visual manifest
        source_mf = lec_dir / "visual_manifest.json"
        if source_mf.exists():
            self.config.visuals_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_mf, self.config.visual_manifest_path)

    def _ensure_sample_or_legacy_indexed(self) -> None:
        """If legacy data exists and no lecture-scoped folder exists, populate sample_node_lecture."""
        legacy_analysis = self.config.analysis_path
        if legacy_analysis.exists():
            sample_id = "sample_node_lecture"
            sample_dir = self.lectures_dir / sample_id
            if not sample_dir.exists():
                logger.info(f"Importing baseline lecture into {sample_id}...")
                sample_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(legacy_analysis, sample_dir / "video_analysis.json")

                if self.config.notes_path.exists():
                    shutil.copy2(self.config.notes_path, sample_dir / "study_notes.json")
                if self.config.visual_manifest_path.exists():
                    shutil.copy2(self.config.visual_manifest_path, sample_dir / "visual_manifest.json")

                summary = {
                    "lecture_id": sample_id,
                    "title": "Introduction to Node.js and Core Concepts",
                    "section_count": 4,
                    "duration": 464,
                }
                with open(sample_dir / "meta.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)


def get_lecture_service() -> LectureService:
    """Return the global LectureService instance."""
    return LectureService()

"""
LectureLens - Data Models and Schemas
Typed Pydantic schemas representing all contracts across the processing,
RAG retrieval, assessment, and export pipelines.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class VisualContentItem(BaseModel):
    """Visual content item identified in lecture analysis."""
    timestamp: float
    type: str = "other"
    description: str = ""
    visible_text: str = ""


class SectionAnalysis(BaseModel):
    """Single section in the multimodal lecture analysis."""
    start_time: float
    end_time: float
    title: str
    spoken_content: List[str] = Field(default_factory=list)
    visual_content: List[VisualContentItem] = Field(default_factory=list)
    audio_visual_relationship: List[str] = Field(default_factory=list)
    key_concepts: List[str] = Field(default_factory=list)


class VideoAnalysis(BaseModel):
    """Complete structured multimodal lecture analysis."""
    video_summary: str
    sections: List[SectionAnalysis] = Field(default_factory=list)
    verified_duration_seconds: Optional[float] = None


class VisualAsset(BaseModel):
    """Extracted physical frame image asset."""
    section_number: int
    section_title: str
    visual_index: int
    timestamp: float
    type: str = "other"
    description: str = ""
    visible_text: str = ""
    image: str


class VisualManifest(BaseModel):
    """Manifest of extracted visual frames."""
    video_summary: str = ""
    visual_count: int = 0
    visuals: List[VisualAsset] = Field(default_factory=list)


class NoteSection(BaseModel):
    """Section in structured study notes."""
    section_number: int
    title: str
    start_time: float = 0.0
    end_time: float = 0.0
    explanation: str = ""
    key_concepts: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)
    visual_explanation: Optional[str] = None
    important_visual_text: List[str] = Field(default_factory=list)
    key_takeaway: Optional[str] = None
    visuals: List[Dict[str, Any]] = Field(default_factory=list)


class StudyNotes(BaseModel):
    """Complete generated study notes."""
    title: str
    overview: str
    duration_seconds: float = 0.0
    sections: List[NoteSection] = Field(default_factory=list)
    final_takeaways: List[str] = Field(default_factory=list)


class LectureDocument(BaseModel):
    """Normalized document constructed from a lecture section for indexing."""
    doc_id: str
    lecture_id: str
    section_number: int
    section_title: str
    start_time: float
    end_time: float
    text: str
    source_type: str = "analyzed_section"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LectureChunk(BaseModel):
    """Text chunk ready for vector embedding and retrieval."""
    chunk_id: str
    doc_id: str
    lecture_id: str
    section_number: int
    section_title: str
    start_time: float
    end_time: float
    text: str
    source_type: str = "analyzed_section"
    chunk_index: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Result of vector similarity retrieval."""
    chunk: LectureChunk
    score: float


class SourceCitation(BaseModel):
    """Citation metadata for retrieved evidence."""
    section_number: int
    section_title: str
    start_time: float
    end_time: float
    timestamp_formatted: str
    snippet: str
    score: float


class RAGResponse(BaseModel):
    """Structured response from the question-answering system."""
    query: str
    answer: str
    is_relevant: bool = True
    is_sufficient: bool = True
    sources: List[SourceCitation] = Field(default_factory=list)
    relevance_reason: str = ""
    refusal_reason: str = ""


class Flashcard(BaseModel):
    """Self-assessment revision flashcard."""
    id: str
    question: str
    answer: str
    section_title: str
    timestamp: float = 0.0


class QuizQuestion(BaseModel):
    """Multiple choice or concept assessment question."""
    id: str
    question: str
    options: List[str]
    correct_index: int
    explanation: str
    section_title: str
    timestamp: float = 0.0

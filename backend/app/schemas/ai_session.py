"""
MemoryVerse - AI Session, VideoSpecification, MemoryContext, and Revision Schemas
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EmotionSpec(BaseModel):
    primary: str = "calm"
    secondary: List[str] = Field(default_factory=list)
    distribution: Dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.85
    intensity: float = 0.5


class MusicSpec(BaseModel):
    style: str = "cinematic_warm"  # energetic, warm, cinematic, acoustic, minimal, emotional
    tempo: int = 90
    energy: float = 0.6


class NarrationSpec(BaseModel):
    enabled: bool = False
    style: str = "warm_storyteller"
    voice: str = "auto"


class MotionSpec(BaseModel):
    style: str = "dynamic_zoom"  # slow_pan, dynamic_zoom, subtle_zoom, steady
    intensity: float = 0.7


class CompositionSpec(BaseModel):
    background_strategy: str = "adaptive_blurred"  # adaptive_blurred, color_gradient, darkened_duplicate, adaptive_cinematic
    crop_policy: str = "subject_aware"  # subject_aware, center, fit
    preserve_source_quality: bool = True


class VideoSpecification(BaseModel):
    version: str = "1.0"
    duration_target_seconds: int = 20
    aspect_ratio: str = "9:16"  # 16:9, 9:16, 1:1, 4:3
    orientation: str = "portrait"  # landscape, portrait, square
    quality_profile: str = "balanced"  # fast, balanced, high_quality

    media_selection: List[str] = Field(default_factory=list)
    selection_strategy: str = "emotion_relevant"  # emotion_relevant, ai_curated, use_all, user_selected

    story_structure: str = "chronological"  # chronological, three_act_arc, highlight_montage, emotional_crescendo
    story: Dict[str, Any] = Field(default_factory=lambda: {"style": "cinematic", "structure": "chronological"})

    emotion: EmotionSpec = Field(default_factory=EmotionSpec)
    music: MusicSpec = Field(default_factory=MusicSpec)
    narration: NarrationSpec = Field(default_factory=NarrationSpec)
    text: Dict[str, Any] = Field(default_factory=lambda: {"enabled": True, "style": "minimal_captions"})
    motion: MotionSpec = Field(default_factory=MotionSpec)
    transitions: Dict[str, str] = Field(default_factory=lambda: {"style": "dissolve"})
    composition: CompositionSpec = Field(default_factory=CompositionSpec)

    @property
    def duration_seconds(self) -> int:
        return self.duration_target_seconds

    @duration_seconds.setter
    def duration_seconds(self, val: int) -> None:
        self.duration_target_seconds = val


class MemoryContext(BaseModel):
    memory_id: str
    user_id: str
    title: str
    description: Optional[str] = None
    event_date: Optional[str] = None
    location_name: Optional[str] = None

    media_count: int = 0
    photo_count: int = 0
    video_count: int = 0
    media_ids: List[str] = Field(default_factory=list)
    candidate_media: List[Dict[str, Any]] = Field(default_factory=list)

    semantic_topics: List[str] = Field(default_factory=list)
    emotion_distribution: Dict[str, float] = Field(default_factory=dict)
    dominant_emotion: str = "calm"
    emotion_confidence: float = 0.85

    current_specification: Optional[VideoSpecification] = None
    previous_revisions: List[str] = Field(default_factory=list)


class AIMessage(BaseModel):
    id: str
    session_id: str
    role: str  # user, assistant, system
    content: str
    suggestions: List[str] = Field(default_factory=list)
    specification: Optional[VideoSpecification] = None
    job_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AISessionResponse(BaseModel):
    session_id: str
    memory_id: str
    user_id: str
    title: str
    specification: VideoSpecification
    context: MemoryContext
    messages: List[AIMessage] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    active_job_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class AISessionCreateRequest(BaseModel):
    memory_id: str
    initial_prompt: Optional[str] = None


class UserMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class GenerateVideoRequest(BaseModel):
    specification: Optional[VideoSpecification] = None


class RevisionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)


class VideoRevisionResponse(BaseModel):
    revision_id: str
    parent_revision_id: Optional[str] = None
    job_id: str
    specification_hash: str
    changed_fields: List[str] = Field(default_factory=list)
    invalidated_stages: List[str] = Field(default_factory=list)
    status: str = "queued"
    created_at: datetime

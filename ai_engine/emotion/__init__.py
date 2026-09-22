"""
MemoryVerse - Emotion Intelligence Package
Provides multimodal emotion extraction, confidence-weighted fusion,
dynamic emotion timelines, and research ablation configurations.
"""
from ai_engine.emotion.emotion_analyzer import (
    EmotionAnalyzer,
    EmotionDistribution,
    EmotionTimeline,
    EmotionAblationMode,
    emotion_analyzer,
)

__all__ = [
    "EmotionAnalyzer",
    "EmotionDistribution",
    "EmotionTimeline",
    "EmotionAblationMode",
    "emotion_analyzer",
]

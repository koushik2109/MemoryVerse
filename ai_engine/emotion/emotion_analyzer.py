"""
MemoryVerse - Multimodal Emotion Analyzer & EmotionTimeline
Extracts affective signals from visual cues (CLIP + color temperature),
acoustic features (energy, tempo), speech transcripts, text descriptions,
and temporal EXIF metadata.
Fuses them with confidence weights and generates an EmotionTimeline that drives
story pacing, scene durations, Ken Burns motion intensity, 4-chord soundtrack mood,
and TTS narration prosody.
"""
import io
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

EMOTION_MODEL_VERSION = "emotion-fusion-v2"

# 6 Core Affective Dimensions
EMOTION_CLASSES = ["joy", "nostalgia", "calm", "sadness", "excitement", "love"]

# Lexical emotion word banks for text & speech analysis
LEXICAL_EMOTION_MAP: Dict[str, Dict[str, float]] = {
    "joy": {
        "happy": 0.8, "fun": 0.7, "celebrate": 0.9, "laugh": 0.8, "party": 0.85,
        "birthday": 0.85, "smile": 0.75, "wonderful": 0.8, "sun": 0.5, "beach": 0.6,
        "cheers": 0.85, "dance": 0.8, "friends": 0.6, "together": 0.6, "great": 0.7
    },
    "nostalgia": {
        "remember": 0.9, "memory": 0.85, "past": 0.8, "old": 0.7, "years": 0.6,
        "childhood": 0.9, "vintage": 0.8, "reminisce": 0.95, "family": 0.7,
        "home": 0.65, "trip": 0.6, "summer": 0.6, "golden": 0.75, "miss": 0.8
    },
    "calm": {
        "peace": 0.9, "quiet": 0.85, "relax": 0.85, "calm": 0.9, "serene": 0.95,
        "sunset": 0.8, "morning": 0.7, "lake": 0.75, "nature": 0.7, "walk": 0.6,
        "gentle": 0.8, "soft": 0.75, "rest": 0.8, "meditation": 0.9
    },
    "sadness": {
        "sad": 0.9, "cry": 0.9, "goodbye": 0.85, "loss": 0.9, "farewell": 0.85,
        "rain": 0.6, "dark": 0.6, "hurt": 0.8, "alone": 0.75, "gloomy": 0.8
    },
    "excitement": {
        "thrill": 0.9, "adventure": 0.9, "rush": 0.85, "fast": 0.7, "climb": 0.75,
        "summit": 0.85, "explore": 0.8, "wild": 0.8, "wow": 0.85, "flight": 0.7,
        "sport": 0.8, "race": 0.85, "concert": 0.9, "festival": 0.85
    },
    "love": {
        "love": 0.95, "wedding": 0.9, "anniversary": 0.9, "romantic": 0.9, "kiss": 0.85,
        "hug": 0.8, "couple": 0.8, "heart": 0.75, "sweet": 0.7, "cherish": 0.85
    },
}


class EmotionAblationMode(str, Enum):
    A_NONE = "no_emotion"
    B_VISUAL_ONLY = "visual_only"
    C_VISUAL_AUDIO = "visual_audio"
    D_VISUAL_AUDIO_TEXT = "visual_audio_text"
    E_FULL_MULTIMODAL_TEMPORAL = "full_multimodal_temporal"


@dataclass
class EmotionDistribution:
    distribution: Dict[str, float]
    dominant_emotion: str
    confidence: float
    secondary_emotions: List[str]
    sources: Dict[str, Any]
    model_version: str = EMOTION_MODEL_VERSION
    ablation_mode: str = EmotionAblationMode.E_FULL_MULTIMODAL_TEMPORAL.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "distribution": {k: round(v, 4) for k, v in self.distribution.items()},
            "dominant_emotion": self.dominant_emotion,
            "confidence": round(self.confidence, 4),
            "secondary_emotions": self.secondary_emotions,
            "sources": self.sources,
            "model_version": self.model_version,
            "ablation_mode": self.ablation_mode,
        }


@dataclass
class EmotionTimelineKeypoint:
    timestamp_offset_sec: float
    media_id: str
    dominant_emotion: str
    intensity: float
    distribution: Dict[str, float]
    recommended_duration_sec: float
    motion_intensity: float  # 0.5 (gentle pan) to 1.5 (dynamic zoom/pan)
    transition_type: str     # "crossfade", "dissolve", "cut", "dip_to_black"


@dataclass
class EmotionTimeline:
    keypoints: List[EmotionTimelineKeypoint] = field(default_factory=list)
    overall_mood: str = "calm"
    average_intensity: float = 0.5

    def get_pacing_modifier(self, emotion: str, intensity: float) -> float:
        """Returns duration multiplier (e.g. excitement -> faster cuts, calm -> longer)."""
        if emotion in ("excitement", "joy"):
            # Shorter, faster-paced shots
            return max(0.7, 1.0 - (intensity * 0.3))
        elif emotion in ("calm", "nostalgia", "love"):
            # Longer, breathing, contemplative shots
            return min(1.4, 1.0 + (intensity * 0.35))
        return 1.0

    def get_motion_preset(self, emotion: str) -> str:
        """Maps dominant emotion to Ken Burns motion style."""
        if emotion == "excitement":
            return "dynamic_zoom_in"
        elif emotion == "joy":
            return "gentle_zoom_in"
        elif emotion == "nostalgia":
            return "slow_pan_horizontal"
        elif emotion == "calm":
            return "subtle_zoom_out"
        elif emotion == "love":
            return "slow_center_zoom"
        return "slow_pan_right"

    def get_audio_mood(self) -> str:
        """Returns soundtrack mood parameter for audio synthesizer."""
        if self.overall_mood in ("joy", "excitement"):
            return "energetic"
        elif self.overall_mood in ("nostalgia", "love"):
            return "nostalgic"
        elif self.overall_mood == "calm":
            return "calm"
        return "neutral"

    def get_tts_profile(self) -> str:
        """Returns neural TTS emotion preset."""
        if self.overall_mood == "joy":
            return "joyful"
        elif self.overall_mood == "excitement":
            return "energetic"
        elif self.overall_mood == "nostalgia":
            return "nostalgic"
        elif self.overall_mood in ("sadness", "dramatic"):
            return "dramatic"
        elif self.overall_mood == "calm":
            return "calm"
        return "calm"


class EmotionAnalyzer:
    """
    Multimodal Emotion Analyzer extracting visual, acoustic, textual, and temporal cues.
    """

    def __init__(self):
        self._clip_model = None

    def _get_clip(self):
        if self._clip_model is None:
            try:
                from ai_engine.models.clip_loader import get_clip_model
                self._clip_model = get_clip_model()
            except Exception as e:
                logger.debug(f"CLIP loader unavailable in EmotionAnalyzer: {e}")
        return self._clip_model

    def analyze_visual_emotion(
        self,
        image_bytes: Optional[bytes] = None,
        clip_embedding: Optional[List[float]] = None,
        color_stats: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], float]:
        """
        Extracts emotion distribution from visual cues:
        1. CLIP embedding cosine similarity to emotion prompt anchors
        2. Color saturation, brightness, and warmth
        """
        raw_scores = {e: 0.166 for e in EMOTION_CLASSES}
        confidence = 0.5

        # 1. CLIP visual embedding match
        if clip_embedding and any(clip_embedding):
            vec = np.array(clip_embedding, dtype=np.float32)
            v_norm = np.linalg.norm(vec)
            if v_norm > 0:
                vec = vec / v_norm
                model = self._get_clip()
                if model is not None:
                    try:
                        prompts = [
                            "a joyful vibrant happy celebration",
                            "a nostalgic warm vintage memory from the past",
                            "a calm peaceful serene quiet natural scene",
                            "a sad gloomy dark somber moment",
                            "an exciting fast adventure thrill",
                            "a loving romantic sweet affectionate moment",
                        ]
                        prompt_embs = model.encode(prompts)
                        sims = [float(np.dot(vec, p / np.linalg.norm(p))) for p in prompt_embs]
                        # Softmax
                        exp_sims = np.exp(np.array(sims) * 4.0)
                        probs = exp_sims / np.sum(exp_sims)
                        for idx, e in enumerate(EMOTION_CLASSES):
                            raw_scores[e] = float(probs[idx])
                        confidence = float(np.max(probs))
                    except Exception as e:
                        logger.debug(f"Visual CLIP emotion failed: {e}")

        # 2. Color temperature heuristic adjustment (if stats provided)
        if color_stats:
            warmth = color_stats.get("warmth", 0.5)      # 0.0 (cool/blue) to 1.0 (warm/golden)
            saturation = color_stats.get("saturation", 0.5)
            brightness = color_stats.get("brightness", 0.5)

            if warmth > 0.65 and brightness > 0.6:
                raw_scores["nostalgia"] += 0.15
                raw_scores["joy"] += 0.10
            elif warmth < 0.35 and brightness < 0.4:
                raw_scores["calm"] += 0.12
                raw_scores["sadness"] += 0.08
            if saturation > 0.7:
                raw_scores["excitement"] += 0.15
                raw_scores["joy"] += 0.10

        # Normalize
        total = sum(raw_scores.values()) or 1.0
        return {k: v / total for k, v in raw_scores.items()}, min(1.0, max(0.1, confidence))

    def analyze_audio_emotion(
        self,
        acoustic_features: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, float], float]:
        """
        Extracts emotion distribution from acoustic features:
        RMS energy, estimated tempo (BPM), spectral centroid, zero-crossing rate.
        """
        if not acoustic_features:
            return {e: 0.166 for e in EMOTION_CLASSES}, 0.1

        energy = float(acoustic_features.get("rms_energy", 0.05))
        tempo = float(acoustic_features.get("tempo", 110.0))
        zcr = float(acoustic_features.get("zcr", 0.05))

        scores = {e: 0.1 for e in EMOTION_CLASSES}

        if tempo > 125.0 and energy > 0.08:
            scores["excitement"] += 0.45
            scores["joy"] += 0.35
        elif tempo < 90.0 and energy < 0.04:
            scores["calm"] += 0.40
            scores["nostalgia"] += 0.35
            scores["sadness"] += 0.15
        elif 90.0 <= tempo <= 125.0:
            scores["joy"] += 0.30
            scores["calm"] += 0.25
            scores["love"] += 0.25

        total = sum(scores.values()) or 1.0
        norm_dist = {k: v / total for k, v in scores.items()}
        conf = min(0.85, 0.4 + (energy * 5.0))
        return norm_dist, conf

    def analyze_text_emotion(
        self,
        text: Optional[str] = None,
    ) -> Tuple[Dict[str, float], float]:
        """
        Extracts emotion distribution from title, descriptions, or transcripts.
        """
        if not text or len(text.strip()) == 0:
            return {e: 0.166 for e in EMOTION_CLASSES}, 0.1

        text_clean = text.lower()
        scores = {e: 0.05 for e in EMOTION_CLASSES}
        matched_words = 0

        for emotion, keywords in LEXICAL_EMOTION_MAP.items():
            for kw, weight in keywords.items():
                if kw in text_clean:
                    scores[emotion] += weight
                    matched_words += 1

        if matched_words == 0:
            return {e: 0.166 for e in EMOTION_CLASSES}, 0.15

        total = sum(scores.values())
        norm_dist = {k: v / total for k, v in scores.items()}
        conf = min(0.95, 0.3 + (matched_words * 0.15))
        return norm_dist, conf

    def analyze_temporal_emotion(
        self,
        taken_at: Optional[str] = None,
    ) -> Tuple[Dict[str, float], float]:
        """
        Extracts contextual emotion bias from capture hour (golden hour, sunset, night).
        """
        if not taken_at:
            return {e: 0.166 for e in EMOTION_CLASSES}, 0.1

        try:
            # Parse hour
            hour = 12
            if "t" in taken_at.lower():
                time_part = taken_at.lower().split("t")[1]
                hour = int(time_part.split(":")[0])
            elif " " in taken_at:
                time_part = taken_at.split(" ")[1]
                hour = int(time_part.split(":")[0])

            scores = {e: 0.1 for e in EMOTION_CLASSES}
            # Golden hour morning (6-8) or evening (17-19)
            if 6 <= hour <= 8 or 17 <= hour <= 19:
                scores["calm"] += 0.35
                scores["nostalgia"] += 0.30
                scores["love"] += 0.20
            # Night party hours (21-3)
            elif hour >= 21 or hour <= 3:
                scores["excitement"] += 0.40
                scores["joy"] += 0.30
            # Midday sunshine (11-15)
            elif 11 <= hour <= 15:
                scores["joy"] += 0.40
                scores["excitement"] += 0.25

            total = sum(scores.values())
            return {k: v / total for k, v in scores.items()}, 0.45
        except Exception:
            return {e: 0.166 for e in EMOTION_CLASSES}, 0.1

    def fuse_emotions(
        self,
        visual_data: Optional[Tuple[Dict[str, float], float]] = None,
        audio_data: Optional[Tuple[Dict[str, float], float]] = None,
        text_data: Optional[Tuple[Dict[str, float], float]] = None,
        temporal_data: Optional[Tuple[Dict[str, float], float]] = None,
        ablation_mode: EmotionAblationMode = EmotionAblationMode.E_FULL_MULTIMODAL_TEMPORAL,
    ) -> EmotionDistribution:
        """
        Confidence-weighted multimodal fusion supporting research ablation modes.
        """
        # Mode A: No emotion awareness -> uniform baseline
        if ablation_mode == EmotionAblationMode.A_NONE:
            uniform = {e: 1.0 / len(EMOTION_CLASSES) for e in EMOTION_CLASSES}
            return EmotionDistribution(
                distribution=uniform,
                dominant_emotion="calm",
                confidence=0.5,
                secondary_emotions=[],
                sources={"ablation": "no_emotion"},
                ablation_mode=ablation_mode.value,
            )

        fused_scores = {e: 0.0 for e in EMOTION_CLASSES}
        total_weight = 0.0
        sources_meta = {}

        # 1. Visual Modality
        if visual_data and ablation_mode in (
            EmotionAblationMode.B_VISUAL_ONLY,
            EmotionAblationMode.C_VISUAL_AUDIO,
            EmotionAblationMode.D_VISUAL_AUDIO_TEXT,
            EmotionAblationMode.E_FULL_MULTIMODAL_TEMPORAL,
        ):
            v_dist, v_conf = visual_data
            weight = max(0.1, v_conf * 1.2)
            for e, val in v_dist.items():
                fused_scores[e] += val * weight
            total_weight += weight
            sources_meta["visual_confidence"] = round(v_conf, 3)

        # 2. Audio Modality
        if audio_data and ablation_mode in (
            EmotionAblationMode.C_VISUAL_AUDIO,
            EmotionAblationMode.D_VISUAL_AUDIO_TEXT,
            EmotionAblationMode.E_FULL_MULTIMODAL_TEMPORAL,
        ):
            a_dist, a_conf = audio_data
            if a_conf > 0.15:
                weight = max(0.1, a_conf * 1.0)
                for e, val in a_dist.items():
                    fused_scores[e] += val * weight
                total_weight += weight
                sources_meta["audio_confidence"] = round(a_conf, 3)

        # 3. Text Modality
        if text_data and ablation_mode in (
            EmotionAblationMode.D_VISUAL_AUDIO_TEXT,
            EmotionAblationMode.E_FULL_MULTIMODAL_TEMPORAL,
        ):
            t_dist, t_conf = text_data
            if t_conf > 0.15:
                weight = max(0.1, t_conf * 1.4)
                for e, val in t_dist.items():
                    fused_scores[e] += val * weight
                total_weight += weight
                sources_meta["text_confidence"] = round(t_conf, 3)

        # 4. Temporal Modality
        if temporal_data and ablation_mode == EmotionAblationMode.E_FULL_MULTIMODAL_TEMPORAL:
            temp_dist, temp_conf = temporal_data
            weight = 0.4
            for e, val in temp_dist.items():
                fused_scores[e] += val * weight
            total_weight += weight
            sources_meta["temporal_confidence"] = round(temp_conf, 3)

        # Fallback to uniform if no modality provided
        if total_weight <= 0.0:
            total_weight = 1.0
            fused_scores = {e: 1.0 / len(EMOTION_CLASSES) for e in EMOTION_CLASSES}

        # Normalize
        normalized_dist = {e: fused_scores[e] / total_weight for e in EMOTION_CLASSES}

        # Find dominant and secondary emotions
        sorted_emotions = sorted(normalized_dist.items(), key=lambda kv: kv[1], reverse=True)
        dominant_emotion = sorted_emotions[0][0]
        dominant_score = sorted_emotions[0][1]

        secondaries = [
            e for e, s in sorted_emotions[1:3]
            if s >= 0.18 and e != dominant_emotion
        ]

        # Overall confidence is proportional to peak sharpness
        entropy = -sum(p * math.log(max(p, 1e-9)) for p in normalized_dist.values())
        max_entropy = math.log(len(EMOTION_CLASSES))
        confidence = max(0.2, min(0.98, 1.0 - (entropy / max_entropy) * 0.7))

        return EmotionDistribution(
            distribution=normalized_dist,
            dominant_emotion=dominant_emotion,
            confidence=confidence,
            secondary_emotions=secondaries,
            sources=sources_meta,
            model_version=EMOTION_MODEL_VERSION,
            ablation_mode=ablation_mode.value,
        )

    def build_emotion_timeline(
        self,
        media_items: List[Dict[str, Any]],
        memory_title: Optional[str] = None,
        user_prompt: Optional[str] = None,
        target_duration: float = 30.0,
        ablation_mode: EmotionAblationMode = EmotionAblationMode.E_FULL_MULTIMODAL_TEMPORAL,
    ) -> EmotionTimeline:
        """
        Builds a chronological EmotionTimeline spanning the media items to direct the video reel.
        """
        if not media_items:
            return EmotionTimeline()

        # Pre-analyze global text context
        text_context = f"{memory_title or ''} {user_prompt or ''}".strip()
        global_text_data = self.analyze_text_emotion(text_context)

        keypoints: List[EmotionTimelineKeypoint] = []
        n_items = len(media_items)
        base_scene_duration = max(2.0, min(5.0, target_duration / max(1, n_items)))

        current_offset = 0.0
        all_emotions = []

        for idx, item in enumerate(media_items):
            mid = str(item.get("id") or f"media_{idx}")
            meta = item.get("metadata") or {}
            ai_tags = meta.get("ai_tags") or {}

            # Visual cues
            clip_emb = item.get("clip_embedding") or meta.get("clip_embedding")
            v_data = self.analyze_visual_emotion(clip_embedding=clip_emb)

            # Audio cues (if video/audio item)
            a_data = self.analyze_audio_emotion(meta.get("acoustic_features"))

            # Temporal cues
            taken_at = item.get("taken_at") or meta.get("taken_at")
            temp_data = self.analyze_temporal_emotion(taken_at)

            # Fuse for item
            item_dist = self.fuse_emotions(
                visual_data=v_data,
                audio_data=a_data,
                text_data=global_text_data,
                temporal_data=temp_data,
                ablation_mode=ablation_mode,
            )

            all_emotions.append(item_dist.dominant_emotion)

            # Pacing and motion modifiers
            modifier = self._calc_pacing_mod(item_dist.dominant_emotion, item_dist.confidence)
            duration = base_scene_duration * modifier

            motion_intensity = 1.0
            if item_dist.dominant_emotion in ("excitement", "joy"):
                motion_intensity = 1.3
            elif item_dist.dominant_emotion in ("calm", "nostalgia"):
                motion_intensity = 0.7

            transition = "crossfade"
            if item_dist.dominant_emotion in ("calm", "nostalgia"):
                transition = "dissolve"
            elif item_dist.dominant_emotion == "excitement":
                transition = "cut"

            keypoint = EmotionTimelineKeypoint(
                timestamp_offset_sec=round(current_offset, 2),
                media_id=mid,
                dominant_emotion=item_dist.dominant_emotion,
                intensity=round(item_dist.confidence, 3),
                distribution=item_dist.distribution,
                recommended_duration_sec=round(duration, 2),
                motion_intensity=motion_intensity,
                transition_type=transition,
            )
            keypoints.append(keypoint)
            current_offset += duration

        # Compute overall mood by plurality
        from collections import Counter
        counts = Counter(all_emotions)
        overall_mood = counts.most_common(1)[0][0] if counts else "calm"
        avg_intensity = float(np.mean([kp.intensity for kp in keypoints])) if keypoints else 0.5

        return EmotionTimeline(
            keypoints=keypoints,
            overall_mood=overall_mood,
            average_intensity=round(avg_intensity, 3),
        )

    def _calc_pacing_mod(self, emotion: str, confidence: float) -> float:
        if emotion in ("excitement", "joy"):
            return max(0.75, 1.0 - (confidence * 0.25))
        elif emotion in ("calm", "nostalgia", "love"):
            return min(1.35, 1.0 + (confidence * 0.30))
        return 1.0

    def analyze_memory_media(
        self,
        media_items: List[Dict[str, Any]],
        memory_title: str,
        memory_description: Optional[str] = None,
    ) -> EmotionDistribution:
        """
        Fuses available text description, EXIF timestamps, and metadata tags into an EmotionDistribution.
        """
        text_content = f"{memory_title} {memory_description or ''}"
        text_data = self.analyze_text_emotion(text_content)

        temporal_data = None
        if media_items:
            first_taken = media_items[0].get("taken_at") or media_items[0].get("created_at")
            if first_taken:
                temporal_data = self.analyze_temporal_emotion(str(first_taken))

        return self.fuse_emotions(
            text_data=text_data,
            temporal_data=temporal_data,
            ablation_mode=EmotionAblationMode.E_FULL_MULTIMODAL_TEMPORAL,
        )


emotion_analyzer = EmotionAnalyzer()

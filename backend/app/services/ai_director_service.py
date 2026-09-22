"""
MemoryVerse - AI Memory Director Orchestrator Service
Coordinates conversational memory direction, context assembly, emotion analysis,
VideoSpecification validation, generation approval, and revision management.
"""
import copy
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, cast

from app.core.cache_service import cache_service
from app.core.db import get_supabase_client
from app.schemas.ai_session import (
    AIMessage,
    AISessionResponse,
    EmotionSpec,
    MemoryContext,
    MusicSpec,
    NarrationSpec,
    VideoRevisionResponse,
    VideoSpecification,
)
from app.services.job_state_manager import job_state_manager
from app.services.revision_engine import revision_engine
from app.services.video_queue import video_worker_pool
from ai_engine.emotion.emotion_analyzer import emotion_analyzer

logger = logging.getLogger(__name__)


class AIDirectorService:
    """
    Intelligent orchestration layer for conversational memory movie creation.
    """

    # In-memory session store with Redis/DB fallback
    _sessions: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def create_or_get_session(
        cls,
        user_id: str,
        memory_id: str,
        initial_prompt: Optional[str] = None,
    ) -> AISessionResponse:
        """
        Initializes or restores an AI director session anchored to a memory.
        """
        # Look for existing active session for this user and memory
        session_key = f"{user_id}:{memory_id}"
        for s_id, s_data in cls._sessions.items():
            if s_data.get("session_key") == session_key:
                return cls._to_response(s_data)

        # 1. Build MemoryContext from existing database & indexed representations
        context = cls._build_memory_context(user_id, memory_id)

        # 2. Derive initial context-aware suggestions
        suggestions = cls._generate_contextual_suggestions(context)

        # 3. Formulate default VideoSpecification
        spec = cls._build_default_specification(context)

        session_id = f"aisess_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        # 4. Formulate initial greeting message
        welcome_content = (
            f"I found {context.photo_count} photos and {context.video_count} videos from \"{context.title}\".\n\n"
            f"I can direct a warm, {spec.emotion.primary} memory movie highlighting the best moments first "
            f"with a smooth nostalgic finish (~{spec.duration_target_seconds}s in {spec.aspect_ratio}).\n\n"
            f"Would you like to customize the music, pacing, or story before I generate it?"
        )

        initial_message = AIMessage(
            id=f"msg_{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            role="assistant",
            content=welcome_content,
            suggestions=["Generate Video", "Make it more emotional", "Make it 20 seconds", "Change music to acoustic"],
            specification=spec,
            created_at=now,
        )

        session_data: Dict[str, Any] = {
            "session_id": session_id,
            "session_key": session_key,
            "user_id": user_id,
            "memory_id": memory_id,
            "title": f"Movie: {context.title}",
            "context": context,
            "specification": spec,
            "messages": [initial_message],
            "suggestions": suggestions,
            "active_job_id": None,
            "created_at": now,
            "updated_at": now,
        }

        cls._sessions[session_id] = session_data

        # If user passed an initial prompt upon entry, process it immediately
        if initial_prompt and initial_prompt.strip():
            return cls.process_user_message(session_id, user_id, initial_prompt.strip())

        return cls._to_response(session_data)

    @classmethod
    def get_session(cls, session_id: str, user_id: str) -> Optional[AISessionResponse]:
        s_data = cls._sessions.get(session_id)
        if not s_data or s_data.get("user_id") != user_id:
            return None
        return cls._to_response(s_data)

    @classmethod
    def process_user_message(
        cls,
        session_id: str,
        user_id: str,
        user_text: str,
    ) -> AISessionResponse:
        """
        Processes conversational interaction with the AI Memory Director.
        Updates VideoSpecification deterministically or with LLM assistance.
        """
        s_data = cls._sessions.get(session_id)
        if not s_data or s_data.get("user_id") != user_id:
            raise ValueError("AI Director session not found or access unauthorized.")

        now = datetime.now(timezone.utc)
        messages: List[AIMessage] = s_data["messages"]
        current_spec: VideoSpecification = s_data["specification"]

        # Record user message
        user_msg = AIMessage(
            id=f"msg_{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            role="user",
            content=user_text,
            created_at=now,
        )
        messages.append(user_msg)

        # Mutate specification and compose AI answer
        updated_spec, reply_text, suggestions = cls._interpret_and_mutate_spec(
            user_text=user_text,
            spec=current_spec,
            context=s_data["context"],
        )

        s_data["specification"] = updated_spec
        s_data["updated_at"] = now

        ai_msg = AIMessage(
            id=f"msg_{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            role="assistant",
            content=reply_text,
            suggestions=suggestions,
            specification=updated_spec,
            created_at=now,
        )
        messages.append(ai_msg)

        return cls._to_response(s_data)

    @classmethod
    def approve_and_generate(
        cls,
        session_id: str,
        user_id: str,
        custom_spec: Optional[VideoSpecification] = None,
    ) -> Tuple[AISessionResponse, str]:
        """
        Explicit user approval step. Initiates durable VideoJob with the approved VideoSpecification.
        """
        s_data = cls._sessions.get(session_id)
        if not s_data or s_data.get("user_id") != user_id:
            raise ValueError("AI Director session not found or access unauthorized.")

        spec = custom_spec or s_data["specification"]
        s_data["specification"] = spec
        context: MemoryContext = s_data["context"]

        # 1. Deterministic job deduplication fingerprint
        fingerprint = cache_service.compute_job_fingerprint(
            user_id=user_id,
            memory_id=context.memory_id,
            selected_media_ids=spec.media_selection or context.media_ids,
            mood=spec.emotion.primary,
            dimension=spec.aspect_ratio,
        )

        existing_job_id = cache_service.get_active_job_by_fingerprint(fingerprint)
        if existing_job_id:
            active_job = job_state_manager.get_job(existing_job_id)
            if active_job and active_job.status in ("queued", "initializing", "processing"):
                s_data["active_job_id"] = existing_job_id
                return cls._to_response(s_data), existing_job_id

        # 2. Persist in database
        supabase = get_supabase_client()
        metadata_payload = {
            "specification": spec.model_dump(),
            "dimension": spec.aspect_ratio,
            "mood": spec.emotion.primary,
            "duration_seconds": spec.duration_target_seconds,
            "current_task": "Waiting for available background worker...",
            "stage": "queued",
            "overall_progress": 0,
        }
        job_res = (
            supabase.table("video_jobs")
            .insert({
                "memory_id": context.memory_id,
                "user_id": user_id,
                "status": "queued",
                "script_metadata": metadata_payload,
            })
            .execute()
        )

        raw_jobs = cast(List[Dict[str, Any]], job_res.data)
        job_id = str(raw_jobs[0]["id"])

        # 3. Create durable JobState
        job_state_manager.create_job(
            job_id=job_id,
            memory_id=context.memory_id,
            user_id=user_id,
            dimension=spec.aspect_ratio,
            mood=spec.emotion.primary,
            selected_media_ids=spec.media_selection or context.media_ids,
            specification=spec.model_dump(),
        )

        cache_service.set_active_job_by_fingerprint(fingerprint, job_id, ttl=600)
        video_worker_pool.submit_job(job_id)

        s_data["active_job_id"] = job_id
        now = datetime.now(timezone.utc)
        s_data["updated_at"] = now

        # Add assistant acknowledgment with job link
        confirm_msg = AIMessage(
            id=f"msg_{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            role="assistant",
            content=f"I've started creating your memory movie ({spec.duration_target_seconds}s, {spec.aspect_ratio}). You can follow the live progress below.",
            suggestions=["Hide", "Check Status"],
            specification=spec,
            job_id=job_id,
            created_at=now,
        )
        s_data["messages"].append(confirm_msg)

        logger.info(f"AI Director started video job {job_id} for session {session_id}")
        return cls._to_response(s_data), job_id

    @classmethod
    def request_revision(
        cls,
        session_id: str,
        user_id: str,
        revision_prompt: str,
    ) -> Tuple[AISessionResponse, VideoRevisionResponse]:
        """
        Applies a natural language revision, diffs specifications, and triggers selective regeneration.
        """
        s_data = cls._sessions.get(session_id)
        if not s_data or s_data.get("user_id") != user_id:
            raise ValueError("AI Director session not found.")

        old_spec: VideoSpecification = s_data["specification"]
        new_spec, reply_text, _ = cls._interpret_and_mutate_spec(
            user_text=revision_prompt,
            spec=old_spec,
            context=s_data["context"],
        )

        # Generate revision diff
        revision = revision_engine.create_revision(
            parent_revision_id=None,
            job_id=s_data.get("active_job_id") or f"rev_job_{uuid.uuid4().hex[:8]}",
            old_spec=old_spec,
            new_spec=new_spec,
        )

        s_data["specification"] = new_spec
        now = datetime.now(timezone.utc)
        s_data["updated_at"] = now

        user_msg = AIMessage(
            id=f"msg_{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            role="user",
            content=revision_prompt,
            created_at=now,
        )
        ai_msg = AIMessage(
            id=f"msg_{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            role="assistant",
            content=f"{reply_text}\n\nI'll selectively re-render only the affected stages ({', '.join(revision.invalidated_stages)}).",
            suggestions=["Generate Revision", "Keep Previous Video"],
            specification=new_spec,
            created_at=now,
        )
        s_data["messages"].extend([user_msg, ai_msg])

        return cls._to_response(s_data), revision

    # ── Internal Helper Methods ────────────────────────────────────────────────

    @classmethod
    def _build_memory_context(cls, user_id: str, memory_id: str) -> MemoryContext:
        """Fetches memory metadata, media records, and cached embeddings."""
        supabase = get_supabase_client()

        # Fetch memory record
        mem_res = supabase.table("memories").select("*").eq("id", memory_id).execute()
        mem_data = cast(List[Dict[str, Any]], mem_res.data or [])
        mem = mem_data[0] if mem_data else {}

        title = str(mem.get("title") or "Memory Movie")
        desc = mem.get("description")
        event_date = str(mem.get("event_date") or mem.get("created_at") or "")[:10]
        loc = mem.get("location_name")

        # Fetch media
        media_res = (
            supabase.table("media")
            .select("*")
            .eq("memory_id", memory_id)
            .order("created_at", desc=False)
            .limit(60)
            .execute()
        )
        raw_media = cast(List[Dict[str, Any]], media_res.data or [])
        media_items = [m for m in raw_media if "reels/" not in (m.get("storage_path") or "")]

        media_ids = [str(m.get("id")) for m in media_items]
        photos = [m for m in media_items if (m.get("media_type") or "image").lower() == "image"]
        videos = [m for m in media_items if (m.get("media_type") or "").lower() == "video"]

        # Run or reuse emotion analysis
        emotion_dist = emotion_analyzer.analyze_memory_media(
            media_items=media_items,
            memory_title=title,
            memory_description=desc,
        )

        return MemoryContext(
            memory_id=memory_id,
            user_id=user_id,
            title=title,
            description=desc,
            event_date=event_date,
            location_name=loc,
            media_count=len(media_items),
            photo_count=len(photos),
            video_count=len(videos),
            media_ids=media_ids,
            candidate_media=media_items,
            semantic_topics=cls._extract_semantic_topics(media_items, title),
            emotion_distribution=emotion_dist.distribution,
            dominant_emotion=emotion_dist.dominant_emotion,
            emotion_confidence=emotion_dist.confidence,
        )

    @classmethod
    def _generate_contextual_suggestions(cls, context: MemoryContext) -> List[str]:
        """
        Dynamically generates suggestions mirroring the layout of image.png
        based on the actual memory's media and context.
        """
        suggestions: List[str] = []

        # 1. Location or Event themed suggestion
        if context.location_name:
            suggestions.append(f"Trip to {context.location_name}")
        elif "trip" in context.title.lower() or "travel" in context.title.lower():
            suggestions.append("Travel diary, with epic music")
        elif "birthday" in context.title.lower() or "party" in context.title.lower():
            suggestions.append("Joyful celebration with friends")
        else:
            suggestions.append("Time at home with friends")

        # 2. Seasonal / Aesthetic themed suggestion
        if "spring" in context.title.lower() or "nature" in context.title.lower():
            suggestions.append("Spring flowers to remember")
        elif "summer" in context.title.lower() or "beach" in context.title.lower():
            suggestions.append("Sunny days and warm vibes")
        elif "winter" in context.title.lower() or "snow" in context.title.lower():
            suggestions.append("Cozy winter memories")
        else:
            suggestions.append("Sweet moments to remember")

        # 3. Emotion / Music themed suggestion
        if context.dominant_emotion in ("joy", "excitement"):
            suggestions.append("Joyful holidays, with epic music")
        elif context.dominant_emotion == "nostalgia":
            suggestions.append("Warm and nostalgic documentary")
        else:
            suggestions.append("Calm reflection with acoustic melody")

        return suggestions

    @classmethod
    def _build_default_specification(cls, context: MemoryContext) -> VideoSpecification:
        """Initializes a reasonable default VideoSpecification."""
        return VideoSpecification(
            version="1.0",
            duration_target_seconds=min(45, max(20, context.media_count * 3)),
            aspect_ratio="16:9",
            orientation="landscape",
            media_selection=context.media_ids[:15],
            selection_strategy="ai_curated",
            story_structure="three_act_arc",
            emotion=EmotionSpec(
                primary=context.dominant_emotion,
                distribution=context.emotion_distribution,
                confidence=context.emotion_confidence,
            ),
            music=MusicSpec(
                style="cinematic_warm" if context.dominant_emotion in ("nostalgia", "calm") else "energetic",
                tempo=95 if context.dominant_emotion in ("joy", "excitement") else 75,
                energy=0.7 if context.dominant_emotion in ("joy", "excitement") else 0.5,
            ),
            narration=NarrationSpec(enabled=False),
        )

    @classmethod
    def _interpret_and_mutate_spec(
        cls,
        user_text: str,
        spec: VideoSpecification,
        context: MemoryContext,
    ) -> Tuple[VideoSpecification, str, List[str]]:
        """
        Parses natural language user guidance and updates VideoSpecification deterministically.
        """
        new_spec = copy.deepcopy(spec)
        text = user_text.lower().strip()
        modifications: List[str] = []

        # Duration modifications
        dur_match = re.search(r"(\d+)\s*(?:s|sec|second)", text)
        if dur_match:
            sec = int(dur_match.group(1))
            sec = max(10, min(120, sec))
            new_spec.duration_target_seconds = sec
            modifications.append(f"set duration to {sec}s")
        elif "shorter" in text or "fast" in text:
            new_spec.duration_target_seconds = max(15, new_spec.duration_target_seconds - 10)
            modifications.append(f"shortened duration to {new_spec.duration_target_seconds}s")
        elif "longer" in text or "extend" in text:
            new_spec.duration_target_seconds = min(90, new_spec.duration_target_seconds + 15)
            modifications.append(f"extended duration to {new_spec.duration_target_seconds}s")

        # Aspect ratio modifications
        if "4:3" in text or "four by three" in text or "retro" in text:
            new_spec.aspect_ratio = "4:3"
            new_spec.orientation = "landscape"
            modifications.append("switched to 4:3 retro format")
        elif "vertical" in text or "9:16" in text or "reel" in text or "story" in text or "tiktok" in text:
            new_spec.aspect_ratio = "9:16"
            new_spec.orientation = "portrait"
            modifications.append("switched to 9:16 vertical format")
        elif "landscape" in text or "16:9" in text or "horizontal" in text or "widescreen" in text:
            new_spec.aspect_ratio = "16:9"
            new_spec.orientation = "landscape"
            modifications.append("switched to 16:9 landscape format")
        elif "square" in text or "1:1" in text:
            new_spec.aspect_ratio = "1:1"
            new_spec.orientation = "square"
            modifications.append("switched to 1:1 square format")

        # Mood & Emotion modifications
        if "emotional" in text or "nostalgic" in text or "sentiment" in text:
            new_spec.emotion.primary = "nostalgia"
            new_spec.music.style = "cinematic_warm"
            new_spec.music.energy = 0.5
            modifications.append("increased nostalgic emotional warmth")
        elif "happy" in text or "energetic" in text or "upbeat" in text or "fun" in text:
            new_spec.emotion.primary = "joy"
            new_spec.music.style = "energetic"
            new_spec.music.energy = 0.8
            modifications.append("tuned vibe to joyful and energetic")
        elif "calm" in text or "peaceful" in text or "slow" in text:
            new_spec.emotion.primary = "calm"
            new_spec.music.style = "minimal"
            new_spec.music.energy = 0.35
            modifications.append("slowed pacing and softened audio")

        # Music modifications
        if "no music" in text or "remove music" in text or "mute" in text:
            new_spec.music.style = "none"
            new_spec.music.energy = 0.0
            modifications.append("disabled background music")
        elif "acoustic" in text:
            new_spec.music.style = "acoustic"
            new_spec.music.energy = 0.5
            modifications.append("changed music to acoustic guitar")
        elif "epic" in text or "cinematic" in text:
            new_spec.music.style = "cinematic_warm"
            new_spec.music.energy = 0.75
            modifications.append("changed music to epic cinematic soundtrack")
        elif "quieter" in text or "softer" in text or "lower volume" in text:
            new_spec.music.energy = max(0.15, new_spec.music.energy - 0.25)
            modifications.append("softened soundtrack volume")

        # Narration modifications
        if "narration" in text or "voiceover" in text or "narrate" in text:
            if "remove" in text or "no" in text or "disable" in text or "off" in text:
                new_spec.narration.enabled = False
                modifications.append("disabled voice narration")
            else:
                new_spec.narration.enabled = True
                if "calm" in text or "soft" in text:
                    new_spec.narration.style = "calm"
                    modifications.append("enabled calm story narration")
                else:
                    modifications.append("enabled AI story narration")

        # Media filters & overrides
        if "only photo" in text or "only images" in text or "photos only" in text:
            photos = [m for m in context.candidate_media if (m.get("media_type") or "image").lower() == "image"]
            new_spec.media_selection = [str(m.get("id")) for m in photos]
            new_spec.selection_strategy = "photos_only"
            modifications.append(f"filtered to only photos ({len(photos)} items)")
        elif "only video" in text or "videos only" in text:
            videos = [m for m in context.candidate_media if (m.get("media_type") or "").lower() == "video"]
            if videos:
                new_spec.media_selection = [str(m.get("id")) for m in videos]
                new_spec.selection_strategy = "videos_only"
                modifications.append(f"filtered to only videos ({len(videos)} items)")
            else:
                modifications.append("retained photos (no video clips found in memory)")
        elif "remove the second" in text or "remove second" in text:
            if len(new_spec.media_selection) > 1:
                new_spec.media_selection.pop(1)
                modifications.append("removed second media item from story")
        elif "more photos" in text or "use all" in text or "all media" in text:
            new_spec.media_selection = context.media_ids
            new_spec.selection_strategy = "use_all"
            modifications.append(f"included all {len(context.media_ids)} media items")

        # Quality profile
        if "high quality" in text or "hd" in text or "1080p" in text:
            new_spec.quality_profile = "high_quality"
            modifications.append("set quality profile to High Quality (1080p)")
        elif "fast" in text or "quick" in text:
            new_spec.quality_profile = "fast"
            modifications.append("set quality profile to Fast")

        # Evidence-backed questions (Media Count, Why media, Storyboard, Insights, Dates, Mood)
        if any(p in text for p in [
            "number of videos", "number of photos", "how many photo", "how many video",
            "how many item", "how many moment", "how many media", "give the number",
            "count of", "inventory", "total media", "how many"
        ]) and ("photo" in text or "video" in text or "media" in text or "moment" in text or "item" in text):
            photo_txt = f"{context.photo_count} photo{'s' if context.photo_count != 1 else ''}"
            video_txt = f"{context.video_count} video{'s' if context.video_count != 1 else ''}"
            reply = f"This memory contains {context.media_count} moments in total: {photo_txt} and {video_txt}."
            return new_spec, reply, ["Generate Video", "Use only photos", "Use only videos", "Switch aspect ratio"]

        if any(p in text for p in ["when was", "what date", "timeline", "when were", "what time"]):
            reply = f"These moments were captured on {context.event_date or 'recent dates'}."
            return new_spec, reply, ["Generate Video", "Change duration", "Switch aspect ratio", "Add narration"]

        if any(p in text for p in ["what emotion", "what mood", "feeling", "emotional tone"]):
            reply = f"The dominant emotion is {context.dominant_emotion.capitalize()} ({int(context.emotion_confidence * 100)}% confidence)."
            if context.emotion_distribution:
                emotions_list = list(context.emotion_distribution.keys())[:3]
                reply += f" Detected emotions across scenes: {', '.join(emotions_list)}."
            return new_spec, reply, ["Make it more emotional", "Make it joyful", "Generate Video", "Change music to acoustic"]

        if "why did you choose" in text or "why did you select" in text or "why this" in text or "why these" in text or "reason" in text or ("why" in text and ("photo" in text or "media" in text or "select" in text or "choose" in text)):
            reply = f"I selected {len(new_spec.media_selection or context.media_ids[:5])} moments based on visual quality, emotional resonance with '{context.dominant_emotion}', and chronological milestones."
            return new_spec, reply, ["Generate Video", "Change duration", "Switch aspect ratio", "Add narration"]

        if "storyboard" in text or "scenes" in text:
            reply = f"Storyboard Plan for '{context.title}':\n" \
                    f"• Scene 1: Opening establishing shot (~3.5s)\n" \
                    f"• Scene 2: People & core action (~4.0s)\n" \
                    f"• Scene 3: Emotional peak moment (~4.5s)\n" \
                    f"• Scene 4: Highlights & activity (~4.0s)\n" \
                    f"• Scene 5: Gentle resolution outro (~4.0s)\n" \
                    f"Total: ~{new_spec.duration_target_seconds}s in {new_spec.aspect_ratio}."
            return new_spec, reply, ["Generate Video", "Change music to acoustic", "Switch aspect ratio", "Make it 20 seconds"]

        if "insight" in text:
            reply = f"Memory Insights for '{context.title}':\n" \
                    f"• Moments: {context.media_count} items ({context.photo_count} photos, {context.video_count} videos)\n" \
                    f"• Dominant Emotion: {context.dominant_emotion.capitalize()} ({int(context.emotion_confidence * 100)}%)\n" \
                    f"• Date: {context.event_date or 'Recent'}"
            return new_spec, reply, ["Generate Video", "Make it 20 seconds", "Switch aspect ratio", "Add narration"]

        if not modifications:
            if any(w in text for w in ["music", "song", "audio", "track"]):
                reply = f"The current soundtrack style is {new_spec.music.style or 'warm acoustic'}. Would you like me to switch to acoustic, cinematic, energetic, or mute?"
                follow_up_suggestions = ["Change music to acoustic", "Change music to cinematic", "No music", "Generate Video"]
            elif any(w in text for w in ["aspect", "ratio", "format", "vertical", "horizontal"]):
                reply = f"The current format is {new_spec.aspect_ratio}. I can switch to 9:16 (Instagram Reel / TikTok), 16:9 (Landscape YouTube), 1:1 (Square), or 4:3."
                follow_up_suggestions = ["Switch aspect ratio to 9:16", "Switch aspect ratio to 16:9", "Switch aspect ratio to 1:1", "Generate Video"]
            elif any(w in text for w in ["duration", "length", "seconds", "how long"]):
                reply = f"The target duration is ~{new_spec.duration_target_seconds} seconds. Would you like a 15s quick highlight, 20s story, or 30s recap?"
                follow_up_suggestions = ["Make it 15 seconds", "Make it 20 seconds", "Make it 30 seconds", "Generate Video"]
            else:
                reply = f"I've tailored the plan around '{user_text}'. What would you like to customize next — music, pacing, aspect ratio, or shall we generate?"
                follow_up_suggestions = ["Generate Video", "Change duration", "Switch aspect ratio", "Add narration"]
        else:
            reply = f"I've updated the plan: {', '.join(modifications)}. Ready to generate whenever you like!"
            follow_up_suggestions = ["Generate Video", "Change duration", "Switch aspect ratio", "Add narration"]

        return new_spec, reply, follow_up_suggestions

    @classmethod
    def _extract_semantic_topics(cls, media_items: List[Dict[str, Any]], title: str) -> List[str]:
        topics = [title.lower()]
        for m in media_items[:10]:
            tags = (m.get("metadata") or {}).get("ai_tags", {})
            scenes = tags.get("scenes", [])
            topics.extend([s.lower() for s in scenes[:2]])
        return list(dict.fromkeys(topics))[:6]

    @classmethod
    def _to_response(cls, s_data: Dict[str, Any]) -> AISessionResponse:
        return AISessionResponse(
            session_id=s_data["session_id"],
            memory_id=s_data["memory_id"],
            user_id=s_data["user_id"],
            title=s_data["title"],
            specification=s_data["specification"],
            context=s_data["context"],
            messages=s_data["messages"],
            suggestions=s_data["suggestions"],
            active_job_id=s_data["active_job_id"],
            created_at=s_data["created_at"],
            updated_at=s_data["updated_at"],
        )


ai_director_service = AIDirectorService()

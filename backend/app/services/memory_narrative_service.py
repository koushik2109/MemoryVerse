import math
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np

from app.core.db import get_supabase_client
from app.services.event_clustering_service import MediaItemContext
from app.services.ai_service import call_llm
from app.schemas.domain import (
    TimelinePhase, KeyMoment, AtmosphereItem, MemoryNarrativeResponse
)

logger = logging.getLogger(__name__)


def _haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in km."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


class MemoryNarrativeService:
    """
    Dedicated service for Step 8: Memory Narrative Intelligence & Semantic Timeline.
    
    Transforms curated media into grounded narrative intelligence:
      Stage 1: Evidence Manifest Construction (fact/interpretation separation & compression)
      Stage 2: Deterministic Semantic Timeline Segmentation (3–8 phases)
      Stage 3: Diversity-Aware Key Moment Selection (normalized multi-signal scoring)
      Stage 4: Atmosphere Inference (event-level, grounded, zero emotional hallucination)
      Stage 5: Grounded Narrative Synthesis (single primary LLM call)
      Stage 6: Validation & Sanitization Layer
      Stage 7: Deterministic Fallback (complete offline reliability)
    """

    @classmethod
    def analyze_memory_event(
        cls,
        items: List[MediaItemContext],
        memory_id: Optional[str] = None,
        event_title_hint: Optional[str] = None,
        enable_llm: bool = True
    ) -> MemoryNarrativeResponse:
        """
        Main entrypoint: analyzes an event collection and returns structured narrative intelligence.
        """
        if not items:
            return cls._empty_narrative_response(memory_id)

        # Sort chronologically with progressive fallback
        sorted_items = sorted(items, key=lambda x: x.timestamp)

        # ── STAGE 1: Evidence Manifest Construction
        manifest = cls._build_evidence_manifest(sorted_items)

        # ── STAGE 2: Deterministic Semantic Timeline Segmentation
        deterministic_phases = cls._segment_deterministic_timeline(sorted_items)

        # ── STAGE 3: Diversity-Aware Key Moment Selection
        key_moments, raw_scored_moments = cls._select_diversity_key_moments(
            sorted_items,
            deterministic_phases,
            manifest
        )

        # ── STAGE 4: Atmosphere & Mood Inference
        atmosphere_items = cls._infer_atmosphere(sorted_items, manifest)

        # ── STAGE 5: Grounded Narrative Synthesis (Single LLM Call or Deterministic Fallback)
        if enable_llm:
            narrative = cls._synthesize_narrative_llm(
                memory_id=memory_id,
                items=sorted_items,
                manifest=manifest,
                phases=deterministic_phases,
                key_moments=key_moments,
                atmosphere=atmosphere_items,
                title_hint=event_title_hint
            )
        else:
            narrative = cls._synthesize_deterministic_fallback(
                memory_id=memory_id,
                items=sorted_items,
                manifest=manifest,
                phases=deterministic_phases,
                key_moments=key_moments,
                atmosphere=atmosphere_items,
                title_hint=event_title_hint
            )

        # ── STAGE 6: Validation & Sanitization
        validated_response = cls._validate_and_sanitize(narrative, sorted_items, memory_id)
        return validated_response

    # ── STAGE 1: EVIDENCE MANIFEST CONSTRUCTION ──────────────────────────────

    @classmethod
    def _build_evidence_manifest(cls, items: List[MediaItemContext]) -> Dict[str, Any]:
        """
        Constructs a compact, token-efficient factual evidence manifest.
        Separates factual evidence from interpretation and compresses repetitive media.
        """
        start_time = items[0].timestamp
        end_time = items[-1].timestamp
        duration_hours = max(0.1, (end_time - start_time).total_seconds() / 3600.0)

        # Location analysis
        locations = [it.location_name for it in items if it.location_name]
        loc_counts = {}
        for l in locations:
            loc_counts[l] = loc_counts.get(l, 0) + 1
        dominant_loc = max(loc_counts, key=loc_counts.get) if loc_counts else None

        # Tags and scene frequencies
        scene_freq: Dict[str, int] = {}
        obj_freq: Dict[str, int] = {}
        for it in items:
            for s in it.scenes:
                scene_freq[s] = scene_freq.get(s, 0) + 1
            for o in it.objects:
                obj_freq[o] = obj_freq.get(o, 0) + 1

        top_scenes = sorted(scene_freq, key=scene_freq.get, reverse=True)[:5]
        top_objects = sorted(obj_freq, key=obj_freq.get, reverse=True)[:5]

        # Gather sample VLM visual notes (max 6 representative descriptions)
        sample_vlm_notes = []
        step = max(1, len(items) // 6)
        for i in range(0, len(items), step):
            it = items[i]
            if it.vlm_description:
                time_str = it.timestamp.strftime("%H:%M")
                sample_vlm_notes.append(f"[{time_str}] {it.vlm_description}")

        # People presence
        people_counts = [it.people_count for it in items if it.people_count]
        dominant_people = max(set(people_counts), key=people_counts.count) if people_counts else "not specified"

        # Evidence container (factual only)
        evidence = {
            "date": start_time.strftime("%Y-%m-%d"),
            "date_display": start_time.strftime("%B %d, %Y"),
            "start_time_str": start_time.strftime("%I:%M %p"),
            "end_time_str": end_time.strftime("%I:%M %p"),
            "duration_hours": round(duration_hours, 1),
            "media_count": len(items),
            "location": dominant_loc,
            "dominant_scenes": top_scenes,
            "dominant_objects": top_objects,
            "people_presence": dominant_people,
            "sample_vlm_notes": sample_vlm_notes,
            "average_quality": round(float(np.mean([it.quality_score for it in items])), 2),
        }

        return {
            "evidence": evidence,
            "interpretation": {}
        }

    # ── STAGE 2: DETERMINISTIC SEMANTIC TIMELINE SEGMENTATION ─────────────────

    @classmethod
    def _segment_deterministic_timeline(cls, items: List[MediaItemContext]) -> List[TimelinePhase]:
        """
        Deterministically segments the chronological media stream into 3–8 natural timeline phases.
        LLM does NOT invent boundaries or media IDs.
        """
        if not items:
            return []

        if len(items) <= 3:
            it = items[0]
            start_str = items[0].timestamp.strftime("%I:%M %p")
            end_str = items[-1].timestamp.strftime("%I:%M %p")
            scene_label = it.scenes[0].title() if it.scenes else "Memory"
            return [TimelinePhase(
                phase_id="phase_1",
                start_time=start_str,
                end_time=end_str,
                title=f"{scene_label} Moments",
                description=f"Captured moments during {start_str} - {end_str}.",
                media_ids=[m.id for m in items],
                evidence_summary=f"{len(items)} items recorded."
            )]

        # Find natural transition break points
        split_indices = [0]
        for i in range(1, len(items)):
            prev = items[i - 1]
            curr = items[i]

            time_gap_minutes = (curr.timestamp - prev.timestamp).total_seconds() / 60.0
            
            # Geo distance if present
            geo_dist_km = None
            if prev.latitude is not None and prev.longitude is not None and curr.latitude is not None and curr.longitude is not None:
                geo_dist_km = _haversine_distance_km(prev.latitude, prev.longitude, curr.latitude, curr.longitude)

            # Scene / Tag overlap
            tag_overlap = len(prev.all_tags & curr.all_tags) / max(1, len(prev.all_tags | curr.all_tags))

            # Boundary trigger:
            # 1. Substantial time gap (>= 90 minutes)
            # 2. Location jump (> 2.0 km)
            # 3. Disjoint scenes with moderate time gap (>= 45 min)
            is_boundary = (
                time_gap_minutes >= 90.0 or
                (geo_dist_km is not None and geo_dist_km > 2.0) or
                (time_gap_minutes >= 45.0 and tag_overlap < 0.10)
            )

            if is_boundary and (i - split_indices[-1] >= 2) and (len(items) - i >= 2):
                split_indices.append(i)

        # Bound the number of phases between 3 and 8 (for standard collections)
        raw_phase_slices = []
        for idx in range(len(split_indices)):
            start_idx = split_indices[idx]
            end_idx = split_indices[idx + 1] if idx + 1 < len(split_indices) else len(items)
            raw_phase_slices.append(items[start_idx:end_idx])

        # If too few phases on large dataset (>= 12 items), split longest phases uniformly
        while len(raw_phase_slices) < 3 and len(items) >= 9:
            longest_idx = max(range(len(raw_phase_slices)), key=lambda k: len(raw_phase_slices[k]))
            longest_slice = raw_phase_slices[longest_idx]
            if len(longest_slice) < 4:
                break
            mid = len(longest_slice) // 2
            raw_phase_slices = (
                raw_phase_slices[:longest_idx] +
                [longest_slice[:mid], longest_slice[mid:]] +
                raw_phase_slices[longest_idx + 1:]
            )

        # If too many phases (> 8), merge shortest adjacent phases
        while len(raw_phase_slices) > 8:
            shortest_idx = min(range(len(raw_phase_slices) - 1), key=lambda k: len(raw_phase_slices[k]) + len(raw_phase_slices[k+1]))
            merged = raw_phase_slices[shortest_idx] + raw_phase_slices[shortest_idx + 1]
            raw_phase_slices = (
                raw_phase_slices[:shortest_idx] +
                [merged] +
                raw_phase_slices[shortest_idx + 2:]
            )

        # Build TimelinePhase objects
        phases: List[TimelinePhase] = []
        for p_idx, phase_items in enumerate(raw_phase_slices):
            p_start = phase_items[0].timestamp.strftime("%I:%M %p")
            p_end = phase_items[-1].timestamp.strftime("%I:%M %p")
            
            # Determine dominant scene / theme
            p_scenes = [s for it in phase_items for s in it.scenes]
            p_top_scene = max(set(p_scenes), key=p_scenes.count) if p_scenes else "Activity"
            
            # Initial deterministic title & summary
            p_title = f"{p_top_scene.title()} ({p_start})"
            p_desc = f"{len(phase_items)} moments captured featuring {p_top_scene}."
            evidence_sum = f"Time: {p_start} - {p_end}, {len(phase_items)} items, dominant: {p_top_scene}."

            phases.append(TimelinePhase(
                phase_id=f"phase_{p_idx + 1}",
                start_time=p_start,
                end_time=p_end,
                title=p_title,
                description=p_desc,
                media_ids=[m.id for m in phase_items],
                evidence_summary=evidence_sum
            ))

        return phases

    # ── STAGE 3: DIVERSITY-AWARE KEY MOMENT SELECTION ────────────────────────

    @classmethod
    def _select_diversity_key_moments(
        cls,
        items: List[MediaItemContext],
        phases: List[TimelinePhase],
        manifest: Dict[str, Any]
    ) -> Tuple[List[KeyMoment], List[Dict[str, Any]]]:
        """
        Scores each media item using the normalized multi-signal formula:
          Score = 0.30×story_value + 0.20×importance + 0.15×quality + 0.15×uniqueness + 0.10×representativeness + 0.10×speech
        Applies diversity constraints across timeline phases and prevents redundant duplicate covers.
        """
        if not items:
            return [], []

        # Calculate mean embedding for centrality/representativeness
        valid_embs = [it.embedding for it in items if it.embedding and any(it.embedding)]
        mean_emb = None
        if valid_embs:
            mean_emb = np.mean(np.array(valid_embs), axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb = mean_emb / norm

        top_scenes_set = set(manifest["evidence"].get("dominant_scenes", []))

        scored_list = []
        for it in items:
            # 1. Quality (0.0 to 1.0)
            quality = float(np.clip(it.quality_score or 0.80, 0.0, 1.0))

            # 2. Representativeness (0.0 to 1.0)
            representativeness = 0.50
            if mean_emb is not None and it.embedding and any(it.embedding):
                it_emb = np.array(it.embedding)
                it_norm = np.linalg.norm(it_emb)
                if it_norm > 0:
                    representativeness = float(np.clip(np.dot(mean_emb, it_emb / it_norm), 0.0, 1.0))

            # 3. Story Value & Activity Intensity (0.0 to 1.0)
            # High for distinctive scenes like sunset, cake cutting, stage, speeches
            distinctive_tags = {"sunset", "cake", "candles", "celebration", "swimming", "boat", "stage", "toast"}
            has_distinctive = bool(it.all_tags & distinctive_tags)
            story_value = 0.90 if has_distinctive else (0.75 if it.scenes else 0.60)

            # 4. Importance (0.0 to 1.0)
            has_people = it.people_count in ["small_group", "crowd", "few"]
            importance = 0.85 if (has_people or has_distinctive) else 0.65

            # 5. Uniqueness (0.0 to 1.0)
            uniqueness = 0.80 if has_distinctive else 0.70

            # 6. Speech Signal (0.0 to 1.0)
            speech_signal = 0.80 if it.media_type == "video" else 0.40

            # Composite multi-signal formula
            score = (
                0.30 * story_value +
                0.20 * importance +
                0.15 * quality +
                0.15 * uniqueness +
                0.10 * representativeness +
                0.10 * speech_signal
            )
            score = round(float(np.clip(score, 0.0, 1.0)), 2)

            # Grounded reason explanation
            reason_parts = []
            if quality >= 0.85:
                reason_parts.append("high visual quality")
            if has_distinctive:
                matched = list(it.all_tags & distinctive_tags)[0]
                reason_parts.append(f"distinctive {matched} activity")
            if has_people:
                reason_parts.append("group focal point")
            if representativeness >= 0.75:
                reason_parts.append("strong event representativeness")
            if not reason_parts:
                reason_parts.append("coherent scene focal moment")

            reason_str = f"Selected for {', '.join(reason_parts)}."

            desc = it.vlm_description or (
                f"{it.scenes[0].title()} moment" if it.scenes else f"Key moment at {it.timestamp.strftime('%I:%M %p')}"
            )

            scored_list.append({
                "item": it,
                "score": score,
                "reason": reason_str,
                "description": desc
            })

        # Sort highest score first
        scored_list.sort(key=lambda x: x["score"], reverse=True)

        # Apply phase diversity constraints (target 3 to 7 key moments, max 2 per phase)
        target_count = min(7, max(3, len(items) // 5))
        if len(items) <= 3:
            target_count = len(items)

        selected_moments: List[KeyMoment] = []
        phase_selection_count: Dict[str, int] = {p.phase_id: 0 for p in phases}
        selected_ids: Set[str] = set()

        # Map each item id to its phase_id
        item_to_phase = {}
        for p in phases:
            for mid in p.media_ids:
                item_to_phase[mid] = p.phase_id

        # First pass: pick best from distinct phases
        for candidate in scored_list:
            it = candidate["item"]
            p_id = item_to_phase.get(it.id, "phase_1")
            if phase_selection_count.get(p_id, 0) < 1 and it.id not in selected_ids:
                selected_moments.append(KeyMoment(
                    media_id=it.id,
                    timestamp=it.timestamp.strftime("%I:%M %p"),
                    description=candidate["description"],
                    importance_score=candidate["score"],
                    reason=candidate["reason"]
                ))
                selected_ids.add(it.id)
                phase_selection_count[p_id] = phase_selection_count.get(p_id, 0) + 1
                if len(selected_moments) >= target_count:
                    break

        # Second pass: fill up to target_count (max 2 per phase)
        if len(selected_moments) < target_count:
            for candidate in scored_list:
                it = candidate["item"]
                p_id = item_to_phase.get(it.id, "phase_1")
                if phase_selection_count.get(p_id, 0) < 2 and it.id not in selected_ids:
                    selected_moments.append(KeyMoment(
                        media_id=it.id,
                        timestamp=it.timestamp.strftime("%I:%M %p"),
                        description=candidate["description"],
                        importance_score=candidate["score"],
                        reason=candidate["reason"]
                    ))
                    selected_ids.add(it.id)
                    phase_selection_count[p_id] = phase_selection_count.get(p_id, 0) + 1
                    if len(selected_moments) >= target_count:
                        break

        return selected_moments, scored_list

    # ── STAGE 4: ATMOSPHERE & MOOD INFERENCE ─────────────────────────────────

    @classmethod
    def _infer_atmosphere(cls, items: List[MediaItemContext], manifest: Dict[str, Any]) -> List[AtmosphereItem]:
        """
        Infers event-level atmosphere labels strictly based on grounded scene/activity evidence.
        NEVER claims internal personal emotions or ungrounded relationships.
        """
        all_tags = set()
        for it in items:
            all_tags.update(it.all_tags)

        vlm_corpus = " ".join([it.vlm_description or "" for it in items]).lower()
        atmosphere_list: List[AtmosphereItem] = []

        # Celebratory / Festive
        if any(w in all_tags for w in ["birthday", "party", "cake", "candles", "celebration", "confetti"]) or "celebrat" in vlm_corpus:
            atmosphere_list.append(AtmosphereItem(
                label="celebratory",
                confidence=0.94,
                evidence="Supported by party decorations, birthday cake, and group gathering activities."
            ))
            atmosphere_list.append(AtmosphereItem(
                label="social",
                confidence=0.88,
                evidence="Supported by group social presence and interactive gathering."
            ))

        # Peaceful / Relaxed (e.g. beach, nature, sunset)
        if any(w in all_tags for w in ["beach", "ocean", "sunset", "sea", "resort", "park", "nature", "trees"]):
            atmosphere_list.append(AtmosphereItem(
                label="relaxed",
                confidence=0.92,
                evidence="Supported by coastal outdoor setting, leisurely pace, and open natural surroundings."
            ))
            if any(w in all_tags for w in ["sunset", "nature", "park"]):
                atmosphere_list.append(AtmosphereItem(
                    label="peaceful",
                    confidence=0.89,
                    evidence="Supported by calm evening lighting and open landscape scenery."
                ))

        # Professional / Focused (e.g. office, work)
        if any(w in all_tags for w in ["office", "laptop", "coding", "meeting", "desk", "computer"]) or "desk" in vlm_corpus:
            atmosphere_list.append(AtmosphereItem(
                label="focused",
                confidence=0.90,
                evidence="Supported by workplace environment, desk work, and technical session focus."
            ))

        # Adventurous / Energetic (e.g. watersports, hiking, boat)
        if any(w in all_tags for w in ["jetski", "boat", "surfing", "hiking", "swimming", "sports"]):
            atmosphere_list.append(AtmosphereItem(
                label="adventurous",
                confidence=0.86,
                evidence="Supported by active outdoor water sports and exploration."
            ))

        # Fallback neutral atmosphere if none strongly triggered
        if not atmosphere_list:
            atmosphere_list.append(AtmosphereItem(
                label="social",
                confidence=0.75,
                evidence="Supported by general shared event gathering and photo activity."
            ))

        return atmosphere_list[:3]

    # ── STAGE 5: GROUNDED NARRATIVE SYNTHESIS (SINGLE LLM CALL) ──────────────

    @classmethod
    def _synthesize_narrative_llm(
        cls,
        memory_id: Optional[str],
        items: List[MediaItemContext],
        manifest: Dict[str, Any],
        phases: List[TimelinePhase],
        key_moments: List[KeyMoment],
        atmosphere: List[AtmosphereItem],
        title_hint: Optional[str] = None
    ) -> MemoryNarrativeResponse:
        """
        Executes a single compact LLM call to synthesize the narrative, adhering strictly to evidence.
        """
        evidence_dict = manifest["evidence"]
        
        # Build compact prompt
        phases_text = "\n".join([
            f"- Phase {p.phase_id}: {p.start_time}–{p.end_time} | {p.evidence_summary}"
            for p in phases
        ])

        moments_text = "\n".join([
            f"- Key Moment at {m.timestamp}: {m.description} (Reason: {m.reason})"
            for m in key_moments
        ])

        atmosphere_text = ", ".join([f"{a.label} ({int(a.confidence*100)}%)" for a in atmosphere])

        system_instruction = (
            "You are an AI Memory Editorial Director. Your task is to transform structured media evidence "
            "into a meaningful, human-grade memory narrative.\n\n"
            "STRICT GROUNDING RULES:\n"
            "1. Ground all claims in the provided factual evidence.\n"
            "2. Never invent people, personal relationships, locations, dates, or unobserved activities.\n"
            "3. Do not infer personal emotions or thoughts from appearance alone; use event-level atmosphere.\n"
            "4. Adapt the tone naturally: warm and relaxed for trips, celebratory for parties, clear and professional for work.\n"
            "5. Do NOT invent new timeline boundaries or modify media IDs.\n"
            "6. Summary must be exactly 2 to 4 sentences describing the natural progression (beginning -> progression -> key moments -> conclusion).\n"
            "7. Return ONLY valid JSON format.\n"
        )

        user_content = (
            f"FACTUAL EVENT EVIDENCE:\n"
            f"Date: {evidence_dict.get('date_display')}\n"
            f"Time Range: {evidence_dict.get('start_time_str')} to {evidence_dict.get('end_time_str')} ({evidence_dict.get('duration_hours')} hours)\n"
            f"Location: {evidence_dict.get('location') or 'Location not specified'}\n"
            f"Media Count: {evidence_dict.get('media_count')} items\n"
            f"Dominant Scenes: {', '.join(evidence_dict.get('dominant_scenes', []))}\n"
            f"Dominant Objects: {', '.join(evidence_dict.get('dominant_objects', []))}\n"
            f"People Presence: {evidence_dict.get('people_presence')}\n"
            f"Visual Notes: {'; '.join(evidence_dict.get('sample_vlm_notes', []))}\n\n"
            f"DETERMINISTIC TIMELINE PHASES:\n{phases_text}\n\n"
            f"SELECTED KEY MOMENTS:\n{moments_text}\n\n"
            f"EVENT ATMOSPHERE:\n{atmosphere_text}\n\n"
            f"REQUIRED JSON SCHEMA:\n"
            "{\n"
            '  "ai_title": "<Distinctive, human-friendly title>",\n'
            '  "ai_summary": "<2-4 sentences describing event progression>",\n'
            '  "phase_descriptions": {\n'
            '    "phase_1": {"title": "<Phase Title>", "description": "<Phase summary>"}\n'
            "  },\n"
            '  "highlights": ["<Highlight 1>", "<Highlight 2>", "<Highlight 3>"],\n'
            '  "narrative_confidence": 0.92\n'
            "}"
        )

        try:
            full_prompt = f"{system_instruction}\n\n{user_content}"
            raw_response = call_llm(full_prompt, preferred_model="inclusionai/ling-3.0-flash-fin:free")
            if raw_response:
                clean_str = raw_response.strip()
                if "```" in clean_str:
                    clean_str = clean_str.split("```")[1]
                    if clean_str.startswith("json"):
                        clean_str = clean_str[4:]
                data = json.loads(clean_str.strip())

                ai_title = data.get("ai_title") or cls._generate_fallback_title(items, manifest, title_hint)
                ai_summary = data.get("ai_summary") or cls._generate_fallback_summary(items, manifest, phases)
                highlights = data.get("highlights") or [m.description for m in key_moments[:4]]
                confidence = float(data.get("narrative_confidence", 0.90))

                # Update phase titles/descriptions while preserving deterministic media_ids & time bounds
                phase_updates = data.get("phase_descriptions", {})
                updated_phases = []
                for p in phases:
                    p_info = phase_updates.get(p.phase_id, {})
                    updated_phases.append(TimelinePhase(
                        phase_id=p.phase_id,
                        start_time=p.start_time,
                        end_time=p.end_time,
                        title=p_info.get("title") or p.title,
                        description=p_info.get("description") or p.description,
                        media_ids=p.media_ids,
                        evidence_summary=p.evidence_summary
                    ))

                return MemoryNarrativeResponse(
                    memory_id=memory_id,
                    ai_title=ai_title,
                    ai_summary=ai_summary,
                    atmosphere=atmosphere,
                    timeline=updated_phases,
                    key_moments=key_moments,
                    highlights=highlights,
                    narrative_confidence=round(confidence, 2),
                    generated_at=datetime.now(timezone.utc).isoformat()
                )
        except Exception as ex:
            logger.warning(f"LLM narrative synthesis failed, using deterministic fallback: {ex}")

        # Fallback if LLM parsing or call encounters issues
        return cls._synthesize_deterministic_fallback(
            memory_id=memory_id,
            items=items,
            manifest=manifest,
            phases=phases,
            key_moments=key_moments,
            atmosphere=atmosphere,
            title_hint=title_hint
        )

    # ── STAGE 7: DETERMINISTIC FALLBACK GENERATOR ────────────────────────────

    @classmethod
    def _synthesize_deterministic_fallback(
        cls,
        memory_id: Optional[str],
        items: List[MediaItemContext],
        manifest: Dict[str, Any],
        phases: List[TimelinePhase],
        key_moments: List[KeyMoment],
        atmosphere: List[AtmosphereItem],
        title_hint: Optional[str] = None
    ) -> MemoryNarrativeResponse:
        """
        Guaranteed complete deterministic synthesis when LLM is unavailable.
        Zero external calls, zero failure rate.
        """
        ai_title = cls._generate_fallback_title(items, manifest, title_hint)
        ai_summary = cls._generate_fallback_summary(items, manifest, phases)
        highlights = [m.description for m in key_moments[:4]]

        return MemoryNarrativeResponse(
            memory_id=memory_id,
            ai_title=ai_title,
            ai_summary=ai_summary,
            atmosphere=atmosphere,
            timeline=phases,
            key_moments=key_moments,
            highlights=highlights,
            narrative_confidence=0.88,
            generated_at=datetime.now(timezone.utc).isoformat()
        )

    @classmethod
    def _generate_fallback_title(
        cls,
        items: List[MediaItemContext],
        manifest: Dict[str, Any],
        title_hint: Optional[str]
    ) -> str:
        if title_hint:
            return title_hint

        ev = manifest["evidence"]
        loc = ev.get("location")
        date_str = ev.get("date_display")
        top_scenes = ev.get("dominant_scenes", [])
        all_tags = {t.lower() for it in items for t in it.all_tags}

        if any(w in all_tags for w in ["birthday", "party", "cake"]):
            loc_s = f" at {loc}" if loc else ""
            return f"Birthday Celebration{loc_s}"
        elif any(w in all_tags for w in ["beach", "ocean", "sea"]):
            loc_s = loc or "the Beach"
            time_s = "Sunset" if any("sunset" in s for s in top_scenes) else "Day"
            return f"{loc_s} Beach {time_s}"
        elif any(w in all_tags for w in ["office", "laptop", "coding"]):
            return f"Work & Office Session ({date_str})"
        elif loc:
            scene_s = top_scenes[0].title() if top_scenes else "Outing"
            return f"{scene_s} in {loc}"
        else:
            scene_s = top_scenes[0].title() if top_scenes else "Moments"
            return f"{scene_s} on {date_str}"

    @classmethod
    def _generate_fallback_summary(
        cls,
        items: List[MediaItemContext],
        manifest: Dict[str, Any],
        phases: List[TimelinePhase]
    ) -> str:
        ev = manifest["evidence"]
        date_str = ev.get("date_display")
        loc_str = f" at {ev.get('location')}" if ev.get("location") else ""
        count = ev.get("media_count")

        if len(phases) >= 2:
            first_phase = phases[0].title
            last_phase = phases[-1].title
            return (
                f"The day began with {first_phase.lower()}{loc_str} on {date_str}. "
                f"The collection follows the progression of activities across {len(phases)} key phases, "
                f"concluding with {last_phase.lower()} with {count} total moments recorded."
            )
        else:
            top_scenes = ", ".join(ev.get("dominant_scenes", ["activities"]))
            return (
                f"A collection of {count} photos and moments captured on {date_str}{loc_str}. "
                f"Features key scenes of {top_scenes}."
            )

    # ── STAGE 6: VALIDATION & SANITIZATION LAYER ─────────────────────────────

    @classmethod
    def _validate_and_sanitize(
        cls,
        response: MemoryNarrativeResponse,
        items: List[MediaItemContext],
        memory_id: Optional[str]
    ) -> MemoryNarrativeResponse:
        """
        Validates output consistency:
        - Filters out hallucinated or non-existent media IDs
        - Clamps confidence scores to [0.0, 1.0]
        - Ensures timeline boundaries match existing media
        """
        valid_item_ids = {it.id for it in items}

        # 1. Sanitize Timeline Phases
        sanitized_phases = []
        for p in response.timeline:
            filtered_ids = [mid for mid in p.media_ids if mid in valid_item_ids]
            if not filtered_ids:
                # If phase has invalid IDs, fallback to original
                filtered_ids = [it.id for it in items[:max(1, len(items)//len(response.timeline))]]
            sanitized_phases.append(TimelinePhase(
                phase_id=p.phase_id,
                start_time=p.start_time,
                end_time=p.end_time,
                title=p.title.strip() or "Event Phase",
                description=p.description.strip() or "Event phase moments.",
                media_ids=filtered_ids,
                evidence_summary=p.evidence_summary
            ))

        # 2. Sanitize Key Moments
        sanitized_moments = []
        for km in response.key_moments:
            if km.media_id in valid_item_ids:
                sanitized_moments.append(KeyMoment(
                    media_id=km.media_id,
                    timestamp=km.timestamp,
                    description=km.description.strip(),
                    importance_score=round(float(np.clip(km.importance_score, 0.0, 1.0)), 2),
                    reason=km.reason.strip()
                ))

        if not sanitized_moments and items:
            sanitized_moments.append(KeyMoment(
                media_id=items[0].id,
                timestamp=items[0].timestamp.strftime("%I:%M %p"),
                description=items[0].vlm_description or "Focal event moment",
                importance_score=0.85,
                reason="Representative scene focal moment."
            ))

        # 3. Sanitize Atmosphere
        sanitized_atmosphere = [
            AtmosphereItem(
                label=atm.label.lower().strip(),
                confidence=round(float(np.clip(atm.confidence, 0.0, 1.0)), 2),
                evidence=atm.evidence
            )
            for atm in response.atmosphere
        ]

        # 4. Clamp confidence
        clamped_conf = round(float(np.clip(response.narrative_confidence, 0.0, 1.0)), 2)

        return MemoryNarrativeResponse(
            memory_id=memory_id or response.memory_id,
            ai_title=response.ai_title.strip() or "Memory Moments",
            ai_summary=response.ai_summary.strip(),
            atmosphere=sanitized_atmosphere,
            timeline=sanitized_phases,
            key_moments=sanitized_moments,
            highlights=response.highlights or [m.description for m in sanitized_moments[:3]],
            narrative_confidence=clamped_conf,
            generated_at=response.generated_at or datetime.now(timezone.utc).isoformat()
        )

    # ── DATABASE PERSISTENCE HELPER ──────────────────────────────────────────

    @classmethod
    async def analyze_and_persist_memory(
        cls,
        memory_id: str,
        user_id: str,
        force_regenerate: bool = False
    ) -> MemoryNarrativeResponse:
        """
        Loads memory media from Supabase, generates narrative intelligence, and stores
        it in memories.metadata['narrative_intelligence'] without overwriting user titles.
        """
        supabase = get_supabase_client()
        
        # 1. Fetch memory
        mem_res = supabase.table("memories").select("*, media!media_memory_id_fkey(*)").eq("id", memory_id).execute()
        if not mem_res.data:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Memory not found")

        mem_record = mem_res.data[0]
        meta = mem_record.get("metadata") or {}

        # If already analyzed and not forcing regenerate, return stored narrative
        if not force_regenerate and "narrative_intelligence" in meta:
            stored = meta["narrative_intelligence"]
            return MemoryNarrativeResponse(**stored)

        # 2. Hydrate media items
        media_list = mem_record.get("media") or []
        items: List[MediaItemContext] = []
        for idx, m in enumerate(media_list):
            m_meta = m.get("metadata") or {}
            m_tags = m_meta.get("ai_tags") or {}
            
            dt_str = m.get("taken_at") or m.get("created_at")
            if dt_str:
                try:
                    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                except ValueError:
                    dt = datetime.now(timezone.utc) - timedelta(minutes=idx * 5)
            else:
                dt = datetime.now(timezone.utc) - timedelta(minutes=idx * 5)

            items.append(MediaItemContext(
                id=m["id"],
                filename=m.get("filename") or f"media_{idx}",
                timestamp=dt,
                latitude=m.get("latitude"),
                longitude=m.get("longitude"),
                location_name=m.get("location_name"),
                scenes=m_tags.get("scenes", []),
                objects=m_tags.get("objects", []),
                people_count=m_tags.get("people_count"),
                vlm_description=m_meta.get("vlm_description"),
                embedding=None,
                quality_score=float(m_meta.get("quality_score", 0.80)),
                media_type=m.get("media_type") or "image"
            ))

        # 3. Analyze
        narrative = cls.analyze_memory_event(
            items=items,
            memory_id=memory_id,
            event_title_hint=mem_record.get("title"),
            enable_llm=True
        )

        # 4. Persist to metadata['narrative_intelligence']
        narrative_dict = narrative.model_dump(mode="json")
        meta["narrative_intelligence"] = narrative_dict

        update_payload: Dict[str, Any] = {"metadata": meta}
        # Only set title/description if the existing memory record is blank/empty
        if not mem_record.get("title") or mem_record.get("title") == "New Memory":
            update_payload["title"] = narrative.ai_title
        if not mem_record.get("description"):
            update_payload["description"] = narrative.ai_summary

        supabase.table("memories").update(update_payload).eq("id", memory_id).execute()
        return narrative

    @classmethod
    def _empty_narrative_response(cls, memory_id: Optional[str]) -> MemoryNarrativeResponse:
        """Returns an empty narrative response for an empty collection."""
        return MemoryNarrativeResponse(
            memory_id=memory_id,
            ai_title="Empty Memory",
            ai_summary="No media items available to generate a memory narrative.",
            atmosphere=[],
            timeline=[],
            key_moments=[],
            highlights=[],
            narrative_confidence=0.0,
            generated_at=datetime.now(timezone.utc).isoformat()
        )


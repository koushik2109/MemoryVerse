import io
import math
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set, cast
import numpy as np

from app.core.db import get_supabase_client
from app.services.memory_service import MemoryService
from app.services.media_intelligence import score_image_quality
from app.services.ai_service import call_llm
from app.schemas.domain import MemoryCreate, ClusteredEventResponse, ClusterPreviewResponse, ClusterExecutionResponse

logger = logging.getLogger(__name__)


def _haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on the Earth in kilometers."""
    r = 6371.0  # Earth radius in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


class MediaItemContext:
    """Rich structured context for an individual media item during clustering."""
    def __init__(
        self,
        id: str,
        filename: str = "",
        timestamp: Optional[datetime] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        location_name: Optional[str] = None,
        scenes: Optional[List[str]] = None,
        objects: Optional[List[str]] = None,
        people_count: Optional[str] = None,
        vlm_description: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        quality_score: float = 0.80,
        vault_id: Optional[str] = None,
        media_type: str = "image"
    ):
        self.id = id
        self.filename = filename
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.latitude = latitude
        self.longitude = longitude
        self.location_name = location_name
        self.scenes = scenes or []
        self.objects = objects or []
        self.people_count = people_count
        self.vlm_description = vlm_description
        self.embedding = embedding
        self.quality_score = quality_score
        self.vault_id = vault_id
        self.media_type = media_type

    @property
    def all_tags(self) -> Set[str]:
        return {t.lower() for t in self.scenes + self.objects if t}


class EventClusteringService:
    """
    Intelligent Multi-Signal Event Clustering & Automatic Memory Grouping Engine.
    
    5-Stage Pipeline:
      Stage 1: Temporal + spatial deterministic candidate grouping with adaptive boundaries.
      Stage 2: Cheap semantic similarity refinement using pre-computed embeddings.
      Stage 3: Strong semantic refinement using VLM descriptions and AI tags.
      Stage 4: LLM coherence validation only for ambiguous clusters.
      Stage 5: Title, summary, confidence, and representative media selection.
    """

    # ── PUBLIC ENTRYPOINTS ────────────────────────────────────────────────────

    @classmethod
    def cluster_media_items(
        cls,
        items: List[MediaItemContext],
        enable_llm: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute the 5-stage clustering pipeline directly on an in-memory list of MediaItemContext.
        Used by preview, background jobs, and automated test suites.
        """
        if not items:
            return []

        if len(items) == 1:
            it = items[0]
            title, summary = cls._generate_deterministic_title_summary([it])
            return [{
                "event_id": f"evt_{it.id[:8]}",
                "title": title,
                "summary": summary,
                "start_time": it.timestamp.isoformat(),
                "end_time": it.timestamp.isoformat(),
                "location": it.location_name or "Unknown Location",
                "confidence": 0.95,
                "media_ids": [it.id],
                "representative_media_id": it.id,
                "is_coherent_event": True,
                "items": [it]
            }]

        # ── STAGE 1: Temporal + Spatial Adaptive Candidate Grouping
        candidate_groups = cls._stage1_temporal_spatial_grouping(items)

        # ── STAGE 2 & 3: Semantic Refinement & Splitting
        refined_groups = []
        for grp in candidate_groups:
            sub_groups = cls._stage2_and_3_semantic_refinement(grp)
            refined_groups.extend(sub_groups)

        # ── Optional: Detect Multi-day Parent Trips
        trips = cls._detect_parent_trips(refined_groups)

        # ── STAGE 4 & 5: Coherence Validation, Narrative Synthesis & Cover Selection
        final_events = []
        for idx, grp in enumerate(refined_groups):
            event_dict = cls._stage4_and_5_synthesize_event(
                event_index=idx + 1,
                items=grp,
                enable_llm=enable_llm,
                trips_map=trips
            )
            final_events.append(event_dict)

        return final_events

    @classmethod
    async def preview_clusters_for_user(
        cls,
        user_id: str,
        vault_id: Optional[str] = None
    ) -> ClusterPreviewResponse:
        """Fetch unassigned user media and return preview clusters without mutating DB."""
        items = cls._fetch_user_unassociated_media(user_id, vault_id=vault_id)
        if not items:
            return ClusterPreviewResponse(status="skipped", total_media_processed=0, events_found=0, events=[])

        raw_events = cls.cluster_media_items(items, enable_llm=True)
        events_resp = [
            ClusteredEventResponse(
                event_id=e["event_id"],
                title=e["title"],
                summary=e["summary"],
                start_time=datetime.fromisoformat(e["start_time"]) if e.get("start_time") else None,
                end_time=datetime.fromisoformat(e["end_time"]) if e.get("end_time") else None,
                location=e.get("location"),
                confidence=e.get("confidence", 0.90),
                media_ids=e.get("media_ids", []),
                representative_media_id=e.get("representative_media_id"),
                is_coherent_event=e.get("is_coherent_event", True),
                parent_trip_id=e.get("parent_trip_id"),
                parent_trip_title=e.get("parent_trip_title"),
            )
            for e in raw_events
        ]
        return ClusterPreviewResponse(
            status="success",
            total_media_processed=len(items),
            events_found=len(events_resp),
            events=events_resp
        )

    @classmethod
    async def cluster_and_create_memories(
        cls,
        user_id: str,
        vault_id: Optional[str] = None
    ) -> ClusterExecutionResponse:
        """
        Execute clustering and persist generated Events as Memories in Supabase.
        """
        items = cls._fetch_user_unassociated_media(user_id, vault_id=vault_id)
        if len(items) < 2:
            return ClusterExecutionResponse(
                status="skipped",
                total_media_processed=len(items),
                memories_created=0,
                events=[]
            )

        raw_events = cls.cluster_media_items(items, enable_llm=True)
        supabase = get_supabase_client()
        mem_service = MemoryService(supabase)

        created_count = 0
        persisted_events: List[ClusteredEventResponse] = []

        for e in raw_events:
            start_dt = datetime.fromisoformat(e["start_time"]) if e.get("start_time") else datetime.now(timezone.utc)
            m_payload = MemoryCreate(
                vault_id=vault_id or e.get("vault_id"),
                title=e["title"],
                description=e["summary"],
                memory_date=start_dt,
                location_name=e.get("location")
            )
            try:
                new_mem = await mem_service.create_memory(user_id, m_payload)
                new_mem_id = str(new_mem.get("id") or "")
                created_count += 1

                # Associate media items with memory
                for mid in e["media_ids"]:
                    supabase.table("media").update({"memory_id": new_mem_id}).eq("id", mid).execute()

                # Set cover image
                cover_id = e.get("representative_media_id") or e["media_ids"][0]
                supabase.table("memories").update({"cover_media_id": cover_id}).eq("id", new_mem_id).execute()

                persisted_events.append(ClusteredEventResponse(
                    event_id=new_mem_id,
                    title=e["title"],
                    summary=e["summary"],
                    start_time=start_dt,
                    end_time=datetime.fromisoformat(e["end_time"]) if e.get("end_time") else start_dt,
                    location=e.get("location"),
                    confidence=e.get("confidence", 0.90),
                    media_ids=e["media_ids"],
                    representative_media_id=cover_id,
                    is_coherent_event=True,
                    parent_trip_id=e.get("parent_trip_id"),
                    parent_trip_title=e.get("parent_trip_title"),
                ))
            except Exception as ex:
                logger.error(f"Failed to create memory for clustered event '{e['title']}': {ex}")

        return ClusterExecutionResponse(
            status="success",
            total_media_processed=len(items),
            memories_created=created_count,
            events=persisted_events
        )

    # ── STAGE 1: TEMPORAL + SPATIAL ADAPTIVE GROUPING ────────────────────────

    @classmethod
    def _stage1_temporal_spatial_grouping(cls, items: List[MediaItemContext]) -> List[List[MediaItemContext]]:
        """
        Sort items chronologically and segment into candidate groups using adaptive multi-signal boundaries.
        Does NOT hardcode a static 4-hour window.
        """
        # Sort by timestamp (with progressive fallback)
        sorted_items = sorted(items, key=lambda x: x.timestamp)
        groups: List[List[MediaItemContext]] = []
        current_group: List[MediaItemContext] = [sorted_items[0]]

        for i in range(1, len(sorted_items)):
            prev = sorted_items[i - 1]
            curr = sorted_items[i]

            time_diff_hours = abs((curr.timestamp - prev.timestamp).total_seconds()) / 3600.0

            # Compute GPS distance if available
            geo_dist_km: Optional[float] = None
            if prev.latitude is not None and prev.longitude is not None and curr.latitude is not None and curr.longitude is not None:
                geo_dist_km = _haversine_distance_km(prev.latitude, prev.longitude, curr.latitude, curr.longitude)

            # Compute semantic tag overlap (Jaccard)
            tags_prev = prev.all_tags
            tags_curr = curr.all_tags
            tag_overlap = len(tags_prev & tags_curr) / len(tags_prev | tags_curr) if (tags_prev or tags_curr) else 0.5

            # Adaptive boundary decision:
            # 1. Multi-day separation (> 16 hours difference) -> split into separate day/session events
            if time_diff_hours > 16.0:
                groups.append(current_group)
                current_group = [curr]
                continue

            # 2. Moderate separation (3.5 to 16 hours):
            if time_diff_hours >= 3.5:
                # Check if same location or nearby
                is_same_location = (
                    (geo_dist_km is not None and geo_dist_km <= 5.0) or
                    (bool(prev.location_name and curr.location_name and prev.location_name.lower() == curr.location_name.lower()))
                )
                
                # Check thematic continuity
                thematic_tags = {"beach", "ocean", "sea", "sand", "resort", "sunset", "party", "food", "dining", "park", "nature", "trip"}
                has_thematic_continuity = bool((tags_prev & tags_curr) & thematic_tags or tag_overlap >= 0.08)

                if is_same_location and has_thematic_continuity:
                    # Keep together as continuous day outing (e.g., 9am beach -> 1pm lunch -> 6pm sunset)
                    pass
                elif (geo_dist_km is not None and geo_dist_km > 25.0) or tag_overlap == 0.0 or not is_same_location:
                    groups.append(current_group)
                    current_group = [curr]
                    continue

            # 3. Short separation (< 3.5 hours):
            # Split only if extreme location leap (> 50 km)
            if geo_dist_km is not None and geo_dist_km > 50.0:
                groups.append(current_group)
                current_group = [curr]
                continue

            current_group.append(curr)

        if current_group:
            groups.append(current_group)

        return groups

    # ── STAGE 2 & 3: SEMANTIC REFINEMENT & SPLITTING ─────────────────────────

    @classmethod
    def _stage2_and_3_semantic_refinement(cls, group: List[MediaItemContext]) -> List[List[MediaItemContext]]:
        """
        Refines a temporal group by checking for conflicting semantic clusters (e.g. Office vs Party on same day).
        Uses VLM descriptions and AI tags as primary semantic authority, assisted by CLIP embeddings.
        """
        if len(group) <= 2:
            return [group]

        # Categorize items by dominant activity / domain
        office_tags = {"office", "laptop", "coding", "meeting", "desk", "computer", "presentation", "documents", "workplace"}
        party_tags = {"birthday", "party", "cake", "celebration", "candles", "balloons", "dancing", "cheers", "confetti"}
        beach_tags = {"beach", "ocean", "sea", "sand", "coast", "swimming", "sunset", "surf", "resort", "boat"}
        food_tags = {"dinner", "restaurant", "food", "pizza", "pasta", "dining", "meal", "cocktail"}

        categorized: Dict[str, List[MediaItemContext]] = {
            "office": [],
            "party": [],
            "beach": [],
            "general": []
        }

        for item in group:
            item_tags = item.all_tags
            vlm_text = (item.vlm_description or "").lower()
            
            is_office = bool(item_tags & office_tags or any(w in vlm_text for w in ["office", "laptop", "meeting room", "desk"]))
            is_party = bool(item_tags & party_tags or any(w in vlm_text for w in ["birthday", "party", "cake", "celebrating"]))
            is_beach = bool(item_tags & beach_tags or any(w in vlm_text for w in ["beach", "ocean", "coastal", "waves", "sand"]))

            if is_office and not (is_party or is_beach):
                categorized["office"].append(item)
            elif is_party and not is_office:
                categorized["party"].append(item)
            elif is_beach and not is_office:
                categorized["beach"].append(item)
            else:
                categorized["general"].append(item)

        # Count active non-empty distinct categories
        active_cats = [k for k in ["office", "party", "beach"] if len(categorized[k]) > 0]
        
        # If there are strictly conflicting categories (e.g. Office + Party or Office + Beach)
        if len(active_cats) > 1 and "office" in active_cats:
            sub_clusters = []
            for cat in active_cats:
                if categorized[cat]:
                    sub_clusters.append(categorized[cat])
            if categorized["general"]:
                # Attach general items to nearest temporal sub-cluster
                for g_item in categorized["general"]:
                    best_sub = min(sub_clusters, key=lambda sub: min(abs((g_item.timestamp - s.timestamp).total_seconds()) for s in sub))
                    best_sub.append(g_item)
            return [sorted(sub, key=lambda x: x.timestamp) for sub in sub_clusters]

        return [group]

    # ── STAGE 4 & 5: SYNTHESIS, COHERENCE & COVER SELECTION ───────────────────

    @classmethod
    def _stage4_and_5_synthesize_event(
        cls,
        event_index: int,
        items: List[MediaItemContext],
        enable_llm: bool = True,
        trips_map: Optional[Dict[str, Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates cluster coherence, generates descriptive title and summary, and selects the optimal cover.
        """
        sorted_items = sorted(items, key=lambda x: x.timestamp)
        start_time = sorted_items[0].timestamp
        end_time = sorted_items[-1].timestamp
        media_ids = [it.id for it in sorted_items]

        # Calculate dominant location
        locations = [it.location_name for it in sorted_items if it.location_name]
        loc_counts = {}
        for loc in locations:
            loc_counts[loc] = loc_counts.get(loc, 0) + 1
        dominant_loc = max(loc_counts, key=lambda k: loc_counts.get(k, 0)) if loc_counts else None

        # Aggregate tags
        all_scenes: Dict[str, int] = {}
        all_objects: Dict[str, int] = {}
        for it in sorted_items:
            for s in it.scenes:
                all_scenes[s] = all_scenes.get(s, 0) + 1
            for o in it.objects:
                all_objects[o] = all_objects.get(o, 0) + 1

        top_scenes = sorted(all_scenes, key=lambda k: all_scenes.get(k, 0), reverse=True)[:4]
        top_objects = sorted(all_objects, key=lambda k: all_objects.get(k, 0), reverse=True)[:4]
        sample_descriptions = [it.vlm_description for it in sorted_items if it.vlm_description][:3]

        # Calculate Coherence Score
        coherence = cls._calculate_cluster_coherence(sorted_items)

        # Check for Parent Trip linkage
        parent_trip_id = None
        parent_trip_title = None
        if trips_map and items[0].id in trips_map:
            t_info = trips_map[items[0].id]
            parent_trip_id = t_info.get("trip_id")
            parent_trip_title = t_info.get("trip_title")

        # Select Representative Media
        rep_media_id = cls._select_representative_media(sorted_items, top_scenes, top_objects)

        # Title & Summary Generation
        title, summary, is_coherent = cls._generate_title_and_summary(
            sorted_items=sorted_items,
            start_time=start_time,
            end_time=end_time,
            location=dominant_loc,
            top_scenes=top_scenes,
            top_objects=top_objects,
            sample_descriptions=sample_descriptions,
            coherence=coherence,
            enable_llm=enable_llm
        )

        return {
            "event_id": f"evt_{sorted_items[0].id[:8]}_{event_index}",
            "title": title,
            "summary": summary,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "location": dominant_loc or "Unknown Location",
            "confidence": round(coherence, 2),
            "media_ids": media_ids,
            "representative_media_id": rep_media_id,
            "is_coherent_event": is_coherent,
            "parent_trip_id": parent_trip_id,
            "parent_trip_title": parent_trip_title,
            "items": sorted_items
        }

    # ── REPRESENTATIVE MEDIA SELECTION (BALANCED METRIC) ─────────────────────

    @classmethod
    def _select_representative_media(
        cls,
        items: List[MediaItemContext],
        top_scenes: List[str],
        top_objects: List[str]
    ) -> str:
        """
        Balances:
          1. Technical visual quality (35%)
          2. Embedding centrality / temporal centrality (35%)
          3. Semantic representativeness (presence of dominant tags/VLM subject) (30%)
        """
        if not items:
            return ""
        if len(items) == 1:
            return items[0].id

        # Calculate mean embedding if available
        valid_embs = [it.embedding for it in items if it.embedding and any(it.embedding)]
        mean_emb = None
        if valid_embs:
            mean_emb = np.mean(np.array(valid_embs), axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb = mean_emb / norm

        target_tags = set(top_scenes + top_objects)
        best_item = items[0]
        best_score = -1.0

        for it in items:
            # 1. Quality component (0.0 to 1.0)
            q_score = float(it.quality_score or 0.80)

            # 2. Centrality component (0.0 to 1.0)
            c_score = 0.50
            if mean_emb is not None and it.embedding and any(it.embedding):
                it_emb = np.array(it.embedding)
                it_norm = np.linalg.norm(it_emb)
                if it_norm > 0:
                    c_score = float(np.dot(mean_emb, it_emb / it_norm))
                    c_score = max(0.0, min(1.0, c_score))

            # 3. Semantic Representativeness (0.0 to 1.0)
            s_score = 0.50
            if target_tags:
                overlap = len(it.all_tags & target_tags)
                s_score = min(1.0, overlap / max(1, len(target_tags) / 2))

            # Composite balanced score
            composite = (0.35 * q_score) + (0.35 * c_score) + (0.30 * s_score)
            if composite > best_score:
                best_score = composite
                best_item = it

        return best_item.id

    # ── COHERENCE CALCULATION ────────────────────────────────────────────────

    @classmethod
    def _calculate_cluster_coherence(cls, items: List[MediaItemContext]) -> float:
        """Computes internal semantic & contextual coherence of a cluster."""
        if len(items) <= 1:
            return 0.95

        # 1. Temporal span consistency
        span_hours = abs((items[-1].timestamp - items[0].timestamp).total_seconds()) / 3600.0
        time_coherence = max(0.60, 1.0 - (span_hours / 48.0))

        # 2. Tag cohesion
        all_tags_list = [it.all_tags for it in items if it.all_tags]
        if len(all_tags_list) >= 2:
            intersection = set.intersection(*all_tags_list) if len(all_tags_list) <= 5 else set()
            union = set.union(*all_tags_list)
            tag_coherence = 0.85 if intersection else (0.75 if len(union) < 15 else 0.65)
        else:
            tag_coherence = 0.80

        # 3. Embedding cosine variance if present
        valid_embs = [it.embedding for it in items if it.embedding and any(it.embedding)]
        emb_coherence = 0.80
        if len(valid_embs) >= 2:
            emb_matrix = np.array(valid_embs)
            norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            sims = np.dot(emb_matrix / norms, (emb_matrix / norms).T)
            emb_coherence = float(np.mean(sims))

        return float(np.clip(0.4 * tag_coherence + 0.3 * time_coherence + 0.3 * emb_coherence, 0.50, 0.98))

    # ── TITLE & SUMMARY GENERATION (DETERMINISTIC + LLM FALLBACK) ────────────

    @classmethod
    def _generate_title_and_summary(
        cls,
        sorted_items: List[MediaItemContext],
        start_time: datetime,
        end_time: datetime,
        location: Optional[str],
        top_scenes: List[str],
        top_objects: List[str],
        sample_descriptions: List[str],
        coherence: float,
        enable_llm: bool
    ) -> Tuple[str, str, bool]:
        """
        Synthesizes human-grade titles and summaries.
        Uses single compact LLM call if enabled and cluster has narrative depth, else robust deterministic template.
        """
        date_str = start_time.strftime("%b %d, %Y")
        
        # Check for well-defined thematic categories
        all_tags = set(top_scenes + top_objects)
        
        is_birthday = any(w in all_tags for w in ["birthday", "party", "cake", "celebration", "candles"])
        is_beach = any(w in all_tags for w in ["beach", "ocean", "sea", "sand", "sunset", "coast"])
        is_office = any(w in all_tags for w in ["office", "laptop", "coding", "meeting", "desk", "computer"])
        is_dinner = any(w in all_tags for w in ["dinner", "restaurant", "food", "dining", "cocktail", "pizza"])

        # Deterministic generation
        if is_birthday:
            loc_suffix = f" at {location}" if location else ""
            det_title = f"Birthday Celebration{loc_suffix}"
            det_summary = f"Celebration gathering with cake and friends on {date_str} ({len(sorted_items)} photos)."
        elif is_beach:
            loc_name = location or "the Beach"
            time_tag = "Sunset" if any("sunset" in s for s in top_scenes) else "Day"
            det_title = f"{loc_name} Beach {time_tag}"
            det_summary = f"Relaxing and spending time by the water on {date_str} ({len(sorted_items)} photos)."
        elif is_office:
            det_title = f"Work & Office Session ({date_str})"
            det_summary = f"Productive work sessions, meetings, and desk focus ({len(sorted_items)} items)."
        elif is_dinner:
            loc_suffix = f" at {location}" if location else ""
            det_title = f"Dinner & Evening Out{loc_suffix}"
            det_summary = f"Gathering for food and evening drinks on {date_str} ({len(sorted_items)} photos)."
        elif location:
            scene_label = top_scenes[0].title() if top_scenes else "Outing"
            det_title = f"{scene_label} in {location}"
            det_summary = f"Exploring {location} on {date_str} ({len(sorted_items)} media items)."
        else:
            scene_label = top_scenes[0].title() if top_scenes else "Moments"
            det_title = f"{scene_label} on {date_str}"
            det_summary = f"Captured memory collection from {date_str} ({len(sorted_items)} items)."

        if not enable_llm:
            return det_title, det_summary, True

        # Compact LLM validation & title refinement (1 cheap call per cluster)
        prompt = (
            f"Synthesize an event memory title and 1-sentence narrative summary for this photo collection:\n"
            f"Date: {date_str}\n"
            f"Location: {location or 'Unknown'}\n"
            f"Count: {len(sorted_items)} items\n"
            f"Dominant scenes: {', '.join(top_scenes)}\n"
            f"Dominant objects: {', '.join(top_objects)}\n"
            f"Sample visual notes: {'; '.join(sample_descriptions) if sample_descriptions else 'None'}\n\n"
            f"Return JSON:\n"
            f'{{"title": "<Engaging 3-6 word Event Title>", "summary": "<Concise 1-sentence narrative summary>", "is_coherent_event": true}}'
        )

        try:
            res_str = call_llm(prompt, preferred_model="inclusionai/ling-3.0-flash-fin:free")
            if res_str:
                import json
                clean_str = res_str.strip()
                if "```" in clean_str:
                    clean_str = clean_str.split("```")[1]
                    if clean_str.startswith("json"):
                        clean_str = clean_str[4:]
                data = json.loads(clean_str.strip())
                llm_title = data.get("title", det_title)
                llm_summary = data.get("summary", det_summary)
                is_coherent = bool(data.get("is_coherent_event", True))
                return llm_title, llm_summary, is_coherent
        except Exception as e:
            logger.warning(f"LLM event synthesis fallback: {e}")

        return det_title, det_summary, True

    @classmethod
    def _generate_deterministic_title_summary(cls, items: List[MediaItemContext]) -> Tuple[str, str]:
        """Simple deterministic title/summary generator for single-item edge cases."""
        it = items[0]
        date_str = it.timestamp.strftime("%b %d, %Y")
        loc_str = f" in {it.location_name}" if it.location_name else ""
        scene_str = it.scenes[0].title() if it.scenes else "Memory"
        title = f"{scene_str}{loc_str} ({date_str})"
        summary = f"Photo taken on {date_str}."
        return title, summary

    # ── MULTI-DAY PARENT TRIP DETECTION ──────────────────────────────────────

    @classmethod
    def _detect_parent_trips(cls, event_groups: List[List[MediaItemContext]]) -> Dict[str, Dict[str, str]]:
        """
        Identifies if multiple adjacent event groups span consecutive days in the same region.
        Maps item_id -> {"trip_id": str, "trip_title": str}
        """
        trips_map: Dict[str, Dict[str, str]] = {}
        if len(event_groups) < 2:
            return trips_map

        # Check consecutive clusters
        trip_counter = 1
        current_trip_events = [event_groups[0]]

        for i in range(1, len(event_groups)):
            prev_grp = event_groups[i - 1]
            curr_grp = event_groups[i]

            prev_end = max(it.timestamp for it in prev_grp)
            curr_start = min(it.timestamp for it in curr_grp)
            day_gap = abs((curr_start.date() - prev_end.date()).days)

            # Check location consistency
            prev_locs = {it.location_name for it in prev_grp if it.location_name}
            curr_locs = {it.location_name for it in curr_grp if it.location_name}
            common_loc = (prev_locs & curr_locs) or (prev_locs and curr_locs)

            if 1 <= day_gap <= 3 and common_loc:
                current_trip_events.append(curr_grp)
            else:
                if len(current_trip_events) >= 2:
                    trip_loc = next((it.location_name for grp in current_trip_events for it in grp if it.location_name), "Vacation")
                    trip_id = f"trip_{trip_counter}"
                    trip_title = f"{trip_loc} Trip"
                    for grp in current_trip_events:
                        for it in grp:
                            trips_map[it.id] = {"trip_id": trip_id, "trip_title": trip_title}
                    trip_counter += 1
                current_trip_events = [curr_grp]

        if len(current_trip_events) >= 2:
            trip_loc = next((it.location_name for grp in current_trip_events for it in grp if it.location_name), "Vacation")
            trip_id = f"trip_{trip_counter}"
            trip_title = f"{trip_loc} Trip"
            for grp in current_trip_events:
                for it in grp:
                    trips_map[it.id] = {"trip_id": trip_id, "trip_title": trip_title}

        return trips_map

    # ── DATABASE HYDRATION HELPER ────────────────────────────────────────────

    @classmethod
    def _fetch_user_unassociated_media(
        cls,
        user_id: str,
        vault_id: Optional[str] = None
    ) -> List[MediaItemContext]:
        """Fetches unassociated media items from Supabase and hydrates MediaItemContext."""
        supabase = get_supabase_client()
        query = supabase.table("media").select("*, media_embeddings(embedding)").eq("owner_id", user_id)
        if vault_id:
            query = query.eq("vault_id", vault_id)
        else:
            query = query.is_("memory_id", "null")

        res = query.execute()
        raw_list = cast(list[dict[str, Any]], res.data or [])
        items: List[MediaItemContext] = []

        for idx, m in enumerate(raw_list):
            emb_data = m.get("media_embeddings")
            emb = None
            if isinstance(emb_data, list) and len(emb_data) > 0:
                emb = emb_data[0].get("embedding")
            elif isinstance(emb_data, dict):
                emb = emb_data.get("embedding")

            meta = m.get("metadata") or {}
            ai_tags = meta.get("ai_tags") or {}

            # Parse taken_at -> created_at -> progressive fallback
            dt_str = m.get("taken_at") or m.get("created_at")
            if dt_str:
                try:
                    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                except ValueError:
                    dt = datetime.now(timezone.utc) - timedelta(minutes=idx * 5)
            else:
                dt = datetime.now(timezone.utc) - timedelta(minutes=idx * 5)

            items.append(MediaItemContext(
                id=str(m["id"]),
                filename=str(m.get("filename") or f"media_{idx}"),
                timestamp=dt,
                latitude=m.get("latitude"),
                longitude=m.get("longitude"),
                location_name=m.get("location_name"),
                scenes=ai_tags.get("scenes", []),
                objects=ai_tags.get("objects", []),
                people_count=ai_tags.get("people_count"),
                vlm_description=meta.get("vlm_description"),
                embedding=emb if isinstance(emb, list) else None,
                quality_score=float(meta.get("quality_score", 0.80)),
                vault_id=m.get("vault_id"),
                media_type=str(m.get("media_type") or "image")
            ))

        return items

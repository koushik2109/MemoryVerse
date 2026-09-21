import logging
import numpy as np
from typing import cast, Any, Dict, List, Optional
from app.core.db import get_supabase_client
from app.schemas.domain import GlobalSearchResponse, VaultResponse, MediaResponse, UserProfile
from app.services.vault_service import VaultService

logger = logging.getLogger(__name__)

# Query semantic keyword mappings for zero-shot tag matching
QUERY_EXPANSIONS: Dict[str, List[str]] = {
    "birthday": ["birthday", "party", "cake", "celebration", "gift", "people celebrating", "friends"],
    "beach": ["beach", "sunset", "nature", "outdoor", "ocean", "sea", "vacation"],
    "celebration": ["celebration", "party", "birthday", "wedding", "people celebrating", "group of people"],
    "party": ["party", "celebration", "birthday", "food", "beverage", "people celebrating", "friends"],
    "trip": ["trip", "vacation", "mountains", "beach", "nature", "cityscape", "outdoor", "car"],
    "vacation": ["vacation", "trip", "beach", "mountains", "nature", "cityscape", "outdoor"],
    "sunset": ["sunset", "outdoor", "nature", "sky", "beach", "mountains"],
    "family": ["family", "group of people", "people celebrating", "one person", "outdoor", "indoor"],
    "friends": ["friends", "group of people", "party", "celebration", "people celebrating"],
    "nature": ["nature", "mountains", "beach", "outdoor", "plant"],
    "mountains": ["mountains", "nature", "outdoor", "trip", "vacation"],
    "food": ["food", "beverage", "party", "indoor"],
}

class SearchService:
    @staticmethod
    def global_search(query: str, user_id: str) -> GlobalSearchResponse:
        if not query or len(query.strip()) == 0:
            return GlobalSearchResponse(vaults=[], media=[], collaborators=[])

        query_clean = query.strip()
        query_lower = query_clean.lower()
        q_wildcard = f"%{query_clean}%"
        supabase = get_supabase_client()

        # 1. Vaults matching query
        user_vaults = VaultService.get_user_vaults(user_id)
        matching_vaults = [
            v for v in user_vaults
            if query_lower in v.name.lower() or (v.description and query_lower in v.description.lower())
        ]

        # 2. Media Search: Multimodal combining Filename, Location, Metadata Tags, and CLIP Embeddings
        media_by_id: Dict[str, Dict[str, Any]] = {}
        relevance_scores: Dict[str, float] = {}

        # 2a. Direct SQL query by filename or location_name
        try:
            m_res = supabase.table("media").select("*").eq("owner_id", user_id).or_(
                f"filename.ilike.{q_wildcard},location_name.ilike.{q_wildcard}"
            ).limit(40).execute()
            for m in (cast(list[dict[str, Any]], m_res.data or [])):
                mid = str(m["id"])
                media_by_id[mid] = m
                relevance_scores[mid] = max(relevance_scores.get(mid, 0.0), 0.85)
        except Exception as e:
            logger.warning(f"Error searching media by filename/location: {e}")

        # 2b. Search all user media and match against AI tags & EXIF in metadata
        try:
            all_media_res = supabase.table("media").select("*").eq("owner_id", user_id).limit(100).execute()
            all_user_media = cast(list[dict[str, Any]], all_media_res.data or [])

            # Determine expanded keywords for semantic tag matching
            target_keywords = set()
            words = [w.strip() for w in query_lower.split() if len(w.strip()) > 2]
            for w in words:
                target_keywords.add(w)
                for key, expansions in QUERY_EXPANSIONS.items():
                    if key in w or w in key:
                        target_keywords.update(expansions)

            for m in all_user_media:
                mid = str(m["id"])
                meta = m.get("metadata") or {}
                ai_tags = meta.get("ai_tags") or {}
                scenes = [s.lower() for s in ai_tags.get("scenes", [])]
                objects = [o.lower() for o in ai_tags.get("objects", [])]
                people = str(ai_tags.get("people_count", "")).lower()
                all_tags = set(scenes + objects + [people])

                # Calculate tag match overlap
                matched_tags = target_keywords.intersection(all_tags)
                if matched_tags:
                    score = min(0.95, 0.50 + 0.15 * len(matched_tags))
                    media_by_id[mid] = m
                    relevance_scores[mid] = max(relevance_scores.get(mid, 0.0), score)
        except Exception as e:
            logger.warning(f"Error searching media by AI tags: {e}")

        # 2c. Semantic Vector Search via CLIP (if model available)
        try:
            from app.services.ai_extractor import get_clip_model
            model = get_clip_model()
            if model is not None:
                # Compute query text embedding
                query_text = f"a photo of {query_clean}"
                query_emb = model.encode([query_text])[0]  # shape (512,)
                query_norm = np.linalg.norm(query_emb)
                if query_norm > 0:
                    query_unit = query_emb / query_norm

                    # Fetch user media embeddings
                    emb_res = supabase.table("media_embeddings").select("media_id, clip_embedding, embedding").execute()
                    emb_records = cast(list[dict[str, Any]], emb_res.data or [])

                    for rec in emb_records:
                        mid = str(rec.get("media_id"))
                        # Prefer clip_embedding, fallback to generic embedding
                        emb_vec = rec.get("clip_embedding") or rec.get("embedding")
                        if emb_vec and len(emb_vec) == len(query_unit):
                            vec = np.array(emb_vec, dtype=np.float32)
                            v_norm = np.linalg.norm(vec)
                            if v_norm > 0:
                                sim = float(np.dot(query_unit, vec / v_norm))
                                # Threshold for CLIP semantic similarity
                                if sim >= 0.18:
                                    if mid not in media_by_id:
                                        # Fetch full media row if not already loaded
                                        single_m = supabase.table("media").select("*").eq("id", mid).eq("owner_id", user_id).execute()
                                        if single_m.data:
                                            media_by_id[mid] = single_m.data[0]
                                    if mid in media_by_id:
                                        relevance_scores[mid] = max(relevance_scores.get(mid, 0.0), sim)
        except Exception as e:
            logger.debug(f"Semantic vector search skipped or failed: {e}")

        # Sort media by relevance score descending
        sorted_media_ids = sorted(media_by_id.keys(), key=lambda mid: relevance_scores.get(mid, 0.0), reverse=True)

        matching_media = [
            MediaResponse(
                id=media_by_id[mid]["id"],
                vault_id=media_by_id[mid].get("vault_id"),
                memory_id=media_by_id[mid].get("memory_id"),
                owner_id=media_by_id[mid]["owner_id"],
                filename=media_by_id[mid]["filename"],
                storage_path=media_by_id[mid]["storage_path"],
                url=media_by_id[mid]["url"],
                thumbnail_url=media_by_id[mid].get("thumbnail_url"),
                media_type=media_by_id[mid]["media_type"],
                file_size=media_by_id[mid]["file_size"],
                mime_type=media_by_id[mid].get("mime_type"),
                width=media_by_id[mid].get("width"),
                height=media_by_id[mid].get("height"),
                duration=media_by_id[mid].get("duration"),
                taken_at=media_by_id[mid].get("taken_at"),
                latitude=media_by_id[mid].get("latitude"),
                longitude=media_by_id[mid].get("longitude"),
                location_name=media_by_id[mid].get("location_name"),
                metadata=media_by_id[mid].get("metadata"),
                created_at=media_by_id[mid]["created_at"]
            )
            for mid in sorted_media_ids[:30]
        ]

        # 3. Collaborators matching name or email
        collaborators: List[UserProfile] = []
        try:
            p_res = supabase.table("profiles").select("*").or_(f"full_name.ilike.{q_wildcard},email.ilike.{q_wildcard}").limit(10).execute()
            collaborators = [
                UserProfile(
                    id=p["id"],
                    email=p["email"],
                    full_name=p.get("full_name"),
                    username=p.get("username"),
                    avatar_url=p.get("avatar_url"),
                    bio=p.get("bio")
                ) for p in (cast(list[dict[str, Any]], p_res.data or [])) if p["id"] != user_id
            ]
        except Exception as e:
            logger.warning(f"Error searching profiles: {e}")

        return GlobalSearchResponse(
            vaults=matching_vaults,
            media=matching_media,
            collaborators=collaborators
        )

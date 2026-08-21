import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from app.core.db import get_supabase_client
from app.services.memory_service import MemoryService
from app.schemas.domain import MemoryCreate

logger = logging.getLogger(__name__)

class EventClusteringService:
    """Service to automatically cluster media into cohesive 'Memories' (Events)."""

    @staticmethod
    async def cluster_user_media(user_id: str, vault_id: str | None = None) -> Dict[str, Any]:
        """
        Retrieves all media for the user (and optionally vault) that are not currently
        associated with a memory, clusters them, creates new memories, and associates them.
        """
        supabase = get_supabase_client()
        
        # 1. Fetch all media without a memory_id
        query = supabase.table("media").select("*, media_embeddings(embedding)").eq("owner_id", user_id)
        if vault_id:
            query = query.eq("vault_id", vault_id)
        else:
            query = query.is_("memory_id", "null")

        res = query.execute()
        media_list = res.data or []

        if len(media_list) < 2:
            return {"status": "skipped", "message": "Not enough media items to cluster (minimum 2 required)."}

        # 2. Extract features, timestamps, and locations
        valid_items: List[Dict[str, Any]] = []
        embeddings: List[List[float]] = []

        for m in media_list:
            # Parse embedding
            emb_data = m.get("media_embeddings")
            emb = None
            if isinstance(emb_data, list) and len(emb_data) > 0:
                emb = emb_data[0].get("embedding")
            elif isinstance(emb_data, dict):
                emb = emb_data.get("embedding")

            if not emb or not any(emb):
                continue

            # Parse taken_at / created_at timestamp
            dt_str = m.get("taken_at") or m.get("created_at")
            if not dt_str:
                continue

            try:
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            except ValueError:
                continue

            valid_items.append({
                "id": m["id"],
                "vault_id": m.get("vault_id"),
                "taken_at": dt,
                "latitude": m.get("latitude"),
                "longitude": m.get("longitude"),
                "metadata": m.get("metadata") or {},
                "location_name": m.get("location_name")
            })
            embeddings.append(emb)

        if len(valid_items) < 2:
            return {"status": "skipped", "message": "Not enough items with valid embeddings and timestamps to cluster."}

        # 3. Build Distance Matrix
        n = len(valid_items)
        dist_matrix = np.zeros((n, n))

        emb_matrix = np.array(embeddings)
        # Normalize embeddings to calculate cosine distance
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid division by zero
        normed_embs = emb_matrix / norms
        cosine_sim = np.dot(normed_embs, normed_embs.T)
        semantic_dists = 1.0 - np.clip(cosine_sim, 0.0, 1.0)

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    dist_matrix[i][j] = 0.0
                    continue

                # A. Temporal distance (normalized: 24h difference is 1.0, scaled capped at 1.0)
                time_diff = abs((valid_items[i]["taken_at"] - valid_items[j]["taken_at"]).total_seconds())
                time_dist = min(time_diff / (24.0 * 3600.0), 1.0)

                # B. Spatial distance (GPS distance if coordinates are present)
                geo_dist = 0.5
                lat1, lon1 = valid_items[i]["latitude"], valid_items[i]["longitude"]
                lat2, lon2 = valid_items[j]["latitude"], valid_items[j]["longitude"]
                if lat1 is not None and lon1 is not None and lat2 is not None and lon2 is not None:
                    # Simple Euclidean degree distance as approximation for clustering
                    deg_diff = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
                    geo_dist = min(deg_diff / 0.1, 1.0)  # ~11km scale

                # C. Semantic distance from CLIP
                sem_dist = semantic_dists[i][j]

                # Combined weighted distance
                # Weights: 40% time, 30% location, 30% semantic CLIP similarity
                w_dist = (0.4 * time_dist) + (0.3 * geo_dist) + (0.3 * sem_dist)
                dist_matrix[i][j] = w_dist
                dist_matrix[j][i] = w_dist

        # 4. Perform DBSCAN Clustering
        # eps=0.25 (weighted distance threshold for forming groups)
        db = DBSCAN(eps=0.25, min_samples=2, metric="precomputed")
        labels = db.fit_predict(dist_matrix)  # type: ignore[arg-type]

        # 5. Process Clusters
        clusters = {}
        for idx, label in enumerate(labels):
            if label == -1:
                # Outliers / noise (can keep as unclustered or place in separate individual memories later)
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(valid_items[idx])

        created_memories_count = 0
        mem_service = MemoryService(supabase)
        for label, items in clusters.items():
            # A. Determine auto-title and description
            title, desc, avg_date, cluster_vault_id = EventClusteringService._analyze_cluster(items)
            
            # Use vault_id from items or the common one
            v_id = vault_id or cluster_vault_id

            # B. Create Memory record
            payload = MemoryCreate(
                vault_id=v_id,
                title=title,
                description=desc,
                memory_date=avg_date.date(),
                location_name=items[0].get("location_name")
            )
            try:
                new_mem = await mem_service.create_memory(user_id, payload)
                created_memories_count += 1
                new_mem_id = new_mem.get("id")

                # C. Link all clustered media to the new memory
                media_ids = [it["id"] for it in items]
                for mid in media_ids:
                    supabase.table("media").update({"memory_id": new_mem_id}).eq("id", mid).execute()

                # D. Update memory's cover_media_id to the first media item
                supabase.table("memories").update({"cover_media_id": media_ids[0]}).eq("id", new_mem_id).execute()
                logger.info(f"Automatically generated event memory '{title}' with {len(media_ids)} media items.")
            except Exception as e:
                logger.error(f"Failed to save auto-clustered memory: {e}")

        return {
            "status": "success",
            "total_items_processed": n,
            "clusters_found": len(clusters),
            "memories_created": created_memories_count
        }

    @staticmethod
    def _analyze_cluster(items: List[Dict[str, Any]]) -> Tuple[str, str, datetime, str | None]:
        """Analyze clustered media items to generate optimal event title, description, and date."""
        # Calculate average date
        timestamps = [it["taken_at"] for it in items]
        avg_timestamp = datetime.fromtimestamp(sum(ts.timestamp() for ts in timestamps) / len(timestamps), tz=timezone.utc)

        # Collect vault_id
        vault_ids = [it.get("vault_id") for it in items if it.get("vault_id")]
        common_vault_id = vault_ids[0] if vault_ids else None

        # Analyze tags/metadata to name the event
        all_scenes = []
        all_objects = []
        for it in items:
            ai_tags = it.get("metadata", {}).get("ai_tags", {})
            all_scenes.extend(ai_tags.get("scenes", []))
            all_objects.extend(ai_tags.get("objects", []))

        # Count frequencies
        scene_freq = {}
        for s in all_scenes:
            scene_freq[s] = scene_freq.get(s, 0) + 1
        
        obj_freq = {}
        for o in all_objects:
            obj_freq[o] = obj_freq.get(o, 0) + 1

        # pyrefly: ignore [no-matching-overload]
        best_scene = max(scene_freq, key=scene_freq.get) if scene_freq else None
        # pyrefly: ignore [no-matching-overload]
        best_obj = max(obj_freq, key=obj_freq.get) if obj_freq else None

        # Build dynamic title
        date_str = avg_timestamp.strftime("%b %d, %Y")
        
        # Check location names
        locations = [it.get("location_name") for it in items if it.get("location_name")]
        common_loc = locations[0] if locations else None

        if common_loc:
            title = f"Gathering in {common_loc.title()}"
            if best_scene:
                title = f"{best_scene.title()} at {common_loc.title()}"
        elif best_scene and best_obj:
            title = f"{best_scene.title()} with {best_obj.title()} ({date_str})"
        elif best_scene:
            title = f"{best_scene.title()} Outing ({date_str})"
        elif best_obj:
            title = f"{best_obj.title()} Outing ({date_str})"
        else:
            title = f"Memory on {date_str}"

        # Build description
        desc_tags = []
        if best_scene: desc_tags.append(f"scenes of {best_scene}")
        if best_obj: desc_tags.append(f"features of {best_obj}")
        
        desc = (
            f"Automatically grouped event from {date_str} containing {len(items)} media items. "
            f"Features: {', '.join(desc_tags) if desc_tags else 'general activities'}."
        )

        return title, desc, avg_timestamp, common_vault_id

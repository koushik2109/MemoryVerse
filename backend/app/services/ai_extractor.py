import io
import logging
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
try:
    import exifread
except ImportError:
    exifread = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from app.core.db import get_supabase_client

logger = logging.getLogger(__name__)

# Lazy-loaded CLIP Model using singleton loader
_clip_model: Optional[Any] = None
_cached_label_embeddings: Optional[Any] = None

def get_clip_model():
    global _clip_model
    try:
        from ai_engine.models.clip_loader import get_clip_model as _get_singleton_clip
        model = _get_singleton_clip()
        if model is not None:
            _clip_model = model
            return _clip_model
    except Exception as e:
        logger.debug(f"ai_engine.models.clip_loader fallback: {e}")

    if SentenceTransformer is None:
        logger.warning("sentence-transformers is not available. CLIP embeddings will be skipped.")
        return None
    if _clip_model is None:
        logger.info("Loading lightweight CLIP model (clip-ViT-B-32) for multimodal embeddings...")
        _clip_model = SentenceTransformer('clip-ViT-B-32')
    return _clip_model


class AIExtractor:
    """Extracts metadata, tags, and embeddings from media files using lightweight local libraries."""

    @staticmethod
    def extract_metadata(file_bytes: bytes, mime_type: str) -> Dict[str, Any]:
        """Extract camera maker/model, taken timestamp, and GPS coordinates from EXIF metadata."""
        metadata = {
            "taken_at": None,
            "latitude": None,
            "longitude": None,
            "camera_maker": None,
            "camera_model": None,
            "device": None
        }

        # EXIF only applies to images
        if not mime_type.startswith("image/") or exifread is None:
            return metadata

        try:
            # Parse EXIF tags using exifread
            tags = exifread.process_file(io.BytesIO(file_bytes), details=False)

            # 1. Taken timestamp
            taken_tag = (
                tags.get("EXIF DateTimeOriginal") or 
                tags.get("EXIF DateTimeDigitized") or 
                tags.get("Image DateTime")
            )
            if taken_tag:
                try:
                    # e.g., "2026:08:15 14:30:00"
                    dt_str = str(taken_tag).strip()
                    dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                    metadata["taken_at"] = dt.replace(tzinfo=timezone.utc).isoformat()
                except ValueError:
                    pass

            # 2. Camera Maker / Model
            maker = tags.get("Image Make")
            model = tags.get("Image Model")
            if maker:
                metadata["camera_maker"] = str(maker).strip()
            if model:
                metadata["camera_model"] = str(model).strip()
            if maker or model:
                metadata["device"] = f"{metadata['camera_maker'] or ''} {metadata['camera_model'] or ''}".strip()

            # 3. GPS Coordinates
            lat, lon = AIExtractor._parse_gps(tags)
            if lat is not None and lon is not None:
                metadata["latitude"] = lat
                metadata["longitude"] = lon

        except Exception as e:
            logger.warning(f"Failed to extract EXIF metadata: {e}")

        return metadata

    @staticmethod
    def extract_features(file_bytes: bytes, mime_type: str) -> Tuple[List[float], Dict[str, Any]]:
        """
        Generate 512-dimension CLIP embedding and detect tags/objects/scenes
        via zero-shot classification.
        """
        embedding: List[float] = []
        tags: Dict[str, Any] = {
            "objects": [],
            "scenes": [],
            "people_count": "unknown",
            "confidence_scores": {}
        }

        if not mime_type.startswith("image/"):
            # Return empty embedding/tags for video (or placeholder)
            return [0.0] * 512, tags

        try:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            model = get_clip_model()

            # 1. Embedding generation
            img_emb = model.encode(img)
            embedding = img_emb.tolist()

            # 2. Zero-shot tag classification using CLIP similarity
            scenes_labels = ["beach", "cityscape", "nature", "mountains", "indoor", "outdoor", "sunset", "party", "office"]
            objects_labels = ["food", "beverage", "car", "dog", "cat", "laptop", "building", "document", "plant"]
            people_labels = ["no people", "one person", "group of people"]
            all_labels = scenes_labels + objects_labels + people_labels

            global _cached_label_embeddings
            if "_cached_label_embeddings" not in globals() or _cached_label_embeddings is None:
                _cached_label_embeddings = model.encode([f"a photo of {x}" for x in all_labels])

            text_embeddings = _cached_label_embeddings

            # Compute similarities
            import numpy as np
            similarities = np.dot(img_emb, text_embeddings.T)
            # Softmax to get confidence scores
            scores = np.exp(similarities) / np.sum(np.exp(similarities))

            scores_dict = {all_labels[idx]: float(scores[idx]) for idx in range(len(all_labels))}
            tags["confidence_scores"] = scores_dict

            # Extract tags with score > 0.08
            for s in scenes_labels:
                if scores_dict[s] > 0.08:
                    tags["scenes"].append(s)
            for o in objects_labels:
                if scores_dict[o] > 0.08:
                    tags["objects"].append(o)

            # Determine people status
            people_scores = {p: scores_dict[p] for p in people_labels}
            # pyrefly: ignore [no-matching-overload]
            best_people = max(people_scores, key=people_scores.get)
            tags["people_count"] = best_people

        except Exception as e:
            logger.warning(f"Failed to generate embeddings/tags: {e}")
            embedding = [0.0] * 512

        return embedding, tags

    @staticmethod
    def process_media_item(media_id: str, file_bytes: bytes, mime_type: str):
        """Run metadata and feature extraction, saving results back to database."""
        logger.info(f"Processing uploaded media {media_id} for metadata and embeddings...")
        
        # 1. Extract EXIF tags
        meta = AIExtractor.extract_metadata(file_bytes, mime_type)

        # 2. Generate CLIP embedding & label tags
        embedding, tags = AIExtractor.extract_features(file_bytes, mime_type)

        # 3. Save to database
        supabase = get_supabase_client()
        try:
            # Check for near-duplicates in the same vault / memory (threshold >= 0.92)
            duplicate_info: Dict[str, Any] = {"is_alternate": False, "label": "Original", "alternate_media_ids": []}
            try:
                if any(embedding):
                    curr_media_row = supabase.table("media").select("vault_id, memory_id, owner_id").eq("id", media_id).execute()
                    if curr_media_row.data:
                        v_id = curr_media_row.data[0].get("vault_id")
                        m_id = curr_media_row.data[0].get("memory_id")
                        u_id = curr_media_row.data[0].get("owner_id")

                        query = supabase.table("media").select("id, metadata").neq("id", media_id)
                        if v_id:
                            query = query.eq("vault_id", v_id)
                        elif m_id:
                            query = query.eq("memory_id", m_id)
                        else:
                            query = query.eq("owner_id", u_id)

                        sibling_media = query.limit(50).execute().data or []
                        if sibling_media:
                            sibling_ids = [s["id"] for s in sibling_media]
                            emb_res = supabase.table("media_embeddings").select("media_id, clip_embedding, embedding").in_("media_id", sibling_ids).execute()

                            curr_arr = np.array(embedding, dtype=np.float32)
                            curr_norm = np.linalg.norm(curr_arr)
                            if curr_norm > 0:
                                curr_unit = curr_arr / curr_norm
                                for sibling_emb in (emb_res.data or []):
                                    s_vec = sibling_emb.get("clip_embedding") or sibling_emb.get("embedding")
                                    if s_vec and len(s_vec) == len(curr_unit):
                                        s_arr = np.array(s_vec, dtype=np.float32)
                                        s_norm = np.linalg.norm(s_arr)
                                        if s_norm > 0:
                                            sim = float(np.dot(curr_unit, s_arr / s_norm))
                                            if sim >= 0.92:
                                                orig_id = str(sibling_emb["media_id"])
                                                duplicate_info = {
                                                    "is_alternate": True,
                                                    "original_media_id": orig_id,
                                                    "similarity": round(sim, 4),
                                                    "label": "Alternate"
                                                }
                                                logger.info(f"Media {media_id} detected as near-duplicate / alternate of {orig_id} (similarity: {sim:.4f})")

                                                # Update original media item to track this alternate
                                                try:
                                                    orig_row = next((s for s in sibling_media if str(s["id"]) == orig_id), None)
                                                    if orig_row:
                                                        orig_meta = orig_row.get("metadata") or {}
                                                        orig_dup = orig_meta.get("duplicate_info") or {"is_alternate": False, "label": "Original", "alternate_media_ids": []}
                                                        alt_list = orig_dup.get("alternate_media_ids", [])
                                                        if media_id not in alt_list:
                                                            alt_list.append(media_id)
                                                            orig_dup["alternate_media_ids"] = alt_list
                                                            orig_dup["alternate_count"] = len(alt_list)
                                                            orig_meta["duplicate_info"] = orig_dup
                                                            supabase.table("media").update({"metadata": orig_meta}).eq("id", orig_id).execute()
                                                except Exception as orig_err:
                                                    logger.warning(f"Could not update original media metadata: {orig_err}")
                                                break
            except Exception as dup_err:
                logger.warning(f"Duplicate detection check skipped: {dup_err}")

            # Combine exif metadata + zero-shot tags + duplicate info into single metadata JSON
            combined_metadata = {
                "exif": meta,
                "ai_tags": tags,
                "duplicate_info": duplicate_info,
                "processed_at": datetime.now(timezone.utc).isoformat()
            }

            # Update PostgreSQL media table with extracted details
            supabase.table("media").update({
                "taken_at": meta.get("taken_at"),
                "latitude": meta.get("latitude"),
                "longitude": meta.get("longitude"),
                "location_name": meta.get("location_name") or (combined_metadata["ai_tags"]["scenes"][0] if combined_metadata["ai_tags"]["scenes"] else None),
                "metadata": combined_metadata
            }).eq("id", media_id).execute()

            # Insert or update embedding in media_embeddings table (with multi-vector support)
            if any(embedding):
                upsert_payload = {
                    "media_id": media_id,
                    "embedding": embedding,
                    "clip_embedding": embedding,
                }
                supabase.table("media_embeddings").upsert(upsert_payload).execute()
                logger.info(f"Successfully processed media {media_id} and stored embedding.")

        except Exception as e:
            logger.error(f"Failed to save AI extraction results for media {media_id}: {e}")

    @staticmethod
    def _parse_gps(tags: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        """Helper to convert EXIF GPS tags into decimal degree floats."""
        def _to_decimal(val):
            try:
                d = float(val.values[0].num) / float(val.values[0].den)
                m = float(val.values[1].num) / float(val.values[1].den)
                s = float(val.values[2].num) / float(val.values[2].den)
                return d + (m / 60.0) + (s / 3600.0)
            except Exception:
                return None

        try:
            gps_lat = tags.get("GPS GPSLatitude")
            gps_lat_ref = tags.get("GPS GPSLatitudeRef")
            gps_lon = tags.get("GPS GPSLongitude")
            gps_lon_ref = tags.get("GPS GPSLongitudeRef")

            if gps_lat and gps_lat_ref and gps_lon and gps_lon_ref:
                lat = _to_decimal(gps_lat)
                lon = _to_decimal(gps_lon)
                if lat is not None and lon is not None:
                    if str(gps_lat_ref) != "N":
                        lat = -lat
                    if str(gps_lon_ref) != "E":
                        lon = -lon
                    return lat, lon
        except Exception:
            pass
        return None, None

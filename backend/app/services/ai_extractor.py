import io
import logging
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

# Lazy-loaded CLIP Model to keep memory footprint low at startup
_clip_model = None

def get_clip_model():
    global _clip_model
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
            text_embeddings = model.encode([f"a photo of {x}" for x in all_labels])

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
            # Combine exif metadata + zero-shot tags into single metadata JSON
            combined_metadata = {
                "exif": meta,
                "ai_tags": tags,
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

            # Insert or update embedding in media_embeddings table
            if any(embedding):
                supabase.table("media_embeddings").upsert({
                    "media_id": media_id,
                    "embedding": embedding
                }).execute()
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

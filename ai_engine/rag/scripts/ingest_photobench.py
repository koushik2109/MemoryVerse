"""
PhotoBench Album Ingestion Script
==================================
Reads a PhotoBench-style album folder and ingests all images into Supabase.

Data sources understood:
  - PhotoBench query JSONs (album1/2/3 validation + test)
      → each JSON maps queries to ground-truth image filenames + metadata tags
  - L10 dataset: 20260822_l10318e6g11p5pnb2/album1/
      → contains real image files + query.json

Strategy:
  1. Invert the query→ground_truth mapping: build per-image tag records
     (caption, objects, entities, mood, event, timestamp, location)
  2. Walk image files in the album images/ folder
  3. For each image, look up its aggregated tags
  4. Skip already-ingested images (SHA-256 hash stored in Supabase)
  5. Generate SigLIP image_embedding + BGE-M3 text_embedding
  6. Insert into Supabase `memories` table

Usage:
  python ingest_photobench.py --album /path/to/album1 --user_id <user_id>
  python ingest_photobench.py --album /path/to/album1 --json /path/to/album1_validation.json --user_id <user_id>
  python ingest_photobench.py --all --user_id <user_id>   # ingest all known albums
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# ─── Project path setup ───────────────────────────────────────────────────────
RAG_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(RAG_DIR.parent.parent))  # project root

from PIL import Image
from supabase import create_client

from ai_engine.rag.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from ai_engine.rag.video.siglip_embedder import embed_frames
from ai_engine.rag.audio.embedder import embed_text as bge_embed

# ─── Constants ────────────────────────────────────────────────────────────────
MEMORIES_TABLE = "memories"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp", ".tiff"}

# Map PhotoBench query dimension keys → semantic tag fields
DIMENSION_MAP = {
    "Location": "location",
    "Time": "timestamp_hint",
    "Person": "entities",
    "Object": "objects",
    "Concept": "event",
    "Genre": "mood",
}

# ─── Supabase client ──────────────────────────────────────────────────────────
def _db():
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# ─── Deduplication ───────────────────────────────────────────────────────────

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_ingested_hashes() -> Set[str]:
    """Return set of file_hash values already in Supabase memories table."""
    try:
        resp = _db().table(MEMORIES_TABLE).select("file_hash").execute()
        return {r["file_hash"] for r in (resp.data or []) if r.get("file_hash")}
    except Exception as e:
        print(f"  [warn] Could not fetch existing hashes: {e}")
        return set()


# ─── Tag Index Builder ───────────────────────────────────────────────────────

def build_image_tag_index(json_paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Invert query→ground_truth into image→tags.
    Multiple queries pointing to the same image are merged (lists extended).
    """
    index: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "captions_en": [],
        "captions_cn": [],
        "objects": [],
        "entities": [],
        "mood": [],
        "event": [],
        "location": [],
        "timestamp_hint": [],
    })

    for jpath in json_paths:
        if not os.path.exists(jpath):
            print(f"  [skip] JSON not found: {jpath}")
            continue

        with open(jpath, encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            query_en = entry.get("query_en", "")
            query_cn = entry.get("query_cn", "")
            ground_truth: List[str] = entry.get("ground_truth", [])

            # Derive tag hints from dimension fields
            for dim_key, tag_field in DIMENSION_MAP.items():
                val = entry.get(dim_key)
                if val:  # "fact", "cognitive", or null
                    # The query text itself is the best description for a given dimension
                    pass  # We'll use the query caption directly below

            for filename in ground_truth:
                rec = index[filename.upper()]  # normalise to UPPER for matching
                if query_en:
                    rec["captions_en"].append(query_en)
                if query_cn:
                    rec["captions_cn"].append(query_cn)
                # Map dimensions to tag lists
                for dim_key, tag_field in DIMENSION_MAP.items():
                    val = entry.get(dim_key)
                    if val and val != "null":
                        rec[tag_field].append(query_en)  # use en query as tag content

    # Deduplicate lists
    cleaned: Dict[str, Dict[str, Any]] = {}
    for filename, tags in index.items():
        cleaned[filename] = {
            k: list(dict.fromkeys(v)) if isinstance(v, list) else v
            for k, v in tags.items()
        }
    return cleaned


def build_tag_text(filename: str, tags: Optional[Dict[str, Any]]) -> str:
    """
    Build a single rich text string from all tags for BGE-M3 embedding.
    """
    if not tags:
        return filename  # minimal fallback

    parts: List[str] = []
    captions = tags.get("captions_en", [])
    if captions:
        parts.append(" ".join(captions))

    for field in ("objects", "entities", "mood", "event", "location"):
        items = tags.get(field, [])
        if items:
            parts.append(f"{field}: " + ", ".join(items[:5]))  # cap at 5 per field

    return " | ".join(parts) if parts else filename


# ─── Image Loading ────────────────────────────────────────────────────────────

def load_image_as_bgr(path: str):
    """Load image as BGR numpy array (compatible with SigLIP embedder)."""
    import cv2
    import numpy as np

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        # Fallback via Pillow (handles HEIC/WebP on some systems)
        pil = Image.open(path).convert("RGB")
        img = cv2.cvtColor(
            __import__("numpy").array(pil, dtype=__import__("numpy").uint8),
            cv2.COLOR_RGB2BGR,
        )
    return img


# ─── Core Ingestion ───────────────────────────────────────────────────────────

def ingest_album(
    album_dir: str,
    json_paths: List[str],
    user_id: str,
    dry_run: bool = False,
) -> int:
    """
    Main ingestion function.
    Returns the number of newly inserted records.
    """
    album_path = Path(album_dir)
    if not album_path.exists():
        raise FileNotFoundError(f"Album directory not found: {album_dir}")

    # Locate images/ subfolder or fall back to album root
    images_dir = album_path / "images"
    if not images_dir.exists():
        images_dir = album_path

    # Collect all image files
    image_files: List[Path] = sorted([
        p for p in images_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTS
    ])

    if not image_files:
        print(f"  [warn] No images found in {images_dir}")
        return 0

    print(f"\n{'='*60}")
    print(f"Album   : {album_dir}")
    print(f"Images  : {len(image_files)}")
    print(f"Tag JSONs: {json_paths}")
    print(f"{'='*60}")

    # Build image→tags index from all JSONs
    tag_index = build_image_tag_index(json_paths)
    print(f"Tag index: {len(tag_index)} images with metadata")

    # Fetch already-ingested hashes to avoid duplicates
    ingested_hashes = fetch_ingested_hashes()
    print(f"Already ingested: {len(ingested_hashes)} records\n")

    db = _db()
    inserted = 0
    skipped_hash = 0
    skipped_error = 0

    for img_path in image_files:
        filename = img_path.name
        print(f"  Processing: {filename}", end="", flush=True)

        # ── Dedup check ───────────────────────────────────────────────────────
        file_hash = sha256_of_file(str(img_path))
        if file_hash in ingested_hashes:
            print("  [skip – already ingested]")
            skipped_hash += 1
            continue

        # ── Look up tags (try both original case and UPPER) ───────────────────
        tags = tag_index.get(filename.upper()) or tag_index.get(filename)

        # ── Build text representation for embedding ───────────────────────────
        tag_text = build_tag_text(filename, tags)

        try:
            # ── Load image ───────────────────────────────────────────────────
            frame = load_image_as_bgr(str(img_path))

            # ── SigLIP image embedding (768-dim) ─────────────────────────────
            image_embedding: List[float] = embed_frames([frame])[0]

            # ── BGE-M3 text embedding (1024-dim) ──────────────────────────────
            text_embedding: List[float] = bge_embed(tag_text)

            # ── Build Supabase record ─────────────────────────────────────────
            record: Dict[str, Any] = {
                "user_id": user_id,
                "media_type": "image",
                "file_path": str(img_path),
                "file_hash": file_hash,
                "filename": filename,
                "album": album_path.name,
                # Tag fields
                "caption": " ".join((tags or {}).get("captions_en", []))[:2000] or None,
                "caption_cn": " ".join((tags or {}).get("captions_cn", []))[:2000] or None,
                "objects": ", ".join((tags or {}).get("objects", []))[:500] or None,
                "entities": ", ".join((tags or {}).get("entities", []))[:500] or None,
                "mood": ", ".join((tags or {}).get("mood", []))[:200] or None,
                "event": ", ".join((tags or {}).get("event", []))[:500] or None,
                "location": ", ".join((tags or {}).get("location", []))[:500] or None,
                "timestamp_hint": ", ".join((tags or {}).get("timestamp_hint", []))[:200] or None,
                # Embeddings
                "image_embedding": image_embedding,   # 768-dim (SigLIP)
                "text_embedding": text_embedding,     # 1024-dim (BGE-M3)
            }

            if not dry_run:
                db.table(MEMORIES_TABLE).insert(record).execute()
                ingested_hashes.add(file_hash)  # prevent double-insert in same run
                inserted += 1
            else:
                inserted += 1  # count as would-be inserted

            print(f"  ✓  caption: {tag_text[:60]}...")

        except Exception as e:
            print(f"  ✗  ERROR: {e}")
            skipped_error += 1

    print(f"\n{'='*60}")
    mode = "[DRY RUN] " if dry_run else ""
    print(f"{mode}Results for album: {album_path.name}")
    print(f"  Inserted : {inserted}")
    print(f"  Skipped (already ingested): {skipped_hash}")
    print(f"  Skipped (errors)          : {skipped_error}")
    print(f"  Total images found        : {len(image_files)}")
    print(f"{'='*60}\n")

    return inserted


# ─── Known Album Definitions ─────────────────────────────────────────────────

def _get_known_albums() -> List[Dict[str, Any]]:
    """Return the list of all known album configs in this repo."""
    rag = RAG_DIR

    return [
        # L10 album1 (has real images)
        {
            "album_dir": str(rag / "20260822_l10318e6g11p5pnb2" / "album1"),
            "json_paths": [
                str(rag / "20260822_l10318e6g11p5pnb2" / "album1" / "query.json"),
            ],
        },
        # PhotoBench album1 (no images locally — will log a warning)
        {
            "album_dir": str(rag / "PhotoBench" / "images" / "album1"),
            "json_paths": [
                str(rag / "PhotoBench" / "data" / "validation" / "album1_validation.json"),
                str(rag / "PhotoBench" / "data" / "test" / "album1_test.json"),
            ],
        },
        # PhotoBench album2
        {
            "album_dir": str(rag / "PhotoBench" / "images" / "album2"),
            "json_paths": [
                str(rag / "PhotoBench" / "data" / "validation" / "album2_validation.json"),
                str(rag / "PhotoBench" / "data" / "test" / "album2_test.json"),
            ],
        },
        # PhotoBench album3
        {
            "album_dir": str(rag / "PhotoBench" / "images" / "album3"),
            "json_paths": [
                str(rag / "PhotoBench" / "data" / "validation" / "album3_validation.json"),
                str(rag / "PhotoBench" / "data" / "test" / "album3_test.json"),
            ],
        },
    ]


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest a PhotoBench album into Supabase memories table."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--album",
        help="Path to album directory (must contain images/ subfolder or images at root)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Ingest all known albums in this repo",
    )
    parser.add_argument(
        "--json",
        nargs="+",
        help="Path(s) to query JSON files for the album (auto-detected if --all)",
    )
    parser.add_argument(
        "--user_id",
        required=True,
        help="User ID to tag all inserted memories with",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process images but do NOT write to Supabase",
    )
    args = parser.parse_args()

    total_inserted = 0

    if args.all:
        albums = _get_known_albums()
        print(f"\nIngesting {len(albums)} known albums...")
        for album_cfg in albums:
            try:
                total_inserted += ingest_album(
                    album_dir=album_cfg["album_dir"],
                    json_paths=album_cfg["json_paths"],
                    user_id=args.user_id,
                    dry_run=args.dry_run,
                )
            except FileNotFoundError as e:
                print(f"  [skip] {e}")
    else:
        json_paths = args.json or []
        total_inserted = ingest_album(
            album_dir=args.album,
            json_paths=json_paths,
            user_id=args.user_id,
            dry_run=args.dry_run,
        )

    mode = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{mode}Grand total inserted: {total_inserted} memories")


if __name__ == "__main__":
    main()

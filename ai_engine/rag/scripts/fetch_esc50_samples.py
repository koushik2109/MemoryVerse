#!/usr/bin/env python3
"""Fetch authentic raw WAV audio recordings from the official ESC-50 dataset.

Selects recordings for the 13 diagnostic categories tested in our cross-modal RAG suite:
  - airplane, chainsaw, chirping_birds, clapping, dog, door_wood_knock,
    fireworks, pouring_water, rooster, sheep, thunderstorm, train, vacuum_cleaner
"""

from __future__ import annotations

import csv
import io
import json
import logging
import ssl
import urllib.request
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_esc50")

REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIO_DIR = REPO_ROOT / "dataset" / "audio" / "esc50"
MANIFEST_PATH = REPO_ROOT / "dataset" / "audio" / "esc50_manifest.json"

CSV_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
RAW_BASE_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/"

# Bypass macOS Python missing local certs
SSL_CTX = ssl._create_unverified_context()

TARGET_CATEGORIES = {
    "airplane",
    "chainsaw",
    "chirping_birds",
    "clapping",
    "dog",
    "door_wood_knock",
    "fireworks",
    "pouring_water",
    "rooster",
    "sheep",
    "thunderstorm",
    "train",
    "vacuum_cleaner",
}

CLIPS_PER_CATEGORY = 4


def main() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Fetching ESC-50 metadata CSV from GitHub...")

    req = urllib.request.Request(CSV_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=SSL_CTX) as resp:
        csv_text = resp.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(csv_text))
    grouped: Dict[str, List[Dict[str, Any]]] = {cat: [] for cat in TARGET_CATEGORIES}

    for row in reader:
        cat = row.get("category", "")
        if cat in TARGET_CATEGORIES:
            grouped[cat].append(row)

    manifest_records: List[Dict[str, Any]] = []
    total_downloaded = 0

    for cat in sorted(TARGET_CATEGORIES):
        rows = grouped[cat][:CLIPS_PER_CATEGORY]
        logger.info("Downloading %d clips for category '%s'...", len(rows), cat)
        for r in rows:
            fname = r["filename"]
            target_path = AUDIO_DIR / fname
            if not target_path.exists():
                url = RAW_BASE_URL + fname
                try:
                    req_clip = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req_clip, context=SSL_CTX) as clip_resp:
                        target_path.write_bytes(clip_resp.read())
                    total_downloaded += 1
                except Exception as exc:
                    logger.warning("Failed to download %s: %s", fname, exc)
                    continue

            manifest_records.append({
                "filename": fname,
                "file_path": str(target_path.relative_to(REPO_ROOT)),
                "category": cat,
                "fold": r.get("fold"),
                "target": r.get("target"),
                "size_bytes": target_path.stat().st_size if target_path.exists() else 0,
            })

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "total_clips": len(manifest_records),
            "categories": sorted(list(TARGET_CATEGORIES)),
            "clips": manifest_records,
        }, f, indent=2)

    logger.info("Downloaded %d new clips. Total manifest records: %d", total_downloaded, len(manifest_records))
    logger.info("Manifest saved to: %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""kaggle/build_frozen_dataset_v14.py
======================================
Builds kaggle/frozen_inputs_v14.zip including all raw benchmark images,
OCR texts, results/v4/ gapfill artifacts, and CLI evaluation runners.
"""

import os
import sys
import zipfile
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ZIP = REPO_ROOT / "kaggle" / "frozen_inputs_v14.zip"

FILES_TO_PACK = [
    # Benchmark core
    "ai_engine/rag/data/v2/public_benchmark_v2.json",
    "ai_engine/rag/data/v2/frozen_probes_with_paraphrases.json",
    "ai_engine/rag/data/v2/sroie_degraded_ocr.json",
    "results/v3/heavy_paraphrases_v3.json",
    "results/v3/sroie_tesseract_clean_ocr.json",
    "results/v3/sroie_tesseract_degraded_ocr.json",
    "results/v3/funsd_apple_vision_ocr.json",
    "results/v3/funsd_tesseract_ocr.json",
    "results/v3/funsd_probes.json",
    # COCO Karpathy data & embeddings
    "ai_engine/rag/data/coco_blip_captions.json",
    "ai_engine/rag/data/coco_karpathy_test.json",
    "ai_engine/rag/data/coco_clip_image_embeddings.npy",
    "results/v3/coco_siglip_image_embeddings.npy",
    "ai_engine/rag/data/expanded_multihop_results.json",
    # Paper & CLI runners
    "eval_pipeline.py",
    "eval_docvqa.py",
    "profile_latency.py",
    "ijcds_body.tex",
    "ijcds_header.tex",
    "main.tex",
    "kaggle/requirements_kaggle.txt",
    "kaggle/kaggle_run_v14_reproducibility.ipynb",
    "kaggle/README_KAGGLE.md",
]

DIRECTORY_TREES_TO_PACK = [
    "results/v4",
    "ai_engine/rag/data/sroie/img",
    "ai_engine/rag/data/sroie/img_degraded",
    "ai_engine/rag/data/coco_1k_images",
]

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("BUILDING FROZEN INPUTS V14 DATASET (kaggle/frozen_inputs_v14.zip)")
    print("=" * 80)

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    packed_count = 0
    total_bytes = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel_str in FILES_TO_PACK:
            p = REPO_ROOT / rel_str
            if p.exists():
                zf.write(p, arcname=rel_str)
                packed_count += 1
                total_bytes += p.stat().st_size
            else:
                print(f"  [Warning: Missing] {rel_str}")

        for dir_rel in DIRECTORY_TREES_TO_PACK:
            dp = REPO_ROOT / dir_rel
            if dp.exists():
                for item in dp.rglob("*"):
                    if item.is_file() and not item.name.startswith(".") and not item.name.endswith(".zip"):
                        arc_name = str(item.relative_to(REPO_ROOT))
                        zf.write(item, arcname=arc_name)
                        packed_count += 1
                        total_bytes += item.stat().st_size

    zip_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    zip_sha = sha256_of_file(OUTPUT_ZIP)

    print(f"Total packed files:      {packed_count}")
    print(f"Final Zip archive size:  {zip_mb:.2f} MB")
    print(f"SHA-256 Digest:          {zip_sha}")
    print(f"[OK] Created: {OUTPUT_ZIP}")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
kaggle/build_frozen_dataset.py: Bundles all frozen inputs, benchmark files,
probe splits, Apple Vision OCR texts, code, and images into frozen_inputs_v5.zip
for Kaggle execution.

Outputs:
  kaggle/frozen_inputs_v5.zip
  Prints total zip size and SHA-256 hash.
"""

import os
import sys
import zipfile
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ZIP = REPO_ROOT / "kaggle" / "frozen_inputs_v5.zip"

FILES_TO_PACK = [
    # Benchmark core
    "ai_engine/rag/data/v2/public_benchmark_v2.json",
    "ai_engine/rag/data/v2/frozen_probes_with_paraphrases.json",
    "ai_engine/rag/data/v2/sroie_degraded_ocr.json",
    "results/v3/heavy_paraphrases_v3.json",
    "results/v3/dev_tuned_parameters_v3.json",
    "results/v3/ocr_error_rates.csv",
    "results/v3/headline_regime_grid_v3.csv",
    "results/v3/headline_hypothesis_tests.csv",
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
    # Documentation, preregistration & configs
    "PREREG_V5.md",
    "CONFIG_LOG.md",
    "MANIFEST.json",
    "kaggle/requirements_kaggle.txt",
    "kaggle/kaggle_run_v5.py",
]

DIRECTORY_TREES_TO_PACK = [
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
    print("BUILDING FROZEN INPUTS DATASET FOR KAGGLE (frozen_inputs_v5.zip)")
    print(f"Repository Root: {REPO_ROOT}")
    print(f"Target Output:   {OUTPUT_ZIP}")
    print("=" * 80)

    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    packed_count = 0
    total_bytes = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Pack single files
        for rel_str in FILES_TO_PACK:
            p = REPO_ROOT / rel_str
            if p.exists():
                zf.write(p, arcname=rel_str)
                packed_count += 1
                total_bytes += p.stat().st_size
                print(f"  [Packed File] {rel_str} ({p.stat().st_size / 1024:.1f} KB)")
            else:
                print(f"  [Warning: Optional/Missing] {rel_str}")

        # Pack directory trees
        for dir_rel in DIRECTORY_TREES_TO_PACK:
            dp = REPO_ROOT / dir_rel
            if dp.exists():
                print(f"  [Packing Directory Tree] {dir_rel}...")
                dir_packed = 0
                for item in dp.rglob("*"):
                    if item.is_file() and not item.name.startswith("."):
                        arc_name = str(item.relative_to(REPO_ROOT))
                        zf.write(item, arcname=arc_name)
                        dir_packed += 1
                        total_bytes += item.stat().st_size
                packed_count += dir_packed
                print(f"    Packed {dir_packed} items from {dir_rel}")
            else:
                print(f"  [Warning: Directory Missing] {dir_rel}")

    zip_bytes = OUTPUT_ZIP.stat().st_size
    zip_mb = zip_bytes / (1024 * 1024)
    zip_sha = sha256_of_file(OUTPUT_ZIP)

    print("\n" + "=" * 80)
    print("FROZEN DATASET BUNDLE COMPLETED SUCCESSFULLY")
    print(f"Total packed files:      {packed_count}")
    print(f"Uncompressed data size:  {total_bytes / (1024 * 1024):.2f} MB")
    print(f"Final Zip archive size:  {zip_mb:.2f} MB ({zip_bytes} bytes)")
    print(f"SHA-256 Digest:          {zip_sha}")
    print("=" * 80)

    # Assert size is < 2 GB
    assert zip_mb < 2000.0, f"Error: frozen dataset zip exceeds 2 GB ({zip_mb:.2f} MB)"

if __name__ == "__main__":
    main()

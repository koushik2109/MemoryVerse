"""
Generate Degraded Images and Real-OCR Noise (Phase 2.2)
========================================================
Applies realistic physical degradations before OCR:
1. Gaussian blur (radius=1.5)
2. 50% spatial downscale
3. +-4 degree rotation (with white border fill)
4. JPEG compression artifact (quality=20)

Runs native Apple Vision OCR on the degraded images and measures resulting CER/WER.
Saves results to ai_engine/rag/data/v2/sroie_degraded_ocr.json
"""

import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[3]
SROIE_DIR = REPO_ROOT / "ai_engine" / "rag" / "data" / "sroie"
IMG_SRC_DIR = SROIE_DIR / "img"
IMG_DEG_DIR = SROIE_DIR / "img_degraded"
SWIFT_BIN = REPO_ROOT / "ai_engine" / "rag" / "scripts" / "run_ocr_apple_vision"
BENCHMARK_PATH = REPO_ROOT / "ai_engine" / "rag" / "data" / "v2" / "public_benchmark_v2.json"
OUTPUT_PATH = REPO_ROOT / "ai_engine" / "rag" / "data" / "v2" / "sroie_degraded_ocr.json"


def compute_cer_wer(ref: str, hyp: str) -> Tuple[float, float]:
    r_words = ref.strip().split()
    h_words = hyp.strip().split()

    def lev(seq1, seq2):
        n, m = len(seq1), len(seq2)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[n][m]

    word_err = lev(r_words, h_words)
    wer = word_err / max(1, len(r_words))

    r_chars = list(ref.strip())
    h_chars = list(hyp.strip())
    char_err = lev(r_chars, h_chars)
    cer = char_err / max(1, len(r_chars))
    return float(cer), float(wer)


def main():
    print("Generating degraded SROIE images (Blur + Downscale + Rotation + JPEG Q20)...")
    IMG_DEG_DIR.mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(IMG_SRC_DIR.glob("*.jpg")))
    print(f"Found {len(img_files)} source images.")

    for idx, img_path in enumerate(img_files):
        stem = img_path.stem
        dst_path = IMG_DEG_DIR / f"{stem}.jpg"
        if not dst_path.exists():
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                # 1. Gaussian blur
                img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
                # 2. Downscale 50%
                w, h = img.size
                img = img.resize((max(1, w // 2), max(1, h // 2)), Image.Resampling.BILINEAR)
                # 3. Deterministic rotation +-4 deg
                angle = 4.0 if (hash(stem) % 2 == 0) else -4.0
                img = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=(255, 255, 255))
                # 4. Save with JPEG quality 20
                img.save(dst_path, "JPEG", quality=20)

    print("Degraded images generated.")

    # Run Apple Vision OCR on degraded images
    from build_public_benchmark_v2 import compile_swift_ocr, run_ocr_on_all_sroie
    swift_bin = Path("/tmp/vision_ocr_worker_bin")
    if not swift_bin.exists():
        compile_swift_ocr(swift_bin)

    deg_image_paths = sorted(list(IMG_DEG_DIR.glob("*.jpg")))
    degraded_ocr_by_stem = run_ocr_on_all_sroie(deg_image_paths, swift_bin, batch_size=50)

    # Load benchmark to compute CER and WER against ground truth
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    # Compute CER and WER
    cer_list, wer_list = [], []
    degraded_corpus_data = {}

    for doc in benchmark["records"]:
        item_id = doc["id"]
        if doc.get("type") == "receipt_document":
            stem = Path(doc["image_path"]).stem
            gt_text = doc["ground_truth_text"]
            lines = degraded_ocr_by_stem.get(stem, [])
            deg_text = " \n ".join(lines)
            cer, wer = compute_cer_wer(gt_text, deg_text)
            cer_list.append(cer)
            wer_list.append(wer)
            degraded_corpus_data[item_id] = {
                "stem": stem,
                "degraded_ocr_lines": lines,
                "degraded_ocr_text": deg_text,
                "cer": cer,
                "wer": wer,
            }

    mean_cer = sum(cer_list) / max(1, len(cer_list))
    mean_wer = sum(wer_list) / max(1, len(wer_list))

    print(f"Degraded OCR Metrics across {len(cer_list)} receipts:")
    print(f"  Mean CER: {mean_cer:.4f}")
    print(f"  Mean WER: {mean_wer:.4f}")

    output_payload = {
        "condition": "real_ocr_degraded",
        "description": "Apple Vision OCR applied to physical images degraded with Gaussian blur (r=1.5), 50% downscale, +/-4 deg rotation, and JPEG Q20",
        "mean_cer": mean_cer,
        "mean_wer": mean_wer,
        "items": degraded_corpus_data,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"Saved degraded OCR results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

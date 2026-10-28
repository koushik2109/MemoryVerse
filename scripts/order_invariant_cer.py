#!/usr/bin/env python3
"""
order_invariant_cer.py
-------------------------
Computes order-invariant / line-aligned Character Error Rate (CER) and Word Error Rate (WER)
on SROIE receipts to separate reading-order permutation artifacts from true optical recognition noise.
Also computes the Bag-of-Words symmetric-difference rate.

Usage:
    python scripts/order_invariant_cer.py [--engine apple|tesseract|easyocr|all] [--condition clean|degraded|all]
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SROIE_BOX_DIR = REPO_ROOT / "ai_engine" / "rag" / "data" / "sroie" / "box"
BENCHMARK_PATH = REPO_ROOT / "ai_engine" / "rag" / "data" / "v2" / "public_benchmark_v2.json"
DEGRADED_OCR_PATH = REPO_ROOT / "ai_engine" / "rag" / "data" / "v2" / "sroie_degraded_ocr.json"
TESS_CLEAN_PATH = REPO_ROOT / "results" / "v3" / "sroie_tesseract_clean_ocr.json"
TESS_DEG_PATH = REPO_ROOT / "results" / "v3" / "sroie_tesseract_degraded_ocr.json"
EASY_CLEAN_PATH = REPO_ROOT / "results" / "v9" / "sroie_easyocr_clean_ocr.json"
EASY_DEG_PATH = REPO_ROOT / "results" / "v9" / "sroie_easyocr_degraded_ocr.json"
OUTPUT_CSV = REPO_ROOT / "results" / "order_invariant_cer_summary.csv"


def normalize_text(text: str) -> str:
    """Normalizes text by lowercasing, stripping punctuation, and collapsing whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_levenshtein(seq1: List[Any], seq2: List[Any]) -> int:
    """Computes exact Levenshtein edit distance between two sequences."""
    if seq1 == seq2:
        return 0
    if not seq1:
        return len(seq2)
    if not seq2:
        return len(seq1)
    if len(seq1) > len(seq2):
        seq1, seq2 = seq2, seq1
    prev = list(range(len(seq1) + 1))
    for j, elem2 in enumerate(seq2):
        curr = [j + 1] * (len(seq1) + 1)
        for i, elem1 in enumerate(seq1):
            cost = 0 if elem1 == elem2 else 1
            curr[i + 1] = min(curr[i] + 1, prev[i + 1] + 1, prev[i] + cost)
        prev = curr
    return prev[-1]


def compute_cer_wer_line_aligned(ref_lines: List[str], hyp_lines: List[str]) -> Tuple[float, float, float]:
    """
    Line-order aligned normalized CER, WER, and BoW symmetric difference rate.
    Matches each OCR line to the best matching ground-truth line to isolate
    character recognition quality from spatial reading-order permutations.
    """
    norm_gt = [normalize_text(l) for l in ref_lines if normalize_text(l)]
    norm_hyp = [normalize_text(l) for l in hyp_lines if normalize_text(l)]

    if not norm_gt:
        total_hyp_c = sum(len(l) for l in norm_hyp)
        total_hyp_w = sum(len(l.split()) for l in norm_hyp)
        return (1.0 if total_hyp_c > 0 else 0.0), (1.0 if total_hyp_w > 0 else 0.0), 1.0

    total_gt_c = sum(len(l) for l in norm_gt)
    total_gt_w = sum(len(l.split()) for l in norm_gt)
    used_hyp = set()
    total_c_err = 0
    total_w_err = 0

    for gl in norm_gt:
        gl_chars = list(gl)
        gl_words = gl.split()
        best_c_err = len(gl_chars)
        best_w_err = len(gl_words)
        best_j = -1
        for j, hl in enumerate(norm_hyp):
            if j in used_hyp:
                continue
            if abs(len(gl) - len(hl)) > len(gl) * 0.70:
                continue
            c_err = compute_levenshtein(gl_chars, list(hl))
            if c_err < best_c_err:
                best_c_err = c_err
                best_j = j
                best_w_err = compute_levenshtein(gl_words, hl.split())

        if best_j != -1 and best_c_err < len(gl_chars) * 0.75:
            used_hyp.add(best_j)
            total_c_err += best_c_err
            total_w_err += best_w_err
        else:
            total_c_err += len(gl_chars)
            total_w_err += len(gl_words)

    for j, hl in enumerate(norm_hyp):
        if j not in used_hyp:
            total_c_err += len(hl)
            total_w_err += len(hl.split())

    cer = total_c_err / max(1, total_gt_c)
    wer_seq = total_w_err / max(1, total_gt_w)

    from collections import Counter
    ref_counts = Counter([w for l in norm_gt for w in l.split()])
    hyp_counts = Counter([w for l in norm_hyp for w in l.split()])
    all_words = set(ref_counts.keys()).union(set(hyp_counts.keys()))
    diff_count = sum(abs(ref_counts[w] - hyp_counts[w]) for w in all_words)
    mer_bow = diff_count / max(1, sum(ref_counts.values()))

    return float(cer), float(wer_seq), float(mer_bow)


def load_dataset():
    """Loads ground truth box lines and all OCR engine outputs across all 626 SROIE documents."""
    gt_by_stem = {}
    for bf in sorted(SROIE_BOX_DIR.glob("*.csv")):
        stem = bf.stem
        lines = []
        with open(bf, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                parts = raw_line.strip().split(",")
                if len(parts) >= 9:
                    lines.append(",".join(parts[8:]).strip())
                elif len(parts) > 0 and parts[0]:
                    lines.append(parts[-1].strip())
        gt_by_stem[stem] = lines

    # Apple Vision
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench = json.load(f)
    apple_clean = {
        Path(r["image_path"]).stem: [l.strip() for l in r.get("real_ocr_text", "").splitlines() if l.strip()]
        for r in bench["records"] if r["type"] == "receipt_document"
    }
    with open(DEGRADED_OCR_PATH, "r", encoding="utf-8") as f:
        deg_data = json.load(f)["items"]
    apple_deg = {v["stem"]: v["degraded_ocr_lines"] for v in deg_data.values()}

    # Tesseract
    with open(TESS_CLEAN_PATH, "r", encoding="utf-8") as f:
        tess_clean = json.load(f)
    with open(TESS_DEG_PATH, "r", encoding="utf-8") as f:
        tess_deg = json.load(f)

    # EasyOCR
    with open(EASY_CLEAN_PATH, "r", encoding="utf-8") as f:
        easy_clean = json.load(f)
    with open(EASY_DEG_PATH, "r", encoding="utf-8") as f:
        easy_deg = json.load(f)

    return gt_by_stem, {
        ("apple", "clean"): apple_clean,
        ("apple", "degraded"): apple_deg,
        ("tesseract", "clean"): tess_clean,
        ("tesseract", "degraded"): tess_deg,
        ("easyocr", "clean"): easy_clean,
        ("easyocr", "degraded"): easy_deg,
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate line-aligned CER and BoW symmetric difference rate.")
    parser.add_argument("--engine", choices=["apple", "tesseract", "easyocr", "all"], default="all")
    parser.add_argument("--condition", choices=["clean", "degraded", "all"], default="all")
    args = parser.parse_args()

    print("=" * 80)
    print("ORDER-INVARIANT & LINE-ALIGNED CER BENCHMARK (SROIE 626 RECEIPTS)")
    print("=" * 80)

    gt_by_stem, ocr_dict = load_dataset()
    stems = sorted(gt_by_stem.keys())
    print(f"Loaded {len(stems)} physical ground-truth receipt documents.\n")

    engines = ["apple", "tesseract", "easyocr"] if args.engine == "all" else [args.engine]
    conditions = ["clean", "degraded"] if args.condition == "all" else [args.condition]

    results = []

    print(f"{'OCR Engine':<25} | {'Condition':<10} | {'Line CER':<10} | {'Line WER':<10} | {'BoW Rate':<10}")
    print("-" * 75)

    for eng in engines:
        for cond in conditions:
            key = (eng, cond)
            ocr_map = ocr_dict.get(key, {})
            cers, wers, bows = [], [], []

            for stem in stems:
                gt_lines = gt_by_stem[stem]
                hyp_lines = ocr_map.get(stem, [])
                c, w, b = compute_cer_wer_line_aligned(gt_lines, hyp_lines)
                cers.append(c)
                wers.append(w)
                bows.append(b)

            mean_c = float(np.mean(cers))
            mean_w = float(np.mean(wers))
            mean_b = float(np.mean(bows))

            eng_name = {"apple": "Apple Vision Neural", "tesseract": "Tesseract 5.5 LSTM", "easyocr": "EasyOCR 1.7.2 Deep CRNN"}[eng]
            cond_name = "Clean OCR" if cond == "clean" else "Degraded Blur"

            print(f"{eng_name:<25} | {cond_name:<10} | {mean_c:.4f}     | {mean_w:.4f}     | {mean_b:.4f}")
            results.append({
                "engine": eng_name,
                "condition": cond_name,
                "line_aligned_cer": f"{mean_c:.4f}",
                "line_aligned_wer": f"{mean_w:.4f}",
                "bow_symmetric_diff_rate": f"{mean_b:.4f}",
            })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["engine", "condition", "line_aligned_cer", "line_aligned_wer", "bow_symmetric_diff_rate"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[OK] Order-invariant error rates saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

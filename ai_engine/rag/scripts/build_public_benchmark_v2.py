"""
Build Public Scene-Text and Document Benchmark (Phase 1)
========================================================
Builds a public multimodal corpus (1,626 items: 626 SROIE receipts + 1,000 MS-COCO images)
and extracts single-target factual probes with strict 50/50 Dev/Test splitting.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
RAG_DATA_DIR = REPO_ROOT / "ai_engine" / "rag" / "data"
SROIE_DIR = RAG_DATA_DIR / "sroie"
COCO_DIR = RAG_DATA_DIR / "coco_1k_images"
V2_DIR = RAG_DATA_DIR / "v2"
V2_DIR.mkdir(parents=True, exist_ok=True)

SWIFT_OCR_CODE = r'''
import Foundation
import Vision
import AppKit

func processImage(path: String) -> [String: Any] {
    guard let img = NSImage(contentsOfFile: path),
          let cgImg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        return ["file": path, "lines": [], "line_count": 0, "status": "failed"]
    }
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = true
    let handler = VNImageRequestHandler(cgImage: cgImg, options: [:])
    try? handler.perform([req])
    var lines: [String] = []
    if let obs = req.results {
        for o in obs {
            if let top = o.topCandidates(1).first?.string {
                lines.append(top)
            }
        }
    }
    return ["file": path, "lines": lines, "line_count": lines.count, "status": "ok"]
}

let paths = Array(CommandLine.arguments.dropFirst())
var results: [[String: Any]] = []
for p in paths {
    results.append(processImage(path: p))
}

if let d = try? JSONSerialization.data(withJSONObject: results, options: []),
   let s = String(data: d, encoding: .utf8) {
    print(s)
} else {
    print("[]")
}
'''


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text)]


def compile_swift_ocr(bin_path: Path):
    swift_src = Path("/tmp/vision_ocr_worker.swift")
    swift_src.write_text(SWIFT_OCR_CODE)
    cmd = ["swiftc", "-O", str(swift_src), "-o", str(bin_path)]
    subprocess.run(cmd, check=True)
    if swift_src.exists():
        swift_src.unlink()


def run_ocr_on_all_sroie(image_paths: List[Path], bin_path: Path, batch_size: int = 50) -> Dict[str, List[str]]:
    ocr_results: Dict[str, List[str]] = {}
    print(f"Running Apple Vision OCR across {len(image_paths)} SROIE images (batch size {batch_size})...")
    t0 = time.time()
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i : i + batch_size]
        cmd = [str(bin_path)] + [str(p) for p in batch]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = proc.stdout.strip()
        idx = out.find("[")
        if idx != -1:
            batch_data = json.loads(out[idx:])
            for item in batch_data:
                stem = Path(item["file"]).stem
                ocr_results[stem] = item["lines"]
        else:
            print(f"Warning: batch {i//batch_size} produced no json: {proc.stderr[:100]}")
    t1 = time.time()
    print(f"Completed OCR on {len(ocr_results)} images in {t1 - t0:.2f}s ({ (t1 - t0)/max(1, len(ocr_results)):.3f}s/img).")
    return ocr_results


def load_sroie_ground_truth() -> Dict[str, Dict[str, Any]]:
    gt_data: Dict[str, Dict[str, Any]] = {}
    box_dir = SROIE_DIR / "box"
    key_dir = SROIE_DIR / "key"

    for csv_file in sorted(box_dir.glob("*.csv")):
        stem = csv_file.stem
        lines: List[str] = []
        try:
            with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 9:
                        lines.append(row[8].strip())
                    elif len(row) > 0:
                        lines.append(row[-1].strip())
        except Exception:
            pass

        key_file = key_dir / f"{stem}.json"
        key_info = {}
        if key_file.exists():
            try:
                with open(key_file, "r", encoding="utf-8", errors="ignore") as f:
                    key_info = json.load(f)
            except Exception:
                pass

        gt_data[stem] = {
            "box_lines": lines,
            "key_info": key_info,
            "full_text": " \n ".join(lines),
        }
    return gt_data


def compute_cer_wer(ref: str, hyp: str) -> Tuple[float, float]:
    """Levenshtein distance based CER and WER."""
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


def extract_single_target_probes(
    gt_data: Dict[str, Dict[str, Any]],
    ocr_results: Dict[str, List[str]],
    min_probes: int = 350,
) -> List[Dict[str, Any]]:
    """Extract factual probes that appear in exactly ONE target item's text."""
    # Build search corpus index for uniqueness checking
    corpus_tokens_by_doc: Dict[str, Set[str]] = {}
    doc_full_texts: Dict[str, str] = {}
    for stem, d in gt_data.items():
        doc_full_texts[stem] = (d["full_text"] + " " + " ".join(ocr_results.get(stem, []))).lower()
        corpus_tokens_by_doc[stem] = set(tokenize(doc_full_texts[stem]))

    all_probes: List[Dict[str, Any]] = []

    for stem in sorted(gt_data.keys()):
        d = gt_data[stem]
        target_id = f"sroie_{stem}"
        candidates: List[Tuple[str, str]] = []

        # Candidate 1: Company name
        comp = d["key_info"].get("company", "").strip()
        if len(tokenize(comp)) >= 2:
            candidates.append((comp, "company_entity"))

        # Candidate 2: Address line
        addr = d["key_info"].get("address", "").strip()
        if len(tokenize(addr)) >= 3:
            # take first chunk of address
            parts = [p.strip() for p in addr.split(",") if len(tokenize(p)) >= 2]
            if parts:
                candidates.append((parts[0], "address_entity"))

        # Candidate 3: Prominent box lines (product or line items)
        for line in d["box_lines"]:
            tokens = tokenize(line)
            if 3 <= len(tokens) <= 7:
                # Avoid generic headers like 'tax invoice' or 'cash bill'
                line_lower = line.lower()
                if not any(w in line_lower for w in ["invoice", "bill", "receipt", "total", "subtotal", "gst", "qty", "cash", "amount"]):
                    candidates.append((line.strip(), "receipt_line"))

        # Test candidate uniqueness
        selected_for_doc = 0
        for cand_text, probe_type in candidates:
            cand_tokens = set(tokenize(cand_text))
            if len(cand_tokens) < 2:
                continue

            # Verify uniqueness across all 626 documents
            matching_docs = []
            cand_lower = cand_text.lower()
            for other_stem, full_text in doc_full_texts.items():
                if cand_lower in full_text:
                    matching_docs.append(f"sroie_{other_stem}")

            if len(matching_docs) == 1 and matching_docs[0] == target_id:
                all_probes.append({
                    "probe_id": f"probe_{len(all_probes):04d}",
                    "query": cand_text,
                    "target_id": target_id,
                    "probe_type": probe_type,
                    "token_count": len(tokenize(cand_text)),
                })
                selected_for_doc += 1
                if selected_for_doc >= 1:
                    break  # 1 high-quality probe per target document

    print(f"Extracted {len(all_probes)} single-target probes across {len(set(p['target_id'] for p in all_probes))} unique target documents.")
    return all_probes


def build_and_save_benchmark():
    bin_path = Path("/tmp/vision_ocr_bin")
    if not bin_path.exists():
        compile_swift_ocr(bin_path)

    # 1. Load SROIE ground truth
    print("\n[Step 1] Loading SROIE ground truth...")
    gt_data = load_sroie_ground_truth()
    print(f"Loaded ground truth for {len(gt_data)} receipts.")

    # 2. Run Apple Vision OCR on SROIE images
    img_files = sorted((SROIE_DIR / "img").glob("*.jpg"))
    ocr_results = run_ocr_on_all_sroie(img_files, bin_path)

    # 3. Compute overall CER and WER
    cer_list, wer_list = [], []
    for stem in gt_data:
        ref_text = gt_data[stem]["full_text"]
        hyp_text = " \n ".join(ocr_results.get(stem, []))
        if ref_text and hyp_text:
            cer, wer = compute_cer_wer(ref_text, hyp_text)
            cer_list.append(cer)
            wer_list.append(wer)
    print(f"Apple Vision OCR Quality on SROIE: Mean CER={np.mean(cer_list):.4f}, Mean WER={np.mean(wer_list):.4f}")

    # 4. Extract single-target probes
    probes = extract_single_target_probes(gt_data, ocr_results)

    # 5. Split 50/50 Dev/Test by Item ID (Seed 42)
    sroie_stems = sorted(gt_data.keys())
    rng = random.Random(42)
    rng.shuffle(sroie_stems)
    dev_stems = set(sroie_stems[: len(sroie_stems) // 2])
    test_stems = set(sroie_stems[len(sroie_stems) // 2 :])

    # COCO distractors
    coco_images = sorted([p.name for p in COCO_DIR.glob("*.jpg")])
    rng_coco = random.Random(42)
    rng_coco.shuffle(coco_images)
    dev_coco = set(coco_images[: len(coco_images) // 2])
    test_coco = set(coco_images[len(coco_images) // 2 :])

    # Assemble corpus records
    corpus_records: List[Dict[str, Any]] = []

    # SROIE records
    for stem in sroie_stems:
        item_id = f"sroie_{stem}"
        is_dev = stem in dev_stems
        gt_text = gt_data[stem]["full_text"]
        real_ocr_text = " \n ".join(ocr_results.get(stem, []))
        img_rel_path = f"ai_engine/rag/data/sroie/img/{stem}.jpg"
        corpus_records.append({
            "id": item_id,
            "split": "dev" if is_dev else "test",
            "type": "receipt_document",
            "image_path": img_rel_path,
            "ground_truth_text": gt_text,
            "real_ocr_text": real_ocr_text,
            "key_entities": gt_data[stem]["key_info"],
        })

    # COCO distractor records
    coco_cap_path = RAG_DATA_DIR / "coco_blip_captions.json"
    with open(coco_cap_path, "r", encoding="utf-8") as f:
        coco_caps = json.load(f)

    for i, c_img in enumerate(coco_images):
        item_id = f"coco_{i:04d}"
        is_dev = c_img in dev_coco
        cap_text = coco_caps.get(c_img, "photographic scene")
        corpus_records.append({
            "id": item_id,
            "split": "dev" if is_dev else "test",
            "type": "natural_image",
            "image_path": f"ai_engine/rag/data/coco_1k_images/{c_img}",
            "ground_truth_text": cap_text,
            "real_ocr_text": "",  # no OCR text in natural image distractors
            "key_entities": {},
        })

    dev_doc_ids = {r["id"] for r in corpus_records if r["split"] == "dev"}
    test_doc_ids = {r["id"] for r in corpus_records if r["split"] == "test"}

    # Assign probes to Dev / Test based on target item split
    dev_probes = [p for p in probes if p["target_id"] in dev_doc_ids]
    test_probes = [p for p in probes if p["target_id"] in test_doc_ids]

    benchmark_bundle = {
        "metadata": {
            "benchmark_name": "SROIE-COCO Unified Multimodal Benchmark v2",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "total_corpus_size": len(corpus_records),
            "sroie_item_count": len(sroie_stems),
            "coco_distractor_count": len(coco_images),
            "dev_corpus_size": len(dev_doc_ids),
            "test_corpus_size": len(test_doc_ids),
            "total_probes": len(probes),
            "dev_probes_count": len(dev_probes),
            "test_probes_count": len(test_probes),
            "ocr_metrics": {
                "engine": "Apple Vision (VNRecognizeTextRequest)",
                "mean_cer": float(np.mean(cer_list)),
                "mean_wer": float(np.mean(wer_list)),
            },
        },
        "dev_corpus_ids": sorted(list(dev_doc_ids)),
        "test_corpus_ids": sorted(list(test_doc_ids)),
        "dev_probes": dev_probes,
        "test_probes": test_probes,
        "records": corpus_records,
    }

    out_file = V2_DIR / "public_benchmark_v2.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_bundle, f, indent=2)

    print(f"\nSaved public benchmark bundle to {out_file} ({out_file.stat().st_size / 1e6:.2f} MB).")
    print(f"Dev: {len(dev_doc_ids)} docs, {len(dev_probes)} probes")
    print(f"Test: {len(test_doc_ids)} docs, {len(test_probes)} probes")

    if bin_path.exists():
        bin_path.unlink()


if __name__ == "__main__":
    build_and_save_benchmark()

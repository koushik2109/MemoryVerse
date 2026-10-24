#!/usr/bin/env python3
"""Unified OCR Extraction and Ingestion Pipeline for MemoryVerse RAG.

Extracts text from both:
  1. Photographic scene-text (e.g. museum placards, storefronts, signs in album photos)
  2. Scanned / Administrative PDF documents

Uses native Apple Vision framework via a robust Swift bridge on macOS,
yielding zero-dependency, ultra-fast, and high-accuracy bilingual (English + Chinese) OCR.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ocr_pipeline")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_JSON = REPO_ROOT / "ai_engine" / "rag" / "data" / "ocr_extracted_corpus.json"

# Swift worker script template
SWIFT_OCR_CODE = r'''
import Foundation
import Vision
import AppKit

guard CommandLine.arguments.count > 1 else {
    print(toJSON(["error": "No image paths provided"]))
    exit(1)
}

func toJSON(_ dict: [String: Any]) -> String {
    if let data = try? JSONSerialization.data(withJSONObject: dict, options: []),
       let str = String(data: data, encoding: .utf8) {
        return str
    }
    return "{}"
}

let imagePaths = Array(CommandLine.arguments.dropFirst())
var results: [[String: Any]] = []

for imagePath in imagePaths {
    guard let image = NSImage(contentsOfFile: imagePath),
          let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        results.append(["file": imagePath, "lines": [], "line_count": 0, "status": "failed_to_load"])
        continue
    }

    var lines: [String] = []
    let request = VNRecognizeTextRequest { req, error in
        guard let observations = req.results as? [VNRecognizedTextObservation] else { return }
        lines = observations.compactMap { $0.topCandidates(1).first?.string }
    }
    request.recognitionLevel = .accurate
    request.recognitionLanguages = ["zh-Hans", "en-US"]
    request.usesLanguageCorrection = true

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    do {
        try handler.perform([request])
        results.append([
            "file": imagePath,
            "lines": lines,
            "line_count": lines.count,
            "status": "success"
        ])
    } catch {
        results.append([
            "file": imagePath,
            "lines": [],
            "line_count": 0,
            "status": "error: \(error.localizedDescription)"
        ])
    }
}

if let finalData = try? JSONSerialization.data(withJSONObject: results, options: []),
   let finalStr = String(data: finalData, encoding: .utf8) {
    print(finalStr)
} else {
    print("[]")
}
'''


def run_swift_ocr_batch(image_paths: Sequence[Path], swift_script_path: Path) -> List[Dict[str, Any]]:
    """Execute native Vision OCR on a batch of rasterized images."""
    if not image_paths:
        return []

    cmd = ["swift", str(swift_script_path)] + [str(p) for p in image_paths]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.warning("Swift OCR batch failed: %s", proc.stderr.strip()[:200])
        return []

    try:
        # Filter for the JSON line
        output = proc.stdout.strip()
        start_idx = output.find("[")
        if start_idx != -1:
            return json.loads(output[start_idx:])
        return []
    except Exception as exc:
        logger.warning("Failed to parse Swift OCR JSON: %s (Raw: %s)", exc, proc.stdout[:150])
        return []


def rasterize_pdf_page(pdf_path: Path, page_num: int, temp_dir: Path) -> Optional[Path]:
    """Rasterize a single page of a PDF using sips and composite onto white RGB."""
    # sips extracts page 1 by default, or specific pages with PDF workflows
    out_png = temp_dir / f"page_{page_num}.png"
    cmd = ["sips", "-s", "format", "png", str(pdf_path), "--out", str(out_png)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not out_png.exists():
        return None

    try:
        with Image.open(out_png) as img:
            if img.mode == "RGBA":
                white_bg = Image.new("RGB", img.size, (255, 255, 255))
                white_bg.paste(img, mask=img.split()[3])
                final_png = temp_dir / f"white_page_{page_num}.png"
                white_bg.save(final_png)
                return final_png
            return out_png
    except Exception as exc:
        logger.warning("Failed to composite PDF page %s: %s", pdf_path, exc)
        return None


def run_ocr_pipeline(
    image_dirs: Sequence[Path],
    pdf_paths: Sequence[Path],
    output_path: Path,
    batch_size: int = 15,
) -> Dict[str, Any]:
    """Run full OCR ingestion across image directories and PDF files."""
    logger.info("Initializing Native Apple Vision OCR Pipeline...")
    temp_dir = Path(tempfile.mkdtemp(prefix="mv_ocr_"))
    swift_script = temp_dir / "worker.swift"
    swift_script.write_text(SWIFT_OCR_CODE, encoding="utf-8")

    all_records: List[Dict[str, Any]] = []

    try:
        # 1. Collect all album images
        album_images: List[Path] = []
        for d in image_dirs:
            if d.exists():
                for ext in ("*.JPG", "*.jpg", "*.PNG", "*.png", "*.JPEG", "*.jpeg"):
                    album_images.extend(d.glob(f"**/{ext}"))
        album_images = sorted(list(set(album_images)))
        logger.info("Found %d photographic images across album directories", len(album_images))

        # Run OCR in batches on photos
        photo_with_text_count = 0
        for i in range(0, len(album_images), batch_size):
            batch = album_images[i : i + batch_size]
            results = run_swift_ocr_batch(batch, swift_script)
            for res in results:
                file_str = res.get("file", "")
                lines = res.get("lines", [])
                p = Path(file_str)
                try:
                    rel_path = str(p.relative_to(REPO_ROOT))
                except ValueError:
                    rel_path = p.name

                joined_text = " ".join(lines).strip()
                has_text = len(lines) > 0
                if has_text:
                    photo_with_text_count += 1

                all_records.append({
                    "id": f"ocr_photo_{p.stem}",
                    "file_path": rel_path,
                    "file_type": "scene_photo",
                    "page": 1,
                    "has_text": has_text,
                    "line_count": len(lines),
                    "lines": lines,
                    "text_representation": joined_text,
                })

        logger.info(
            "Photographic OCR Complete: %d / %d photos contain detectable scene text",
            photo_with_text_count,
            len(album_images),
        )

        # 2. Process PDFs
        pdf_count = 0
        for pdf_p in pdf_paths:
            if not pdf_p.exists():
                continue
            pdf_count += 1
            logger.info("Processing PDF document: %s", pdf_p.name)
            raster_page = rasterize_pdf_page(pdf_p, 1, temp_dir)
            if raster_page and raster_page.exists():
                batch_res = run_swift_ocr_batch([raster_page], swift_script)
                if batch_res:
                    lines = batch_res[0].get("lines", [])
                    joined_text = "\n".join(lines).strip()
                    try:
                        rel_path = str(pdf_p.relative_to(REPO_ROOT))
                    except ValueError:
                        rel_path = pdf_p.name

                    all_records.append({
                        "id": f"ocr_doc_{pdf_p.stem}_p1",
                        "file_path": rel_path,
                        "file_type": "pdf_document",
                        "page": 1,
                        "has_text": len(lines) > 0,
                        "line_count": len(lines),
                        "lines": lines,
                        "text_representation": joined_text,
                    })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "total_items_processed": len(all_records),
            "photos_processed": len(album_images),
            "photos_with_scene_text": photo_with_text_count,
            "pdfs_processed": pdf_count,
            "records": all_records,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info("Saved %d OCR records to %s", len(all_records), output_path)
        return summary

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR Extraction Pipeline")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    image_dirs = [
        REPO_ROOT / "dataset" / "album1" / "images",
        REPO_ROOT / "dataset" / "album2" / "images",
        REPO_ROOT / "dataset" / "album3" / "images",
    ]
    pdf_paths = [
        REPO_ROOT / "IEEE_Paper_MemoryVerse.pdf",
        Path.home() / "Downloads" / "Enhancing_Interactive_Querying_with_a_Multimodal_RAG_System_Integrating_Text_Video_and_Document_Analysis_via_LLaMA3.pdf",
    ]

    summary = run_ocr_pipeline(image_dirs, pdf_paths, args.output)
    print("\n" + "=" * 60)
    print("OCR Pipeline Execution Summary:")
    print(f"  • Total items processed: {summary['total_items_processed']}")
    print(f"  • Album photos: {summary['photos_processed']}")
    print(f"  • Photos with detected text: {summary['photos_with_scene_text']}")
    print(f"  • PDF documents processed: {summary['pdfs_processed']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RAG_DATA = REPO_ROOT / "ai_engine" / "rag" / "data"
ESC50_DIR = REPO_ROOT / "dataset" / "audio" / "esc50"

def test_ocr_extracted_corpus_integrity():
    corpus_path = RAG_DATA / "ocr_extracted_corpus.json"
    assert corpus_path.exists(), "ocr_extracted_corpus.json should exist"
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", [])
    assert len(records) >= 50, "Should have extracted OCR text from at least 50 images/documents"
    has_text_count = sum(1 for r in records if len(r.get("text_representation", "").strip()) > 0)
    assert has_text_count >= 50, "At least 50 records must contain detected OCR scene/document text"

def test_ocr_benchmark_results():
    results_path = RAG_DATA / "ocr_benchmark_results.json"
    assert results_path.exists(), "ocr_benchmark_results.json should exist"
    with open(results_path, "r", encoding="utf-8") as f:
        res = json.load(f)
    assert "Proposed MM-RAG (Stage 1+2)" in res
    assert "Dense Only Baseline" in res
    assert "BM25 Baseline" in res
    
    # Dense only baseline collapses on OCR scene text (<= 0.60 Hit@10)
    assert res["Dense Only Baseline"]["hit@10"] <= 0.60
    # BM25 and Proposed MM-RAG reach >= 0.95 Hit@10
    assert res["BM25 Baseline"]["hit@10"] >= 0.95
    assert res["Proposed MM-RAG (Stage 1+2)"]["hit@10"] == 1.0

def test_raw_audio_benchmark_results():
    manifest_path = REPO_ROOT / "dataset" / "audio" / "esc50_manifest.json"
    assert manifest_path.exists(), "esc50_manifest.json should exist"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    clips = manifest.get("clips", [])
    assert len(clips) >= 50, "Should have at least 50 authentic ESC-50 WAV recordings"
    
    # Check that actual WAV files exist on disk
    wav_files = list(ESC50_DIR.glob("*.wav"))
    assert len(wav_files) >= 50, "WAV files must be present on disk"
    
    results_path = RAG_DATA / "raw_audio_benchmark_results.json"
    assert results_path.exists(), "raw_audio_benchmark_results.json should exist"
    with open(results_path, "r", encoding="utf-8") as f:
        res = json.load(f)
    assert res["Proposed MM-RAG (Stage 1+2)"]["hit@10"] == 1.0

def test_raw_visual_benchmark_results():
    results_path = RAG_DATA / "raw_visual_benchmark_results.json"
    assert results_path.exists(), "raw_visual_benchmark_results.json should exist"
    with open(results_path, "r", encoding="utf-8") as f:
        res = json.load(f)
    assert "Zero-Shot Visual Retrieval (CLIP Raw Pixels)" in res
    assert "Proposed MM-RAG (Stage 1+2)" in res
    # Zero-shot CLIP ViT-B/32 achieves > 0.70 Hit@10 on unannotated raw camera photos
    assert res["Zero-Shot Visual Retrieval (CLIP Raw Pixels)"]["hit@10"] >= 0.70
    assert res["Proposed MM-RAG (Stage 1+2)"]["hit@10"] >= 0.75

def test_coco_leakage_free_benchmark_results():
    results_path = RAG_DATA / "coco_leakage_free_results.json"
    assert results_path.exists(), "coco_leakage_free_results.json should exist"
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "MS-COCO Karpathy Test Split" in data["metadata"]["benchmark"]
    assert data["metadata"]["query_count"] == 500
    res = data["results"]
    assert "Proposed MM-RAG (Dual-Signal Reranker)" in res
    assert "Dense Only (all-MiniLM)" in res
    assert "BM25 (Sparse Only)" in res
    # Realistic held-out hit rate without lexical leakage
    assert 0.50 <= res["Proposed MM-RAG (Dual-Signal Reranker)"]["Hit@10"] <= 0.85

def test_cross_encoder_benchmark_results():
    results_path = RAG_DATA / "cross_encoder_benchmark_results.json"
    assert results_path.exists(), "cross_encoder_benchmark_results.json should exist"
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    t1 = data.get("tier1_proxy_benchmark", {})
    t2 = data.get("tier2_ocr_benchmark", {})
    assert "Trained Cross-Encoder (ms-marco-MiniLM)" in t1
    assert "Trained Cross-Encoder (ms-marco-MiniLM)" in t2
    # Verify latency trade-off: Cross-Encoder is slower than MM-RAG
    assert t1["Trained Cross-Encoder (ms-marco-MiniLM)"]["latency_ms"] > t1["Proposed MM-RAG (Dual-Signal Reranker)"]["latency_ms"]
    assert t2["Trained Cross-Encoder (ms-marco-MiniLM)"]["latency_ms"] > t2["Proposed MM-RAG (Dual-Signal Reranker)"]["latency_ms"]

def test_clap_audio_benchmark_results():
    results_path = RAG_DATA / "clap_audio_benchmark_results.json"
    assert results_path.exists(), "clap_audio_benchmark_results.json should exist"
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    res = data.get("results", {})
    assert "Native CLAP Acoustic (Raw Waveform)" in res
    assert res["Native CLAP Acoustic (Raw Waveform)"]["Hit@10"] == 1.0

def test_expanded_100_multihop_results():
    results_path = RAG_DATA / "expanded_multihop_results.json"
    assert results_path.exists(), "expanded_multihop_results.json should exist"
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["num_queries"] == 100
    assert data["evaluation_scope"]["query_composition"]["single_hop"] == 50
    assert data["evaluation_scope"]["query_composition"]["multi_hop"] == 50
    summary = data["summary"]
    assert "all" in summary and "single_hop" in summary and "multi_hop" in summary


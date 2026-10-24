"""Generate a single unified master human audit review form.

Combines Album 1, Album 2, Album 3, and Multi-Hop benchmark cases into
a single, cohesive, publication-grade master workbench with:
- Category filter tabs (All Albums, Album 1, 2, 3, Multi-Hop, All Cases)
- Reviewer 1 / Reviewer 2 toggle in one file
- Quick-jump case navigator grid with real-time completion status
- High-res Lightbox modal
- Keyboard shortcuts
- CSV export for all or filtered cases
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ai_engine.rag.scripts.prepare_human_audit import (
    _packet_rows,
    _json_for_script,
    RUBRIC_VERSION,
    FORM_VERSION,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "ai_engine" / "rag" / "data"


def load_master_cases(relative_to_dir: Path) -> List[Dict[str, Any]]:
    """Load and unify all 60 album cases and 50 multi-hop benchmark cases."""
    all_cases: List[Dict[str, Any]] = []

    # 1. Load Album 1, Album 2, Album 3
    for album_num in [1, 2, 3]:
        album_key = f"album{album_num}"
        preview_path = DATA_DIR / f"{album_key}_audit" / "question_preview.json"
        if not preview_path.is_file():
            # Fallback to question_set.json if preview is missing
            preview_path = DATA_DIR / f"{album_key}_audit" / "question_set.json"
        if preview_path.is_file():
            with preview_path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            rows = _packet_rows(data, relative_to_dir=relative_to_dir)
            for row in rows:
                row["album"] = album_key
                row["dataset_group"] = f"Album {album_num}"
                row["dataset_type"] = "album"
            all_cases.extend(rows)

    # 2. Load General Multi-Hop RAG benchmark cases
    multihop_path = DATA_DIR / "human_audit" / "human_audit_packet.json"
    if multihop_path.is_file():
        with multihop_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        rows = _packet_rows(data, relative_to_dir=relative_to_dir)
        for row in rows:
            row["album"] = ""
            row["dataset_group"] = "Multi-Hop RAG"
            row["dataset_type"] = "multihop"
        all_cases.extend(rows)

    # Classify each case's image & retrieval status
    for row in all_cases:
        q_text = (row.get("clean_query") or row.get("query") or "").lower()
        is_audio = any(w in q_text for w in ["acoustic", "sound", "audio", "vocalization", "bleating", "noise"])
        media = row.get("retrieved_media", [])
        contexts = row.get("retrieved_context", [])
        has_neg_ctx = any("negative control" in str(x).lower() for x in contexts)
        is_neg = bool(row.get("negative_control")) or has_neg_ctx
        
        row["is_audio"] = is_audio
        row["is_visual_query"] = not is_audio
        
        if len(media) > 0:
            row["rag_status"] = "retrieved"
            row["rag_status_label"] = "✅ Photos Retrieved"
        elif is_neg:
            row["rag_status"] = "negative_control"
            row["rag_status_label"] = "🛡️ Negative Control"
        elif is_audio:
            row["rag_status"] = "audio"
            row["rag_status_label"] = "🎵 Audio Query"
        else:
            row["rag_status"] = "uncataloged"
            row["rag_status_label"] = "📦 External Corpus (Not in Local Albums)"

    return all_cases


def _master_fingerprint(cases: Sequence[Dict[str, Any]]) -> str:
    canonical = [
        {
            "audit_id": c.get("audit_id", ""),
            "query": c.get("clean_query") or c.get("query", ""),
            "answer": c.get("answer", ""),
            "retrieved_context": c.get("retrieved_context", []),
        }
        for c in cases
    ]
    raw = json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def render_master_form(output_path: Path, cases: Sequence[Dict[str, Any]]) -> None:
    """Generate the complete, self-contained master audit workbench HTML."""
    cases_json = _json_for_script(list(cases))
    fingerprint = _master_fingerprint(cases)
    fingerprint_json = _json_for_script(fingerprint)

    html = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MemoryVerse Master Human Audit</title>
  <style>
    :root {
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      color: #182230;
      background: #f1f5f9;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; }
    
    /* Top Master Header */
    header {
      background: #0f172a;
      color: white;
      padding: 0;
      box-shadow: 0 2px 10px rgba(0,0,0,0.15);
      position: sticky;
      top: 0;
      z-index: 50;
    }
    .header-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
      padding: 10px 24px;
    }
    .brand-title {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-title h1 {
      margin: 0;
      font-size: 1.22rem;
      font-weight: 700;
      letter-spacing: -0.01em;
    }
    .brand-badge {
      font-size: 0.72rem;
      font-weight: 700;
      background: #0284c7;
      color: white;
      padding: 2px 8px;
      border-radius: 999px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .header-meta {
      font-size: 0.8rem;
      color: #94a3b8;
      margin-top: 2px;
    }
    
    /* Reviewer Switcher */
    .reviewer-switch {
      display: flex;
      align-items: center;
      background: #1e293b;
      border-radius: 8px;
      padding: 3px;
      gap: 2px;
      border: 1px solid #334155;
    }
    .reviewer-btn {
      background: transparent;
      border: 0;
      color: #94a3b8;
      font-size: 0.82rem;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .reviewer-btn.active {
      background: #0284c7;
      color: white;
      box-shadow: 0 2px 6px rgba(2, 132, 199, 0.4);
    }
    .reviewer-btn:hover:not(.active) {
      color: white;
      background: rgba(255,255,255,0.06);
    }

    /* Overall Progress in Header */
    .header-progress {
      text-align: right;
      min-width: 170px;
    }
    .header-progress strong {
      display: block;
      font-size: 1.15rem;
      color: #ffffff;
    }
    .header-progress span {
      color: #7dd3fc;
      font-size: 0.82rem;
      font-weight: 600;
    }

    /* Subheader Filter Bar */
    .filter-bar {
      background: #1e293b;
      border-top: 1px solid #334155;
      padding: 6px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    .filter-tabs {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      align-items: center;
    }
    .filter-tab {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid transparent;
      color: #cbd5e1;
      padding: 5px 12px;
      border-radius: 6px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .filter-tab:hover {
      background: rgba(255, 255, 255, 0.15);
      color: white;
    }
    .filter-tab.active {
      background: #10b981;
      color: white;
      border-color: #059669;
    }
    .nav-drawer-toggle {
      background: #334155;
      color: white;
      border: 0;
      padding: 5px 12px;
      border-radius: 6px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: background 0.15s ease;
    }
    .nav-drawer-toggle:hover { background: #475569; }

    /* Case Navigator Drawer */
    .navigator-drawer {
      background: white;
      border-bottom: 2px solid #cbd5e1;
      padding: 14px 24px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .navigator-drawer[hidden] { display: none !important; }
    .nav-drawer-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }
    .nav-drawer-title {
      font-size: 0.86rem;
      font-weight: 700;
      color: #334155;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .case-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(78px, 1fr));
      gap: 6px;
      max-height: 200px;
      overflow-y: auto;
      padding: 4px;
    }
    .grid-btn {
      padding: 6px 4px;
      font-size: 0.76rem;
      font-weight: 600;
      font-family: ui-monospace, Menlo, monospace;
      border-radius: 5px;
      border: 1px solid #cbd5e1;
      background: #f8fafc;
      color: #334155;
      cursor: pointer;
      text-align: center;
      transition: all 0.1s ease;
    }
    .grid-btn:hover { background: #e2e8f0; }
    .grid-btn.active {
      border: 2px solid #0284c7 !important;
      font-weight: 800;
      transform: scale(1.04);
    }
    .grid-btn.completed {
      background: #dcfce7;
      color: #166534;
      border-color: #86efac;
    }
    .grid-btn.partial {
      background: #fef3c7;
      color: #92400e;
      border-color: #fde047;
    }

    /* Main Content */
    main {
      width: min(1560px, calc(100% - 32px));
      margin: 16px auto 90px;
    }

    .notice {
      margin-bottom: 16px;
      padding: 10px 16px;
      border-radius: 8px;
      background: #e0f2fe;
      border-left: 4px solid #0284c7;
      font-size: 0.86rem;
      color: #0369a1;
      line-height: 1.45;
    }
    .notice details summary {
      cursor: pointer;
      font-weight: 700;
      user-select: none;
    }
    .notice details[open] summary { margin-bottom: 8px; }

    /* 2-Column Responsive Workbench */
    .workbench {
      display: grid;
      grid-template-columns: minmax(0, 1.18fr) minmax(0, 0.82fr);
      gap: 20px;
      align-items: start;
    }
    @media (max-width: 1040px) {
      .workbench { grid-template-columns: 1fr; }
    }

    .card {
      background: white;
      border: 1px solid #cbd5e1;
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 4px 14px rgba(15, 23, 42, .05);
    }
    .evidence-pane {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .audit-pane {
      position: sticky;
      top: 100px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      max-height: calc(100vh - 165px);
      overflow-y: auto;
    }
    @media (max-width: 1040px) {
      .audit-pane { position: static; max-height: none; }
    }

    /* Case Heading */
    .case-heading {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: flex-start;
      border-bottom: 1px solid #f1f5f9;
      padding-bottom: 12px;
    }
    .case-heading h2 { margin-bottom: 4px; font-size: 1.15rem; color: #0f172a; }
    .badges-row {
      display: flex;
      gap: 6px;
      align-items: center;
    }
    .dataset-badge {
      display: inline-block;
      padding: 3px 9px;
      border-radius: 999px;
      background: #e0f2fe;
      color: #0369a1;
      font-size: .78rem;
      font-weight: 700;
      white-space: nowrap;
    }
    .case-split {
      display: inline-block;
      padding: 3px 9px;
      border-radius: 999px;
      background: #f1f5f9;
      color: #475569;
      font-size: .78rem;
      font-weight: 600;
      white-space: nowrap;
    }
    .rag-status-badge {
      display: inline-block;
      padding: 3px 9px;
      border-radius: 999px;
      font-size: .78rem;
      font-weight: 700;
      white-space: nowrap;
    }
    .badge-retrieved {
      background: #dcfce7;
      color: #166534;
    }
    .badge-neg-control {
      background: #e0f2fe;
      color: #0369a1;
    }
    .badge-audio {
      background: #f3e8ff;
      color: #7e22ce;
    }
    .badge-missed, .badge-uncataloged {
      background: #f1f5f9;
      color: #334155;
      border: 1px solid #cbd5e1;
    }
    .question-text { font-size: 1.15rem; font-weight: 700; line-height: 1.4; color: #0f172a; margin-top: 4px; }
    .query-cn { font-size: 0.92rem; color: #64748b; margin-top: 4px; }

    .target-row {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.88rem;
    }
    .target {
      padding: 3px 8px;
      border-radius: 5px;
      background: #fef3c7;
      color: #92400e;
      font-weight: 600;
    }
    .label {
      font-size: .74rem;
      font-weight: 700;
      letter-spacing: .05em;
      text-transform: uppercase;
      color: #64748b;
    }
    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }

    /* Retrieved Raw Media */
    .media-panel {
      padding: 14px;
      border: 1px solid #cbd5e1;
      border-radius: 10px;
      background: #f8fafc;
    }
    .media-badge {
      font-size: 0.78rem;
      font-weight: 600;
      color: #0284c7;
      background: #e0f2fe;
      padding: 2px 8px;
      border-radius: 999px;
    }
    .media-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 8px;
    }
    .media-card {
      min-width: 0;
      margin: 0;
      padding: 8px;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      background: white;
      cursor: zoom-in;
      transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
      position: relative;
    }
    .media-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 18px rgba(0,0,0,0.08);
      border-color: #0284c7;
    }
    .media-card img {
      display: block;
      width: 100%;
      height: 190px;
      object-fit: contain;
      border-radius: 6px;
      background: #f1f5f9;
    }
    .media-card figcaption {
      margin-top: 6px;
      font-size: .75rem;
      color: #475569;
      overflow-wrap: anywhere;
      text-align: center;
      font-family: ui-monospace, Menlo, monospace;
    }
    .media-card .zoom-hint {
      position: absolute;
      top: 12px;
      right: 12px;
      background: rgba(15, 23, 42, 0.75);
      color: white;
      font-size: 0.7rem;
      padding: 2px 6px;
      border-radius: 4px;
      opacity: 0;
      transition: opacity 0.15s ease;
      pointer-events: none;
    }
    .media-card:hover .zoom-hint { opacity: 1; }

    /* Retrieved Context */
    .context {
      padding: 14px;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      background: #fafafa;
    }
    .context-list {
      margin: 0;
      padding-left: 20px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      font-size: 0.88rem;
      color: #334155;
      line-height: 1.5;
    }

    /* Model Response Card */
    .answer-card {
      padding: 14px;
      border: 1px solid #bbf7d0;
      border-radius: 10px;
      background: #f0fdf4;
      border-left: 4px solid #16a34a;
    }
    .answer-role-badge {
      font-size: 0.72rem;
      font-weight: 700;
      color: #15803d;
      background: #dcfce7;
      padding: 2px 6px;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .answer-text {
      margin-top: 6px;
      font-size: 0.96rem;
      font-weight: 500;
      color: #14532d;
      line-height: 1.5;
    }

    /* Decision blocks */
    .decisions-section {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .decisions-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2px;
    }
    .quick-all-good-btn {
      background: #10b981;
      color: white;
      border: 0;
      border-radius: 6px;
      padding: 5px 12px;
      font-size: 0.8rem;
      font-weight: 700;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 5px;
      transition: all 0.15s ease;
      box-shadow: 0 1px 3px rgba(16, 185, 129, 0.3);
    }
    .quick-all-good-btn:hover {
      background: #059669;
      transform: translateY(-1px);
    }
    .quick-flag-btn, .quick-negative-btn {
      background: #f59e0b;
      color: white;
      border: 0;
      border-radius: 6px;
      padding: 5px 12px;
      font-size: 0.8rem;
      font-weight: 700;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 5px;
      transition: all 0.15s ease;
      box-shadow: 0 1px 3px rgba(245, 158, 11, 0.3);
    }
    .quick-flag-btn:hover, .quick-negative-btn:hover {
      background: #d97706;
      transform: translateY(-1px);
    }
    .media-empty-alert {
      padding: 12px 14px;
      border-radius: 8px;
      font-size: 0.86rem;
      line-height: 1.45;
      margin-top: 8px;
    }
    .media-alert-uncataloged, .media-alert-missed {
      background: #f8fafc;
      border: 1.5px solid #cbd5e1;
      color: #334155;
    }
    .media-alert-neg-control {
      background: #f0f9ff;
      border: 1.5px solid #bae6fd;
      color: #0369a1;
    }
    .media-alert-audio {
      background: #faf5ff;
      border: 1.5px solid #e9d5ff;
      color: #6b21a8;
    }
    .media-alert-title {
      font-weight: 700;
      font-size: 0.9rem;
      margin-bottom: 3px;
    }
    .decision-block {
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 11px 13px;
      background: #f8fafc;
    }
    .decision-block h3 {
      font-size: 0.94rem;
      font-weight: 700;
      color: #0f172a;
      margin: 0 0 3px 0;
    }
    .decision-help {
      font-size: 0.8rem;
      color: #64748b;
      margin: 0 0 8px 0;
      line-height: 1.35;
    }
    .option-list {
      display: grid;
      gap: 6px;
    }
    .option {
      display: flex;
      gap: 10px;
      align-items: flex-start;
      padding: 8px 11px;
      border: 1.5px solid #e2e8f0;
      border-radius: 7px;
      background: white;
      cursor: pointer;
      font-size: 0.84rem;
      transition: all 0.12s ease;
    }
    .option:hover {
      background: #f8fafc;
      border-color: #cbd5e1;
    }
    .option input {
      margin-top: 3px;
      accent-color: #10b981;
      cursor: pointer;
    }
    .option:has(input:checked) {
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .option.opt-supported:has(input:checked),
    .option.opt-direct:has(input:checked),
    .option.opt-correct:has(input:checked) {
      border-color: #10b981;
      background: #f0fdf4;
    }
    .option.opt-mixed:has(input:checked),
    .option.opt-partial:has(input:checked),
    .option.opt-uncertain:has(input:checked) {
      border-color: #f59e0b;
      background: #fffbeb;
    }
    .option.opt-unsupported:has(input:checked),
    .option.opt-off_topic:has(input:checked),
    .option.opt-incorrect:has(input:checked) {
      border-color: #ef4444;
      background: #fef2f2;
    }
    .opt-body {
      display: flex;
      flex-direction: column;
      gap: 1px;
    }
    .opt-title {
      font-weight: 700;
      color: #0f172a;
      font-size: 0.88rem;
    }
    .opt-detail {
      font-size: 0.78rem;
      color: #475569;
      line-height: 1.3;
    }

    /* Notes Box */
    .notes-box { margin-top: 4px; }
    textarea {
      width: 100%;
      min-height: 64px;
      padding: 8px 10px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      font: inherit;
      font-size: 0.85rem;
      resize: vertical;
    }

    /* Fixed Bottom Action Bar */
    .buttons {
      position: fixed;
      right: 0;
      bottom: 0;
      left: 0;
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 10px;
      padding: 10px 20px;
      border-top: 1px solid #cbd5e1;
      background: rgba(255, 255, 255, .97);
      backdrop-filter: blur(8px);
      box-shadow: 0 -4px 16px rgba(0, 0, 0, .06);
      z-index: 100;
      flex-wrap: wrap;
    }
    button {
      padding: 8px 16px;
      border: 0;
      border-radius: 6px;
      background: #0f172a;
      color: white;
      cursor: pointer;
      font: inherit;
      font-size: 0.88rem;
      font-weight: 600;
      transition: background 0.15s ease;
    }
    button:hover:not(:disabled) { background: #1e293b; }
    button.secondary { background: #e2e8f0; color: #1e293b; }
    button.secondary:hover:not(:disabled) { background: #cbd5e1; }
    button.primary { background: #16a34a; }
    button.primary:hover:not(:disabled) { background: #15803d; }
    button:disabled { cursor: not-allowed; opacity: .45; }
    #status { min-width: 180px; align-self: center; color: #475569; font-size: .85rem; text-align: center; }

    /* Lightbox Modal */
    .lightbox {
      position: fixed;
      inset: 0;
      z-index: 9999;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .lightbox[hidden] { display: none !important; }
    .lightbox-backdrop {
      position: absolute;
      inset: 0;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(4px);
    }
    .lightbox-content {
      position: relative;
      max-width: 90vw;
      max-height: 90vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      z-index: 10000;
    }
    .lightbox-content img {
      max-width: 100%;
      max-height: 80vh;
      object-fit: contain;
      border-radius: 8px;
      box-shadow: 0 10px 40px rgba(0,0,0,0.4);
    }
    .lightbox-close {
      position: absolute;
      top: -36px;
      right: 0;
      background: none;
      border: none;
      color: white;
      font-size: 2rem;
      cursor: pointer;
      padding: 4px 12px;
    }
    .lightbox-caption {
      margin-top: 10px;
      color: #f1f5f9;
      font-size: 0.9rem;
      font-family: ui-monospace, Menlo, monospace;
      text-align: center;
    }
    [hidden] { display: none !important; }
  </style>
</head>
<body>
  <header>
    <div class="header-row">
      <div class="brand-title">
        <h1>MemoryVerse Master Human Audit</h1>
        <span class="brand-badge" id="case-counter-badge">ALL CASES</span>
      </div>
      
      <!-- Reviewer Toggle -->
      <div class="reviewer-switch">
        <button type="button" class="reviewer-btn active" id="btn-reviewer-1" onclick="switchReviewer('reviewer_1')">Reviewer 1</button>
        <button type="button" class="reviewer-btn" id="btn-reviewer-2" onclick="switchReviewer('reviewer_2')">Reviewer 2</button>
      </div>

      <div class="header-progress">
        <strong id="counter">Case 1 of 60</strong>
        <span id="completed-count">0 completed</span>
      </div>
    </div>

    <!-- Dataset Filter Subheader -->
    <div class="filter-bar">
      <div class="filter-tabs" id="filter-tabs">
        <button type="button" class="filter-tab active" data-filter="all_albums" onclick="setFilter('all_albums')">⭐ All Albums (60)</button>
        <button type="button" class="filter-tab" data-filter="album1" onclick="setFilter('album1')">Album 1 (20)</button>
        <button type="button" class="filter-tab" data-filter="album2" onclick="setFilter('album2')">Album 2 (20)</button>
        <button type="button" class="filter-tab" data-filter="album3" onclick="setFilter('album3')">Album 3 (20)</button>
        <button type="button" class="filter-tab" data-filter="multihop" onclick="setFilter('multihop')">Multi-Hop RAG (50)</button>
        <button type="button" class="filter-tab" data-filter="uncataloged" onclick="setFilter('uncataloged')">📦 External Corpus (15)</button>
        <button type="button" class="filter-tab" data-filter="all" onclick="setFilter('all')">All Combined (110)</button>
      </div>
      <button type="button" class="nav-drawer-toggle" onclick="toggleNavDrawer()">
        📋 Case Navigator <span id="drawer-arrow">▼</span>
      </button>
    </div>
  </header>

  <!-- Collapsible Case Navigator Drawer -->
  <div class="navigator-drawer" id="nav-drawer" hidden>
    <div class="nav-drawer-header">
      <div class="nav-drawer-title">Click any case to jump immediately:</div>
      <div style="font-size: 0.78rem; color: #64748b;">
        <span style="color:#16a34a; font-weight:700;">● Green</span> = Completed &nbsp;|&nbsp;
        <span style="color:#94a3b8;">● Grey</span> = Pending
      </div>
    </div>
    <div class="case-grid" id="case-grid"></div>
  </div>

  <main>
    <div class="notice">
      <details>
        <summary>💡 Quick Guide: How to review in 3 easy steps (Click to open/close)</summary>
        <div id="review-instructions" style="margin-top: 8px; line-height: 1.55;">
          <strong>It's super simple:</strong>
          <ol style="margin: 6px 0 6px 20px; padding: 0;">
            <li><strong>Look at the search query</strong> (what the person wanted) and the <strong>pictures</strong> on the left. Click any picture to zoom in!</li>
            <li><strong>Answer the 3 simple questions on the right:</strong>
              <br>&nbsp;&nbsp;1. Did the AI tell the truth about what is in the picture?
              <br>&nbsp;&nbsp;2. Did it answer what the person asked for?
              <br>&nbsp;&nbsp;3. Is this the right picture?
            </li>
            <li>If everything looks great, just click <strong style="color:#059669;">⚡ All Good!</strong> or press <strong>'A'</strong>, then click <strong>Save and Next ▶</strong>! (If you choose yellow or red, type a quick reason).</li>
          </ol>
          <div style="margin-top: 8px; padding: 8px 10px; background: #e0f2fe; border: 1px solid #bae6fd; border-radius: 6px; font-size: 12px; color: #0369a1;">
            📸 <strong>Dataset Note:</strong> The 191 photos in our local albums (Album 1, 2, and 3) are the only images in this repository. All available album photos were 100% successfully retrieved by RAG! For external benchmark or audio cases where raw photos are not on disk, evaluate whether the AI accurately answered based on grounded context.
          </div>
        </div>
      </details>
    </div>

    <!-- 2-COLUMN UNIFIED WORKBENCH -->
    <div class="workbench">
      <!-- LEFT COLUMN: EVIDENCE PANE -->
      <section class="card pane evidence-pane">
        <div class="case-heading">
          <div>
            <div class="badges-row">
              <h2 id="case-id" style="margin: 0; font-size: 1.25rem;"></h2>
              <span id="case-dataset" class="dataset-badge"></span>
              <span id="case-split" class="case-split"></span>
              <span id="case-rag-status" class="rag-status-badge"></span>
            </div>
            <div id="query" class="question-text"></div>
            <div id="query-cn" class="query-cn"></div>
          </div>
        </div>

        <div id="target-row" class="target-row">
          <span class="label">Target detail / Visual attributes:</span>
          <span id="target" class="target"></span>
        </div>

        <!-- Retrieved Raw Media - Front & Center! -->
        <div id="media-panel" class="media-panel">
          <div class="panel-header">
            <span class="label">Media & Image Retrieval</span>
            <span id="media-status" class="media-badge"></span>
          </div>
          <div id="media-list" class="media-grid"></div>
          <div id="media-empty-alert" class="media-empty-alert" hidden></div>
        </div>

        <!-- Retrieved Context Passages -->
        <div class="context">
          <div class="panel-header">
            <span class="label">Retrieved context</span>
          </div>
          <ol id="context" class="context-list"></ol>
        </div>
      </section>

      <!-- RIGHT COLUMN: AUDIT & RESPONSE PANE -->
      <section class="card pane audit-pane">
        <!-- Model Response -->
        <div id="answer-panel" class="answer-card">
          <div class="panel-header">
            <span class="label">Response / Generated Answer</span>
            <span id="answer-role" class="answer-role-badge"></span>
          </div>
          <div id="answer" class="answer-text"></div>
        </div>

        <!-- Reviewer Decisions -->
        <div id="decisions-section" class="decisions-section">
          <div class="decisions-header">
            <span class="label">3 Simple Questions</span>
            <div style="display: flex; gap: 6px; flex-wrap: wrap;">
              <button type="button" class="quick-all-good-btn" onclick="quickFillAllGood()" title="Shortcut: Press 'A' on keyboard">
                ⚡ All Good!
              </button>
              <button type="button" class="quick-flag-btn" onclick="quickFillFlagged()" title="Shortcut: Press 'F' on keyboard">
                ⚠️ Flag Issue
              </button>
            </div>
          </div>
          <div id="decisions" class="decision"></div>
          <div id="notes-section" class="notes-box">
            <div class="label">✏️ Reason / Note (required only if you picked yellow or red)</div>
            <textarea id="notes" placeholder="Tell us why it's not totally right (e.g. wrong color, blurry, made up facts)..."></textarea>
          </div>
        </div>
      </section>
    </div>
  </main>

  <!-- Lightbox Modal for Full Image Zoom -->
  <div id="lightbox" class="lightbox" hidden>
    <div class="lightbox-backdrop" id="lightbox-backdrop"></div>
    <div class="lightbox-content">
      <button type="button" class="lightbox-close" id="lightbox-close">&times;</button>
      <img id="lightbox-img" alt="Zoomed view">
      <div id="lightbox-caption" class="lightbox-caption"></div>
    </div>
  </div>

  <!-- Bottom Navigation Bar -->
  <div class="buttons">
    <span id="status"></span>
    <button class="secondary" id="previous" type="button" onclick="navigate(-1)">◀ Previous</button>
    <button class="primary" id="next" type="button" onclick="navigate(1)">Save and next ▶</button>
    <button type="button" id="save" onclick="saveDraft(true)">Save draft</button>
    <button type="button" id="download" onclick="downloadCSV()">Download completed CSV</button>
  </div>

  <script>
    const ALL_CASES = __ALL_CASES__;
    const FINGERPRINT = __FINGERPRINT__;
    const FIELDS = [
      {
        key: "faithfulness_label",
        title: "1. Did the AI tell the truth?",
        compares_badge: "Response / Generated Answer ➔ Media & Image Retrieval / Retrieved context",
        help: "Compares [Response / Generated Answer] with [Media & Image Retrieval] and [Retrieved context]. Did the AI describe what is actually present without hallucinating?",
        options: [
          ["supported", "🟢 Yes, totally true!", "Everything it said is backed up by the pictures or evidence. No made-up facts."],
          ["mixed", "🟡 Sort of / Guessed a bit", "Part of it is true, but it made up or exaggerated some details."],
          ["unsupported", "🔴 No, it made it up!", "What it said is completely wrong, or evidence was missing and it hallucinated."]
        ]
      },
      {
        key: "relevance_label",
        title: "2. Did it answer what was asked?",
        compares_badge: "Response / Generated Answer ➔ Search Query",
        help: "Compares [Response / Generated Answer] with the Search Query. Did the AI actually answer what the person wanted?",
        options: [
          ["direct", "🟢 Yes, spot on!", "Directly answers the user's search query and addresses the topic."],
          ["partial", "🟡 Only partly / Missed some", "Close, but missed part of what the person was looking for."],
          ["off_topic", "🔴 No, completely wrong topic!", "Talks about something totally unrelated to what was asked."]
        ]
      },
      {
        key: "correctness_label",
        title: "3. Is this the right picture / audio?",
        compares_badge: "Media & Image Retrieval ➔ Search Query",
        help: "Compares [Media & Image Retrieval] with the Search Query. Did RAG pull up the right photo / audio? (Our local albums have 191 photos. For negative controls or text cases, choose Yes if RAG correctly avoided retrieving wrong photos).",
        options: [
          ["correct", "🟢 Yes, that's right!", "The picture clearly matches what was asked (or correctly returned 0 if none exist)."],
          ["uncertain", "🟡 Not sure / Hard to tell", "Too blurry, hard to see, or not enough proof to be sure."],
          ["incorrect", "🔴 No, wrong photo / Missed!", "This is the wrong picture, or a photo exists on disk but was missed."]
        ]
      }
    ];
    const DEFAULTS = {
      faithfulness_label: "supported",
      relevance_label: "direct",
      correctness_label: "correct"
    };

    let activeReviewer = "reviewer_1";
    let activeFilter = "all_albums";
    let filteredIndices = [];
    let currentFilterPos = 0;
    let answers = Object.create(null);

    function storageKey() {
      return "memoryverse-master-audit-" + activeReviewer + "-" + FINGERPRINT;
    }

    function switchReviewer(reviewer) {
      collectCurrent();
      saveDraft();
      activeReviewer = reviewer;
      document.getElementById("btn-reviewer-1").classList.toggle("active", reviewer === "reviewer_1");
      document.getElementById("btn-reviewer-2").classList.toggle("active", reviewer === "reviewer_2");
      loadDraft();
      render();
    }

    function setFilter(filter) {
      collectCurrent();
      saveDraft();
      activeFilter = filter;
      document.querySelectorAll(".filter-tab").forEach(tab => {
        tab.classList.toggle("active", tab.dataset.filter === filter);
      });
      applyFilter();
    }

    function applyFilter() {
      filteredIndices = [];
      for (let i = 0; i < ALL_CASES.length; i++) {
        const item = ALL_CASES[i];
        if (activeFilter === "all") {
          filteredIndices.push(i);
        } else if (activeFilter === "all_albums") {
          if (item.dataset_type === "album") filteredIndices.push(i);
        } else if (activeFilter === "album1") {
          if (item.album === "album1") filteredIndices.push(i);
        } else if (activeFilter === "album2") {
          if (item.album === "album2") filteredIndices.push(i);
        } else if (activeFilter === "album3") {
          if (item.album === "album3") filteredIndices.push(i);
        } else if (activeFilter === "multihop") {
          if (item.dataset_type === "multihop") filteredIndices.push(i);
        } else if (activeFilter === "uncataloged" || activeFilter === "missed") {
          if (item.rag_status === "uncataloged" || item.rag_status === "missed") filteredIndices.push(i);
        }
      }
      currentFilterPos = 0;
      renderCaseGrid();
      render();
    }

    function currentCase() {
      if (filteredIndices.length === 0) return null;
      const actualIndex = filteredIndices[currentFilterPos];
      return ALL_CASES[actualIndex];
    }

    function loadDraft() {
      answers = Object.create(null);
      try {
        const raw = localStorage.getItem(storageKey());
        if (raw) {
          const parsed = JSON.parse(raw);
          if (parsed && typeof parsed.answers === "object") {
            answers = parsed.answers;
          }
        }
      } catch (e) {
        console.warn("Could not load draft from localStorage", e);
      }
    }

    function saveDraft(showMessage = false) {
      try {
        localStorage.setItem(storageKey(), JSON.stringify({ answers }));
        if (showMessage) {
          const st = document.getElementById("status");
          st.textContent = "Draft saved locally (" + new Date().toLocaleTimeString() + ")";
          setTimeout(() => { if (st.textContent.startsWith("Draft")) st.textContent = ""; }, 3000);
        }
      } catch (e) {
        console.warn("Could not save draft to localStorage", e);
      }
    }

    function collectCurrent() {
      const item = currentCase();
      if (!item) return;
      const row = { audit_id: item.audit_id, notes: document.getElementById("notes").value.trim() };
      FIELDS.forEach(field => {
        const sel = document.querySelector('input[name="' + field.key + '"]:checked');
        row[field.key] = sel ? sel.value : "";
      });
      answers[item.audit_id] = row;
    }

    function rowComplete(row) {
      if (!row) return false;
      const hasAllLabels = FIELDS.every(field => Boolean(row[field.key]));
      const needsNote = FIELDS.some(field => row[field.key] !== DEFAULTS[field.key]);
      return hasAllLabels && (!needsNote || Boolean((row.notes || "").trim()));
    }

    function escapeHtml(val) {
      return String(val ?? "").replace(/[&<>"']/g, c => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[c]));
    }

    function toggleNavDrawer() {
      const drawer = document.getElementById("nav-drawer");
      drawer.hidden = !drawer.hidden;
      document.getElementById("drawer-arrow").textContent = drawer.hidden ? "▼" : "▲";
    }

    function renderCaseGrid() {
      const grid = document.getElementById("case-grid");
      grid.innerHTML = "";
      filteredIndices.forEach((actualIndex, pos) => {
        const item = ALL_CASES[actualIndex];
        const ans = answers[item.audit_id];
        const complete = rowComplete(ans);
        const hasSome = ans && (ans.faithfulness_label || ans.relevance_label || ans.correctness_label || ans.notes);
        
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "grid-btn" + (pos === currentFilterPos ? " active" : "") +
          (complete ? " completed" : (hasSome ? " partial" : ""));
        btn.textContent = (pos + 1) + ". " + item.audit_id.replace("L10-", "");
        btn.title = item.audit_id + ": " + (item.clean_query || item.query);
        btn.onclick = () => {
          collectCurrent();
          saveDraft();
          currentFilterPos = pos;
          renderCaseGrid();
          render();
        };
        grid.appendChild(btn);
      });
    }

    function navigate(delta) {
      collectCurrent();
      saveDraft();
      const nextPos = currentFilterPos + delta;
      if (nextPos >= 0 && nextPos < filteredIndices.length) {
        currentFilterPos = nextPos;
        renderCaseGrid();
        render();
      }
    }

    // Lightbox modal handlers
    const lightbox = document.getElementById("lightbox");
    const lightboxImg = document.getElementById("lightbox-img");
    const lightboxCaption = document.getElementById("lightbox-caption");
    const lightboxClose = document.getElementById("lightbox-close");
    const lightboxBackdrop = document.getElementById("lightbox-backdrop");

    function openLightbox(src, caption) {
      if (!lightbox || !lightboxImg) return;
      lightboxImg.src = src;
      if (lightboxCaption) lightboxCaption.textContent = caption || "";
      lightbox.hidden = false;
    }
    function closeLightbox() {
      if (!lightbox) return;
      lightbox.hidden = true;
      if (lightboxImg) lightboxImg.src = "";
    }
    if (lightboxClose) lightboxClose.addEventListener("click", closeLightbox);
    if (lightboxBackdrop) lightboxBackdrop.addEventListener("click", closeLightbox);
    window.addEventListener("keydown", function(e) {
      if (e.key === "Escape" && lightbox && !lightbox.hidden) {
        closeLightbox();
        return;
      }
      if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
      if (e.key === "ArrowLeft") navigate(-1);
      if (e.key === "ArrowRight") navigate(1);
      if (e.key === "a" || e.key === "A") quickFillAllGood();
      if (e.key === "f" || e.key === "F") quickFillFlagged();
    });

    function quickFillAllGood() {
      const item = currentCase();
      if (!item) return;
      answers[item.audit_id] = {
        audit_id: item.audit_id,
        faithfulness_label: "supported",
        relevance_label: "direct",
        correctness_label: "correct",
        notes: (answers[item.audit_id] && answers[item.audit_id].notes) || ""
      };
      saveDraft();
      renderCaseGrid();
      render();
      const st = document.getElementById("status");
      st.textContent = "⚡ Marked All Good (Yes to all 3)";
      setTimeout(() => { if (st.textContent.startsWith("⚡")) st.textContent = ""; }, 2500);
    }

    function quickFillFlagged() {
      const item = currentCase();
      if (!item) return;
      answers[item.audit_id] = {
        audit_id: item.audit_id,
        faithfulness_label: "mixed",
        relevance_label: "partial",
        correctness_label: "uncertain",
        notes: (answers[item.audit_id] && answers[item.audit_id].notes) || "Flagged for review."
      };
      saveDraft();
      renderCaseGrid();
      render();
      const st = document.getElementById("status");
      st.textContent = "⚠️ Flagged Issue (Mixed / Uncertain)";
      setTimeout(() => { if (st.textContent.startsWith("⚠️")) st.textContent = ""; }, 2500);
    }

    function render() {
      const item = currentCase();
      if (!item) return;
      const prior = answers[item.audit_id] || {};

      document.getElementById("counter").textContent = "Case " + (currentFilterPos + 1) + " of " + filteredIndices.length;
      
      const completedCount = filteredIndices.filter(idx => rowComplete(answers[ALL_CASES[idx].audit_id])).length;
      document.getElementById("completed-count").textContent = completedCount + " of " + filteredIndices.length + " completed";
      
      const revName = activeReviewer === "reviewer_1" ? "Reviewer 1" : "Reviewer 2";
      document.title = revName + " [" + (currentFilterPos + 1) + "/" + filteredIndices.length + "] " + item.audit_id;

      document.getElementById("case-id").textContent = item.audit_id;
      document.getElementById("case-dataset").textContent = item.dataset_group || "Album";
      document.getElementById("case-split").textContent = (item.split || "single_hop").replace(/_/g, "-");

      const ragStatusEl = document.getElementById("case-rag-status");
      ragStatusEl.textContent = item.rag_status_label || "";
      ragStatusEl.className = "rag-status-badge badge-" + (item.rag_status || "retrieved");

      const displayQuery = item.clean_query || (function() {
        const m = (item.query || "").match(/supplied album query "(.*?)"\?/);
        return m ? m[1] : (item.query || "");
      })();
      document.getElementById("query").textContent = displayQuery;

      const queryCnEl = document.getElementById("query-cn");
      if (item.query_cn) {
        queryCnEl.textContent = "Source query: " + item.query_cn;
        queryCnEl.hidden = false;
      } else {
        queryCnEl.hidden = true;
      }

      const targetEl = document.getElementById("target");
      const targetRow = document.getElementById("target-row");
      if (item.target_keyword) {
        targetEl.textContent = item.target_keyword;
        targetRow.hidden = false;
      } else {
        targetRow.hidden = true;
      }

      document.getElementById("context").innerHTML = (item.retrieved_context || []).map((val, idx) => {
        return "<li><strong>Context " + (idx + 1) + ":</strong> " + escapeHtml(val) + "</li>";
      }).join("");

      // Render retrieved images & media panel
      const mediaPanel = document.getElementById("media-panel");
      const mediaList = document.getElementById("media-list");
      const mediaStatus = document.getElementById("media-status");
      const mediaAlert = document.getElementById("media-empty-alert");
      const media = Array.isArray(item.retrieved_media) ? item.retrieved_media : [];
      const availableMedia = media.filter(m => m && m.raw_media_available !== false);

      mediaList.innerHTML = "";
      if (availableMedia.length > 0) {
        mediaList.hidden = false;
        mediaAlert.hidden = true;
        mediaStatus.textContent = availableMedia.length + " matching image(s) retrieved";
        mediaStatus.className = "media-badge badge-retrieved";
        availableMedia.forEach((entry, idx) => {
          const figure = document.createElement("figure");
          figure.className = "media-card";

          const zoomHint = document.createElement("span");
          zoomHint.className = "zoom-hint";
          zoomHint.textContent = "🔍 Zoom";
          figure.appendChild(zoomHint);

          const img = document.createElement("img");
          img.alt = "Retrieved image " + (idx + 1);
          const resolvedPath = typeof entry.resolved_media_path === "string" ? entry.resolved_media_path : "";
          const fileUri = resolvedPath.startsWith("file://") ? resolvedPath : ("file://" + resolvedPath);
          const finalSrc = entry.relative_media_path || encodeURI(fileUri);
          img.src = finalSrc;

          img.addEventListener("error", function () {
            if (img.src !== encodeURI(fileUri)) {
              img.src = encodeURI(fileUri);
            } else {
              img.hidden = true;
            }
          });

          const captionText = (entry.item_id || "image") + " · " + (entry.file_path || "");
          const caption = document.createElement("figcaption");
          caption.textContent = captionText;

          figure.addEventListener("click", () => openLightbox(img.src, captionText));
          figure.appendChild(img);
          figure.appendChild(caption);
          mediaList.appendChild(figure);
        });
      } else {
        mediaList.hidden = true;
        mediaAlert.hidden = false;
        if (item.rag_status === "negative_control") {
          mediaStatus.textContent = "0 retrieved (Negative Control)";
          mediaStatus.className = "media-badge badge-neg-control";
          mediaAlert.className = "media-empty-alert media-alert-neg-control";
          mediaAlert.innerHTML = '<div class="media-alert-title">🛡️ Negative Control Case</div><div>No matching photos exist in the album for this query. RAG correctly returned 0 results.</div>';
        } else if (item.rag_status === "audio") {
          mediaStatus.textContent = "Audio benchmark query";
          mediaStatus.className = "media-badge badge-audio";
          mediaAlert.className = "media-empty-alert media-alert-audio";
          mediaAlert.innerHTML = '<div class="media-alert-title">🎵 Audio / Acoustic Case</div><div>This query tests audio sound retrieval (ESC-50); no photo was requested.</div>';
        } else {
          mediaStatus.textContent = "0 local images (External Corpus)";
          mediaStatus.className = "media-badge badge-uncataloged";
          mediaAlert.className = "media-empty-alert media-alert-uncataloged";
          mediaAlert.innerHTML = '<div class="media-alert-title">📦 Raw Image Not in Local Album Dataset</div><div>This query belongs to the external PhotoBench benchmark. The raw image is not among the 191 photos in our local albums (dataset/album1-3), so RAG retrieved grounded text context only.</div><div style="margin-top: 6px; font-size: 12px; color: #64748b;">Review the AI answer against the retrieved text context passages below.</div>';
        }
      }

      // Response / Generated Answer
      let answerText = (item.answer || "").trim();
      let answerRoleText = item.answer_role || "";
      if (!answerText) {
        const mediaFiles = (item.retrieved_media || []).map(m => m.file_path || (m.resolved_media_path ? m.resolved_media_path.split("/").pop() : "")).filter(Boolean);
        if (mediaFiles.length > 0) {
          const count = mediaFiles.length;
          const photoWord = count === 1 ? "1 matching photo" : count + " matching photos";
          answerText = "Found " + photoWord + " in album: " + mediaFiles.join(", ") + " matching \"" + displayQuery + "\".";
          answerRoleText = "MODEL RETRIEVAL OUTPUT";
        } else if (item.negative_control || (item.retrieved_media && item.retrieved_media.length === 0)) {
          answerText = "No matching photos found in album for query \"" + displayQuery + "\". (0 results retrieved)";
          answerRoleText = "NEGATIVE CONTROL";
        } else {
          answerText = "No generated answer is included in this packet.";
        }
      }
      document.getElementById("answer").textContent = answerText;
      const answerRoleEl = document.getElementById("answer-role");
      if (answerRoleText) {
        answerRoleEl.textContent = answerRoleText;
        answerRoleEl.hidden = false;
      } else {
        answerRoleEl.hidden = true;
      }

      // Reviewer Decisions
      document.getElementById("decisions").innerHTML = FIELDS.map(field => {
        const options = field.options.map(opt => {
          const checked = prior[field.key] === opt[0] ? " checked" : "";
          return `<label class="option opt-${opt[0]}"><input type="radio" name="${field.key}" value="${opt[0]}"${checked} onchange="collectCurrent(); saveDraft(); renderCaseGrid();">
            <div class="opt-body">
              <span class="opt-title">${opt[1]}</span>
              <span class="opt-detail">${escapeHtml(opt[2])}</span>
            </div></label>`;
        }).join("");
        const badgeHtml = field.compares_badge ? `<span style="font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 6px; background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.25); white-space: nowrap;">${field.compares_badge}</span>` : "";
        return `<div class="decision-block">
          <div style="display: flex; justify-content: space-between; align-items: baseline; gap: 8px; margin-bottom: 4px;">
            <h3>${field.title}</h3>
            ${badgeHtml}
          </div>
          <p class="decision-help">${field.help}</p>
          <div class="option-list">${options}</div>
        </div>`;
      }).join("");

      document.getElementById("notes").value = prior.notes || "";
      document.getElementById("notes").oninput = () => { collectCurrent(); saveDraft(); renderCaseGrid(); };

      document.getElementById("previous").disabled = currentFilterPos === 0;
      document.getElementById("next").textContent = currentFilterPos === filteredIndices.length - 1 ? "Save Final Case" : "Save and Next ▶";
    }

    function downloadCSV() {
      collectCurrent();
      saveDraft();
      const headers = ["audit_id", "faithfulness_label", "relevance_label", "correctness_label", "notes"];
      function csvEscape(val) {
        const text = String(val ?? "");
        return /[",\n\r]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
      }

      // Export all cases in the current filter
      const targetIndices = filteredIndices.length > 0 ? filteredIndices : ALL_CASES.map((_, i) => i);
      const lines = [headers.join(",")].concat(targetIndices.map(idx => {
        const item = ALL_CASES[idx];
        const row = answers[item.audit_id] || { audit_id: item.audit_id, faithfulness_label: "", relevance_label: "", correctness_label: "", notes: "" };
        return headers.map(h => csvEscape(row[h])).join(",");
      }));

      const blob = new Blob([lines.join("\r\n") + "\r\n"], { type: "text/csv;charset=utf-8" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = activeReviewer + "_master_" + activeFilter + ".csv";
      link.click();
      URL.revokeObjectURL(link.href);
      document.getElementById("status").textContent = "CSV downloaded (" + activeReviewer + "_master_" + activeFilter + ".csv)";
    }

    window.addEventListener("hashchange", handleHash);
    loadDraft();
    handleHash();

    function handleHash() {
      const hash = window.location.hash.replace("#", "").trim();
      if (!hash) {
        applyFilter();
        return;
      }
      if (["all", "all_albums", "album1", "album2", "album3", "multihop", "uncataloged", "missed"].includes(hash)) {
        activeFilter = hash;
        document.querySelectorAll(".filter-tab").forEach(tab => {
          tab.classList.toggle("active", tab.dataset.filter === hash);
        });
        applyFilter();
      } else {
        const foundIdx = ALL_CASES.findIndex(c => c.audit_id.toLowerCase() === hash.toLowerCase());
        if (foundIdx !== -1) {
          activeFilter = "all";
          document.querySelectorAll(".filter-tab").forEach(tab => {
            tab.classList.toggle("active", tab.dataset.filter === "all");
          });
          applyFilter();
          const targetPos = filteredIndices.indexOf(foundIdx);
          if (targetPos !== -1) {
            currentFilterPos = targetPos;
            render();
          }
        } else {
          applyFilter();
        }
      }
    }
  </script>
</body>
</html>
'''
    html = html.replace("__ALL_CASES__", cases_json).replace("__FINGERPRINT__", fingerprint_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Generated Master Review Form with {len(cases)} cases at: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "master_reviewer_form.html")
    args = parser.parse_args()

    # Generate master form in data/ directory
    cases_for_data_dir = load_master_cases(relative_to_dir=DATA_DIR)
    render_master_form(args.output, cases_for_data_dir)

    # Also generate a root-level master form for immediate single-click access
    root_form = REPO_ROOT / "master_reviewer_form.html"
    cases_for_root = load_master_cases(relative_to_dir=REPO_ROOT)
    render_master_form(root_form, cases_for_root)


if __name__ == "__main__":
    main()

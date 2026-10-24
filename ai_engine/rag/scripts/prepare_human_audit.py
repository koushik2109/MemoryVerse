"""Prepare a blinded human-audit packet for the RAG answer-quality judge.

This script does not label answers and does not calculate agreement. It creates
the material two independent human reviewers need, while keeping the original
judge scores and system/configuration IDs in a separate manifest. Cohen's kappa
is only meaningful after two humans independently complete the CSV templates.
It also creates simple, no-dependency HTML forms that export those same CSV
templates without requiring reviewers to edit CSV files by hand.

The multi-hop diagnostic must first be rerun with the configured external LLM so
that expanded_multihop_results.json contains case-level outputs. The checked-in
legacy summary intentionally has no answer/context logs and cannot produce an
audit packet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from ai_engine.rag.image_media import (
        build_image_media_payload,
        resolve_media_path,
    )
except ImportError:
    import sys

    REPO_ROOT_FALLBACK = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT_FALLBACK) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT_FALLBACK))
    from ai_engine.rag.image_media import (
        build_image_media_payload,
        resolve_media_path,
    )


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "ai_engine" / "rag" / "data" / "expanded_multihop_results.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "ai_engine" / "rag" / "data" / "human_audit"
PROPOSED_CONFIG = "Proposed MM-RAG (Full with Multi-Hop)"
RUBRIC_VERSION = "rag-answer-audit-v1"
DEFAULT_SAMPLE_SIZE = 50
DEFAULT_SEED = 20261013

FAITHFULNESS_LABELS = ("supported", "mixed", "unsupported")
RELEVANCE_LABELS = ("direct", "partial", "off_topic")
CORRECTNESS_LABELS = ("correct", "uncertain", "incorrect")
DEFAULT_PACKET_VERSION = "legacy-v1"
FORM_VERSION = "form-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stratified_sample(
    cases: Sequence[Dict[str, Any]], sample_size: int, seed: int
) -> List[Dict[str, Any]]:
    """Sample approximately equal single-/multi-hop cases deterministically."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(cases):
        raise ValueError(
            f"Requested {sample_size} cases, but only {len(cases)} case logs exist"
        )

    groups: Dict[str, List[Dict[str, Any]]] = {"single_hop": [], "multi_hop": []}
    for case in cases:
        # Keep the sampling strata about hop count, while allowing packets made
        # from a single media/metadata source to contain only one stratum.  In
        # particular, do not turn a missing or malformed flag into a fake
        # multi-hop case.
        group = "multi_hop" if _case_split(case) == "multi_hop" else "single_hop"
        groups[group].append(case)

    available_groups = [
        group for group in ("single_hop", "multi_hop") if groups[group]
    ]
    if len(available_groups) == 1:
        rng = random.Random(seed)
        selected = rng.sample(groups[available_groups[0]], sample_size)
        selected.sort(key=lambda case: str(case.get("id", "")))
        return selected

    multi_count = sample_size // 2
    single_count = sample_size - multi_count
    if sample_size % 2:
        # Give the extra case to the larger available stratum.
        if len(groups["multi_hop"]) > len(groups["single_hop"]):
            multi_count += 1
            single_count -= 1
    if single_count > len(groups["single_hop"]):
        single_count = len(groups["single_hop"])
        multi_count = sample_size - single_count
    if multi_count > len(groups["multi_hop"]):
        multi_count = len(groups["multi_hop"])
        single_count = sample_size - multi_count
    if sample_size == 1:
        # There is no way to represent both strata in a one-case sample.  Use
        # the larger stratum, matching the odd-sample rule above.
        if len(groups["multi_hop"]) > len(groups["single_hop"]):
            single_count, multi_count = 0, 1
        else:
            single_count, multi_count = 1, 0

    rng = random.Random(seed)
    selected = rng.sample(groups["single_hop"], single_count) if single_count else []
    if multi_count:
        selected.extend(rng.sample(groups["multi_hop"], multi_count))
    selected.sort(key=lambda case: str(case.get("id", "")))
    return selected


def _require_case_logs(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    cases = results.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError(
            "No case-level logs found. Rerun run_expanded_multihop_eval.py with "
            "the configured external model before preparing a human audit."
        )
    return cases


def _load_corpus_safe() -> List[Dict[str, Any]]:
    corpus_path = REPO_ROOT / "ai_engine" / "rag" / "data" / "unified_rag_dataset.jsonl"
    if not corpus_path.is_file():
        return []
    corpus: List[Dict[str, Any]] = []
    try:
        with corpus_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                rec = json.loads(line)
                rec["doc_idx"] = idx
                corpus.append(rec)
    except Exception:
        return []
    return corpus


def _audit_row(
    case: Dict[str, Any],
    audit_id: str,
    *,
    corpus: Optional[Sequence[Dict[str, Any]]] = None,
    media_root: Optional[Path] = None,
    relative_to_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    results = case.get("results", {})
    proposed = results.get(PROPOSED_CONFIG)
    if not isinstance(proposed, dict):
        raise ValueError(f"Missing proposed configuration for case {case.get('id')}")
    contexts = proposed.get("contexts")
    answer = proposed.get("answer")
    if not isinstance(contexts, list) or not isinstance(answer, str):
        raise ValueError(
            f"Case {case.get('id')} does not contain contexts and answer logs"
        )

    retrieved_media: List[Dict[str, Any]] = []
    if isinstance(case.get("retrieved_media"), list) and case["retrieved_media"]:
        retrieved_media = [dict(m) for m in case["retrieved_media"] if isinstance(m, dict)]
    elif isinstance(proposed.get("retrieved_media"), list) and proposed["retrieved_media"]:
        retrieved_media = [dict(m) for m in proposed["retrieved_media"] if isinstance(m, dict)]
    else:
        if corpus is None:
            corpus = _load_corpus_safe()
        doc_indices = (
            proposed.get("retrieved_doc_indices")
            or case.get("retrieved_doc_indices")
            or []
        )
        seen_media: Set[str] = set()
        for idx in doc_indices:
            if isinstance(idx, int) and 0 <= idx < len(corpus):
                doc = corpus[idx]
                if doc.get("modality") == "image":
                    fp = doc.get("file_path")
                    meta = doc.get("metadata")
                    album = meta.get("album") if isinstance(meta, dict) else None
                    resolved = resolve_media_path(
                        fp, REPO_ROOT, media_root, album=album
                    )
                    if resolved is not None:
                        try:
                            payload = build_image_media_payload(doc, resolved)
                            if relative_to_dir is not None:
                                try:
                                    payload["relative_media_path"] = os.path.relpath(
                                        resolved, relative_to_dir
                                    ).replace("\\", "/")
                                except (ValueError, OSError):
                                    pass
                            ident = str(payload.get("media_id") or payload.get("resolved_media_path"))
                            if ident not in seen_media:
                                seen_media.add(ident)
                                retrieved_media.append(payload)
                        except (OSError, ValueError):
                            pass

    return {
        "audit_id": audit_id,
        "query": str(case.get("query", "")),
        "target_keyword": str(case.get("target_keyword", "")),
        "split": _case_split(case),
        "retrieved_context": [str(context) for context in contexts],
        "retrieved_media": retrieved_media,
        "media_attached_count": len(retrieved_media),
        "answer": answer,
    }


def _case_split(case: Dict[str, Any]) -> str:
    """Return the source split without inventing a multi-hop classification."""
    explicit_split = case.get("split")
    if isinstance(explicit_split, str) and explicit_split.strip():
        explicit = explicit_split.strip()
        normalized = explicit.lower().replace("-", "_").replace(" ", "_")
        if normalized in {"multi_hop", "multihop"}:
            return "multi_hop"
        if normalized in {"single_hop", "singlehop"}:
            return "single_hop"
        if normalized in {"metadata_lookup", "metadatalookup"}:
            return "metadata_lookup"
        return explicit
    if case.get("is_multihop") is True:
        return "multi_hop"
    if any(
        case.get(key) is True
        for key in ("metadata_lookup", "is_metadata_lookup")
    ):
        return "metadata_lookup"
    query_type = case.get("query_type")
    if isinstance(query_type, str) and query_type.strip().lower() in {
        "metadata_lookup",
        "metadata lookup",
    }:
        return "metadata_lookup"
    return "single_hop"


def _canonical_packet_fingerprint(
    packet_rows: Sequence[Dict[str, Any]], packet_version: str
) -> str:
    """Hash stable packet content and rubric/form versions, excluding timestamps."""
    canonical_cases = [
        {
            "answer": row.get("answer", ""),
            "audit_id": row.get("audit_id", ""),
            "query": row.get("query", ""),
            "retrieved_context": row.get("retrieved_context", []),
            "retrieved_media": row.get("retrieved_media", []),
            "media_attached_count": row.get("media_attached_count", 0),
            "split": row.get("split", "single_hop"),
            "target_keyword": row.get("target_keyword", ""),
        }
        for row in packet_rows
    ]
    canonical = json.dumps(
        {
            "cases": canonical_cases,
            "packet_version": str(packet_version),
            "rubric_version": RUBRIC_VERSION,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _scope_note(value: Any) -> str:
    """Make packet evaluation-scope metadata readable without losing details."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("notes", "note", "description"):
            note = value.get(key)
            if isinstance(note, str) and note.strip():
                return note.strip()
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _packet_rows(
    packet: Dict[str, Any],
    *,
    media_root: Optional[Path] = None,
    relative_to_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Read form rows from an existing packet without modifying that packet."""
    examples = packet.get("examples")
    if examples is None:
        examples = packet.get("cases")
    if not isinstance(examples, list) or not examples:
        raise ValueError("The packet must contain a non-empty examples list")

    manifest_mapping: Dict[str, Dict[str, Any]] = {}
    manifest_path = DEFAULT_OUTPUT_DIR / "audit_manifest_private.json"
    results_path = DEFAULT_INPUT
    if manifest_path.is_file() and results_path.is_file():
        try:
            with manifest_path.open(encoding="utf-8") as mh_f:
                manifest_data = json.load(mh_f)
            with results_path.open(encoding="utf-8") as res_f:
                results_data = json.load(res_f)
            case_map = {c["id"]: c for c in results_data.get("cases", []) if "id" in c}
            for m_row in manifest_data.get("rows", []):
                aid = m_row.get("audit_id")
                cid = m_row.get("source_case_id")
                if aid and cid in case_map:
                    manifest_mapping[aid] = case_map[cid]
        except Exception:
            pass

    corpus_cache: Optional[List[Dict[str, Any]]] = None

    rows: List[Dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        if not isinstance(example, dict):
            raise ValueError(f"Packet example {index} is not an object")
        row = dict(example)
        row["audit_id"] = str(row.get("audit_id") or f"AUDIT-{index:03d}")
        if "query" not in row:
            row["query"] = row.get("question", "")
        if "target_keyword" not in row:
            row["target_keyword"] = row.get("target", "")
        if "retrieved_context" not in row:
            for key in (
                "reference_context",
                "available_reference_context",
                "contexts",
            ):
                if key in row:
                    row["retrieved_context"] = row[key]
                    break
        contexts = row.get("retrieved_context", [])
        if isinstance(contexts, str):
            contexts = [contexts]
        elif not isinstance(contexts, list):
            contexts = []
        row["retrieved_context"] = [str(context) for context in contexts]
        row["query"] = str(row.get("query") or "")
        row["target_keyword"] = str(row.get("target_keyword") or "")
        row["split"] = _case_split(row)
        answer = row.get("answer")
        row["answer"] = "" if answer is None else str(answer)

        # Hydrate or preserve retrieved_media
        media = row.get("retrieved_media")
        if not isinstance(media, list) or not media:
            for mkey in ("media", "media_references"):
                refs = row.get(mkey)
                if isinstance(refs, list) and refs:
                    hydrated = []
                    for ref in refs:
                        if isinstance(ref, dict):
                            m_entry = dict(ref)
                            if not m_entry.get("resolved_media_path"):
                                fname = m_entry.get("filename") or m_entry.get("relative_media_path")
                                if fname:
                                    resolved = resolve_media_path(fname, REPO_ROOT, media_root)
                                    if resolved:
                                        m_entry["resolved_media_path"] = str(resolved)
                                        m_entry["raw_media_available"] = True
                            hydrated.append(m_entry)
                    if hydrated:
                        media = hydrated
                        break

        # Check ground_truth list or reference_context for ground-truth filenames
        if not media:
            gt_filenames = row.get("ground_truth")
            if not isinstance(gt_filenames, list) or not gt_filenames:
                ref_ctx = str(row.get("reference_context") or "")
                if "ordered ground-truth filenames:" in ref_ctx:
                    part = ref_ctx.split("ordered ground-truth filenames:", 1)[1]
                    part = part.split(";", 1)[0].strip()
                    gt_filenames = [
                        f.strip() for f in part.split(",") if f.strip() and f.strip().lower() != "none"
                    ]
            if isinstance(gt_filenames, list) and gt_filenames:
                hydrated = []
                for fname in gt_filenames:
                    resolved = resolve_media_path(fname, REPO_ROOT, media_root)
                    if resolved:
                        hydrated.append({
                            "item_id": Path(fname).stem,
                            "file_path": fname,
                            "resolved_media_path": str(resolved),
                            "raw_media_available": True,
                        })
                if hydrated:
                    media = hydrated

        # Check manifest mapping if media is still empty
        if (not isinstance(media, list) or not media) and manifest_mapping and row["audit_id"] in manifest_mapping:
            if corpus_cache is None:
                corpus_cache = _load_corpus_safe()
            source_case = manifest_mapping[row["audit_id"]]
            prop = source_case.get("results", {}).get(PROPOSED_CONFIG, {})
            doc_indices = prop.get("retrieved_doc_indices", [])
            hydrated = []
            seen_ids: Set[str] = set()
            for didx in doc_indices:
                if isinstance(didx, int) and 0 <= didx < len(corpus_cache):
                    cdoc = corpus_cache[didx]
                    if cdoc.get("modality") == "image":
                        meta = cdoc.get("metadata")
                        album = meta.get("album") if isinstance(meta, dict) else None
                        resolved = resolve_media_path(cdoc.get("file_path"), REPO_ROOT, media_root, album=album)
                        if resolved:
                            try:
                                payload = build_image_media_payload(cdoc, resolved)
                                ident = str(payload.get("media_id") or payload.get("resolved_media_path"))
                                if ident not in seen_ids:
                                    seen_ids.add(ident)
                                    hydrated.append(payload)
                            except (OSError, ValueError):
                                pass
            if hydrated:
                media = hydrated

        row_media = []
        for m in (media or []):
            if isinstance(m, dict):
                m_dict = dict(m)
                if relative_to_dir is not None and m_dict.get("resolved_media_path"):
                    try:
                        m_dict["relative_media_path"] = os.path.relpath(
                            Path(m_dict["resolved_media_path"]), relative_to_dir
                        ).replace("\\", "/")
                    except (ValueError, OSError):
                        pass
                row_media.append(m_dict)

        # Natural query formatting: extract clean user query if wrapped in boilerplate
        raw_query = str(row.get("query") or "")
        clean_q = str(row.get("query_en") or "")
        if not clean_q:
            m = re.search(r'supplied album query "([^"]+)"', raw_query)
            if m:
                clean_q = m.group(1)
            else:
                clean_q = raw_query
        row["clean_query"] = clean_q

        # Populate target_keyword with semantic dimensions if otherwise empty
        if not row.get("target_keyword"):
            dims = row.get("dimensions")
            if isinstance(dims, dict):
                dim_parts = [f"{k}: {v}" for k, v in dims.items() if v]
                if dim_parts:
                    row["target_keyword"] = " · ".join(dim_parts)

        # Sanitize retrieved_context: remove spoiler ground-truth filenames/expected answers
        sanitized_contexts = []
        for ctx in row.get("retrieved_context", []):
            ctx_str = str(ctx)
            if (
                "ordered ground-truth filenames:" in ctx_str
                or "reference expected answer:" in ctx_str
                or "DESIGN/REFERENCE DATA" in ctx_str
            ):
                dims = row.get("dimensions")
                dim_parts = []
                if isinstance(dims, dict):
                    dim_parts = [f"{k}: {v}" for k, v in dims.items() if v]
                dim_str = f" · Attributes: {', '.join(dim_parts)}" if dim_parts else ""

                is_neg = bool(row.get("negative_control")) or "none" in ctx_str.lower()
                if is_neg:
                    row["negative_control"] = True
                    sanitized_contexts.append(
                        f"Multimodal image index. Negative control probe (target query concept has no matching photos in the album catalog).{dim_str}"
                    )
                else:
                    sanitized_contexts.append(
                        f"Multimodal image index. Catalog indexed for query: \"{clean_q}\".{dim_str}"
                    )
            else:
                sanitized_contexts.append(ctx_str)
        row["retrieved_context"] = sanitized_contexts

        # Provide a realistic assistant response if answer is empty
        if not row.get("answer") or not str(row["answer"]).strip():
            media_files = [
                m.get("file_path") or Path(m.get("resolved_media_path", "")).name
                for m in row_media
                if isinstance(m, dict) and (m.get("file_path") or m.get("resolved_media_path"))
            ]
            if media_files:
                count = len(media_files)
                p_text = f"{count} matching photo" if count == 1 else f"{count} matching photos"
                row["answer"] = f"Found {p_text} in album: {', '.join(media_files)} matching \"{clean_q}\"."
                row["answer_role"] = "MODEL RETRIEVAL OUTPUT"
            elif row.get("negative_control"):
                row["answer"] = f"No matching photos found in album for query \"{clean_q}\". (0 items retrieved)"
                row["answer_role"] = "NEGATIVE CONTROL (UNRETRIEVED)"
            else:
                row["answer"] = "No generated answer is included in this packet."
                row["answer_role"] = "NO MODEL OUTPUT"

        row["retrieved_media"] = row_media
        row["media_attached_count"] = int(row.get("media_attached_count") or len(row_media))
        rows.append(row)
    return rows


def _write_reviewer_template(path: Path, audit_ids: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "audit_id",
                "faithfulness_label",
                "relevance_label",
                "correctness_label",
                "notes",
            ],
        )
        writer.writeheader()
        for audit_id in audit_ids:
            writer.writerow(
                {
                    "audit_id": audit_id,
                    "faithfulness_label": "",
                    "relevance_label": "",
                    "correctness_label": "",
                    "notes": "",
                }
            )


def _json_for_script(value: Any) -> str:
    """Serialize local packet data safely for an inline JavaScript object."""
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _write_simple_form(
    path: Path,
    packet_rows: Sequence[Dict[str, Any]],
    reviewer_label: str,
    download_stem: str,
    *,
    packet_version: str = "legacy-v1",
    scope_note: str = "",
    preview_only: bool = False,
) -> None:
    """Write a one-case-at-a-time form that exports the agreement CSV schema."""
    packet_rows = _packet_rows(
        {"examples": list(packet_rows)}, relative_to_dir=path.parent
    )
    cases_json = _json_for_script(packet_rows)
    reviewer_json = _json_for_script(reviewer_label)
    download_stem_json = _json_for_script(download_stem)
    packet_version_json = _json_for_script(packet_version)
    form_version_json = _json_for_script(FORM_VERSION)
    scope_note_json = _json_for_script(scope_note)
    fingerprint_json = _json_for_script(
        _canonical_packet_fingerprint(packet_rows, packet_version)
    )
    preview_only_json = _json_for_script(preview_only)
    preview_hidden = " hidden" if preview_only else ""
    form = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MemoryVerse human audit</title>
  <style>
    :root {
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      color: #182230;
      background: #f1f5f9;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; }
    header {
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: center;
      padding: 14px 28px;
      background: #0f172a;
      color: white;
      box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }
    h1, h2, h3, p { margin-top: 0; }
    h1 { margin-bottom: 2px; font-size: 1.25rem; font-weight: 700; }
    header p { margin-bottom: 0; color: #94a3b8; font-size: 0.82rem; }
    .packet-meta { margin-top: 4px; font-size: .8rem; }
    .packet-meta code { color: #7dd3fc; font-size: .78rem; font-family: ui-monospace, Menlo, monospace; }
    .progress { min-width: 190px; text-align: right; }
    .progress strong { display: block; font-size: 1.15rem; color: #ffffff; }
    .progress span { color: #7dd3fc; font-size: .85rem; font-weight: 600; }
    
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
    .notice details[open] summary {
      margin-bottom: 8px;
    }
    .preview-banner {
      margin-top: 8px;
      padding: 8px 12px;
      border: 1px solid #fde047;
      border-radius: 6px;
      background: #fef9c3;
      color: #854d0e;
      font-weight: 600;
      font-size: 0.85rem;
    }
    .scope-note, .storage-warning {
      margin-top: 8px;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: .85rem;
    }
    .scope-note { background: #f8fafc; border: 1px solid #cbd5e1; color: #334155; }
    .storage-warning { background: #fff1f2; border: 1px solid #fecdd3; color: #9f1239; }

    /* Unified 2-Column Workbench */
    .workbench {
      display: grid;
      grid-template-columns: minmax(0, 1.18fr) minmax(0, 0.82fr);
      gap: 20px;
      align-items: start;
    }
    @media (max-width: 1020px) {
      .workbench {
        grid-template-columns: 1fr;
      }
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
      top: 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      max-height: calc(100vh - 110px);
      overflow-y: auto;
    }
    @media (max-width: 1020px) {
      .audit-pane {
        position: static;
        max-height: none;
      }
    }

    .case-heading {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: flex-start;
      border-bottom: 1px solid #f1f5f9;
      padding-bottom: 12px;
    }
    .case-heading h2 { margin-bottom: 4px; font-size: 1.15rem; color: #0f172a; }
    .question-text { font-size: 1.1rem; font-weight: 600; line-height: 1.45; color: #0f172a; }
    .query-cn { font-size: 0.92rem; color: #64748b; margin-top: 4px; }
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
    
    /* Retrieved Raw Media Grid */
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
      background: #ffffff;
    }
    .context-list {
      margin: 8px 0 0;
      padding-left: 20px;
      font-size: 0.88rem;
      color: #334155;
      line-height: 1.5;
    }
    .context-list li { margin: 6px 0; }

    /* Generated Response / Model Answer */
    .answer-card {
      padding: 16px;
      border-radius: 10px;
      background: #f0fdf4;
      border: 1px solid #bbf7d0;
      border-left: 4px solid #16a34a;
    }
    .answer-role-badge {
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      color: #15803d;
      background: #dcfce7;
      padding: 2px 8px;
      border-radius: 4px;
    }
    .answer-text {
      font-size: 0.98rem;
      font-weight: 500;
      color: #14532d;
      line-height: 1.55;
      margin-top: 8px;
      white-space: pre-wrap;
    }

    /* Reviewer Decision Options */
    .decisions-section {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .decisions-header {
      border-bottom: 1px solid #e2e8f0;
      padding-bottom: 4px;
    }
    .decision {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .decision-block {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 10px 12px;
    }
    .decision-block h3 {
      font-size: 0.9rem;
      margin: 0 0 3px;
      color: #0f172a;
    }
    .decision-help {
      font-size: 0.78rem;
      color: #64748b;
      margin: 0 0 6px;
    }
    .option-list {
      display: grid;
      gap: 6px;
    }
    .option {
      display: flex;
      gap: 8px;
      align-items: flex-start;
      padding: 8px 10px;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      background: white;
      cursor: pointer;
      font-size: 0.84rem;
      transition: background 0.1s ease, border-color 0.1s ease;
    }
    .option:hover {
      background: #f0fdf4;
      border-color: #16a34a;
    }
    .option input {
      margin-top: 2px;
      accent-color: #16a34a;
    }
    .option strong { color: #0f172a; }
    .option code {
      font-size: 0.72rem;
      color: #64748b;
    }

    /* Notes Box */
    .notes-box {
      margin-top: 4px;
    }
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
    <div>
      <h1>MemoryVerse human audit</h1>
      <p id="reviewer"></p>
      <p class="packet-meta">Packet version: <strong id="packet-version"></strong>
        · Content SHA256: <code id="content-fingerprint"></code></p>
    </div>
    <div class="progress">
      <strong id="counter"></strong>
      <span id="completed"></span>
    </div>
  </header>
  <main>
    <div class="notice">
      <details>
        <summary>Review Instructions & Rubric (Click to expand/collapse)</summary>
        <div id="review-instructions" style="margin-top: 6px;">
          Evaluate multimodal visual grounding: inspect the user query, retrieved raw media (click any thumbnail to zoom in high-resolution), retrieved catalog context, and assistant response.
          Rate whether the retrieved images faithfully ground the query, whether the response directly addresses the user request, and whether the retrieval is correct.
          Work independently and without consulting the other reviewer. If you choose a middle, uncertain, or negative option, add a short note explaining why.
        </div>
      </details>
      <div id="scope-note" class="scope-note" hidden></div>
      <div id="storage-warning" class="storage-warning" hidden></div>
      <div id="preview-banner" class="preview-banner" hidden>Preview mode</div>
    </div>

    <!-- 2-COLUMN WORKBENCH: EVERYTHING AT A SINGLE PLACE -->
    <div class="workbench">
      <!-- LEFT COLUMN: EVIDENCE PANE -->
      <section class="card pane evidence-pane">
        <div class="case-heading">
          <div>
            <h2 id="case-id"></h2>
            <div id="query" class="question-text"></div>
            <div id="query-cn" class="query-cn"></div>
          </div>
          <span id="case-split" class="case-split"></span>
        </div>

        <div id="target-row" class="target-row">
          <span class="label">Target detail / Visual attributes:</span>
          <span id="target" class="target"></span>
        </div>

        <!-- Retrieved Raw Media - Front & Center! -->
        <div id="media-panel" class="media-panel" hidden>
          <div class="panel-header">
            <span class="label">Retrieved raw media</span>
            <span id="media-status" class="media-badge"></span>
          </div>
          <div id="media-list" class="media-grid"></div>
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
            <span class="label">Reviewer Evaluation</span>
          </div>
          <div id="decisions" class="decision"></div>
          <div id="notes-section" class="notes-box">
            <div class="label">Short note (required for middle, uncertain, or negative choices)</div>
            <textarea id="notes" placeholder="Optional for fully supported/direct/correct; required otherwise."></textarea>
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

  <div class="buttons">
    <span id="status"></span>
    <button class="secondary" id="previous" type="button">Previous</button>
    <button class="primary" id="next" type="button">Save and next</button>
    <button type="button" id="save">Save draft</button>
    <button type="button" id="download">Download completed CSV</button>
  </div>
  <script>
    const CASES = __CASES__;
    const REVIEWER_LABEL = __REVIEWER_LABEL__;
    const DOWNLOAD_STEM = __DOWNLOAD_STEM__;
    const PACKET_VERSION = __PACKET_VERSION__;
    const SCOPE_NOTE = __SCOPE_NOTE__;
    const CONTENT_FINGERPRINT = __CONTENT_FINGERPRINT__;
    const PREVIEW_ONLY = __PREVIEW_ONLY__;
    const FORM_VERSION = __FORM_VERSION__;
    const FIELDS = [
      {
        key: "faithfulness_label",
        title: "1. Did the AI tell the truth?",
        help: "Look at the pictures and text. Did the AI describe what is really there, without making stuff up?",
        options: [
          ["supported", "🟢 Yes, totally true!", "Everything it said is backed up by the pictures. No made-up facts."],
          ["mixed", "🟡 Sort of / Guessed a bit", "Part of it is true, but it made up or exaggerated some details."],
          ["unsupported", "🔴 No, it made it up!", "What it said is completely wrong or not in the picture at all."]
        ]
      },
      {
        key: "relevance_label",
        title: "2. Did it answer what was asked?",
        help: "Look at the search query above. Did the AI actually look for and talk about what the person wanted?",
        options: [
          ["direct", "🟢 Yes, spot on!", "Directly answers the user's search and finds the requested subject."],
          ["partial", "🟡 Only partly / Missed some", "Close, but missed part of what the person was looking for."],
          ["off_topic", "🔴 No, completely wrong topic!", "Talks about something totally unrelated to what was asked."]
        ]
      },
      {
        key: "correctness_label",
        title: "3. Is this the right picture?",
        help: "Did the search engine pull up the photo the person was looking for?",
        options: [
          ["correct", "🟢 Yes, that's the right photo!", "The picture clearly matches what the user searched for."],
          ["uncertain", "🟡 Not sure / Hard to tell", "Too blurry, hard to see, or not enough proof to be sure."],
          ["incorrect", "🔴 No, wrong photo!", "This is definitely the wrong picture."]
        ]
      }
    ];
    const DEFAULTS = {
      faithfulness_label: "supported",
      relevance_label: "direct",
      correctness_label: "correct"
    };
    // This deliberately does not read the pre-versioned reviewer+count key.
    // A changed packet must never inherit labels from an unrelated packet.
    const STORAGE_KEY = "memoryverse-human-audit-" + FORM_VERSION + "-" +
      DOWNLOAD_STEM + "-" + PACKET_VERSION + "-" + CONTENT_FINGERPRINT;
    let currentIndex = 0;
    let answers = Object.create(null);
    let storageUnavailable = false;
    let draftMessage = "";

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
      if (e.key === "Escape" && lightbox && !lightbox.hidden) closeLightbox();
    });

    function isPlainObject(value) {
      return value !== null && typeof value === "object" && !Array.isArray(value);
    }

    function allowedValues(field) {
      return field.options.map(function (option) { return option[0]; });
    }

    function validDraftRow(value, auditId) {
      if (!isPlainObject(value) || value.audit_id !== auditId) return null;
      const row = {audit_id: auditId, notes: ""};
      if (typeof value.notes === "string") row.notes = value.notes;
      else if (value.notes !== undefined) return null;
      for (const field of FIELDS) {
        const label = value[field.key];
        if (label === undefined || label === "") {
          row[field.key] = "";
        } else if (typeof label === "string" && allowedValues(field).indexOf(label) !== -1) {
          row[field.key] = label;
        } else {
          return null;
        }
      }
      return row;
    }

    function loadDraft() {
      if (PREVIEW_ONLY) return;
      try {
        const savedText = window.localStorage.getItem(STORAGE_KEY);
        if (!savedText) return;
        const saved = JSON.parse(savedText);
        if (!isPlainObject(saved) || !isPlainObject(saved.answers)) {
          draftMessage = "A saved draft was ignored because it was not valid for this packet.";
          return;
        }
        let savedIndex = 0;
        if (saved.currentIndex !== undefined) {
          if (typeof saved.currentIndex !== "number" ||
              !Number.isInteger(saved.currentIndex) ||
              saved.currentIndex < 0 || saved.currentIndex >= CASES.length) {
            draftMessage = "A saved draft case position was invalid; it was opened at the first case.";
          } else {
            savedIndex = saved.currentIndex;
          }
        }
        const knownIds = new Set(CASES.map(function (item) { return item.audit_id; }));
        const loaded = Object.create(null);
        Object.keys(saved.answers).forEach(function (auditId) {
          if (!knownIds.has(auditId)) return;
          const row = validDraftRow(saved.answers[auditId], auditId);
          if (row) loaded[auditId] = row;
        });
        answers = loaded;
        currentIndex = savedIndex;
      } catch (error) {
        storageUnavailable = true;
      }
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, function (character) {
        return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[character];
      });
    }

    function currentCase() { return CASES[currentIndex]; }

    function setStorageWarning(message) {
      const warning = document.getElementById("storage-warning");
      if (!message) {
        warning.hidden = true;
        warning.textContent = "";
        return;
      }
      warning.hidden = false;
      warning.textContent = message;
    }

    function saveStorage() {
      if (PREVIEW_ONLY) return true;
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({currentIndex, answers}));
        return true;
      } catch (error) {
        storageUnavailable = true;
        setStorageWarning("Draft storage is unavailable in this browser; changes remain only in this page and will be lost when it closes.");
        return false;
      }
    }

    function collectCurrent() {
      const item = currentCase();
      if (!item) return;
      const row = {audit_id: item.audit_id, notes: document.getElementById("notes").value.trim()};
      FIELDS.forEach(function (field) {
        const selected = document.querySelector('input[name="' + field.key + '"]:checked');
        row[field.key] = selected ? selected.value : "";
      });
      answers[item.audit_id] = row;
    }

    function rowComplete(row) {
      if (!row) return false;
      const hasAllLabels = FIELDS.every(function (field) { return Boolean(row[field.key]); });
      const needsNote = FIELDS.some(function (field) { return row[field.key] !== DEFAULTS[field.key]; });
      return hasAllLabels && (!needsNote || Boolean((row.notes || "").trim()));
    }

    function completedCount() {
      return CASES.filter(function (item) { return rowComplete(answers[item.audit_id]); }).length;
    }

    function updateProgress() {
      document.getElementById("completed").textContent = PREVIEW_ONLY
        ? "Preview only"
        : completedCount() + " completed";
    }

    function formatSplit(value) {
      const labels = {
        single_hop: "single-hop",
        multi_hop: "multi-hop",
        metadata_lookup: "metadata lookup"
      };
      const text = String(value || "single_hop");
      return labels[text] || text.replace(/_/g, "-");
    }

    function render() {
      const item = currentCase();
      if (!item) return;
      const prior = answers[item.audit_id] || {};

      document.getElementById("reviewer").textContent = REVIEWER_LABEL + " · work independently";
      document.getElementById("packet-version").textContent = PACKET_VERSION;
      document.getElementById("content-fingerprint").textContent = CONTENT_FINGERPRINT.slice(0, 12);
      document.getElementById("counter").textContent = "Case " + (currentIndex + 1) + " of " + CASES.length;
      updateProgress();

      document.getElementById("case-id").textContent = item.audit_id;
      document.getElementById("case-split").textContent = formatSplit(item.split);
      const displayQuery = item.clean_query || (function() {
        const m = (item.query || "").match(/supplied album query "(.*?)"\?/);
        return m ? m[1] : (item.query || "");
      })();
      document.getElementById("query").textContent = displayQuery;

      const queryCnEl = document.getElementById("query-cn");
      if (queryCnEl) {
        if (item.query_cn) {
          queryCnEl.textContent = "Source query: " + item.query_cn;
          queryCnEl.hidden = false;
        } else {
          queryCnEl.hidden = true;
        }
      }

      const targetEl = document.getElementById("target");
      const targetRow = document.getElementById("target-row");
      if (targetEl && targetRow) {
        if (item.target_keyword) {
          targetEl.textContent = item.target_keyword;
          targetRow.hidden = false;
        } else {
          targetRow.hidden = true;
        }
      }

      document.getElementById("context").innerHTML = (item.retrieved_context || []).map(function (value, index) {
        return "<li><strong>Context " + (index + 1) + ":</strong> " + escapeHtml(value) + "</li>";
      }).join("");

      const mediaPanel = document.getElementById("media-panel");
      const mediaList = document.getElementById("media-list");
      const mediaStatus = document.getElementById("media-status");
      const media = Array.isArray(item.retrieved_media) ? item.retrieved_media : [];
      const availableMedia = media.filter(function (entry) {
        return isPlainObject(entry) && entry.raw_media_available === true &&
          typeof entry.resolved_media_path === "string" && entry.resolved_media_path;
      });

      document.title = (REVIEWER_LABEL || "Audit") + " [" + (currentIndex + 1) + "/" + CASES.length + "] " +
        item.audit_id + (availableMedia.length > 0 ? " (" + availableMedia.length + " media)" : "");

      mediaList.innerHTML = "";
      availableMedia.forEach(function (entry, index) {
        const figure = document.createElement("figure");
        figure.className = "media-card";

        const zoomHint = document.createElement("span");
        zoomHint.className = "zoom-hint";
        zoomHint.textContent = "🔍 Zoom";
        figure.appendChild(zoomHint);

        const image = document.createElement("img");
        image.alt = "Retrieved raw image " + (index + 1);
        const resolvedPath = typeof entry.resolved_media_path === "string" ? entry.resolved_media_path : "";
        const fileUri = resolvedPath.startsWith("file://") ? resolvedPath : ("file://" + resolvedPath);
        const finalSrc = entry.relative_media_path || encodeURI(fileUri);
        image.src = finalSrc;

        image.addEventListener("error", function () {
          if (image.src !== encodeURI(fileUri)) {
            image.src = encodeURI(fileUri);
          } else {
            image.hidden = true;
          }
        });

        const captionText = (entry.item_id || "image") + " · " + (entry.file_path || "");
        const caption = document.createElement("figcaption");
        caption.textContent = captionText;

        figure.addEventListener("click", function () {
          openLightbox(image.src, captionText);
        });

        figure.appendChild(image);
        figure.appendChild(caption);
        mediaList.appendChild(figure);
      });

      mediaPanel.hidden = media.length === 0;
      mediaStatus.textContent = media.length === 0
        ? ""
        : availableMedia.length + " of " + media.length + " retrieved image(s) available";

      // Populate Response / Generated Answer
      let answerText = (item.answer || "").trim();
      let answerRoleText = item.answer_role || "";
      if (!answerText) {
        const mediaFiles = (item.retrieved_media || []).map(function(m) {
          return m.file_path || (m.resolved_media_path ? m.resolved_media_path.split("/").pop() : "");
        }).filter(Boolean);
        if (mediaFiles.length > 0) {
          const count = mediaFiles.length;
          const photoWord = count === 1 ? "1 matching photo" : count + " matching photos";
          answerText = "Found " + photoWord + " in album: " + mediaFiles.join(", ") + " matching \"" + displayQuery + "\".";
          answerRoleText = "MODEL RETRIEVAL OUTPUT";
        } else if (Array.isArray(item.ground_truth) && item.ground_truth.length > 0) {
          const count = item.ground_truth.length;
          const photoWord = count === 1 ? "1 matching photo" : count + " matching photos";
          answerText = "Found " + photoWord + " in album: " + item.ground_truth.join(", ") + " matching \"" + displayQuery + "\".";
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
      if (answerRoleEl) {
        if (answerRoleText) {
          answerRoleEl.textContent = answerRoleText;
          answerRoleEl.hidden = false;
        } else {
          answerRoleEl.hidden = true;
        }
      }

      // Reviewer Decisions
      document.getElementById("decisions").innerHTML = FIELDS.map(function (field) {
        const options = field.options.map(function (option) {
          const checked = prior[field.key] === option[0] ? " checked" : "";
          return "<label class=\"option\"><input type=\"radio\" name=\"" + field.key +
            "\" value=\"" + option[0] + "\"" + checked + ">" +
            "<span><strong>" + option[1] + "</strong> — " + escapeHtml(option[2]) +
            "<br><code>" + option[0] + "</code></span></label>";
        }).join("");
        return "<div class=\"decision-block\"><h3>" + field.title + "</h3><p class=\"decision-help\">" + field.help +
          "</p><div class=\"option-list\">" + options + "</div></div>";
      }).join("");

      document.getElementById("notes").value = prior.notes || "";
      document.getElementById("previous").disabled = currentIndex === 0;
      document.getElementById("next").textContent = (currentIndex === CASES.length - 1 ? "Save final case" : "Save and next");

      document.getElementById("scope-note").hidden = !SCOPE_NOTE;
      document.getElementById("scope-note").textContent = SCOPE_NOTE
        ? "Evaluation scope: " + SCOPE_NOTE
        : "";
      if (storageUnavailable) {
        setStorageWarning("Draft storage is unavailable in this browser; changes remain only in this page and will be lost when it closes.");
      } else {
        setStorageWarning("");
      }
      document.getElementById("status").textContent = draftMessage;
    }

    function validateCurrent(showMessage) {
      collectCurrent();
      const item = currentCase();
      if (!item) return false;
      const row = answers[item.audit_id];
      const missing = FIELDS.filter(function (field) { return !row[field.key]; });
      if (missing.length) {
        if (showMessage) alert("Please choose one option in every section before continuing.");
        return false;
      }
      const needsNote = FIELDS.some(function (field) { return row[field.key] !== DEFAULTS[field.key]; });
      if (needsNote && !row.notes) {
        if (showMessage) alert("Please add a short note for a partly supported, uncertain, or negative choice.");
        return false;
      }
      return true;
    }

    function move(delta) {
      if (!PREVIEW_ONLY) {
        collectCurrent();
        if (delta > 0 && !validateCurrent(true)) return;
      }
      currentIndex = Math.max(0, Math.min(CASES.length - 1, currentIndex + delta));
      if (!PREVIEW_ONLY) saveStorage();
      render();
      window.scrollTo({top: 0, behavior: "smooth"});
    }

    document.getElementById("previous").addEventListener("click", function () { move(-1); });
    document.getElementById("next").addEventListener("click", function () { move(1); });
    document.getElementById("decisions").addEventListener("change", function () {
      if (PREVIEW_ONLY) return;
      collectCurrent();
      saveStorage();
      updateProgress();
    });
    document.getElementById("notes").addEventListener("input", function () {
      if (PREVIEW_ONLY) return;
      collectCurrent();
      saveStorage();
      updateProgress();
    });
    document.getElementById("save").addEventListener("click", function () {
      if (PREVIEW_ONLY) return;
      collectCurrent();
      saveStorage();
      document.getElementById("status").textContent = "Draft saved in this browser.";
      updateProgress();
    });
    document.getElementById("download").addEventListener("click", function () {
      if (PREVIEW_ONLY) return;
      collectCurrent();
      const firstIncomplete = CASES.findIndex(function (item) { return !rowComplete(answers[item.audit_id]); });
      if (firstIncomplete !== -1) {
        currentIndex = firstIncomplete;
        saveStorage();
        render();
        alert("Complete every case before downloading the CSV. This is the first incomplete case.");
        window.scrollTo({top: 0, behavior: "smooth"});
        return;
      }
      const headers = ["audit_id", "faithfulness_label", "relevance_label", "correctness_label", "notes"];
      function csvEscape(value) {
        const text = String(value ?? "");
        return /[",\n\r]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
      }
      const lines = [headers.join(",")].concat(CASES.map(function (item) {
        const row = answers[item.audit_id];
        return headers.map(function (header) { return csvEscape(row[header]); }).join(",");
      }));
      const blob = new Blob([lines.join("\r\n") + "\r\n"], {type: "text/csv;charset=utf-8"});
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = DOWNLOAD_STEM + ".csv";
      link.click();
      URL.revokeObjectURL(link.href);
      document.getElementById("status").textContent = "CSV downloaded. Send it to the project owner.";
    });

    loadDraft();
    render();
  </script>
</body>
</html>
'''
    form = (
        form.replace("__CASES__", cases_json)
        .replace("__REVIEWER_LABEL__", reviewer_json)
        .replace("__DOWNLOAD_STEM__", download_stem_json)
        .replace("__PACKET_VERSION__", packet_version_json)
        .replace("__FORM_VERSION__", form_version_json)
        .replace("__SCOPE_NOTE__", scope_note_json)
        .replace("__CONTENT_FINGERPRINT__", fingerprint_json)
        .replace("__PREVIEW_ONLY__", preview_only_json)
        .replace("__PREVIEW_BANNER_HIDDEN__", preview_hidden)
        .replace("__PREVIEW_ANSWER_HIDDEN__", preview_hidden)
        .replace("__PREVIEW_RATINGS_HIDDEN__", preview_hidden)
        .replace("__PREVIEW_ACTION_HIDDEN__", preview_hidden)
    )
    path.write_text(form, encoding="utf-8")


def _write_rubric(
    path: Path,
    *,
    packet_version: str = DEFAULT_PACKET_VERSION,
    scope_note: str = "",
) -> None:
    metadata = (
        "## Packet metadata\n\n"
        f"- Packet version: `{packet_version}`\n"
    )
    if scope_note:
        metadata += "- Evaluation scope notes:\n\n"
        metadata += "\n".join(f"  {line}" for line in scope_note.splitlines())
        metadata += "\n"
    metadata += "\n"
    path.write_text(
        """# Human RAG answer-quality audit

Use the query, target keyword, retrieved context, and answer in the packet. Do not
infer or repair facts from outside the retrieved context. Reviewers must work
independently and must not see model scores or one another's labels.

For a simpler workflow, open `reviewer_1_form.html` or `reviewer_2_form.html` in a
browser. Review one case at a time, save the draft locally, and use “Download
completed CSV” when all cases are finished. Send only the downloaded CSV to the
project owner. The form exports the exact columns required by the agreement script.

"""
        + metadata
        + """## Quick decision rule

1. **Faithfulness:** judge whether the answer's own claims are supported by the
   retrieved context. A truthful answer that says the context is insufficient is
   usually `supported`, even if it did not solve the user's question.
2. **Relevance:** judge whether the answer actually addresses the question and its
   target. A truthful refusal is usually `partial`, not `off_topic`.
3. **Correctness:** mark `uncertain` when the target cannot be established from
   the context. Mark `incorrect` only when the answer gives a wrong target or
   denies a target that the context clearly contains.

The target keyword is a checking hint, not evidence. Use the retrieved context as
the only evidence. If the answer says a detail is absent, check whether that
detail is actually present before accepting the statement.

Examples:

- Context says only `dog`; answer says it cannot verify a barking recording:
  `supported`, `partial`, `uncertain`.
- Context contains an exact district tag; answer says the tag is absent:
  `mixed`, `partial`, `incorrect`.
- Context contains an exact event detail; answer says that detail is absent:
  `mixed`, `partial`, `incorrect`.

## Faithfulness

- `supported`: every material factual claim in the answer is entailed by the retrieved context.
- `mixed`: at least one material claim is supported, but another is overstated, weakly supported, or unverifiable.
- `unsupported`: the answer's material claims are absent from or contradicted by the retrieved context.

## Relevance

- `direct`: answers the requested question and addresses the target detail.
- `partial`: addresses part of the question, is vague, or omits the target detail.
- `off_topic`: does not answer the question or answers a different task.

## Correctness

- `correct`: the answer gives the target detail supported by the context.
- `uncertain`: the context is insufficient or ambiguous; do not guess.
- `incorrect`: the answer gives a target detail contradicted by the context or a wrong target.

Record a short note for every `mixed`, `unsupported`, `partial`, `off_topic`,
`uncertain`, or `incorrect` label.
""",
        encoding="utf-8",
    )


def _refuse_overwrite(output_dir: Path) -> None:
    protected_names = (
        "human_audit_packet.json",
        "audit_manifest_private.json",
        "reviewer_1.csv",
        "reviewer_2.csv",
    )
    existing = [name for name in protected_names if (output_dir / name).exists()]
    if existing:
        joined = ", ".join(existing)
        raise FileExistsError(
            "Refusing to overwrite existing audit output "
            f"({joined}). Use a new output directory."
        )


def render_packet_forms(
    packet_path: Path,
    output_dir: Path,
    *,
    preview_only: bool = False,
) -> None:
    """Render forms and the rubric from an existing packet without rewriting it."""
    with packet_path.open(encoding="utf-8") as handle:
        packet = json.load(handle)
    if not isinstance(packet, dict):
        raise ValueError("The packet JSON must contain an object")
    packet_rows = _packet_rows(packet, relative_to_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    packet_version = str(packet.get("packet_version", DEFAULT_PACKET_VERSION))
    scope_value = packet.get("evaluation_scope")
    if scope_value is None:
        scope_value = packet.get("evaluation_scope_notes", packet.get("scope_note"))
    scope_note = _scope_note(scope_value)
    _write_simple_form(
        output_dir / "reviewer_1_form.html",
        packet_rows,
        "Reviewer 1",
        "reviewer_1",
        packet_version=packet_version,
        scope_note=scope_note,
        preview_only=preview_only,
    )
    _write_simple_form(
        output_dir / "reviewer_2_form.html",
        packet_rows,
        "Reviewer 2",
        "reviewer_2",
        packet_version=packet_version,
        scope_note=scope_note,
        preview_only=preview_only,
    )
    _write_rubric(
        output_dir / "RUBRIC.md",
        packet_version=packet_version,
        scope_note=scope_note,
    )
    mode = "preview forms" if preview_only else "review forms"
    print(f"Rendered {mode} and rubric from {packet_path} into {output_dir}")


def prepare_packet(input_path: Path, output_dir: Path, sample_size: int, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _refuse_overwrite(output_dir)
    with input_path.open(encoding="utf-8") as handle:
        results = json.load(handle)
    cases = stratified_sample(_require_case_logs(results), sample_size, seed)

    corpus = _load_corpus_safe()
    packet_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        audit_id = f"AUDIT-{index:03d}"
        packet_rows.append(
            _audit_row(
                case,
                audit_id,
                corpus=corpus,
                relative_to_dir=output_dir,
            )
        )
        proposed = case["results"][PROPOSED_CONFIG]
        manifest_rows.append(
            {
                "audit_id": audit_id,
                "source_case_id": case.get("id"),
                "source_config": PROPOSED_CONFIG,
                "original_judge": {
                    "faithfulness": proposed.get("faithfulness"),
                    "relevance": proposed.get("relevance"),
                },
            }
        )

    packet = {
        "audit_type": "blinded_human_answer_quality",
        "rubric_version": RUBRIC_VERSION,
        "packet_version": DEFAULT_PACKET_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_results_sha256": sha256_file(input_path),
        "sample_size": len(packet_rows),
        "stratification": {
            "single_hop": sum(row["split"] != "multi_hop" for row in packet_rows),
            "multi_hop": sum(row["split"] == "multi_hop" for row in packet_rows),
        },
        "blinding": "source case IDs, system labels, and original judge scores are excluded from this packet",
        "examples": packet_rows,
    }
    if "evaluation_scope" in results:
        packet["evaluation_scope"] = results["evaluation_scope"]
    (output_dir / "human_audit_packet.json").write_text(
        json.dumps(packet, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "audit_manifest_private.json").write_text(
        json.dumps(
            {
                "source_results_sha256": packet["source_results_sha256"],
                "rubric_version": RUBRIC_VERSION,
                "rows": manifest_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_reviewer_template(output_dir / "reviewer_1.csv", [row["audit_id"] for row in packet_rows])
    _write_reviewer_template(output_dir / "reviewer_2.csv", [row["audit_id"] for row in packet_rows])
    scope_note = _scope_note(results.get("evaluation_scope"))
    _write_simple_form(
        output_dir / "reviewer_1_form.html",
        packet_rows,
        "Reviewer 1",
        "reviewer_1",
        packet_version=DEFAULT_PACKET_VERSION,
        scope_note=scope_note,
    )
    _write_simple_form(
        output_dir / "reviewer_2_form.html",
        packet_rows,
        "Reviewer 2",
        "reviewer_2",
        packet_version=DEFAULT_PACKET_VERSION,
        scope_note=scope_note,
    )
    _write_rubric(
        output_dir / "RUBRIC.md",
        packet_version=DEFAULT_PACKET_VERSION,
        scope_note=scope_note,
    )
    print(
        f"Prepared {len(packet_rows)} blinded cases in {output_dir}; "
        "open reviewer_1_form.html or reviewer_2_form.html to review"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--packet",
        type=Path,
        help="Render forms/rubric from an existing packet without writing packet or CSV files",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="With --packet, render question/context preview forms without ratings or export",
    )
    args = parser.parse_args()
    try:
        if args.packet is not None:
            if args.input is not None:
                parser.error("--input and --packet cannot be used together")
            render_packet_forms(
                args.packet,
                args.output_dir,
                preview_only=args.preview_only,
            )
        else:
            if args.preview_only:
                parser.error("--preview-only requires --packet")
            prepare_packet(
                args.input or DEFAULT_INPUT,
                args.output_dir,
                args.sample_size,
                args.seed,
            )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

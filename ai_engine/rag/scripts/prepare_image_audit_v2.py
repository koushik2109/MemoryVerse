"""Prepare an image-description human-audit packet (v2).

This module audits only the image rows in ``unified_rag_dataset.jsonl``.  The
corpus also contains ESC-50 category-only audio metadata, but those rows are
not recordings and are never included here.  Image rows retain their text
annotation proxy and symbolic ``file_path``; when the referenced raw image is
available, it is resolved and attached to the vision request as additional
evidence.  Missing media is reported explicitly and falls back to the
text-proxy-only behavior.

The 25 questions are an annotation-derived design set, not an independent
held-out evaluation.  Each question is a single-hop lookup of one field in an
exact source-proxy phrase.  The question specification keeps the source row,
zero-based corpus ``doc_idx``, exact support, and expected detail as design
references separate from the preview and generated answer records.

Preview is the default and makes no model/API calls.  ``--generate-answers``
is intentionally opt-in; it uses the existing image-row retrieval helpers,
attaches any resolved raw images to the repository's configured OpenAI client,
checkpoints only real responses, and refuses to overwrite final audit artifacts
or CSVs.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import json
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS_PATH = REPO_ROOT / "ai_engine" / "rag" / "data" / "unified_rag_dataset.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "ai_engine" / "rag" / "data" / "human_audit_image_v2"
DEFAULT_MEDIA_ROOT = REPO_ROOT / "ai_engine" / "rag"

PACKET_VERSION = "image-descriptions-v2"
GENERATOR_MODEL = "gpt-4o-mini"
DENSE_MODEL = "all-MiniLM-L6-v2"
RETRIEVAL_TOP_K = 10
CONTEXT_DEPTH = 5
LLM_SEED = 42
MAX_IMAGE_BYTES = 10 * 1024 * 1024


try:
    from ai_engine.rag.image_media import (
        build_image_media_payload,
        resolve_media_path,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ai_engine.rag.image_media import (
        build_image_media_payload,
        resolve_media_path,
    )

SCOPE_NOTE = (
    "Annotation-derived single-hop design over image text proxies, with "
    "optional raw-image attachments when the referenced files are available. The "
    "3,356-row corpus contains 2,000 ESC-50 category-only audio metadata rows "
    "and 1,356 image annotation/text-proxy rows. Audio files/recordings are not "
    "used here. If raw image files are absent, the affected cases remain "
    "text-proxy-only and are labelled as such. Audio labels are not presented as "
    "real recordings, and image descriptions are not treated as raw-image "
    "understanding unless an image is actually attached. These questions are not "
    "independent held-out evaluation items."
)

DESIGN_REFERENCE_LABEL = "DESIGN REFERENCE — not a generated model output"


class OutputConflict(RuntimeError):
    """Raised when a requested artifact would overwrite an existing artifact."""


class GenerationFailure(RuntimeError):
    """Raised for a safe, non-secret model-generation failure."""


# The source rows below were selected from the first image rows and additional
# image rows later in the corpus.  Every support phrase is checked against the
# selected row's text_representation at build time; no metadata or cross-modal
# association is used as evidence.
_QUESTION_BLUEPRINTS: Tuple[Dict[str, Any], ...] = (
    {
        "id": "IMG-V2-001",
        "query": "According to the image annotation, what date is associated with the marathon volunteers handing water?",
        "field": "date",
        "target_keyword": "December 17, 2023",
        "expected_detail": "December 17, 2023",
        "source_item_id": "IMG_8266",
        "support_phrases": ("marathon volunteers handing water", "December 17, 2023"),
        "qualification": "Use the date phrase in the annotation; this is not a verified raw-image timestamp.",
    },
    {
        "id": "IMG-V2-002",
        "query": "What time period is explicitly named in the image annotation for the Putuo Mountain coast photos?",
        "field": "date",
        "target_keyword": "first half of 2024",
        "expected_detail": "first half of 2024",
        "source_item_id": "IMG_2457",
        "support_phrases": ("Putuo Mountain coast", "first half of 2024"),
        "qualification": "Report the stated period only; do not infer a more precise capture date.",
    },
    {
        "id": "IMG-V2-003",
        "query": "What action is described for the image annotation about the bridge at sunset?",
        "field": "described_action",
        "target_keyword": "driving over bridge at sunset",
        "expected_detail": "driving over bridge at sunset",
        "source_item_id": "IMG_3326",
        "support_phrases": ("driving over bridge at sunset",),
        "qualification": "Use the described action and setting, not visual inference from an unavailable image.",
    },
    {
        "id": "IMG-V2-004",
        "query": "What action is described in the party image annotation?",
        "field": "described_action",
        "target_keyword": "friends toasting at party",
        "expected_detail": "friends toasting at party",
        "source_item_id": "IMG_3076",
        "support_phrases": ("friends toasting at party",),
        "qualification": "Return the action phrase supplied by the annotation, without adding identities or objects.",
    },
    {
        "id": "IMG-V2-005",
        "query": "What celebration detail is explicitly named in the birthday image annotation?",
        "field": "described_object",
        "target_keyword": "22nd birthday celebration",
        "expected_detail": "22nd birthday celebration",
        "source_item_id": "IMG_3853(1)",
        "support_phrases": ("22nd birthday celebration",),
        "qualification": "Report the named celebration detail; do not infer a person's identity or exact birthday date.",
    },
    {
        "id": "IMG-V2-006",
        "query": "Which place tag is explicitly present in the annotation for the toasting photo?",
        "field": "place",
        "target_keyword": "Qiantang",
        "expected_detail": "Qiantang",
        "source_item_id": "IMG_4438",
        "support_phrases": ("Qiantang",),
        "qualification": "Treat Qiantang as the annotation's place tag, not as an independently inferred geographic location.",
    },
    {
        "id": "IMG-V2-007",
        "query": "What object is described in the art image annotation?",
        "field": "described_object",
        "target_keyword": "art studio wall decor",
        "expected_detail": "art studio wall decor",
        "source_item_id": "IMG_4903",
        "support_phrases": ("art studio wall decor",),
        "qualification": "Use the supplied object/setting phrase only; do not describe unobserved visual details.",
    },
    {
        "id": "IMG-V2-008",
        "query": "Which place name is explicitly included in the quiet-beach image annotation?",
        "field": "place",
        "target_keyword": "Phuket's Karon",
        "expected_detail": "Phuket's Karon",
        "source_item_id": "IMG_4595",
        "support_phrases": ("Phuket's Karon",),
        "qualification": "Return the place wording in the proxy; do not infer a broader region or exact coordinates.",
    },
    {
        "id": "IMG-V2-009",
        "query": "What drink is explicitly described in the fancy-drink image annotation?",
        "field": "described_object",
        "target_keyword": "layered cocktail",
        "expected_detail": "layered cocktail",
        "source_item_id": "IMG_5180",
        "support_phrases": ("layered cocktail",),
        "qualification": "Report the named drink only; do not infer ingredients or preparation.",
    },
    {
        "id": "IMG-V2-010",
        "query": "Which place is explicitly named for the driverless-car image annotation?",
        "field": "place",
        "target_keyword": "Pingyao",
        "expected_detail": "Pingyao",
        "source_item_id": "IMG_2668",
        "support_phrases": ("driverless car in Pingyao", "Pingyao"),
        "qualification": "Use Pingyao as the named place in the proxy; do not infer where the car was geographically beyond that phrase.",
    },
    {
        "id": "IMG-V2-011",
        "query": "What material is explicitly named for the figurine in the image annotation?",
        "field": "material",
        "target_keyword": "ceramic",
        "expected_detail": "ceramic",
        "source_item_id": "IMG_9087",
        "support_phrases": ("cute ceramic figurine", "ceramic"),
        "qualification": "Answer the material as ceramic; do not identify the figurine's character or rely on raw-image inspection.",
    },
    {
        "id": "IMG-V2-012",
        "query": "What state is explicitly described for the sculpture annotation?",
        "field": "described_action",
        "target_keyword": "handmade sculpture before painting",
        "expected_detail": "handmade sculpture before painting",
        "source_item_id": "IMG_9088",
        "support_phrases": ("handmade sculpture before painting",),
        "qualification": "Report the stated pre-painting condition; do not infer the sculpture's character, color, or material.",
    },
    {
        "id": "IMG-V2-013",
        "query": "What object is explicitly described in the couple-photo annotation?",
        "field": "described_object",
        "target_keyword": "matching rings",
        "expected_detail": "matching rings",
        "source_item_id": "IMG_7951",
        "support_phrases": ("matching rings",),
        "qualification": "Return the named object only; do not infer relationship details beyond the proxy wording.",
    },
    {
        "id": "IMG-V2-014",
        "query": "What snack is explicitly described next to the deer statue?",
        "field": "described_object",
        "target_keyword": "chocolate popsicle",
        "expected_detail": "chocolate popsicle",
        "source_item_id": "IMG_2666",
        "support_phrases": ("chocolate popsicle", "cartoon deer statue"),
        "qualification": "Use the named snack in the proxy; do not infer its size, brand, or visual appearance.",
    },
    {
        "id": "IMG-V2-015",
        "query": "What food is explicitly described in the noodle image annotation?",
        "field": "described_object",
        "target_keyword": "handmade noodles",
        "expected_detail": "handmade noodles",
        "source_item_id": "IMG_0428",
        "support_phrases": ("handmade noodles",),
        "qualification": "Report the named food only; do not infer ingredients or preparation steps.",
    },
    {
        "id": "IMG-V2-016",
        "query": "What hand gesture is explicitly described with the dry-pot dish?",
        "field": "described_action",
        "target_keyword": "peace sign",
        "expected_detail": "peace sign",
        "source_item_id": "IMG_0403",
        "support_phrases": ("dry pot dish with peace sign", "peace sign"),
        "qualification": "Return the described gesture; do not infer who made it or inspect an unavailable image.",
    },
    {
        "id": "IMG-V2-017",
        "query": "Which place is explicitly named in the New Year's Eve image annotation?",
        "field": "place",
        "target_keyword": "Hangzhou",
        "expected_detail": "Hangzhou",
        "source_item_id": "IMG_8684",
        "support_phrases": ("New Year’s Eve Hangzhou", "Hangzhou"),
        "qualification": "Return the place word in the annotation; do not infer a venue or event location beyond it.",
    },
    {
        "id": "IMG-V2-018",
        "query": "What month and year are explicitly named in the concert-support image annotation?",
        "field": "date",
        "target_keyword": "April 2023",
        "expected_detail": "April 2023",
        "source_item_id": "IMG_1148",
        "support_phrases": ("April 2023",),
        "qualification": "Report the month and year phrase; do not infer a specific day.",
    },
    {
        "id": "IMG-V2-019",
        "query": "Which district tag is explicitly present in the Ningbo night-bridge image annotation?",
        "field": "district",
        "target_keyword": "Yinzhou District",
        "expected_detail": "Yinzhou District",
        "source_item_id": "IMG_1438",
        "support_phrases": ("Yinzhou District",),
        "qualification": "Answer the explicit district tag, not a geographic inference from the image description.",
    },
    {
        "id": "IMG-V2-020",
        "query": "Which place is explicitly named in the night boat-ride image annotation?",
        "field": "place",
        "target_keyword": "West Lake",
        "expected_detail": "West Lake",
        "source_item_id": "IMG_2277 2",
        "support_phrases": ("West Lake night boat ride", "West Lake"),
        "qualification": "Return the named place only; do not infer the boat route or surrounding geography.",
    },
    {
        "id": "IMG-V2-021",
        "query": "Which district tag is explicitly present in this night-installation image annotation?",
        "field": "district",
        "target_keyword": "Binjiang District",
        "expected_detail": "Binjiang District",
        "source_item_id": "IMG_8300",
        "support_phrases": ("Binjiang District",),
        "qualification": "Treat this as an explicit district tag, not as a location inferred from visual content.",
    },
    {
        "id": "IMG-V2-022",
        "query": "What place name is explicitly present in the subway-at-sunset image annotation?",
        "field": "place",
        "target_keyword": "Hangzhou Qingchun Road",
        "expected_detail": "Hangzhou Qingchun Road",
        "source_item_id": "IMG_1518",
        "support_phrases": ("Hangzhou Qingchun Road",),
        "qualification": "Return the place wording in the proxy; do not infer a station entrance or exact coordinates.",
    },
    {
        "id": "IMG-V2-023",
        "query": "What activity is described in the April 2023 image annotation?",
        "field": "described_action",
        "target_keyword": "concert fan support moments",
        "expected_detail": "concert fan support moments",
        "source_item_id": "IMG_1051",
        "support_phrases": ("concert fan support moments",),
        "qualification": "Report the described activity; do not infer the performer, venue, or audience identities.",
    },
    {
        "id": "IMG-V2-024",
        "query": "What place is explicitly named in the riverside image annotation?",
        "field": "place",
        "target_keyword": "Anqiao Port",
        "expected_detail": "Anqiao Port",
        "source_item_id": "IMG_5016",
        "support_phrases": ("Anqiao Port",),
        "qualification": "Use the named place phrase; do not infer a broader district or waterway location.",
    },
    {
        "id": "IMG-V2-025",
        "query": "What date is explicitly named in the black-pigeon image annotation?",
        "field": "date",
        "target_keyword": "November 15, 2025",
        "expected_detail": "November 15, 2025",
        "source_item_id": "IMG_1619",
        "support_phrases": ("November 15, 2025",),
        "qualification": "Report the date phrase in the annotation; it is not independently verified file metadata.",
    },
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _json_bytes(value: Any, *, sort_keys: bool = False) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=sort_keys)
        + "\n"
    ).encode("utf-8")


def _hash_json(value: Any) -> str:
    return _sha256_bytes(_json_bytes(value, sort_keys=True))


def _load_corpus(path: Path) -> Tuple[List[Dict[str, Any]], str, Dict[str, int]]:
    """Load JSONL records without writing or changing the source file."""
    corpus: List[Dict[str, Any]] = []
    digest = hashlib.sha256()
    counts = {"total_rows": 0, "image_rows": 0, "audio_rows": 0, "other_rows": 0}

    with path.open("rb") as handle:
        for doc_idx, raw_line in enumerate(handle):
            digest.update(raw_line)
            if not raw_line.strip():
                raise ValueError(f"Blank corpus line at zero-based doc_idx {doc_idx}")
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid JSONL corpus row at doc_idx {doc_idx}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Corpus row at doc_idx {doc_idx} is not an object")
            # doc_idx is added only to the in-memory copy used for retrieval and
            # references; the corpus JSONL is never modified.
            loaded = dict(record)
            loaded["doc_idx"] = doc_idx
            corpus.append(loaded)
            counts["total_rows"] += 1
            modality = record.get("modality")
            if modality == "image":
                counts["image_rows"] += 1
            elif modality == "audio":
                counts["audio_rows"] += 1
            else:
                counts["other_rows"] += 1

    if not corpus:
        raise ValueError("The diagnostic corpus is empty")
    return corpus, digest.hexdigest(), counts


def _relative_source_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _resolve_source_record(
    corpus: Sequence[Mapping[str, Any]], blueprint: Mapping[str, Any]
) -> Dict[str, Any]:
    item_id = str(blueprint["source_item_id"])
    phrases = tuple(str(value) for value in blueprint["support_phrases"])
    matches = [
        dict(record)
        for record in corpus
        if record.get("modality") == "image"
        and str(record.get("item_id", "")) == item_id
        and all(phrase in str(record.get("text_representation", "")) for phrase in phrases)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Question {blueprint['id']} source did not resolve uniquely from image rows"
        )
    source = matches[0]
    if not isinstance(source.get("doc_idx"), int) or source["doc_idx"] < 0:
        raise ValueError(f"Question {blueprint['id']} source lacks its original doc_idx")
    source_text = str(source.get("text_representation", ""))
    if not source_text:
        raise ValueError(f"Question {blueprint['id']} source has no text proxy")
    # Recheck exact support against the final source string.  This intentionally
    # does not normalize or broaden phrases.
    for phrase in phrases:
        if phrase not in source_text:
            raise ValueError(f"Question {blueprint['id']} has unsupported source phrase")
    return source


def _make_question_specification(
    blueprint: Mapping[str, Any], source: Mapping[str, Any]
) -> Dict[str, Any]:
    source_text = str(source["text_representation"])
    doc_idx = int(source["doc_idx"])
    exact_phrases = [str(value) for value in blueprint["support_phrases"]]
    support_reference = {
        "doc_idx": doc_idx,
        "item_id": str(source["item_id"]),
        "modality": "image",
        "source_proxy_type": "annotation/text-proxy",
        "source_proxy": source_text,
        # The complete proxy is retained as an exact snippet, while the
        # separately listed phrases identify the one supported detail.
        "support_snippets": [source_text],
        "exact_support_phrases": exact_phrases,
        "raw_media_available": False,
    }
    return {
        "id": str(blueprint["id"]),
        "query": str(blueprint["query"]),
        "split": "single_hop",
        "modality": "image",
        "field": str(blueprint["field"]),
        "source_doc_idx": doc_idx,
        "source_reference": {
            "doc_idx": doc_idx,
            "item_id": str(source["item_id"]),
            "modality": "image",
        },
        "support_references": [support_reference],
        "support_snippets": [source_text],
        "exact_support_phrases": exact_phrases,
        "expected_detail": str(blueprint["expected_detail"]),
        "expected_detail_role": DESIGN_REFERENCE_LABEL,
        "expected_answer": str(blueprint["expected_detail"]),
        "expected_answer_role": DESIGN_REFERENCE_LABEL,
        "qualification": str(blueprint["qualification"]),
        "source_proxy_only": True,
        "raw_media_available": False,
    }


def build_question_set(corpus_path: Path) -> dict:
    """Build the deterministic 25-question image-description specification.

    The returned object contains public question records plus a separate
    ``question_specification`` section with exact support and design-reference
    answers.  It never calls a model and never writes the corpus.
    """
    corpus_path = Path(corpus_path)
    corpus, corpus_sha256, counts = _load_corpus(corpus_path)
    if len(_QUESTION_BLUEPRINTS) != 25:
        raise RuntimeError("The image-description v2 design must contain 25 questions")

    specifications: List[Dict[str, Any]] = []
    questions: List[Dict[str, Any]] = []
    used_doc_indices = set()
    for blueprint in _QUESTION_BLUEPRINTS:
        source = _resolve_source_record(corpus, blueprint)
        spec = _make_question_specification(blueprint, source)
        doc_idx = int(spec["source_reference"]["doc_idx"])
        if doc_idx in used_doc_indices:
            raise RuntimeError("The v2 design unexpectedly reuses a source corpus row")
        used_doc_indices.add(doc_idx)
        specifications.append(spec)
        questions.append(
            {
                "id": str(blueprint["id"]),
                "audit_id": str(blueprint["id"]),
                "query": str(blueprint["query"]),
                # This is a checking hint/design reference, never a model input.
                "target_keyword": str(blueprint["target_keyword"]),
                "target_keyword_role": DESIGN_REFERENCE_LABEL,
                "split": "single_hop",
                "modality": "image",
                "question_type": str(blueprint["field"]),
                "specification_id": str(blueprint["id"]),
            }
        )

    scope_counts = {
        **counts,
        "selected_image_rows": len(specifications),
        "question_count": len(questions),
        "raw_media_available": False,
    }
    return {
        "packet_version": PACKET_VERSION,
        "annotation_derived_design": True,
        "independent_held_out_evaluation": False,
        "scope_note": SCOPE_NOTE,
        "source_corpus_path": _relative_source_path(corpus_path),
        "source_corpus_sha256": corpus_sha256,
        "corpus_sha256": corpus_sha256,
        "scope_counts": scope_counts,
        "questions": questions,
        "question_specification": specifications,
        "expected_answers_are_design_references": True,
        "audio_scope": {
            "audio_rows_in_source": counts["audio_rows"],
            "audio_files_available": False,
            "audio_recordings_used": 0,
            "audio_labels_are_recordings": False,
        },
        "raw_media_scope": {
            "available": False,
            "image_understanding_claim": False,
            "audio_understanding_claim": False,
        },
    }


def build_preview(question_set: dict) -> dict:
    """Build a navigation/review preview with no model results.

    The required ``retrieved_context`` field contains source-proxy evidence for
    design inspection only.  Each entry is explicitly labelled as a design
    reference and is not a retrieval result.
    """
    if question_set.get("packet_version") != PACKET_VERSION:
        raise ValueError("question_set has an unsupported packet version")
    questions = question_set.get("questions")
    specifications = question_set.get("question_specification")
    if not isinstance(questions, list) or not isinstance(specifications, list):
        raise ValueError("question_set is missing questions or question_specification")
    spec_by_id = {str(spec.get("id")): spec for spec in specifications if isinstance(spec, dict)}
    examples: List[Dict[str, Any]] = []
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("question_set contains a non-object question")
        question_id = str(question.get("id", ""))
        spec = spec_by_id.get(question_id)
        if spec is None:
            raise ValueError(f"No specification found for {question_id}")
        supports = spec.get("support_references")
        if not isinstance(supports, list) or not supports:
            raise ValueError(f"No support reference found for {question_id}")
        contexts: List[str] = []
        for support in supports:
            if not isinstance(support, dict):
                raise ValueError(f"Invalid support reference for {question_id}")
            snippets = support.get("support_snippets")
            if not isinstance(snippets, list):
                raise ValueError(f"Invalid support snippets for {question_id}")
            for snippet in snippets:
                contexts.append(
                    "DESIGN REFERENCE — source proxy evidence only; not a retrieval "
                    "result and not raw-media understanding: " + str(snippet)
                )
        examples.append(
            {
                "audit_id": question_id,
                "query": str(question["query"]),
                "target_keyword": str(question["target_keyword"]),
                "target_keyword_role": DESIGN_REFERENCE_LABEL,
                "split": "single_hop",
                "retrieved_context": contexts,
                "answer": "",
            }
        )
    return {
        "packet_version": PACKET_VERSION,
        "preview_only": True,
        "annotation_derived_design": True,
        "scope_note": str(question_set["scope_note"]),
        "source_corpus_sha256": str(question_set["source_corpus_sha256"]),
        "scope_counts": dict(question_set["scope_counts"]),
        "examples": examples,
        "results": [],
    }


def _safe_create_or_verify(path: Path, content: bytes) -> None:
    """Create an artifact, or accept it only when it is byte-identical."""
    if path.exists():
        if path.is_file() and path.read_bytes() == content:
            return
        raise OutputConflict(f"Refusing to overwrite existing artifact: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.is_file() and path.read_bytes() == content:
            return
        raise OutputConflict(f"Refusing to overwrite existing artifact: {path.name}")


def _navigation_html(preview: Mapping[str, Any]) -> bytes:
    examples = preview.get("examples", [])
    links = []
    for example in examples:
        if isinstance(example, dict):
            links.append(
                "<li><code>"
                + html.escape(str(example.get("audit_id", "")))
                + "</code> &mdash; "
                + html.escape(str(example.get("query", "")))
                + "</li>"
            )
    body = "\n".join(links)
    scope = html.escape(str(preview.get("scope_note", "")))
    return (
        "<!doctype html>\n"
        "<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>MemoryVerse image descriptions v2 preview</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:960px;margin:40px auto;padding:0 20px;color:#182230}"
        "nav a{margin-right:18px}li{margin:8px 0}.notice{padding:14px;border-left:4px solid #2474c6;background:#eef6ff}"
        "code{color:#526274}</style></head><body>"
        "<h1>MemoryVerse image-descriptions-v2</h1>"
        "<p class=\"notice\"><strong>Preview only.</strong> This page is navigation-only; it contains no model results.</p>"
        "<p><strong>packet_version:</strong> image-descriptions-v2 &nbsp; "
        "<strong>preview_only:</strong> true</p>"
        "<p>"
        "<a href=\"question_set.json\">Question specification JSON</a> &nbsp;"
        "<a href=\"question_preview.json\">Question preview JSON</a>"
        "</p>"
        "<p>"
        + scope
        + "</p><h2>Questions</h2><ol>"
        + body
        + "</ol></body></html>\n"
    ).encode("utf-8")


def _legacy_simple_form_writer():
    """Import the existing writer in both package and direct-script modes."""
    try:
        from .prepare_human_audit import _write_simple_form
    except ImportError:
        from prepare_human_audit import _write_simple_form
    return _write_simple_form


def _call_preview_form_writer(
    writer: Any,
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    reviewer_label: str,
    download_stem: str,
) -> None:
    """Call the current form writer with v2 metadata when supported.

    The checked-in legacy writer predates the preview metadata keyword
    arguments.  Passing them first keeps this call compatible with an updated
    writer; the narrow fallback keeps direct use with the current writer
    working without editing that legacy file.
    """
    try:
        writer(
            path,
            rows,
            reviewer_label,
            download_stem,
            packet_version=PACKET_VERSION,
            scope_note=SCOPE_NOTE,
            preview_only=True,
        )
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        writer(path, rows, reviewer_label, download_stem)


def _preview_form_bytes(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    reviewer_label: str,
    download_stem: str,
) -> bytes:
    """Render a preview form in a temporary file without touching its target."""
    writer = _legacy_simple_form_writer()
    with tempfile.TemporaryDirectory(
        prefix=".image-v2-preview-form-", dir=str(output_dir)
    ) as temporary_dir:
        temporary_path = Path(temporary_dir) / (download_stem + ".html")
        _call_preview_form_writer(
            writer,
            temporary_path,
            rows,
            reviewer_label,
            download_stem,
        )
        return temporary_path.read_bytes()


def _write_preview_forms(output_dir: Path, preview: Mapping[str, Any]) -> None:
    """Create or byte-verify inspectable reviewer preview forms only."""
    rows = preview.get("examples", [])
    for reviewer_label, filename, download_stem in (
        (
            "Reviewer 1 · image-descriptions-v2 preview · no model results",
            "reviewer_1_preview.html",
            "reviewer_1_preview",
        ),
        (
            "Reviewer 2 · image-descriptions-v2 preview · no model results",
            "reviewer_2_preview.html",
            "reviewer_2_preview",
        ),
    ):
        content = _preview_form_bytes(
            output_dir,
            rows,
            reviewer_label,
            download_stem,
        )
        _safe_create_or_verify(output_dir / filename, content)


def _write_preview_index(path: Path, preview: Mapping[str, Any]) -> None:
    """Use the legacy writer compatibility hook, then retain navigation-only HTML.

    The checked-in legacy helper currently accepts four positional arguments.
    A future-compatible helper may accept the requested preview metadata, so we
    first pass ``packet_version``, ``scope_note``, and ``preview_only`` and fall
    back only for the current signature.  Its audit-form output is immediately
    replaced by the deterministic navigation-only page required for this v2
    preview; no legacy audit is modified.
    """
    expected = _navigation_html(preview)
    if path.exists():
        if path.is_file() and path.read_bytes() == expected:
            return
        raise OutputConflict(f"Refusing to overwrite existing artifact: {path.name}")

    rows = preview.get("examples", [])
    writer = _legacy_simple_form_writer()
    _call_preview_form_writer(
        writer,
        path,
        rows,
        "Image-description v2 preview",
        "image_descriptions_v2_preview",
    )
    path.write_bytes(expected)


def _generation_config(
    runtime: Mapping[str, Any], media_root: Path = DEFAULT_MEDIA_ROOT
) -> Dict[str, Any]:
    return {
        "image_only": True,
        "retrieval_helper": "run_expanded_multihop_eval.rank_candidates",
        "retrieval_mode": "full",
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "context_depth": CONTEXT_DEPTH,
        "dense_model": DENSE_MODEL,
        "bm25_index": "rank_bm25.BM25Okapi",
        "tie_break": "ascending original zero-based corpus doc_idx",
        "rrf_k": int(runtime.get("RRF_K", 60)),
        "rrf_depth": int(runtime.get("RRF_DEPTH", 100)),
        "candidate_depth": int(runtime.get("CANDIDATE_DEPTH", 30)),
        "rerank_dense_weight": float(runtime.get("RERANK_DENSE_WEIGHT", 0.65)),
        "generator_model": GENERATOR_MODEL,
        "temperature": 0.0,
        "max_tokens": 200,
        "llm_seed": LLM_SEED,
        "prompt_builder": "run_expanded_multihop_eval.build_answer_prompt",
        "raw_media": {
            "resolution": "image row file_path resolved under media_root",
            "media_root": str(media_root.resolve()),
            "attached_to_vision_request": True,
            "max_image_bytes": MAX_IMAGE_BYTES,
        },
    }


_INDEX_CACHE: Dict[str, Any] = {}


def _build_or_get_image_index(
    corpus: Sequence[Mapping[str, Any]], corpus_sha256: str, runtime: Mapping[str, Any]
) -> Any:
    """Build one in-memory BM25/dense index and reuse it for all 25 queries."""
    cache_key = corpus_sha256 + ":image-only:" + DENSE_MODEL
    if cache_key in _INDEX_CACHE:
        return _INDEX_CACHE[cache_key]
    try:
        import numpy as np
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
    except Exception as error:
        raise GenerationFailure("retrieval dependencies are unavailable") from error

    image_corpus = [dict(record) for record in corpus if record.get("modality") == "image"]
    if not image_corpus:
        raise GenerationFailure("the corpus has no image rows for generation")
    try:
        corpus_tokens = [runtime["tokenize"](str(row.get("text_representation", ""))) for row in image_corpus]
        bm25 = BM25Okapi(corpus_tokens)
        encoder = SentenceTransformer(DENSE_MODEL)
        corpus_texts = [str(row.get("text_representation", "")) for row in image_corpus]
        embeddings = encoder.encode(
            corpus_texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as error:
        raise GenerationFailure("image retrieval index could not be prepared") from error

    index = {
        "corpus": image_corpus,
        "bm25": bm25,
        "encoder": encoder,
        "embeddings": np.asarray(embeddings),
        "numpy": np,
        "rank_candidates": runtime["rank_candidates"],
        "tokenize": runtime["tokenize"],
    }
    _INDEX_CACHE[cache_key] = index
    return index


def _retrieve_documents(index: Mapping[str, Any], query: str) -> List[Dict[str, Any]]:
    """Retrieve image rows using the query only (no design references)."""
    np = index["numpy"]
    q_tokens = index["tokenize"](query)
    bm25_scores = np.asarray(index["bm25"].get_scores(q_tokens))
    query_embedding = np.asarray(
        index["encoder"].encode(query, normalize_embeddings=True)
    )
    if query_embedding.ndim > 1:
        query_embedding = query_embedding[0]
    dense_scores = np.dot(index["embeddings"], query_embedding)
    documents = index["rank_candidates"](
        query,
        index["corpus"],
        dense_scores,
        bm25_scores,
        mode="full",
        top_k=RETRIEVAL_TOP_K,
    )
    return [dict(document) for document in documents]


def _retrieve_contexts(index: Mapping[str, Any], query: str) -> List[str]:
    """Retrieve image proxy text using the query only (no design references).

    This compatibility wrapper intentionally returns text only.  Callers that
    need the associated raw media should use ``_retrieve_documents`` and then
    hydrate the returned rows with ``_media_payload``.
    """
    documents = _retrieve_documents(index, query)
    return [
        str(document.get("text_representation", ""))
        for document in documents[:CONTEXT_DEPTH]
    ]


def _media_payload(record: Mapping[str, Any], media_root: Path) -> Dict[str, Any]:
    """Describe the raw image associated with a retrieved corpus row.

    The corpus may contain a symbolic ``file_path`` without the corresponding
    media file.  In that case the returned payload explicitly reports
    ``raw_media_available=False`` instead of pretending that the annotation
    text is the image itself.
    """
    file_path = str(record.get("file_path", ""))
    payload: Dict[str, Any] = {
        "item_id": record.get("item_id"),
        "modality": record.get("modality", "image"),
        "file_path": file_path,
        "resolved_media_path": None,
        "raw_media_available": False,
    }
    if not file_path:
        return payload

    metadata = record.get("metadata")
    album = metadata.get("album") if isinstance(metadata, Mapping) else None
    resolved = resolve_media_path(file_path, REPO_ROOT, media_root, album=album)
    if resolved is None:
        # PhotoBench's public instructions also describe a raw_albums/albumN
        # layout, while the checked-in proxy corpus uses PhotoBench/images/.
        # Use the row's album metadata only for these explicit, deterministic
        # fallbacks; never search the whole workspace by basename.
        filename = Path(file_path).name
        if isinstance(album, str) and album.strip() and filename:
            for candidate in (
                media_root / album / filename,
                media_root / album / "images" / filename,
                media_root / "raw_albums" / album / filename,
                media_root / "PhotoBench" / "images" / album / filename,
                REPO_ROOT / "dataset" / album / "images" / filename,
            ):
                resolved = resolve_media_path(candidate, REPO_ROOT, media_root, album=album)
                if resolved is None and candidate.parent.is_dir():
                    # The proxy corpus normalises filenames to .jpg while the
                    # source albums commonly preserve camera-case .JPG names.
                    casefold_matches = sorted(
                        path
                        for path in candidate.parent.iterdir()
                        if path.is_file()
                        and path.name.casefold() == candidate.name.casefold()
                    )
                    if casefold_matches:
                        resolved = resolve_media_path(
                            casefold_matches[0], REPO_ROOT, media_root, album=album
                        )
                if resolved is not None:
                    break
    if resolved is None:
        return payload
    try:
        return build_image_media_payload(record, resolved)
    except (OSError, ValueError):
        # A file can disappear or become unreadable between resolution and
        # hashing.  Preserve an explicit unavailable state for the audit.
        return payload


def _retrieved_media(
    documents: Sequence[Mapping[str, Any]], media_root: Path
) -> List[Dict[str, Any]]:
    """Hydrate unique raw-media descriptors for the visible context rows."""
    media: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for document in documents[:CONTEXT_DEPTH]:
        descriptor = _media_payload(document, media_root)
        identity = str(
            descriptor.get("media_id")
            or descriptor.get("resolved_media_path")
            or descriptor.get("file_path")
            or descriptor.get("item_id")
        )
        if identity in seen:
            continue
        seen.add(identity)
        media.append(descriptor)
    return media


def _build_multimodal_messages(
    prompt: str, media: Sequence[Mapping[str, Any]]
) -> Tuple[List[Dict[str, Any]], int]:
    """Attach available retrieved images to the model request.

    Text-only behavior is preserved when no media is available.  Image bytes
    stay in the request and are never serialized into the audit checkpoint.
    """
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    attached = 0
    for descriptor in media:
        if not descriptor.get("raw_media_available"):
            continue
        raw_path = descriptor.get("resolved_media_path")
        if not isinstance(raw_path, str):
            continue
        try:
            image_path = Path(raw_path)
            if not image_path.is_file() or image_path.stat().st_size > MAX_IMAGE_BYTES:
                continue
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        except OSError:
            continue
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{encoded}",
                    "detail": "low",
                },
            }
        )
        attached += 1

    if attached == 0:
        return [{"role": "user", "content": prompt}], 0
    content[0]["text"] = (
        prompt
        + "\n\nRaw image attachments follow. Use the actual image pixels only for "
        "claims they visibly support; keep annotation text and visual evidence "
        "distinct."
    )
    return [{"role": "user", "content": content}], attached


def _load_runtime_helpers() -> Dict[str, Any]:
    """Load existing retrieval/prompt/client helpers without changing them."""
    try:
        from .run_expanded_multihop_eval import (
            CANDIDATE_DEPTH,
            RRF_DEPTH,
            RRF_K,
            RERANK_DENSE_WEIGHT,
            build_answer_prompt,
            get_openai_client,
            rank_candidates,
            tokenize,
        )
    except ImportError:
        from run_expanded_multihop_eval import (
            CANDIDATE_DEPTH,
            RRF_DEPTH,
            RRF_K,
            RERANK_DENSE_WEIGHT,
            build_answer_prompt,
            get_openai_client,
            rank_candidates,
            tokenize,
        )
    return {
        "CANDIDATE_DEPTH": CANDIDATE_DEPTH,
        "RRF_DEPTH": RRF_DEPTH,
        "RRF_K": RRF_K,
        "RERANK_DENSE_WEIGHT": RERANK_DENSE_WEIGHT,
        "build_answer_prompt": build_answer_prompt,
        "get_openai_client": get_openai_client,
        "rank_candidates": rank_candidates,
        "tokenize": tokenize,
    }


def _is_real_answer(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    normalized = value.strip().lower()
    # These are failure/fallback forms, not model answers.  Exceptions are
    # never interpolated into an answer or written to a checkpoint.
    return not normalized.startswith(("error:", "error generating answer:"))


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_checkpoint(path: Path, checkpoint: Mapping[str, Any]) -> None:
    content = _json_bytes(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _matching_receipt(
    receipt: Any,
    question: Mapping[str, Any],
    question_set_sha256: str,
    corpus_sha256: str,
    model_sha256: str,
    config_sha256: str,
    build_answer_prompt: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(receipt, dict):
        return None
    identity = receipt.get("identity")
    contexts = receipt.get("retrieved_context")
    retrieved_media = receipt.get("retrieved_media")
    media_attached_count = receipt.get("media_attached_count")
    answer = receipt.get("answer")
    if (
        not isinstance(identity, dict)
        or not isinstance(contexts, list)
        or not isinstance(retrieved_media, list)
        or not isinstance(media_attached_count, int)
    ):
        return None
    if not all(isinstance(context, str) for context in contexts):
        return None
    if not all(isinstance(media, dict) for media in retrieved_media):
        return None
    expected_identity = {
        "question_id": str(question["id"]),
        "question_set_sha256": question_set_sha256,
        "corpus_sha256": corpus_sha256,
        "model_sha256": model_sha256,
        "model": GENERATOR_MODEL,
        "config_sha256": config_sha256,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            return None
    prompt = build_answer_prompt(str(question["query"]), contexts)
    if identity.get("prompt_sha256") != _sha256_text(prompt):
        return None
    if not _is_real_answer(answer):
        return None
    return {
        "identity": dict(identity),
        "retrieved_context": list(contexts),
        "retrieved_media": [dict(media) for media in retrieved_media],
        "media_attached_count": media_attached_count,
        "answer": str(answer).strip(),
    }


def _generate_answers(
    question_set: Mapping[str, Any],
    question_set_sha256: str,
    corpus_path: Path,
    output_dir: Path,
    media_root: Path = DEFAULT_MEDIA_ROOT,
    require_raw_media: bool = False,
) -> Dict[str, Any]:
    """Generate all answers, resuming only exact matching receipts."""
    final_names = (
        "human_audit_packet.json",
        "reviewer_1_form.html",
        "reviewer_2_form.html",
        "reviewer_1.csv",
        "reviewer_2.csv",
    )
    for name in final_names:
        if (output_dir / name).exists():
            raise OutputConflict(f"Refusing to overwrite existing final artifact: {name}")
    # A reviewer CSV is a completed/possibly in-progress human artifact even
    # when it does not use one of the two standard names.  Generation never
    # overwrites any CSV in the new directory.
    for csv_path in output_dir.glob("*.csv"):
        raise OutputConflict(f"Refusing to overwrite existing CSV: {csv_path.name}")

    corpus, corpus_sha256, _ = _load_corpus(Path(corpus_path))
    media_root = Path(media_root).expanduser().resolve()
    if require_raw_media and not any(
        _media_payload(record, media_root).get("raw_media_available")
        for record in corpus
        if record.get("modality") == "image"
    ):
        raise GenerationFailure(
            "no raw image files could be resolved from the corpus; provide "
            "the source images and pass their root with --media-root"
        )
    try:
        runtime = _load_runtime_helpers()
    except Exception as error:
        raise GenerationFailure("existing retrieval/model helpers are unavailable") from error
    config = _generation_config(runtime, media_root)
    config_sha256 = _hash_json(config)
    model_sha256 = _sha256_text(GENERATOR_MODEL)
    checkpoint_path = output_dir / "answer_checkpoint.json"
    prior = _load_checkpoint(checkpoint_path)
    prior_receipts = prior.get("receipts", {}) if isinstance(prior, dict) else {}
    if not isinstance(prior_receipts, dict):
        prior_receipts = {}
    receipts: Dict[str, Dict[str, Any]] = {}
    index: Optional[Mapping[str, Any]] = None
    client: Any = None

    questions = question_set.get("questions")
    if not isinstance(questions, list) or len(questions) != 25:
        raise GenerationFailure("the question set does not contain 25 questions")

    for question in questions:
        if not isinstance(question, dict):
            raise GenerationFailure("the question set contains an invalid question")
        question_id = str(question["id"])
        receipt = _matching_receipt(
            prior_receipts.get(question_id),
            question,
            question_set_sha256,
            corpus_sha256,
            model_sha256,
            config_sha256,
            runtime["build_answer_prompt"],
        )
        if receipt is not None:
            receipts[question_id] = receipt
            continue

        if index is None:
            index = _build_or_get_image_index(corpus, corpus_sha256, runtime)
        query = str(question["query"])
        try:
            # Retrieval receives only the question query.  In particular, the
            # expected detail, target keyword, source doc_idx, and evidence IDs
            # are not used in retrieval or generation.
            documents = _retrieve_documents(index, query)
            visible_documents = documents[:CONTEXT_DEPTH]
            contexts = [
                str(document.get("text_representation", ""))
                for document in visible_documents
            ]
            retrieved_media = _retrieved_media(visible_documents, media_root)
            if require_raw_media and not any(
                media.get("raw_media_available") for media in retrieved_media
            ):
                raise GenerationFailure(
                    f"raw image media is unavailable for {question_id}; "
                    "provide the source images or omit --require-raw-media"
                )
            prompt = runtime["build_answer_prompt"](query, contexts)
            messages, media_attached_count = _build_multimodal_messages(
                prompt, retrieved_media
            )
        except Exception as error:
            raise GenerationFailure("image retrieval or prompt construction failed") from error

        if client is None:
            try:
                client = runtime["get_openai_client"]()
            except Exception as error:
                raise GenerationFailure("the configured model client is unavailable") from error
        try:
            response = client.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=200,
                seed=LLM_SEED,
            )
            answer_value = response.choices[0].message.content
        except Exception as error:
            raise GenerationFailure("the model did not return a usable response") from error
        if not _is_real_answer(answer_value):
            raise GenerationFailure("the model did not return a usable response")

        identity = {
            "question_id": question_id,
            "question_set_sha256": question_set_sha256,
            "corpus_sha256": corpus_sha256,
            "prompt_sha256": _sha256_text(prompt),
            "model_sha256": model_sha256,
            "model": GENERATOR_MODEL,
            "config_sha256": config_sha256,
            "media_attached_count": media_attached_count,
        }
        receipts[question_id] = {
            "identity": identity,
            "retrieved_context": list(contexts),
            "retrieved_media": list(retrieved_media),
            "media_attached_count": media_attached_count,
            "answer": str(answer_value).strip(),
        }
        _write_checkpoint(
            checkpoint_path,
            {
                "packet_version": PACKET_VERSION,
                "question_set_sha256": question_set_sha256,
                "corpus_sha256": corpus_sha256,
                "model": GENERATOR_MODEL,
                "model_sha256": model_sha256,
                "config_sha256": config_sha256,
                "receipts": receipts,
            },
        )

    if len(receipts) != len(questions) or not all(
        _is_real_answer(receipt.get("answer")) for receipt in receipts.values()
    ):
        raise GenerationFailure("not every question has a real model response")

    rows: List[Dict[str, Any]] = []
    for question in questions:
        question_id = str(question["id"])
        receipt = receipts[question_id]
        rows.append(
            {
                "audit_id": question_id,
                "query": str(question["query"]),
                "target_keyword": str(question["target_keyword"]),
                "target_keyword_role": DESIGN_REFERENCE_LABEL,
                "split": "single_hop",
                "retrieved_context": list(receipt["retrieved_context"]),
                "retrieved_media": list(receipt["retrieved_media"]),
                "media_attached_count": int(receipt["media_attached_count"]),
                "answer": str(receipt["answer"]),
            }
        )

    final_packet = {
        "packet_version": PACKET_VERSION,
        "audit_type": "image_description_answer_quality",
        "preview_only": False,
        "annotation_derived_design": True,
        "independent_held_out_evaluation": False,
        "scope_note": SCOPE_NOTE,
        "source_corpus_sha256": corpus_sha256,
        "question_set_sha256": question_set_sha256,
        "scope_counts": dict(question_set["scope_counts"]),
        "generation": {
            "model": GENERATOR_MODEL,
            "config_sha256": config_sha256,
            "retrieval_scope": (
                "image rows only; contexts are text proxies and available raw "
                "image files are attached to the vision request"
            ),
            "raw_media_attached": sum(
                int(receipt.get("media_attached_count", 0))
                for receipt in receipts.values()
            ),
            "reference_answers_fed_to_model": False,
            "target_keywords_fed_to_model": False,
            "evidence_ids_fed_to_model": False,
        },
        "examples": rows,
    }

    # Render all final files before creating any of them, so a writer error does
    # not occur before the all-answers gate.  Existing final files were checked
    # above and are never overwritten.
    try:
        form_writer = _legacy_simple_form_writer()
        rendered_forms: Dict[str, bytes] = {}
        for reviewer, filename, stem in (
            ("Reviewer 1", "reviewer_1_form.html", "reviewer_1"),
            ("Reviewer 2", "reviewer_2_form.html", "reviewer_2"),
        ):
            with tempfile.TemporaryDirectory(prefix=".image-v2-form-", dir=str(output_dir)) as temp_dir:
                temp_path = Path(temp_dir) / filename
                form_writer(
                    temp_path,
                    rows,
                    reviewer,
                    stem,
                    packet_version=PACKET_VERSION,
                    scope_note=SCOPE_NOTE,
                    preview_only=False,
                )
                rendered_forms[filename] = temp_path.read_bytes()
    except Exception as error:
        raise GenerationFailure("reviewer forms could not be prepared") from error

    def blank_csv_bytes() -> bytes:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "audit_id",
                "faithfulness_label",
                "relevance_label",
                "correctness_label",
                "notes",
            ],
            lineterminator="\r\n",
        )
        writer.writeheader()
        for question in questions:
            writer.writerow(
                {
                    "audit_id": str(question["id"]),
                    "faithfulness_label": "",
                    "relevance_label": "",
                    "correctness_label": "",
                    "notes": "",
                }
            )
        return stream.getvalue().encode("utf-8")

    final_bytes = {
        "human_audit_packet.json": _json_bytes(final_packet),
        **rendered_forms,
        "reviewer_1.csv": blank_csv_bytes(),
        "reviewer_2.csv": blank_csv_bytes(),
    }
    for name in final_bytes:
        if (output_dir / name).exists():
            raise OutputConflict(f"Refusing to overwrite existing final artifact: {name}")
    for name, content in final_bytes.items():
        with (output_dir / name).open("xb") as handle:
            handle.write(content)
    return final_packet


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "--corpus",
        dest="input_path",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="JSONL corpus path (image rows are selected by modality)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="new v2 audit directory",
    )
    parser.add_argument(
        "--generate-answers",
        action="store_true",
        help="opt in to configured gpt-4o-mini generation after preview creation",
    )
    parser.add_argument(
        "--media-root",
        type=Path,
        default=DEFAULT_MEDIA_ROOT,
        help=(
            "root corresponding to corpus file_path values, such as the "
            "ai_engine/rag directory containing PhotoBench/images"
        ),
    )
    parser.add_argument(
        "--require-raw-media",
        action="store_true",
        help="fail generation instead of falling back when no retrieved image file is available",
    )
    args = parser.parse_args(argv)

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        question_set = build_question_set(args.input_path)
        preview = build_preview(question_set)
        question_set_content = _json_bytes(question_set)
        preview_content = _json_bytes(preview)
        _safe_create_or_verify(args.output_dir / "question_set.json", question_set_content)
        _safe_create_or_verify(args.output_dir / "question_preview.json", preview_content)
        _write_preview_index(args.output_dir / "index.html", preview)
        _write_preview_forms(args.output_dir, preview)

        if args.generate_answers:
            final_packet = _generate_answers(
                question_set,
                _sha256_bytes(question_set_content),
                args.input_path,
                args.output_dir,
                args.media_root,
                args.require_raw_media,
            )
            print(
                f"Generated {len(final_packet['examples'])} image-description answers "
                f"and prepared final audit artifacts in {args.output_dir}"
            )
        else:
            print(
                f"Prepared image-description v2 preview with {len(question_set['questions'])} "
                f"questions in {args.output_dir}"
            )
        return 0
    except (OSError, OutputConflict, RuntimeError, ValueError) as error:
        # GenerationFailure/OutputConflict messages are deliberately generic;
        # model/client exception payloads are never printed or serialized.
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    main()

"""Build a deterministic, data-driven audit question set for an L10 album.

The album query annotations are the design/reference data for this packet.  No
model, network service, or image decoder is used: media availability is checked
with deterministic filesystem resolution only.  The generated question set is
intended to make the source annotation and its ordered ground-truth filenames
explicit, including the two source queries that contain no ground-truth files.

Normal use writes ``question_set.json`` and ``question_preview.json`` into a
new output directory::

    python -m ai_engine.rag.scripts.prepare_album3_audit \
        --album-dir ai_engine/rag/data/raw_media/l10/album3 \
        --output-dir ai_engine/rag/data/album3_audit

``--preview-only`` writes only the preview JSON.  Existing files are accepted
only when their bytes are identical to the deterministic output; a different
file is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
RAG_ROOT = REPO_ROOT / "ai_engine" / "rag"
DEFAULT_ALBUM_DIR = RAG_ROOT / "data" / "raw_media" / "l10" / "album3"
DEFAULT_OUTPUT_DIR = RAG_ROOT / "data" / "album3_audit"

QUESTION_COUNT = 20
SUPPORTED_ALBUM_NUMBERS = (1, 2, 3)
PACKET_VERSION_BY_ALBUM = {
    album_number: f"l10-album{album_number}-v1"
    for album_number in SUPPORTED_ALBUM_NUMBERS
}
QUESTION_ID_PREFIX_BY_ALBUM = {
    album_number: f"L10-A{album_number}-"
    for album_number in SUPPORTED_ALBUM_NUMBERS
}
# Keep the album3 constants as aliases for callers of the original script.
PACKET_VERSION = PACKET_VERSION_BY_ALBUM[3]
QUESTION_ID_PREFIX = QUESTION_ID_PREFIX_BY_ALBUM[3]
NEGATIVE_CONTROL_ANSWER = "No ground-truth image is listed for this query."
DESIGN_REFERENCE_LABEL = "DESIGN/REFERENCE DATA — not a model output"
QUESTION_TEMPLATE = (
    "Which image file(s), if any, are the ground-truth matches for the supplied "
    'album query "{query_en}"?'
)

# These are the semantic dimensions supplied by the copied L10 query export.
# Their values are retained exactly, including nulls and the source field's
# original capitalization in the input JSON.
DIMENSION_KEYS = ("Location", "Time", "Person", "Object", "Concept", "Genre")

# Counting and resolving paths does not require Pillow.  The extension list is
# deliberately local so importing this script remains standard-library-only.
SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {
        ".avif",
        ".bmp",
        ".gif",
        ".heic",
        ".jpeg",
        ".jfif",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }
)


class OutputConflict(RuntimeError):
    """Raised when an output artifact differs from an existing file."""


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 digest of *path* without changing or rewriting it."""

    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def album_number_from_dir(album_dir: Path | str) -> int:
    """Return the supported L10 album number from the directory's final name."""

    name = _canonical_path(album_dir).name.casefold()
    supported_names = {f"album{number}" for number in SUPPORTED_ALBUM_NUMBERS}
    if name not in supported_names:
        raise ValueError(
            "Album directory name must be album1, album2, or album3; "
            f"found {name or '<empty>'}"
        )
    return int(name[len("album") :])


def packet_version_for_album(album_number: int) -> str:
    """Return the packet version for a supported L10 album number."""

    try:
        return PACKET_VERSION_BY_ALBUM[album_number]
    except KeyError as error:
        raise ValueError(
            f"Unsupported L10 album number {album_number}; expected 1, 2, or 3"
        ) from error


def question_id_prefix_for_album(album_number: int) -> str:
    """Return the question ID prefix for a supported L10 album number."""

    try:
        return QUESTION_ID_PREFIX_BY_ALBUM[album_number]
    except KeyError as error:
        raise ValueError(
            f"Unsupported L10 album number {album_number}; expected 1, 2, or 3"
        ) from error


def _portable_path(path: Path) -> str:
    """Use a repository-relative path where possible, otherwise an absolute path."""

    resolved = _canonical_path(path)
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _json_bytes(value: Any) -> bytes:
    """Serialize JSON with stable key ordering, Unicode, indentation, and newline."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def serialize_json(value: Any) -> str:
    """Return the deterministic JSON representation used by output artifacts."""

    return _json_bytes(value).decode("utf-8")


def _normalise_relative(value: str) -> str:
    """Normalize a user/source path for portable, case-insensitive matching."""

    return value.strip().replace("\\", "/").lstrip("./")


def _validate_query_entry(entry: Any, index: int, query_path: Path) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"Query entry {index} in {query_path} must be a JSON object"
        )

    loaded = dict(entry)
    query_en = loaded.get("query_en")
    if not isinstance(query_en, str) or not query_en.strip():
        raise ValueError(
            f"Query entry {index} in {query_path} must have a non-empty query_en"
        )

    query_cn = loaded.get("query_cn")
    if not isinstance(query_cn, str):
        raise ValueError(f"Query entry {index} in {query_path} must have query_cn text")

    ground_truth = loaded.get("ground_truth")
    if not isinstance(ground_truth, list):
        raise ValueError(
            f"Query entry {index} in {query_path} must have a ground_truth list"
        )
    for filename_index, filename in enumerate(ground_truth):
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(
                f"Query entry {index} ground_truth[{filename_index}] in "
                f"{query_path} must be a non-empty filename"
            )

    source = loaded.get("Source", loaded.get("source"))
    if not isinstance(source, str) or not source.strip():
        raise ValueError(
            f"Query entry {index} in {query_path} must have a non-empty Source"
        )

    # The copied source includes ground_truth_count, but its value is metadata
    # from the exporter and is not always equal to the ordered list length.  It
    # is validated as a scalar when present, while the list remains authoritative.
    if "ground_truth_count" in loaded and (
        not isinstance(loaded["ground_truth_count"], int)
        or isinstance(loaded["ground_truth_count"], bool)
        or loaded["ground_truth_count"] < 0
    ):
        raise ValueError(
            f"Query entry {index} ground_truth_count in {query_path} must be a "
            "non-negative integer"
        )

    return loaded


def load_query_entries(
    album_dir: Path | str,
) -> tuple[list[dict[str, Any]], Path, str]:
    """Load and validate the album's exactly-20 query entries.

    Returns ``(entries, query_json_path, query_json_sha256)``.  Entry order is
    preserved exactly as it appears in ``query.json``; no set or sort operation
    is applied to query entries or their ground-truth lists.
    """

    root = _canonical_path(album_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Album directory not found: {root}")
    query_path = root / "query.json"
    if not query_path.is_file():
        raise FileNotFoundError(f"Album query JSON not found: {query_path}")

    raw = query_path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to parse album query JSON: {query_path}") from error

    if not isinstance(payload, list):
        raise ValueError(f"Album query JSON must contain a list: {query_path}")
    if len(payload) != QUESTION_COUNT:
        raise ValueError(
            f"Expected exactly {QUESTION_COUNT} query entries in {query_path}; "
            f"found {len(payload)}"
        )

    entries = [
        _validate_query_entry(entry, index, query_path)
        for index, entry in enumerate(payload)
    ]
    return entries, query_path, hashlib.sha256(raw).hexdigest()


# A short alias is convenient for callers that use "queries" rather than
# "entries" while retaining one implementation and one validation contract.
load_queries = load_query_entries


def discover_image_paths(album_dir: Path | str) -> list[Path]:
    """Return supported files below ``album_dir/images`` in stable order."""

    root = _canonical_path(album_dir)
    image_root = root / "images"
    if not image_root.is_dir():
        return []

    paths = [
        path.resolve()
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(
        paths,
        key=lambda path: (
            path.relative_to(root).as_posix().casefold(),
            path.relative_to(root).as_posix(),
        ),
    )


def _media_lookup(
    album_dir: Path,
    image_paths: Iterable[Path],
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    by_relative: dict[str, list[Path]] = {}
    by_basename: dict[str, list[Path]] = {}
    for path in image_paths:
        relative = path.relative_to(album_dir).as_posix()
        by_relative.setdefault(relative.casefold(), []).append(path)
        by_basename.setdefault(path.name.casefold(), []).append(path)
    for values in (*by_relative.values(), *by_basename.values()):
        values.sort(
            key=lambda path: (
                path.relative_to(album_dir).as_posix().casefold(),
                path.relative_to(album_dir).as_posix(),
            )
        )
    return by_relative, by_basename


def resolve_media_path(
    filename: str,
    album_dir: Path | str,
    image_paths: Sequence[Path] | None = None,
) -> Path | None:
    """Resolve a ground-truth filename to one deterministic album image.

    Exact relative paths are preferred.  If the annotation contains only a
    basename, a case-insensitive basename match is used, which supports the
    copied album's filename casing and nested image directories.  A missing
    reference returns ``None`` rather than guessing from outside ``images/``.
    """

    root = _canonical_path(album_dir)
    if not isinstance(filename, str) or not filename.strip():
        return None
    paths = list(image_paths) if image_paths is not None else discover_image_paths(root)
    by_relative, by_basename = _media_lookup(root, paths)

    normalised = _normalise_relative(filename)
    if not normalised:
        return None

    relative_candidates = [normalised]
    if not normalised.casefold().startswith("images/"):
        relative_candidates.append(f"images/{normalised}")
    for candidate in relative_candidates:
        matches = by_relative.get(candidate.casefold(), [])
        if matches:
            return matches[0]

    basename = PurePosixPath(normalised).name.casefold()
    matches = by_basename.get(basename, [])
    return matches[0] if matches else None


def _media_reference(
    filename: str,
    album_dir: Path,
    image_paths: Sequence[Path],
) -> dict[str, Any]:
    resolved = resolve_media_path(filename, album_dir, image_paths)
    relative_path = (
        resolved.relative_to(album_dir).as_posix() if resolved is not None else None
    )
    return {
        "filename": filename,
        "raw_media_available": resolved is not None,
        "relative_media_path": relative_path,
    }


def _reference_context(
    *,
    source_query_index: int,
    query_en: str,
    ground_truth: Sequence[str],
    expected_answer: Any,
) -> str:
    filenames = ", ".join(ground_truth) if ground_truth else "none"
    return (
        f"{DESIGN_REFERENCE_LABEL} — source album annotation only; not a "
        f"retrieval result or generated answer. Query index {source_query_index}; "
        f'query_en "{query_en}"; ordered ground-truth filenames: {filenames}; '
        f"reference expected answer: {expected_answer}"
    )


def _dimensions(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {key: entry.get(key) for key in DIMENSION_KEYS}


def _question_id(index: int, question_id_prefix: str = QUESTION_ID_PREFIX) -> str:
    return f"{question_id_prefix}{index + 1:03d}"


def _make_question(
    entry: Mapping[str, Any],
    source_query_index: int,
    album_dir: Path,
    image_paths: Sequence[Path],
    question_id_prefix: str = QUESTION_ID_PREFIX,
) -> dict[str, Any]:
    query_en = str(entry["query_en"])
    query_cn = str(entry["query_cn"])
    ground_truth = [str(filename) for filename in entry["ground_truth"]]
    source = entry.get("Source", entry.get("source"))
    expected_answer: Any = (
        list(ground_truth) if ground_truth else NEGATIVE_CONTROL_ANSWER
    )
    media = [
        _media_reference(filename, album_dir, image_paths)
        for filename in ground_truth
    ]
    relative_media_paths = [
        descriptor["relative_media_path"] for descriptor in media
    ]
    raw_media_availability = [
        descriptor["raw_media_available"] for descriptor in media
    ]
    question_text = QUESTION_TEMPLATE.format(query_en=query_en)
    question_id = _question_id(source_query_index, question_id_prefix)

    return {
        "id": question_id,
        "audit_id": question_id,
        "question": question_text,
        # Existing audit tooling calls the prompt-facing text ``query``; keep
        # it equal to the human-readable question for interoperability.
        "query": question_text,
        "question_type": "album_ground_truth_image_file_match",
        "source_query_index": source_query_index,
        "query_en": query_en,
        "query_cn": query_cn,
        "dimensions": _dimensions(entry),
        "source": str(source),
        "ground_truth": list(ground_truth),
        "expected_answer": expected_answer,
        "expected_answer_role": DESIGN_REFERENCE_LABEL,
        "negative_control": not ground_truth,
        "media": media,
        "media_references": media,
        "raw_media_available": bool(media) and all(raw_media_availability),
        "raw_media_availability": raw_media_availability,
        "relative_media_paths": relative_media_paths,
        "reference_data_role": DESIGN_REFERENCE_LABEL,
    }


def build_question_set(album_dir: Path | str = DEFAULT_ALBUM_DIR) -> dict[str, Any]:
    """Build the deterministic 20-question specification for an L10 album."""

    root = _canonical_path(album_dir)
    album_number = album_number_from_dir(root)
    packet_version = packet_version_for_album(album_number)
    question_id_prefix = question_id_prefix_for_album(album_number)
    entries, query_path, query_sha256 = load_query_entries(root)
    image_paths = discover_image_paths(root)
    questions = [
        _make_question(
            entry,
            index,
            root,
            image_paths,
            question_id_prefix=question_id_prefix,
        )
        for index, entry in enumerate(entries)
    ]
    specifications = []
    for question in questions:
        specification = dict(question)
        specification["question_specification_role"] = DESIGN_REFERENCE_LABEL
        specifications.append(specification)

    supported_count = sum(bool(entry["ground_truth"]) for entry in entries)
    negative_control_count = len(entries) - supported_count
    source_album_path = _portable_path(root)
    source_query_path = _portable_path(query_path)
    return {
        "packet_version": packet_version,
        "source_album_path": source_album_path,
        "source_query_path": source_query_path,
        "query_json_sha256": query_sha256,
        "source_query_json_sha256": query_sha256,
        "image_count": len(image_paths),
        "query_count": len(questions),
        "supported_query_count": supported_count,
        "negative_control_count": negative_control_count,
        "questions": questions,
        "question_specification": specifications,
        "design_reference_label": DESIGN_REFERENCE_LABEL,
        "question_specification_note": (
            "Expected answers and reference contexts are copied from the source "
            "album annotation for audit design; they are not model outputs."
        ),
    }


def build_preview(question_set: Mapping[str, Any]) -> dict[str, Any]:
    """Build preview rows with explicit reference-context and no model output."""

    packet_version = question_set.get("packet_version")
    if packet_version not in PACKET_VERSION_BY_ALBUM.values():
        raise ValueError("question_set has an unsupported packet version")
    questions = question_set.get("questions")
    if not isinstance(questions, list) or len(questions) != QUESTION_COUNT:
        raise ValueError(
            f"question_set must contain exactly {QUESTION_COUNT} questions"
        )

    rows: list[dict[str, Any]] = []
    for question in questions:
        if not isinstance(question, Mapping):
            raise ValueError("question_set contains a non-object question")
        ground_truth = question.get("ground_truth")
        if not isinstance(ground_truth, list) or not all(
            isinstance(filename, str) for filename in ground_truth
        ):
            raise ValueError("question_set contains an invalid ground_truth list")
        expected_answer = question.get("expected_answer")
        context = _reference_context(
            source_query_index=int(question["source_query_index"]),
            query_en=str(question["query_en"]),
            ground_truth=ground_truth,
            expected_answer=expected_answer,
        )
        rows.append(
            {
                "audit_id": str(question["id"]),
                "id": str(question["id"]),
                "question": str(question["question"]),
                "query": str(question["query"]),
                "source_query_index": int(question["source_query_index"]),
                "query_en": str(question["query_en"]),
                "query_cn": str(question["query_cn"]),
                "reference_context": context,
                "reference_context_role": DESIGN_REFERENCE_LABEL,
                "retrieved_context": [context],
                "answer": "",
                "answer_role": "MODEL OUTPUT PLACEHOLDER — no model inference run",
            }
        )

    query_sha256 = str(question_set.get("query_json_sha256", ""))
    return {
        "packet_version": packet_version,
        "preview_only": True,
        "design_reference_label": DESIGN_REFERENCE_LABEL,
        "source_album_path": str(question_set.get("source_album_path", "")),
        "query_json_sha256": query_sha256,
        "image_count": int(question_set.get("image_count", 0)),
        "query_count": len(rows),
        "supported_query_count": int(question_set.get("supported_query_count", 0)),
        "negative_control_count": int(
            question_set.get("negative_control_count", 0)
        ),
        "examples": rows,
        "reference_context_note": (
            "Every retrieved_context value is design/reference data from the "
            "album annotation, not a retrieval result and not a model output."
        ),
    }


def _safe_create_or_verify(path: Path, content: bytes) -> None:
    """Create *path*, or accept it only if its existing bytes are identical."""

    if path.exists():
        if path.is_file() and path.read_bytes() == content:
            return
        raise OutputConflict(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.is_file() and path.read_bytes() == content:
            return
        raise OutputConflict(f"Refusing to overwrite existing artifact: {path}")


def write_artifacts(
    question_set: Mapping[str, Any],
    preview: Mapping[str, Any],
    output_dir: Path | str,
    *,
    preview_only: bool = False,
) -> list[Path]:
    """Write deterministic artifacts without overwriting non-identical files."""

    destination = _canonical_path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifact_values: list[tuple[Path, bytes]] = []
    if not preview_only:
        artifact_values.append(
            (destination / "question_set.json", _json_bytes(question_set))
        )
    artifact_values.append((destination / "question_preview.json", _json_bytes(preview)))

    # Check every target before creating any new file, so one conflict cannot
    # leave a partially written pair of artifacts behind.
    for path, content in artifact_values:
        if path.exists() and (not path.is_file() or path.read_bytes() != content):
            raise OutputConflict(f"Refusing to overwrite existing artifact: {path}")
    for path, content in artifact_values:
        _safe_create_or_verify(path, content)
    return [path for path, _ in artifact_values]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--album-dir",
        type=Path,
        default=DEFAULT_ALBUM_DIR,
        help=f"album directory containing query.json and images/ (default: {DEFAULT_ALBUM_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"new output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="write only question_preview.json; do not create question_set.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        question_set = build_question_set(args.album_dir)
        preview = build_preview(question_set)
        written = write_artifacts(
            question_set,
            preview,
            args.output_dir,
            preview_only=args.preview_only,
        )
    except (OSError, OutputConflict, ValueError) as error:
        parser.error(str(error))
        return 2

    mode = "preview" if args.preview_only else "question set and preview"
    album_number = album_number_from_dir(args.album_dir)
    print(
        f"Prepared {question_set['query_count']} album{album_number} audit questions "
        f"({mode}) in {args.output_dir}"
    )
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI use
    raise SystemExit(main())

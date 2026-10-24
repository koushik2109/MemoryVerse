"""Small helpers for resolving and loading raw image media.

The RAG dataset stores image paths relative to its data root, while callers
may also provide an absolute path.  This module deliberately keeps its import
surface limited to the standard library: image backends are imported only by
``load_image_bgr`` when an image is actually loaded.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Union


PathInput = Union[str, os.PathLike[str]]

# Keep this set in lower-case because suffix comparisons are case-insensitive.
# These are formats commonly supported by OpenCV or Pillow and used by the RAG
# image datasets.  ``.heic`` is accepted for path resolution when a compatible
# runtime decoder is available.
SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {
        ".avif",
        ".bmp",
        ".gif",
        ".heic",
        ".jpeg",
        ".jpg",
        ".jfif",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }
)


def _normalise_root(root: PathInput, *, relative_to: Optional[Path] = None) -> Path:
    """Return an absolute, non-strictly-resolved root path."""

    root_path = Path(root).expanduser()
    if not root_path.is_absolute() and relative_to is not None:
        root_path = relative_to / root_path
    return root_path.resolve()


def _supported_image_path(path: Path) -> bool:
    """Return whether ``path`` has one of the supported image suffixes."""

    return path.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS


def _rooted_candidate(root: Path, relative_path: Path) -> Optional[Path]:
    """Build a candidate while preventing ``..`` from escaping its root."""

    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def resolve_media_path(
    file_path: PathInput,
    repo_root: PathInput,
    rag_root: Optional[PathInput] = None,
    *,
    album: Optional[str] = None,
) -> Optional[Path]:
    """Resolve a dataset image path to an existing supported regular file.

    Absolute paths are checked directly.  Relative paths are tried under the
    supplied ``rag_root`` first, then under ``repo_root``.  If no RAG root is
    supplied, the conventional ``repo_root/ai_engine/rag`` location is also
    tried before the repository root itself.  If a direct candidate does not
    exist, the search falls back to dataset/ and raw_media/l10/ album folders.
    A path is returned only when it exists, is a regular file, and has a
    supported image extension.

    The function intentionally returns ``None`` for missing, unsupported, or
    invalid paths so callers can distinguish an unavailable raw media asset
    without guessing that a similarly named file exists elsewhere.
    """

    if file_path is None:
        return None

    try:
        requested = Path(file_path).expanduser()
        repository_root = _normalise_root(repo_root)
    except (TypeError, ValueError, OSError):
        return None

    candidates: list[Path] = []
    if requested.is_absolute():
        candidates.append(requested.resolve())
    else:
        relative_path = Path(os.fspath(requested))
        roots: list[Path] = []
        if rag_root is not None:
            try:
                roots.append(_normalise_root(rag_root, relative_to=repository_root))
            except (TypeError, ValueError, OSError):
                pass
        else:
            # Dataset folders in this repository live below ai_engine/rag.
            roots.append((repository_root / "ai_engine" / "rag").resolve())
        roots.append(repository_root)

        seen_roots: set[Path] = set()
        for root in roots:
            if root in seen_roots:
                continue
            seen_roots.add(root)
            candidate = _rooted_candidate(root, relative_path)
            if candidate is not None:
                candidates.append(candidate)

    seen_candidates: set[Path] = set()
    for candidate in candidates:
        if candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)
        if candidate.is_file() and _supported_image_path(candidate):
            return candidate

    # If the file is not found at the direct rooted candidate locations, check
    # whether it lives in the repository's dataset or raw_media directory
    # (e.g. dataset/album1/images/<name>, dataset/album2/images/<name>, etc.).
    if not requested.is_absolute() and _supported_image_path(requested):
        filename = requested.name
        search_roots: list[Path] = []
        dataset_root = repository_root / "dataset"
        if dataset_root.is_dir():
            search_roots.append(dataset_root)
        raw_media_root = (
            repository_root / "ai_engine" / "rag" / "data" / "raw_media" / "l10"
        )
        if raw_media_root.is_dir():
            search_roots.append(raw_media_root)
        if rag_root is not None:
            try:
                norm_rag = _normalise_root(rag_root, relative_to=repository_root)
                if norm_rag.is_dir():
                    search_roots.append(norm_rag)
                    rag_raw = norm_rag / "data" / "raw_media" / "l10"
                    if rag_raw.is_dir():
                        search_roots.append(rag_raw)
            except (TypeError, ValueError, OSError):
                pass

        albums = (
            [album]
            if album
            else [p for p in requested.parts if p in {"album1", "album2", "album3"}]
            or ["album1", "album2", "album3"]
        )

        for base_root in search_roots:
            for alb in albums:
                for sub in ("images", ""):
                    cand_dir = base_root / alb / sub if sub else base_root / alb
                    if not cand_dir.is_dir():
                        continue
                    direct = cand_dir / filename
                    if direct.is_file() and _supported_image_path(direct):
                        return direct.resolve()
                    for f in cand_dir.iterdir():
                        if (
                            f.name.casefold() == filename.casefold()
                            and f.is_file()
                            and _supported_image_path(f)
                        ):
                            return f.resolve()

    return None


def load_image_bgr(path: PathInput) -> Any:
    """Load an image as a BGR array using a lazy OpenCV/Pillow backend.

    OpenCV is preferred because ``cv2.imread`` already returns BGR data.  If
    OpenCV is unavailable or cannot decode the file, Pillow is used as a
    fallback and the RGB result is converted to BGR with NumPy.  Neither
    backend is imported until this function is called.

    Raises:
        FileNotFoundError: if ``path`` does not exist or is not a regular file.
        ValueError: if available decoders cannot read the image.
        ImportError: if neither an image decoder nor its array dependency is
            installed.
    """

    image_path = Path(path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image_path = image_path.resolve()

    cv2 = None
    cv2_import_error: Optional[ImportError] = None
    try:
        import cv2 as cv2_module

        cv2 = cv2_module
    except ImportError as exc:
        cv2_import_error = exc

    if cv2 is not None:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is not None:
            return image

    try:
        from PIL import Image
        import numpy as np
    except ImportError as exc:
        if cv2_import_error is not None:
            raise ImportError(
                "load_image_bgr requires OpenCV or Pillow with NumPy installed"
            ) from exc
        raise ValueError(f"Unable to decode image file: {image_path}") from exc

    try:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            # Pillow exposes RGB channel order; reverse it for the BGR contract.
            return np.asarray(rgb)[:, :, ::-1].copy()
    except Exception as exc:
        raise ValueError(f"Unable to decode image file: {image_path}") from exc


def _json_safe_record_value(value: Any) -> Any:
    """Keep common scalar record fields intact while guaranteeing JSON safety."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_image_media_payload(
    record: Mapping[str, Any],
    media_path: PathInput,
) -> dict[str, Any]:
    """Build a JSON-serializable raw-image payload for a dataset record.

    ``media_id`` is content-addressed: two files with identical bytes receive
    the same SHA-256 hex digest regardless of their path or record metadata.
    The supplied media path must be an existing regular file; this prevents a
    payload from claiming raw media is available when it cannot be hashed.
    """

    requested_path = Path(media_path).expanduser()
    resolved_path = requested_path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Image file not found: {requested_path}")

    file_path = record.get("file_path")
    if file_path is None:
        file_path = str(requested_path)

    payload = {
        "item_id": _json_safe_record_value(record.get("item_id")),
        "modality": _json_safe_record_value(record.get("modality", "image")),
        "file_path": _json_safe_record_value(file_path),
        "resolved_media_path": str(resolved_path),
        "raw_media_available": True,
        "media_id": _sha256_file(resolved_path),
    }
    # Keep this invariant local to the utility if a future record shape changes.
    json.dumps(payload)
    return payload


__all__ = [
    "SUPPORTED_IMAGE_EXTENSIONS",
    "build_image_media_payload",
    "load_image_bgr",
    "resolve_media_path",
]

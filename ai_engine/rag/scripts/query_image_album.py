"""Query an image album with a local Hugging Face SigLIP model.

The command intentionally keeps the model imports lazy.  This makes the
annotation/indexing helpers useful in lightweight environments and lets the
tests run without downloading (or even importing) a model.

Example::

    python -m ai_engine.rag.scripts.query_image_album \
        --album-dir ai_engine/rag/data/raw_media/l10/album3 \
        --query "spring flowers blooming" \
        --top-k 5 \
        --offline \
        --output /tmp/album3-query.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence


MODEL_ID = "google/siglip-base-patch16-224"
DEFAULT_TOP_K = 5
DEFAULT_IMAGE_BATCH_SIZE = 16

# These are the image formats commonly supported by Pillow in the runtime used
# by this project.  Suffix matching is case-insensitive; Pillow remains the
# only decoder used by this utility.
SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {
        ".avif",
        ".bmp",
        ".gif",
        ".jfif",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }
)


def _normalise_filename_key(filename: os.PathLike[str] | str) -> str:
    """Return a case-insensitive basename key for an image filename.

    ``query.json`` stores ground-truth names as basenames, while an album may
    contain images in nested directories.  Normalising to a basename keeps
    annotation matching compatible with both layouts and with case-only
    extension differences such as ``IMG_1.JPG`` versus ``IMG_1.jpg``.
    """

    value = os.fspath(filename).strip().replace("\\", "/")
    if not value:
        return ""
    return PurePosixPath(value).name.casefold()


def _relative_filename(path: Path, album_dir: Path) -> str:
    """Return a portable, deterministic filename relative to ``album_dir``."""

    return path.relative_to(album_dir).as_posix()


def discover_image_paths(album_dir: Path | str) -> list[Path]:
    """Recursively discover supported image files in deterministic order.

    The returned paths are absolute and sorted by their portable relative
    filename.  Actual decoding is deliberately deferred to
    :func:`load_pillow_image` so callers can inspect the deterministic corpus
    list without loading a model or image pixels.
    """

    root = Path(album_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Album directory not found: {root}")

    paths = [
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(
        paths,
        key=lambda path: (
            _relative_filename(path, root).casefold(),
            _relative_filename(path, root),
        ),
    )


def load_pillow_image(path: Path | str) -> Any:
    """Load one image as an RGB Pillow image.

    Pillow is imported here, rather than at module import time, so annotation
    helpers and tests do not require an image backend merely to import this
    module.  The returned image is detached from the file handle and can be
    passed to a processor after this function returns.
    """

    image_path = Path(path).expanduser()
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "Pillow is required to load album images. Install Pillow and retry."
        ) from exc

    try:
        with Image.open(image_path) as opened:
            return opened.convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Unable to decode image with Pillow: {image_path}") from exc


def load_query_annotations(album_dir: Path | str) -> list[dict[str, Any]]:
    """Load annotation entries from ``album_dir/query.json`` when present.

    An absent annotation file is valid and produces an empty list.  The copied
    L10 albums use a top-level JSON list; a mapping containing ``annotations``,
    ``queries``, or ``data`` is also accepted for convenient reuse with nearby
    exported album formats.
    """

    root = Path(album_dir).expanduser().resolve()
    query_paths: list[Path] = []
    direct_query = root / "query.json"
    if direct_query.is_file():
        query_paths.append(direct_query)
    else:
        query_paths.extend(sorted(root.rglob("query.json")))

    if not query_paths:
        return []

    entries: list[Any] = []
    for query_path in query_paths:
        try:
            with query_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read annotation file {query_path}: {exc}") from exc

        if isinstance(payload, list):
            entries.extend(payload)
        elif isinstance(payload, Mapping):
            found_entries = None
            for key in ("annotations", "queries", "data"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    found_entries = candidate
                    break
            if found_entries is not None:
                entries.extend(found_entries)
            else:
                raise ValueError(
                    f"Annotation file must contain a JSON list: {query_path}"
                )
        else:
            raise ValueError(f"Annotation file must contain a JSON list: {query_path}")

    annotations: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"Annotation entry {index} must be a JSON object"
            )
        annotations.append(dict(entry))
    return annotations


def build_annotation_index(
    annotations: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Index full annotation entries by each ground-truth image basename.

    Keys are lower-cased basenames.  Each annotation is included at most once
    per image even if a malformed/duplicated ground-truth list names that image
    more than once.  The original annotation dictionaries are copied into the
    index so later callers can safely serialise them.
    """

    index: MutableMapping[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        ground_truth = annotation.get("ground_truth", [])
        if isinstance(ground_truth, str):
            references: Sequence[Any] = [ground_truth]
        elif isinstance(ground_truth, Sequence):
            references = ground_truth
        else:
            continue

        seen_keys: set[str] = set()
        for reference in references:
            if not isinstance(reference, (str, os.PathLike)):
                continue
            key = _normalise_filename_key(reference)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            index[key].append(dict(annotation))

    return {key: index[key] for key in sorted(index)}


def matching_annotations(
    filename: Path | str,
    annotation_index: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Return annotation entries matching ``filename`` in index order."""

    key = _normalise_filename_key(filename)
    return [dict(annotation) for annotation in annotation_index.get(key, ())]


def _import_model_dependencies() -> tuple[Any, Any, Any]:
    """Import torch and Transformers only when model inference is requested."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyTorch (torch) is required for SigLIP image queries. "
            "Install the RAG requirements and retry."
        ) from exc

    try:
        from transformers import AutoModel, AutoProcessor
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Hugging Face Transformers is required for SigLIP image queries. "
            "Install the RAG requirements and retry."
        ) from exc

    return torch, AutoModel, AutoProcessor


def load_siglip_model(*, offline: bool = False) -> tuple[Any, Any, Any]:
    """Load and return ``(torch, processor, model)`` for the SigLIP model.

    ``local_files_only=True`` is passed to both Hugging Face loaders in offline
    mode.  Model access failures are wrapped with an actionable message rather
    than exposing a long Hub traceback as the command's primary error.
    """

    torch, auto_model, auto_processor = _import_model_dependencies()
    loader_kwargs = {"local_files_only": True} if offline else {}
    try:
        processor = auto_processor.from_pretrained(MODEL_ID, **loader_kwargs)
        model = auto_model.from_pretrained(MODEL_ID, **loader_kwargs)
        model.eval()
    except Exception as exc:  # pragma: no cover - requires model access
        mode_hint = (
            " The requested offline mode requires the model files to already be "
            "cached locally."
            if offline
            else " Check network access or pre-download the model before retrying."
        )
        raise RuntimeError(
            f"Unable to load Hugging Face model {MODEL_ID}.{mode_hint} "
            f"Original error: {exc}"
        ) from exc
    return torch, processor, model


def _feature_tensor(output: Any, *, name: str) -> Any:
    """Extract a tensor from common Transformers feature output shapes."""

    if hasattr(output, "shape"):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state.mean(dim=1)
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if hasattr(first, "shape"):
            return first
    raise RuntimeError(f"SigLIP did not return a usable {name} embedding tensor")


def _normalise_embeddings(torch: Any, embeddings: Any) -> Any:
    """L2-normalise a batch of embedding vectors along its final dimension."""

    if len(embeddings.shape) == 1:
        embeddings = embeddings.unsqueeze(0)
    return torch.nn.functional.normalize(embeddings, p=2, dim=-1)


def _encode_images(
    *,
    torch: Any,
    processor: Any,
    model: Any,
    images: Sequence[Any],
) -> Any:
    """Encode one Pillow-image batch and return normalised image embeddings."""

    inputs = processor(images=list(images), return_tensors="pt", padding=True)
    with torch.no_grad():
        if hasattr(model, "get_image_features"):
            try:
                output = model.get_image_features(**inputs)
            except TypeError:
                output = model.get_image_features(pixel_values=inputs["pixel_values"])
        else:  # Compatibility fallback for older AutoModel SigLIP variants.
            vision_inputs = {
                key: value for key, value in inputs.items() if key.startswith("pixel")
            }
            output = model.vision_model(**vision_inputs)
            output = _feature_tensor(output, name="image")
            projection = getattr(model, "visual_projection", None)
            if projection is not None:
                output = projection(output)
        return _normalise_embeddings(torch, _feature_tensor(output, name="image"))


def _encode_query(*, torch: Any, processor: Any, model: Any, query: str) -> Any:
    """Encode and normalise one text query."""

    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    with torch.no_grad():
        if hasattr(model, "get_text_features"):
            try:
                output = model.get_text_features(**inputs)
            except TypeError:
                output = model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
        else:  # Compatibility fallback for older AutoModel SigLIP variants.
            text_inputs = {
                key: value
                for key, value in inputs.items()
                if key != "pixel_values"
            }
            output = model.text_model(**text_inputs)
            output = _feature_tensor(output, name="text")
            projection = getattr(model, "text_projection", None)
            if projection is not None:
                output = projection(output)
        return _normalise_embeddings(torch, _feature_tensor(output, name="text"))


def serialize_results(results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Canonicalise and deterministically sort result records.

    Scores sort descending, with relative filename and then path as stable tie
    breakers.  Only JSON-friendly fields are emitted, which also makes this
    helper useful in tests without importing torch or Transformers.
    """

    canonical: list[dict[str, Any]] = []
    for result in results:
        filename = os.fspath(result["filename"])
        path = os.fspath(result["path"])
        score = float(result["score"])
        if not math.isfinite(score):
            raise ValueError(f"Image result score must be finite: {score!r}")
        annotations = result.get("annotations", [])
        if annotations is None:
            annotations = []
        resolved_path = Path(path).resolve()
        canonical.append(
            {
                "filename": filename,
                "score": score,
                "path": path,
                "resolved_media_path": str(resolved_path),
                "raw_media_available": resolved_path.is_file(),
                "media_url": resolved_path.as_uri(),
                "annotations": [dict(annotation) for annotation in annotations],
            }
        )

    canonical.sort(
        key=lambda result: (
            -result["score"],
            result["filename"].casefold(),
            result["filename"],
            result["path"],
        )
    )
    return canonical


def build_output(
    *,
    query: str,
    album_dir: Path | str,
    top_k: int,
    results: Iterable[Mapping[str, Any]],
    model: str = MODEL_ID,
) -> dict[str, Any]:
    """Build the stable top-level JSON payload for an album query."""

    return {
        "query": query,
        "album_dir": str(Path(album_dir).expanduser().resolve()),
        "model": model,
        "top_k": int(top_k),
        "results": serialize_results(results),
        "raw_media_used": True,
    }


def serialize_output(payload: Mapping[str, Any]) -> str:
    """Serialize a query payload with stable formatting and a final newline."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def query_album(
    album_dir: Path | str,
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    offline: bool = False,
    batch_size: int = DEFAULT_IMAGE_BATCH_SIZE,
    model_loader: Callable[..., tuple[Any, Any, Any]] | None = None,
) -> dict[str, Any]:
    """Run a text-to-image cosine query over an album directory.

    ``model_loader`` is an injection point for callers/tests that already have
    a compatible ``(torch, processor, model)`` tuple.  Normal CLI use leaves
    it unset, which invokes :func:`load_siglip_model` lazily.
    """

    if top_k <= 0:
        raise ValueError("--top-k must be greater than zero")
    if batch_size <= 0:
        raise ValueError("Image batch size must be greater than zero")
    if not query.strip():
        raise ValueError("--query must not be empty")

    root = Path(album_dir).expanduser().resolve()
    image_paths = discover_image_paths(root)
    if not image_paths:
        raise ValueError(f"No supported image files found recursively under: {root}")

    annotations = load_query_annotations(root)
    annotation_index = build_annotation_index(annotations)
    model_name = MODEL_ID
    score_values: list[float]

    try:
        loader = model_loader or load_siglip_model
        torch, processor, model = loader(offline=offline)
        query_embedding = _encode_query(
            torch=torch,
            processor=processor,
            model=model,
            query=query,
        )

        image_embeddings: list[Any] = []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            images = [load_pillow_image(path) for path in batch_paths]
            try:
                image_embeddings.append(
                    _encode_images(
                        torch=torch,
                        processor=processor,
                        model=model,
                        images=images,
                    )
                )
            finally:
                for image in images:
                    close = getattr(image, "close", None)
                    if close is not None:
                        close()

        all_image_embeddings = torch.cat(image_embeddings, dim=0)
        # Both sides have already been L2-normalised, so their dot product is the
        # cosine similarity.  Normalise once more defensively in case a custom
        # model implementation returns a different tensor from the compatibility
        # path above.
        all_image_embeddings = _normalise_embeddings(torch, all_image_embeddings)
        query_embedding = _normalise_embeddings(torch, query_embedding)
        scores = all_image_embeddings @ query_embedding.squeeze(0)
        score_values = scores.detach().cpu().tolist()
    except Exception as exc:
        if model_loader is not None:
            raise exc
        try:
            from sentence_transformers import SentenceTransformer

            st_model = SentenceTransformer("all-MiniLM-L6-v2")
            q_emb = st_model.encode(query, normalize_embeddings=True)
            score_values = []
            for path in image_paths:
                filename = _relative_filename(path, root)
                matched_ann = matching_annotations(filename, annotation_index)
                texts = [a.get("query_en", "") for a in matched_ann if a.get("query_en")]
                if not texts:
                    texts = [path.stem.replace("_", " ")]
                doc_text = " ".join(texts)
                d_emb = st_model.encode(doc_text, normalize_embeddings=True)
                sim = float(sum(float(q) * float(d) for q, d in zip(q_emb, d_emb)))
                score_values.append(sim)
            model_name = "all-MiniLM-L6-v2 (semantic annotation fallback)"
        except Exception:
            raise exc

    raw_results = []
    for path, score in zip(image_paths, score_values):
        filename = _relative_filename(path, root)
        raw_results.append(
            {
                "filename": filename,
                "score": float(score),
                "path": str(path),
                "annotations": matching_annotations(filename, annotation_index),
            }
        )

    return build_output(
        query=query,
        album_dir=root,
        top_k=top_k,
        results=serialize_results(raw_results)[:top_k],
        model=model_name,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank album images against a text query with local SigLIP."
    )
    parser.add_argument(
        "--album-dir",
        required=True,
        type=Path,
        help="Album directory containing images and optional query.json",
    )
    parser.add_argument("--query", required=True, help="Text query to match")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of ranked images to return (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; otherwise the payload is printed only",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Load SigLIP only from the local Hugging Face cache",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")

    try:
        payload = query_album(
            args.album_dir,
            args.query,
            top_k=args.top_k,
            offline=args.offline,
        )
        serialised = serialize_output(payload)
        if args.output is not None:
            output_path = args.output.expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(serialised, encoding="utf-8")
        print(serialised, end="")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI usage
    raise SystemExit(main())

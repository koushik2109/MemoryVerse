import csv
import json
from collections import defaultdict
from pathlib import Path


RAG_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = RAG_DIR / "data"
OUTPUT_PATH = DATA_DIR / "unified_rag_dataset.jsonl"


def _unique_sorted(values):
    """Return non-empty values in a deterministic order without duplicates."""
    return sorted({value for value in values if value})


def _normalise_filename(filename):
    """Use one key for filename references that differ only by case."""
    return filename.casefold()


def _item_id(dataset, filename):
    """Build an ID that cannot collide across datasets or file extensions."""
    return f"{dataset}:{Path(filename).name.casefold()}"

def build_unified_dataset():
    unified_data = []

    base_dir = RAG_DIR

    # 1. ESC-50
    esc50_meta_path = base_dir / "ESC-50-master" / "meta" / "esc50.csv"
    if esc50_meta_path.exists():
        with esc50_meta_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                unified_data.append({
                    "item_id": _item_id("ESC-50", row["filename"]),
                    "modality": "audio",
                    "file_path": f"ESC-50-master/audio/{row['filename']}",
                    "text_representation": row["category"],
                    "metadata": {
                        "dataset": "ESC-50",
                        "category": row["category"],
                        "target": int(row["target"]) if row.get("target") else None,
                        "esc10": row["esc10"] == "True"
                    }
                })
        print(f"Processed ESC-50 dataset.")
    else:
        print(f"Warning: ESC-50 meta file not found at {esc50_meta_path}")

    # 2. PhotoBench
    # Map from image_filename to list of query dicts
    photobench_images = defaultdict(lambda: {"queries_en": [], "queries_cn": [], "metadata": {"dataset": "PhotoBench"}})
    photobench_filenames = {}
    photobench_files = sorted((base_dir / "PhotoBench" / "data").rglob("*.json"))

    pb_count = 0
    for pb_file in photobench_files:
        try:
            with pb_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                album_name = pb_file.name.split("_")[0]
                for item in data:
                    ground_truth = item.get("ground_truth", [])
                    query_en = item.get("query_en")
                    query_cn = item.get("query_cn")
                    for img in ground_truth:
                        if query_en:
                            key = _normalise_filename(img)
                            photobench_images[key]["queries_en"].append(query_en)
                        if query_cn:
                            key = _normalise_filename(img)
                            photobench_images[key]["queries_cn"].append(query_cn)
                        key = _normalise_filename(img)
                        photobench_filenames[key] = min(photobench_filenames.get(key, img), img)
                        photobench_images[key]["metadata"]["album"] = album_name
        except Exception as e:
            print(f"Error reading {pb_file}: {e}")

    for image_key in sorted(photobench_images):
        img = photobench_filenames[image_key]
        data = photobench_images[image_key]
        # Using English queries as the primary text representation, space-separated
        queries_en = _unique_sorted(data["queries_en"])
        queries_cn = _unique_sorted(data["queries_cn"])
        text_rep = " ".join(queries_en)
        unified_data.append({
            "item_id": _item_id("PhotoBench", img),
            "modality": "image",
            "file_path": f"PhotoBench/images/{img}", # Assuming an images folder structure, adjust later if needed
            "text_representation": text_rep,
            "metadata": {
                **data["metadata"],
                "queries_en": queries_en,
                "queries_cn": queries_cn,
            }
        })
        pb_count += 1
    print(f"Processed PhotoBench dataset: {pb_count} images.")

    # 3. L10
    l10_images = defaultdict(lambda: {"queries_en": [], "queries_cn": [], "metadata": {"dataset": "L10", "album": "album1"}})
    l10_filenames = {}
    l10_query_path = base_dir / "20260822_l10318e6g11p5pnb2" / "album1" / "query.json"
    if l10_query_path.exists():
        with l10_query_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                ground_truth = item.get("ground_truth", [])
                query_en = item.get("query_en")
                query_cn = item.get("query_cn")
                for img in ground_truth:
                    key = _normalise_filename(img)
                    l10_filenames[key] = min(l10_filenames.get(key, img), img)
                    if query_en:
                        l10_images[key]["queries_en"].append(query_en)
                    if query_cn:
                        l10_images[key]["queries_cn"].append(query_cn)
    
    l10_count = 0
    for image_key in sorted(l10_images):
        img = l10_filenames[image_key]
        data = l10_images[image_key]
        queries_en = _unique_sorted(data["queries_en"])
        queries_cn = _unique_sorted(data["queries_cn"])
        text_rep = " ".join(queries_en)
        unified_data.append({
            "item_id": _item_id("L10", img),
            "modality": "image",
            "file_path": f"20260822_l10318e6g11p5pnb2/album1/images/{img}",
            "text_representation": text_rep,
            "metadata": {
                **data["metadata"],
                "queries_en": queries_en,
                "queries_cn": queries_cn,
            }
        })
        l10_count += 1
    print(f"Processed L10 dataset: {l10_count} images.")

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for entry in unified_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Successfully wrote {len(unified_data)} entries to {OUTPUT_PATH}")

if __name__ == "__main__":
    build_unified_dataset()

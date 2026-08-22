import json
import csv
import os
import glob
from collections import defaultdict

def build_unified_dataset():
    unified_data = []

    base_dir = "/home/koushik_2109/MemoryVerse/ai_engine/rag"

    # 1. ESC-50
    esc50_meta_path = os.path.join(base_dir, "ESC-50-master", "meta", "esc50.csv")
    if os.path.exists(esc50_meta_path):
        with open(esc50_meta_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                unified_data.append({
                    "item_id": row["filename"].split(".")[0],
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
    photobench_files = glob.glob(os.path.join(base_dir, "PhotoBench", "data", "**", "*.json"), recursive=True)
    
    pb_count = 0
    for pb_file in photobench_files:
        if not os.path.exists(pb_file):
            continue
        try:
            with open(pb_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                album_name = os.path.basename(pb_file).split('_')[0]
                for item in data:
                    ground_truth = item.get("ground_truth", [])
                    query_en = item.get("query_en")
                    query_cn = item.get("query_cn")
                    for img in ground_truth:
                        if query_en:
                            photobench_images[img]["queries_en"].append(query_en)
                        if query_cn:
                            photobench_images[img]["queries_cn"].append(query_cn)
                        photobench_images[img]["metadata"]["album"] = album_name
        except Exception as e:
            print(f"Error reading {pb_file}: {e}")

    for img, data in photobench_images.items():
        # Using English queries as the primary text representation, space-separated
        text_rep = " ".join(data["queries_en"])
        unified_data.append({
            "item_id": img.split(".")[0],
            "modality": "image",
            "file_path": f"PhotoBench/images/{img}", # Assuming an images folder structure, adjust later if needed
            "text_representation": text_rep,
            "metadata": {
                **data["metadata"],
                "queries_en": list(set(data["queries_en"])), # Remove duplicates
                "queries_cn": list(set(data["queries_cn"]))
            }
        })
        pb_count += 1
    print(f"Processed PhotoBench dataset: {pb_count} images.")

    # 3. L10
    l10_images = defaultdict(lambda: {"queries_en": [], "queries_cn": [], "metadata": {"dataset": "L10", "album": "album1"}})
    l10_query_path = os.path.join(base_dir, "20260822_l10318e6g11p5pnb2", "album1", "query.json")
    if os.path.exists(l10_query_path):
        with open(l10_query_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                ground_truth = item.get("ground_truth", [])
                query_en = item.get("query_en")
                query_cn = item.get("query_cn")
                for img in ground_truth:
                    if query_en:
                        l10_images[img]["queries_en"].append(query_en)
                    if query_cn:
                        l10_images[img]["queries_cn"].append(query_cn)
    
    l10_count = 0
    for img, data in l10_images.items():
        text_rep = " ".join(data["queries_en"])
        unified_data.append({
            "item_id": img.split(".")[0],
            "modality": "image",
            "file_path": f"20260822_l10318e6g11p5pnb2/album1/images/{img}",
            "text_representation": text_rep,
            "metadata": {
                **data["metadata"],
                "queries_en": list(set(data["queries_en"])),
                "queries_cn": list(set(data["queries_cn"]))
            }
        })
        l10_count += 1
    print(f"Processed L10 dataset: {l10_count} images.")

    # Save
    out_dir = os.path.join(base_dir, "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "unified_rag_dataset.jsonl")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for entry in unified_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Successfully wrote {len(unified_data)} entries to {out_path}")

if __name__ == "__main__":
    build_unified_dataset()

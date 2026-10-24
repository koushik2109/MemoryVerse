"""
Download the 1,000 selected MS-COCO images for Task 4.
Uses 32 concurrent threads with retries.
"""

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
COCO_DATA_PATH = REPO_ROOT / "ai_engine" / "rag" / "data" / "coco_karpathy_test.json"
TARGET_DIR = REPO_ROOT / "ai_engine" / "rag" / "data" / "coco_1k_images"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

with open(COCO_DATA_PATH, encoding="utf-8") as f:
    coco_records = json.load(f)

rng = np.random.default_rng(RANDOM_SEED)
selected_indices = rng.choice(len(coco_records), size=1000, replace=False)
selected_records = [coco_records[i] for i in selected_indices]

tasks = []
for i, r in enumerate(selected_records):
    rel_path = r["image"]
    filename = Path(rel_path).name
    dest_path = TARGET_DIR / filename
    url = f"http://images.cocodataset.org/{rel_path}"
    tasks.append((url, dest_path))

print(f"Total images to download: {len(tasks)}")

def download_one(item):
    url, dest = item
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
                with open(dest, "wb") as f:
                    f.write(data)
            return True
        except Exception:
            time.sleep(0.5)
    return False

t0 = time.time()
success = 0
with ThreadPoolExecutor(max_workers=32) as executor:
    futures = [executor.submit(download_one, item) for item in tasks]
    for fut in as_completed(futures):
        if fut.result():
            success += 1
        if success % 100 == 0:
            print(f"Downloaded {success}/{len(tasks)} images ({time.time() - t0:.1f}s)...")

print(f"Completed! Downloaded {success}/{len(tasks)} images in {time.time() - t0:.1f}s.")

"""
Generate BLIP Caption Proxies for the 1,000 MS-COCO Images
===========================================================
Generates realistic VLM captions using Salesforce/blip-image-captioning-base.
Saves mapping: filename -> blip_caption into ai_engine/rag/data/coco_blip_captions.json
"""

import json
import time
from pathlib import Path
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIR = REPO_ROOT / "ai_engine" / "rag" / "data" / "coco_1k_images"
OUTPUT_PATH = REPO_ROOT / "ai_engine" / "rag" / "data" / "coco_blip_captions.json"

BATCH_SIZE = 16

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading BLIP on {device}...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", local_files_only=True).to(device)
    model.eval()

    img_paths = sorted(list(IMAGE_DIR.glob("*.jpg")))
    print(f"Found {len(img_paths)} images to caption.")

    captions = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            captions = json.load(f)
        print(f"Loaded {len(captions)} already generated captions.")

    remaining = [p for p in img_paths if p.name not in captions]
    print(f"Generating captions for {len(remaining)} images...")

    t0 = time.time()
    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i : i + BATCH_SIZE]
        images = []
        valid_batch = []
        for p in batch:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_batch.append(p)
            except Exception as e:
                print(f"Error reading {p}: {e}")
        
        if not images:
            continue

        inputs = processor(images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=25)
        batch_captions = processor.batch_decode(outputs, skip_special_tokens=True)

        for p, cap in zip(valid_batch, batch_captions):
            captions[p.name] = cap.strip()

        if (i + BATCH_SIZE) % 100 == 0 or (i + BATCH_SIZE) >= len(remaining):
            print(f"Processed {min(i + BATCH_SIZE, len(remaining))}/{len(remaining)} images in {time.time() - t0:.1f}s...")
            # Incremental save
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(captions, f, indent=2)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2)

    print(f"Finished generating all {len(captions)} BLIP captions in {time.time() - t0:.1f}s.")
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

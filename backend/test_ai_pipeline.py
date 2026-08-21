import io
import os
import sys
from datetime import datetime, timedelta, timezone
from PIL import Image

# Ensure backend folder is on PATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.ai_extractor import AIExtractor
from app.services.event_clustering_service import EventClusteringService

def create_dummy_image() -> bytes:
    """Generate a tiny mock JPEG image in memory."""
    img = Image.new("RGB", (100, 100), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def run_tests():
    print("=== STARTING AI PIPELINE VERIFICATION TESTS ===")
    
    # 1. Generate dummy image bytes
    img_bytes = create_dummy_image()
    print("[1/5] Dummy image generated successfully.")
    
    # 2. Test Metadata Extractor
    print("[2/5] Testing EXIF Metadata Extraction...")
    meta = AIExtractor.extract_metadata(img_bytes, "image/jpeg")
    print(f"-> Extracted Metadata: {meta}")
    assert "taken_at" in meta
    assert "latitude" in meta
    
    # 3. Test Feature Extractor & CLIP Embeddings
    print("[3/5] Testing CLIP Embedding and Tag Generation...")
    print("-> Loading CLIP model (this might take a few seconds)...")
    emb, tags = AIExtractor.extract_features(img_bytes, "image/jpeg")
    print(f"-> Embedding Dimensions: {len(emb)}")
    print(f"-> Detected Tags: {tags}")
    
    assert len(emb) == 512, f"Expected 512 dimensions, got {len(emb)}"
    assert isinstance(tags, dict), "Tags must be a dictionary"
    print("-> CLIP feature extraction PASSED!")
    
    # 4. Test Event Clustering Heuristics & Analysis
    print("[4/5] Testing Clustering Heuristics & Title Generator...")
    now = datetime.now(timezone.utc)
    mock_media_items = [
        {
            "id": "1",
            "taken_at": now,
            "location_name": "Goa Beach",
            "metadata": {
                "ai_tags": {
                    "scenes": ["beach", "sunset"],
                    "objects": ["beverage"]
                }
            }
        },
        {
            "id": "2",
            "taken_at": now + timedelta(minutes=30),
            "location_name": "Goa Beach",
            "metadata": {
                "ai_tags": {
                    "scenes": ["beach"],
                    "objects": ["food"]
                }
            }
        }
    ]
    
    title, desc, avg_date, vault_id = EventClusteringService._analyze_cluster(mock_media_items)
    print(f"-> Generated Event Title: '{title}'")
    print(f"-> Generated Event Description: '{desc}'")
    print(f"-> Clustered Average Timestamp: {avg_date}")
    
    assert "Beach" in title or "beach" in title or "Gathering" in title, f"Unexpected title: {title}"
    print("-> Title and description auto-generation PASSED!")

    print("[5/5] All local pipeline units verified successfully!")
    print("=== TESTS PASSED ===")

if __name__ == "__main__":
    run_tests()

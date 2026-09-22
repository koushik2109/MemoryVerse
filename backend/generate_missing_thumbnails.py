import os
import sys
import io
import time
from datetime import datetime, timezone
import urllib.request

# Ensure backend folder is on path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Any, cast
from app.core.db import get_supabase_client
from PIL import Image

def generate_thumbnail(image_bytes: bytes, max_dim: int = 480) -> bytes | None:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        if img.width > max_dim or img.height > max_dim:
            img.thumbnail((max_dim, max_dim), cast(Any, Image.Resampling.LANCZOS))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80, optimize=True)
        return buf.getvalue()
    except Exception as e:
        print(f"Failed to generate thumbnail: {e}")
        return None

def main():
    print("=== STARTING RETROACTIVE THUMBNAIL GENERATION ===")
    supabase = get_supabase_client()
    
    # 1. Fetch all images where thumbnail_url == url or thumbnail_url is null
    res = supabase.table("media").select("*").eq("media_type", "image").execute()
    if not res.data:
        print("No media records found in database.")
        return

    items = cast(list[dict[str, Any]], res.data or [])
    print(f"Found {len(items)} total image records. Inspecting for uncompressed thumbnails...")
    
    updated_count = 0
    
    for item in items:
        media_id = str(item["id"])
        url = str(item["url"])
        thumb_url = str(item.get("thumbnail_url") or "")
        storage_path = str(item.get("storage_path") or "")
        owner_id = str(item.get("owner_id") or "")
        filename = str(item.get("filename") or "")
        
        # If thumbnail is already pointing to a custom thumb folder, skip it
        if thumb_url and "/thumbs/" in thumb_url:
            print(f"Skipping already optimized image: {filename}")
            continue
            
        print(f"Processing unoptimized image: {filename}...")
        
        try:
            # 2. Download original image bytes from URL
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                img_bytes = response.read()
                
            # 3. Generate thumbnail bytes
            thumb_bytes = generate_thumbnail(img_bytes)
            if not thumb_bytes:
                print(f"Could not generate thumbnail for {filename}. Skipping.")
                continue
                
            # 4. Upload thumbnail to Supabase Storage
            ts = int(time.time() * 1000)
            safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
            thumb_path = f"{owner_id}/thumbs/{ts}_{safe_name}.jpg"
            
            supabase.storage.from_("memories").upload(
                path=thumb_path,
                file=thumb_bytes,
                file_options={"content-type": "image/jpeg", "upsert": "true"},
            )
            
            # Create a signed URL valid for 1 year
            signed_res = cast(dict[str, Any], supabase.storage.from_("memories").create_signed_url(thumb_path, 31536000))
            new_thumb_url = signed_res.get("signedURL") or signed_res.get("signed_url") or ""
            
            if not new_thumb_url:
                print(f"Failed to get signed URL for thumbnail of {filename}. Skipping.")
                continue
                
            # 5. Update DB record
            supabase.table("media").update({
                "thumbnail_url": new_thumb_url
            }).eq("id", media_id).execute()
            
            print(f"Successfully generated thumbnail for {filename} (~{len(thumb_bytes)//1024}KB)!")
            updated_count += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    print(f"=== COMPLETED: Updated {updated_count} media records! ===")

if __name__ == "__main__":
    main()

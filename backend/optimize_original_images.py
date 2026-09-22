import os
import sys
import io
import time
import urllib.request

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Any, cast
from app.core.db import get_supabase_client
from PIL import Image

def optimize_original(image_bytes: bytes, max_dim: int = 1920, quality: int = 80) -> bytes | None:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        
        # Downscale only if it exceeds target dimension
        if img.width > max_dim or img.height > max_dim:
            img.thumbnail((max_dim, max_dim), cast(Any, Image.Resampling.LANCZOS))
            
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception as e:
        print(f"Failed to optimize image: {e}")
        return None

def main():
    print("=== STARTING RETROACTIVE ORIGINAL IMAGE OPTIMIZATION ===")
    supabase = get_supabase_client()
    
    # 1. Fetch all images
    res = supabase.table("media").select("*").eq("media_type", "image").execute()
    if not res.data:
        print("No media records found.")
        return

    items = cast(list[dict[str, Any]], res.data or [])
    print(f"Found {len(items)} image records to inspect.")
    
    optimized_count = 0
    
    for item in items:
        media_id = str(item["id"])
        url = str(item["url"])
        storage_path = str(item.get("storage_path") or "")
        filename = str(item.get("filename") or "")
        file_size = int(item.get("file_size") or 0)
        
        # Only optimize if file size is larger than 1MB (1024 * 1024 bytes)
        if file_size <= 1 * 1024 * 1024:
            print(f"Skipping already optimized image: {filename} ({file_size // 1024}KB)")
            continue
            
        print(f"Optimizing original image: {filename} ({file_size // 1024 // 1024}MB)...")
        
        try:
            # 2. Download original
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                img_bytes = response.read()
                
            # 3. Downscale / Compress
            opt_bytes = optimize_original(img_bytes, max_dim=1920, quality=80)
            if not opt_bytes:
                print(f"Failed to optimize bytes for {filename}.")
                continue
                
            # 4. Upload back to same path (upsert=True)
            supabase.storage.from_("memories").upload(
                path=storage_path,
                file=opt_bytes,
                file_options={"content-type": "image/jpeg", "upsert": "true"},
            )
            
            # Re-generate the signed URL just in case, and update DB file size
            signed_res = cast(dict[str, Any], supabase.storage.from_("memories").create_signed_url(storage_path, 31536000))
            new_url = signed_res.get("signedURL") or signed_res.get("signed_url") or ""
            
            update_data: dict[str, Any] = {
                "file_size": len(opt_bytes)
            }
            if new_url:
                update_data["url"] = new_url
                
            supabase.table("media").update(update_data).eq("id", media_id).execute()
            
            print(f"Successfully optimized {filename} down to {len(opt_bytes)//1024}KB!")
            optimized_count += 1
            
        except Exception as e:
            print(f"Error optimizing {filename}: {e}")
            
    print(f"=== COMPLETED: Optimized {optimized_count} original images! ===")

if __name__ == "__main__":
    main()

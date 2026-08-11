import os
import uuid
import httpx
from datetime import datetime, timezone
from typing import cast, Any
from fastapi import HTTPException
from app.core.db import get_supabase_client
from app.schemas.domain import MediaResponse

class VideoService:
    @staticmethod
    @staticmethod
    async def process_video_job(job_id: str, memory_id: str, user_id: str, dimension: str | None = None) -> None:
        supabase = get_supabase_client()
        
        def update_job_status(status: str, error_msg: str | None = None, result_media_id: str | None = None):
            now = datetime.now(timezone.utc).isoformat()
            data = {"status": status, "updated_at": now}
            if error_msg:
                data["error_message"] = error_msg
            if result_media_id:
                data["result_media_id"] = result_media_id
            supabase.table("video_jobs").update(data).eq("id", job_id).execute()

        # Update to processing
        update_job_status("processing")
        
        try:
            # 1. Fetch ALL media from the memory
            res = supabase.table("media")\
                .select("*")\
                .eq("memory_id", memory_id)\
                .order("created_at", desc=False)\
                .limit(30)\
                .execute()
            
            media_items = cast(list[dict[str, Any]], res.data or [])
            if len(media_items) < 2:
                update_job_status("failed", "Not enough media items in this memory to create a video.")
                return

            # 2. Download media to /tmp
            tmp_dir = f"/tmp/memoryverse_reel_{uuid.uuid4().hex}"
            os.makedirs(tmp_dir, exist_ok=True)
            
            downloaded_paths = []
            async with httpx.AsyncClient() as client:
                for idx, item in enumerate(media_items):
                    url = item["url"]
                    if not url:
                        continue
                    try:
                        resp = await client.get(url, timeout=30)
                        resp.raise_for_status()
                        ext = ".mp4" if item["media_type"] == "video" else ".jpg"
                        file_path = os.path.join(tmp_dir, f"media_{idx:03d}{ext}")
                        with open(file_path, "wb") as f:
                            f.write(resp.content)
                        downloaded_paths.append((file_path, item["media_type"]))
                    except Exception as e:
                        print(f"Failed to download {url}: {e}")
            
            if len(downloaded_paths) < 2:
                update_job_status("failed", "Not enough valid media items could be downloaded.")
                return

            # 3. Use MoviePy to stitch
            output_filename = f"memory_video_{uuid.uuid4().hex}.mp4"
            output_path = os.path.join(tmp_dir, output_filename)
            
            from PIL import Image, ImageOps
            if dimension == "9:16":
                TARGET_SIZE = (1080, 1920)
            else:
                TARGET_SIZE = (1920, 1080)
            
            # Since moviepy requires a uniform timeline, we'll convert images to clips 
            # and resize videos to target size
            from moviepy import ImageSequenceClip, VideoFileClip, concatenate_videoclips
            clips = []
            
            for path, mtype in downloaded_paths:
                if mtype == "image":
                    with Image.open(path) as pil_img:
                        pil_img = pil_img.convert("RGB")
                        pil_img = ImageOps.fit(pil_img, TARGET_SIZE, Image.Resampling.LANCZOS)
                        proc_path = path + "_proc.jpg"
                        pil_img.save(proc_path, "JPEG")
                        # 2 second duration for images
                        img_clip = ImageSequenceClip([proc_path], fps=24, durations=[2.0])
                        clips.append(img_clip)
                else:
                    try:
                        v_clip = VideoFileClip(path)
                        # Crop/Resize to match target
                        # Basic resize for now to avoid complex moviepy resizing logic in this scope
                        v_clip = v_clip.resized(new_size=TARGET_SIZE)
                        # Limit long videos to 10 seconds to avoid giant outputs
                        if v_clip.duration and v_clip.duration > 10:
                            v_clip = v_clip.subclipped(0, 10)
                        clips.append(v_clip)
                    except Exception as e:
                        print(f"Failed to process video clip {path}: {e}")
            
            if not clips:
                update_job_status("failed", "Media processing failed.")
                return

            final_clip = concatenate_videoclips(clips, method="compose")
            final_clip.write_videofile(
                output_path,
                codec="libx264",
                audio=True,
                preset="ultrafast",
                logger=None,
                fps=24
            )
            final_clip.close()
            for c in clips:
                c.close()

            # 4. Upload to Supabase Storage
            storage_path = f"{user_id}/reels/{output_filename}"
            with open(output_path, "rb") as f:
                supabase.storage.from_("memories").upload(
                    file=f,
                    path=storage_path,
                    file_options={"content-type": "video/mp4", "upsert": "true"}
                )

            public_url = supabase.storage.from_("memories").get_public_url(storage_path)

            # 5. Insert into media table
            now = datetime.now(timezone.utc).isoformat()
            
            # Get the vault_id from the memory if possible
            mem_res = supabase.table("memories").select("vault_id").eq("id", memory_id).execute()
            vault_id = mem_res.data[0].get("vault_id") if mem_res.data else None

            media_data = {
                "memory_id": memory_id,
                "vault_id": vault_id,
                "owner_id": user_id,
                "filename": output_filename,
                "storage_path": storage_path,
                "url": public_url,
                "media_type": "video",
                "file_size": os.path.getsize(output_path),
                "mime_type": "video/mp4",
                "duration": int(final_clip.duration) if final_clip.duration else 0,
                "created_at": now
            }
            
            insert_res = supabase.table("media").insert(media_data).execute()
            if not insert_res.data:
                update_job_status("failed", "Failed to save generated media metadata.")
                return
                
            new_media = insert_res.data[0]
            update_job_status("completed", result_media_id=new_media["id"])

        except Exception as e:
            update_job_status("failed", str(e))


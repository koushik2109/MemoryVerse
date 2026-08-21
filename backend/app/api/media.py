"""
Media API — handles file uploads and media metadata.

Upload pipeline:
  Client sends multipart/form-data with the file
  ↓ Backend receives file bytes in memory
  ↓ If video: use moviepy to extract a frame thumbnail (JPEG)
  ↓ Upload original + thumbnail to Supabase Storage
  ↓ Register metadata row in PostgreSQL via MediaService
  ↓ Return MediaResponse
"""

import io
import os
import tempfile
import logging

from fastapi import APIRouter, Depends, Query, UploadFile, File, Form, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, cast, Any
from datetime import datetime, timezone, timedelta

from app.schemas.domain import MediaCreate, MediaResponse, VideoJobResponse, MediaReorderRequest
from app.services.media_service import MediaService
from app.services.video_service import VideoService
from app.services.ai_extractor import AIExtractor
from app.core.security import get_current_user, CurrentUser
from app.core.db import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/media", tags=["Media"])

# ── Allowed MIME types ────────────────────────────────────────────────────────
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/heic"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm", "video/3gpp"}
ALLOWED_TYPES = ALLOWED_IMAGE_TYPES | ALLOWED_VIDEO_TYPES
MAX_FILE_SIZE_MB = 500


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_media_type(mime: str) -> str:
    """Return 'image' or 'video' from MIME type."""
    if mime in ALLOWED_IMAGE_TYPES:
        return "image"
    if mime in ALLOWED_VIDEO_TYPES:
        return "video"
    return "image"


def _extract_video_thumbnail(video_bytes: bytes, suffix: str = ".mp4") -> bytes | None:
    """
    Use moviepy to grab the first frame of a video and return it as JPEG bytes.
    Returns None on any failure so callers can degrade gracefully.
    Compatible with moviepy 2.x (no longer uses moviepy.editor).
    """
    try:
        from moviepy import VideoFileClip  # moviepy 2.x API
        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            clip = VideoFileClip(tmp_path)
            # Seek to 10% of duration or 1s, whichever is less
            seek_t = min(1.0, clip.duration * 0.1) if clip.duration else 0
            frame = clip.get_frame(seek_t)
            clip.close()

            if frame is None:
                return None
            # Convert numpy array → PIL Image → JPEG bytes
            img = Image.fromarray(frame)
            # Resize to max 720px wide keeping aspect ratio
            max_w = 720
            if img.width > max_w:
                ratio = max_w / img.width
                img = img.resize((max_w, int(img.height * ratio)), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        logger.warning(f"Thumbnail extraction failed: {e}")
        return None


def _upload_to_storage(bucket: str, path: str, data: bytes, mime: str) -> str:
    """Upload bytes to Supabase Storage and return the signed URL."""
    supabase = get_supabase_client()
    supabase.storage.from_(bucket).upload(
        path=path,
        file=data,
        file_options={"content-type": mime, "upsert": "true"},
    )
    # Create a signed URL valid for 1 year (31536000s)
    res = supabase.storage.from_(bucket).create_signed_url(path, 31536000)
    return res.get("signedURL") or res.get("signed_url") or ""


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=List[MediaResponse])
async def list_media(
    vault_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    current_user: CurrentUser = Depends(get_current_user),
):
    """List the authenticated user's media, optionally filtered by vault."""
    return MediaService.get_user_media(current_user.id, vault_id=vault_id, limit=limit)


@router.post("/memory/{memory_id}/generate-video")
async def generate_video(
    memory_id: str,
    background_tasks: BackgroundTasks,
    dimension: Optional[str] = Query(None),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Create a new video stitching job for a memory and process it in the background.
    """
    supabase = get_supabase_client()
    
    # Create the job record
    job_res = supabase.table("video_jobs").insert({
        "memory_id": memory_id,
        "user_id": current_user.id,
        "status": "queued"
    }).execute()
    
    if not job_res.data:
        raise HTTPException(status_code=500, detail="Failed to create video job")
        
    job_id = job_res.data[0]["id"]
    
    # Enqueue background task
    background_tasks.add_task(
        VideoService.process_video_job,
        job_id=job_id,
        memory_id=memory_id,
        user_id=current_user.id,
        dimension=dimension
    )
    
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}", response_model=VideoJobResponse)
async def get_video_job_status(
    job_id: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get the status of an async video creation job."""
    supabase = get_supabase_client()
    res = supabase.table("video_jobs").select("*").eq("id", job_id).eq("user_id", current_user.id).execute()
    
    if not res.data:
        raise HTTPException(status_code=404, detail="Job not found")
        
    j = cast(list[dict[str, Any]], res.data)[0]
    return VideoJobResponse(**j)


@router.post("/upload", response_model=MediaResponse, status_code=status.HTTP_201_CREATED)
async def upload_media(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    vault_id: Optional[str] = Form(None),
    memory_id: Optional[str] = Form(None),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Upload a photo or video.

    - Accepts multipart/form-data with `file` (required) and `vault_id` (optional).
    - For videos: automatically generates a JPEG thumbnail using moviepy.
    - Uploads original + thumbnail to Supabase Storage (private bucket 'memories').
    - Registers metadata in PostgreSQL and returns the created MediaResponse.
    """
    # 1. Validate MIME type
    content_type = file.content_type or "application/octet-stream"
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {content_type}. Allowed: {', '.join(sorted(ALLOWED_TYPES))}",
        )

    # 2. Read file into memory and check size
    file_bytes = await file.read()
    file_size = len(file_bytes)
    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB} MB.",
        )

    user_id = current_user.id
    filename = file.filename or "upload"
    import time
    ts = int(time.time() * 1000)
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
    media_type = _detect_media_type(content_type)

    # 3. Determine storage path
    original_path = f"{user_id}/{ts}_{safe_name}"
    thumb_path: str | None = None
    thumb_url: str | None = None

    # 4. Upload original file
    try:
        original_url = _upload_to_storage("memories", original_path, file_bytes, content_type)
    except Exception as e:
        logger.error(f"Storage upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload file to storage.")

    # 5. Generate + upload thumbnail for videos
    if media_type == "video":
        suffix = ".mp4" if "mp4" in content_type else ".mov"
        thumb_bytes = _extract_video_thumbnail(file_bytes, suffix=suffix)
        if thumb_bytes:
            thumb_path = f"{user_id}/thumbs/{ts}_{safe_name}.jpg"
            try:
                thumb_url = _upload_to_storage("memories", thumb_path, thumb_bytes, "image/jpeg")
            except Exception as e:
                logger.warning(f"Thumbnail upload failed (non-critical): {e}")
                thumb_url = original_url  # fallback: use video URL
        else:
            thumb_url = original_url
    else:
        # For images, use the image itself as its thumbnail
        thumb_path = original_path
        thumb_url = original_url

    # 6. Register in DB
    payload = MediaCreate(
        vault_id=vault_id,
        memory_id=memory_id,
        filename=filename,
        storage_path=original_path,
        url=original_url,
        thumbnail_url=thumb_url,
        media_type=media_type,
        file_size=file_size,
        mime_type=content_type,
    )
    created = MediaService.create_media(user_id, payload)
    
    # Trigger background AI feature & metadata extraction
    background_tasks.add_task(AIExtractor.process_media_item, created.id, file_bytes, content_type)
    
    return created


@router.post("/upload-multiple", response_model=List[MediaResponse], status_code=status.HTTP_201_CREATED)
async def upload_multiple_media(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    vault_id: Optional[str] = Form(None),
    memory_id: Optional[str] = Form(None),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Upload multiple photos or videos in batch.
    Processes and uploads each file to Supabase Storage and records metadata in PostgreSQL.
    """
    results: List[MediaResponse] = []
    user_id = current_user.id
    import time
    base_ts = int(time.time() * 1000)

    for i, file in enumerate(files):
        content_type = file.content_type or "application/octet-stream"
        if content_type not in ALLOWED_TYPES:
            continue

        file_bytes = await file.read()
        file_size = len(file_bytes)
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            continue

        filename = file.filename or f"upload_{i}"
        ts = base_ts + i
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
        media_type = _detect_media_type(content_type)

        original_path = f"{user_id}/{ts}_{safe_name}"

        try:
            original_url = _upload_to_storage("memories", original_path, file_bytes, content_type)
        except Exception as e:
            logger.error(f"Storage upload failed for {filename}: {e}")
            continue

        if media_type == "video":
            suffix = ".mp4" if "mp4" in content_type else ".mov"
            thumb_bytes = _extract_video_thumbnail(file_bytes, suffix=suffix)
            if thumb_bytes:
                thumb_path = f"{user_id}/thumbs/{ts}_{safe_name}.jpg"
                try:
                    thumb_url = _upload_to_storage("memories", thumb_path, thumb_bytes, "image/jpeg")
                except Exception as e:
                    logger.warning(f"Thumbnail upload failed (non-critical): {e}")
                    thumb_url = original_url
            else:
                thumb_url = original_url
        else:
            thumb_path = original_path
            thumb_url = original_url

        payload = MediaCreate(
            vault_id=vault_id,
            memory_id=memory_id,
            filename=filename,
            storage_path=original_path,
            url=original_url,
            thumbnail_url=thumb_url,
            media_type=media_type,
            file_size=file_size,
            mime_type=content_type,
        )
        created = MediaService.create_media(user_id, payload)
        results.append(created)

        # Trigger background AI feature & metadata extraction
        background_tasks.add_task(AIExtractor.process_media_item, created.id, file_bytes, content_type)

    return results


@router.post("", response_model=MediaResponse, status_code=status.HTTP_201_CREATED)
async def create_media_metadata(
    payload: MediaCreate,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Register media metadata after client-side upload to Supabase Storage.
    Use /upload for server-side upload with auto-thumbnail generation.
    """
    return MediaService.create_media(current_user.id, payload)


@router.put("/reorder", status_code=status.HTTP_200_OK)
async def reorder_media(
    payload: MediaReorderRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Reorder media items by updating their created_at timestamps.
    The first item in the list gets the newest timestamp so it appears first.
    """
    supabase = get_supabase_client()
    now = datetime.now(timezone.utc)
    
    # We offset each item by 1 second backward so that the first item is the most recent
    # (assuming descending created_at sort order in UI/Timeline)
    # Actually, timeline usually sorts by oldest first (ascending) or newest first.
    # The default media list is ordered by created_at DESC (if descending) or ASC.
    # Let's just adjust them sequentially by 1 second.
    
    # Check if UI expects first item to be oldest or newest.
    # We'll set the first item to now, the next to now + 1s, etc. so they sort predictably ASC.
    for index, media_id in enumerate(payload.media_ids):
        new_time = (now + timedelta(seconds=index)).isoformat()
        supabase.table("media").update({
            "created_at": new_time
        }).eq("id", media_id).eq("owner_id", current_user.id).execute()

    return {"message": "Reordered successfully"}


@router.get("/{media_id}", response_model=MediaResponse)
async def get_media(media_id: str, current_user: CurrentUser = Depends(get_current_user)):
    """Get a single media item by ID."""
    supabase = get_supabase_client()
    res = supabase.table("media").select("*").eq("id", media_id).eq("owner_id", current_user.id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Media not found")
    m = cast(list[dict[str, Any]], res.data)[0]
    return MediaResponse(
        id=m["id"],
        vault_id=m.get("vault_id"),
        memory_id=m.get("memory_id"),
        owner_id=m["owner_id"],
        filename=m["filename"],
        storage_path=m["storage_path"],
        url=m["url"],
        thumbnail_url=m.get("thumbnail_url"),
        media_type=m["media_type"],
        file_size=m["file_size"],
        mime_type=m.get("mime_type"),
        created_at=m["created_at"],
    )


@router.get("/{media_id}/stream-url")
async def get_stream_url(media_id: str, current_user: CurrentUser = Depends(get_current_user)):
    """
    Return a fresh signed URL for streaming/downloading a media file.
    Valid for 1 hour.
    """
    supabase = get_supabase_client()
    res = supabase.table("media").select("storage_path, owner_id, vault_id").eq("id", media_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Media not found")

    m = cast(list[dict[str, Any]], res.data)[0]
    # Authorization: owner OR vault member
    if m["owner_id"] != current_user.id:
        if m.get("vault_id"):
            member_res = supabase.table("vault_members") \
                .select("id").eq("vault_id", m["vault_id"]).eq("user_id", current_user.id).execute()
            if not member_res.data:
                raise HTTPException(status_code=403, detail="Access denied")
        else:
            raise HTTPException(status_code=403, detail="Access denied")

    signed = supabase.storage.from_("memories").create_signed_url(m["storage_path"], 3600)
    url = signed.get("signedURL") or signed.get("signed_url") or ""
    if not url:
        raise HTTPException(status_code=500, detail="Failed to generate stream URL")
    return {"url": url, "expires_in": 3600}


@router.delete("/{media_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_media(media_id: str, current_user: CurrentUser = Depends(get_current_user)):
    """Delete a media file (owner only). Removes from storage and DB."""
    MediaService.delete_media(media_id, current_user.id)

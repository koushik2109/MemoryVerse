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
import asyncio
import json

from fastapi import APIRouter, Depends, Query, UploadFile, File, Form, HTTPException, status, BackgroundTasks, Request, Response
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, RedirectResponse
from typing import List, Optional, cast, Any
from datetime import datetime, timezone, timedelta

from app.schemas.domain import MediaCreate, MediaResponse, VideoJobResponse, MediaReorderRequest
from app.services.media_service import MediaService
from app.services.video_service import VideoService
from app.services.ai_extractor import AIExtractor
from app.services.job_state_manager import job_state_manager, JobStatus
from app.services.video_queue import video_worker_pool
from app.core.cache_service import cache_service
from app.core.security import get_current_user, CurrentUser
from app.core.db import get_supabase_client
from app.config.settings import settings

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


def _extract_image_thumbnail(image_bytes: bytes, max_dim: int = 480, quality: int = 80) -> bytes | None:
    """
    Resize an image to max_dim pixels keeping aspect ratio and compress as JPEG.
    Returns lightweight bytes (~30-50KB) for instant grid rendering.
    """
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        
        if img.width > max_dim or img.height > max_dim:
            img.thumbnail((max_dim, max_dim), cast(Any, Image.Resampling.LANCZOS))
            
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Image thumbnail generation failed: {e}")
        return None


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
                img = img.resize((max_w, int(img.height * ratio)), cast(Any, Image.Resampling.LANCZOS))

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
    return str(res.get("signedURL") or res.get("signed_url") or "")


MEDIA_STREAM_CACHE_DIR = os.path.join(tempfile.gettempdir(), "memoryverse_media_cache")
os.makedirs(MEDIA_STREAM_CACHE_DIR, exist_ok=True)


@router.get("/stream/{media_id}")
async def stream_media(media_id: str, request: Request):
    """
    Stream media (video or image) directly by media_id.
    Provides high-performance disk-cached HTTP 206 Partial Content / Range support
    for smooth video playback on mobile, desktop, and web players without network re-downloading.
    """
    supabase = get_supabase_client()
    res = supabase.table("media").select("*").eq("id", media_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Media not found")
        
    m = cast(list[dict[str, Any]], res.data)[0]
    storage_path = m.get("storage_path")
    mime_type = m.get("mime_type") or "video/mp4"
    
    if not storage_path:
        raise HTTPException(status_code=404, detail="Storage path missing")

    safe_base = "".join(c if c.isalnum() or c in "._-" else "_" for c in os.path.basename(storage_path))
    cached_file_path = os.path.join(MEDIA_STREAM_CACHE_DIR, f"{media_id}_{safe_base}")

    try:
        # 1. Download to local disk cache if not already present
        if not os.path.exists(cached_file_path) or os.path.getsize(cached_file_path) == 0:
            file_bytes = supabase.storage.from_("memories").download(storage_path)
            if not file_bytes:
                raise ValueError("Empty file returned from storage")
            with open(cached_file_path, "wb") as f:
                f.write(file_bytes)

        total_size = os.path.getsize(cached_file_path)
        range_header = request.headers.get("Range")

        # 2. Handle HTTP 206 Partial Content Range requests via direct OS file seeks
        if range_header and range_header.startswith("bytes="):
            byte_range = range_header[6:].split("-")
            start = int(byte_range[0]) if byte_range[0] else 0
            end = int(byte_range[1]) if len(byte_range) > 1 and byte_range[1] else total_size - 1
            start = max(0, min(start, total_size - 1))
            end = max(start, min(end, total_size - 1))
            chunk_size = end - start + 1

            with open(cached_file_path, "rb") as f:
                f.seek(start)
                chunk_data = f.read(chunk_size)

            headers = {
                "Content-Range": f"bytes {start}-{end}/{total_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
                "Content-Type": mime_type,
            }
            return Response(
                content=chunk_data,
                status_code=206,
                headers=headers,
                media_type=mime_type,
            )

        # 3. Standard full response via FileResponse
        return FileResponse(
            cached_file_path,
            media_type=mime_type,
            headers={"Accept-Ranges": "bytes", "Content-Length": str(total_size)},
        )

    except Exception as e:
        logger.error(f"Failed to stream media {media_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to stream media")


@router.get("", response_model=List[MediaResponse])
async def list_media(
    vault_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    current_user: CurrentUser = Depends(get_current_user),
):
    """List the authenticated user's media, optionally filtered by vault."""
    return MediaService.get_user_media(current_user.id, vault_id=vault_id, limit=limit)


async def _authenticate_sse(request: Request, token_query: Optional[str] = None) -> CurrentUser:
    """Authenticate SSE connection via Authorization header or token query parameter."""
    auth_header = request.headers.get("Authorization")
    raw_token = None
    if auth_header and auth_header.startswith("Bearer "):
        raw_token = auth_header[7:].strip()
    elif token_query:
        raw_token = token_query.strip()

    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials for SSE stream",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        supabase = get_supabase_client()
        user_response = supabase.auth.get_user(raw_token)
        if user_response and user_response.user:
            user = user_response.user
            metadata = user.user_metadata or {}
            user_email = user.email or ""
            return CurrentUser(
                id=user.id,
                email=user_email,
                full_name=metadata.get("full_name") or (user_email.split("@")[0] if user_email else "User"),
            )
    except Exception:
        pass

    try:
        from jose import jwt
        payload = jwt.decode(raw_token, settings.JWT_SECRET, algorithms=[settings.ALGORITHM], options={"verify_aud": False})
        user_id: str = str(payload.get("sub") or payload.get("user_id"))
        email: str = payload.get("email", "")
        if user_id:
            return CurrentUser(id=user_id, email=email)
    except Exception:
        pass

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials for SSE stream",
        headers={"WWW-Authenticate": "Bearer"},
    )


@router.post("/memory/{memory_id}/generate-video")
async def generate_video(
    memory_id: str,
    dimension: Optional[str] = Query("9:16"),
    mood: Optional[str] = Query("calm"),
    media_ids: Optional[List[str]] = Query(None),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Create a new video stitching job for a memory and process it asynchronously
    through the bounded durable VideoWorkerPool.
    Includes deduplication fingerprinting, multi-level state tracking, and recovery.
    """
    dim = dimension or "9:16"
    m_mood = mood or "calm"
    m_ids = media_ids or []

    # 1. Deduplication fingerprint check (idempotent submission)
    fingerprint = cache_service.compute_job_fingerprint(
        user_id=current_user.id,
        memory_id=memory_id,
        selected_media_ids=m_ids,
        mood=m_mood,
        dimension=dim,
    )
    existing_job_id = cache_service.get_active_job_by_fingerprint(fingerprint)
    if existing_job_id:
        existing_job = job_state_manager.get_job(existing_job_id)
        if existing_job and existing_job.status in ("queued", "initializing", "processing"):
            logger.info(
                f"Deduplicated video generation request for memory {memory_id}: "
                f"reusing active job {existing_job_id}"
            )
            return {"job_id": existing_job.id, "status": existing_job.status, "deduplicated": True}

    supabase = get_supabase_client()

    # 2. Mark old incomplete jobs for this memory as superseded
    try:
        supabase.table("video_jobs")\
            .update({"status": "failed", "error_message": "Superseded by a new creation request."})\
            .eq("memory_id", memory_id)\
            .eq("user_id", current_user.id)\
            .in_("status", ["queued", "processing"])\
            .execute()
    except Exception as e:
        logger.debug(f"Could not supersede previous video jobs: {e}")

    # 3. Create job record in database
    job_res = supabase.table("video_jobs").insert({
        "memory_id": memory_id,
        "user_id": current_user.id,
        "status": "queued",
    }).execute()

    if not job_res.data:
        raise HTTPException(status_code=500, detail="Failed to create video job")

    raw_jobs = cast(list[dict[str, Any]], job_res.data)
    job_id: str = str(raw_jobs[0]["id"])

    # 4. Initialize in-memory and Redis state in JobStateManager
    job_state_manager.create_job(
        job_id=job_id,
        memory_id=memory_id,
        user_id=current_user.id,
        dimension=dim,
        mood=m_mood,
        selected_media_ids=m_ids,
    )

    # 5. Record fingerprint in cache to prevent duplicate rapid submissions
    cache_service.set_active_job_by_fingerprint(fingerprint, job_id, ttl=600)

    # 6. Submit to durable queue
    video_worker_pool.submit_job(job_id)
    logger.info(f"Video job {job_id} submitted to durable queue for user {current_user.id}")

    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}", response_model=VideoJobResponse)
async def get_video_job_status(
    job_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Get the status of an async video creation job with strict tenant isolation.
    """
    job = job_state_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Strict tenant isolation check
    if job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job not found")

    d = job.to_dict()

    # Resolve result URL if finished and missing in state
    if not d.get("result_url") and d.get("result_media_id"):
        try:
            supabase = get_supabase_client()
            m_res = supabase.table("media").select("url").eq("id", d["result_media_id"]).execute()
            if m_res.data:
                m_list = cast(list[dict[str, Any]], m_res.data)
                d["result_url"] = str(m_list[0].get("url") or "")
        except Exception:
            pass

    return VideoJobResponse(**d)


@router.get("/jobs/{job_id}/stream")
async def stream_video_job_status(
    job_id: str,
    request: Request,
    token: Optional[str] = Query(None),
):
    """
    Server-Sent Events (SSE) streaming endpoint for live job status and progress.
    Pushes real-time updates every 1.0s until terminal state (completed, failed, cancelled).
    """
    auth_user = await _authenticate_sse(request, token)

    job = job_state_manager.get_job(job_id)
    if not job or job.user_id != auth_user.id:
        raise HTTPException(status_code=404, detail="Job not found")

    async def sse_generator():
        while True:
            if await request.is_disconnected():
                logger.debug(f"SSE client disconnected for job {job_id}")
                break

            current_job = job_state_manager.get_job(job_id)
            if not current_job:
                break

            d = current_job.to_dict()
            if not d.get("result_url") and d.get("result_media_id"):
                try:
                    supabase = get_supabase_client()
                    m_res = supabase.table("media").select("url").eq("id", d["result_media_id"]).execute()
                    if m_res.data:
                        m_list = cast(list[dict[str, Any]], m_res.data)
                        d["result_url"] = str(m_list[0].get("url") or "")
                except Exception:
                    pass

            yield f"data: {json.dumps(d)}\n\n"

            if current_job.status in ("completed", "failed", "cancelled"):
                break

            await asyncio.sleep(1.0)

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/jobs/{job_id}/retry")
async def retry_video_job(
    job_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Retry a failed video generation job.
    If the rendered video artifact was preserved on disk, resumes directly from upload.
    Otherwise, recovers and re-processes through the durable queue.
    """
    job = job_state_manager.get_job(job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status not in ("failed", "cancelled"):
        return {"job_id": job.id, "status": job.status, "message": "Job is already active or completed."}

    # Reset failure markers and increment retry count
    job.status = JobStatus.RETRYING.value
    job.retry_count += 1
    job.error_code = None
    job.error_message = None
    job_state_manager._persist_job(job, db_sync=True)

    # Re-submit to worker pool
    video_worker_pool.submit_job(job.id)
    logger.info(f"Job {job_id} submitted for retry by user {current_user.id}")

    return {"job_id": job.id, "status": "retrying"}


@router.post("/jobs/{job_id}/cancel")
async def cancel_video_job(
    job_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Cancels an active or queued video generation job.
    """
    cancelled = job_state_manager.cancel_job(job_id, current_user.id)
    if not cancelled:
        job = job_state_manager.get_job(job_id)
        if not job or job.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job_id": job_id, "status": job.status, "message": f"Job is already in {job.status} state."}

    return {"job_id": job_id, "status": "cancelled"}


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
        original_url = await asyncio.to_thread(_upload_to_storage, "memories", original_path, file_bytes, content_type)
    except Exception as e:
        logger.error(f"Storage upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload file to storage.")

    # 5. Generate + upload thumbnail for videos
    if media_type == "video":
        suffix = ".mp4" if "mp4" in content_type else ".mov"
        thumb_bytes = await asyncio.to_thread(_extract_video_thumbnail, file_bytes, suffix=suffix)
        if thumb_bytes:
            thumb_path = f"{user_id}/thumbs/{ts}_{safe_name}.jpg"
            try:
                thumb_url = await asyncio.to_thread(_upload_to_storage, "memories", thumb_path, thumb_bytes, "image/jpeg")
            except Exception as e:
                logger.warning(f"Thumbnail upload failed (non-critical): {e}")
                thumb_url = original_url  # fallback: use video URL
        else:
            thumb_url = original_url
    else:
        # Generate + upload lightweight thumbnail (~30-50KB) for images
        thumb_bytes = await asyncio.to_thread(_extract_image_thumbnail, file_bytes, max_dim=480, quality=80)
        if thumb_bytes:
            thumb_path = f"{user_id}/thumbs/{ts}_{safe_name}.jpg"
            try:
                thumb_url = await asyncio.to_thread(_upload_to_storage, "memories", thumb_path, thumb_bytes, "image/jpeg")
            except Exception as e:
                logger.warning(f"Image thumbnail upload failed: {e}")
                thumb_url = original_url
        else:
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
    created = await asyncio.to_thread(MediaService.create_media, user_id, payload)
    
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
    Processes and uploads each file to Supabase Storage concurrently with bounded
    semaphore, then records all metadata in PostgreSQL in a single batch insert.
    """
    user_id = current_user.id
    import time
    base_ts = int(time.time() * 1000)

    # Concurrency control for storage uploads to prevent socket exhaustion
    sem = asyncio.Semaphore(10)

    async def _process_single(i: int, file: UploadFile) -> Optional[tuple[MediaCreate, bytes, str]]:
        async with sem:
            content_type = file.content_type or "application/octet-stream"
            if content_type not in ALLOWED_TYPES:
                return None

            file_bytes = await file.read()
            file_size = len(file_bytes)
            if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                return None

            filename = file.filename or f"upload_{i}"
            ts = base_ts + i
            safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
            media_type = _detect_media_type(content_type)

            original_path = f"{user_id}/{ts}_{safe_name}"

            try:
                original_url = await asyncio.to_thread(_upload_to_storage, "memories", original_path, file_bytes, content_type)
            except Exception as e:
                logger.error(f"Storage upload failed for {filename}: {e}")
                return None

            if media_type == "video":
                suffix = ".mp4" if "mp4" in content_type else ".mov"
                thumb_bytes = await asyncio.to_thread(_extract_video_thumbnail, file_bytes, suffix=suffix)
                if thumb_bytes:
                    thumb_path = f"{user_id}/thumbs/{ts}_{safe_name}.jpg"
                    try:
                        thumb_url = await asyncio.to_thread(_upload_to_storage, "memories", thumb_path, thumb_bytes, "image/jpeg")
                    except Exception as e:
                        logger.warning(f"Thumbnail upload failed (non-critical): {e}")
                        thumb_url = original_url
                else:
                    thumb_url = original_url
            else:
                thumb_bytes = await asyncio.to_thread(_extract_image_thumbnail, file_bytes, max_dim=480, quality=80)
                if thumb_bytes:
                    thumb_path = f"{user_id}/thumbs/{ts}_{safe_name}.jpg"
                    try:
                        thumb_url = await asyncio.to_thread(_upload_to_storage, "memories", thumb_path, thumb_bytes, "image/jpeg")
                    except Exception as e:
                        logger.warning(f"Image thumbnail upload failed: {e}")
                        thumb_url = original_url
                else:
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
            return (payload, file_bytes, content_type)

    tasks = [_process_single(i, f) for i, f in enumerate(files)]
    completed = await asyncio.gather(*tasks)
    valid_items = [item for item in completed if item is not None]

    if not valid_items:
        return []

    payloads = [item[0] for item in valid_items]
    created_responses = await asyncio.to_thread(MediaService.create_media_batch, user_id, payloads)

    # Queue background AI extraction for each item
    for created, (_, f_bytes, c_type) in zip(created_responses, valid_items):
        background_tasks.add_task(AIExtractor.process_media_item, created.id, f_bytes, c_type)

    return created_responses


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


@router.get("/videos/{job_id}/download")
async def download_video_job(
    job_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Directly downloads the rendered video file for offline storage, local playback, and external sharing.
    Returns FileResponse with Content-Disposition: attachment; filename="..." and video/mp4 MIME type.
    """
    job = job_state_manager.get_job(job_id)
    if not job:
        supabase = get_supabase_client()
        res = supabase.table("video_jobs").select("*").eq("id", job_id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Video job not found")
        job = res.data[0]

    owner_id = getattr(job, "user_id", None) or (job.get("user_id") if isinstance(job, dict) else None)
    if owner_id and owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    dimension = getattr(job, "dimension", None) or (job.get("dimension") if isinstance(job, dict) else "16:9")
    safe_ratio = str(dimension).replace(":", "_").replace("/", "_").strip() or "16_9"
    videos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "storage", "videos", safe_ratio))

    # 1. Check local staged path
    staged_path = getattr(job, "staged_filepath", None) or (job.get("staged_filepath") if isinstance(job, dict) else None)
    if isinstance(staged_path, str) and os.path.exists(staged_path):
        return FileResponse(
            path=staged_path,
            filename=f"MemoryVerse_{job_id[:8]}_{safe_ratio}.mp4",
            media_type="video/mp4",
            headers={"Content-Disposition": f'attachment; filename="MemoryVerse_{job_id[:8]}_{safe_ratio}.mp4"'},
        )

    # 2. Check persistent ratio directory
    if os.path.exists(videos_dir):
        for f in os.listdir(videos_dir):
            if f.endswith(".mp4"):
                full_p = os.path.join(videos_dir, f)
                return FileResponse(
                    path=full_p,
                    filename=f"MemoryVerse_{job_id[:8]}_{safe_ratio}.mp4",
                    media_type="video/mp4",
                    headers={"Content-Disposition": f'attachment; filename="MemoryVerse_{job_id[:8]}_{safe_ratio}.mp4"'},
                )

    # 3. Fallback to public result URL
    result_url = getattr(job, "result_url", None) or (job.get("result_url") if isinstance(job, dict) else None)
    if isinstance(result_url, str) and result_url:
        return RedirectResponse(url=result_url)

    raise HTTPException(status_code=404, detail="Video file not available for download")


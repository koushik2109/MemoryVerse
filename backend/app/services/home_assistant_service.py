"""
HomeAssistantService - Layer 1 AI Assistant for MemoryVerse.
Handles application-level commands, entity resolution, controlled tool execution,
and structured action responses for the Flutter frontend navigation dispatcher.
"""

import re
import difflib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, model_validator

from app.core.db import get_supabase_client
from app.schemas.domain import MemoryCreate, VaultCreate
from app.services.memory_service import MemoryService
from app.services.vault_service import VaultService
from app.services.memory_readiness_service import memory_readiness_service
from app.services.job_state_manager import job_state_manager
from app.services.video_queue import video_worker_pool
from app.services.video_service import VideoService

logger = logging.getLogger(__name__)


class AssistantAction(BaseModel):
    type: str  # OPEN_MEMORY, OPEN_VAULT, WATCH_VIDEO_JOB, OPEN_VIDEO, OPEN_AI_DIRECTOR, SHOW_READINESS, SHOW_LIST
    memory_id: Optional[str] = None
    vault_id: Optional[str] = None
    job_id: Optional[str] = None
    media_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class HomeAssistantRequest(BaseModel):
    query: Optional[str] = None
    message: Optional[str] = None
    context_memory_id: Optional[str] = None
    context_vault_id: Optional[str] = None
    client_context: Optional[Dict[str, Any]] = None

    def get_query(self) -> str:
        return (self.query or self.message or "").strip()


class HomeAssistantResponse(BaseModel):
    message: str
    intent: str
    actions: List[AssistantAction] = []
    structured_action: Optional[Dict[str, Any]] = None
    suggested_chips: List[str] = []
    data: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def populate_structured_action(self):
        if not self.structured_action and self.actions:
            first = self.actions[0]
            params: Dict[str, Any] = {}
            if first.memory_id:
                params["memory_id"] = first.memory_id
            if first.vault_id:
                params["vault_id"] = first.vault_id
            if first.job_id:
                params["job_id"] = first.job_id
            if first.media_id:
                params["media_id"] = first.media_id
            if first.data:
                params.update(first.data)
            self.structured_action = {
                "action": first.type,
                "params": params,
            }
        return self


class HomeAssistantService:
    @staticmethod
    def process_command(user_id: str, query: str, context: Optional[Dict[str, Any]] = None) -> HomeAssistantResponse:
        """
        Main entry point for processing Home Assistant natural language commands.
        Uses deterministic pattern matching & entity resolution first, ensuring sub-50ms latency
        and strict server-side authorization.
        """
        q = query.strip()
        q_lower = q.lower()
        supabase = get_supabase_client()

        # -------------------------------------------------------------
        # 1. CREATE MEMORY
        # Examples: "Create a memory Summer Trip", "Create memory Birthday 2026"
        # -------------------------------------------------------------
        create_journal_match = re.search(
            r"^(?:create|make|new)\s+(?:a\s+)?(?:journal|memory)\s+(?:of|about|for|called|named|titled)\s+[\"']?(.+?)[\"']?$",
            q,
            re.IGNORECASE,
        )
        if create_journal_match:
            prompt = create_journal_match.group(1).strip(" .\"'")
            if prompt:
                return HomeAssistantResponse(
                    message=(
                        f"I’ll scan your accessible media for evidence of '{prompt}', "
                        "then let you review the grounded matches before creating the journal."
                    ),
                    intent="CREATE_MEMORY_DISCOVERY",
                    actions=[
                        AssistantAction(
                            type="CREATE_MEMORY",
                            data={
                                "initial_prompt": f"Create journal of {prompt}",
                                "auto_start_discovery": True,
                            },
                        )
                    ],
                    suggested_chips=["Grant Media Access", "Add Photos Manually"],
                )

        create_mem_match = re.search(
            r"^(?:create|make|add|new)\s+(?:a\s+)?memory(?:\s+(?:called|named|for|about|titled))?\s+[\"']?(.+?)[\"']?$",
            q,
            re.IGNORECASE,
        )
        if create_mem_match:
            title = create_mem_match.group(1).strip(" .\"'")
            if title:
                mem_service = MemoryService(supabase)
                now_date = datetime.now(timezone.utc)
                mem_payload = MemoryCreate(
                    title=title,
                    description=f"Created via AI Assistant on {now_date.strftime('%B %d, %Y')}",
                    memory_date=now_date,
                    tags=["assistant", "ai"],
                )
                try:
                    import asyncio
                    # Run sync/async safely
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as pool:
                                created_mem = pool.submit(asyncio.run, mem_service.create_memory(user_id, mem_payload)).result()
                        else:
                            created_mem = loop.run_until_complete(mem_service.create_memory(user_id, mem_payload))
                    except RuntimeError:
                        created_mem = asyncio.run(mem_service.create_memory(user_id, mem_payload))

                    mem_id = created_mem["id"]
                    return HomeAssistantResponse(
                        message=f"I've created your memory '{title}'. Let's add some photos to it.",
                        intent="CREATE_MEMORY",
                        actions=[
                            AssistantAction(type="OPEN_MEMORY", memory_id=mem_id, data={"title": title})
                        ],
                        suggested_chips=["Add Media", "Create with AI", "Check Readiness"],
                    )
                except Exception as e:
                    logger.exception(f"Error creating memory: {e}")
                    return HomeAssistantResponse(
                        message=f"Could not create memory '{title}': {str(e)}",
                        intent="CREATE_MEMORY",
                    )

        # -------------------------------------------------------------
        # 2. CREATE VAULT
        # Examples: "Create a vault called Family", "Create vault Friends"
        # -------------------------------------------------------------
        create_vault_match = re.search(
            r"^(?:create|make|new)\s+(?:a\s+)?vault(?:\s+(?:called|named|for))?\s+[\"']?(.+?)[\"']?$",
            q,
            re.IGNORECASE,
        )
        if create_vault_match:
            vault_name = create_vault_match.group(1).strip(" .\"'")
            if vault_name:
                try:
                    vault_payload = VaultCreate(
                        name=vault_name,
                        description=f"Personal Room created via AI Assistant",
                    )
                    created_vault = VaultService.create_vault(user_id, vault_payload)
                    return HomeAssistantResponse(
                        message=f"Created room '{vault_name}'. You can now organize shared memories here.",
                        intent="CREATE_VAULT",
                        actions=[
                            AssistantAction(type="OPEN_VAULT", vault_id=created_vault.id, data={"name": vault_name})
                        ],
                        suggested_chips=["Add Memories", "Share Room", "View Room"],
                    )
                except Exception as e:
                    logger.exception(f"Error creating vault: {e}")
                    return HomeAssistantResponse(
                        message=f"Could not create vault '{vault_name}': {str(e)}",
                        intent="CREATE_VAULT",
                    )

        # -------------------------------------------------------------
        # 3. LIST MEMORIES
        # Examples: "Show my memories", "List memories", "What memories do I have"
        # -------------------------------------------------------------
        if any(phrase in q_lower for phrase in ["show my memories", "list memories", "show memories", "what memories"]):
            mems = HomeAssistantService._fetch_user_memories(user_id)
            if not mems:
                return HomeAssistantResponse(
                    message="You haven't created any memories yet. Would you like to create one now?",
                    intent="LIST_MEMORIES",
                    suggested_chips=["Create Memory", "Explore Timeline"],
                )
            titles = [f"• {m['title']} ({len(m.get('media', []))} items)" for m in mems[:5]]
            summary = "\n".join(titles)
            count = len(mems)
            return HomeAssistantResponse(
                message=f"You have {count} memory{'ies' if count != 1 else ''}:\n{summary}",
                intent="LIST_MEMORIES",
                actions=[AssistantAction(type="SHOW_LIST", data={"type": "memories"})],
                suggested_chips=[m["title"] for m in mems[:3]],
            )

        # -------------------------------------------------------------
        # 4. LIST VAULTS
        # Examples: "Show my vaults", "List vaults", "Show rooms"
        # -------------------------------------------------------------
        if any(phrase in q_lower for phrase in ["show my vaults", "list vaults", "show rooms", "show vaults", "list rooms"]):
            vaults = HomeAssistantService._fetch_user_vaults(user_id)
            if not vaults:
                return HomeAssistantResponse(
                    message="You don't have any rooms or vaults yet. Would you like to create a Family or Friends room?",
                    intent="LIST_VAULTS",
                    suggested_chips=["Create Room Family", "Create Room Travel"],
                )
            names = [f"• {v['name']}" for v in vaults[:5]]
            return HomeAssistantResponse(
                message=f"You have {len(vaults)} room(s):\n" + "\n".join(names),
                intent="LIST_VAULTS",
                actions=[AssistantAction(type="SHOW_LIST", data={"type": "vaults"})],
                suggested_chips=[v["name"] for v in vaults[:3]],
            )

        # -------------------------------------------------------------
        # 5. HOW MANY PHOTOS / MEDIA COUNT
        # Examples: "How many photos are in Summer Trip?", "How many items in Swarnagiri Temple?"
        # -------------------------------------------------------------
        count_match = re.search(
            r"^(?:how many\s+(?:photos|videos|images|items|moments|pictures)\s+(?:are\s+)?(?:in|for)|media count (?:of|in))\s+[\"']?(.+?)[\"']?\??$",
            q,
            re.IGNORECASE,
        )
        if count_match:
            entity_name = count_match.group(1).strip(" ?.\"")
            resolved_mem = HomeAssistantService._resolve_memory_by_name(user_id, entity_name)
            if resolved_mem:
                media_list = resolved_mem.get("media") or []
                photos = sum(1 for m in media_list if m.get("media_type") == "image")
                videos = sum(1 for m in media_list if m.get("media_type") == "video")
                total = len(media_list)
                return HomeAssistantResponse(
                    message=f"'{resolved_mem['title']}' has {total} moment(s): {photos} photo(s) and {videos} video(s).",
                    intent="GET_MEMORY_MEDIA_COUNT",
                    actions=[AssistantAction(type="OPEN_MEMORY", memory_id=resolved_mem["id"])],
                    suggested_chips=["Check Readiness", "Generate Video", "Open Memory"],
                )
            else:
                return HomeAssistantResponse(
                    message=f"I couldn't find a memory named '{entity_name}'. Would you like to create it?",
                    intent="GET_MEMORY_MEDIA_COUNT",
                    suggested_chips=[f"Create memory {entity_name}", "Show my memories"],
                )

        # -------------------------------------------------------------
        # 6. CHECK READINESS
        # Examples: "Is Summer Trip ready for a video?", "Is Swarnagiri Temple ready?", "Check readiness of Summer Trip"
        # -------------------------------------------------------------
        ready_match = re.search(
            r"^(?:is\s+[\"']?(.+?)[\"']?\s+ready(?:\s+for\s+(?:a\s+)?video)?|check\s+readiness(?:\s+(?:of|for))?\s+[\"']?(.+?)[\"']?)\??$",
            q,
            re.IGNORECASE,
        )
        if ready_match:
            entity_name = (ready_match.group(1) or ready_match.group(2) or "").strip(" ?.\"")
            resolved_mem = HomeAssistantService._resolve_memory_by_name(user_id, entity_name)
            if resolved_mem:
                readiness = memory_readiness_service.evaluate_memory_readiness(user_id, resolved_mem["id"])
                return HomeAssistantResponse(
                    message=f"Readiness check for '{resolved_mem['title']}':\n{readiness.status_summary}",
                    intent="CHECK_MEMORY_READINESS",
                    actions=[
                        AssistantAction(
                            type="SHOW_READINESS",
                            memory_id=resolved_mem["id"],
                            data=readiness.model_dump(),
                        )
                    ],
                    suggested_chips=["Generate Video", "Open AI Director", "Open Memory"],
                )
            else:
                return HomeAssistantResponse(
                    message=f"I couldn't locate a memory called '{entity_name}' to check its readiness.",
                    intent="CHECK_MEMORY_READINESS",
                    suggested_chips=["Show my memories"],
                )

        # -------------------------------------------------------------
        # 7. GENERATE VIDEO
        # Examples: "Generate the video of Summer Trip", "Make a video from Swarnagiri Temple", "Generate Summer Trip"
        # -------------------------------------------------------------
        gen_match = re.search(
            r"^(?:generate|make|create|render)(?:\s+(?:the|a))?\s+(?:video|movie|reel)?(?:\s+(?:of|from|for))?\s+[\"']?(.+?)[\"']?(?:\s+video)?\??$",
            q,
            re.IGNORECASE,
        )
        if gen_match and not any(k in q_lower for k in ["memory", "vault", "room"]):
            entity_name = gen_match.group(1).strip(" ?.\"")
            resolved_mem = HomeAssistantService._resolve_memory_by_name(user_id, entity_name)
            if resolved_mem:
                mem_id = resolved_mem["id"]
                # 1. Check readiness
                readiness = memory_readiness_service.evaluate_memory_readiness(user_id, mem_id)
                if not readiness.is_ready_for_video:
                    if readiness.active_job_id:
                        return HomeAssistantResponse(
                            message=f"'{resolved_mem['title']}' is already being rendered ({readiness.active_job_progress}%).",
                            intent="GENERATE_MEMORY_VIDEO",
                            actions=[
                                AssistantAction(
                                    type="WATCH_VIDEO_JOB",
                                    memory_id=mem_id,
                                    job_id=readiness.active_job_id,
                                )
                            ],
                            suggested_chips=["Check Status", "Open Memory"],
                        )
                    else:
                        blocker_text = "; ".join(readiness.blockers)
                        return HomeAssistantResponse(
                            message=f"'{resolved_mem['title']}' is not ready for video generation yet: {blocker_text}",
                            intent="GENERATE_MEMORY_VIDEO",
                            actions=[AssistantAction(type="OPEN_MEMORY", memory_id=mem_id)],
                            suggested_chips=["Add Media", "Check Readiness"],
                        )

                # 2. Extract optional duration / aspect ratio if present
                duration = 30
                dur_match = re.search(r"(\d+)\s*(?:sec|second|s)", q_lower)
                if dur_match:
                    duration = int(dur_match.group(1))

                aspect_ratio = "16:9"
                if "reel" in q_lower or "portrait" in q_lower or "9:16" in q_lower or "tiktok" in q_lower:
                    aspect_ratio = "9:16"

                # 3. Create durable VideoJob
                new_job = video_worker_pool.submit_video_job(
                    user_id=user_id,
                    memory_id=mem_id,
                    conversation_id=f"home_assistant_{user_id[:8]}",
                    specification={
                        "memory_id": mem_id,
                        "duration_seconds": duration,
                        "aspect_ratio": aspect_ratio,
                        "mood": "nostalgic",
                        "music_style": "cinematic_ambient",
                        "pacing": "moderate",
                    },
                )

                return HomeAssistantResponse(
                    message=(
                        f"Started generating a {duration}s ({aspect_ratio}) memory video for '{resolved_mem['title']}'. "
                        "You can track live progress below."
                    ),
                    intent="GENERATE_MEMORY_VIDEO",
                    actions=[
                        AssistantAction(
                            type="WATCH_VIDEO_JOB",
                            memory_id=mem_id,
                            job_id=new_job.job_id,
                            data={"title": resolved_mem["title"], "duration": duration},
                        )
                    ],
                    suggested_chips=["Check Status", "Open AI Director"],
                )

        # -------------------------------------------------------------
        # 8. CHECK JOB STATUS / HOW IS PROCESSING
        # Examples: "How is Summer Trip processing?", "Check video status", "How is the video doing?"
        # -------------------------------------------------------------
        if any(k in q_lower for k in ["how is", "processing", "job status", "video status", "check status"]):
            # Extract memory name if provided
            user_jobs = job_state_manager.get_user_jobs(user_id)
            if not user_jobs:
                return HomeAssistantResponse(
                    message="You don't have any active video generation or media processing jobs right now.",
                    intent="CHECK_GENERATION_STATUS",
                    suggested_chips=["Show my memories", "Generate Video"],
                )

            # Match specific memory if named
            matched_job = None
            for j in user_jobs:
                mem = HomeAssistantService._get_memory_by_id(user_id, j.memory_id)
                if mem and (mem["title"].lower() in q_lower):
                    matched_job = j
                    break

            if not matched_job:
                matched_job = user_jobs[0]  # Most recent job

            mem_info = HomeAssistantService._get_memory_by_id(user_id, matched_job.memory_id)
            title = mem_info["title"] if mem_info else "Memory"

            status_str = matched_job.status.value if hasattr(matched_job.status, "value") else str(matched_job.status)
            progress = matched_job.overall_progress
            task = matched_job.current_task or matched_job.stage

            if status_str in ["completed"]:
                return HomeAssistantResponse(
                    message=f"The video for '{title}' is complete (100%) and ready to play!",
                    intent="CHECK_GENERATION_STATUS",
                    actions=[
                        AssistantAction(
                            type="OPEN_VIDEO",
                            memory_id=matched_job.memory_id,
                            job_id=matched_job.job_id,
                            media_id=matched_job.result_media_id,
                        )
                    ],
                    suggested_chips=["Play Video", "Open Memory"],
                )
            elif status_str in ["failed"]:
                return HomeAssistantResponse(
                    message=f"Video generation for '{title}' failed: {matched_job.error_message or 'Unknown error'}.",
                    intent="CHECK_GENERATION_STATUS",
                    actions=[AssistantAction(type="WATCH_VIDEO_JOB", job_id=matched_job.job_id)],
                    suggested_chips=["Retry Generation", "Open Memory"],
                )
            else:
                return HomeAssistantResponse(
                    message=f"'{title}' is generating: {progress}% completed ({task}).",
                    intent="CHECK_GENERATION_STATUS",
                    actions=[
                        AssistantAction(
                            type="WATCH_VIDEO_JOB",
                            memory_id=matched_job.memory_id,
                            job_id=matched_job.job_id,
                        )
                    ],
                    suggested_chips=["Check Status", "Cancel Video"],
                )

        # -------------------------------------------------------------
        # 9. CANCEL GENERATION
        # Examples: "Cancel the Summer Trip video", "Cancel job"
        # -------------------------------------------------------------
        if "cancel" in q_lower and ("video" in q_lower or "job" in q_lower or "generation" in q_lower):
            user_jobs = job_state_manager.get_user_jobs(user_id)
            active_jobs = [
                j for j in user_jobs
                if (j.status.value if hasattr(j.status, "value") else str(j.status)) in ["queued", "processing", "retrying"]
            ]
            if not active_jobs:
                return HomeAssistantResponse(
                    message="There are no active generation jobs to cancel.",
                    intent="CANCEL_GENERATION",
                )
            target_job = active_jobs[0]
            success = job_state_manager.cancel_job(target_job.job_id, user_id)
            if success:
                return HomeAssistantResponse(
                    message=f"Cancelled video generation job {target_job.job_id[:8]}.",
                    intent="CANCEL_GENERATION",
                    suggested_chips=["Show my memories"],
                )
            else:
                return HomeAssistantResponse(
                    message="Could not cancel the job. It may have already finished or failed.",
                    intent="CANCEL_GENERATION",
                )

        # -------------------------------------------------------------
        # 10. OPEN MEMORY VIDEO
        # Examples: "Open the Summer Trip video", "Play Swarnagiri video", "Watch Summer Trip"
        # -------------------------------------------------------------
        play_match = re.search(
            r"^(?:open|play|watch|view)(?:\s+(?:the|my))?\s+[\"']?(.+?)[\"']?\s+video$",
            q,
            re.IGNORECASE,
        )
        if play_match:
            entity_name = play_match.group(1).strip(" .\"'")
            resolved_mem = HomeAssistantService._resolve_memory_by_name(user_id, entity_name)
            if resolved_mem:
                mem_id = resolved_mem["id"]
                # Find latest video media in memory
                media_list = resolved_mem.get("media") or []
                video_items = [m for m in media_list if m.get("media_type") == "video"]
                if video_items:
                    latest_video = video_items[0]
                    return HomeAssistantResponse(
                        message=f"Opening video reel for '{resolved_mem['title']}'.",
                        intent="OPEN_MEMORY_VIDEO",
                        actions=[
                            AssistantAction(
                                type="OPEN_VIDEO",
                                memory_id=mem_id,
                                media_id=latest_video["id"],
                                data=latest_video,
                            )
                        ],
                        suggested_chips=["Open Memory", "Download Video", "Share Video"],
                    )
                else:
                    return HomeAssistantResponse(
                        message=f"'{resolved_mem['title']}' doesn't have a generated video yet. Would you like to generate one?",
                        intent="OPEN_MEMORY_VIDEO",
                        actions=[AssistantAction(type="OPEN_MEMORY", memory_id=mem_id)],
                        suggested_chips=["Generate Video", "Open AI Director"],
                    )

        # -------------------------------------------------------------
        # 11. OPEN MEMORY
        # Examples: "Open Summer Trip", "Take me to Swarnagiri Temple", "Show Summer Trip"
        # -------------------------------------------------------------
        open_mem_match = re.search(
            r"^(?:open|show|go to|take me to|navigate to)(?:\s+(?:the|my))?\s+(?:memory\s+)?[\"']?(.+?)[\"']?$",
            q,
            re.IGNORECASE,
        )
        if open_mem_match and not any(k in q_lower for k in ["vault", "room"]):
            entity_name = open_mem_match.group(1).strip(" .\"'")
            resolved_mem = HomeAssistantService._resolve_memory_by_name(user_id, entity_name)
            if resolved_mem:
                return HomeAssistantResponse(
                    message=f"Opening '{resolved_mem['title']}'.",
                    intent="OPEN_MEMORY",
                    actions=[
                        AssistantAction(
                            type="OPEN_MEMORY",
                            memory_id=resolved_mem["id"],
                            data={"title": resolved_mem["title"]},
                        )
                    ],
                    suggested_chips=["Add Media", "Check Readiness", "Generate Video"],
                )

        # -------------------------------------------------------------
        # 12. OPEN VAULT
        # Examples: "Open the Family vault", "Open Family vault", "Open room Friends"
        # -------------------------------------------------------------
        open_vault_match = re.search(
            r"^(?:open|show|go to|take me to)(?:\s+(?:the|my))?\s+(?:vault|room)\s+[\"']?(.+?)[\"']?$",
            q,
            re.IGNORECASE,
        )
        if open_vault_match:
            entity_name = open_vault_match.group(1).strip(" .\"'")
            resolved_vault = HomeAssistantService._resolve_vault_by_name(user_id, entity_name)
            if resolved_vault:
                return HomeAssistantResponse(
                    message=f"Opening room '{resolved_vault['name']}'.",
                    intent="OPEN_VAULT",
                    actions=[
                        AssistantAction(
                            type="OPEN_VAULT",
                            vault_id=resolved_vault["id"],
                            data={"name": resolved_vault["name"]},
                        )
                    ],
                    suggested_chips=["Add Memories", "Share Room"],
                )

        # -------------------------------------------------------------
        # 13. OPEN AI DIRECTOR
        # Examples: "Open AI Director for Summer Trip", "Help me make Summer Trip emotional"
        # -------------------------------------------------------------
        if "director" in q_lower or "emotional" in q_lower or "storyboard" in q_lower:
            mems = HomeAssistantService._fetch_user_memories(user_id)
            target_mem = None
            for m in mems:
                if m["title"].lower() in q_lower:
                    target_mem = m
                    break
            if not target_mem and mems:
                target_mem = mems[0]

            if target_mem:
                return HomeAssistantResponse(
                    message=f"Opening AI Memory Director for '{target_mem['title']}' to customize story, emotion, and music.",
                    intent="OPEN_AI_DIRECTOR",
                    actions=[
                        AssistantAction(
                            type="OPEN_AI_DIRECTOR",
                            memory_id=target_mem["id"],
                            data={"title": target_mem["title"]},
                        )
                    ],
                    suggested_chips=["Customize Music", "Change Pacing", "Generate Reel"],
                )

        # -------------------------------------------------------------
        # Fallback: Conversational Guidance
        # -------------------------------------------------------------
        return HomeAssistantResponse(
            message=(
                f"I'm here to help manage your memories and videos. You can say:\n"
                f"• 'Create a memory <name>'\n"
                f"• 'Open <memory name>'\n"
                f"• 'Is <memory> ready for a video?'\n"
                f"• 'Generate the video of <memory>'\n"
                f"• 'Show my rooms'"
            ),
            intent="UNKNOWN",
            suggested_chips=["Show my memories", "Show my vaults", "Create Memory"],
        )

    # -----------------------------------------------------------------
    # Helper Entity Resolution Methods
    # -----------------------------------------------------------------
    @staticmethod
    def _fetch_user_memories(user_id: str) -> List[Dict[str, Any]]:
        supabase = get_supabase_client()
        try:
            res = (
                supabase.table("memories")
                .select("id, title, memory_date, description, media!media_memory_id_fkey(id, media_type, url, thumbnail_url)")
                .eq("owner_id", user_id)
                .order("memory_date", desc=True)
                .execute()
            )
            return res.data or []
        except Exception as e:
            logger.error(f"Error fetching user memories: {e}")
            return []

    @staticmethod
    def _fetch_user_vaults(user_id: str) -> List[Dict[str, Any]]:
        supabase = get_supabase_client()
        try:
            res = (
                supabase.table("vaults")
                .select("id, name, description, cover_image_url")
                .eq("owner_id", user_id)
                .execute()
            )
            return res.data or []
        except Exception as e:
            logger.error(f"Error fetching user vaults: {e}")
            return []

    @staticmethod
    def _get_memory_by_id(user_id: str, memory_id: str) -> Optional[Dict[str, Any]]:
        supabase = get_supabase_client()
        try:
            res = supabase.table("memories").select("id, title, owner_id").eq("id", memory_id).execute()
            if res.data and res.data[0].get("owner_id") == user_id:
                return res.data[0]
        except Exception:
            pass
        return None

    @staticmethod
    def _resolve_memory_by_name(user_id: str, name: str) -> Optional[Dict[str, Any]]:
        """
        Fuzzy & case-insensitive matching for memory names.
        """
        clean_target = name.strip().lower()
        mems = HomeAssistantService._fetch_user_memories(user_id)
        if not mems:
            return None

        # 1. Exact match (case-insensitive)
        for m in mems:
            if m.get("title", "").strip().lower() == clean_target:
                return m

        # 2. Substring match
        for m in mems:
            t = m.get("title", "").strip().lower()
            if clean_target in t or t in clean_target:
                return m

        # 3. Fuzzy match via difflib
        titles = [m.get("title", "").strip() for m in mems]
        matches = difflib.get_close_matches(clean_target, [t.lower() for t in titles], n=1, cutoff=0.55)
        if matches:
            matched_lower = matches[0]
            for m in mems:
                if m.get("title", "").strip().lower() == matched_lower:
                    return m

        return None

    @staticmethod
    def _resolve_vault_by_name(user_id: str, name: str) -> Optional[Dict[str, Any]]:
        """
        Fuzzy & case-insensitive matching for vault names.
        """
        clean_target = name.strip().lower()
        vaults = HomeAssistantService._fetch_user_vaults(user_id)
        if not vaults:
            return None

        # 1. Exact match
        for v in vaults:
            if v.get("name", "").strip().lower() == clean_target:
                return v

        # 2. Substring match
        for v in vaults:
            vn = v.get("name", "").strip().lower()
            if clean_target in vn or vn in clean_target:
                return v

        # 3. Fuzzy match
        names = [v.get("name", "").strip() for v in vaults]
        matches = difflib.get_close_matches(clean_target, [n.lower() for n in names], n=1, cutoff=0.55)
        if matches:
            matched_lower = matches[0]
            for v in vaults:
                if v.get("name", "").strip().lower() == matched_lower:
                    return v

        return None


home_assistant_service = HomeAssistantService()

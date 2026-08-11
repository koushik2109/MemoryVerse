from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Any
from datetime import datetime

# --- AUTH SCHEMAS ---
class SignUpRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None

class SignInRequest(BaseModel):
    email: EmailStr
    password: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    full_name: Optional[str] = None

# --- PROFILE SCHEMAS ---
class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    username: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None

class UserProfile(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    username: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    vault_count: int = 0
    media_count: int = 0
    created_at: Optional[datetime] = None

# --- VAULT SCHEMAS ---
class VaultCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    cover_image_url: Optional[str] = None

class VaultUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    cover_image_url: Optional[str] = None
    is_archived: Optional[bool] = None

class VaultMemberSchema(BaseModel):
    id: str
    user_id: str
    full_name: Optional[str] = None
    email: Optional[str] = None
    avatar_url: Optional[str] = None
    role: str = "editor" # owner, editor, viewer
    joined_at: Optional[datetime] = None

class VaultMemberRoleUpdate(BaseModel):
    role: str # editor, viewer

class VaultResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    cover_image_url: Optional[str] = None
    is_archived: bool = False
    owner_id: str
    created_at: datetime
    updated_at: datetime
    member_count: int = 1
    media_count: int = 0
    invite_code: Optional[str] = None
    members: List[VaultMemberSchema] = []

class VaultJoinRequest(BaseModel):
    invite_code: str

# --- MEMORY SCHEMAS ---
class MemoryCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    vault_id: Optional[str] = None
    cover_media_id: Optional[str] = None
    memory_date: Optional[datetime] = None
    location_name: Optional[str] = None

class MemoryUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    vault_id: Optional[str] = None
    cover_media_id: Optional[str] = None
    memory_date: Optional[datetime] = None
    location_name: Optional[str] = None

class MemoryResponse(BaseModel):
    id: str
    owner_id: str
    vault_id: Optional[str] = None
    title: str
    description: Optional[str] = None
    cover_media_id: Optional[str] = None
    memory_date: datetime
    location_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    media_count: int = 0
    # To be populated if related media is fetched
    media: List['MediaResponse'] = []

# --- MEDIA SCHEMAS ---
class MediaCreate(BaseModel):
    vault_id: Optional[str] = None
    memory_id: Optional[str] = None
    filename: str
    storage_path: str
    url: str
    thumbnail_url: Optional[str] = None
    media_type: str = "image" # image, video
    file_size: int = 0
    mime_type: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[int] = None
    metadata: Optional[dict] = {}

class MediaReorderRequest(BaseModel):
    media_ids: List[str]

class MediaResponse(BaseModel):
    id: str
    vault_id: Optional[str] = None
    memory_id: Optional[str] = None
    owner_id: str
    filename: str
    storage_path: str
    url: str
    thumbnail_url: Optional[str] = None
    media_type: str
    file_size: int
    mime_type: Optional[str] = None
    created_at: datetime

# --- VIDEO JOBS SCHEMAS ---
class VideoJobResponse(BaseModel):
    id: str
    memory_id: str
    user_id: str
    status: str # queued, processing, completed, failed
    result_media_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

# --- INVITATION SCHEMAS ---
class InviteCreateResponse(BaseModel):
    invite_code: str
    invite_link: str
    vault_id: str
    vault_name: str
    expires_at: Optional[datetime] = None

class InviteInfoResponse(BaseModel):
    invite_code: str
    vault_id: str
    vault_name: str
    vault_description: Optional[str] = None
    owner_name: Optional[str] = None
    member_count: int = 0

# --- NOTIFICATION SCHEMAS ---
class NotificationResponse(BaseModel):
    id: str
    title: str
    message: str
    type: str
    data: Optional[dict] = {}
    is_read: bool = False
    created_at: datetime

# --- SEARCH SCHEMAS ---
class GlobalSearchResponse(BaseModel):
    vaults: List[VaultResponse] = []
    media: List[MediaResponse] = []
    collaborators: List[UserProfile] = []

# --- SETTINGS SCHEMAS ---
class SettingsUpdate(BaseModel):
    dark_mode: Optional[bool] = None
    notifications_enabled: Optional[bool] = None

# --- TIMELINE SCHEMAS ---
class TimelineMediaItem(BaseModel):
    id: str
    url: str
    thumbnail_url: Optional[str] = None
    media_type: str  # image | video
    filename: str
    created_at: datetime

class TimelineDayGroup(BaseModel):
    date_label: str       # e.g. "Aug 10"
    date: datetime
    memories: List[MemoryResponse] = []

class TimelineMonthGroup(BaseModel):
    month: str            # e.g. "August"
    days: List[TimelineDayGroup] = []

class TimelineYearGroup(BaseModel):
    year: str
    months: List[TimelineMonthGroup] = []

class TimelineResponse(BaseModel):
    groups: List[TimelineYearGroup] = []
    total_items: int = 0

# --- AI / CHAT SCHEMAS ---
class AIChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: Optional[str] = None  # pass to continue an existing chat

class AIMessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str  # user | assistant
    content: str
    related_memory_ids: List[str] = []
    created_at: datetime

class AIChatResponse(BaseModel):
    conversation_id: str
    user_message: AIMessageResponse
    assistant_message: AIMessageResponse

class AIConversationResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

MemoryResponse.model_rebuild()


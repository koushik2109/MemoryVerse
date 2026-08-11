from fastapi import APIRouter, Depends
from app.schemas.domain import InviteCreateResponse, InviteInfoResponse, VaultResponse
from app.services.invite_service import InviteService
from app.core.security import get_current_user, CurrentUser

router = APIRouter(prefix="/invites", tags=["Collaboration & Invites"])

@router.post("/vaults/{vault_id}/invite-link", response_model=InviteCreateResponse)
async def create_invite_link(vault_id: str, current_user: CurrentUser = Depends(get_current_user)):
    return InviteService.generate_invite(vault_id, current_user.id)

@router.get("/{code}", response_model=InviteInfoResponse)
async def get_invite_info(code: str):
    return InviteService.get_invite_info(code)

@router.post("/{code}/accept", response_model=VaultResponse)
async def accept_invite(code: str, current_user: CurrentUser = Depends(get_current_user)):
    return InviteService.join_vault(code, current_user.id)

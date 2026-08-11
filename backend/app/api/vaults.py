from fastapi import APIRouter, Depends, status
from app.schemas.domain import VaultCreate, VaultUpdate, VaultResponse, VaultMemberSchema, VaultJoinRequest
from app.services.vault_service import VaultService
from app.core.security import get_current_user, CurrentUser
from typing import List

router = APIRouter(prefix="/vaults", tags=["Vaults"])


@router.get("", response_model=List[VaultResponse], summary="List vaults the user belongs to")
async def list_vaults(current_user: CurrentUser = Depends(get_current_user)):
    return VaultService.get_user_vaults(current_user.id)


@router.post("", response_model=VaultResponse, status_code=status.HTTP_201_CREATED, summary="Create a new vault")
async def create_vault(payload: VaultCreate, current_user: CurrentUser = Depends(get_current_user)):
    return VaultService.create_vault(current_user.id, payload)


@router.post("/join", response_model=VaultResponse, summary="Join a vault via invite code")
async def join_vault(payload: VaultJoinRequest, current_user: CurrentUser = Depends(get_current_user)):
    return VaultService.join_by_code(payload.invite_code, current_user.id)


@router.get("/{vault_id}", response_model=VaultResponse, summary="Get vault details")
async def get_vault(vault_id: str, current_user: CurrentUser = Depends(get_current_user)):
    return VaultService.get_vault_by_id(vault_id, current_user.id)


@router.put("/{vault_id}", response_model=VaultResponse, summary="Update vault")
async def update_vault(vault_id: str, payload: VaultUpdate, current_user: CurrentUser = Depends(get_current_user)):
    return VaultService.update_vault(vault_id, current_user.id, payload)


@router.patch("/{vault_id}", response_model=VaultResponse, summary="Partially update vault")
async def patch_vault(vault_id: str, payload: VaultUpdate, current_user: CurrentUser = Depends(get_current_user)):
    return VaultService.update_vault(vault_id, current_user.id, payload)


@router.delete("/{vault_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete vault")
async def delete_vault(vault_id: str, current_user: CurrentUser = Depends(get_current_user)):
    VaultService.delete_vault(vault_id, current_user.id)


@router.post("/{vault_id}/leave", summary="Leave a vault")
async def leave_vault(vault_id: str, current_user: CurrentUser = Depends(get_current_user)):
    VaultService.leave_vault(vault_id, current_user.id)
    return {"message": "Left vault"}


# ── Member endpoints ──────────────────────────────────────────────────────────

@router.get("/{vault_id}/members", response_model=List[VaultMemberSchema], summary="List vault members")
async def get_members(vault_id: str, current_user: CurrentUser = Depends(get_current_user)):
    return VaultService.get_members(vault_id, current_user.id)


@router.delete(
    "/{vault_id}/members/{target_user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a member from vault",
)
async def remove_member(
    vault_id: str,
    target_user_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    VaultService.remove_member(vault_id, target_user_id, current_user.id)


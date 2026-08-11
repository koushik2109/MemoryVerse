import secrets
from app.core.db import get_supabase_client
from app.schemas.domain import InviteCreateResponse, InviteInfoResponse, VaultResponse
from app.services.vault_service import VaultService
from fastapi import HTTPException, status
from datetime import datetime, timezone
from typing import cast, Any
from postgrest.types import CountMethod

class InviteService:
    @staticmethod
    def generate_invite(vault_id: str, user_id: str) -> InviteCreateResponse:
        supabase = get_supabase_client()
        # Verify user is owner or editor in vault
        v_res = supabase.table("vaults").select("name, owner_id").eq("id", vault_id).execute()
        if not v_res.data:
            raise HTTPException(status_code=404, detail="Vault not found")
        
        # Check existing invite
        existing = supabase.table("invitations").select("*").eq("vault_id", vault_id).execute()
        if existing.data:
            code = cast(list[dict[str, Any]], existing.data)[0]["invite_code"]
        else:
            code = secrets.token_urlsafe(8)
            supabase.table("invitations").insert({
                "vault_id": vault_id,
                "invite_code": code,
                "created_by": user_id,
                "role": "editor",
                "created_at": datetime.now(timezone.utc).isoformat()
            }).execute()

        link = f"memoryverse://join/{code}"
        return InviteCreateResponse(
            invite_code=code,
            invite_link=link,
            vault_id=vault_id,
            vault_name=cast(list[dict[str, Any]], v_res.data)[0]["name"]
        )

    @staticmethod
    def get_invite_info(code: str) -> InviteInfoResponse:
        supabase = get_supabase_client()
        inv_res = supabase.table("invitations").select("*").eq("invite_code", code).execute()
        if not inv_res.data:
            raise HTTPException(status_code=404, detail="Invalid or expired invite link")
        
        inv = cast(list[dict[str, Any]], inv_res.data)[0]
        v_res = supabase.table("vaults").select("*").eq("id", inv["vault_id"]).execute()
        if not v_res.data:
            raise HTTPException(status_code=404, detail="Vault no longer exists")
        
        v = cast(list[dict[str, Any]], v_res.data)[0]
        p_res = supabase.table("profiles").select("full_name").eq("id", v["owner_id"]).execute()
        owner_name = cast(list[dict[str, Any]], p_res.data)[0].get("full_name") if p_res.data else "Owner"
        
        mem_cnt = supabase.table("vault_members").select("id", count="exact").eq("vault_id", v["id"]).execute().count or 1  # type: ignore

        return InviteInfoResponse(
            invite_code=code,
            vault_id=v["id"],
            vault_name=v["name"],
            vault_description=v.get("description"),
            owner_name=owner_name,
            member_count=mem_cnt
        )

    @staticmethod
    def join_vault(code: str, user_id: str) -> VaultResponse:
        supabase = get_supabase_client()
        inv_res = supabase.table("invitations").select("*").eq("invite_code", code).execute()
        if not inv_res.data:
            raise HTTPException(status_code=404, detail="Invalid invite code")
        
        inv = cast(list[dict[str, Any]], inv_res.data)[0]
        vault_id = inv["vault_id"]
        role = inv.get("role", "editor")

        # Check if already a member
        existing = supabase.table("vault_members").select("*").eq("vault_id", vault_id).eq("user_id", user_id).execute()
        if not existing.data:
            now = datetime.now(timezone.utc).isoformat()
            supabase.table("vault_members").insert({
                "vault_id": vault_id,
                "user_id": user_id,
                "role": role,
                "joined_at": now
            }).execute()

            # Create notification for owner
            v_res = supabase.table("vaults").select("owner_id, name").eq("id", vault_id).execute()
            if v_res.data:
                owner_id = cast(list[dict[str, Any]], v_res.data)[0]["owner_id"]
                p_res = supabase.table("profiles").select("full_name").eq("id", user_id).execute()
                joiner_name = cast(list[dict[str, Any]], p_res.data)[0].get("full_name") if p_res.data else "A new user"
                supabase.table("notifications").insert({
                    "user_id": owner_id,
                    "title": "New Collaborator",
                    "message": f"{joiner_name} joined your vault '{cast(list[dict[str, Any]], v_res.data)[0]['name']}'",
                    "type": "member_joined",
                    "data": {"vault_id": vault_id, "user_id": user_id},
                    "created_at": now
                }).execute()

        return VaultService.get_vault_by_id(vault_id, user_id)

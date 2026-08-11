from app.core.db import get_supabase_client
from app.schemas.domain import VaultCreate, VaultUpdate, VaultResponse, VaultMemberSchema
from fastapi import HTTPException, status
from typing import cast, Any
from datetime import timezone
import uuid
import random
import string
from datetime import datetime
from postgrest.types import CountMethod

class VaultService:
    @staticmethod
    def create_vault(user_id: str, payload: VaultCreate) -> VaultResponse:
        supabase = get_supabase_client()
        now = datetime.now(timezone.utc).isoformat()
        vault_data = {
            "name": payload.name,
            "description": payload.description,
            "cover_image_url": payload.cover_image_url,
            "owner_id": user_id,
            "is_archived": False,
            "created_at": now,
            "updated_at": now,
        }
        res = supabase.table("vaults").insert(vault_data).execute()
        if not res.data:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create vault")
        
        vault = cast(list[dict[str, Any]], res.data)[0]
        # Add owner to vault_members
        supabase.table("vault_members").insert({
            "vault_id": vault["id"],
            "user_id": user_id,
            "role": "owner",
            "joined_at": now,
        }).execute()
        
        # Generate permanent invite code for Room
        # Format: MV-XXXXXX
        code = "MV-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
        
        supabase.table("vault_invitations").insert({
            "vault_id": vault["id"],
            "created_by": user_id,
            "invite_code": code,
            "invite_link": f"memoryverse://room/{code}",
            "created_at": now
        }).execute()

        return VaultResponse(
            id=vault["id"],
            name=vault["name"],
            description=vault.get("description"),
            cover_image_url=vault.get("cover_image_url"),
            is_archived=vault["is_archived"],
            owner_id=vault["owner_id"],
            created_at=vault["created_at"],
            updated_at=vault["updated_at"],
            member_count=1,
            media_count=0,
            invite_code=code,
            members=[]
        )

    @staticmethod
    def get_user_vaults(user_id: str) -> list[VaultResponse]:
        supabase = get_supabase_client()
        # Find vaults owned or joined by user
        member_res = supabase.table("vault_members").select("vault_id").eq("user_id", user_id).execute()
        vault_ids = [m["vault_id"] for m in (cast(list[dict[str, Any]], member_res.data or []))]
        
        if not vault_ids:
            return []

        vaults_res = supabase.table("vaults").select("*").in_("id", vault_ids).order("updated_at", desc=True).execute()
        vaults = cast(list[dict[str, Any]], vaults_res.data or [])

        result = []
        for v in vaults:
            # Count members & media
            mem_cnt = supabase.table("vault_members").select("id", count="exact").eq("vault_id", v["id"]).execute().count or 1  # type: ignore
            med_cnt = supabase.table("media").select("id", count="exact").eq("vault_id", v["id"]).execute().count or 0  # type: ignore
            
            # Get invite code
            inv_res = supabase.table("vault_invitations").select("invite_code").eq("vault_id", v["id"]).execute()
            inv_code = cast(list[dict[str, Any]], inv_res.data)[0]["invite_code"] if inv_res.data else None
            
            result.append(VaultResponse(
                id=v["id"],
                name=v["name"],
                description=v.get("description"),
                cover_image_url=v.get("cover_image_url"),
                is_archived=v.get("is_archived", False),
                owner_id=v["owner_id"],
                created_at=v["created_at"],
                updated_at=v["updated_at"],
                member_count=mem_cnt,
                media_count=med_cnt,
                invite_code=inv_code,
                members=[]
            ))
        return result

    @staticmethod
    def get_vault_by_id(vault_id: str, user_id: str) -> VaultResponse:
        supabase = get_supabase_client()
        v_res = supabase.table("vaults").select("*").eq("id", vault_id).execute()
        if not v_res.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vault not found")
        
        v = cast(list[dict[str, Any]], v_res.data)[0]

        # Get members with profile info
        members_res = supabase.table("vault_members").select("id, user_id, role, joined_at").eq("vault_id", vault_id).execute()
        members_data = cast(list[dict[str, Any]], members_res.data or [])
        
        members_list = []
        for m in members_data:
            prof = supabase.table("profiles").select("*").eq("id", m["user_id"]).execute()
            p = cast(list[dict[str, Any]], prof.data)[0] if prof.data else {}
            members_list.append(VaultMemberSchema(
                id=m["id"],
                user_id=m["user_id"],
                full_name=p.get("full_name"),
                email=p.get("email"),
                avatar_url=p.get("avatar_url"),
                role=m["role"],
                joined_at=m.get("joined_at")
            ))

        med_cnt = supabase.table("media").select("id", count="exact").eq("vault_id", vault_id).execute().count or 0  # type: ignore

        # Get invite code
        inv_res = supabase.table("vault_invitations").select("invite_code").eq("vault_id", vault_id).execute()
        inv_code = cast(list[dict[str, Any]], inv_res.data)[0]["invite_code"] if inv_res.data else None

        return VaultResponse(
            id=v["id"],
            name=v["name"],
            description=v.get("description"),
            cover_image_url=v.get("cover_image_url"),
            is_archived=v.get("is_archived", False),
            owner_id=v["owner_id"],
            created_at=v["created_at"],
            updated_at=v["updated_at"],
            member_count=len(members_list),
            media_count=med_cnt,
            invite_code=inv_code,
            members=members_list
        )

    @staticmethod
    def update_vault(vault_id: str, user_id: str, payload: VaultUpdate) -> VaultResponse:
        supabase = get_supabase_client()
        # Verify ownership
        v_res = supabase.table("vaults").select("*").eq("id", vault_id).execute()
        if not v_res.data:
            raise HTTPException(status_code=404, detail="Vault not found")
        if cast(list[dict[str, Any]], v_res.data)[0]["owner_id"] != user_id:
            raise HTTPException(status_code=403, detail="Only owner can modify vault settings")

        update_dict = {}
        if payload.name is not None: update_dict["name"] = payload.name
        if payload.description is not None: update_dict["description"] = payload.description
        if payload.cover_image_url is not None: update_dict["cover_image_url"] = payload.cover_image_url
        if payload.is_archived is not None: update_dict["is_archived"] = payload.is_archived
        update_dict["updated_at"] = datetime.now(timezone.utc).isoformat()

        supabase.table("vaults").update(update_dict).eq("id", vault_id).execute()
        return VaultService.get_vault_by_id(vault_id, user_id)

    @staticmethod
    def delete_vault(vault_id: str, user_id: str) -> bool:
        supabase = get_supabase_client()
        v_res = supabase.table("vaults").select("owner_id").eq("id", vault_id).execute()
        if not v_res.data:
            raise HTTPException(status_code=404, detail="Vault not found")
        if cast(list[dict[str, Any]], v_res.data)[0]["owner_id"] != user_id:
            raise HTTPException(status_code=403, detail="Only owner can delete vault")
        
        supabase.table("vaults").delete().eq("id", vault_id).execute()
        return True

    @staticmethod
    def leave_vault(vault_id: str, user_id: str) -> bool:
        supabase = get_supabase_client()
        v_res = supabase.table("vaults").select("owner_id").eq("id", vault_id).execute()
        if v_res.data and cast(list[dict[str, Any]], v_res.data)[0]["owner_id"] == user_id:
            raise HTTPException(status_code=400, detail="Owner cannot leave vault. Transfer ownership or delete vault instead.")
        
        supabase.table("vault_members").delete().eq("vault_id", vault_id).eq("user_id", user_id).execute()
        return True

    # ── Member Management ────────────────────────────────────────────────────

    @staticmethod
    def get_members(vault_id: str, user_id: str) -> list[VaultMemberSchema]:
        """Get all members of a vault (requires membership)."""
        supabase = get_supabase_client()
        # Verify requester is a member
        check = supabase.table("vault_members").select("id") \
            .eq("vault_id", vault_id).eq("user_id", user_id).execute()
        if not check.data:
            raise HTTPException(status_code=403, detail="Not a member of this vault")

        members_res = supabase.table("vault_members").select("id, user_id, role, joined_at") \
            .eq("vault_id", vault_id).execute()
        members_data = cast(list[dict[str, Any]], members_res.data or [])

        result = []
        for m in members_data:
            prof = supabase.table("profiles").select("*").eq("id", m["user_id"]).execute()
            p = cast(list[dict[str, Any]], prof.data)[0] if prof.data else {}
            result.append(VaultMemberSchema(
                id=m["id"],
                user_id=m["user_id"],
                full_name=p.get("full_name"),
                email=p.get("email"),
                avatar_url=p.get("avatar_url"),
                role=m["role"],
                joined_at=m.get("joined_at")
            ))
        return result

    @staticmethod
    def remove_member(vault_id: str, target_user_id: str, requesting_user_id: str) -> bool:
        """Remove a member from a vault (owner only, or self-removal)."""
        supabase = get_supabase_client()
        vault_res = supabase.table("vaults").select("owner_id").eq("id", vault_id).execute()
        if not vault_res.data:
            raise HTTPException(status_code=404, detail="Vault not found")

        is_owner = cast(list[dict[str, Any]], vault_res.data)[0]["owner_id"] == requesting_user_id
        is_self = target_user_id == requesting_user_id

        if not is_owner and not is_self:
            raise HTTPException(status_code=403, detail="Only vault owner can remove other members")

        if target_user_id == cast(list[dict[str, Any]], vault_res.data)[0]["owner_id"]:
            raise HTTPException(status_code=400, detail="Cannot remove the vault owner")

        supabase.table("vault_members").delete() \
            .eq("vault_id", vault_id).eq("user_id", target_user_id).execute()
        return True

    @staticmethod
    def join_by_code(invite_code: str, user_id: str) -> VaultResponse:
        """Join a vault using an invite code."""
        supabase = get_supabase_client()
        inv_res = supabase.table("vault_invitations").select("*") \
            .eq("invite_code", invite_code).execute()
        if not inv_res.data:
            raise HTTPException(status_code=404, detail="Invalid or expired invite code")

        inv = cast(list[dict[str, Any]], inv_res.data)[0]

        # Check expiry
        if inv.get("expires_at"):
            exp = datetime.fromisoformat(inv["expires_at"].replace("Z", "+00:00"))
            if exp < datetime.now(timezone.utc):
                raise HTTPException(status_code=400, detail="Invite code has expired")

        vault_id = inv["vault_id"]

        # Check if already a member
        existing = supabase.table("vault_members").select("id") \
            .eq("vault_id", vault_id).eq("user_id", user_id).execute()
        if existing.data:
            return VaultService.get_vault_by_id(vault_id, user_id)

        # Add as editor
        supabase.table("vault_members").insert({
            "vault_id": vault_id,
            "user_id": user_id,
            "role": "editor",
            "joined_at": datetime.now(timezone.utc).isoformat(),
        }).execute()

        # Increment use_count
        supabase.table("vault_invitations").update({"use_count": int(inv.get("use_count") or 0) + 1}) \
            .eq("id", inv["id"]).execute()

        return VaultService.get_vault_by_id(vault_id, user_id)


from fastapi import APIRouter

router = APIRouter(prefix="/api/v1")

@router.get("/health")
async def health_check():
    return {"status": "ok", "message": "MemoryVerse API is running"}

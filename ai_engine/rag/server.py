"""
MemoryVerse - Standalone RAG Microservice
Runs Video RAG and Audio RAG pipelines independently on port 8001
without requiring main backend dependencies.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ai_engine.rag.video.router import router as video_router
from ai_engine.rag.audio.router import router as audio_router

app = FastAPI(
    title="MemoryVerse Standalone RAG Engine",
    version="1.0.0",
    description="Multimodal Audio and Video Retrieval-Augmented Generation Engine",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video_router)
app.include_router(audio_router)

import asyncio
from pydantic import BaseModel
from typing import Any, Dict, Optional

class MultiAgentFlowRequest(BaseModel):
    query: str
    user_id: str = "default_user"
    ablation_mode: str = "model_4_full"
    target_duration: float = 30.0
    media_type: str = "all"

@app.post("/api/v1/agents/flow", tags=["Multi-Agent Flow"])
async def execute_agent_flow(req: MultiAgentFlowRequest):
    """
    Executes the multi-agent cognitive architecture
    (Planner, Scorer, Storyteller, Auditor, Video Director).
    """
    from ai_engine.langgraph.graph import run_memory_flow
    result = await asyncio.to_thread(
        run_memory_flow,
        query=req.query,
        user_id=req.user_id,
        ablation_mode=req.ablation_mode,
        target_duration=req.target_duration,
        media_type=req.media_type,
    )
    return result

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": "MemoryVerse Standalone RAG & Multi-Agent Engine",
        "pipelines": [
            "Video RAG (SigLIP + BGE-M3 + Qwen-VL)",
            "Audio RAG (Librosa + Whisper + BGE-M3)",
            "Multi-Agent LangGraph (Planner, Scorer, Storyteller, Auditor, Video Director)",
        ],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "ai_engine.rag.server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        reload_dirs=["ai_engine"],
        reload_excludes=[".venv", "*/.venv/*", "*/site-packages/*", "__pycache__", "cache", "tmp", "storage", "*.pyc"]
    )

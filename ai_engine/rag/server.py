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

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": "MemoryVerse Standalone RAG Engine",
        "pipelines": ["Video RAG (SigLIP + BGE-M3 + Qwen-VL)", "Audio RAG (Librosa + Whisper + BGE-M3)"],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_engine.rag.server:app", host="0.0.0.0", port=8001, reload=True)

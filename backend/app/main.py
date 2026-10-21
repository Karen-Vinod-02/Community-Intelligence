from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from app.api.analysis import router as analysis_router
from app.services.analysis_cache import analysis_cache
from app.services.generation import warm_ollama

WARMUP_EXECUTOR = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("OLLAMA_WARMUP", "true").lower() == "true":
        WARMUP_EXECUTOR.submit(warm_ollama)
    yield

app = FastAPI(title="Community Intelligence Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    analysis_router,
    prefix="/api"
)


@app.get("/api/health")
def health():
    return {"status": "ok", "supabase": analysis_cache.health()}

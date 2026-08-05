from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.analysis import router as analysis_router

app = FastAPI(title="Community Intelligence Engine")

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
    return {"status": "ok"}
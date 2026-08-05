from fastapi import FastAPI

app = FastAPI(title="Community Intelligence Engine")

@app.get("/")
def root():
    return {
        "message": "Community Intelligence Engine API",
        "docs": "/docs",
        "health": "/api/health"
    }

@app.get("/api/health")
def health():
    return {"status": "ok"}
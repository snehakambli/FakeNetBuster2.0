"""
FakeNetBuster Backend API
FastAPI application entry point.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from backend.routes.upload_routes import router as upload_router
from backend.routes.analysis_routes import router as analysis_router
from backend.routes.report_routes import router as report_router
from backend.routes.preview_routes import router as preview_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("reports/generated", exist_ok=True)
    os.makedirs("reports/visualizations", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    print("FakeNetBuster API started.")
    yield
    # Shutdown
    print("FakeNetBuster API shutting down.")


app = FastAPI(
    title="FakeNetBuster 2.0 API",
    description="AI-Powered Multi-Modal Fake Detection Platform",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for visualizations
if os.path.exists("reports/visualizations"):
    app.mount("/visualizations", StaticFiles(directory="reports/visualizations"),
              name="visualizations")

# Register routers
app.include_router(upload_router)
app.include_router(analysis_router)
app.include_router(report_router)
app.include_router(preview_router)


@app.get("/")
async def root():
    return {
        "name": "FakeNetBuster 2.0",
        "version": "2.0.0",
        "status": "running",
        "endpoints": ["/upload", "/analyze/file", "/analyze/news", "/report/history"]
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    import yaml
    try:
        with open("configs/system_configs.yaml") as f:
            cfg = yaml.safe_load(f)["server"]
    except Exception:
        cfg = {"host": "0.0.0.0", "port": 8000}

    uvicorn.run("backend.main:app", host=cfg["host"], port=cfg["port"],
                reload=cfg.get("reload", False))

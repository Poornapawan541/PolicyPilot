"""
PolicyPilot FastAPI application entry point.

Provides lightweight, local-first REST API endpoints for PolicyPilot:
- GET  /health : Instant health check without loading heavy ML models
- POST /ask    : Runs the PolicyPilot LangGraph workflow on a query
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(
    title="PolicyPilot API",
    description="Deterministic, verifiable enterprise policy assistant with citation verification and conflict resolution.",
    version="1.0.0",
)

# Optional CORS middleware for local frontend/tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "PolicyPilot",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
    }

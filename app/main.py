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

ALLOWED_ORIGINS = [
    "https://policy-pilot-eta.vercel.app",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

# Tightly constrained regex matching official project Vercel preview domains only
# (e.g., https://policy-pilot-fullstack-*.vercel.app or https://policy-pilot-*.vercel.app)
VERCEL_PREVIEW_ORIGIN_REGEX = r"^https://policy-pilot(-fullstack)?(-[a-zA-Z0-9_-]+)?\.vercel\.app$"

# CORS middleware for Vercel production frontend, verified project previews, and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=VERCEL_PREVIEW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Register routes for both root and /api prefixes
app.include_router(router)
app.include_router(router, prefix="/api")


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "PolicyPilot",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
    }

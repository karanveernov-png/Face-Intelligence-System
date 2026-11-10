from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.api.ws import router as ws_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="AI Face Intelligence System",
    description=(
        "Local-first face detection, consent-based recognition, expression "
        "classification, and approximate age estimation. See /docs for the "
        "interactive API reference, and docs/PRIVACY.md in the repo for the "
        "consent and data-retention model."
    ),
    version="0.1.0",
)

# Local-first app: frontend is expected to run on localhost during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(ws_router)


@app.get("/")
def root():
    return {"message": "AI Face Intelligence System API. See /docs for the interactive API reference."}

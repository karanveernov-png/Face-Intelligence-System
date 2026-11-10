from __future__ import annotations

from pydantic import BaseModel, Field


class EnrollResponse(BaseModel):
    face_id: str
    name: str
    retained_image: bool


class EnrolledEntry(BaseModel):
    face_id: str
    name: str
    model: str
    enrolled_at: float


class DeleteResponse(BaseModel):
    face_id: str
    deleted: bool


class ModelStatus(BaseModel):
    name: str
    purpose: str
    loaded: bool
    backend: str
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str


class AnalyzeOptions(BaseModel):
    run_recognition: bool = Field(default=True)
    run_expression: bool = Field(default=True)
    run_age: bool = Field(default=False)

"""
Upload validation and safe temporary-file handling.

Keeps four concerns out of the route handlers:
  1. Reject oversized / wrong-extension uploads before they touch disk or a model.
  2. Never trust a client-supplied filename for a filesystem path (classic path-
     traversal risk) — we always generate our own name.
  3. Guarantee temp files are cleaned up, even on failure.
  4. Reject a decoded image/video frame whose pixel dimensions are absurd, even
     though its compressed file size was within the upload cap -- see
     `reject_oversized_decoded_image` for why compressed size alone doesn't
     bound memory use.

BUG FIX (see README/CHANGELOG): these were previously consumed in route handlers
as `async for path in validated_image_upload(file): ...` with an early `return`
inside the loop body. Breaking out of an `async for` early does not reliably
call the generator's `finally` block in all cases/interpreters — Python only
guarantees that on explicit `.aclose()`, which a plain loop doesn't call. It
happened to work in testing under CPython's reference-counting GC, but that's
an implementation detail, not a guarantee. These are now consumed via FastAPI's
`Depends()`, which is the documented, guaranteed mechanism for yield-dependency
cleanup — FastAPI itself calls `.aclose()` via an exit stack after the response
is sent, regardless of which exception path was taken.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
from fastapi import HTTPException, UploadFile, File

from app.core.config import settings, TMP_UPLOAD_DIR


def _safe_suffix(filename: str, allowed: tuple[str, ...]) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix or '(none)'}'. Allowed: {', '.join(allowed)}",
        )
    return suffix


async def _write_validated(upload: UploadFile, allowed: tuple[str, ...]) -> Path:
    suffix = _safe_suffix(upload.filename, allowed)

    dest = TMP_UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    size = 0
    with open(dest, "wb") as out:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds max upload size of {settings.max_upload_size_mb}MB",
                )
            out.write(chunk)

    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    return dest


async def validated_image_upload(file: UploadFile = File(...)) -> AsyncGenerator[Path, None]:
    """FastAPI yield-dependency: inject with `Depends(validated_image_upload)`.
    FastAPI guarantees the code after `yield` runs once the request is done,
    success or failure — that's what makes cleanup reliable here."""
    path = await _write_validated(file, settings.allowed_image_ext)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


async def validated_video_upload(file: UploadFile = File(...)) -> AsyncGenerator[Path, None]:
    """Same guarantee as validated_image_upload, for video uploads."""
    path = await _write_validated(file, settings.allowed_video_ext)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def reject_oversized_dimensions(w: int, h: int) -> None:
    """Dimensions-only check that never itself allocates a buffer the size
    being checked -- safe to call on a width/height read from a container's
    metadata (e.g. cv2.VideoCapture's CAP_PROP_FRAME_WIDTH/HEIGHT) before any
    frame is decoded. See reject_oversized_decoded_image for the full
    rationale; that function delegates here after extracting img.shape."""
    if h <= 0 or w <= 0:
        return
    if max(h, w) > settings.max_decoded_image_dimension_px or h * w > settings.max_decoded_image_pixels:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Decoded image is {w}x{h} ({h * w:,} px), which exceeds the "
                f"{settings.max_decoded_image_pixels:,}px / "
                f"{settings.max_decoded_image_dimension_px}px-per-side limit. "
                "This guards against decompression-bomb-style files that are "
                "small on disk but huge once decoded; resize and retry."
            ),
        )


def reject_oversized_decoded_image(img: np.ndarray) -> None:
    """Call this right after cv2.imread/VideoCapture.read succeeds, before the
    array reaches any model. A compressed file's size on disk says nothing
    about its decoded size: a highly-compressible image (large areas of flat
    color, which real photos rarely have but a crafted file can) can be a few
    MB on disk and gigabytes once decoded -- `max_upload_size_mb` alone does
    not guard against this. Raises 413 (too large), matching the status this
    API already uses for the compressed-size check, rather than silently
    downscaling, since a caller who intended a precise analysis should decide
    whether to resize and retry rather than have us guess for them.

    img is already a decoded array by the time this runs, so this itself
    allocates nothing beyond reading .shape -- for a check that must run
    *before* decoding (e.g. from container metadata), use
    reject_oversized_dimensions instead."""
    if img is None:
        return
    reject_oversized_dimensions(w=img.shape[1], h=img.shape[0])

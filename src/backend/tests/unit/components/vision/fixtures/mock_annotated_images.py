"""
Mock AnnotatedImage fixtures for VisionOutputComponent tests.
Defined in ACCEPTANCE.MD § "Node: Output".
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Minimal valid base64 data URIs — PNG 1×1 pixel, different colors per image.
# Content is irrelevant for output-structure tests; only the URI format matters.
# ---------------------------------------------------------------------------

_B64_RED   = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADklEQVQI12P4z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
_B64_GREEN = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADklEQVQI12Nk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
_B64_BLUE  = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADklEQVQI12NgYGBgAAAABAABJzQnCgAAAABJRU5ErkJggg=="

# ---------------------------------------------------------------------------
# Canonical single AnnotatedImage fixtures
# ---------------------------------------------------------------------------

ANNOTATED_SUCCESS: dict = {
    "index": 0,
    "filename": "car_0.jpg",
    "annotated_base64": _B64_RED,
    "findings": {
        "index": 0,
        "filename": "car_0.jpg",
        "brand": "Toyota",
        "model": "Camry",
        "color": "White",
        "confidence": 0.92,
        "bbox": {"x": 50, "y": 40, "width": 100, "height": 80},
        "status": "success",
        "error": None,
    },
}

# brand=None → "Không nhận dạng được" in output, status still "success"
ANNOTATED_UNKNOWN: dict = {
    "index": 1,
    "filename": "car_1.jpg",
    "annotated_base64": _B64_GREEN,
    "findings": {
        "index": 1,
        "filename": "car_1.jpg",
        "brand": None,
        "model": None,
        "color": None,
        "confidence": 0.3,
        "bbox": None,
        "status": "success",
        "error": None,
    },
}

# status="failed" → shows error message in block, not counted as "nhận dạng được"
ANNOTATED_FAILED: dict = {
    "index": 2,
    "filename": "car_2.jpg",
    "annotated_base64": _B64_BLUE,
    "findings": {
        "index": 2,
        "filename": "car_2.jpg",
        "brand": None,
        "model": None,
        "color": None,
        "confidence": 0.0,
        "bbox": None,
        "status": "failed",
        "error": "API rate limit exceeded",
    },
}

# ---------------------------------------------------------------------------
# Pre-built batch fixtures
# ---------------------------------------------------------------------------

# All three combined: 1 recognized, 1 unknown, 1 failed → "1/3 ảnh nhận dạng được"
BATCH_MIXED = [ANNOTATED_SUCCESS, ANNOTATED_UNKNOWN, ANNOTATED_FAILED]

# Two recognized images → "2/2 ảnh nhận dạng được"
BATCH_ALL_SUCCESS = [
    ANNOTATED_SUCCESS,
    {
        "index": 1,
        "filename": "car_1.jpg",
        "annotated_base64": _B64_GREEN,
        "findings": {
            "index": 1,
            "filename": "car_1.jpg",
            "brand": "Honda",
            "model": "Civic",
            "color": "Red",
            "confidence": 0.88,
            "bbox": None,
            "status": "success",
            "error": None,
        },
    },
]

# All failed → "0/2 ảnh nhận dạng được"
BATCH_ALL_FAILED = [
    {**ANNOTATED_FAILED, "index": 0, "filename": "car_0.jpg"},
    {**ANNOTATED_FAILED, "index": 1, "filename": "car_1.jpg"},
]


def make_annotated_image(
    index: int,
    filename: str,
    brand: str | None = "Toyota",
    model_car: str | None = "Camry",
    color: str | None = "White",
    confidence: float = 0.92,
    status: str = "success",
    error: str | None = None,
    annotated_base64: str = _B64_RED,
) -> dict:
    """Factory for AnnotatedImage dicts with full schema."""
    return {
        "index": index,
        "filename": filename,
        "annotated_base64": annotated_base64,
        "findings": {
            "index": index,
            "filename": filename,
            "brand": brand,
            "model": model_car,
            "color": color,
            "confidence": confidence,
            "bbox": None,
            "status": status,
            "error": error,
        },
    }

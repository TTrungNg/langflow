"""
Mock VisionResult fixtures and image helpers for ImageAnnotatorComponent tests.
Defined in ACCEPTANCE.MD § "Node: Image Annotator".
"""
from __future__ import annotations

import base64
import io

import numpy as np

# ---------------------------------------------------------------------------
# Canonical VisionResult fixtures
#
# Bbox values are chosen so they fit inside the solid test image (300×200 resized)
# and scale cleanly by ×2 into the original space (600×400):
#   resized bbox  x=50, y=40, w=100, h=80
#   scaled bbox   x=100, y=80, w=200, h=160  (all within 600×400)
# ---------------------------------------------------------------------------

MOCK_RESULT_SUCCESS: dict = {
    "index": 0,
    "filename": "car_0.jpg",
    "brand": "Toyota",
    "model": "Camry",
    "color": "White",
    "confidence": 0.92,
    "bbox": {"x": 50, "y": 40, "width": 100, "height": 80},
    "status": "success",
    "error": None,
}

MOCK_RESULT_NO_BBOX: dict = {
    "index": 0,
    "filename": "car_0.jpg",
    "brand": "Honda",
    "model": "Civic",
    "color": "Red",
    "confidence": 0.85,
    "bbox": None,
    "status": "success",
    "error": None,
}

MOCK_RESULT_UNKNOWN: dict = {
    "index": 0,
    "filename": "car_0.jpg",
    "brand": None,
    "model": None,
    "color": None,
    "confidence": 0.3,
    "bbox": None,
    "status": "success",
    "error": None,
}

MOCK_RESULT_FAILED: dict = {
    "index": 0,
    "filename": "car_0.jpg",
    "brand": None,
    "model": None,
    "color": None,
    "confidence": 0.0,
    "bbox": None,
    "status": "failed",
    "error": "API rate limit exceeded",
}


def make_vision_result(
    index: int,
    filename: str,
    brand: str | None = "Toyota",
    model_car: str | None = "Camry",
    color: str | None = "White",
    confidence: float = 0.92,
    bbox: dict | None = None,
    status: str = "success",
    error: str | None = None,
) -> dict:
    return {
        "index": index,
        "filename": filename,
        "brand": brand,
        "model": model_car,
        "color": color,
        "confidence": confidence,
        "bbox": bbox if bbox is not None else {"x": 50, "y": 40, "width": 100, "height": 80},
        "status": status,
        "error": error,
    }


# ---------------------------------------------------------------------------
# ImageData factory — solid-color PNG for reliable pixel-level assertions
#
# Default geometry:
#   resized  300×200  (w×h)
#   original 600×400  → scale_x = scale_y = 2.0
#
# Background color: pure blue RGB(0, 0, 200) so the G-channel ≈ 0 everywhere.
# After a green (0, 255, 0) bbox annotation the G-channel at the boundary ≈ 255 —
# trivially distinguishable even if the component re-encodes as JPEG.
# ---------------------------------------------------------------------------

_DEFAULT_RESIZED: tuple[int, int] = (300, 200)    # width × height
_DEFAULT_ORIGINAL: tuple[int, int] = (600, 400)
_BG_RGB: tuple[int, int, int] = (0, 0, 200)       # blue: no green, no red


def make_solid_image_data(
    index: int,
    filename: str,
    resized_size: tuple[int, int] = _DEFAULT_RESIZED,
    original_size: tuple[int, int] = _DEFAULT_ORIGINAL,
    color_rgb: tuple[int, int, int] = _BG_RGB,
) -> dict:
    """Return ImageData backed by a solid-color lossless PNG.

    PNG is lossless so pixel values after decode are exact — no JPEG compression
    artifacts to worry about in assertions.
    """
    from PIL import Image

    w, h = resized_size
    img = Image.new("RGB", (w, h), color_rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode()
    return {
        "index": index,
        "filename": filename,
        "base64": f"data:image/png;base64,{b64}",
        "mime_type": "image/png",
        "size_bytes": len(raw),
        "original_width": original_size[0],
        "original_height": original_size[1],
        "resized_width": w,
        "resized_height": h,
    }


# Prebuilt stubs used directly in tests
SOLID_IMG_0 = make_solid_image_data(0, "car_0.jpg")
SOLID_IMG_1 = make_solid_image_data(1, "car_1.jpg")
SOLID_IMG_2 = make_solid_image_data(2, "car_2.jpg")

# Expected scaled bbox for MOCK_RESULT_SUCCESS on SOLID_IMG_0 (scale = 2)
# resized: x=50,y=40,w=100,h=80  →  original: x=100,y=80,w=200,h=160
SCALED_BBOX = {"x": 100, "y": 80, "width": 200, "height": 160}


# ---------------------------------------------------------------------------
# Output decoding helper
# ---------------------------------------------------------------------------

def decode_annotated_image(annotated_base64: str) -> np.ndarray:
    """Decode annotated_base64 → BGR uint8 numpy array (cv2 convention).

    Accepts both ``data:image/png;base64,...`` and ``data:image/jpeg;base64,...``.
    """
    import cv2

    _, encoded = annotated_base64.split(",", 1)
    arr = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode returned None — invalid image data")
    return img

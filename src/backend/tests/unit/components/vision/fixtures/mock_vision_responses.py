"""
Mock responses and helpers for VisionAnalysisComponent tests.
Defined in ACCEPTANCE.MD § "Node: Vision Analysis — Mock để test".
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# Canonical mock API responses (from ACCEPTANCE.MD)
# ---------------------------------------------------------------------------

SUCCESS_MOCK: dict = {
    "brand": "Toyota",
    "model": "Camry",
    "color": "White",
    "confidence": 0.92,
    "bbox": {"x": 10, "y": 20, "width": 200, "height": 150},
}

NO_CAR_MOCK: dict = {
    "brand": None,
    "model": None,
    "color": None,
    "confidence": 0.3,
    "bbox": None,
}

API_FAIL_MOCK = RuntimeError("API rate limit exceeded")


# ---------------------------------------------------------------------------
# ImageData factory — matches the schema from ACCEPTANCE.MD
# ---------------------------------------------------------------------------

def make_image_data(index: int, filename: str = "test.jpg") -> dict:
    return {
        "index": index,
        "filename": filename,
        "base64": f"data:image/jpeg;base64,/9j/fakedata{index}==",
        "mime_type": "image/jpeg",
        "size_bytes": 51_200,
        "original_width": 1500,
        "original_height": 1000,
        "resized_width": 1024,
        "resized_height": 683,
    }


# ---------------------------------------------------------------------------
# Mock LLM factory
# ---------------------------------------------------------------------------

def make_mock_llm(responses: list) -> MagicMock:
    """Return a mock LLM whose ainvoke() yields *responses* in order.

    Each entry in *responses* can be:
    - dict            → serialised as JSON, returned as an AIMessage-like object
    - Exception instance → raised when ainvoke is called for that slot
    """
    side_effects: list = []
    for resp in responses:
        if isinstance(resp, BaseException):
            side_effects.append(resp)
        else:
            msg = MagicMock()
            msg.content = json.dumps(resp)
            side_effects.append(msg)

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=side_effects)
    return mock_llm


def make_mock_llm_raw(raw_responses: list[str]) -> MagicMock:
    """Return a mock LLM whose ainvoke() yields raw string content (not auto-serialised).

    Use this to test how the component handles non-standard responses such as
    markdown-wrapped JSON or empty strings returned by some Ollama models.
    """
    side_effects = []
    for text in raw_responses:
        msg = MagicMock()
        msg.content = text
        side_effects.append(msg)

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=side_effects)
    return mock_llm

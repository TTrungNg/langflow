from __future__ import annotations

import asyncio
import json
from typing import Any

from lfx.base.models.unified_models import get_language_model_options, get_llm, handle_model_input_update
from lfx.custom.custom_component.component import Component
from lfx.inputs.inputs import IntInput, ModelInput
from lfx.io import DataInput, MultilineInput, Output

_SYSTEM_PROMPT = """\
Analyze the car in the image. Respond ONLY with valid JSON:
{
  "brand": "string or null",
  "model": "string or null",
  "color": "string or null",
  "confidence": float 0.0-1.0,
  "bbox": {"x": int, "y": int, "width": int, "height": int} or null — tight bounding box that MUST enclose the ENTIRE VEHICLE (leftmost to rightmost point, top of roof to bottom of wheels, bumper to bumper),
  "explanation": "string — describe what visual evidence supports your answer, and any ambiguities or factors that reduce confidence"
}"""

# Models known to NOT support image input — blocklist intentionally minimal.
# Modern models from major providers (gpt-4o, claude-3+, gemini) all support vision.
_NON_VISION_PATTERNS = frozenset([
    "gpt-3.5",
    "text-davinci",
    "text-ada",
    "embed",
    "whisper",
    "tts",
    "dall-e",
    "claude-2",
    "claude-instant",
])


def _vision_warning(model_name: str) -> str | None:
    name = model_name.lower()
    if any(p in name for p in _NON_VISION_PATTERNS):
        return (
            f"⚠️ '{model_name}' có thể không hỗ trợ vision. "
            "Hãy chọn gpt-4o, claude-3-*, gemini-1.5+, hoặc llava."
        )
    return None


def _extract_json(text: str, image_filename: str) -> dict:
    """Extract JSON dict from model response, handling markdown wrappers and empty content.

    Ollama models often wrap JSON in ```json ... ``` code blocks or return
    plain text when they cannot analyze an image.
    """
    if not text or not text.strip():
        raise ValueError(
            f"Model returned empty response for '{image_filename}'. "
            "Ensure the model supports vision (e.g. llava, llama3.2-vision)."
        )
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` wrappers
    if text.startswith("```"):
        lines = text.splitlines()
        end_idx = next((i for i in range(len(lines) - 1, 0, -1) if lines[i].strip() == "```"), None)
        inner = lines[1:end_idx] if end_idx else lines[1:]
        text = "\n".join(inner).strip()
    # Find JSON object boundaries — ignore any surrounding prose
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(
            f"No JSON object found in model response for '{image_filename}'. "
            f"Raw response: {text!r:.200}"
        )
    return json.loads(text[start:end])


class VisionAnalysisComponent(Component):
    display_name = "Vision Analysis"
    description = "Nhận List[ImageData], gọi Vision API song song, trả về List[VisionResult]."
    icon = "ScanEye"
    name = "VisionAnalysis"

    inputs = [
        DataInput(
            name="images",
            display_name="Images",
            info="List[ImageData] from Image Input component.",
            is_list=True,
        ),
        ModelInput(
            name="model",
            display_name="Language Model",
            info="Yêu cầu model hỗ trợ vision (gpt-4o, claude-3-*, gemini-1.5+, llava...).",
            real_time_refresh=True,
        ),
        MultilineInput(
            name="system_prompt",
            display_name="System Prompt",
            value=_SYSTEM_PROMPT,
            input_types=[],
        ),
        IntInput(
            name="max_concurrent",
            display_name="Max Concurrent Calls",
            value=3,
            info="Maximum number of parallel Vision API calls.",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Vision Results",
            name="results",
            method="analyze_images",
            types=["Data"],
        ),
    ]

    async def update_build_config(
        self, build_config: dict, field_value: Any, field_name: str | None = None
    ) -> dict:
        return handle_model_input_update(
            component=self,
            build_config=dict(build_config),
            field_value=field_value,
            field_name=field_name,
            get_options_func=lambda user_id=None: get_language_model_options(user_id=user_id),
        )

    async def analyze_images(self) -> list[dict]:
        if not self.model:
            raise ValueError("Model name is required")

        model_entry = self.model[0] if isinstance(self.model, list) and self.model else {}
        model_name: str = model_entry.get("name", "") if isinstance(model_entry, dict) else str(model_entry)

        if not model_name:
            raise ValueError("Model name is required")

        warning = _vision_warning(model_name)
        if warning:
            self.status = warning

        images: list[dict] = self.images or []
        if not images:
            return []

        max_concurrent: int = getattr(self, "max_concurrent", 3) or 3
        system_prompt: str = getattr(self, "system_prompt", _SYSTEM_PROMPT) or _SYSTEM_PROMPT

        llm = get_llm(
            model=self.model,
            user_id=self._user_id,
            api_key=getattr(self, "api_key", None),
            ollama_base_url=getattr(self, "ollama_base_url", None),
            watsonx_url=getattr(self, "base_url_ibm_watsonx", None),
            watsonx_project_id=getattr(self, "project_id", None),
        )

        sem = asyncio.Semaphore(max_concurrent)

        async def _analyze_one(img: dict) -> dict:
            async with sem:
                # Include image dimensions in the prompt so LLM knows the exact bbox space
                img_width = img.get("resized_width", img.get("original_width", 0))
                img_height = img.get("resized_height", img.get("original_height", 0))
                size_context = f"\n[Image size: {img_width}x{img_height} pixels]"

                content = [
                    {"type": "text", "text": f"Analyze this image{size_context}"},
                    {"type": "image_url", "image_url": {"url": img["base64"]}},
                ]
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ]
                response = await llm.ainvoke(messages)
                parsed: dict = _extract_json(response.content, img["filename"])
                # Pass through all fields from parsed JSON so user-defined fields
                # in the system prompt (e.g. year, plate) are preserved automatically.
                result = dict(parsed)
                result["confidence"] = float(parsed.get("confidence", 0.0))
                result["index"] = img["index"]
                result["filename"] = img["filename"]
                result["status"] = "success"
                result["error"] = None
                return result

        tasks = [_analyze_one(img) for img in images]
        raw = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[dict] = []
        for img, outcome in zip(images, raw):
            if isinstance(outcome, BaseException):
                raise RuntimeError(str(outcome)) from outcome
            results.append(outcome)

        return results

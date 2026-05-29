from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path

import aiofiles
import cv2
import numpy as np
import platformdirs
from PIL import Image

from lfx.custom.custom_component.component import Component
from lfx.io import DataInput, Output
from lfx.services.deps import get_storage_service

_GREEN = (0, 255, 0)   # BGR
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.8
_THICKNESS = 2


def _decode_base64_image(b64_str: str) -> np.ndarray:
    """Decode data:image/...;base64,... → BGR uint8 numpy array."""
    _, encoded = b64_str.split(",", 1)
    arr = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode returned None — invalid image data")
    return img


def _encode_image_base64(img: np.ndarray, mime_type: str = "image/png") -> str:
    ext = ".png" if "png" in mime_type else ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"cv2.imencode failed for {ext}")
    b64 = base64.b64encode(buf.tobytes()).decode()
    return f"data:{mime_type};base64,{b64}"


def _build_label(result: dict) -> str:
    brand = result.get("brand")
    if not brand:
        return "Unknown Vehicle"
    model = result.get("model") or ""
    color = result.get("color") or ""
    return f"{brand} {model} - {color}".strip(" -")




class ImageAnnotatorComponent(Component):
    display_name = "Image Annotator"
    description = "Scale bounding box về ảnh gốc, vẽ annotation, trả về List[AnnotatedImage]."
    icon = "PencilLine"
    name = "ImageAnnotator"

    inputs = [
        DataInput(name="images", display_name="Images", is_list=True),
        DataInput(name="vision_results", display_name="Vision Results", is_list=True),
    ]

    outputs = [
        Output(
            display_name="Annotated Images",
            name="annotated",
            method="annotate_images",
            types=["Data"],
        ),
    ]

    async def annotate_images(self) -> list[dict]:
        images: list[dict] = self.images or []
        vision_results: list[dict] = self.vision_results or []

        if len(images) != len(vision_results):
            raise ValueError(
                f"images count ({len(images)}) != vision_results count ({len(vision_results)})"
            )

        if not images:
            return []

        # Get flow_id from vertex context
        vertex = getattr(self, "_vertex", None)
        flow_id = None
        if vertex and hasattr(vertex, "graph"):
            flow_id = str(vertex.graph.flow_id) if vertex.graph.flow_id else None

        results = []
        for img, res in zip(images, vision_results):
            # Decode base64 image to original dimensions
            img_array = _decode_base64_image(img["base64"])
            original_width = img.get("original_width", img_array.shape[1])
            original_height = img.get("original_height", img_array.shape[0])
            resized_width = img.get("resized_width", img_array.shape[1])
            resized_height = img.get("resized_height", img_array.shape[0])

            # Scale image to original dimensions
            scale_x = original_width / resized_width if resized_width > 0 else 1
            scale_y = original_height / resized_height if resized_height > 0 else 1
            if scale_x != 1 or scale_y != 1:
                img_array = cv2.resize(
                    img_array,
                    (original_width, original_height),
                    interpolation=cv2.INTER_LINEAR,
                )

            # Draw annotations if bbox exists
            bbox = res.get("bbox")
            if bbox:
                # Convert bbox from {x, y, width, height} to [x1, y1, x2, y2]
                x1 = int(bbox.get("x", 0) * scale_x)
                y1 = int(bbox.get("y", 0) * scale_y)
                x2 = int((bbox.get("x", 0) + bbox.get("width", 0)) * scale_x)
                y2 = int((bbox.get("y", 0) + bbox.get("height", 0)) * scale_y)

                # Clamp bbox to image bounds
                img_height, img_width = img_array.shape[:2]
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(x1 + 1, min(x2, img_width))
                y2 = max(y1 + 1, min(y2, img_height))

                # Only draw if bbox has valid dimensions
                if x2 > x1 and y2 > y1:
                    # Draw green rectangle
                    cv2.rectangle(img_array, (x1, y1), (x2, y2), _GREEN, _THICKNESS)

                    # Build and draw label
                    label = _build_label(res)
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, _FONT, _FONT_SCALE, _THICKNESS
                    )
                    label_y = max(y1 - 5, text_height + 5)
                    cv2.rectangle(
                        img_array,
                        (x1, label_y - text_height - 5),
                        (x1 + text_width + 5, label_y),
                        _BLACK,
                        -1,
                    )
                    cv2.putText(
                        img_array,
                        label,
                        (x1 + 2, label_y - 2),
                        _FONT,
                        _FONT_SCALE,
                        _WHITE,
                        _THICKNESS,
                    )

            # Encode back to base64
            annotated_base64 = _encode_image_base64(img_array)

            # Save annotated image to disk if flow_id is available
            file_path = None
            if flow_id:
                filename = f"ImageAnnotator_{uuid.uuid4().hex}.jpg"
                cache_dir = platformdirs.user_cache_dir("langflow", "langflow")
                image_dir = Path(cache_dir) / flow_id
                image_dir.mkdir(parents=True, exist_ok=True)

                file_path_obj = image_dir / filename
                _, buf = cv2.imencode(".jpg", img_array)
                async with aiofiles.open(file_path_obj, "wb") as f:
                    await f.write(buf.tobytes())

                file_path = f"{flow_id}/{filename}"

            results.append(
                {
                    "annotated_base64": annotated_base64,
                    "file_path": file_path,
                }
            )

        return results

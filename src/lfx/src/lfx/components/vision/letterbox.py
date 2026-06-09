from __future__ import annotations

import base64

import cv2
import numpy as np

from lfx.custom.custom_component.component import Component
from lfx.inputs.inputs import IntInput
from lfx.io import DataInput, Output


def _decode_base64_image(b64_str: str) -> tuple[np.ndarray, str]:
    """Decode data:image/...;base64,... → (BGR uint8 array, mime_type)."""
    header, encoded = b64_str.split(",", 1)
    mime_type = header.split(":")[1].split(";")[0]
    arr = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode returned None — invalid image data")
    return img, mime_type


def _encode_image_base64(img: np.ndarray, mime_type: str) -> str:
    ext = ".png" if "png" in mime_type else ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"cv2.imencode failed for {ext}")
    b64 = base64.b64encode(buf.tobytes()).decode()
    return f"data:{mime_type};base64,{b64}"


def _letterbox(img: np.ndarray, target_size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize by longest edge then pad to target_size × target_size with black.

    Returns (letterboxed_image, scale, pad_left, pad_top).
    scale and pad values are needed downstream to decode bbox coordinates.
    """
    h, w = img.shape[:2]
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    pad_right = target_size - new_w - pad_left
    pad_bottom = target_size - new_h - pad_top

    letterboxed = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return letterboxed, scale, pad_left, pad_top


class LetterboxComponent(Component):
    display_name = "Letterbox Preprocess"
    description = (
        "Resize ảnh về target_size × target_size bằng letterbox (giữ aspect ratio, pad đen). "
        "Lưu scale và pad để decode bbox sau. Output: List[ProcessedImage]."
    )
    icon = "Crop"
    name = "LetterboxPreprocess"

    inputs = [
        DataInput(
            name="images",
            display_name="Images",
            is_list=True,
            info="List[ImageData] từ Image Input component.",
        ),
        IntInput(
            name="target_size",
            display_name="Target Size (px)",
            value=640,
            info="Kích thước ảnh vuông đầu ra (mặc định 640). Configurable cho mọi model.",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Processed Images",
            name="processed_images",
            method="letterbox_images",
            types=["Data"],
        ),
    ]

    async def letterbox_images(self) -> list[dict]:
        images: list[dict] = self.images or []
        target_size: int = getattr(self, "target_size", 640) or 640

        results: list[dict] = []
        for image_data in images:
            img, mime_type = _decode_base64_image(image_data["base64"])
            original_height, original_width = img.shape[:2]

            letterboxed, scale, pad_left, pad_top = _letterbox(img, target_size)
            base64_resized = _encode_image_base64(letterboxed, mime_type)

            results.append({
                "index": image_data["index"],
                "filename": image_data["filename"],
                "base64_resized": base64_resized,
                "target_size": target_size,
                "scale": float(scale),
                "pad_left": pad_left,
                "pad_top": pad_top,
                "original_width": original_width,
                "original_height": original_height,
            })

        return results

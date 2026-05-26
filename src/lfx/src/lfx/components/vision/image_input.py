from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageFile

# Allow processing of truncated/incomplete image files (e.g. partial uploads).
ImageFile.LOAD_TRUNCATED_IMAGES = True

from lfx.custom.custom_component.component import Component
from lfx.inputs.inputs import IntInput
from lfx.io import FileInput, Output

_SUPPORTED = {".jpg", ".jpeg", ".png"}
_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
_MAX_BYTES = 2 * 1024 * 1024  # 2 MB


class ImageInputComponent(Component):
    display_name = "Image Input"
    description = "Nhận ảnh upload từ user, validate, resize, convert sang List[ImageData]."
    icon = "ImageIcon"
    name = "ImageInput"

    inputs = [
        FileInput(
            name="files",
            display_name="Images",
            file_types=["jpg", "jpeg", "png"],
            info="Image files to process. Multiple files allowed.",
            is_list=True,
            temp_file=True,
        ),
        IntInput(
            name="max_size",
            display_name="Max Size (px)",
            value=1024,
            info="Maximum dimension (longest edge) after resize. PIL thumbnail keeps aspect ratio and never upscales.",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Image List",
            name="images",
            method="process_images",
            types=["Data"],
        ),
    ]

    async def process_images(self) -> list[dict]:
        files: list[str] = self.files or []
        if isinstance(files, str):
            files = [files]
        files = [f for f in files if f]

        max_size: int = getattr(self, "max_size", 1024) or 1024

        results: list[dict] = []
        skip_reasons: list[str] = []

        for index, file_path in enumerate(files):
            path = Path(file_path)

            if not path.exists():
                self.log(f"Skipped {path.name}: file not found")
                skip_reasons.append(f"{path.name}: file not found")
                continue

            if path.suffix.lower() not in _SUPPORTED:
                self.log(f"Skipped {path.name}: unsupported format")
                skip_reasons.append(f"{path.name}: unsupported format")
                continue

            size_bytes = path.stat().st_size
            if size_bytes > _MAX_BYTES:
                self.log(f"Skipped {path.name}: exceeds 2mb limit")
                skip_reasons.append(f"{path.name}: exceeds 2mb limit")
                continue

            img = Image.open(path)
            original_width, original_height = img.size
            mime_type = _MIME[path.suffix.lower()]

            # thumbnail() shrinks in-place to fit (max_size, max_size) keeping aspect ratio;
            # it never upscales, so small images stay unchanged.
            img.thumbnail((max_size, max_size))
            resized_width, resized_height = img.size

            buf = io.BytesIO()
            fmt = "JPEG" if mime_type == "image/jpeg" else "PNG"
            # JPEG encoder requires RGB; PNG can handle RGBA or P-mode directly.
            save_img = img.convert("RGB") if fmt == "JPEG" else img
            save_img.save(buf, format=fmt)
            b64 = base64.b64encode(buf.getvalue()).decode()

            results.append({
                "index": index,
                "filename": path.name,
                "base64": f"data:{mime_type};base64,{b64}",
                "mime_type": mime_type,
                "size_bytes": size_bytes,
                "original_width": original_width,
                "original_height": original_height,
                "resized_width": resized_width,
                "resized_height": resized_height,
            })

        if not results:
            if skip_reasons:
                reasons = "; ".join(skip_reasons)
                raise ValueError(f"No valid images to process. Skipped: {reasons}")
            raise ValueError("No valid images to process")

        return results

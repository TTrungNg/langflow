"""
Unit tests for LetterboxComponent.
All tests FAIL until lfx.components.vision.letterbox is implemented — intentional.

Acceptance source: ACCEPTANCE_FACE.MD §§ "Tư duy thiết kế", "Context kỹ thuật",
"Node: Letterbox Preprocess"
"""
import base64
import math
from pathlib import Path

import pytest

from lfx.components.vision.letterbox import LetterboxComponent
from tests.base import ComponentTestBaseWithoutClient
from tests.unit.components.vision.fixtures.mock_preprocess import MOCK_LARGE_IMAGE, MOCK_SMALL_IMAGE

FIXTURES = Path(__file__).parent / "fixtures"
VALID_JPEG = FIXTURES / "valid_car.jpeg"       # 284×177 — matches MOCK_SMALL_IMAGE
VALID_LARGE = FIXTURES / "valid_car_large.jpg"  # 1500×1000

DEFAULT_TARGET = 640

# VALID_LARGE letterbox math (1500×1000, target=640)
_LARGE_SCALE = DEFAULT_TARGET / 1500          # ~0.4267
_LARGE_RESIZED_H = int(1000 * _LARGE_SCALE)  # 426
_LARGE_PAD_TOP = (DEFAULT_TARGET - _LARGE_RESIZED_H) // 2  # 107
_LARGE_PAD_LEFT = 0


def _image_data(path: Path, index: int = 0) -> dict:
    """Build an ImageData dict from a fixture file (mimics ImageInputComponent output)."""
    from PIL import Image

    raw = path.read_bytes()
    with Image.open(path) as img:
        orig_w, orig_h = img.size
    encoded = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
    return {
        "index": index,
        "filename": path.name,
        "base64": encoded,
        "mime_type": "image/jpeg",
        "size_bytes": len(raw),
        "original_width": orig_w,
        "original_height": orig_h,
        "resized_width": orig_w,
        "resized_height": orig_h,
    }


class TestLetterboxComponent(ComponentTestBaseWithoutClient):
    # ------------------------------------------------------------------
    # Required base-class fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def component_class(self):
        return LetterboxComponent

    @pytest.fixture
    def default_kwargs(self):
        return {
            "images": [_image_data(VALID_JPEG)],
            "target_size": DEFAULT_TARGET,
        }

    @pytest.fixture
    def file_names_mapping(self):
        return []  # new component, no version history yet

    # ------------------------------------------------------------------
    # Acceptance: output là List[ProcessedImage]
    # ------------------------------------------------------------------

    async def test_output_is_list(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.letterbox_images()
        assert isinstance(result, list)

    async def test_single_image_returns_one_item(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.letterbox_images()
        assert len(result) == 1

    async def test_multiple_images_all_processed(self, component_class):
        component = await self.component_setup(
            component_class,
            {
                "images": [_image_data(VALID_JPEG, 0), _image_data(VALID_LARGE, 1)],
                "target_size": DEFAULT_TARGET,
            },
        )
        result = await component.letterbox_images()
        assert len(result) == 2

    # ------------------------------------------------------------------
    # Acceptance: ProcessedImage schema — tất cả fields bắt buộc phải có
    # ------------------------------------------------------------------

    async def test_processedimage_has_all_required_fields(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.letterbox_images()
        item = result[0]
        required = {
            "index",
            "filename",
            "base64_resized",
            "target_size",
            "scale",
            "pad_left",
            "pad_top",
            "original_width",
            "original_height",
        }
        assert required.issubset(item.keys())

    async def test_bbox_decode_fields_are_present(self, component_class, default_kwargs):
        """scale, pad_left, pad_top bắt buộc để decode bbox — không được thiếu."""
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.letterbox_images()
        item = result[0]
        assert "scale" in item
        assert "pad_left" in item
        assert "pad_top" in item

    async def test_field_types_are_correct(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.letterbox_images()
        item = result[0]
        assert isinstance(item["index"], int)
        assert isinstance(item["filename"], str)
        assert isinstance(item["base64_resized"], str)
        assert isinstance(item["target_size"], int)
        assert isinstance(item["scale"], float)
        assert isinstance(item["pad_left"], int)
        assert isinstance(item["pad_top"], int)
        assert isinstance(item["original_width"], int)
        assert isinstance(item["original_height"], int)

    # ------------------------------------------------------------------
    # Acceptance: base64_resized — ảnh sau letterbox hợp lệ
    # ------------------------------------------------------------------

    async def test_base64_resized_starts_with_data_image(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.letterbox_images()
        assert result[0]["base64_resized"].startswith("data:image/")

    async def test_base64_resized_is_decodable(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.letterbox_images()
        b64_str = result[0]["base64_resized"]
        _, encoded = b64_str.split(",", 1)
        decoded = base64.b64decode(encoded)
        assert len(decoded) > 0

    # ------------------------------------------------------------------
    # Acceptance: target_size configurable (default 640)
    # ------------------------------------------------------------------

    async def test_default_target_size_is_640(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.letterbox_images()
        assert result[0]["target_size"] == DEFAULT_TARGET

    async def test_custom_target_size_is_reflected_in_output(self, component_class):
        component = await self.component_setup(
            component_class,
            {"images": [_image_data(VALID_JPEG)], "target_size": 320},
        )
        result = await component.letterbox_images()
        assert result[0]["target_size"] == 320

    # ------------------------------------------------------------------
    # Acceptance: scale theo cạnh dài
    # ------------------------------------------------------------------

    async def test_small_image_scale_is_ratio_of_target_to_longest_edge(self, component_class):
        """284×177 — cạnh dài là 284 → scale = 640/284."""
        component = await self.component_setup(
            component_class,
            {"images": [_image_data(VALID_JPEG)], "target_size": DEFAULT_TARGET},
        )
        result = await component.letterbox_images()
        item = result[0]
        expected_scale = MOCK_SMALL_IMAGE["expected_scale"]  # 640/284
        assert math.isclose(item["scale"], expected_scale, rel_tol=1e-4)

    async def test_large_image_scale_is_ratio_of_target_to_longest_edge(self, component_class):
        """1500×1000 — cạnh dài là 1500 → scale = 640/1500."""
        component = await self.component_setup(
            component_class,
            {"images": [_image_data(VALID_LARGE)], "target_size": DEFAULT_TARGET},
        )
        result = await component.letterbox_images()
        item = result[0]
        assert math.isclose(item["scale"], _LARGE_SCALE, rel_tol=1e-4)

    async def test_scale_is_strictly_positive(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.letterbox_images()
        assert result[0]["scale"] > 0.0

    # ------------------------------------------------------------------
    # Acceptance: pad_left và pad_top đúng theo letterbox algorithm
    # ------------------------------------------------------------------

    async def test_landscape_image_pad_left_is_zero(self, component_class):
        """284×177 — landscape → longest edge là width → pad_left = 0."""
        component = await self.component_setup(
            component_class,
            {"images": [_image_data(VALID_JPEG)], "target_size": DEFAULT_TARGET},
        )
        result = await component.letterbox_images()
        assert result[0]["pad_left"] == MOCK_SMALL_IMAGE["expected_pad_left"]  # 0

    async def test_landscape_image_pad_top_is_correct(self, component_class):
        """284×177 → resized_h=399 → pad_top=(640-399)//2=120."""
        component = await self.component_setup(
            component_class,
            {"images": [_image_data(VALID_JPEG)], "target_size": DEFAULT_TARGET},
        )
        result = await component.letterbox_images()
        assert result[0]["pad_top"] == MOCK_SMALL_IMAGE["expected_pad_top"]  # 120

    async def test_large_landscape_image_pad_left_is_zero(self, component_class):
        """1500×1000 — landscape → pad_left = 0."""
        component = await self.component_setup(
            component_class,
            {"images": [_image_data(VALID_LARGE)], "target_size": DEFAULT_TARGET},
        )
        result = await component.letterbox_images()
        assert result[0]["pad_left"] == _LARGE_PAD_LEFT  # 0

    async def test_large_landscape_image_pad_top_is_correct(self, component_class):
        """1500×1000 → resized_h=426 → pad_top=(640-426)//2=107."""
        component = await self.component_setup(
            component_class,
            {"images": [_image_data(VALID_LARGE)], "target_size": DEFAULT_TARGET},
        )
        result = await component.letterbox_images()
        assert result[0]["pad_top"] == _LARGE_PAD_TOP  # 107

    async def test_portrait_image_pad_top_is_zero(self, component_class, tmp_path):
        """Portrait ảnh (100×300) — longest edge là height → pad_top = 0."""
        from PIL import Image

        portrait = tmp_path / "portrait.jpg"
        img = Image.new("RGB", (100, 300), color=(128, 64, 32))
        img.save(portrait, format="JPEG")
        raw = portrait.read_bytes()
        encoded = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
        image_data = {
            "index": 0,
            "filename": "portrait.jpg",
            "base64": encoded,
            "mime_type": "image/jpeg",
            "size_bytes": len(raw),
            "original_width": 100,
            "original_height": 300,
            "resized_width": 100,
            "resized_height": 300,
        }
        component = await self.component_setup(
            component_class,
            {"images": [image_data], "target_size": DEFAULT_TARGET},
        )
        result = await component.letterbox_images()
        assert result[0]["pad_top"] == 0

    async def test_portrait_image_pad_left_is_correct(self, component_class, tmp_path):
        """Portrait 100×300, target=640 → scale=640/300 → resized_w=213 → pad_left=(640-213)//2=213."""
        from PIL import Image

        portrait = tmp_path / "portrait.jpg"
        img = Image.new("RGB", (100, 300), color=(128, 64, 32))
        img.save(portrait, format="JPEG")
        raw = portrait.read_bytes()
        encoded = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
        image_data = {
            "index": 0,
            "filename": "portrait.jpg",
            "base64": encoded,
            "mime_type": "image/jpeg",
            "size_bytes": len(raw),
            "original_width": 100,
            "original_height": 300,
            "resized_width": 100,
            "resized_height": 300,
        }
        scale = DEFAULT_TARGET / 300
        expected_resized_w = int(100 * scale)
        expected_pad_left = (DEFAULT_TARGET - expected_resized_w) // 2
        component = await self.component_setup(
            component_class,
            {"images": [image_data], "target_size": DEFAULT_TARGET},
        )
        result = await component.letterbox_images()
        assert result[0]["pad_left"] == expected_pad_left

    async def test_square_image_has_zero_padding(self, component_class, tmp_path):
        """Square ảnh 300×300 → pad_left=0, pad_top=0."""
        from PIL import Image

        square = tmp_path / "square.jpg"
        img = Image.new("RGB", (300, 300), color=(100, 100, 100))
        img.save(square, format="JPEG")
        raw = square.read_bytes()
        encoded = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
        image_data = {
            "index": 0,
            "filename": "square.jpg",
            "base64": encoded,
            "mime_type": "image/jpeg",
            "size_bytes": len(raw),
            "original_width": 300,
            "original_height": 300,
            "resized_width": 300,
            "resized_height": 300,
        }
        component = await self.component_setup(
            component_class,
            {"images": [image_data], "target_size": DEFAULT_TARGET},
        )
        result = await component.letterbox_images()
        item = result[0]
        assert item["pad_left"] == 0
        assert item["pad_top"] == 0

    # ------------------------------------------------------------------
    # Acceptance: original_width/height được giữ nguyên
    # ------------------------------------------------------------------

    async def test_original_dimensions_are_preserved(self, component_class):
        component = await self.component_setup(
            component_class,
            {"images": [_image_data(VALID_JPEG)], "target_size": DEFAULT_TARGET},
        )
        result = await component.letterbox_images()
        item = result[0]
        assert item["original_width"] == MOCK_SMALL_IMAGE["original_width"]   # 284
        assert item["original_height"] == MOCK_SMALL_IMAGE["original_height"]  # 177

    async def test_large_image_original_dimensions_are_preserved(self, component_class):
        component = await self.component_setup(
            component_class,
            {"images": [_image_data(VALID_LARGE)], "target_size": DEFAULT_TARGET},
        )
        result = await component.letterbox_images()
        item = result[0]
        assert item["original_width"] == 1500
        assert item["original_height"] == 1000

    # ------------------------------------------------------------------
    # Acceptance: giữ đúng thứ tự index
    # ------------------------------------------------------------------

    async def test_output_preserves_input_order(self, component_class):
        component = await self.component_setup(
            component_class,
            {
                "images": [_image_data(VALID_JPEG, 0), _image_data(VALID_LARGE, 1)],
                "target_size": DEFAULT_TARGET,
            },
        )
        result = await component.letterbox_images()
        assert result[0]["filename"] == VALID_JPEG.name
        assert result[1]["filename"] == VALID_LARGE.name

    async def test_output_indices_match_input_indices(self, component_class):
        component = await self.component_setup(
            component_class,
            {
                "images": [_image_data(VALID_JPEG, 0), _image_data(VALID_LARGE, 1)],
                "target_size": DEFAULT_TARGET,
            },
        )
        result = await component.letterbox_images()
        assert [item["index"] for item in result] == [0, 1]

    async def test_filename_is_passed_through(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.letterbox_images()
        assert result[0]["filename"] == VALID_JPEG.name

    # ------------------------------------------------------------------
    # Acceptance: pad + scale consistency (pad không vượt quá target_size / 2)
    # ------------------------------------------------------------------

    async def test_pad_top_does_not_exceed_half_target_size(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.letterbox_images()
        item = result[0]
        assert 0 <= item["pad_top"] <= DEFAULT_TARGET // 2

    async def test_pad_left_does_not_exceed_half_target_size(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.letterbox_images()
        item = result[0]
        assert 0 <= item["pad_left"] <= DEFAULT_TARGET // 2

    async def test_scale_pad_consistent_with_target_size(self, component_class, default_kwargs):
        """resized_h = round(orig_h * scale) → pad_top*2 + resized_h ≤ target_size."""
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.letterbox_images()
        item = result[0]
        resized_h = int(item["original_height"] * item["scale"])
        assert item["pad_top"] * 2 + resized_h <= item["target_size"]

"""
Unit tests for ImageInputComponent.
All tests FAIL until lfx.components.vision.image_input is implemented — intentional.

Acceptance source: ACCEPTANCE.MD §§ "Tư duy thiết kế", "Context kỹ thuật", "Node: Image Input"
"""
from pathlib import Path

import pytest

from lfx.components.vision.image_input import ImageInputComponent
from tests.base import ComponentTestBaseWithoutClient

FIXTURES = Path(__file__).parent / "fixtures"
VALID_JPG = FIXTURES / "valid_car.jpg"          # 300×168, well below 1024px
VALID_JPEG = FIXTURES / "valid_car.jpeg"         # 284×177, well below 1024px
VALID_LARGE = FIXTURES / "valid_car_large.jpg"   # 1500×1000, longest edge > 1024px
INVALID_TXT = FIXTURES / "invalid_file.txt"
LARGE_FILE = FIXTURES / "large_image.jpg"        # > 2 MB
MISSING = FIXTURES / "does_not_exist.jpg"

# PIL thumbnail rule: longest edge of valid_car_large.jpg is 1500 → scaled to 1024
LARGE_ORIGINAL_WIDTH = 1500
LARGE_ORIGINAL_HEIGHT = 1000
MAX_SIZE = 1024  # default configurable value


class TestImageInputComponent(ComponentTestBaseWithoutClient):
    # ------------------------------------------------------------------
    # Required base-class fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def component_class(self):
        return ImageInputComponent

    @pytest.fixture
    def default_kwargs(self):
        return {"files": [str(VALID_JPG)]}

    @pytest.fixture
    def file_names_mapping(self):
        return []  # new component, no version history yet

    # ------------------------------------------------------------------
    # Acceptance: nhận một hoặc nhiều file upload cùng lúc
    # ------------------------------------------------------------------

    async def test_accepts_single_file(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.process_images()
        assert len(result) == 1

    async def test_accepts_multiple_files_simultaneously(self, component_class):
        component = await self.component_setup(
            component_class,
            {"files": [str(VALID_JPG), str(VALID_JPEG)]},
        )
        result = await component.process_images()
        assert len(result) == 2

    # ------------------------------------------------------------------
    # Acceptance: chấp nhận .jpg, .jpeg, .png
    # ------------------------------------------------------------------

    async def test_accepts_jpg_format(self, component_class):
        component = await self.component_setup(component_class, {"files": [str(VALID_JPG)]})
        result = await component.process_images()
        assert len(result) == 1

    async def test_accepts_jpeg_format(self, component_class):
        component = await self.component_setup(component_class, {"files": [str(VALID_JPEG)]})
        result = await component.process_images()
        assert len(result) == 1

    async def test_accepts_png_format(self, component_class, tmp_path):
        # Minimal valid 1×1 PNG (67 bytes)
        png_bytes = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde"
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        png_file = tmp_path / "test_car.png"
        png_file.write_bytes(png_bytes)
        component = await self.component_setup(component_class, {"files": [str(png_file)]})
        result = await component.process_images()
        assert len(result) == 1

    # ------------------------------------------------------------------
    # Acceptance: validate trước khi process — skip + log warning
    # ------------------------------------------------------------------

    async def test_unsupported_format_is_skipped(self, component_class):
        component = await self.component_setup(
            component_class,
            {"files": [str(INVALID_TXT), str(VALID_JPG)]},
        )
        result = await component.process_images()
        assert len(result) == 1
        assert result[0]["filename"] == VALID_JPG.name

    async def test_unsupported_format_logs_correct_warning(self, component_class):
        component = await self.component_setup(
            component_class,
            {"files": [str(INVALID_TXT), str(VALID_JPG)]},
        )
        await component.process_images()
        expected = f"Skipped {INVALID_TXT.name}: unsupported format"
        warning_texts = [str(c) for c in component.log.call_args_list]
        assert any(expected in w for w in warning_texts)

    async def test_nonexistent_file_is_skipped(self, component_class):
        component = await self.component_setup(
            component_class,
            {"files": [str(MISSING), str(VALID_JPG)]},
        )
        result = await component.process_images()
        assert len(result) == 1

    async def test_nonexistent_file_logs_warning(self, component_class):
        component = await self.component_setup(
            component_class,
            {"files": [str(MISSING), str(VALID_JPG)]},
        )
        await component.process_images()
        warning_texts = [str(c) for c in component.log.call_args_list]
        assert any(MISSING.name in w for w in warning_texts)

    async def test_file_exceeding_2mb_is_skipped(self, component_class):
        component = await self.component_setup(
            component_class,
            {"files": [str(LARGE_FILE), str(VALID_JPG)]},
        )
        result = await component.process_images()
        assert len(result) == 1

    async def test_file_exceeding_2mb_logs_correct_warning(self, component_class):
        component = await self.component_setup(
            component_class,
            {"files": [str(LARGE_FILE), str(VALID_JPG)]},
        )
        await component.process_images()
        expected = f"Skipped {LARGE_FILE.name}: exceeds 2mb limit"
        warning_texts = [str(c) for c in component.log.call_args_list]
        assert any(expected in w for w in warning_texts)

    async def test_all_invalid_raises_value_error(self, component_class):
        component = await self.component_setup(
            component_class,
            {"files": [str(INVALID_TXT), str(MISSING), str(LARGE_FILE)]},
        )
        with pytest.raises(ValueError, match="No valid images to process"):
            await component.process_images()

    # ------------------------------------------------------------------
    # Acceptance: resize — PIL thumbnail, max_size=1024, giữ aspect ratio
    # ------------------------------------------------------------------

    async def test_large_image_resized_dimensions_within_max_size(self, component_class):
        """ảnh 1500×1000 → resized_width ≤ 1024 và resized_height ≤ 1024."""
        component = await self.component_setup(
            component_class,
            {"files": [str(VALID_LARGE)]},
        )
        result = await component.process_images()
        item = result[0]
        assert item["resized_width"] <= MAX_SIZE
        assert item["resized_height"] <= MAX_SIZE

    async def test_large_image_longest_edge_equals_max_size(self, component_class):
        """Cạnh dài nhất sau resize phải chính xác bằng max_size (1024)."""
        component = await self.component_setup(
            component_class,
            {"files": [str(VALID_LARGE)]},
        )
        result = await component.process_images()
        item = result[0]
        assert max(item["resized_width"], item["resized_height"]) == MAX_SIZE

    async def test_resize_preserves_aspect_ratio(self, component_class):
        """1500×1000 → 1024×682 (ratio 3:2 giữ nguyên trong sai số 1px)."""
        component = await self.component_setup(
            component_class,
            {"files": [str(VALID_LARGE)]},
        )
        result = await component.process_images()
        item = result[0]
        original_ratio = item["original_width"] / item["original_height"]
        resized_ratio = item["resized_width"] / item["resized_height"]
        assert abs(original_ratio - resized_ratio) < 0.02

    async def test_small_image_is_not_upscaled(self, component_class, default_kwargs):
        """300×168 < 1024 → resized_width == original_width (thumbnail không upscale)."""
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.process_images()
        item = result[0]
        assert item["resized_width"] == item["original_width"]
        assert item["resized_height"] == item["original_height"]

    async def test_original_dimensions_recorded_before_resize(self, component_class):
        """original_width/height phải là kích thước file gốc, không phải sau resize."""
        component = await self.component_setup(
            component_class,
            {"files": [str(VALID_LARGE)]},
        )
        result = await component.process_images()
        item = result[0]
        assert item["original_width"] == LARGE_ORIGINAL_WIDTH
        assert item["original_height"] == LARGE_ORIGINAL_HEIGHT

    # ------------------------------------------------------------------
    # Acceptance: ImageData schema — tất cả fields bắt buộc phải có
    # ------------------------------------------------------------------

    async def test_imagedata_has_all_required_fields(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.process_images()
        item = result[0]
        required = {
            "index", "filename",
            "base64", "mime_type", "size_bytes",
            "original_width", "original_height",
            "resized_width", "resized_height",
        }
        assert required.issubset(item.keys())

    async def test_bbox_scaling_fields_exist_in_imagedata(self, component_class, default_kwargs):
        """original_width/height + resized_width/height phải có để Image Annotator scale bbox."""
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.process_images()
        item = result[0]
        assert "original_width" in item
        assert "original_height" in item
        assert "resized_width" in item
        assert "resized_height" in item

    async def test_dimension_fields_are_positive_integers(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.process_images()
        item = result[0]
        for field in ("original_width", "original_height", "resized_width", "resized_height"):
            assert isinstance(item[field], int), f"{field} must be int"
            assert item[field] > 0, f"{field} must be positive"

    async def test_imagedata_base64_starts_with_data_image(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.process_images()
        assert result[0]["base64"].startswith("data:image/")

    async def test_imagedata_base64_has_correct_structure(self, component_class, default_kwargs):
        """Format: data:{mime_type};base64,{data}"""
        import base64
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.process_images()
        b64_str = result[0]["base64"]
        prefix, encoded = b64_str.split(",", 1)
        assert prefix.endswith(";base64")
        decoded = base64.b64decode(encoded)
        assert len(decoded) > 0

    async def test_imagedata_mime_type_for_jpg(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.process_images()
        assert result[0]["mime_type"] == "image/jpeg"

    async def test_imagedata_mime_type_for_png(self, component_class, tmp_path):
        png_bytes = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde"
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        png_file = tmp_path / "test.png"
        png_file.write_bytes(png_bytes)
        component = await self.component_setup(component_class, {"files": [str(png_file)]})
        result = await component.process_images()
        assert result[0]["mime_type"] == "image/png"

    async def test_imagedata_size_bytes_is_original_file_size(self, component_class, default_kwargs):
        """size_bytes phải là kích thước file gốc, không phải sau resize."""
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.process_images()
        assert result[0]["size_bytes"] == VALID_JPG.stat().st_size

    async def test_imagedata_filename_is_original_basename(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.process_images()
        assert result[0]["filename"] == VALID_JPG.name

    # ------------------------------------------------------------------
    # Acceptance: output là List, giữ thứ tự, index 0-based
    # ------------------------------------------------------------------

    async def test_output_is_always_list(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.process_images()
        assert isinstance(result, list)

    async def test_output_preserves_upload_order(self, component_class):
        component = await self.component_setup(
            component_class,
            {"files": [str(VALID_JPG), str(VALID_JPEG)]},
        )
        result = await component.process_images()
        assert result[0]["filename"] == VALID_JPG.name
        assert result[1]["filename"] == VALID_JPEG.name

    async def test_output_indices_are_zero_based_sequential(self, component_class):
        component = await self.component_setup(
            component_class,
            {"files": [str(VALID_JPG), str(VALID_JPEG)]},
        )
        result = await component.process_images()
        assert [item["index"] for item in result] == [0, 1]

    async def test_skipped_file_index_reflects_original_position(self, component_class):
        """Khi file đầu bị skip, file hợp lệ thứ hai giữ index=1 (vị trí gốc trong batch)."""
        component = await self.component_setup(
            component_class,
            {"files": [str(INVALID_TXT), str(VALID_JPG)]},
        )
        result = await component.process_images()
        assert result[0]["index"] == 1

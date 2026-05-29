"""
Unit tests for ImageAnnotatorComponent.
All tests FAIL until lfx.components.vision.image_annotator is implemented — intentional.

Acceptance source: ACCEPTANCE.MD §§ "Tư duy thiết kế", "Context kỹ thuật", "Node: Image Annotator"
"""
from __future__ import annotations

import pytest

from lfx.components.vision.image_annotator import ImageAnnotatorComponent
from tests.base import ComponentTestBaseWithoutClient
from tests.unit.components.vision.fixtures.mock_vision_results import (
    MOCK_RESULT_FAILED,
    MOCK_RESULT_NO_BBOX,
    MOCK_RESULT_SUCCESS,
    MOCK_RESULT_UNKNOWN,
    SCALED_BBOX,
    SOLID_IMG_0,
    SOLID_IMG_1,
    SOLID_IMG_2,
    decode_annotated_image,
    make_solid_image_data,
    make_vision_result,
)

# cv2 BGR channel indices
_CH_B, _CH_G, _CH_R = 0, 1, 2

# Green threshold: annotation color is (0,255,0) BGR; background G-channel ≈ 0.
# Using > 200 instead of == 255 to tolerate JPEG re-encoding by the component.
_GREEN_THRESHOLD = 200


class TestImageAnnotatorComponent(ComponentTestBaseWithoutClient):

    # ------------------------------------------------------------------
    # Required base-class fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def component_class(self):
        return ImageAnnotatorComponent

    @pytest.fixture
    def default_kwargs(self):
        return {
            "images": [SOLID_IMG_0],
            "vision_results": [MOCK_RESULT_SUCCESS],
        }

    @pytest.fixture
    def file_names_mapping(self):
        return []  # new component, no version history

    # ------------------------------------------------------------------
    # Output structure — List[AnnotatedImage]
    # ------------------------------------------------------------------

    async def test_output_is_list(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.annotate_images()
        assert isinstance(result, list)

    async def test_annotatedimage_has_all_required_fields(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.annotate_images()
        item = result[0]
        for field in ("index", "filename", "annotated_base64", "findings"):
            assert field in item, f"AnnotatedImage missing field: {field}"

    async def test_annotated_base64_is_string(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.annotate_images()
        assert isinstance(result[0]["annotated_base64"], str)

    async def test_annotated_base64_starts_with_data_image(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.annotate_images()
        assert result[0]["annotated_base64"].startswith("data:image/")

    async def test_findings_matches_vision_result(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.annotate_images()
        assert result[0]["findings"] == MOCK_RESULT_SUCCESS

    # ------------------------------------------------------------------
    # Count and order preservation
    # ------------------------------------------------------------------

    async def test_single_pair_returns_one_result(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.annotate_images()
        assert len(result) == 1

    async def test_multiple_pairs_returns_correct_count(self, component_class):
        results = [
            make_vision_result(i, f"car_{i}.jpg") for i in range(3)
        ]
        component = await self.component_setup(
            component_class,
            {"images": [SOLID_IMG_0, SOLID_IMG_1, SOLID_IMG_2], "vision_results": results},
        )
        result = await component.annotate_images()
        assert len(result) == 3

    async def test_output_order_matches_input_index(self, component_class):
        r0 = make_vision_result(0, "car_0.jpg")
        r1 = make_vision_result(1, "car_1.jpg")
        component = await self.component_setup(
            component_class,
            {"images": [SOLID_IMG_0, SOLID_IMG_1], "vision_results": [r0, r1]},
        )
        result = await component.annotate_images()
        assert result[0]["index"] == 0
        assert result[1]["index"] == 1

    async def test_output_filename_matches_image_filename(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.annotate_images()
        assert result[0]["filename"] == SOLID_IMG_0["filename"]

    # ------------------------------------------------------------------
    # Output image dimensions — must equal original (not resized)
    # SOLID_IMG_0: resized 300×200, original 600×400
    # ------------------------------------------------------------------

    async def test_output_dimensions_equal_original_width_height(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.annotate_images()
        img = decode_annotated_image(result[0]["annotated_base64"])
        h, w = img.shape[:2]
        assert w == SOLID_IMG_0["original_width"], f"expected width {SOLID_IMG_0['original_width']}, got {w}"
        assert h == SOLID_IMG_0["original_height"], f"expected height {SOLID_IMG_0['original_height']}, got {h}"

    async def test_output_dimensions_respect_custom_original_size(self, component_class):
        img_data = make_solid_image_data(0, "car.jpg", resized_size=(200, 100), original_size=(800, 400))
        r = make_vision_result(0, "car.jpg", bbox={"x": 10, "y": 10, "width": 50, "height": 30})
        component = await self.component_setup(
            component_class, {"images": [img_data], "vision_results": [r]}
        )
        result = await component.annotate_images()
        img = decode_annotated_image(result[0]["annotated_base64"])
        h, w = img.shape[:2]
        assert w == 800
        assert h == 400

    # ------------------------------------------------------------------
    # Bbox drawing — pixel-level verification
    #
    # SOLID_IMG_0 background: blue (BGR 200,0,0) → G-channel ≈ 0 everywhere.
    # Green bbox annotation (BGR 0,255,0) → G-channel ≈ 255 at boundary.
    # SCALED_BBOX on 600×400 image: x=100, y=80, w=200, h=160.
    # ------------------------------------------------------------------

    async def test_green_pixels_present_at_top_edge_of_bbox(self, component_class, default_kwargs):
        """Top edge of bbox: row y=80, cols 100..300 should contain green pixels."""
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.annotate_images()
        img = decode_annotated_image(result[0]["annotated_base64"])
        # Sample G-channel along the top edge (middle section, avoid corners)
        y = SCALED_BBOX["y"]
        x_mid = SCALED_BBOX["x"] + SCALED_BBOX["width"] // 2
        assert img[y, x_mid, _CH_G] > _GREEN_THRESHOLD, (
            f"Expected green pixel at top edge ({x_mid},{y}), G={img[y, x_mid, _CH_G]}"
        )

    async def test_green_pixels_present_at_left_edge_of_bbox(self, component_class, default_kwargs):
        """Left edge of bbox: col x=100, rows 80..240 should contain green pixels."""
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.annotate_images()
        img = decode_annotated_image(result[0]["annotated_base64"])
        x = SCALED_BBOX["x"]
        y_mid = SCALED_BBOX["y"] + SCALED_BBOX["height"] // 2
        assert img[y_mid, x, _CH_G] > _GREEN_THRESHOLD, (
            f"Expected green pixel at left edge ({x},{y_mid}), G={img[y_mid, x, _CH_G]}"
        )

    async def test_background_has_no_green_before_annotation(self, component_class):
        """Sanity check: unmodified solid blue image has no green pixels (G < 10)."""
        img = decode_annotated_image(SOLID_IMG_0["base64"])
        assert img[:, :, _CH_G].max() < 10, "Background image must have no green pixels"

    # ------------------------------------------------------------------
    # bbox: null — no rectangle, only label at top-left corner
    # ------------------------------------------------------------------

    async def test_null_bbox_produces_no_green_bbox_pixels(self, component_class):
        component = await self.component_setup(
            component_class,
            {"images": [SOLID_IMG_0], "vision_results": [MOCK_RESULT_NO_BBOX]},
        )
        result = await component.annotate_images()
        img = decode_annotated_image(result[0]["annotated_base64"])
        # A rectangle at SCALED_BBOX position should NOT exist
        y = SCALED_BBOX["y"]
        x_mid = SCALED_BBOX["x"] + SCALED_BBOX["width"] // 2
        assert img[y, x_mid, _CH_G] < _GREEN_THRESHOLD, (
            "No green bbox should be drawn when bbox is null"
        )

    async def test_null_bbox_still_produces_output_image(self, component_class):
        component = await self.component_setup(
            component_class,
            {"images": [SOLID_IMG_0], "vision_results": [MOCK_RESULT_NO_BBOX]},
        )
        result = await component.annotate_images()
        assert result[0]["annotated_base64"].startswith("data:image/")

    # ------------------------------------------------------------------
    # status: "failed" — return original image unchanged, no annotation
    # ------------------------------------------------------------------

    async def test_failed_status_produces_no_green_pixels(self, component_class):
        component = await self.component_setup(
            component_class,
            {"images": [SOLID_IMG_0], "vision_results": [MOCK_RESULT_FAILED]},
        )
        result = await component.annotate_images()
        img = decode_annotated_image(result[0]["annotated_base64"])
        assert img[:, :, _CH_G].max() < _GREEN_THRESHOLD, (
            "Failed status must not draw any annotation"
        )

    async def test_failed_status_still_returns_output_image(self, component_class):
        component = await self.component_setup(
            component_class,
            {"images": [SOLID_IMG_0], "vision_results": [MOCK_RESULT_FAILED]},
        )
        result = await component.annotate_images()
        assert result[0]["annotated_base64"].startswith("data:image/")

    async def test_failed_status_findings_preserved(self, component_class):
        component = await self.component_setup(
            component_class,
            {"images": [SOLID_IMG_0], "vision_results": [MOCK_RESULT_FAILED]},
        )
        result = await component.annotate_images()
        assert result[0]["findings"]["status"] == "failed"
        assert result[0]["findings"]["error"] == "API rate limit exceeded"

    # ------------------------------------------------------------------
    # Label text — "brand model - color" or "Unknown Vehicle"
    # ------------------------------------------------------------------

    async def test_success_result_produces_annotated_image(self, component_class, default_kwargs):
        """Component runs without error and produces output — label content not pixel-verifiable."""
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.annotate_images()
        img = decode_annotated_image(result[0]["annotated_base64"])
        assert img is not None
        assert img.size > 0

    async def test_unknown_brand_label_does_not_crash(self, component_class):
        """brand=null → label 'Unknown Vehicle'; component must not raise."""
        component = await self.component_setup(
            component_class,
            {"images": [SOLID_IMG_0], "vision_results": [MOCK_RESULT_UNKNOWN]},
        )
        result = await component.annotate_images()
        assert len(result) == 1
        assert result[0]["annotated_base64"].startswith("data:image/")

    # ------------------------------------------------------------------
    # Bbox scaling — ACCEPTANCE.MD formula
    # scale_x = original_width / resized_width
    # scale_y = original_height / resized_height
    # ------------------------------------------------------------------

    async def test_bbox_green_pixels_at_scaled_position_not_unscaled(self, component_class, default_kwargs):
        """Green pixels must appear at SCALED coords (100,80), NOT unscaled (50,40)."""
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.annotate_images()
        img = decode_annotated_image(result[0]["annotated_base64"])

        # Scaled position (correct): green expected
        y_scaled = SCALED_BBOX["y"]
        x_scaled = SCALED_BBOX["x"] + SCALED_BBOX["width"] // 2
        assert img[y_scaled, x_scaled, _CH_G] > _GREEN_THRESHOLD, (
            f"Green pixel expected at scaled position ({x_scaled},{y_scaled})"
        )

        # Unscaled position (wrong): should NOT have green from the bbox top edge
        y_raw = MOCK_RESULT_SUCCESS["bbox"]["y"]           # 40
        x_raw = MOCK_RESULT_SUCCESS["bbox"]["x"] + MOCK_RESULT_SUCCESS["bbox"]["width"] // 2  # 100
        # Only assert if unscaled coords are different from scaled ones
        if y_raw != y_scaled:
            assert img[y_raw, x_raw, _CH_G] < _GREEN_THRESHOLD, (
                f"Green pixel should NOT appear at unscaled position ({x_raw},{y_raw})"
            )

    async def test_scale_1_unchanged_no_upscale_needed(self, component_class):
        """When resized == original (scale=1), bbox coords must not be changed."""
        img_data = make_solid_image_data(0, "car.jpg", resized_size=(300, 200), original_size=(300, 200))
        bbox = {"x": 50, "y": 40, "width": 100, "height": 80}
        r = make_vision_result(0, "car.jpg", bbox=bbox)
        component = await self.component_setup(
            component_class, {"images": [img_data], "vision_results": [r]}
        )
        result = await component.annotate_images()
        img = decode_annotated_image(result[0]["annotated_base64"])
        # bbox top edge at y=40 (no scaling)
        x_mid = bbox["x"] + bbox["width"] // 2  # 100
        assert img[40, x_mid, _CH_G] > _GREEN_THRESHOLD

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    async def test_count_mismatch_raises_value_error(self, component_class):
        """len(images) != len(vision_results) → ValueError."""
        component = await self.component_setup(
            component_class,
            {"images": [SOLID_IMG_0, SOLID_IMG_1], "vision_results": [MOCK_RESULT_SUCCESS]},
        )
        with pytest.raises(ValueError):
            await component.annotate_images()

    async def test_empty_inputs_return_empty_list(self, component_class):
        component = await self.component_setup(
            component_class, {"images": [], "vision_results": []}
        )
        result = await component.annotate_images()
        assert result == []

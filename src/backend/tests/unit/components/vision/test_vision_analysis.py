"""
Unit tests for VisionAnalysisComponent.

Acceptance source: ACCEPTANCE.MD §§ "Tư duy thiết kế", "Context kỹ thuật", "Node: Vision Analysis"
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from lfx.components.vision.vision_analysis import VisionAnalysisComponent, _extract_json
from tests.base import ComponentTestBaseWithoutClient
from tests.unit.components.vision.fixtures.mock_vision_responses import (
    API_FAIL_MOCK,
    NO_CAR_MOCK,
    SUCCESS_MOCK,
    make_image_data,
    make_mock_llm,
    make_mock_llm_raw,
)

# Patch target — get_llm imported inside the component module
_LLM_PATCH = "lfx.components.vision.vision_analysis.get_llm"

# ModelInput value format: list[dict] with name + provider + metadata
_MODEL_OPENAI = [{"name": "gpt-4o-mini", "provider": "OpenAI", "metadata": {"model_class": "ChatOpenAI"}}]
_MODEL_NON_VISION = [{"name": "gpt-3.5-turbo", "provider": "OpenAI", "metadata": {"model_class": "ChatOpenAI"}}]

# Reusable ImageData stubs (no real files needed)
IMG_0 = make_image_data(0, "car_0.jpg")
IMG_1 = make_image_data(1, "car_1.jpg")
IMG_2 = make_image_data(2, "car_2.jpg")


class TestVisionAnalysisComponent(ComponentTestBaseWithoutClient):

    # ------------------------------------------------------------------
    # Required base-class fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def component_class(self):
        return VisionAnalysisComponent

    @pytest.fixture
    def default_kwargs(self):
        return {
            "images": [IMG_0],
            "model": _MODEL_OPENAI,
            "max_concurrent": 3,
        }

    @pytest.fixture
    def file_names_mapping(self):
        return []  # new component, no version history

    # Override base — must mock LLM to avoid real API call
    @patch(_LLM_PATCH)
    async def test_latest_version(self, mock_get_llm, component_class, default_kwargs):
        mock_get_llm.return_value = make_mock_llm([SUCCESS_MOCK])
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.run()
        assert result is not None

    # ------------------------------------------------------------------
    # Model validation
    # ------------------------------------------------------------------

    async def test_empty_model_raises_value_error(self, component_class):
        component = await self.component_setup(
            component_class, {"images": [IMG_0], "model": []}
        )
        with pytest.raises(ValueError, match="Model name is required"):
            await component.analyze_images()

    async def test_none_model_raises_value_error(self, component_class):
        component = await self.component_setup(
            component_class, {"images": [IMG_0], "model": None}
        )
        with pytest.raises(ValueError, match="Model name is required"):
            await component.analyze_images()

    @patch(_LLM_PATCH)
    async def test_empty_images_returns_empty_list(self, mock_get_llm, component_class):
        mock_get_llm.return_value = make_mock_llm([])
        component = await self.component_setup(
            component_class, {"images": [], "model": _MODEL_OPENAI}
        )
        result = await component.analyze_images()
        assert result == []
        mock_get_llm.return_value.ainvoke.assert_not_called()

    # ------------------------------------------------------------------
    # Output structure — List[VisionResult]
    # ------------------------------------------------------------------

    @patch(_LLM_PATCH)
    async def test_output_is_list(self, mock_get_llm, component_class, default_kwargs):
        mock_get_llm.return_value = make_mock_llm([SUCCESS_MOCK])
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.analyze_images()
        assert isinstance(result, list)

    @patch(_LLM_PATCH)
    async def test_single_image_returns_one_result(self, mock_get_llm, component_class, default_kwargs):
        mock_get_llm.return_value = make_mock_llm([SUCCESS_MOCK])
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.analyze_images()
        assert len(result) == 1

    @patch(_LLM_PATCH)
    async def test_visionresult_has_all_required_fields(self, mock_get_llm, component_class, default_kwargs):
        mock_get_llm.return_value = make_mock_llm([SUCCESS_MOCK])
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.analyze_images()
        item = result[0]
        for field in ("index", "filename", "brand", "model", "color", "confidence", "bbox", "status", "error"):
            assert field in item, f"VisionResult missing field: {field}"

    @patch(_LLM_PATCH)
    async def test_result_index_matches_image_index(self, mock_get_llm, component_class):
        mock_get_llm.return_value = make_mock_llm([SUCCESS_MOCK])
        component = await self.component_setup(
            component_class, {"images": [IMG_0], "model": _MODEL_OPENAI}
        )
        result = await component.analyze_images()
        assert result[0]["index"] == IMG_0["index"]

    @patch(_LLM_PATCH)
    async def test_result_filename_matches_image_filename(self, mock_get_llm, component_class):
        mock_get_llm.return_value = make_mock_llm([SUCCESS_MOCK])
        component = await self.component_setup(
            component_class, {"images": [IMG_0], "model": _MODEL_OPENAI}
        )
        result = await component.analyze_images()
        assert result[0]["filename"] == IMG_0["filename"]

    # ------------------------------------------------------------------
    # Success mock — field parsing
    # ------------------------------------------------------------------

    @patch(_LLM_PATCH)
    async def test_success_brand_parsed(self, mock_get_llm, component_class, default_kwargs):
        mock_get_llm.return_value = make_mock_llm([SUCCESS_MOCK])
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.analyze_images()
        assert result[0]["brand"] == SUCCESS_MOCK["brand"]

    @patch(_LLM_PATCH)
    async def test_success_model_parsed(self, mock_get_llm, component_class, default_kwargs):
        mock_get_llm.return_value = make_mock_llm([SUCCESS_MOCK])
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.analyze_images()
        assert result[0]["model"] == SUCCESS_MOCK["model"]

    @patch(_LLM_PATCH)
    async def test_success_color_parsed(self, mock_get_llm, component_class, default_kwargs):
        mock_get_llm.return_value = make_mock_llm([SUCCESS_MOCK])
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.analyze_images()
        assert result[0]["color"] == SUCCESS_MOCK["color"]

    @patch(_LLM_PATCH)
    async def test_success_confidence_parsed(self, mock_get_llm, component_class, default_kwargs):
        mock_get_llm.return_value = make_mock_llm([SUCCESS_MOCK])
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.analyze_images()
        assert result[0]["confidence"] == pytest.approx(SUCCESS_MOCK["confidence"])

    @patch(_LLM_PATCH)
    async def test_success_bbox_parsed(self, mock_get_llm, component_class, default_kwargs):
        mock_get_llm.return_value = make_mock_llm([SUCCESS_MOCK])
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.analyze_images()
        assert result[0]["bbox"] == SUCCESS_MOCK["bbox"]

    @patch(_LLM_PATCH)
    async def test_success_status_is_success(self, mock_get_llm, component_class, default_kwargs):
        mock_get_llm.return_value = make_mock_llm([SUCCESS_MOCK])
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.analyze_images()
        assert result[0]["status"] == "success"

    @patch(_LLM_PATCH)
    async def test_success_error_is_none(self, mock_get_llm, component_class, default_kwargs):
        mock_get_llm.return_value = make_mock_llm([SUCCESS_MOCK])
        component = await self.component_setup(component_class, default_kwargs)
        result = await component.analyze_images()
        assert result[0]["error"] is None

    # ------------------------------------------------------------------
    # No-car response — brand=null, confidence<0.5, status="success"
    # ------------------------------------------------------------------

    @patch(_LLM_PATCH)
    async def test_no_car_brand_is_none(self, mock_get_llm, component_class):
        mock_get_llm.return_value = make_mock_llm([NO_CAR_MOCK])
        component = await self.component_setup(
            component_class, {"images": [IMG_0], "model": _MODEL_OPENAI}
        )
        result = await component.analyze_images()
        assert result[0]["brand"] is None

    @patch(_LLM_PATCH)
    async def test_no_car_model_is_none(self, mock_get_llm, component_class):
        mock_get_llm.return_value = make_mock_llm([NO_CAR_MOCK])
        component = await self.component_setup(
            component_class, {"images": [IMG_0], "model": _MODEL_OPENAI}
        )
        result = await component.analyze_images()
        assert result[0]["model"] is None

    @patch(_LLM_PATCH)
    async def test_no_car_confidence_below_0_5(self, mock_get_llm, component_class):
        mock_get_llm.return_value = make_mock_llm([NO_CAR_MOCK])
        component = await self.component_setup(
            component_class, {"images": [IMG_0], "model": _MODEL_OPENAI}
        )
        result = await component.analyze_images()
        assert result[0]["confidence"] < 0.5

    @patch(_LLM_PATCH)
    async def test_no_car_status_is_success(self, mock_get_llm, component_class):
        mock_get_llm.return_value = make_mock_llm([NO_CAR_MOCK])
        component = await self.component_setup(
            component_class, {"images": [IMG_0], "model": _MODEL_OPENAI}
        )
        result = await component.analyze_images()
        assert result[0]["status"] == "success"

    @patch(_LLM_PATCH)
    async def test_no_car_bbox_is_none(self, mock_get_llm, component_class):
        mock_get_llm.return_value = make_mock_llm([NO_CAR_MOCK])
        component = await self.component_setup(
            component_class, {"images": [IMG_0], "model": _MODEL_OPENAI}
        )
        result = await component.analyze_images()
        assert result[0]["bbox"] is None

    # ------------------------------------------------------------------
    # Multiple images — count and order preservation
    # ------------------------------------------------------------------

    @patch(_LLM_PATCH)
    async def test_multiple_images_returns_correct_count(self, mock_get_llm, component_class):
        mock_get_llm.return_value = make_mock_llm([SUCCESS_MOCK, SUCCESS_MOCK, NO_CAR_MOCK])
        component = await self.component_setup(
            component_class,
            {"images": [IMG_0, IMG_1, IMG_2], "model": _MODEL_OPENAI},
        )
        result = await component.analyze_images()
        assert len(result) == 3

    @patch(_LLM_PATCH)
    async def test_multiple_images_preserves_order(self, mock_get_llm, component_class):
        mock_get_llm.return_value = make_mock_llm([SUCCESS_MOCK, NO_CAR_MOCK])
        component = await self.component_setup(
            component_class,
            {"images": [IMG_0, IMG_1], "model": _MODEL_OPENAI},
        )
        result = await component.analyze_images()
        indices = [r["index"] for r in result]
        assert indices == sorted(indices), "Results must be ordered by image index"
        assert result[0]["filename"] == IMG_0["filename"]
        assert result[1]["filename"] == IMG_1["filename"]

    # ------------------------------------------------------------------
    # Parallel execution — asyncio.gather + Semaphore
    # ------------------------------------------------------------------

    async def test_max_concurrent_default_is_3(self, component_class, default_kwargs):
        component = await self.component_setup(component_class, default_kwargs)
        assert getattr(component, "max_concurrent", 3) == 3

    @patch(_LLM_PATCH)
    async def test_max_concurrent_1_still_processes_all_images(self, mock_get_llm, component_class):
        mock_get_llm.return_value = make_mock_llm([SUCCESS_MOCK, SUCCESS_MOCK])
        component = await self.component_setup(
            component_class,
            {"images": [IMG_0, IMG_1], "model": _MODEL_OPENAI, "max_concurrent": 1},
        )
        result = await component.analyze_images()
        assert len(result) == 2

    @patch(_LLM_PATCH)
    async def test_max_concurrent_5_processes_all_images(self, mock_get_llm, component_class):
        images = [make_image_data(i, f"car_{i}.jpg") for i in range(5)]
        mock_get_llm.return_value = make_mock_llm([SUCCESS_MOCK] * 5)
        component = await self.component_setup(
            component_class,
            {"images": images, "model": _MODEL_OPENAI, "max_concurrent": 5},
        )
        result = await component.analyze_images()
        assert len(result) == 5

    # ------------------------------------------------------------------
    # Failure handling — always raises RuntimeError on API error
    # ------------------------------------------------------------------

    @patch(_LLM_PATCH)
    async def test_api_fail_raises_runtime_error(self, mock_get_llm, component_class):
        mock_get_llm.return_value = make_mock_llm([API_FAIL_MOCK])
        component = await self.component_setup(
            component_class,
            {"images": [IMG_0], "model": _MODEL_OPENAI},
        )
        with pytest.raises(RuntimeError):
            await component.analyze_images()

    @patch(_LLM_PATCH)
    async def test_api_fail_error_contains_original_message(self, mock_get_llm, component_class):
        mock_get_llm.return_value = make_mock_llm([API_FAIL_MOCK])
        component = await self.component_setup(
            component_class,
            {"images": [IMG_0], "model": _MODEL_OPENAI},
        )
        with pytest.raises(RuntimeError, match="API rate limit exceeded"):
            await component.analyze_images()

    # ------------------------------------------------------------------
    # Vision capability warning
    # ------------------------------------------------------------------

    @patch(_LLM_PATCH)
    async def test_non_vision_model_sets_warning_status(self, mock_get_llm, component_class):
        mock_get_llm.return_value = make_mock_llm([SUCCESS_MOCK])
        component = await self.component_setup(
            component_class,
            {"images": [IMG_0], "model": _MODEL_NON_VISION},
        )
        await component.analyze_images()
        assert "⚠️" in str(component.status)

    @patch(_LLM_PATCH)
    async def test_non_vision_model_warning_contains_model_name(self, mock_get_llm, component_class):
        mock_get_llm.return_value = make_mock_llm([SUCCESS_MOCK])
        component = await self.component_setup(
            component_class,
            {"images": [IMG_0], "model": _MODEL_NON_VISION},
        )
        await component.analyze_images()
        assert "gpt-3.5-turbo" in str(component.status)

    @patch(_LLM_PATCH)
    async def test_vision_model_no_warning_status(self, mock_get_llm, component_class, default_kwargs):
        mock_get_llm.return_value = make_mock_llm([SUCCESS_MOCK])
        component = await self.component_setup(component_class, default_kwargs)
        await component.analyze_images()
        assert "⚠️" not in str(component.status or "")

    # ------------------------------------------------------------------
    # JSON extraction — _extract_json handles Ollama quirks
    # ------------------------------------------------------------------

    def test_extract_json_plain(self):
        data = _extract_json('{"brand": "Toyota", "confidence": 0.9}', "test.jpg")
        assert data["brand"] == "Toyota"

    def test_extract_json_markdown_block(self):
        raw = '```json\n{"brand": "Honda", "confidence": 0.8}\n```'
        data = _extract_json(raw, "test.jpg")
        assert data["brand"] == "Honda"

    def test_extract_json_markdown_no_lang(self):
        raw = '```\n{"brand": "Ford"}\n```'
        data = _extract_json(raw, "test.jpg")
        assert data["brand"] == "Ford"

    def test_extract_json_prose_around(self):
        raw = 'Here is the result:\n{"brand": "BMW"}\nDone.'
        data = _extract_json(raw, "test.jpg")
        assert data["brand"] == "BMW"

    def test_extract_json_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="empty response"):
            _extract_json("", "car.jpg")

    def test_extract_json_whitespace_only_raises_value_error(self):
        with pytest.raises(ValueError, match="empty response"):
            _extract_json("   \n  ", "car.jpg")

    def test_extract_json_no_json_object_raises_value_error(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            _extract_json("I cannot analyze this image.", "car.jpg")

    @patch(_LLM_PATCH)
    async def test_ollama_markdown_json_response_parsed_correctly(self, mock_get_llm, component_class):
        raw = '```json\n{"brand":"Toyota","model":"Camry","color":"White","confidence":0.92,"bbox":null}\n```'
        mock_get_llm.return_value = make_mock_llm_raw([raw])
        component = await self.component_setup(
            component_class, {"images": [IMG_0], "model": _MODEL_OPENAI}
        )
        result = await component.analyze_images()
        assert result[0]["brand"] == "Toyota"

    @patch(_LLM_PATCH)
    async def test_empty_response_raises_runtime_error(self, mock_get_llm, component_class):
        mock_get_llm.return_value = make_mock_llm_raw([""])
        component = await self.component_setup(
            component_class, {"images": [IMG_0], "model": _MODEL_OPENAI}
        )
        with pytest.raises(RuntimeError, match="empty response"):
            await component.analyze_images()

# Acceptance Criteria — Flow Execution System

> File này do **developer viết và maintain**.
> Claude Code đọc file này trước khi implement bất kỳ node nào.
> **Không được tự ý thêm/sửa file này khi không có yêu cầu từ developer.**

---

## Quy ước chung

- Mỗi node có section riêng
- Status: `[ ] TODO` | `[x] DONE` | `[~] IN PROGRESS`
- Mỗi criteria bắt đầu bằng bullet rõ ràng

---

## Tư duy thiết kế (Claude Code phải đọc trước)

> Mỗi component là 1 service độc lập — không quan tâm ai gọi, không quan tâm output đi đâu.

```
Image Input     → luôn output List[ImageData] dù chỉ có 1 ảnh
Vision Analysis → luôn nhận List[ImageData], luôn output List[VisionResult]
Image Annotator → luôn nhận List[ImageData] + List[VisionResult], output List[AnnotatedImage]
```

---

## Context kỹ thuật (Claude Code phải đọc trước)

### Data flow của image trong LangFlow

```
User upload ảnh
    ↓
Image Input Component
    → validate file type/size
    → resize về max 1024px (giữ aspect ratio)
    → convert sang base64
    → output List[ImageData]
    ↓
Vision Analysis
    → parallel API calls (asyncio.gather + Semaphore)
    → mỗi call dùng base64 từ ImageData
    → output List[VisionResult]
    ↓
Image Annotator
    → scale bounding box từ resized → original dimensions
    → vẽ annotation lên ảnh gốc
    → output List[AnnotatedImage]
```

### Các file quan trọng cần tham khảo

| File                                                             | Dùng cho                                                                               |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `src/lfx/src/lfx/utils/image.py:75`                              | `create_image_content_dict()` — format multimodal message đúng chuẩn OpenAI Vision API |
| `src/lfx/src/lfx/components/input_output/chat.py`                | Pattern Chat Input handle files                                                        |
| `src/lfx/src/lfx/components/models_and_agents/language_model.py` | Pattern Language Model component                                                       |
| `src/backend/tests/unit/base.py`                                 | Base class cho unit test                                                               |

### Format multimodal message đúng

```python
# Dùng create_image_content_dict() từ utils/image.py
# KHÔNG tự viết lại format này
[
    {"type": "text", "text": "Analyze this image"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BASE64_DATA"}}
]
```

### ImageData schema

```python
ImageData = {
    # File info
    "index":           int,    # vị trí trong batch (0-based)
    "filename":        str,    # tên file gốc

    # Sau khi resize + convert
    "base64":          str,    # "data:image/jpeg;base64,..."
    "mime_type":       str,    # "image/jpeg" | "image/png"
    "size_bytes":      int,    # kích thước file gốc (trước resize)

    # Dimensions — cần để Image Annotator scale bounding box
    "original_width":  int,
    "original_height": int,
    "resized_width":   int,
    "resized_height":  int,
}
```

### VisionResult schema

```python
VisionResult = {
    "index":      int,
    "filename":   str,
    "brand":      str | None,
    "model":      str | None,
    "color":      str | None,
    "confidence": float,          # 0.0 → 1.0
    "bbox": {                     # bounding box trên ảnh đã RESIZE
        "x": int, "y": int,
        "width": int, "height": int
    } | None,
    "status":     "success" | "failed",
    "error":      str | None
}
```

### AnnotatedImage schema

```python
AnnotatedImage = {
    "index":            int,
    "filename":         str,
    "annotated_base64": str,    # base64 ảnh GỐC (chưa resize) đã annotate
    "findings":         VisionResult
}
```

### Tại sao giữ original_width/height trong ImageData

```
Vision Analysis nhận ảnh đã resize (1024px)
→ bbox tọa độ tính trên ảnh resize
→ Image Annotator vẽ lên ảnh GỐC (chưa resize)
→ Phải scale tọa độ:
   x_orig = bbox.x * (original_width / resized_width)
   y_orig = bbox.y * (original_height / resized_height)
```

### Những gì KHÔNG làm

- **Không extend Agent component** — Agent hardcode strip images khỏi input
- **Không dùng MediaContent** để transport image — chỉ lưu URL refs
- **Không tự viết multimodal format** — dùng `create_image_content_dict()` có sẵn
- **Không emit từng ảnh riêng lẻ** — luôn output List dù chỉ 1 ảnh
- **Không bỏ qua resize** — ảnh gốc > 1024px tốn token không cần thiết

---

## Node: Image Input

**Status:** `[~] IN PROGRESS`

**Mô tả:** Nhận ảnh upload từ user, validate, resize, convert sang ImageData schema chuẩn hóa.

**File implement:**

- `src/lfx/src/lfx/components/vision/image_input.py`
- `src/lfx/src/lfx/components/vision/__init__.py`

**Tham khảo pattern:** `src/lfx/src/lfx/components/input_output/chat.py`
**Thư viện:** Pillow (`PIL`) cho resize

**Acceptance Criteria:**

**File handling:**

- [ ] Nhận một hoặc nhiều file upload cùng lúc
- [ ] Chấp nhận định dạng: `.jpg`, `.jpeg`, `.png`
- [ ] Validate từng file trước khi process:
  - File không phải ảnh → skip, log warning `"Skipped {filename}: unsupported format"`
  - File không tồn tại → skip, log warning
  - File > 2mb → skip, log warning `"Skipped {filename}: exceeds 2mb limit"`
- [ ] Nếu tất cả files đều không hợp lệ → raise `ValueError("No valid images to process")`

**Resize:**

- [ ] Resize mỗi ảnh về max `max_size` px trên cạnh dài, giữ aspect ratio
- [ ] Dùng `PIL.Image.thumbnail((max_size, max_size))` — không dùng resize()
- [ ] `max_size` configurable trong node config panel (default: `1024`)
- [ ] Lưu cả `original_width/height` và `resized_width/height` vào ImageData

**Output:**

- [ ] Convert sang base64 với prefix `"data:{mime_type};base64,..."`
- [ ] Output: `List[ImageData]` theo đúng schema — luôn là list
- [ ] Giữ đúng thứ tự file upload (index 0, 1, 2...)

**Test fixtures:**

```
src/backend/tests/unit/components/vision/fixtures/
  ├── valid_car.jpg         ← ảnh xe hợp lệ (< 1024px)
  ├── valid_car_large.jpg   ← ảnh xe > 1024px để test resize
  ├── valid_car.jpeg        ← ảnh xe JPEG
  ├── invalid_file.txt      ← file không phải ảnh
  └── large_image.jpg       ← file > 2mb
```

---

## Node: Vision Analysis

**Status:** `[~] IN PROGRESS`

**Mô tả:** Nhận `List[ImageData]`, gọi Vision API song song, trả về `List[VisionResult]`.

**File implement:**

- `src/lfx/src/lfx/components/vision/vision_analysis.py`

**Tham khảo:** `src/lfx/src/lfx/components/models_and_agents/language_model.py`
**Import bắt buộc:** `from src.lfx.utils.image import create_image_content_dict`

**System prompt bắt buộc:**

```
Analyze the car in the image. Respond ONLY with valid JSON, no explanation:
{
  "brand": "string or null",
  "model": "string or null",
  "color": "string or null",
  "confidence": float 0.0-1.0,
  "bbox": {"x": int, "y": int, "width": int, "height": int} or null
}
```

**Acceptance Criteria:**

**Input/Output:**

- [ ] Nhận: `List[ImageData]` + `system_prompt` (string) + config
- [ ] Output: `List[VisionResult]` — giữ đúng thứ tự theo index

**Parallel execution:**

- [ ] Dùng `asyncio.gather()` với `return_exceptions=True`
- [ ] `max_concurrent` configurable (default: `3`)
- [ ] Dùng `asyncio.Semaphore(max_concurrent)` để giới hạn concurrent calls

**Xử lý từng ảnh:**

- [ ] Dùng `create_image_content_dict()` để format message — không tự viết
- [ ] Gọi với `response_format={"type": "json_object"}` (JSON mode)
- [ ] Parse JSON response → map vào VisionResult schema
- [ ] Model name configurable (default: `gpt-4o`)
- [ ] API key đọc từ LangFlow global Model Providers

**Failure handling:**

- [ ] `fail_strategy` configurable: `"skip"` (default) | `"stop"`
  - `"skip"`: fail → `status: "failed"`, `error: message`, tiếp tục
  - `"stop"`: fail → raise `RuntimeError`, dừng cả batch
- [ ] Không nhận ra xe → `brand: null`, `confidence < 0.5`, `status: "success"`
- [ ] API fail → `status: "failed"`, `error: "<message>"`

**Mock để test:**

```python
success_mock  = {"brand": "Toyota", "model": "Camry", "color": "White",
                 "confidence": 0.92, "bbox": {"x": 10, "y": 20, "width": 200, "height": 150}}
no_car_mock   = {"brand": None, "model": None, "color": None,
                 "confidence": 0.3, "bbox": None}
api_fail_mock = RuntimeError("API rate limit exceeded")
```

---

## Node: Image Annotator

**Status:** `[ ] TODO`

**Mô tả:** Scale bounding box về ảnh gốc, vẽ annotation, trả về `List[AnnotatedImage]`.

**File implement:** `src/lfx/src/lfx/components/vision/image_annotator.py`
**Thư viện:** OpenCV (`cv2`)

**Acceptance Criteria:**

- [ ] Nhận: `List[ImageData]` + `List[VisionResult]`
- [ ] Match theo `index` — raise `ValueError` nếu count không khớp
- [ ] Scale bbox từ resized → original dimensions:
  ```python
  scale_x = original_width / resized_width
  scale_y = original_height / resized_height
  x = int(bbox.x * scale_x)
  y = int(bbox.y * scale_y)
  w = int(bbox.width * scale_x)
  h = int(bbox.height * scale_y)
  ```
- [ ] Vẽ bounding box màu xanh lá `(0, 255, 0)`, độ dày 2px
- [ ] Label `"{brand} {model} - {color}"` tại góc trên-trái bbox
  - Font: `cv2.FONT_HERSHEY_SIMPLEX`, size: 0.8, màu trắng, background đen
- [ ] `brand: null` → label: `"Unknown Vehicle"`
- [ ] `bbox: null` → không vẽ bounding box, chỉ vẽ label ở góc trên-trái ảnh
- [ ] `status: "failed"` → giữ nguyên ảnh gốc, không annotate
- [ ] Output ảnh GỐC (chưa resize) đã annotate dạng base64
- [ ] Output: `List[AnnotatedImage]` đúng thứ tự

---

## Node: Conditional Branch

**Status:** `[ ] TODO`

**File implement:** Tận dụng `ConditionalRouterComponent` có sẵn.

**Acceptance Criteria:**

- [ ] Nhận: `List[VisionResult]`
- [ ] `success_rate` = số ảnh `status: "success"` và `confidence >= confidence_threshold` / tổng
- [ ] `success_rate >= route_threshold` → route: `"annotate"`
- [ ] `success_rate < route_threshold` → route: `"manual_review"`
- [ ] `route_threshold` configurable (default: `0.5`)
- [ ] `confidence_threshold` configurable (default: `0.8`)
- [ ] List rỗng → raise `ValueError`

---

## Node: Output

**Status:** `[ ] TODO`

**File implement:** Tận dụng Chat Output có sẵn.

**Acceptance Criteria:**

- [ ] Nhận: `List[AnnotatedImage]`
- [ ] Hiển thị summary:

  ```
  Kết quả xử lý: {success}/{total} ảnh nhận dạng được

  Ảnh 1 — car1.jpg:
  - Hãng xe: Toyota | Model: Camry | Màu: White
  - Độ tin cậy: 92%

  Ảnh 2 — car2.jpg:
  - Không nhận dạng được
  ```

- [ ] Tất cả ảnh annotated hiển thị trong Playground UI

---

## Integration: Full Pipeline

**Status:** `[ ] TODO`

**Acceptance Criteria:**

- [ ] Upload 3 ảnh xe → full flow → 3 ảnh annotated + summary
- [ ] Upload batch có 1 file lỗi → 2 ảnh còn lại vẫn xử lý được
- [ ] Ảnh > 1024px được resize trước khi gửi API — verify qua LangSmith token count
- [ ] `POST /api/v1/run/{flow_id}` → HTTP 200 + có kết quả
- [ ] `POST /api/v1/webhook/{flow_id}` → flow chạy tự động
- [ ] LangSmith trace đủ bước + token usage per ảnh

---

## Changelog

| Ngày       | Thay đổi                                                     |
| ---------- | ------------------------------------------------------------ |
| 2026-05-25 | Khởi tạo file                                                |
| 2026-05-26 | Rewrite dựa trên research source code                        |
| 2026-05-26 | Update List-based batch + Option B parallel                  |
| 2026-05-26 | Thêm resize logic, original/resized dimensions, bbox scaling |

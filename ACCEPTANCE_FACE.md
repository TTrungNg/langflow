# Acceptance Criteria — SCRFD Face Detection Pipeline

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

> Mỗi component là 1 service độc lập. Generic components có thể tái sử dụng cho bất kỳ detection model nào.

```
Image Input         → List[ImageData]           (tái sử dụng từ car pipeline)
Letterbox           → List[ProcessedImage]       (generic — configurable target_size)
Normalize           → List[NormalizedImage]      (generic — configurable preset/mean/std)
To Tensor           → List[TensorData]           (generic — configurable format)
SCRFD Inference     → List[RawDetection]         (SCRFD-specific)
Score Threshold     → List[FilteredDetection]    (generic — configurable threshold)
NMS                 → List[FaceResult]           (generic — configurable algorithm)
Face Annotator      → List[AnnotatedImage]       (extend Image Annotator)
Image Output        → UI render                  (tái sử dụng từ car pipeline)
```

**Phân loại:**
```
Generic (dùng lại cho mọi model):  Letterbox, Normalize, To Tensor,
                                   Score Threshold, NMS
SCRFD-specific:                    SCRFD Inference (inference + decode anchor gộp)
Extend từ car pipeline:            Face Annotator (extend Image Annotator)
Tái sử dụng hoàn toàn:             Image Input, Image Output
```

---

## Context kỹ thuật (Claude Code phải đọc trước)

### Full data flow

```
User upload bulk ảnh
        ↓
Image Input → List[ImageData]
        ↓
Letterbox → List[ProcessedImage]  (640×640, lưu scale/pad)
        ↓
Normalize → List[NormalizedImage] (mean=0, std=1, /255, BGR→RGB)
        ↓
To Tensor → List[TensorData]      (NCHW 1×3×640×640, float32)
        ↓
SCRFD Inference → List[RawDetection]  (ONNX local, decode anchor)
        ↓
Score Threshold → List[FilteredDetection]  (loại score < 0.5)
        ↓
NMS → List[FaceResult]            (loại bbox chồng, IoU > 0.4)
        ↓
        ├──────────────────────────┐
        ↓                         ↓
Face Annotator             List[FaceResult]
        ↓                  (raw data, dùng sau)
Image Output
```

### Các file quan trọng cần tham khảo

| File | Dùng cho |
|---|---|
| `src/lfx/src/lfx/components/vision/image_input.py` | Tái sử dụng hoàn toàn |
| `src/lfx/src/lfx/components/vision/image_annotator.py` | Base class để extend Face Annotator |
| `src/lfx/src/lfx/components/vision/vision_output.py` | Tái sử dụng hoàn toàn |
| `src/backend/tests/unit/components/vision/` | Pattern test có sẵn |

### ImageData schema (tái sử dụng từ car pipeline)

```python
ImageData = {
    "index":           int,
    "filename":        str,
    "base64":          str,    # "data:image/jpeg;base64,..."
    "mime_type":       str,
    "size_bytes":      int,
    "original_width":  int,
    "original_height": int,
    "resized_width":   int,
    "resized_height":  int,
}
```

### ProcessedImage schema

```python
ProcessedImage = {
    "index":           int,
    "filename":        str,
    "base64_resized":  str,    # ảnh sau letterbox, chưa normalize
    "target_size":     int,    # 640
    "scale":           float,  # tỉ lệ resize — bắt buộc để decode bbox
    "pad_left":        int,    # bắt buộc để decode bbox
    "pad_top":         int,    # bắt buộc để decode bbox
    "original_width":  int,
    "original_height": int,
}
```

### NormalizedImage schema

```python
NormalizedImage = {
    "index":              int,
    "filename":           str,
    "base64_normalized":  str,
    "target_size":        int,
    "scale":              float,   # pass-through
    "pad_left":           int,     # pass-through
    "pad_top":            int,     # pass-through
    "original_width":     int,
    "original_height":    int,
    "mean":               list,    # lưu lại để debug
    "std":                float | list,
}
```

### TensorData schema

```python
TensorData = {
    "index":          int,
    "filename":       str,
    "tensor_b64":     str,    # numpy array serialized → base64
    "shape":          list,   # [1, 3, 640, 640]
    "dtype":          str,    # "float32"
    "format":         str,    # "NCHW"
    "scale":          float,  # pass-through
    "pad_left":       int,    # pass-through
    "pad_top":        int,    # pass-through
    "original_width": int,
    "original_height":int,
}
```

### RawDetection schema

```python
RawDetection = {
    "index":      int,
    "filename":   str,
    "boxes":      list,   # [[x1,y1,x2,y2], ...] trên ảnh 640×640
    "scores":     list,   # [0.92, 0.87, ...]
    "landmarks":  list | None,  # [[x1,y1,...,x5,y5], ...] nếu KPS
    "scale":      float,  # pass-through
    "pad_left":   int,    # pass-through
    "pad_top":    int,    # pass-through
    "original_width":  int,
    "original_height": int,
}
```

### FilteredDetection schema

```python
FilteredDetection = {
    "index":      int,
    "filename":   str,
    "boxes":      list,   # sau khi filter score < threshold
    "scores":     list,
    "landmarks":  list | None,
    "scale":      float,
    "pad_left":   int,
    "pad_top":    int,
    "original_width":  int,
    "original_height": int,
}
```

### FaceResult schema

```python
FaceResult = {
    "index":      int,
    "filename":   str,
    "face_count": int,
    "faces": [
        {
            "x1": int, "y1": int,   # tọa độ trên ảnh GỐC (đã scale)
            "x2": int, "y2": int,
            "score": float,
            "landmarks": [          # 5 điểm trên ảnh GỐC
                {"x": int, "y": int}
            ] | None
        }
    ],
    "status": "success" | "failed",
    "error":  str | None
}
```

### Công thức decode bbox về ảnh gốc

```python
# Áp dụng trong SCRFD Inference sau khi decode anchor
x1_orig = (x1_640 - pad_left) / scale
y1_orig = (y1_640 - pad_top)  / scale
x2_orig = (x2_640 - pad_left) / scale
y2_orig = (y2_640 - pad_top)  / scale

# Tương tự cho landmarks
lm_x_orig = (lm_x_640 - pad_left) / scale
lm_y_orig = (lm_y_640 - pad_top)  / scale
```

### Những gì KHÔNG làm

- **Không hardcode target_size=640** trong Letterbox — phải configurable
- **Không hardcode mean/std** trong Normalize — phải dùng preset
- **Không để NCHW conversion ra ngoài** To Tensor node
- **Không decode anchor bên ngoài** SCRFD Inference node
- **Không build Face Annotator từ đầu** — extend Image Annotator

---

## Node: Letterbox Preprocess

**Status:** `[ ] TODO`

**Mô tả:** Resize ảnh về target_size × target_size bằng letterbox (giữ aspect ratio, pad đen). Lưu scale và pad để decode bbox sau.

**File implement:** `src/lfx/src/lfx/components/vision/letterbox.py`
**Thư viện:** OpenCV (`cv2`)

**Acceptance Criteria:**

- [ ] Nhận: `List[ImageData]` + config
- [ ] `target_size` dùng `IntInput`, configurable (default: `640`)
- [ ] Scale theo cạnh dài → pad màu đen phần còn lại
- [ ] Dùng `cv2.copyMakeBorder()` — không resize thẳng về square
- [ ] Lưu đúng `scale`, `pad_left`, `pad_top` — bắt buộc để decode bbox
- [ ] Output: `List[ProcessedImage]` đúng schema
- [ ] Giữ đúng thứ tự index

---

## Node: Normalize

**Status:** `[ ] TODO`

**Mô tả:** Apply normalization theo config của từng model. User chọn preset hoặc tự nhập mean/std.

**File implement:** `src/lfx/src/lfx/components/vision/normalize.py`
**Thư viện:** NumPy

**Acceptance Criteria:**

- [ ] Nhận: `List[ProcessedImage]` + config
- [ ] `preset` dùng `DropdownInput` với `combobox=True`:
  ```
  SCRFD    → mean=[0,0,0],             std=1.0,   /255=True
  YOLOv8   → mean=[0,0,0],             std=1.0,   /255=True
  ImageNet → mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225], /255=True
  TinaFace → mean=[127.5,127.5,127.5], std=128.0, /255=False
  Custom   → user tự nhập mean và std
  ```
- [ ] Khi chọn preset → tự điền mean/std fields
- [ ] Khi chọn Custom → mean/std fields trống, user nhập tay
- [ ] Convert BGR → RGB trước khi normalize
- [ ] Công thức nếu `/255=True`:  `(pixel / 255 - mean) / std`
- [ ] Công thức nếu `/255=False`: `(pixel - mean) / std`
- [ ] Output: `List[NormalizedImage]` đúng schema

---

## Node: To Tensor

**Status:** `[ ] TODO`

**Mô tả:** Convert ảnh đã normalize thành tensor NCHW/NHWC, serialize thành base64.

**File implement:** `src/lfx/src/lfx/components/vision/to_tensor.py`
**Thư viện:** NumPy

**Acceptance Criteria:**

- [ ] Nhận: `List[NormalizedImage]` + config
- [ ] `format` dùng `DropdownInput`: `NCHW` (default) | `NHWC`
- [ ] NCHW: `np.transpose(arr, (2,0,1))` → `np.expand_dims(axis=0)`
- [ ] NHWC: chỉ `np.expand_dims(axis=0)`
- [ ] dtype: `float32`
- [ ] Serialize tensor → bytes → base64
- [ ] Lưu `shape` và `format` vào output để Inference node deserialize đúng
- [ ] Output: `List[TensorData]` đúng schema

---

## Node: SCRFD Inference

**Status:** `[ ] TODO`

**Mô tả:** Load ONNX model local, chạy inference, decode anchor boxes về tọa độ thật trên ảnh 640×640. Đây là node SCRFD-specific — decode anchor gộp vào bên trong.

**File implement:** `src/lfx/src/lfx/components/vision/scrfd_inference.py`
**Thư viện:** `onnxruntime`, NumPy
**Setup:** `pip install onnxruntime`

**Acceptance Criteria:**

**Model loading:**
- [ ] `model_path` dùng `StrInput` — đường dẫn tuyệt đối đến file `.onnx`
- [ ] Load model 1 lần duy nhất khi component init, cache `_session`
- [ ] `provider` dùng `DropdownInput`: `CPUExecutionProvider` (default) | `CUDAExecutionProvider` | `CoreMLExecutionProvider`
- [ ] Nếu file không tồn tại → raise `FileNotFoundError`

**Inference:**
- [ ] `input_name` dùng `StrInput` (default: `"input.1"`) — lấy từ Netron
- [ ] `output_names` dùng `StrInput` comma-separated
  (default: `"score_8,score_16,score_32,bbox_8,bbox_16,bbox_32"`)
- [ ] `use_kps` dùng `BoolInput` (default: `True`)
  — nếu True: output_names phải bao gồm `kps_8,kps_16,kps_32`
- [ ] Deserialize `tensor_b64` → numpy array đúng `shape` và `dtype`
- [ ] Chạy `session.run(output_names, {input_name: tensor})`

**Decode anchor:**
- [ ] Decode delta values `(dx, dy, dw, dh)` về tọa độ thật trên ảnh 640×640:
  ```
  cx = cx_anchor + dx × stride
  cy = cy_anchor + dy × stride
  w  = w_anchor  × exp(dw)
  h  = h_anchor  × exp(dh)
  ```
- [ ] Scale bbox về tọa độ ảnh gốc dùng `scale`, `pad_left`, `pad_top`
- [ ] Nếu `use_kps=True`: decode và scale landmarks tương tự
- [ ] Output: `List[RawDetection]` đúng schema — tọa độ trên ảnh GỐC

---

## Node: Score Threshold

**Status:** `[ ] TODO`

**Mô tả:** Generic filter — loại bỏ detection có score thấp hơn threshold.

**File implement:** `src/lfx/src/lfx/components/vision/score_threshold.py`

**Acceptance Criteria:**

- [ ] Nhận: `List[RawDetection]` + config
- [ ] `threshold` dùng `FloatInput` (default: `0.5`, range: 0.0–1.0)
- [ ] Filter: giữ lại boxes có `score >= threshold`
- [ ] Giữ alignment giữa `boxes`, `scores`, `landmarks` sau filter
- [ ] Nếu tất cả bị filter → trả về item với `boxes=[]`, `scores=[]`
- [ ] KHÔNG raise error khi không còn detection — đây là kết quả hợp lệ
- [ ] Output: `List[FilteredDetection]` đúng schema

---

## Node: NMS

**Status:** `[ ] TODO`

**Mô tả:** Generic Non-Maximum Suppression — loại bỏ bbox chồng nhau.

**File implement:** `src/lfx/src/lfx/components/vision/nms.py`
**Thư viện:** `torchvision.ops.nms` hoặc `cv2.dnn.NMSBoxes`

**Acceptance Criteria:**

- [ ] Nhận: `List[FilteredDetection]` + config
- [ ] `algorithm` dùng `DropdownInput`: `Standard` (default) | `Soft`
- [ ] `iou_threshold` dùng `FloatInput` (default: `0.4`, range: 0.0–1.0)
- [ ] Standard NMS: dùng `torchvision.ops.nms(boxes, scores, iou_threshold)`
- [ ] Sau NMS: convert boxes về `FaceResult.faces[]` format
- [ ] Scale bbox đã được decode từ SCRFD Inference — KHÔNG scale lại
- [ ] Tính `face_count` = số faces còn lại sau NMS
- [ ] Nếu `faces=[]` → `face_count=0`, `status="success"` (không phải lỗi)
- [ ] Output: `List[FaceResult]` đúng schema

---

## Node: Face Annotator

**Status:** `[ ] TODO`

**Mô tả:** Extend Image Annotator từ car pipeline. Override để handle nhiều faces per ảnh và vẽ thêm 5-point landmarks.

**File implement:** `src/lfx/src/lfx/components/vision/face_annotator.py`
**Extend:** `src/lfx/src/lfx/components/vision/image_annotator.py`
**Thư viện:** OpenCV (`cv2`)

**Khác biệt so với Image Annotator (car):**

| | Image Annotator (car) | Face Annotator |
|---|---|---|
| bbox per ảnh | 1 | nhiều (loop `faces[]`) |
| Label | brand/model/color | score % |
| Landmarks | ❌ | ✅ 5 điểm |
| Input schema | `List[VisionResult]` | `List[FaceResult]` |

**Acceptance Criteria:**

- [ ] Extend `ImageAnnotatorComponent` — kế thừa base logic
- [ ] Nhận: `List[ImageData]` + `List[FaceResult]`
- [ ] Match theo `index`
- [ ] **Loop qua `faces[]`** trong mỗi FaceResult — vẽ bbox cho TẤT CẢ faces
- [ ] Mỗi face: vẽ bounding box màu xanh lá `(0, 255, 0)`, độ dày 2px
- [ ] Label: `"{score*100:.0f}%"` tại góc trên-trái bbox
- [ ] Nếu `face_count=0` → giữ nguyên ảnh gốc, không annotate
- [ ] Nếu `use_kps=True` và `landmarks` không None: vẽ 5 điểm
  ```
  Điểm 0,1 (mắt):   màu xanh dương (255, 0, 0)
  Điểm 2   (mũi):   màu vàng       (0, 255, 255)
  Điểm 3,4 (miệng): màu đỏ         (0, 0, 255)
  Radius: 3px, filled
  ```
- [ ] `use_kps` dùng `BoolInput` (default: `True`)
- [ ] Output ảnh GỐC (chưa resize) đã annotate
- [ ] Output: `List[AnnotatedImage]` đúng schema (tái sử dụng từ car pipeline)

---

## Integration: Full Pipeline

**Status:** `[ ] TODO`

**Acceptance Criteria:**

- [ ] Upload 3 ảnh bulk → full flow → 3 ảnh annotated với bbox + landmarks
- [ ] Ảnh có 0 mặt → face_count=0, ảnh gốc giữ nguyên
- [ ] Ảnh có nhiều mặt → tất cả faces được annotate
- [ ] `POST /api/v1/run/{flow_id}` → HTTP 200
- [ ] `POST /api/v1/webhook/{flow_id}` → flow chạy tự động
- [ ] LangSmith trace đủ bước + latency per node

---

## Dataset test

- Stanford Face Dataset hoặc WiderFace validation subset
- Cần ảnh cover các case: 0 face, 1 face, nhiều faces, faces nhỏ

---

## Checklist setup local

- [ ] `pip install onnxruntime`
- [ ] Download model: `scrfd_2.5g_bnkps.onnx` từ InsightFace
- [ ] Inspect model bằng Netron — confirm input/output names
- [ ] Chạy thử inference đơn lẻ trước khi integrate vào flow

---

## Changelog

| Ngày | Thay đổi |
|---|---|
| 2026-05-29 | Khởi tạo file |

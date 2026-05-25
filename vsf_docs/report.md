# Báo cáo nghiên cứu: Flow Execution System

> **Người thực hiện:** TrungTomNg
>
> **Thời gian:** Tuần 1 — Thực tập  
> **Mục tiêu:** Research và đề xuất kiến trúc cho task được giao

---

## 1. Task Overview

### Yêu cầu

**Input:**

- User tự thiết kế flow configuration trên Frontend
- Tham khảo component-based design của LangFlow (kéo thả, kết nối, configure)

**Output:**

- Hệ thống thực thi flow hoạt động được
- Hỗ trợ **manual trigger** (user bấm Run)
- Hỗ trợ **automated trigger** từ external services (webhook)

### Phạm vi và Timeline

- Dự án mang tính **workaround / thử nghiệm nội bộ**
- Trong tương lai: tích hợp với MCP pipeline (nhóm khác đang làm)
- **Deadline: 6 tuần thực tập**

---

## 2. Nghiên cứu: Hệ sinh thái công cụ

### 2.1 Nguồn gốc các công cụ

| Tool          | Tổ chức                   | Vai trò                        |
| ------------- | ------------------------- | ------------------------------ |
| **LangChain** | LangChain Inc.            | Framework nền tảng             |
| **LangGraph** | LangChain Inc.            | Execution engine (graph-based) |
| **LangFlow**  | Logspace → DataStax → IBM | Visual builder UI              |
| **LangSmith** | LangChain Inc.            | Observability & monitoring     |

### 2.2 Có tool "all-in-one" chưa? — Dify

Sau khi research sâu hơn, **đã có** một tool gần với all-in-one: **Dify** (github.com/langgenius/dify, ~124K stars).

#### Dify là gì?

Dify là open-source LLM app platform tích hợp trong một package duy nhất:

#### So sánh Dify vs LangFlow cho task này

| Tiêu chí              | LangFlow                      | Dify                     |
| --------------------- | ----------------------------- | ------------------------ |
| LangGraph integration | ✅ Native                     | ❌ Không dùng LangGraph  |
| License               | ✅ MIT — tự do fork/customize | ⚠️ Có hạn chế commercial |
| Observability         | Cần thêm LangSmith            | ✅ Built-in              |
| Production queue      | Cần tự thêm                   | ✅ Celery+Redis sẵn      |
| Phù hợp với task      | ✅                            | ⚠️                       |

#### Tại sao vẫn chọn LangFlow?

```
Dify  → Execution engine tự viết riêng → KHÓ kiểm soát hơn LangGraph
```

Ngoài ra, LangFlow dùng **MIT license** — tự do fork, customize, và tích hợp vào hệ thống nội bộ mà không có ràng buộc thương mại.

> **Kết luận cập nhật:** Dify là tool gần nhất với all-in-one trong thị trường hiện tại, nhưng không phù hợp cho task này.

---

## 3. Phân tích LangFlow Open Source

**Repository:** https://github.com/langflow-ai/langflow  
**Stars:** ~148K | **License:** MIT

### 3.1 Kiến trúc kỹ thuật

```
Frontend  (React 19 + TypeScript)
  └── @xyflow/react       ← Canvas kéo thả nodes/edges (v12)
  └── reactflow           ← v11 vẫn còn (đang migration sang @xyflow/react)
  └── Zustand             ← State management
  └── Component Library   ← Sidebar các component có sẵn

Backend  (Python + FastAPI)
  └── create_app()        ← Tạo FastAPI app, mount routers
  └── setup_app()         ← Entry point production (static files + backend-only mode)
  └── Package lfx         ← Execution core riêng biệt (Graph, Vertex, Component)
  └── REST API            ← /run, /webhook, /build endpoints
  └── SSE streaming       ← Stream kết quả qua text/event-stream (KHÔNG phải WebSocket)
  └── WebSocket           ← Chỉ dùng cho voice mode (/ws/flow_as_tool)

Database
  └── SQLite (dev) / PostgreSQL (prod)
  └── Flow storage, session management
```

### 3.2 Những gì LangFlow đã giải quyết sẵn

| Bài toán               | LangFlow đã có                                               |
| ---------------------- | ------------------------------------------------------------ |
| Canvas kéo thả nodes   | ✅ `@xyflow/react` integration                               |
| Serialize flow → JSON  | ✅ Flow JSON schema hoàn chỉnh                               |
| JSON → Graph execution | ✅ Component class system                                    |
| State management       | ✅ Tự động từ component inputs/outputs                       |
| Conditional branching  | ✅ Router node, Conditional component                        |
| Manual trigger         | ✅ `POST /api/v1/run/{flow_id}`                              |
| Automated trigger      | ✅ `POST /api/v1/webhook/{flow_id}`                          |
| Stream kết quả         | ✅ SSE (Server-Sent Events) qua `GET /build/{job_id}/events` |
| Component library      | ✅ 50+ components có sẵn                                     |

### 3.3 Kết luận về LangFlow OSS

> **Fork và customize thay vì build từ đầu.** LangFlow đã giải quyết các bài toán khó nhất (UI canvas, flow serialization, execution mapping). Việc cần làm là extend thêm các tính năng đặc thù của team, không phải viết lại.

---

## 4. Đề xuất Image Processing trong dự án

> Dự án sẽ build các component xử lý hình ảnh để phục vụ use case nhận diện xe oto

### 4.1 Đề xuất 3 nodes Image Processing cho MVP

**Node 1 — `Image Input`**

```
Chức năng: Nhận ảnh đầu vào từ user
Input:     Upload file ảnh xe (jpg, png...)
Output:    Image data dạng base64 → truyền vào state
```

**Node 2 — `Vision Analysis`**

```
Chức năng: Gửi ảnh cho AI model nhận dạng xe
Input:     Image data + system prompt
           ("Nhận dạng hãng xe, model, màu sắc, biển số...")
Model:     GPT-4o Vision hoặc Claude Vision (configurable)
Output:    Findings dạng text (hãng xe, model, màu, biển số...)
```

**Node 3 — `Image Annotator`**

```
Chức năng: Vẽ kết quả nhận dạng lên ảnh gốc
Input:     Image data + findings từ Vision Analysis node
Thư viện:  OpenCV (bounding box, label) hoặc Pillow
Output:    Ảnh đã annotate → hiển thị trên UI hoặc lưu file
```

---

### 4.3 Pipeline đề xuất cho use case nhận dạng xe ô tô

```
[Image Input]
      ↓ image data (base64)
[Vision Analysis]  ←  prompt: "Nhận dạng xe ô tô trong ảnh"
      ↓ findings: { hãng, model, màu, biển số, confidence }
[Conditional Branch]
    ↙ nhận dạng được              ↘ không nhận dạng được
[Image Annotator]             [Output: "Không tìm thấy xe"]
      ↓ ảnh đã annotate
[Output: ảnh + thông tin xe]
```

## 5. Kiến trúc hệ thống đề xuất

### 5.1 Sơ đồ tổng thể

```
┌──────────────────────────────────────────────────────┐
│                     FRONTEND                          │
│   Component Sidebar │ Canvas Editor │ Config Panel    │
│   (fork LangFlow)   │ (@xyflow/react)│ (node settings)│
│                     │               │                 │
│                     └───────────────┘                 │
│                        Flow Config JSON               │
└────────────────────────────┬─────────────────────────┘
                             │ HTTP / WebSocket
              ┌──────────────┼──────────────┐
              │ Manual       │              │ Auto
              │ POST /run    │              │ POST /webhook
              ▼              ▼              ▼
┌──────────────────────────────────────────────────────┐
│                     BACKEND                           │
│                                                       │
│   ┌─────────────────────────────────────────────┐    │
│   │         LangGraph Execution Engine           │    │
│   │  Parse JSON → Build Graph → Execute DAG      │    │
│   │  State | Branching | Loops | Human-in-loop   │    │
│   └─────────────────────────────────────────────┘    │
│                         ↓                            │
│   ┌─────────────────────────────────────────────┐    │
│   │           LangChain Tool Layer               │    │
│   │   LLMs │ Tools │ Memory │ Image Processing   │    │
│   └─────────────────────────────────────────────┘    │
│                         ↓                            │
│   ┌─────────────────────────────────────────────┐    │
│   │         LangSmith (Observability)            │    │
│   │         Trace │ Debug │ Benchmark            │    │
│   └─────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
                         ↓ (future)
              MCP Integration Layer
              (nhóm khác đang xây dựng)
```

### 5.2 Phân công công cụ

| Layer            | Tool            | Nguồn           | Ghi chú            |
| ---------------- | --------------- | --------------- | ------------------ |
| Canvas UI        | `@xyflow/react` | Open source MIT | Dùng trực tiếp     |
| Component system | LangFlow source | Fork MIT        | Tham khảo + extend |
| Flow JSON schema | LangFlow        | Fork            | Giữ nguyên format  |
| Execution engine | LangGraph       | Pip package     | Core backend       |
| Tools/Models     | LangChain       | Pip package     | Building blocks    |
| Observability    | LangSmith       | Pip package     | Debug & benchmark  |
| API server       | FastAPI         | Pip package     | Wrapper mỏng       |

### 5.3 Trigger system

```
Manual Trigger:
  User bấm Run trên UI
    → Frontend: POST /api/v1/build/{flow_id}/flow → nhận job_id
    → Frontend: GET /api/v1/build/{job_id}/events (SSE stream)
    → Backend: LangGraph execute flow, emit events từng bước
    → Kết quả hiển thị realtime qua text/event-stream

Automated Trigger:
  External service (webhook, cron, event)
    → POST /api/v1/webhook/{flow_id}
    → Backend: parse raw request body, trigger flow execution async
    → Không cần user online
```

---

## 6. MVP Definition

### Mục tiêu MVP

> Một hệ thống cho phép user **thiết kế flow trên UI** và **chạy được flow đó** — theo cả 2 cách manual và webhook — trong môi trường internal/test.

### In scope ✅

**Frontend**

- Canvas kéo thả (fork LangFlow UI)
- 8 loại node (xem bảng bên dưới)
- Kết nối nodes bằng edges
- Config panel cho từng node
- Save / Load flow
- Nút Run thủ công

**Backend**

- Parse flow JSON → build LangGraph graph
- Execute flow, trả kết quả
- `POST /run/{flow_id}` — manual trigger
- `POST /webhook/{flow_id}` — automated trigger
- Stream kết quả từng bước qua SSE

**Observability**

- LangSmith integration — trace từng lần chạy
- Log token usage cơ bản

### Danh sách 8 nodes MVP

| #   | Node                 | Chức năng                                          |
| --- | -------------------- | -------------------------------------------------- |
| 1   | `Input`              | Nhận text từ user                                  |
| 2   | `Image Input`        | Nhận ảnh upload (jpg, png, dicom...)               |
| 3   | `LLM`                | Gọi OpenAI / Claude text                           |
| 4   | `Vision Analysis`    | Gọi GPT-4o Vision / Claude Vision phân tích ảnh    |
| 5   | `Tool`               | Tavily search, custom tool                         |
| 6   | `Image Annotator`    | Vẽ bounding box / label lên ảnh bằng OpenCV/Pillow |
| 7   | `Conditional Branch` | Rẽ nhánh if/else dựa trên output của node trước    |
| 8   | `Output`             | Hiển thị kết quả (text hoặc ảnh)                   |

### Out of scope ❌

- Custom Component API (user tự viết node bằng code)
- Multi-tenant / auth / RBAC
- Async job queue (Celery/Redis)
- MCP integration
- Production deployment

### Definition of Done

```
✅ Tạo flow ≥ 3 nodes trên UI, bấm Run → có kết quả
✅ POST /webhook/{id} từ curl → flow chạy tự động
✅ Upload ảnh → Vision Analysis → Image Annotator → trả ảnh đã annotate
✅ Conditional Branch hoạt động — flow rẽ đúng nhánh
✅ LangSmith hiển thị trace của mỗi lần chạy
```

---

## 7. Lộ trình 6 tuần

> **Nguyên tắc:** Dễ trước khó sau — mỗi tuần deliver 1 tính năng chạy được end-to-end, build trực tiếp trên LangFlow fork thay vì tách riêng Frontend/Backend. Dùng Claude Code (vibe coding) để tăng tốc implement.

| Tuần          | Việc làm                                                                 | Deliverable                              |
| ------------- | ------------------------------------------------------------------------ | ---------------------------------------- |
| **Tuần 1** ✅ | Fork LangFlow, chạy local, chạy thử flow có sẵn, setup LangSmith         | LangFlow chạy local, có trace LangSmith  |
| **Tuần 2**    | Viết `Image Input` + `Vision Analysis` node, test trên UI LangFlow       | Upload ảnh → AI phân tích → kết quả text |
| **Tuần 3**    | Viết `Image Annotator` + `Conditional Branch` node, test flow đầy đủ     | Image pipeline end-to-end chạy được      |
| **Tuần 4**    | Confirm webhook trigger, test automated trigger, kết nối hệ thống nội bộ | Cả 2 trigger ổn định                     |
| **Tuần 5**    | Buffer — xử lý trễ từ tuần trước; nếu đúng tiến độ: thêm text nodes      | MVP đủ 5 Definition of Done              |
| **Tuần 6**    | Viết README, chuẩn bị demo 2 flow (text + image), demo cho mentor/team   | Demo thành công, có documentation        |

---

## 8. Câu hỏi cần confirm với mentor/team

1. Xin review của team về phần process image do bản thân chưa có kinh nghiệm ở mảng này ?

---

## 9. Chiến lược kiểm thử (Vibe Coding)

### 9.1 Bốn tầng kiểm thử

**Tầng 1 — Unit Test từng Node (pytest)**

LangFlow đã có sẵn base class tại `src/backend/tests/unit/base.py`. Claude Code kế thừa và viết test theo đúng pattern

**Tầng 2 — Mock LLM Test cho Vision Analysis**

Không gọi API thật mỗi lần test — dùng mock để test logic xử lý output

```python
@pytest.fixture
def mock_car_detected():
    return {"brand": "Toyota", "model": "Camry",
            "color": "White", "confidence": 0.92}

@pytest.fixture
def mock_car_not_detected():
    return {"brand": None, "confidence": 0.3}

def test_high_confidence_routes_to_annotate(mock_car_detected):
    result = conditional_branch(mock_car_detected)
    assert result["route"] == "annotate"

def test_low_confidence_routes_to_manual_review(mock_car_not_detected):
    result = conditional_branch(mock_car_not_detected)
    assert result["route"] == "manual_review"
```

---

**Tầng 3 — Integration Test với LangSmith**

Chuẩn bị dataset ảnh xe thật, chạy toàn bộ flow và đo kết quả:

```
Dataset: 10-20 ảnh xe với expected output
  { input: "toyota_camry.jpg", expected_brand: "Toyota" }

Chạy qua LangSmith → xem:
  - Accuracy: bao nhiêu % nhận dạng đúng?
  - Token cost: mỗi lần chạy tốn bao nhiêu?
  - Node nào thường fail hoặc chậm nhất?
```

---

**Tầng 4 — Manual Trigger Test (Definition of Done)**

Chạy trước khi kết thúc mỗi tuần — kiểm tra đúng 5 DoD:

```bash
# Test manual trigger
curl -X POST http://localhost:7860/api/v1/run/{flow_id} \
  -d '{"input_value": "test"}' → expect 200 + có kết quả

# Test webhook trigger
curl -X POST http://localhost:7860/api/v1/webhook/{flow_id} \
  -d '{"image_url": "..."}' → expect flow chạy tự động
```

---

## 10. Tài nguyên tham khảo

- LangFlow OSS: https://github.com/langflow-ai/langflow
- LangGraph docs: https://langchain-ai.github.io/langgraph/
- @xyflow/react: https://reactflow.dev/
- LangSmith: https://smith.langchain.com/
- LangGraph multimodal example: https://aws-samples.github.io/amazon-bedrock-samples/agents-and-function-calling/open-source-agents/langgraph/langgraph-agents-multimodal/
- Dify OSS: https://github.com/langgenius/dify
- So sánh Dify vs LangFlow vs Flowise: https://blog.elest.io/dify-vs-langflow-vs-flowise-which-open-source-llm-app-builder-actually-ships-to-production/
- Open Source AI Agent Platform Comparison 2026: https://jimmysong.io/blog/open-source-ai-agent-workflow-comparison/

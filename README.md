# -AI- · 校园基础设施智能巡检系统

基于无人机/摄像头图像，使用 AI 自动识别维修需求的智能系统。

---

## 项目架构

```
1️⃣ 数据采集层（无人机或手机）  →  2️⃣ 感知识别层（本模块）  →  3️⃣ 决策与生成层  →  4️⃣ 调度与优化层
```

本仓库当前实现了 **感知识别层（Perception Layer）**，基于 YOLOv8 + OpenCV。

---

## 感知层功能

| 功能    | 实现                                                                        |
|-------|---------------------------------------------------------------------------|
| 目标检测  | YOLOv8（`ultralytics`）                                                     |
| 图像预处理 | OpenCV：基础去噪（高斯模糊）                                                         |
| 结构化输出 | JSON（类型、置信度、位置、严重程度、边界框）                                                  |
| 可检测对象 | Crack、Manhole、Net、Pothole、Patch-Crack、Patch-Net、Patch-Pothole、other、Other |
| 入口形态  | 单路径输入（`main.py`）：图片/视频/流地址                                           |
| 可视化   | 彩色边界框 + 标签叠加                                                              |

### 输出示例（JSON）

```json
[
  {
    "type": "裂缝",
    "confidence": 0.9200,
    "location": "未知区域",
    "severity": "high",
    "bounding_box": { "x1": 120.0, "y1": 80.0, "x2": 340.0, "y2": 200.0 }
  }
]
```

严重程度分级：

| 严重程度 | 置信度范围 |
|----------|-----------|
| `high`   | ≥ 0.75    |
| `medium` | 0.50 – 0.74 |
| `low`    | < 0.50    |

---

## 目录结构

```
.
├── config/
│   └── detection_config.yaml   # 模型、检测阈值、基础预处理配置
├── models/                     # 存放自定义训练权重（.pt 文件）
├── perception/
│   ├── __init__.py
│   ├── detector.py             # YOLODetector：核心检测类
│   ├── preprocessor.py         # ImagePreprocessor：基础图片预处理
│   └── utils/
│       ├── __init__.py
│       ├── output.py           # 结构化 JSON 输出格式化
│       └── visualization.py    # 边界框可视化
├── tests/
│   ├── test_main_entry.py      # 主入口行为测试
│   ├── test_perception.py      # 感知层核心测试
│   ├── test_class_map_sync.py  # class_map 同步/校验测试
│   └── ...
└── main.py                     # CLI 入口
```

---

## 快速开始

### 1. 安装依赖（Windows PowerShell）

```powershell
Set-Location "C:\Users\24053\PycharmProjects\-AI-"
uv sync
```

### 2. 配置

编辑 `config/detection_config.yaml`，主要参数：

```yaml
model:
  weights: "models/best.pt"      # 默认使用本地训练权重
  confidence_threshold: 0.4
  iou_threshold: 0.45
  max_detections: 100
  image_size: 640

class_map:                       # YOLO 类别索引 → 损坏类型名称
  0: Crack
  1: Manhole
  2: Net
  3: Pothole
  4: Patch-Crack
  5: Patch-Net
  6: Patch-Pothole
  7: other
  8: Other

# 可选：从训练 data.yaml 自动校验 class_map（防止类别顺序错位）
class_map_sync:
  data_yaml: "D:/.../lumian"    # 可填 data.yaml 文件，或其所在目录
  mode: "warn"                  # warn | strict | overwrite
```

### 3. 运行

```powershell
uv run ai-inspect "path/to/image.jpg"
```

程序会：
- 在终端打印检测 JSON 字符串
- 同时把结果保存到 `outputs/<图片名>/`：
  - `outputs/<图片名>/<图片名>.json`
  - `outputs/<图片名>/<图片名>_annotated.jpg`

视频文件或流地址同样支持：

```powershell
uv run ai-inspect "path/to/video.mp4" --frame-interval 5 --location "A区-3号楼"
```

视频模式会保存到 `outputs/<视频名>/`：
- `outputs/<视频名>/<视频名>.json`（按帧结果，包含 `frame_index` 和 `timestamp_ms`）
- `outputs/<视频名>/<视频名>_annotated.mp4`

也支持摄像头索引和RTSP/HTTP流：

```powershell
uv run ai-inspect 0
uv run ai-inspect "rtsp://127.0.0.1:8554/live"
```

### 4. 在代码中使用

```python
from main import detect_image_json

json_str = detect_image_json("image.jpg")
print(json_str)
```

如果你更习惯直接运行脚本，也可以使用：

```powershell
uv run python main.py "path/to/image.jpg"
uv run python main.py "path/to/video.mp4" --frame-interval 3
```

---

## 训练自定义模型

若需要针对校园场景进行微调，可参考 [Ultralytics YOLOv8 训练文档](https://docs.ultralytics.com/modes/train/)：

```powershell
yolo detect train data=campus_damage.yaml model=yolov8n.pt epochs=100 imgsz=640
```

训练完成后，将 `runs/detect/train/weights/best.pt` 的路径填入 `config/detection_config.yaml` 的 `model.weights` 字段。
> 本项目已预设了 `best.pt` 作为默认权重，适合快速测试和小规模数据集微调。
---

## 运行测试

```powershell
uv run pytest tests/ -v
```

---

## 技术栈

- **Python 3.10+**
- **YOLOv8** (`ultralytics`) – 目标检测
- **OpenCV** – 图像预处理（基础去噪）
- **NumPy** – 数组操作
- **PyYAML** – 配置文件解析
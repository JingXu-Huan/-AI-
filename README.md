# -AI- · 校园基础设施智能巡检系统

基于无人机/摄像头图像，使用 AI 自动识别维修需求的智能系统。

---

## 项目架构

```
1️⃣ 数据采集层  →  2️⃣ 感知识别层（本模块）  →  3️⃣ 决策与生成层  →  4️⃣ 调度与优化层
```

本仓库当前实现了 **感知识别层（Perception Layer）**，基于 YOLOv8 + OpenCV。

---

## 感知层功能

| 功能 | 实现 |
|------|------|
| 目标检测 | YOLOv8（`ultralytics`） |
| 图像预处理 | OpenCV：去噪、CLAHE 对比度增强、边缘增强 |
| 结构化输出 | JSON（类型、置信度、位置、严重程度、边界框） |
| 可检测对象 | 路面裂缝、坑洞、管道泄漏、积水/漏水、设备损坏 |
| 视频支持 | 逐帧检测，可配置采样间隔 |
| 可视化 | 彩色边界框 + 标签叠加 |

### 输出示例（JSON）

```json
[
  {
    "type": "pipe_leak",
    "confidence": 0.9200,
    "location": "A区-3号楼",
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
│   └── detection_config.yaml   # 模型、检测阈值、预处理开关等配置
├── models/                     # 存放自定义训练权重（.pt 文件）
├── perception/
│   ├── __init__.py
│   ├── detector.py             # YOLODetector：核心检测类
│   ├── preprocessor.py         # ImagePreprocessor：OpenCV 预处理流水线
│   └── utils/
│       ├── __init__.py
│       ├── output.py           # 结构化 JSON 输出格式化
│       └── visualization.py    # 边界框可视化
├── tests/
│   └── test_perception.py      # 单元测试（25 个测试用例）
├── main.py                     # CLI 入口
└── requirements.txt
```

---

## 快速开始

### 1. 安装依赖（推荐 `uv`）

```bash
uv sync
```

### 2. 配置

编辑 `config/detection_config.yaml`，主要参数：

```yaml
model:
  weights: "yolov8n.pt"          # 预训练或自定义权重路径
  confidence_threshold: 0.4
  iou_threshold: 0.45

class_map:                       # YOLO 类别索引 → 损坏类型名称
  0: "road_crack"
  1: "pothole"
  2: "pipe_leak"
  3: "water_accumulation"
  4: "equipment_damage"
```

### 3. 运行

```bash
# 检测单张图片
uv run ai-inspect --image path/to/image.jpg --location "A区-3号楼"

# 检测图片并保存可视化结果
uv run ai-inspect --image path/to/image.jpg --location "A区-3号楼" \
                 --save-viz output_annotated.jpg --output results.json

# 检测视频（每30帧采样一次）
uv run ai-inspect --video path/to/video.mp4 --location "B区" --frame-interval 15
```

### 4. 在代码中使用

```python
from perception import YOLODetector
from perception.utils import detection_to_json, draw_detections
import cv2

detector = YOLODetector("config/detection_config.yaml")

# 检测图片
detections = detector.detect_image("image.jpg", location="A区-3号楼")
print(detection_to_json(detections))

# 可视化
img = cv2.imread("image.jpg")
annotated = draw_detections(img, detections)
cv2.imwrite("annotated.jpg", annotated)
```

如果你更习惯直接运行脚本，也可以使用：

```bash
uv run python main.py --image path/to/image.jpg --location "A区-3号楼"
```

---

## 训练自定义模型

若需要针对校园场景进行微调，可参考 [Ultralytics YOLOv8 训练文档](https://docs.ultralytics.com/modes/train/)：

```bash
yolo detect train data=campus_damage.yaml model=yolov8n.pt epochs=100 imgsz=640
```

训练完成后，将 `runs/detect/train/weights/best.pt` 的路径填入 `config/detection_config.yaml` 的 `model.weights` 字段。

---

## 运行测试

```bash
uv run pytest tests/ -v
```

---

## 技术栈

- **Python 3.10+**
- **YOLOv8** (`ultralytics`) – 目标检测
- **OpenCV** – 图像预处理（去噪、CLAHE、边缘增强）
- **NumPy** – 数组操作
- **PyYAML** – 配置文件解析
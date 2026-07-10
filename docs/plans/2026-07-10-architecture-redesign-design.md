# 红灯压斑马线检测系统 — v2.0 分层架构重构设计文档

> **日期**: 2026-07-10  
> **方法论**: Superpowers Spec-First TDD + Grill-Me 压力测试  
> **状态**: 🟡 DRAFT — 待用户 review & grill-me 质疑  
> **硬门控**: ⚠️ 用户批准此设计前，禁止任何编码工作

---

## 1. 项目概述与目标

### 1.1 核心使命
从手机拍摄的视频中自动检测"红灯时车辆停在斑马线上"的交通违规行为，并识别车牌号。

### 1.2 当前能力矩阵（已验证可工作 ✅）
| 能力 | 方案 | 实测状态 |
|------|------|----------|
| 车辆检测 | ultralytics YOLOv8n (torch CPU) | ✅ 6.9 fps, 框车精准 |
| 斑马线检测 | 纯 CV v7 (HSV+形态学+位置过滤) | ✅ 掩膜生成正常 |
| 红绿灯状态 | HSV 颜色兜底 (RED/GREEN/UNKNOWN) | ✅ 切换准确 |
| 车牌识别 | HyperLPR3 HIGH (中文车牌) | ✅ 读出号码(如京B13703) |
| 车辆跟踪 | SimpleTracker (IoU贪婪匹配) | ✅ ID稳定 |
| 违规判定 | 三条件状态机 (红+静止+压线) | ⚠️ 判定阈值需调优 |
| 输出产物 | 标注视频(mp4) + CSV + 证据图(jpg) | ✅ 全部产出 |

### 1.3 重构驱动力
用户明确提出的 **0-7 条工程化要求**（2026-07-10）：

| # | 要求 | 对应本章节 |
|---|------|-----------|
| 0 | 工程大，做好基本要求 | §2 经验教训 |
| 1 | **先写文档 → 再设计架构 → 最后开发**（避免反复） | 本文全文 |
| 2 | **测试用例 + 单元测试**（避免质量反复） | §11 测试策略 |
| 3 | **重大修改 git commit**（避免无法回退） | §12 Git 规范 |
| 4 | **分层架构**: 数据管道 / 任务步骤 / 工程化 / 模型训练 / 评测工具 / 训练数据 / 评测集 / 模型参数 | §3~§9 |
| 5 | **总结经验教训**，引用 superpowers + grill-me，保证不跑偏 | §2 + §15 |
| 6 | **标准评测指标**: precision / recall / F1 / accuracy / ROUGE 等 | §8 评测工具 |
| 7 | **泛化性**，不要 overfitting 到这几个用例 | §14 泛化保障 |

---

## 2. 经验教训（来自 v1.0 开发过程）

### ❌ 已犯过的错误

| # | 错误 | 影响 | 修正措施 |
|---|------|------|----------|
| E1 | PowerShell 内联多行 Python（`\n`被吞） | 调试浪费 2h+ | ✅ 改为脚本文件+落盘日志 |
| E2 | VC++ 运行库缺失导致 DLL 初始化失败 | torch/onnxruntime 无法加载 | ✅ 安装 VC++ 2015-2022 redist |
| E3 | HyperLPR3 API 名字猜错（`LPR3` vs `LicensePlateCatcher`） | 浪费 3 轮试错 | ✅ 先读源码再调用 |
| E4 | pipeline.py 是 God Object（165行包一切） | 难以单独测试每个模块 | → 本次重构解决 |
| E5 | scripts/ 积累 21 个临时脚本未清理 | 杂乱、不可复用 | → 本次重构解决 |
| E6 | 无评测指标体系 | 不知道"好"的标准是什么 | → 新增评测工具层 |
| E7 | 违规判定阈值（speed=15px/s）对蠕行太严 | 两段视频都 viol=0 | → 需要数据驱动调参 |
| E8 | 斑马线掩膜偏大（延伸到草地） | overlap_ratio 失真 | → 需要掩膜质量评测 |
| E9 | 无单元测试 → 任何改动都可能引入回归 bug | → 新增测试策略 |
| E10 | 未建立 git 提交规范 | 无法追溯变更 | → 新增 git 规范 |

### ✅ 已验证的最佳实践
- **CPU-only 环境**: Win10 1709, i7-10750H, torch/onnxruntime 均可用
- **YOLOv8n > 纯CV**: 车辆检测精度远超背景减除法
- **HyperLPR3 HIGH > 默认**: 大车牌 OCR 准确率极高
- **脚本文件 > 内联Python**: 可靠、可调试、可落盘
- **config.yaml 集中参数**: 一处改参即可调（这个设计保留）

---

## 3. 目标架构总览（分层设计）

### 3.1 六层架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    L6: 应用层 (Application)                   │
│    CLI 入口 / Web API / 批处理调度器                           │
├─────────────────────────────────────────────────────────────┤
│                    L5: 任务编排层 (Orchestration)              │
│    Pipeline DAG / 状态机引擎 / 结果聚合器                      │
├───────────────┬───────────────┬───────────────────────────────┤
│  L4a: 推理层   │  L4b: 训练层   │  L4c: 评测层                  │
│  模型推理引擎   │  模型训练工具   │  评测指标计算器               │
├───────────────┴───────────────┴───────────────────────────────┤
│                    L3: 模型层 (Model Zoo)                      │
│    YOLOv8n / HyperLPR3 / CV模块 / 未来扩展模型                 │
├─────────────────────────────────────────────────────────────┤
│                    L2: 数据管道层 (Data Pipeline)              │
│    视频读取 / 抽帧 / 预处理 / 数据增强 / 格式转换               │
├─────────────────────────────────────────────────────────────┤
│                    L1: 基础设施层 (Infrastructure)             │
│    Config / Logging / Git / CI / 路径管理 / 依赖管理            │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 各层职责定义

| 层级 | 名称 | 职责 | 关键组件 |
|------|------|------|----------|
| **L1** | 基础设施 | 配置、日志、路径、依赖、Git钩子 | `config/`, `logging`, `paths`, `requirements` |
| **L2** | 数据管道 | 视频→帧的输入管道；数据增强；格式标准化 | `data_pipeline/video_reader.py`, `frame_sampler.py`, `augmentor.py` |
| **L3** | 模型层 | 所有检测/识别模型的封装和抽象 | `models/yolo_detector.py`, `plate_recognizer.py`, `cv_crosswalk.py`, `cv_traffic_light.py` |
| **L4a** | 推理引擎 | 模型推理的统一接口、批处理、缓存 | `inference/engine.py`, `batch_processor.py` |
| **L4b** | 训练工具 | 模型微调/训练的工具链 | `training/trainer.py`, `dataset_builder.py` |
| **L4c** | 评测工具 | 标准指标计算、对比报告、可视化 | `evaluation/metrics.py`, `evaluator.py` |
| **L5** | 任务编排 | Pipeline DAG、违规判定状态机、结果聚合 | `pipeline/dag.py`, `violation_engine.py` |
| **L6** | 应用层 | 用户交互入口 | `cli/main.py` |

---

## 4. 目标目录结构

```
project_root/
├── configs/                    # 配置文件
│   ├── config.yaml            # 主配置（运行时参数）
│   └── model_registry.yaml    # 模型注册表（权重路径/版本）
│
├── docs/                       # 文档（新增）
│   ├── plans/                 # 设计文档
│   └── lessons/               # 经验教训
│
├── data/                       # 数据（重组）
│   ├── raw/                   # 原始视频（只读）
│   ├── processed/             # 处理后（帧/标注/证据）
│   ├── output/                # 运行输出（视频/报告/截图）
│   └── cache/                 # 缓存
│
├── datasets/                   # 数据集管理（新增）
│   ├── train/                 # 训练集 (images + labels)
│   ├── val/                   # 验证集
│   ├── test/                  # 测试集
│   └── metadata.json          # 数据集元信息
│
├── models/                     # 模型权重
├── src/                        # 源码（重组为分层包）
│   ├── infrastructure/        # L1: 基础设施
│   ├── data_pipeline/         # L2: 数据管道
│   ├── models/                # L3: 模型层（统一BaseModel接口）
│   ├── inference/             # L4a: 推理引擎
│   ├── training/              # L4b: 训练工具（预留骨架）
│   ├── evaluation/            # L4c: 评测工具 ⭐
│   ├── pipeline/              # L5: 编排（DAG+状态机+可视化）
│   └── app/                   # L6: 入口
│
├── tests/                      # 测试（新增）⭐
│   ├── unit/                  # 单元测试（每个模块至少1个）
│   ├── integration/           # 集成测试
│   ├── fixtures/              # 测试fixture
│   └── conftest.py            # pytest配置
│
├── scripts/                    # 工具脚本（精简）
├── .gitignore
├── .pre-commit-config.yaml     # Pre-commit hooks
├── pyproject.toml              # 项目元信息
└── CHANGELOG.md                # 变更日志
```

---

## 5. L2-L3 详细设计要点

### 5.1 统一模型接口（BaseModel ABC）
```python
class BaseModel(ABC):
    """所有模型的统一接口 —— 关键抽象"""
    @abstractmethod
    def load(self, weights_path=None) -> None
    @abstractmethod
    def infer(self, image: np.ndarray) -> ModelResult
    @abstractmethod
    def get_info() -> ModelInfo  # name/version/classes/input_size
    def benchmark(self, n_runs=100) -> BenchmarkResult
```
**为什么重要?**: 当前 YOLO 返回 dets dict、HyperLPR3 返回 [text,conf,type,box]，格式不同导致评测和替换困难。统一接口让评测和 A/B 测试变得简单。

### 5.2 FrameSampler 策略升级
```python
class FrameSampler:
    STRATEGIES = ["fixed_fps", "uniform", "keyframe", "scene_change"]
```
当前固定 `fps=8`，新设计支持策略切换。场景变化检测可在画面突变时触发额外采样。

### 5.3 ViolationEngine V2（多档灵敏度）
```python
SENSITIVITY_PRESETS = {
    "strict":   {"speed": 15, "sustain": 8, "duration": 8, "overlap": 0.30},
    "balanced": {"speed": 30, "sustain": 5, "duration": 5, "overlap": 0.20},
    "loose":    {"speed": 50, "sustain": 3, "duration": 3, "overlap": 0.15},
}
```
实测发现 speed=15 对蠕行太严（违章01 全 stop=0）。V2 支持**滑动窗口速度计算**（抗抖动），并预设三档灵敏度供评测选择最优。

### 5.4 Pipeline DAG（替代 monolithic pipeline.py）
将当前 165 行线性函数拆分为独立节点:
```
read_frame → sample → detect_vehicle → track → [crosswalk/light/plate] → evaluate → visualize → output
```
每个节点可单独 mock 测试、替换、并行化。

---

## 6. L4c: 评测工具 + 标准指标 ⭐

### 6.1 四类指标体系

#### 目标检测指标（车辆/斑马线/红绿灯）
| 指标 | 含义 |
|------|------|
| Precision | 检出的里面多少是真的？TP/(TP+FP) |
| Recall | 真实的多少被检出了？TP/(TP+FN) |
| F1-Score | P/R 的调和平均 |
| mAP@0.5 | COCO 标准 IoU≥0.5 的 AP 均值 |
| mAP@0.5:0.95 | 更严格的 mAP |

#### 序列级指标（违规事件检测）
| 指标 | 含义 |
|------|------|
| Event P/R/F1 | 事件级的精确率/召回率/F1 |
| Temporal IoU | 检出时间段与真实时间段的交并比 |

#### OCR 指标（车牌识别）
| 指标 | 含义 |
|------|------|
| Char Accuracy | 字符级正确率 |
| Plate Accuracy | 完全匹配车牌的正确率 |
| Edit Distance | Levenshtein 近似度 |
| Province Accuracy | 省份识别专项准确率 |

#### 系统级指标
| 指标 | 含义 |
|------|------|
| Throughput (fps) | 推理速度 |
| Latency p99 | 99分位延迟 |
| End-to-End Accuracy | 视频→违规事件的端到端正确率 |

### 6.2 Ground Truth Schema（标注格式）
```json
{
  "video_id": "违章02",
  "frame_id": 1500,
  "annotations": {
    "vehicles": [{"id": 1, "bbox": [x1,y1,x2,y2], "label": "car"}],
    "traffic_light_state": "red",
    "license_plates": [{"vehicle_id": 1, "plate_text": "京A12345"}]
  },
  "violations": [{"track_id": 1, "start_ts": 45.0, "end_ts": 55.0}]
}
```

---

## 7. 测试策略 ⭐

### 金字塔测试模型
```
        ╱╲       E2E Test (完整视频→报告, 发布前必跑)
       ╱  ╲
      ╱Integration╲  (模块间协作验证)
     ╱──────────────╲
    ╱    Unit Test    ╲  (每个函数, 提交前必跑 pytest)
   ╱  (pytest, 快速)   ╲
  ╱────────────────────╲
```

### 必须覆盖的测试矩阵
| 模块 | 测试类型 | 最少 Case 数 | Fixture |
|------|----------|-------------|---------|
| VideoReader | Unit | 3 (open/metadata/seek) | 5s测试视频 |
| YoloDetector | Unit | 2 (车辆图/无车图) | 固定图片 |
| PlateRecognizer | Unit | 3 (近景/远景/侧角) | 多尺寸车牌图 |
| CrosswalkDetector | Unit | 2 (有条纹/无条纹) | 合成图 |
| TrafficLightDetector | Unit | 3 (红/绿/未知) | 纯色块 |
| Tracker | Unit | 2 (稳定轨迹/遮挡) | 序列数据 |
| ViolationEngine | Unit | 3 (全满足/缺一/全不满足) | 构造states |
| Evaluator | Unit | 2 (完美预测/全错预测) | 构造pred+gt |
| Full Pipeline | Integration | 1 | 违章01或02片段 |

---

## 8. Git Commit 规范 ⭐

Conventional Commits 格式：`<type>(<scope>): <subject>`
- Types: `feat` / `fix` / `refactor` / `test` / `docs` / `chore` / `perf`
- Scopes: 对应层级名 (`models` / `evaluation` / `pipeline` / ...)
- 强制规则: 重大修改必须立即 commit; 不 commit `__pycache__/data/output/cache/`

---

## 9. 泛化性保障 ⭐

| Overfitting 风险 | 防护措施 |
|------------------|----------|
| 只在这几个视频上调参 | 建立独立评测集(≥3个视频)，严格分离调参与评测 |
| 斑马线参数只适配当前路口 | 多路口/多时段验证; 参数按场景分组 |
| 光照/天气单一 | 数据增强模拟不同光照条件 |
| 车型分布偏差 | 数据集分析类别分布; 确保 car/bus/truck/suv 都有样本 |
| 车牌大小敏感 | 多分辨率分别评测 |

**评测集建设原则**: 至少 3 个不同地点/机位的视频; 至少 1 个含明确违规案例; 至少 1 个正常交通负样本; 评测集绝不用来调参。

---

## 10. 迁移计划

| Phase | 内容 | 交付物 | 预估时间 |
|-------|------|--------|----------|
| **A** | 基础设施搭建 | 新目录结构+迁移代码+基础测试+.gitignore | Day 1 |
| **B** | 接口统一 | BaseModel ABC + InferenceEngine + 模型包装 | Day 2 |
| **C** | 评测工具 | metrics.py + Evaluator + baseline评测 | Day 3 |
| **D** | Pipeline DAG | 拆分monolithic为DAG + ViolationEngineV2 | Day 4 |
| **E** | 调参优化 | 用评测指导选balanced preset + 多视频验证 | Day 5 |

每 Phase 完成后立即 git commit。Phase 可独立交付价值。

---

## 11. Grill-Me 自我压力测试 🔥

### Q1: 六层架构是否过度设计？（杀鸡用牛刀？）
**自辩**: 当前核心问题不是"模块太少"，而是**没有边界**——pipeline.py 是 God Object，改任何东西都可能影响全局。6层的价值在于**清晰的职责边界**。
**裁决**: ✅ 合理。但 L4b(训练) 和 L6(API) 先只建空壳+接口定义，避免 YAGNI违反。

### Q2: DAG 是否比顺序调用更好？
**自辩**: 价值在于**可测试性**和**可替换性**。无法单独测"给定 states，violation engine 怎么判"。自制轻量 DAG（字典+拓扑排序），不引第三方库。
**裁决**: ✅ 自制轻量 DAG。

### Q3: GT 标注工作量是否现实？
**自辩**: 务实方案: 第一版用**合成数据**(已知GT); 第二版逐步标注真实视频(优先已确认含违规的片段); 第三版考虑半自动标注(预标注+人工校正)。
**裁决**: ⚠️ **需用户确认投入**。如不想标注，合成数据是最小可行路径。

### Q4: 5天迁移是否乐观？
**自辩**: Phase A-D 是结构性迁移(搬文件+写接口+写测试)，不需新算法。Phase E 才是需要迭代的调参。理想5天，实际可能7-10天。以 Phase 为单位交付，降低风险。
**裁决**: ⚠️ Optimistic estimate。建议逐 Phase 交付。

### Q5: CPU 能支撑评测吗？
**自辩**: 评测只是 pred vs GT 的数学运算，不需 GPU。推理瓶颈已在跑(6.9fps)。评测几乎不加额外时间。
**裁决**: ✅ 完全支撑。

### Q6: HyperLPR3 作为黑盒如何保证泛化？
**自辩**: 评测工具的价值正是**量化它在不同场景的表现**。PlateAccuracy < 60% 就需要收集该场景数据并微调或更换。BaseModel 接口隔离保证了**随时可换成 CRNN/PaddleOCR/其他方案**。
**裁决**: ✅ 接口隔离保证可替换性，评测数据告诉我们何时需要换。

---

## 12. 待用户确认事项

| # | 问题 | 选项 | 我的推荐 |
|---|------|------|----------|
| D1 | GT标注策略? | A)纯合成 B)合成+部分真实 C)全量人工 | **B)** 平衡成本质量 |
| D2 | 迁移节奏? | A)一次全量 B)逐Phase交付 | **B)** 降低风险 |
| D3 | 要API入口? | A)仅CLI够用 B)预留API骨架 | **A)** YAGNI原则 |
| D4 | 评测集规模? | A)最少2个(1正1负) B)5个以上 | **A)** 先最小可行 |
| D5 | 先做覆盖率? | A)是(pytest-cov) B)先跑通再说 | **B)** 先跑通 |

---

> **下一步**: 用户 review 本文档 → grill-me 质疑/修改 → 批准 → 进入 Phase 2 (Writing Plans) → Phase 3 (TDD Build)

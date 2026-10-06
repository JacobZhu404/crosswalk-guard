# Phase A 重训数据集方案: 行人灯 vs 环境绿判别器

> **署名**: wb(执行者)  · **收件**: cc(plan-gate) / Jacob(拍板 Phase B/C)  
> **红线**: 不改生产 prior/权重/`_sample_roi`/`ped_signal.pt`; 独立 worktree; 不 push/merge。

---

## 1. 目标

训练一个小型图像分类器 `SignalDiscriminator`, 输入为生产 prior ROI 裁图(与生产 `_sample_prior_color` 同范围), 输出 3 类:

- `green_signal`: 该 ROI 中心区域包含**行人信号灯且为绿**;
- `red_signal`: 该 ROI 中心区域包含**行人信号灯且为红**;
- `no_signal`: ROI 中无行人信号灯, 或灯灭, 或仅有环境绿/车辆绿/反光(重点覆盖 [85-106] 类假绿)。

部署时作为 `_sample_prior_color` 的第二意见: 即使颜色 vote 返回 green, 若判别器输出 `no_signal`, 则当前帧应降级为 unknown/off, 从而抑制 [85-106] 类持续假绿。

---

## 2. 训练样本来源与标注口径

### 2.1 素材来源

全部 11 个本地视频(`input_video/违章*.mp4`, gitignored)。以生产 `light_priors.json` 中的 per-video 坐标 `(cx,cy)` 为准, 每帧截取与 `_sample_prior_color` 相同的 **160×160 ROI** 作为分类器输入。

**不使用** `ped_signal.pt` 及其训练数据; 避免把已发现的 overfit 带入新模型。

### 2.2 时间标签(弱监督)

以 `datasets/gt/light_states.csv` 的 frame-level 状态作为时间标签:

| GT 状态 | 分类标签 | 用途 |
|---|---|---|
| `green` | `green_signal` | 正例: 真实行人灯绿 |
| `red` | `red_signal` | 正例: 真实行人灯红 |
| `unknown` / `occluded` / `off` / `tentative` | `no_signal` | 灯态不可知或已灭; 重点 hard negative |
| `None`(未标注) | **丢弃** | 避免引入 label noise(05 `[30,65.7]` 即属此类) |

### 2.3 Hard Negative 增强

除了按 GT 时间标签采的 `no_signal` 外, 额外加入 Phase A mining 中发现的 **color-vote 假绿帧**:

- 09 `[72.7-106.1]` 中 color-vote 输出 green 但 GT=unknown 的 34 帧(含车辆绿、车身绿);
- 01/02/03/04/06/07/10/11 definitive 假绿帧约 60 帧;
- 这些帧明确告诉模型: "HSV 颜色 vote 说 green, 但你应该判 no_signal"。

### 2.4 视觉抽检锚点

对 hard negative 做人工抽检(已抽取 `data/output/mine_false_green/crops/`), 按来源分为子标签(内部使用, 不进入模型标签):

- 车辆绿(taxi / 私家车 / 绿牌反光)
- 树叶绿(树冠/路边绿植)
- 交通绿牌/路牌
- 反光/白平衡泛绿

用于分析模型错误模式和后续针对性增广。

---

## 3. 数据规模与采样策略

### 3.1 基础采样

当前 Phase A 用 1fps, 全 11 视频约 700 帧。训练时需要更密集帧 + 增广:

| 配置 | 帧率 | 预估原始帧 | 增广后 |
|---|---|---|---|
| 保守 | 1fps | ~700 | ~3,500(5x) |
| 推荐 | 3fps | ~2,100 | ~10,500(5x) |
| 激进 | 6fps | ~4,200 | ~21,000(5x) |

**建议采用 3fps**: 兼顾时序多样性与避免相邻帧高度相关。

### 3.2 类别平衡

按标签统计(以 3fps 估算):

- `green_signal`: ~350 帧(GT green 窗口集中在 violation 段)
- `red_signal`: ~400 帧(GT red 窗口与部分 true-negative 视频)
- `no_signal`: ~1,350 帧(GT unknown + hard negative)

`no_signal` 占多数, 符合生产场景(灯灭/环境绿远多于亮灯)。采用 **class-weighted loss** 或 **按类别采样** 使每批 balance, 重点上采 hard negative(09 车辆绿、树叶绿)。

### 3.3 数据增广

- 随机 crop ±10% 模拟 prior 漂移(不模拟大漂移, 因为生产 prior 是固定的);
- 亮度 ±20%、对比度 ±15%、高斯噪声;
- 轻微旋转 ±5°;
- 水平翻转仅限对称灯形(训练时排除翻转 red_signal 除非数据量大, 避免红绿方向混淆);
- 颜色抖动 H/S/V 在合理范围, 但不能改变"灯 vs 环境"的本质。

---

## 4. 模型选型

### 4.1 推荐: MobileNetV3-Small 微调

- 预训练 ImageNet 权重(轻量, CPU/GPU 均可);
- 替换最后分类头为 3 类;
- 输入 160×160×3;
- 训练 30-50 epoch, early stopping on 09 验证 recall;
- 推理耗时 <5ms/帧, 不拖慢 per-frame pipeline。

### 4.2 备选: 轻量自训练 CNN

若避免引入外部权重或网络限制: 3-4 层 Conv + GAP + Dense, 约 100k-300k 参数。需要 10k+ 样本才能稳定, 因此更依赖 6fps 采样。

### 4.3 输出与阈值

输出三 logits; 部署时:
- `green_signal` P ≥ 0.7 → 可 trust 绿;
- `red_signal` P ≥ 0.7 → 可 trust 红;
- `no_signal` P ≥ 0.7 → 即使颜色 vote 为 green, 也降级为 off/unknown。
- 三类别均低置信 → 回退当前颜色 vote(避免新模型不确定时引入新错误)。

---

## 5. 训练/验证划分

### 5.1 严禁随机划分

相邻帧高度相关, 随机划分会泄漏。采用 **leave-one-video-out (LOVO)**:

- 每次留 1 个视频作验证, 其余 10 个训练;
- 重点关注 09 作验证时的 recall(能否把 09 `[72.7-106.1]` 的 no_signal 召回为 no_signal, 同时不压降 `[11,72]` 的 green_signal)。

### 5.2 关键验收指标

- **09 硬负例 recall**: 在 09 `[72.7-106.1]` 的所有 hard negative 帧上, `no_signal` 预测比例 ≥ 90%。
- **11 视频 green/red 真值 recall**: GT green/red 窗口内, 对应 `green_signal`/`red_signal` 比例 ≥ 95%(不能为了抑制假绿而屠真绿)。
- **零新增记分 FP**: 若接入生产, 全 11 视频端到端 F1 ≥ 0.941 / P = 1.000 须保持。

---

## 6. 数据产出物

Phase B 重训前需产出并 commit:

| 产物 | 路径 | 说明 |
|---|---|---|
| 训练索引 CSV | `data/output/discriminator_train/train_index.csv` | 字段: video, frame_idx, prior_x, prior_y, label, split |
| 样本图块 | `data/output/discriminator_train/crops/` | 160×160 PNG, gitignored, 仅本地使用 |
| 类别统计 | `data/output/discriminator_train/class_stats.json` | 每类数量、来源分布 |
| hard negative 清单 | `data/output/discriminator_train/hard_negatives.json` | 09/06/01 等假绿帧元数据, 含来源子标签 |

因 `data/output/` gitignored, commit 的只有训练脚本和索引生成器; 实际图块随运行生成。

---

## 7. Phase B/C 排期(待 Jacob 拍板)

### Phase B: 重训(本 worktree)

1. 按 3fps 生成训练索引 + 提取 160×160 crops;
2. LOVO 训练 MobileNetV3-Small;
3. 调阈值, 重点看 09/06 假绿 recall 与 11 视频真绿/真红 recall;
4. 产出生成 `signal_discriminator.pt`。

### Phase C: 接线(新 worktree, 等 cc 效果 gate)

1. 修改 `TrafficLightDetector._sample_prior_color` 接入 discriminator 第二意见(保留颜色 vote fallback);
2. 跑全 11 视频端到端 gate: F1≥0.941 / P=1.000 且 09 记分无新增 FP;
3. cc 验收通过后由 Jacob 拍板 merge。

---

## 8. 红线与风险

- **不碰生产 prior 坐标**: 训练输入仍用生产 `light_priors.json`, 不改坐标。
- **不碰 `_sample_roi` 颜色阈值**: 训练集归一化用独立参数, 生产阈值不变。
- **ped_signal.pt 不入库**: 新模型权重 `signal_discriminator.pt` 在 gate 通过前不入主仓。
- **GT 不进生产推理**: 训练/评估用 GT, 部署时只用 discriminator 输出。
- **最大风险**: 新模型在 04/11 等"短暗绿"或"遮挡绿"上过度压低 recall, 反而漏真绿。需 LOVO + 严格 recall 验收控制。

Co-Authored-By: 白板 <wb@crosswalk-guard.agents>

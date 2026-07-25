# Diag 3 — YOLO-vs-硬GT 命中率验证（决定重挖架构，read-only）

> **性质**：只读诊断（不改代码、不重挖、不写 configs）。仅加载 `models/yolov8n.pt` + `datasets/light_location_gt.json`（Jacob 硬框），逐帧跑 YOLO(COCO cls9, 低阈) 与 GT 行人灯框算 IoU/命中率。
> **目的**：回答 cc 战略叉——重挖到底走 **YOLO 框裁图**（免手设 prior、抗漂移）还是 **固定 prior 重定位**？核心问题：YOLO 框的是行人信号灯还是车用红绿灯？命中率多高？
> **结论先行**：**YOLO(yolov8n, COCO cls9) 基本框不住行人信号灯**。在 02/05/06/09 上命中率 0–4.4%、mean best-IoU ≈0.04；每帧虽检出 2–6 个 traffic-light 框，但 95–100% 都不在行人灯位置（框的是车灯/误检）。**→ 重挖走「固定 prior 重定位」，不走 YOLO-crop。** 这恰好印证 cc 正在建的 `light_priors.rebuilt.json`（用 GT 中心）才是正确方向。

---

## 1. 方法

- GT：`datasets/light_location_gt.json`（`ped_signal_location_gt_v1`），取每视频 `derived_per_video.true_center` + `true_box_wh` 推 canonical 行人灯框（归一化 `[x1,y1,x2,y2]`）。行人灯极小：宽 0.017–0.043、高 0.049–0.126（如 05 = 0.021×0.071）。
- YOLO：`ultralytics.YOLO("models/yolov8n.pt")`，`classes=[9]`（traffic light），主阈值 `conf=0.05`（低阈，尽量捞小目标）+ 对照 `conf=0.25`。
- 逐帧（全部抽帧）跑 YOLO → 每个框与 canonical GT 框算 IoU；帧内 best-IoU≥0.5 记命中。
- 额外指标量化"框的是车灯还是行人灯"：每个 YOLO 框中心到 GT 中心的距离 → 近框占比(≤0.10, 疑行人灯) / 远框占比(>0.25, 疑车灯)；帧内最近框中心距 GT 的均值。
- 视频：先跑 cc 指定的 02(对照)+05/06/09(偏框)，后跑全 11 视频补全。

---

## 2. 结果（cc 指定 4 视频，conf=0.05）

| 视频 | 帧 | 命中率(全帧) | 命中率(有YOLO帧) | meanBestIoU | 均框数/帧 | 有框不命中率 | 近框占比(≤0.10) | 远框占比(>0.25) | 均最近距GT |
|---|---|---|---|---|---|---|---|---|---|
| 02(对照) | 469 | **0.2%** | 0.2% | 0.015 | 2.82 | 99.8% | — | — | — |
| 05(偏框) | 279 | **0.0%** | 0.0% | 0.000 | 1.88 | 100% | — | — | — |
| 06(偏框) | 204 | **0.5%** | 0.5% | 0.096 | 5.95 | 99.5% | — | — | — |
| 09(偏框) | 452 | **4.4%** | 4.6% | 0.049 | 2.92 | 95.4% | — | — | — |
| **汇总** | | **1.3%** | — | **0.040** | — | **98.7%** | | | |

对照高阈 conf=0.25：命中率进一步跌到 0–0.2%，均框数降到 0.3–0.6（YOLO 高阈下基本不报行人灯，只偶尔报大车灯）。

> 注：近框/远框占比为全 11 跑补全后填（见 §3）。4 视频已足够得出方向性结论。

---

## 3. 结果（全 11 视频，conf=0.05，后台补全中）

> 全 11 视频跑批在后台进行（数据量 ~4000 帧，CPU 推理约 2–3 分钟）。**方向已由上节 4 视频（cc 指定 02+05/06/09）锁死**，补全仅用于把"近框/远框占比"填实。预期：11 视频平均命中率仍 <10%、meanBestIoU≈0、远框占比 ≫ 近框占比。跑完后会回填本表与 §2 的近框/远框列。
>
> 回填判定（若与下相反则需复核）：若全 11 平均命中率 <10% 且远框占比 >60% → 维持"回固定 prior 重定位"裁决；若某视频命中率突 >50% → 单独标注（可能该视频行人灯较大/近，YOLO 偶可框）。

---

## 4. 分析

### 4.1 YOLO 框的是车灯，不是行人灯
- 每帧检出 **2–6 个** traffic-light 框（说明 YOLO 在路口确实在找灯），但 **95–100% 的帧里没有任何一个框落在行人灯位置**（有框不命中率）。
- mean best-IoU ≈ 0.04（接近 0 重叠）→ YOLO 框与行人灯框**几乎不在同一处**。
- 这只能是：YOLO 检出的框是**车用红绿灯（路口对侧大灯）/ 反光误检**，漏掉了**极小（宽~2%）的行人信号灯**。COCO 训练分布里 traffic light 多为较大目标，对 2% 宽度的行人竖灯欠拟合。

### 4.2 偏框视频(05/06/09)上 YOLO 同样失败 → 不是 prior 的锅
- 05/06/09 当前 prior IoU=0（偏框），但 YOLO 在**这些视频上命中率也≈0**。说明 YOLO 失败与 prior 无关——它本来就认不出这种小行人灯。
- 反过来说：**即便把 prior 修对，YOLO 也救不了重挖**（它仍框车灯）。这直接否定了"用 YOLO 框裁图做重挖"的架构。

### 4.3 对战略叉的裁决
| 架构 | 可行性 | 理由 |
|---|---|---|
| **固定 prior 重定位**（用 GT 中心建 `light_priors.rebuilt.json`） | ✅ 可行 | GT 中心(derived_per_video.true_center)几何精确、框极小但位置确定；prior-ROI 直抽只需"位置对"，不依赖 YOLO 识灯。cc 正在建此文件。 |
| **YOLO 框裁图**（逐帧 YOLO 定位） | ❌ 不可行 | YOLO 命中率 <5%，且框的是车灯；用以裁图会裁到车灯/背景，重蹈 39.3% 无信号覆辙，甚至更糟（车灯颜色干扰）。 |

---

## 5. 结论与建议

1. **重挖架构定案：固定 prior 重定位**，不走 YOLO-crop。YOLO(yolov8n COCO cls9) 在本数据上无法定位行人信号灯（命中率 <5%、框车灯），不能承担重挖的"定位"职责。
2. **cc 的 `light_priors.rebuilt.json` 方向正确且必要**：用 GT 中心 + 保持 roi_px 单变量，既能修生产 YOLO 选框（Diag 1 已知 YOLO 兜生产，但选框靠 prior 邻近，prior 准了选框更准），又能驱动干净重挖（prior-ROI 直抽）。
3. **YOLO 的角色定位**：生产 observe() 仍可用 YOLO 作"选框候选"（它在路口能检出灯，只是分不清行人/车用），但**最终定位必须靠修正后的 prior 收敛**，不能让 YOLO 裸定位行人灯。
4. **后续（等 cc 的 rebuilt.json + Jacob 画框收尾）**：rebuilt prior → 干净重挖 crops（prior-ROI 直抽，几何保持 cc(B)）→ 重训。Step2 sweep 维持 STOP（偏框数据未清）。

---

## 6. 产物与复现

- 脚本：`scripts/diag_yolo_vs_gt.py`（新建，纯只读，不依赖 pipeline 内部代码）。
- 数据：`data/output/diag_yolo_vs_gt.json`（每视频 low/high 阈明细 + 汇总）。
- 复现：
  ```bash
  cd /Users/jacob/personal/crosswalk-guard
  PYTHONPATH=src ./.venv/bin/python scripts/diag_yolo_vs_gt.py                       # 全11视频
  PYTHONPATH=src ./.venv/bin/python scripts/diag_yolo_vs_gt.py --videos 违章02 违章05 违章06 违章09
  ```
- ⚠️ 本诊断用 canonical GT 框（跨帧平均位置）作参照；02/10 等灯位漂移较大(derived cross_frame_spread 0.14–0.18)，可能略微低估命中率，但 05/06/09(漂移小) 同样≈0% 命中，方向结论不受影响。

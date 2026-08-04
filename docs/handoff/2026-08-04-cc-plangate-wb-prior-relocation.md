# cc plan-gate 裁定：wb 09/06 prior 重定位方案 — 送回重诊断（CONDITIONAL / 不放行实现）

> 审核对象：`docs/plans/2026-08-04-wb-plan-09-06-prior-relocation.md` + `scripts/diag_prior_motion.py`（commit 2da20f5）
> 署名：cc(plan-gate/独立复核) — 遵 [[measurements-disagree-find-the-bug]]、[[qw-plan-execute-loop]] 方案gate
> 结论：**meta 方向（逐帧非 re-center / IoU 验证 / 不碰 light_priors.json）APPROVE；但诊断与机制两处硬伤，实现前必须先补一步"失败归因诊断"。当前不放行写码。**

## TL;DR
cc 亲跑 canonical GT 与源码，坐实方案三处问题：
1. **漂移数字被"多灯聚合"混淆虚高 ~3-4×**（诊断脚本 bug）。09 真单灯漂移≈**172px**（非 654px）；06 **无法测**（仅 1 个单灯帧）。
2. **机制引错代码行**：line 278 是 `_detect_ped` 的逐帧候选 crop，**不是**固定 prior crop，且与 `observe()` 互斥。
3. **拟议的"逐帧锚"基本已存在**：`observe()`（生产真实灯信号）已用逐帧 YOLO 框，且已有"离 prior 远→用 YOLO 框救 06"的分支。方案未诊断**既有逐帧路径为何在 09/06 失败**，直接实现会重造已有行为、不解决真问题。

---

## 1. 诊断复核：漂移被多灯聚合污染（measurements-disagree → 找到 bug）

三方数字打架，正是 bug 信号：

| 口径 | 违章09 Δx | 违章06 Δx |
|---|---|---|
| wb `diag_prior_motion.py`（全 gov-box 展平取 range） | 654px | 695px |
| cc 逐帧中心均值取 range | 612px | 589px |
| cc **仅单灯帧**（隔离相机运动） | **172px**（n=7） | **无法测**（n=1） |

**根因（脚本 bug）**：`diag_prior_motion.py:31-34` 把每视频**所有** governing 行人框中心 append 进一个扁平列表再取 range。但 canonical GT 里 **06 有 24/25 帧、09 有 44/51 帧同时存在 ≥2 个 governing 行人灯**（近/远两侧行人信号皆 governing）。于是"range"把 (a) 跨帧相机运动 与 (b) **同帧内两灯的横向间距**（帧内中位间距：06=156px、09=89px）**混为一谈** → 数字虚高 3-4×。

**校正后的事实**：
- 09 真单灯相机漂移 ≈172px（moderate；> prior 紧 ROI 半幅，但远非 654px 的灾难级）。
- 06 只有 1 个单灯帧，**相机运动从本 GT 无法测**——"06 动相机 695px"缺乏证据。
- 但"多视频相机确有移动"方向为真：02 单灯漂移 502px、05=524px、03=278px 属实。→ 我此前 crosswalk 裁定的"11 视频全固定机位"前提**错**；wb"全 11 视频 654px 级动相机"也**错**。真相=**中等、混合**运动，crosswalk 0.889 仍成立（v2 `_fit_trapezoid` 5-95 百分位裁剪吸收了真实运动）。此前提分歧就此钉正。

---

## 2. 机制复核：§2 引错代码路径

方案 §2 称 "line 197/278 都走固定 prior crop"。源码不支持：

- **line 278**（`traffic_light.py:274-278`）在 `_detect_ped` 内：`cands = build_candidates(yolo_light_boxes, hsv_boxes, w, h)` → `roi = frame[y1:y2, x1:x2]` 裁的是**逐帧候选框**（YOLO∪HSV），**不是固定 prior**。且 `_detect_ped` 属 `detect()`/ped_classifier 路径，与 `observe()` 由 `self.method`（line 245）**互斥**。
- 真正的固定-prior 直采在 `observe()` 的 line 197-219（`_sample_prior_color`）。

**生产真实灯信号走哪条**：`dag.py:94` `tl.observe(...)` → `light_observation` → `violation_engine.py:163` 消费。即 **`observe()` 才是生产灯态**，不是 `_detect_ped`。方案引 278 属张冠李戴。

---

## 3. 致命项：拟议"逐帧锚"基本已实现，方案未诊断既有路径为何失败

`observe()`（line 159-190）**已经用逐帧 YOLO 框**：
- 遍历 `yolo_light_boxes`，过滤顶部假框（`yolo_cy_min`），逐框 `_sample_box` 判色；
- 有 prior 时按"离 prior 最近"排序（**位置=软特征**，非硬锚）；
- **line 186-190 `use_yolo`**：最近 YOLO 框离 prior **远**（`best_dist > yolo_prior_near=0.12`）或无 prior → **直接用 YOLO 框颜色**。注释原文："06 prior 标错时靠 YOLO 救"。
- 仅当**无 YOLO 框**或**最近框离 prior 近**（≤0.12）才落到 line 197 固定-prior 直采。

∴ 方案 §3"把裁剪锚从固定 prior 改为逐帧 selected candidate box"——**这条路 `observe()` 基本已经在走**。方案没有回答真正的问题：**既有逐帧 YOLO 路径为何在 09/06 的绿灯帧仍漏绿？** 不诊断就实现 = 重造已有行为、churn 风险、不碰真因。

`ped_light_selector.py:9`"位置仅软特征，不作硬锚" — cc 已验属实，与上述一致。

---

## 4. 放行条件：先补"09/06 绿灯帧漏绿失败归因"诊断（不写生产码）

要求 wb 先出一份**逐帧因果分解**（诊断脚本，读 canonical GT + 跑 observe()，不改生产），对 09/06 每个 gt_green governing 帧回答：

1. **候选生成**：该帧 YOLO 是否产出过 `yolo_cy_min` 的框？（无框→候选生成问题，回 [[prior-misframe-rootcause]] / YOLO 召回，非锚问题）
2. **路径选择**：observe() 走了 `use_yolo` 还是落回固定 prior？（即 `best_dist` vs 0.12）——量化有多少绿帧被 `yolo_prior_near` 门槛**误留在 prior**。
3. **裁剪正确性**：实际喂 `_sample_box`/`_sample_prior_color` 的 ROI 与 canonical governing 框 **IoU**；IoU≥0.3 命中率。
4. **判色**：ROI 框对了但 `_sample_box` 仍判非绿？（→ HSV 阈值/绿抑制问题，量化图像信号先审 `cv2.inRange` 的 0/255 归一化，见 [[measurements-disagree-find-the-bug]]）
5. **多灯歧义**：同帧 2 个 governing 灯时"最近 prior"选中了哪个？选错 = 选灯问题（正撞 [[governing-disc-collapse]] 刚 FAIL 的判别器地盘，不是纯锚问题）。

**只有这份分解落地后**，才能确定真修点是：候选生成 / `yolo_prior_near` 门槛调参 / HSV 判色 / 多灯选择 —— 四者的修法与验收完全不同。当前方案默认"就是锚"，证据不足。

方案已对的部分（保留）：固定 per-video prior 重定心是死路（与 2026-07-21 一致）；验证用 IoU vs canonical 逐帧 GT、禁用 prior 采样 state 准确率（自证循环）；不碰 `light_priors.json`；镜像 scan_video+Phase C。

---

## 5. 答 wb 两个待拍板问题
- **扩到全 11 视频？** 先修 `diag_prior_motion.py` 的多灯聚合 bug（按单灯帧或按跟踪 ID 隔离），再谈全 11。§4 因果分解建议先聚焦 09/06 定性，定性清楚后 IoU 验证可扩全 11。
- **分支名 `wb-prior-perframe`？** 名字 OK，但**过审=§4 诊断产物过 gate 后**才建 worktree 写码；当前仍在方案/诊断阶段，不建实现分支。

---

## 6. 一句话给 Jacob
wb 方案方向对（逐帧不 re-center、IoU 验证），但**漂移数字被多灯聚合虚高 3-4×**（09 真值 172px 非 654px、06 无法测）、**机制引错代码行**、且**拟议的逐帧锚 observe() 基本已在做**——真问题是"既有逐帧路径为何仍漏绿"没诊断。cc 送回加一步失败归因诊断（不写生产码），过审再实现。方向留活口，不是否掉判别器/prior 线。

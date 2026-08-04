# wb 方案：09/06 prior 诊断 + 重定位（per-frame 路径）

> 状态：诊断已完成；本方案待 cc **plan-gate** 审核。审核通过前不动代码。
> 署名：wb(执行/诊断/方案) / cc(复核/plan-gate/验效果)

## 0. TL;DR
- 09、06 **都是动相机**（canonical 逐帧 GT 实测灯位横向漂移 654px / 695px）。**重定心固定 prior 对两者皆死路**，与 2026-07-21 裁定一致。
- 根因不在 ranking：base 选择器 `select_gtfree` 已不依赖位置 prior（line 9）。根因在 `observe()` 仍把 **signal 裁剪 ROI 锚定固定 `signal_prior` 点**（line 197/278）→ 动相机下 crop 套空 → 漏绿。
- 重定位 = 把 signal 裁剪锚点从固定 prior **改为逐帧 selected candidate box（YOLO/tracker）**，即走架构已锁定的「逐帧定位」路径，不是 re-center 固定点。
- 验证只用**位置级 GT（IoU vs canonical 逐帧灯 GT）**，禁用 prior 处采样 state 准确率（自证循环）。

---

## 1. 诊断结论（engages 护栏#1 的第一问：09/06 到底动不动）

**方法**：用 `datasets/gt/light_canonical_gt.json`（schema `light_canonical_gt_v1`，399 帧全 11 视频逐帧行人灯框）量每个视频 governing 框中心的跨帧标准差与范围。这是 cc 指定的位置级真值（护栏#2），不是 3 帧稀疏样本。

**结果（横向漂移，W=1280 假设）**：

| video | n帧 | governing框 | cx±std | cy±std | Δx范围 | Δy范围 |
|---|---|---|---|---|---|---|
| **违章06** | 25 | 77 | 0.620±**0.111** | 0.218±0.070 | **695px** | 243px |
| **违章09** | 54 | 135 | 0.557±**0.091** | 0.283±0.035 | **654px** | 110px |
| 违章02 | 56 | 119 | 0.567±0.171 | 0.050±0.049 | 773px | 171px |
| 违章05 | 34 | 53 | 0.635±0.086 | 0.354±0.027 | 540px | 91px |
| 违章07 | 44 | 135 | 0.564±0.082 | 0.236±0.043 | 654px | 151px |

（其余视频见诊断脚本输出）

**关键判读**：
1. **09/06 均大幅移动**（654/695px 横向），远超任何固定 ROI 可吸收范围（2026-07-21 估 ~400px，实测更大）。→ **固定 per-video prior 重定心对两者都是死路**。
2. **3 帧稀疏样本是陷阱**：`datasets/light_location_gt.json` 每视频仅 3 帧标注，09 显示 ±0.044、06 仅 ±0.024——看起来很稳，但那是抽样窗口内的局部稳定，掩盖了帧间大运动。护栏#1 警告的「默认 re-center」正是会踩这个坑。已识破。
3. **与 crosswalk 线「11 视频全固定机位」表述有张力**：canonical GT 显示 09/06/02/05/07 等灯位漂移均 ≥500px，根本不是固定机位。此表述需向 Jacob/cc 钉死修正——至少 09/06 实测为动相机，重定心假设不成立。

**结论（护栏#1 第一问答案）**：09/06 是「动相机、任何固定点皆错」，**需逐帧定位**，不可 re-center 固定 prior。

---

## 2. 根因机制（固定 prior 在哪咬人）

读 `src/redlight/models/traffic_light.py`：

- `observe()`（line 140）接收 `yolo_light_boxes`，但在 line 197 `if self.signal_prior is not None and frame is not None:` 仍走 **固定 prior 点采样**：`_sample_prior_color`（line 197→605）以 `signal_prior=(px,py)` 为中心、`prior_roi_px`（默认 160）为边长取 ROI。
- line 278 `roi = frame[y1:y2, x1:x2]`（ROI 来自固定 prior 中心）→ line 279 `self.classifier.classify(roi)`。
- **即 signal 裁剪 ROI 锚定固定 `signal_prior` 点，而非逐帧检测到的灯框。**

对照选择器：`ped_light_selector.py` line 9 明确「位置仅软特征，不作硬锚（固定 prior 已证死路）」；`select_gtfree` 用 L1 几何 + YOLO 偏好 + L2 时序，**不依赖位置 prior**。→ base 选择器本身已迁移到逐帧，09 漏绿**不是** ranking 距离问题。

**链路（解释 neg_a 的「09 prior 偏框 → crop 错位 → 选不中绿」）**：
动相机下 true light 离开固定 `signal_prior` 框 → `observe()` 在固定点采到背景/树叶 → 颜色判定非绿 → 该帧 gt_green 但 best_color≠green → 漏绿。base 09 漏绿恒 22、neg_a 09 worst 29 均源于此，而非判别器质量。

---

## 3. 重定位方案（per-frame，非固定 re-center）

**核心改动**：把 signal 裁剪锚点从固定 `signal_prior` 改为**逐帧 selected candidate box**（即 `select_gtfree` 选中的 YOLO/tracker 框）。

- 在 `observe()` 的产绿/裁剪路径（line 197/278 区域），用当前帧 `yolo_light_boxes` / `select_gtfree` 选中的 governing 框作为裁剪中心与 ROI，替代 `signal_prior` 固定点。
- 固定 `signal_prior` 仅在「逐帧检测完全失效」时作 fallback（沿用 2026-07-21 裁定：prior 仅静态段/YOLO 失效 fallback），不作为主路径。
- **镜像进 `scan_video` + Phase C**（working-memory 红线：重定心须镜像进 scan_video+Phase C 才上，否则 train/生产 skew）。
- **生产 `light_priors.json` 不动**（红线）。

**为何这是正解而非 re-center**：
- 09/06 已证动相机，re-center 固定点必跟丢（2026-07-21 已证伪）。
- YOLO 逐帧框跟得住运动（qw 天花板诊断：09 候选池 IoU≥0.3 天花板高 → 好框存在），用逐帧框作锚 = 架构已锁定的「逐帧定位」路径落地。

---

## 4. 验证（engages 护栏#2：位置级 GT，禁用 prior 采样准确率）

**主指标（位置级）**：
- 对 09/06（及建议全 11 视频）每一 governing 帧，取**实际喂给 classifier 的 crop box**，与 `light_canonical_gt.json` 该帧 governing 框算 **IoU**。
- 报告：mean IoU、IoU≥0.3 命中率、逐视频分解。
- 这是「crop 是否框住真灯」的直接判据，**不依赖 prior 位置**。

**辅助（仍位置级，非 state 准确率）**：
- 09/06 重跑漏绿：目标从 base09=22 / neg_a09 worst=29 降到接近其他视频水平（如 ≤10），且误绿不回退（≤ base 2.19% 扣05）。
- LOVO 复核：`--use-negative-a` 重跑 neg_a（09 prior 修好后）→ 全量口径下 +11 缺口大头在 09，修好后应逼近门槛，可重新评估接线。

**明令禁止**：用 `signal_prior` 处采样的「state 准确率」当验收——那是自证循环（prior 偏到哪就在哪采样打分），2026-07-21 前的 prior 调优史一律不认。

---

## 5. 流程（engages 护栏#3：双关口不变）

1. ✅ 诊断（本方案 §1，已用位置级 GT 实测，bit-for-bit 可复现）
2. 📋 **本方案 → cc plan-gate 审核**（当前步，未过不动代码）
3. 实现（在独立 worktree，scoped git，署名）
4. cc 验效果（位置级 IoU + 漏绿/误绿 gate）

---

## 6. 红线 / scope
- 不碰 `light_priors.json` 生产；prior 仅 fallback。
- gate 不过不接线；neg_a/discriminator 权重不入库。
- 镜像 scan_video + Phase C，防 train/生产 skew。
- scoped git + 署名；不擅启实现，等 cc plan-gate。

## 7. 待 cc 拍板的小问题
- 验证是否扩到全 11 视频（建议是，因 canonical GT 显示多视频漂移大）。
- worktree 分支命名（建议 `wb-prior-perframe`）。

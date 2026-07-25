# Diag 1 — 生产影响面：偏框 prior 是否误导 observe()，还是被 YOLO 兜住？

> **性质**：只读诊断（不改代码、不重训、不重挖）。仅用当前 `configs/light_priors.json`（cc 已确认 05/06/09 偏框、IoU=0）在真视频上跑评测。
> **目的**：量化 cc 根因结论里"prior 也喂生产 observe()"这一环的真实影响面 —— 偏框 prior 到底有没有把生产灯态带偏。
> **结论先行**：**YOLO 把生产 observe() 兜住了**（尤其 06 从 60.3%→90.7%）；生产管线目前"可用但虚高"。**真正的重灾区是训练数据挖矿**（prior-ROI 直抽 → 39.3% 无信号 crop），生产侧 YOLO 救得了、训练侧没得救。

---

## 1. 方法

用 `scripts/eval_temporal_fusion.py` 在 05/06/09 上各跑两种模式，对照 GT `datasets/gt/light_states.csv`：

| 模式 | 含义 | 对应生产路径 |
|---|---|---|
| `--no-yolo` | 强制走 **纯 prior** 路径（无 YOLO，observe() 直接按偏框 prior ROI 采样） | 关闭 YOLO 时的回退路径 |
| 默认（含 YOLO） | YOLO 优先 → prior → brightspot | **生产默认路径** |

评测输出：每视频 `confirmed` 准确率、融合时间线、混淆矩阵，以及 `pred_*.csv` / `mismatch_all.csv` / `fused_segments.json`。

GT 关键段（来自 `light_states.csv`）：
- 05：green 0–30s（评测窗口内全程绿）
- 06：**green 0–29s → red 29s+**（有真实红段）
- 09：red 0–7s → unknown 7–11 → green 11–72 → unknown 72+

---

## 2. 结果

### 2.1 逐视频 confirmed 准确率

| 视频 | 纯 prior（--no-yolo） | 默认含 YOLO | Δ | GT 是否有红段 |
|---|---|---|---|---|
| 05 | **100.0%** (279/279) | **100.0%** (279/279) | 0 | 无（全程绿） |
| 06 | **60.3%** (123/204) | **90.7%** (185/204) | **+30.4pp** | **有（29s+）** |
| 09 | **100.0%** (289/289) | **99.3%** (287/289) | -0.7pp | 有（但尾段 unknown） |
| **总体** | **89.5%** (691/772) | **97.3%** (751/772) | +7.8pp | — |

### 2.2 融合时间线 vs GT（06 是分水岭）

- **06 纯 prior**：融合时间线 `green[0.0-48.0]` —— **把整个视频判成全绿，完全漏掉 29s 后的红段** → 40% 帧红判绿 → 60.3%。
- **06 含 YOLO**：融合时间线 `green[0.0-33.4] red[33.4-48.0]` —— 抓回了红段（边界 33.4 vs GT 29，过渡滞后，但主状态对）→ 90.7%。
- 05 / 09 两种模式都接近满分：05 是"全程绿"的易样本（任何路径都绿），09 红→绿主结构清晰、YOLO 兜底尾段 unknown。

### 2.3 06 的剩余误差（含 YOLO 仍有 10.7%）

含 YOLO 模式下 06 的全部 mismatch 集中在 **t=31.2–33.4s（绿判红边界）**——即 red 段 onset 的过渡带，属于真实的硬边界问题，与 prior 偏框无直接关系（YOLO 已把主状态救回）。

---

## 3. 分析

### 3.1 偏框 prior 在生产侧"部分误导、被 YOLO 吸收"
`observe()` 的选择逻辑（traffic_light.py）：有 YOLO 框且距 prior 近 → 用 YOLO；**无 YOLO 框距 prior 近 → 直接按偏框 prior ROI 采样背景**。06 的偏框 prior 离真灯很远，导致大量帧"无 YOLO 框在 prior 附近"→ 回退采背景 → 全绿。YOLO 默认开启后，在能检出灯框的帧上覆盖了回退路径，把 06 从 60% 拉回 91%。

**所以 cc 的疑问有答案**：偏框 prior 确实会把生产 observe() 引去采背景（06 实证：60.3%），但 **YOLO 在生产默认路径上把它兜住了（90.7%）**。05/09 因是易样本/结构清晰，偏框与否不影响结果。

### 3.2 生产"可用但虚高"——且虚高来自 Diag 2 的指标自证循环
注意：上面所有绝对准确率数字（89.5% / 97.3%）都来自 `eval_temporal_fusion.py`，而该指标是 **prior 为匹配 GT 而选、再在 prior 处采样比 GT**（见 Diag 2）。因此：
- **绝对数字不能当真实正确率**——它们度量的是"在 GT 调过的 prior 处采样能否复现 GT 时间线"，是套娃。
- **只有相对比较是可信的**：关掉 YOLO 让 06 从 90.7%→60.3%（塌 30pp），这一相对量真实证明了"偏框 prior 单独无法检出 06 红段、YOLO 是生产浮木"。

### 3.3 真正的重灾区是训练挖矿，不是生产
训练 crop 的产出路径（`mine_classifier_retrain.py:96/153/160/237`、`build_ped_signal_crops.py:116`）是 **prior-ROI 直抽**：walk/stand = "GT 段 + prior ROI 抠图"，off = "prior 外随机"。**生产 observe() 至少有 YOLO 兜底，训练 crop 没有任何兜底**——它就在偏框 prior 那块背景上抠，于是出现 Phase B 的 39.3% 无信号 crop（1269/3233）。这正是 cc 根因结论的核心：Step2 在偏框数据上过滤 = "用背景洗背景"，是死路。

---

## 4. 结论与建议

1. **生产 observe() 暂不崩**：YOLO 默认路径把 05/06/09 维持在 ~97%（相对意义），偏框 prior 只在某些帧触发回退采背景，未造成系统性崩坏。生产侧**不需要立刻停摆**。
2. **但生产正确率被指标虚高掩盖**：绝对数字来自自证循环（Diag 2），真实定位质量存疑——尤其 06 的红段边界、09 尾段 unknown 仍是隐患。
3. **训练数据是必须修的根**：偏框 prior → prior-ROI 直抽 → 39.3% 无信号 crop → 模型记背景。这与生产侧"YOLO 能救"完全不同，是**重定位 prior + 干净重挖**才能解决（cc 根因计划已定，Step2 sweep 维持 STOP）。
4. **动作建议**：
   - 等 Jacob 11 视频画框完成 → 重建 `light_priors.json`（几何保持 cc(B)，只过滤/重定位，不动 ROI 框几何进生产需镜像 scan_video）→ 干净重挖 crops → 再谈重训。
   - **切记**：不在偏框数据上跑任何训练/Step2 sweep（cc 红线，本次诊断全程只读）。

---

## 5. 产物与复现

- 评测输出：`data/output/diag_prior_noyolo/`、`data/output/diag_prior_yolo/`（pred_*.csv / mismatch_all.csv / fused_segments.json）。
- 复现命令：
  ```bash
  cd /Users/jacob/personal/crosswalk-guard
  PYTHONPATH=src ./.venv/bin/python scripts/eval_temporal_fusion.py --videos 违章05 违章06 违章09 --no-yolo --out data/output/diag_prior_noyolo
  PYTHONPATH=src ./.venv/bin/python scripts/eval_temporal_fusion.py --videos 违章05 违章06 违章09 --out data/output/diag_prior_yolo
  ```
- ⚠️ 本报告绝对准确率数字受 Diag 2 指标自证循环影响，仅相对量（YOLO 开关的 Δ）可直接采信。

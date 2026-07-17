# Plan v5 — VLM 自动标注管线（GT 自举 → 评测 / 训 CV）

日期：2026-07-17 | 作者：wb | 状态：交 cc 复核

## 0. 背景与目标

Part B（模块级评测）当前唯一卡点是**人工标注 GT**：
- B1 斑马线掩膜：每视频 3–5 关键帧标 `poly`（透视梯形，4–6 个 `[x,y]` 顶点）。
- B2 跟踪锚框：每违章窗 1–2 锚帧标 `box: [x,y,w,h]`。

Jacob 手工标 ~36 个 poly + ~18 个 box 成本高、慢。本计划用 **VLM 出 draft 标注 + 人工复核确认** 把成本降为「看一眼改几个点」，并同时给未来「训 CV 模型替代 v11 启发式」铺路。

## 1. 关键认知（cc / Jacob 已确认）

**VLM = WorkBuddy / hy3 多模态（即本 agent 自身）**，不需要外部 API、key 或本地权重。
- 标注动作 = wb 用 `Read` 工具读关键帧图像 → 输出结构化 `poly`/`box` JSON。
- 零外部依赖，环境内即可跑。这是本计划相对「接 GPT-4o / 本地模型」方案的压倒性简化。

## 2. 管线（对应架构图）

```
切片 → VLM标注(我) → 解析校验 → 人工复核(HITL) → GT落地 → 下游(评测/训CV)
                ↑________ prompt 迭代 ________│
```

## 3. 各阶段设计

### S1 切片（复用既有关键帧选法）
- 复用 `scripts/gen_gt_skeleton.py` 的关键帧选法：每违章窗取 10%/50%/90% ts 帧 + 中性帧（已落 `datasets/gt/crosswalk/<video>.json` 的 `frames[].ts/note`）。
- 新增：把对应帧抽成图片落到 `datasets/gt/_frames/<video>/<ts>.jpg`，供 VLM（我）读取。
- 漂移已天然吸收：per-window 选帧，不逐帧标。

### S2 VLM 标注（我读图出 JSON）
- 输入：单张关键帧图 + 该帧的 `note`（如「违章窗[15-28]@50%」）。
- 输出（严格对齐骨架 schema）：
  - B1：`{"poly": [[x,y]*4..6]}`，顶点按透视梯形顺序。
  - B2：`{"box": [x,y,w,h]}`，框住违章车（用违章窗空间约束排除背景车）。
- **坐标精度缓解手段（务必做，见 §5）**：
  1. 渲染带**坐标网格叠加**的帧（每 40px 一条线 + 角标尺寸），我按网格读数。
  2. 斑马线区域再做一次**放大裁剪**让我读顶点更准。
  3. 输出后由 S3 做合理性校验，异常交 HITL。

### S3 解析校验（适配器）
- 新脚本 `scripts/label_vlm_to_gt.py`：读我输出的 JSON → 写回 `crosswalk/*.json` 的 `poly` / `tracking/*.json` 的 `box`。
- 校验：顶点数 4–6、坐标在图内、poly 面积合理（非退化）、box 宽高 >0。失败标红交人工。
- 纯写骨架字段，不改既有评测逻辑。

### S4 人工复核（HITL，Jacob）
- 我标完不代表 GT 生效。**Jacob 过一遍 diff（图 + 我标的 poly/box 叠加）**，确认或微调后，骨架 `poly`/`box` 由 `null` 变真实值 = 生效 GT。
- 这是 cc 红线「wb 不许瞎标」的满足方式：VLM 出稿，Jacob 拍板。

### S5 GT 落地 + 下游
- 落地即现有骨架文件（已存在，只填 `null`）。
- 下游 Phase 1：`eval_crosswalk_mask.py` + `eval_tracking.py` 跑首个模块级基线（band-IoU / ID 碎片化 / 静止准确率）。
- 下游 Phase 2（未来，本计划不含）：拿 polygon GT 训 crosswalk 分割/检测头，替代 v11 条纹扫描兜底——这是 v3 挂起后更稳的正路。

## 4. 分阶段

- **Phase 1（本次计划范围）**：S1–S5 跑通 Part B 的 GT 自举 + 首个评测基线。
- **Phase 2（未来，不在本计划）**：基于 GT 训 CV 模型。待 Phase 1 掩膜 band-IoU 基线看清失准幅度后再立项。

## 5. 精度风险与缓解（诚实列出）

| 风险 | 影响 | 缓解 |
|---|---|---|
| 我输出像素坐标有偏差（尤其透视梯形顶点） | 污染 band-IoU | 网格叠加 + 区域放大 + HITL 兜底 |
| 同一视频不同帧风格抖动 | 跨帧不一致 | 只标关键帧、per-window，影响可控 |
| 误把背景车当违章车（B2） | 锚定错 | 违章窗空间约束 + HITL |
| 弱光/遮挡帧我标不准 | 个别帧差 | 标不出的帧交 Jacob 手标，不强行 |

**首步 POC**：先标 1 帧（违章11 违章窗@50%，透视最强）验证我的坐标精度是否可接受，再决定是否全量。这是计划获批后的第一步。

## 6. 红线遵守

- 不自动接受 VLM 标签为 GT（必经 S4 Jacob 确认）。
- 不代 Jacob 拍板标注正确性。
- 只改评测/标注工具，不碰 `match_violation_events` 与管线决策逻辑。
- 碎片仅度量分离，不改判定行为（沿用 v4）。

## 7. 待 cc 复核的开放点

1. **坐标精度接受阈值**：band-IoU 目标多少算「VLM 标可用」？建议 POC 后定（如 ≥0.85 免复核，否则人工）。
2. **B2 是否模型辅助**：box 比 poly 简单，是否允许用检测器辅助 box、我只攻 crosswalk poly？请裁定。
3. **Phase 2 是否现在立项**：本计划只到 Phase 1，训 CV 留待基线出来后。确认。

## 8. 文件清单

- 新增：`scripts/label_vlm_to_gt.py`（S3 适配器）、`datasets/gt/_frames/` 抽帧目录、POC 标注样例。
- 改动：`gen_gt_skeleton.py` 增抽帧步骤（S1）；`eval_crosswalk_mask.py` / `eval_tracking.py` 不变（等 GT 填好即可跑）。
- 复用：`datasets/gt/crosswalk/*.json`、`datasets/gt/tracking/*.json`（已有骨架，`null` 待填）。

## 9. 第一步（计划获批后）

1. 写 `gen_gt_skeleton.py` 抽帧步骤 → 抽违章11 违章窗@50% 一帧。
2. 我读该帧（带网格叠加）输出 `poly` → 你判精度。
3. 精度可接受 → 全量标 B1/B2 → 跑首个模块级基线。

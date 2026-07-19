# GT Schema / Provenance 约定（统一 GT 集合 · Phase 1）

> 本文件只**定义约定**，不改任何现有文件内容。目标是让未来（Phase 2 的 `build_gt.py`）重建 canonical GT 时，每条真值都带可追溯的来源与置信度。

## 1. 三层输入源

| 层 | 目录/文件 | 含义 | 写入方 |
|----|-----------|------|--------|
| **source** | `source/annotation.csv` | 原始人工描述（每个视频一段自然语言 GT 描述） | 初始标注 |
| **feedback** | `feedback/light_feedback.csv`、`feedback/crosswalk_feedback.csv` | 画廊（gallery）逐帧人工修正：灯态 pred→gt、斑马线多边形 | 画廊工具落盘 `data/output/annotated/`，定期入库快照 |
| **badcases** | `badcases/template.csv`（append-only） | 每轮迭代发现的算法 bad case（expected vs actual） | 评测复盘时追加 |

## 2. provenance 列约定（适用于 canonical GT 每条真值）

未来重建的 canonical 文件，建议每条真值带以下元列（已有范例见 `light_states.csv` 的 `confidence` / `note`）：

| 列 | 取值 | 说明 |
|----|------|------|
| `source` | `source` \| `feedback` \| `badcase` | 该真值最终来自哪一层 |
| `confidence` | `confirmed` \| `occluded` \| `inferred` \| `low` | 置信度（范例：`light_states.csv` 已用 `confirmed`/`occluded`） |
| `ts` | ISO-8601 时间戳 | 该真值被确立/修正的时间（范例：feedback 文件已有 `ts` 列，如 `2026-07-14T18:44:31`） |
| `note` | 自由文本 | 来源说明/修正理由（范例：`light_states.csv` 的 `从label_result_01.csv解析`、`人工画廊标注`） |

> 现有 canonical 文件（`events.csv` / `light_states.csv` / `crosswalk/*.json` / `tracking/*.json` / `videos.csv` / `light_regression.csv`）**本 Phase 不动**。上表是给 Phase 2 `build_gt.py` 的契约，要求其产出的 canonical 行携带这些列。

## 3. 现有文件 provenance 现状（已核实，供 Phase 2 参考）

- **`light_states.csv`**：已有 `confidence` + `note`，note 中混用 `从label_result_XX.csv解析`（算法解析）与 `人工画廊标注`（画廊修正）。是 provenance 的范本。
- **`feedback/light_feedback.csv`**（593 行）：列 `video,t_sec,frame_idx,pred,gt,verdict,reason,note,ts`，`gt`=人工修正真值、`ts`=修正时间（2026-07-14/15）。
- **`feedback/crosswalk_feedback.csv`**（36 行）：列 `video,t_sec,frame_idx,verdict,reason,note,annotated,imgh,imgw,poly,ts`，`poly`=斑马线多边形、`ts`=2026-07-16。
- **`light_state/违章01_gt.csv`、`违章02_gt.csv`**：旧版灯态"肉眼确认"段（`gt_state` 列空），已被 `light_states.csv` 取代，作历史留存。
- **`tracking/*.json`**：骨架在，但 `box: null` 未标（需 Jacob 标框，本 Phase 不做）。

## 4. 优先级规则（合并冲突时）

```
badcase（最新迭代）  >  feedback（画廊修正）  >  source（原始描述）  >  algorithm default（算法默认）
```

详见 `README.md`。

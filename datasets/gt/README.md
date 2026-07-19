# datasets/gt/ —— 统一真值（GT）集合

> **目的**：融合「原始标注 + 画廊修正 + 迭代 bad case」的权威真值集，可复现、可随迭代扩充。
> **阶段**：本 README 与目录结构是 **Phase 1（只打地基）**。重建脚本 `build_gt.py`、把 593 行修正真正 merge 进 `light_states`、重新生成任何 canonical 文件，均属 **Phase 2**，未做。
> **红线**：本 Phase **不动任何 canonical 文件**（见下「canonical vs 输入」），eval 消费方不受影响。

## 目录结构

```
datasets/gt/
├── README.md                      # 本文件（覆盖矩阵 + 优先级 + 扩充闭环）
├── SCHEMA.md                      # provenance/source/confidence/ts/note 约定
├── events.csv                     # [canonical] 违章事件 GT（9 处引用）
├── light_states.csv               # [canonical] 灯态 GT（4 处引用，带 confidence/note）
├── light_regression.csv           # [canonical] 灯态硬 case 回归集（4 处引用）
├── videos.csv                     # [canonical] 视频清单 + has_violation
├── crosswalk/                     # [canonical] 斑马线多边形 GT（4 处引用，9 视频）
│   └── 违章{02..09,11}.json
├── tracking/                      # [canonical] 跟踪骨架 GT（4 处引用，9 视频，box=null 未标）
│   └── 违章{02..09,11}.json
├── light_state/                   # [legacy] 旧版灯态肉眼确认段（gt_state 空，已取代）
│   └── 违章01_gt.csv  违章02_gt.csv
├── violation_events/              # [legacy] 旧版逐视频事件 csv（02,03），已被 events.csv 取代
│   └── 违章02.csv  违章03.csv
├── source/                        # [input] 原始人工描述
│   └── annotation.csv
├── feedback/                      # [input] 画廊修正（入库快照，详见 feedback/README.md）
│   ├── light_feedback.csv                     # 593 行，灯态逐帧修正
│   ├── crosswalk_feedback.csv                 # 36 行，斑马线多边形修正
│   └── light_feedback_2026-07-12_original.csv # 36 行，早期版，历史留存
└── badcases/                      # [input] 迭代 bad case（append-only，Phase 1 仅模板）
    ├── template.csv
    └── README.md
```

## 覆盖矩阵（11 视频 × 模态）

图例：✅ 已填 · 🟡 骨架在/部分 · ⬜ 无/N-A · ⚠️ 仅图像集/待补

| 视频 | 违章 | events | light_states | crosswalk | tracking | plate | 关键备注 |
|------|------|--------|--------------|-----------|----------|-------|----------|
| 违章01 | 0(负) | ✅ | ✅ label_result解析 | ⬜ N/A | ⬜ N/A | ⬜ | 真负例；legacy light_state/01_gt.csv（gt_state空）已取代 |
| 违章02 | 1 | ✅ | ✅ 混合¹ | ✅ poly | 🟡 box=null | ✅ | feedback 灯/斑马线均有 |
| 违章03 | 1 | ✅ | ✅ 灯仅反光推断 | ✅ poly | 🟡 box=null | ✅ | 车牌 京ABV3428+无牌 |
| 违章04 | 1 | ✅ | ✅ 旧GT肉眼确认¹ | ✅ poly | 🟡 box=null | ✅ | 车牌 京N07YK6 |
| 违章05 | 1 | ✅ | ✅ | ✅ poly | 🟡 box=null | ✅ | ⚠️ **灯态画廊未修正**（缺 light_feedback） |
| 违章06 | 1 | ✅ | ✅ | ✅ poly | 🟡 box=null | ✅ | 三车 |
| 违章07 | 1 | ✅ | ✅ | ✅ poly | 🟡 box=null | ✅ | 三车 |
| 违章08 | 1 | ✅ | ✅ 灯中段间隙推断 | ✅ poly | 🟡 box=null | ✅ | ⚠️ **灯态画廊未修正**（缺 light_feedback） |
| 违章09 | 1 | ✅ | ✅ | ✅ poly | 🟡 box=null | ✅ | ⚠️ **灯态画廊未修正**（缺 light_feedback） |
| 违章10 | 0(负) | ✅ | ✅ | ⬜ N/A | ⬜ N/A | ⬜ | 真负例 |
| 违章11 | 1 | ✅ | ✅ 车牌看不清 | ✅ poly | 🟡 box=null | ⚠️ 看不清 | 车牌无法结构化 |

¹ 灯态 provenance 混合：02/03/04 初版依赖旧 GT 肉眼确认（legacy `light_state/*_gt.csv`，现仅存 01/02 且 `gt_state` 列空）；其余视频由 `label_result_*` 解析。最终以 `light_states.csv` 为准，其 `note` 字段已标注各段来源。
² 车牌：结构化真值仅以 `events.csv` 的 `violating_plates` 字符串存在（覆盖全部 11 视频）；独立的**车牌图像评测集**在 `datasets/plate_eval_set/`（50 个 crop：6 correct / 24 incorrect / 20 missed + `meta.csv`，帧级，未按 02/03/04 分），属 plate 模态图像资产，矩阵未逐视频展开。
³ `datasets/ped_signal/{02,03,04}`（含 `labels.csv`）是**行人信号灯训练 crop**（标签 `stand`/`off`/`walk`，由 `build_ped_signal_crops.py` 造、喂 `ped_signal.pt` 灯分类器），属**灯态模态**训练资产，**不是车牌** —— 为未来灯态立项保留，勿与 `plate_eval_set` 混淆。

**画廊修正覆盖**（独立视角，非 canonical）：
- `light_feedback.csv`（593 行）：违章 01,02,03,04,06,07,11（**7/11**，缺 05,08,09,10）
- `crosswalk_feedback.csv`（36 行）：违章 02,03,04,05,06,07,08,09,11（**9/11**，缺 01,10）
- `light_regression.csv`：违章 02,03,04（3/11）

> 缺口提示：05/08/09 的灯态、01/10 的斑马线是画廊尚未修正处，属后续迭代补 feedback 的候选。

## 优先级规则（合并冲突时）

```
badcase（最新迭代）  >  feedback（画廊修正）  >  source（原始描述）  >  algorithm default（算法默认）
```

- 同位置出现多层真值冲突时，高优先级层覆盖低优先级层。
- `feedback` 已含人工修正（`gt`/`poly`），可信度高于 `source` 的原始描述。
- `badcase` 是最高优先级，用于推翻历史错误、锁定迭代修复。

## canonical vs 输入

**canonical（eval 直接消费，本 Phase 零改动）**：
`events.csv`(9 处引用)、`light_states.csv`(4)、`crosswalk/*.json`(4)、`tracking/*.json`(4)、`videos.csv`、`light_regression.csv`。

**input（重建源，不直接被 eval 消费）**：
`source/annotation.csv`、`feedback/*`、`badcases/*`、`light_state/*_gt.csv`(legacy)、`violation_events/*`(legacy)。

> Phase 2 的 `build_gt.py` 职责：读 input 三层 → 按优先级合并 → **重建** 上方 canonical。本 Phase 只把 input 整理入库，不动 canonical。

## 扩充闭环（每轮迭代）

```
评测跑出 bad case
   → 追加到 badcases/（append-only，带 video/t_sec/modality/expected/actual）
   → （Phase 2）build_gt.py 依"badcase>feedback>source"合并
   → 重建 canonical GT
   → 重跑 eval_violations 确认修复 + 回归（F1 不应无理由下降）
   → 必要时回画廊补 feedback，重新入库快照
```

画廊修正的同步：画廊实时写 `data/output/annotated/*_feedback.csv`（工作区），需**定期复制**进 `feedback/` 作为受跟踪的快照（详见 `feedback/README.md`）。本 Phase 是"复制不是搬移"，不改动画廊写入路径。

## provenance 约定

每条 canonical 真值建议携带 `source` / `confidence` / `ts` / `note` 元列，详见 `SCHEMA.md`。现有 `light_states.csv` 的 `confidence`+`note` 已是范本。

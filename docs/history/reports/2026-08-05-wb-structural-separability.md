# 结构信号可分性横测报告 (plan-gate #4 结构版前置, wb)

> 任务: cc 放行 (Jacob 2026-08-05 relay) 的结构信号横测, 带两条护栏 ——
> (1) 不预锁"遮挡", 同时横测两个候选结构信号; (2) 只用生产真能消费的信号(非 oracle),
> 证明它在 01 假绿段 fire 且真绿段不 fire, 三条全过才出 #3 方案。
> 结论:**两个候选信号都过不了第三条 (真绿段不 fire)**, #3 字面范围两条路全死。

---

## 1. 横测对象与可得性

| 候选 | 信号 | 生产消费点 | 灯决策点可得? |
|---|---|---|---|
| (a) 生产遮挡 | `ViolationEngineV2._is_occluded(mask)` | `evaluate` / `tag_evidence` 用 `_occ_samples` | ✅ `ctx["mask"]` 由 `n_crosswalk` 在 `n_light` 之前产出 (DAG 边 crosswalk→light), 每帧可得 |
| (b) YOLO 灯形 | `observe` 入参 `yolo_light_boxes` 数量 (`det.last_light_boxes`, COCO traffic-light, conf≥0.25) | `observe`/`detect` 选灯 | ✅ 直接作为 `observe(frame, yolo_light_boxes=boxes)` 参数传入 (dag.py:94) |

**两个信号在灯决策点都可达** —— 所以"落地可行性"不是瓶颈, **分离度才是**。

---

## 2. 横测结果 (仅在 obs=='green' 决策帧上比较)

```
label         n_green  occ_fire%   yolo0%  yolo>=1
false_green       104     100.0%   100.0%        0     ← 01 (全程红, 任何绿=误绿)
true_green       2203      90.4%    86.2%      304     ← 05/06/07/08/09 真绿

逐视频(真绿) occ/yolo0 占比:
  违章05     n_green= 481 occ_fire=100.0% yolo0= 78.4%
  违章06     n_green= 296 occ_fire=100.0% yolo0= 94.6%   ← 06 低饱和石
  违章07     n_green= 336 occ_fire= 48.8% yolo0= 95.2%
  违章08     n_green= 355 occ_fire= 91.0% yolo0= 91.0%
  违章09     n_green= 735 occ_fire= 98.9% yolo0= 81.5%
```

> 数据: `scripts/diag_structural_separability.py` 复用 `cli.run` 零循环复制, 逐帧抓
> `occ_fire=_is_occluded(mask)` 与 `n_yolo_light_boxes`, 仅对 obs=='green' 帧做标签。
> 01 标签=false_green (GT 全程红); 05–09 标签=true_green (生产 确认违规=1 待复核=0 即真 TP)。

---

## 3. 候选(a) 生产遮挡信号 —— 死 (死亡方式: 真绿段也 fire)

- **cc 的核心疑问** "01 假绿段, 生产遮挡信号到底 fire 不 fire?" → **答: fire, 100%**。
- **但** 真绿段 fire=90.4% (05/06/09 近 100%, 07 48.8%, 08 91%)。**场景常量**。
- **根因 (cc 预警的语义陷阱坐实)**: `_is_occluded(mask)` (violation_engine.py:53-66) 度量的是
  `mask[-1,:].any()` —— 斑马线掩膜触底边(看不到完整斑马线)即判遮挡。固定机位下斑马线
  常态触底边, 故几乎所有帧都 fire, **与"绿车挡住信号灯"毫无关系**。
- **若用作 #3 降级闸门**: 把 occ_fire 时段的 prior 绿降级 unknown → 会连带把 ~90% 真绿降级
  → 屠真绿 (尤其 06 石: occ_fire=100%)。**正是 ped_signal.pt 屠真绿重演, cc 明令禁止。**
- **判定**: `死 —— 真绿段也 fire (无 margin, 降级屠真绿)`。

---

## 4. 候选(b) YOLO 灯形缺席 —— 死 (死亡方式: 真绿段也缺席)

- 01 假绿段 yolo0=100% (窗口内 0 个灯框) —— 符合"信号灯被绿车挡/不在画面"的结构直觉。
- **但** 真绿段 yolo0=86.2% (05 78.4% / 06 94.6% / 07 95.2% / 08 91.0% / 09 81.5%)。
- **根因**: 本数据集 COCO traffic-light 检测召回极低 (竖排行人灯 + 远景小灯), YOLO 灯框
  在**绝大多数帧都不触发** —— 生产之所以仍工作, 是因为绿判定走 **prior 直采 ROI**, YOLO 只是
  可选的二次确认 (且在 06 这类 prior 标错时靠 YOLO 救)。所以"YOLO 缺席"是近常量。
- **若用作 #3 降级闸门**:
  - 用"YOLO 缺席→降级": 屠 86% 真绿 → 死。
  - 用"YOLO 在场→才信绿"(正确认): 真绿帧仅 13.8% 有框 → 仍屠 86% 真绿 → 死。
- **判定**: `死 —— 真绿段也无灯框 (无 margin, 降级屠真绿)`。

---

## 5. 结论: #3 字面范围两条路全死, 不可直接落地

| 条件 (cc 三条护栏) | 候选(a) 遮挡 | 候选(b) YOLO 缺席 |
|---|---|---|
| 01 假绿段 fire | ✅ 100% | ✅ 100% |
| 真绿段不 fire | ❌ 90.4% fire | ❌ 86.2% 缺席 |
| 过 06 石 (真绿不屠) | ❌ 06 occ=100% | ❌ 06 yolo0=94.6% |

**两条候选都没能"留 margin 分开 + 过 06 石"**。这与 cc 的护栏精神一致 —— 没有预锁遮挡,
实测证明遮挡(生产语义)和 YOLO 缺席都**不是有效判别器**。这避免了第三次栽坑
(预锁单一信号 → 看似分开实则屠真绿)。

---

## 6. 下一步 (不预锁, 待 cc 定 scope)

两个命名信号都死, 但 cc 也说过"**谁能留 margin 分开就用谁**" —— 开放给任一可分离的生产信号。
在出 #3 方案前, 建议再横测 1–2 个**尚未测、且非外观阈值**的候选轴 (同样只读、用生产信号):

1. **prior 绿 vs 全局候选绿一致性**: 01 假绿是 ROI 局部绿 (车/叶), 全局亮斑绿应很低;
   真绿视频灯绿是否也在全局亮斑体现? 这是"两路独立观测一致"的结构一致性, 非单特征阈值。
   *(需测量, 不预锁结论)*
2. **视频级"是否曾出现 YOLO 灯框"**: 01 全程 0; 真绿视频至少偶发。但这是**视频级**信号,
   与"禁 per-video 灯参"红线可能冲突, 需 cc 裁定能否用作闸门。

**wb 立场**: 不应急于出 #3 方案 —— 前置条件 (存在一个可分离的生产信号) 目前**不成立**。
先与 cc 确认下一道测量轴, 拿到分离证据, 再出方案走正式 plan-gate #4 (结构版)。
绝不预锁、绝不拿非分离信号硬写修法。

---

## 7. 红线合规

- ✅ 全程只读, 未写生产码, 未建 worktree/分支。
- ✅ 禁用 `select_gtfree` (脚本未引用)。
- ✅ 用生产真消费信号 (`_is_occluded` / `yolo_light_boxes`), 非 oracle/GT 遮挡。
- ✅ 未碰 `light_priors.json` 坐标 / `ped_signal.pt` / 权重。
- ✅ 诊断产物 gitignored (`data/output/diag_struct_sep/`), 本次仅新增脚本 + 本报告。

## 8. 交付物

- `scripts/diag_structural_separability.py` —— 只读双候选横测脚本 (cc 可亲跑复核)。
- `docs/reports/2026-08-05-wb-structural-separability.md` —— 本报告。
- `data/output/diag_struct_sep/diag_struct_sep.json` —— 逐视频聚合数据 (gitignored)。

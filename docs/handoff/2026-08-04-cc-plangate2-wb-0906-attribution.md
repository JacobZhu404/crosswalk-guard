# cc plan-gate #2 裁定：wb 09/06 漏绿归因(c498112)— 诊断复现 PASS,但修法标的是**非生产指标**,HALT selector⑤,escalate scope 给 Jacob

> 审核对象：`docs/reports/2026-08-04-wb-prior-failure-attribution.md` + `scripts/diag_0906_green_failure.py` + `scripts/diag_prior_motion.py`(commit `c498112`)
> 署名：cc(plan-gate/独立复核) — 遵 [[measurements-disagree-find-the-bug]]、[[eval-methodology-gap-overfit]]、[[qw-plan-execute-loop]] 方案gate
> 结论：**wb 诊断 bit-for-bit 复现、第一反转(observe 工作、漏绿在 select_gtfree 排序)成立 → 记 wb 一功。但 cc 独立追出第二反转:`select_gtfree` 根本不在生产违章路径上,09/06 端到端已是 TP。∴ ⑤ 是"修一个不上线的指标"的幻影,HALT selector⑤ 实现;真生产缺口只剩 01 FP / 04 FN。scope 决定权在 Jacob,cc 附推荐。**

## TL;DR
1. **wb 五分归因 cc 亲跑 bit-for-bit 复现**:09=⑤28/①2/⑥4(34帧),06=⑤5/④1/①2/⑥7(15帧),总 ⑤33/⑥11/①4/④1(49帧),与报告 §2 表逐格一致。
2. **接受 wb 第一反转**:`observe()` 逐帧本就出绿(§3),漏绿不在信号 crop,在 `select_gtfree` 排序。原 prior 重定位方案(改 observe crop)确实修不了这个数字。
3. **cc 追出第二反转(决定性)**:`select_gtfree` **不在生产违章路径**。生产灯态 = `dag.py:94 observe()` → `violation_engine`;`observe()` 有自己独立的选框逻辑(`traffic_light.py:159-194`,近 prior/最亮),从不调用 `select_gtfree`。全 `src/` grep:`select_gtfree` 只出现在 `ped_light_selector.py`(定义 + `leave_some_out_eval` 评测代理)与 `governing_disc.py`(仅 import `iou`,且 governing_disc 未接线)。
4. **端到端铁证**:C4 终验(`cc2414b`, F1=0.889)逐视频 —— **违章09 TP=1/FN=0、违章06 TP=1/FN=0**,两个都正确命中(唯一 FN=04,唯一 FP=01)。**09/06 漏绿在生产里根本不存在。**
5. **∴ ⑤ 是非生产指标的幻影**:修 `select_gtfree` 排序只动 selection-quality 报告的数字,对 F1 零影响(09/06 已 TP)。更糟:observe 在 09/06 **赢** select_gtfree,把 select_gtfree "上线"(stale 文档所称"生产唯一选灯路径")会让 09/06 **TP→FN 回退**。

---

## 1. 承重项独立复核(cc 亲跑,非采信 wb)

| 项 | wb 报 | cc 独立结果 | 判定 |
|---|---|---|---|
| 09 五分归因 | ⑤28/①2/⑥4 | 亲跑 `diag_0906_green_failure.py --videos 违章09` → ⑤28/①2/⑥4 | ✓ bit-for-bit |
| 06 五分归因 | ⑤5/④1/①2/⑥7 | 亲跑 --videos 违章06 → ⑤5/④1/①2/⑥7 | ✓ bit-for-bit |
| 多灯 bug 修复 | tracklet IoU 关联 + 单灯帧隔离 | git diff `bde3a09→c498112` 确认已修 | ✓ |
| observe 出绿 | 09 绿帧几乎全 green | 脚本 ⑥/⑤ 分类用真 `det.observe()`,obs_color=green 主导 | ✓ |
| select_gtfree 口径 | governing_scores=None(纯几何) | 读脚本 line 104 确认 = base 生产选灯参数 | ✓ |

诊断脚本方法学干净:真 `select_gtfree` + 真 `observe()`,GT 仅做诊断裁判(不进推理)。**wb 第一反转成立,记一功。**

---

## 2. ⚠️ 第二反转(cc 追加):select_gtfree 不是生产路径

wb §2/§6 已埋线索:"select_gtfree = selection-quality 报告漏绿的来源;observe = prior 重定位要改的另一消费者"。cc 顺线核到底:

**结构证据(grep 全 `src/`)**:
- `select_gtfree` 仅在 `ped_light_selector.py`(定义 + line 216 `leave_some_out_eval`,该函数 docstring 自注"新视频泛化的代理,**非生产保证**")。
- 生产唯一 import:`governing_disc.py:27 from ...ped_light_selector import iou`(只取几何 helper),且 governing_disc **未接线**([[governing-disc-collapse]] Jacob 拍板不接)。
- `pipeline/`、`app/` 里 **零** `select_gtfree`。

**生产灯态路径(dag.py:85-94)**:`tl.detect()`(state,供可视化)+ `tl.observe(frame, yolo_light_boxes=boxes)` → `ctx["light_observation"]` → `violation_engine.accumulate(..., light_observation, ...)`(`violation_engine.py:163` 明注"light_observation: observe() 输出")。

**observe() 有自己独立的选框(traffic_light.py:159-194)**:遍历 YOLO 灯框 → 过滤顶部假框(cy<yolo_cy_min)→ 有 prior 选**离 prior 最近**、无 prior 选**最亮** → use_yolo/prior 直采二选一。**与 select_gtfree 的 L1几何+0.15 YOLO+L2/L3 时序排序是两套完全独立的实现。**

**∴ `ped_light_selector.py:5` docstring "select_gtfree 是生产唯一选灯路径" = stale/错。** selection-quality 台(`eval_selection_quality.py:7` "驱动生产上线路径 select_gtfree")测的是一个**没接线**的子系统。

---

## 3. 端到端铁证:09/06 生产已 TP

C4 终验(`cc2414b`,`docs/reports/2026-08-03-qw-crosswalk-v2-e2e-report.md`,F1=0.889)逐视频:

| 视频 | confirmed | TP | FP | FN |
|---|---|---|---|---|
| 违章06 | 1 | **1** | 0 | **0** |
| 违章09 | 1 | **1** | 0 | **0** |
| 违章04 | 0 | 0 | 0 | **1**(唯一 FN,灯态,v6 出局) |
| 违章01 | 1 | 0 | **1**(唯一 FP,负例误报,v11 基线已有) | 0 |

**09/06 都是 TP。** 违章要成立需 ped-green([[project-semantics-spec]]),09/06 能 confirmed 正说明生产 observe() 读到了绿。selection-quality 的"09 漏绿=22"与端到端 F1 **零耦合**。

---

## 4. 裁定:HALT selector⑤,这是修幻影

- ⑤(select_gtfree 排序)修好 → 只动 selection-quality 报告数字,**F1 不变**(09/06 已 TP)。
- 且 select_gtfree 在 09/06 **输给** observe(observe 出绿、select_gtfree ⑤ 选非绿)→ 把 select_gtfree "上线"会**回退** 09/06 TP→FN。
- 这正是 [[eval-methodology-gap-overfit]]("先修度量再优化,别优化不上线的指标")+ [[measurements-disagree-find-the-bug]] 的教科书案例,与车牌线 eval_plate 84.6% vs 生产 50% 同一类("台"≠"生产")。
- **附带含义(需独立审计,本裁定不据此下结论)**:整套基于 select_gtfree 的度量(selection-quality 漏绿、[[selection-precision-ranking-bottleneck]] 召回天花板 80.5% vs 40.8%、[[governing-disc-collapse]] neg_a 91>80)若都测的是这个未接线子系统,则可能都在量幻影。这**放大**了 Jacob 暂停 governing-disc 的正确性,也意味着真正的灯态工作面在 observe() 上,小得多。**先只钉死 09/06 漏绿=幻影这一条(证据闭环),其余留审计。**

**唯一能翻此裁定的前提**:若产品意图是**用 select_gtfree 替换 observe 的选框**(把 observe 的 ad-hoc 近-prior/最亮 换成 select_gtfree 的排序器)。但(a)现在没接;(b)接了 09/06 会退步;(c)这是一个大改+回归风险决定,不是 wb 一句"修 ⑤"。**此前提的拍板权在 Jacob。**

---

## 5. wb 的真任务(cc 推荐,scope 拍板权在 Jacob)

09/06 selector 线是幻影,**不派 wb 去修 ⑤**。C4 里真正的生产灯态缺口只剩两个,都在 **observe() 路径**上、都有端到端指标兜底:

- **① 违章01 FP(唯一 FP,负例误报)= 最高杠杆**。负例被开罚单 = 生产 observe 在该视频**误绿/误判**(与 [[project-semantics-spec]] 违章条件冲突)。降掉这 1 个 FP 直接把 F1 抬过 0.889。**第一步(只读诊断)**:走生产 observe→violation_engine 路径,定位 01 FP 是 observe 误绿、还是车辆/事件形成误判(拆到 dag 逐帧 light_observation + violation_engine 累积)。这是真生产 bug,与 select_gtfree 无关。
- **② 违章04 FN(唯一 FN,已知灯态,v6 出局)**。真生产漏检,但被显式 de-scope("04 fn 可接受")。若 Jacob 重纳入,才诊断。

**建议给 wb 的下一步 = 只读诊断 01 FP 的生产根因(observe 路径),出归因报告 → cc plan-gate #3**。诊断前不建 worktree/不写生产码(沿 [[multi-agent-worktree-isolation]])。

---

## 6. 一句话给 Jacob
wb 的 09/06 归因 cc 亲跑全复现、第一反转(observe 工作、漏绿在 select_gtfree)对,记一功。但 cc 追出**第二反转**:`select_gtfree` 根本没接进生产违章路径(生产灯态=observe),而端到端 C4 里 **09/06 早就是 TP**——所以"09/06 漏绿"是一个**不上线的评测台指标**,修 selector⑤ 不动 F1,反而"上线"它会让 09/06 回退。**cc 判 HALT selector⑤(修幻影)**。真生产灯态缺口只剩 **01 FP(负例误绿,最高杠杆)** 与 04 FN(已 de-scope)。**请 Jacob 拍板**:(a) 把 wb 重定向到"只读诊断 01 FP 生产根因"(cc 推荐),还是 (b) 另有意图要把 select_gtfree 真正替换 observe 上线(那才轮到 ⑤,但会先在 09/06 回退,需谨慎)。

---
*署名:cc(plan-gate #2/独立复核)。本裁定只读诊断,未改生产码;结论 09/06 漏绿=非生产幻影,证据=grep src/ + dag/violation_engine 接线 + C4 逐视频 TP。*

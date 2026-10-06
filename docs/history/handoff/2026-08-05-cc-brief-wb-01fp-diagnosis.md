# cc→wb brief:违章01 FP 生产根因只读诊断(plan-gate #3 前置)

> 派发:cc(coordinator/arbiter)→ wb。Jacob 已拍板 scope = 转诊 01 FP(HALT selector⑤,见 `docs/handoff/2026-08-04-cc-plangate2-wb-0906-attribution.md`)。
> 状态:**只读诊断,不写生产码,过 gate 前不建 worktree**。产出归因报告 → cc plan-gate #3 → 才谈修法。

## 0. 为什么是 01 FP(scope 依据)
plan-gate #2 已坐实:09/06 漏绿是 `select_gtfree`(**未接线**评测台)的幻影,生产 09/06 端到端已 TP。C4 终验(`cc2414b`, F1=0.889)里真正的生产灯态缺口只剩:
- **违章01 FP = 唯一 FP,最高杠杆**(负例被开罚单;降掉它直接 F1>0.889)。
- 违章04 FN(已 de-scope,本次不碰)。

01 是**真负例**:`events.csv` 记「违章01,0,66.5,red,occluded,…全程红灯(部分被遮挡) 输出0(真负例)」——**全程红灯 0-66.5s**。而生产在 **[48.4-61.3s]** 形成并确认了一个违章 → 生产在红相位期间误判违章成立。

## 1. 唯一诊断问题(二分)
违章确认路径(`violation_engine.py`):
- **:87** `on_crosswalk and light_state in ("green","flashing")` → 正路确认。若 01 FP 走这条,则 **observe() 在 [48.4-61.3] 读出了 green/flashing = 误绿**(真值全程红)。这是项目核心要杀的 false-green,与 05/06 同族但发生在"全程红的负例"上。
- **:90** `on_crosswalk and light_state=="unknown" and self.unknown_to_review and occluded` → 遮挡旁路进 review。01 note 明写"部分被遮挡" + `_is_occluded` 存在 → FP 有可能是**遮挡帧被判 unknown→自动进 review→计成 confirmed**,而非误绿。

**wb 要回答的唯一问题:01 FP 在 [48.4-61.3] 是走 :87(observe 误绿)还是 :90(遮挡 unknown 进 review)?** 二者修法完全不同(前者修 observe 判色/排干扰;后者修 review 门控策略),定错=白修。

## 2. 方法(生产路径,非评测台)
1. **走生产 DAG**:`dag.py:94 tl.observe(frame, yolo_light_boxes)` → `ctx["light_observation"]` → `violation_engine.accumulate/evaluate`。**必须用 observe() 的真实逐帧输出**,不许碰 `select_gtfree`(它不在生产路径,plan-gate #2 已定)。
2. **逐帧打点 [48.4-61.3](含前后 ~3s 余量)**:每帧记 `light_observation={obs,conf}`、`on_crosswalk`(ratio vs overlap 阈值)、`stationary`、`occluded`、最终走 :87 还是 :90。定位是哪几帧触发确认。
3. **误绿归因(若走 :87)**:那些帧 observe 的绿是从哪来的?——YOLO 框选中了什么(信号灯背面/车灯/反光)?还是 prior 直采/亮斑兜底采到环境绿(树叶/车窗反光,同 05/06 家族)?给出干扰源类型 + 帧号 + 叠框截图证据。
4. **诊断裁判 GT**:01 **无** canonical 逐帧灯 GT(`light_canonical_gt.json` 仅 5 视频不含 01),但 `events.csv` 段级=**全程红 0-66.5**已足够裁判"[48.4-61.3] 任何 green 读数都是误绿"。GT 仅诊断裁判,**不进推理**。

## 3. 交付物
- 归因报告 `docs/reports/2026-08-05-wb-01fp-attribution.md`:二分结论(:87 误绿 / :90 遮挡 review)+ 触发帧号 + 若误绿则干扰源类型与证据 + 修法方向候选(仅方向,不实现)。
- 复现脚本(只读)`scripts/diag_01fp_*.py`:走生产 observe 路径逐帧打点,cc 要能亲跑 bit-for-bit 复现你的帧级结论。

## 4. 红线
- 只读诊断,**不写生产码,过 gate 前不建 worktree/分支**(沿 [[multi-agent-worktree-isolation]])。
- **禁用 select_gtfree**(非生产路径);生产灯态只认 observe()。
- GT 不进推理(诊断裁判 OK);scoped git(**禁 `git add -A`**);wb 署名 `Co-Authored-By: 千问办公 <qw@crosswalk-guard.agents>`(若 wb 用独立署名请沿其既有约定)。
- 不碰 `ped_signal.pt` / B-only s1 缓存 / 权重不入库。
- 先诊断后修:本 brief 只到"定性清楚 01 FP 的因",修法走 plan-gate #3 再动代码。

## 5. 一句话
09/06 selector⑤ 是幻影已 HALT。wb 转攻唯一真生产 FP=违章01(全程红却在 48.4-61.3 开罚单)。**先只读回答一个二分:observe 误绿(:87)还是遮挡 unknown 进 review(:90)?** 定因后走 plan-gate #3 再修。

---
*署名:cc(brief/coordinator)。承 [[prior-misframe-rootcause]] plan-gate #2 第二反转、[[eval-methodology-gap-overfit]]、[[e2e-violation-eval-baseline]]。*

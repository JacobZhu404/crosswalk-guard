# CC 验 qw 斑马线 B1 实现(caae40d+1b5c237)— 近端指标关(mask-IoU)PASS, cc 独立复现+证伪了自己的 cadence 顾虑; 放行跑端到端(C4 终验待端到端证据)

> 出自 cc(arbiter)。qw 交 B1 实现: eval 加 `--temporal`(v2 时序聚合)+ sweep 脚本, 报 mask-IoU 0.249→0.441 / recall 0.385→0.818。**这是"近端指标"检查点, 不是 C4 终验**(端到端 F1 + 负例 FP 证据尚未产出)。cc 独立核实: **近端数据真实、C1 cadence 顾虑被 cc 亲手证伪(不是刷分)、C2 加性 bit-level 干净 → 近端 PASS, 放行 qw 跑 sweep+eval_violations 出端到端证据。**

## 0. cc 独立核实
- **C1 cadence(cc 亲验, 原顾虑证伪)**: 代码 `--temporal` 实为**每 8fps 采样帧喂一次 detect(每 0.125s)**, 而生产 DAG 是 8fps 再每 `crosswalk_interval`=4 帧调一次(每 0.5s)→ **qw 喂帧比生产密 4×**, 且 commit/报告"8fps×interval4 / 与 DAG 每4帧完全一致"措辞**不准**(`interval=round(fps/8)` 是降采样步长, 非 crosswalk_interval)。cc 原担心密喂会把 IoU 刷高于生产。**亲跑对拍(4 视频, A=qw 8fps vs B=生产 2fps)**:
  | 视频 | A(qw 8fps) | B(生产 2fps) |
  |---|---|---|
  | 违章02 | 0.405 | 0.401 |
  | 违章05 | 0.364 | 0.398 |
  | 违章06 | 0.646 | 0.640 |
  | 违章09 | 0.364 | 0.369 |
  | **聚合** | **0.445** | **0.452** |
  **running-max 饱和 → 密喂不刷分, 生产节奏反而略高(0.452≥0.445)。** cc 的 4 视频聚合 0.445 ≈ qw 报的 0.441 → **数字真实**。**结论: cadence 措辞需更正, 但 0.441 不虚高、是生产的忠实(略保守)估计 → 不要求重跑。**
- **C2 加性(bit-level)**: `SENSITIVITY_PRESETS` 4 档全含 `box_overlap`(tracker.py:19-22); `violation_engine.py:144` `overlap_thr = p["box_overlap"] if occ_denom=="box" else p["overlap"]`, 默认 `occ_denom="mask"`(:138)→ 走 `p["overlap"]` **不变**。caae40d 只改 eval+sweep, 未碰 tracker/violation_engine(box_overlap 是 wb v6 既有, qw 正确没重做)。**默认 mask 路径 bit-identical, 确认。**
- **C4 近端半(mask-IoU 升)**: ✅ 0.011→0.441(+77%)、recall→0.818、3/9 视频≥0.5(06/08/11), 与 v6 改定1 预测 ~0.4 吻合。**必要条件达成, 但非充分**(端到端未验)。

## 1. 要 qw 更正(不阻塞, 但记录必须准)
- **报告 §C1 + commit 的 cadence 描述改对**: 不是"8fps×interval4/与 DAG 每4帧完全一致", 实为"每 8fps 采样帧喂 detect(比生产 crosswalk_interval 密 4×)"。**幸而 cc 已证密喂不刷分(生产 2fps 聚合 0.452≥qw 0.445), 故 0.441 结论成立** —— 但报告不能留错误的"完全一致"表述。qw 下个交付里改一句即可(或按需让 eval 真按 crosswalk_interval 节流, 二选一; cc 倾向"改措辞"因结果等价, 省一次重跑)。

## 2. 放行下一步(仍在已批 B1 §5 范围内, 非新 gate)
sweep + eval_violations + 负例检查**都是 B1 §5 已批的活, 且经注入跑评测不是"接线"(C3 的 Jacob 拍板只约束改默认), 属合法证据采集** → **qw 直接跑, 不需再等 cc:**
- `sweep_box_overlap.py`(9正+2负, 5 档, v2+denom=box)定最优阈值 —— cc 已审脚本: 临时改 `box_overlap` 带恢复、按 F1+负例FP=0 选, 合理。**注意**: sweep 只选阈值, "7 好视频 TP 不降"要靠下一步 eval_violations 对基线比, sweep 本身不比基线。
- `eval_violations.py`(11 视频, 禁 --reuse)v2+box+新阈值 → 端到端 P/R/F1。
- 负例 01/10 零新误报。

## 3. C4 终验(qw 跑完端到端后, cc 独立复核)
qw 交端到端结果后, cc 会独立核: (1) 端到端 F1 vs 基线 0.636 不回退且最好升 (2) 负例 01/10 零新 FP (3) 7 好视频 TP 不降/FP 不增 (4) flicker 根除。**全过才是 B1 收官; 接线仍待 Jacob 拍板(C3)。** 若端到端 F1 上不去(precision 0.506 mask 过宽稀释 denom=mask, 需 denom=box 或收紧梯形 5-95→10-90), 按方案记 Phase 2 seg 独立立项。

## 4. 红线
近端检查点 PASS。cc 未改生产代码(对拍用临时脚本已删)。sweep/eval_violations 是 qw 跑评测(注入, 非接线), 合法。C4 终验 + 接线为后续。

---
**一句话**: qw B1 实现近端指标 PASS —— mask-IoU 0.011→0.441/recall→0.818, cc 独立复现(4 视频 0.445≈0.441)。**cc 亲手证伪了自己的 cadence 顾虑**: `--temporal` 虽比生产密喂 4×(措辞需更正), 但 running-max 饱和 → 生产 2fps 聚合 0.452≥密喂 0.445, 数字不虚高。C2 加性 bit-level 干净(默认 mask 路径不变)。**放行 qw 直接跑 sweep+eval_violations+负例检查(都在已批 §5 内、注入非接线); 端到端证据出来后 cc 做 C4 终验。**

# CC 审 qw 斑马线 B1 方案(d5b5314)— 过关(APPROVED), cc 已独立 de-risk 核心前提, 附 4 条落地条件

> 出自 cc(arbiter)。qw 交 B1 修法方案(d5b5314): 激活 v2 的 running-max 时序聚合(现被 eval 每帧新建实例架空)+ denom=box 阈值解耦重扫 + 端到端验。**裁定: 方案打在真瓶颈上、可验、加性门控、守红线, 且 cc 独立跑了核心前提的 de-risk 确认它成立 → APPROVED, 放行进实现。附 4 条必须遵守的落地条件。**

## 0. cc 审前独立核实(逐条核了方案的技术断言)
- **v2 聚合代码真存在且真被架空**: 读 `crosswalk_v2.py:73-80` —— `self._accum = np.maximum(self._accum, resp)` 是逐像素 running-max、**无 N 帧门槛、每次 detect 都累积**(monotonic 不衰减)。`eval_crosswalk_mask.py:52` 确在帧循环内 `_make_detector` → 每帧新 `_accum` → 聚合形同虚设。qw 判断正确。
- **生产路径天然能跑聚合**: 核 `cli.py:75` = `crosswalk_detector if ... else CrosswalkDetector(cfg)`, **每视频新建一次**; DAG `n_crosswalk` 每 `crosswalk_interval`(=4, config:16)帧调 detect, sample_fps=8(config:12)→ 生产每 ~0.5s 喂一帧, `_accum` 跨全视频累积。注入点在、默认仍 v11。qw 断言正确。
- **★ cc 独立 de-risk 核心前提**(方案成败全押"顺序喂帧能抬 IoU"上, cc 亲跑验证): 单 v2 实例、按生产节奏(每 0.5s 喂一帧)顺序跑到各 anchor ts 快照, 对比 qw CSV 的单帧值:
  | 帧 | 单帧(qw CSV) | 时序聚合(cc) |
  |---|---|---|
  | 违章02@25.7 | 0.448 | **0.546** |
  | 违章02@44.5 | 0.123 | **0.507** |
  | 违章09@17.1 | 0.265 | **0.412** |
  | 违章09@41.5 | 0.308 | **0.472** |
  | 违章05(单帧已好) | ~0.42 | ~0.40(基本持平) |
  违章窗内帧 IoU 被显著抬升(往 v6 预期 ~0.4 走), 冷启动中性帧(ts=1.0, 仅 ~2 帧)如期仍低。**前提成立, 不是空想。** 这是绿灯的底气。

## 1. 四条 gate 判定
- **(a) 打真瓶颈**: ✅ 核心改动(激活时序聚合)正对 Phase A 诊断的"v2 有梯形但聚合没跑", cc de-risk 已证有效。
- **(b) 可验**: ✅ 近端 mask-IoU before/after + 北极星端到端(`eval_violations.py` 禁 --reuse + `diag_gt_crosswalk_ceiling.py` 喂真 detector)+ 负例 01/10。
- **(c) 加性/门控不静默换默认**: ✅ cli.run 默认仍 v11(cli.py:75), v2 走注入; `box_overlap` 缺失回退 `overlap`(旧口径不破); 接线待 Jacob 拍板。
- **(d) 守红线**: ✅ GT 不进生产、无 per-video 硬编码、`test_no_gt_leakage` 扩展、独立 worktree、scoped、原子写、qw 署名。

## 2. 四条落地条件(实现时必须遵守, cc 验收会查)
- **C1 — eval 必须镜像生产节奏, 别喂全帧刷高分**: 方案 §2.1 说"按 cfg.inference.fps 采样"要精确到 **sample_fps=8 且每 crosswalk_interval=4 帧喂一次(即每 0.5s)**, 与生产 DAG 一致; 因 `_accum` monotonic 不衰减, 在每个 anchor ts **快照当前 mask**(cc de-risk 就是这么做的)。若 eval 按全视频原生帧率喂, 会得到比生产乐观的上界 → mask-IoU 与端到端对不上。**近端只作参考, 真 gate 是端到端。**
- **C2 — tracker.py/violation_engine.py 改动必须纯加性**: 加 `box_overlap` + denom 分流后, **默认路径(denom=mask, 无 box_overlap 键)必须与今日 bit-identical**。交付时附一条回归证据: 默认 v11+mask 的 `eval_violations.py`(禁 --reuse)在改动前后逐视频 TP/FP **完全不变**。
- **C3 — denom=box 采用 + 接线是 Jacob 拍板点(v6 §9.1)**: 即便效果 gate 过, qw **只交 before/after 证据, 不 merge 进默认**。denom 从 mask 切 box 是语义变更(推翻 D2), 归 Jacob 定。
- **C4 — 效果 gate(cc 第二关口, 独立复核)**: 全满足才接受 ——
  (1) mask-IoU 有升(**必要, 不硬卡 ≥0.5**; train-free 预期触顶 ~0.4);
  (2) 端到端违章 F1 **不回退且最好升**(`eval_violations.py` 禁 --reuse, 含负例);
  (3) **负例 01/10 零新误报**(硬约束);
  (4) 7 好视频(02/03/05/06/07/08/09)TP 不降、FP 不增;
  (5) flicker 根除(定性: mask 不再逐帧跳变)。
  若确认触顶 ~0.4 且端到端 F1 上不去 → 记 **Phase 2(seg 微调)独立立项交 Jacob**, 不硬凑参数。

## 3. 小提醒(非阻塞)
- 冷启动: 违章窗早段 `_accum` 浅(前几帧不完整)—— 生产同样有冷启动, 且违章需持续 `duration` 帧才触发, 可接受(方案风险表已列)。别为冷启动帧的低 IoU 焦虑。
- 负例 FP 机理: 斑马线检测**单独**不造 FP —— 需 crosswalk 占道 ∧ 绿灯 ∧ 静止 三条同时。v2 检测更全一般不会凭空造 FP(除非灯/track 也误), 但仍按 C4(3) 实测确认。
- 参数微调(§2.2)先按兵不动, 等 §2.1 时序聚合的 mask-IoU 出来再决定, 对。

## 4. 红线
方案审毕 APPROVED。qw 可进实现(独立 worktree, 按执行顺序 §5)。cc 未改任何生产代码(de-risk 用临时脚本, 已删)。实现完 cc 独立验收(C1–C4), 接线待 Jacob。

---
**一句话**: qw B1 方案(d5b5314)**过关**。核心(激活 v2 时序聚合)打在真瓶颈上, cc 独立 de-risk 亲验前提成立(违章窗帧 IoU 0.12/0.27/0.31→0.51/0.41/0.47)。加性门控、守红线、可验。**放行进实现, 附 4 条件: C1 eval 镜像生产节奏(8fps×interval4, 别喂全帧刷分)、C2 默认路径改动后 bit-identical(附回归证据)、C3 denom=box 接线归 Jacob 拍板只交证据、C4 效果 gate=mask-IoU 升(不硬卡0.5)∧端到端F1不回退∧负例零新误报∧7好视频不退∧flicker根除。触顶~0.4 则 Phase 2 seg 独立立项。**

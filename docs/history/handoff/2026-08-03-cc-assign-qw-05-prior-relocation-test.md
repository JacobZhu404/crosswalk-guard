# CC 派活 qw:违章05 prior 重定位可救性验证(只读,为第③步 de-risk)

> 出自 cc(arbiter)派给 qw。承 `docs/handoff/2026-08-03-cc-verify-qw-05-candidate-gen.md`:05 有两个真 bug——(i) prior `[0.7,0.15]` 偏框 → 生产 prior 直采在 15/15 红灯帧返假绿;(ii) gov 灯低饱和(S<130)被候选口径的 `sat_min=130` 过滤。修法是重定位 prior 到 ~[0.58,0.34],但那是**写操作**(改 `configs/light_priors.json` / 跑 `derive_priors.py`),属第③步。**动写之前,先只读地验证这个修法到底成不成立**——这就是本任务。**只读,不训练,不改生产代码,不写 configs,不碰 s1/`_A` 缓存,不动 wb 训练。**

## 0. 关键机制(cc 已核, qw 据此设计)
- `_sample_prior_color`(`traffic_light.py:597`)返回**颜色字符串** `'green'|'red'|None`,**不是框**。它在 `signal_prior` 归一化中心取边长 `prior_roi_px`(05=160)的方形 ROI,按 ROI 内绿/红像素占比投票(`_sample_roi:626` 阈值 S≥60/V≥40/min_frac 0.2%;`g_frac>r_frac×1.3`→green,反之→red,否则按大者);紧 ROI 无色时自适应 2× 扩展重采。
- 这是**独立于候选/选灯的并行颜色信号路径**。所以 05 的部署相关问题**不是"天花板/框 IoU"**,而是"**prior 直采在每帧返回的颜色对不对**"。

## 1. 任务:三臂对照,量 prior 直采颜色正确率
对违章05 的 30 个 gov 帧(过滤口径同 `eval_video:117-124`),分别设三种 prior 位置,各调 `_sample_prior_color`,记返回颜色 + `_last_sample`(g_n,r_n):
- **Arm C(现状 `[0.7,0.15]`)**:基线(已知 15/15 红帧返假绿)。复算作对照。
- **Arm F(固定重定位 `[0.58,0.34]`)**:现实的静态修法(GT-free,就是个常数)。
- **Arm O(oracle 逐帧 gov 中心)**:把 `signal_prior` 设成**该帧 gov box 中心**。上界:prior 直采在正确位置到底能不能救。**GT 仅作诊断 oracle,绝非生产路径**(生产无逐帧 GT,红线"GT 不进生产推理"不违反——这是量"可救性上界"的诊断)。
- 三臂都用 `prior_roi_px=160`(与生产同),照走 2× 自适应扩展。

## 2. 每臂报这些指标(逐帧写 CSV + 聚合)
- **非 None 率**:返回了颜色的帧数/30。
- **颜色正确率**(核心部署指标):返回颜色 == `gov_color_gt` 的帧数/30。
- **假绿数**:`gov_color_gt=="red"` 但返回 `"green"` 的帧数(要杀的 bug)。
- **假红数**:`gov_color_gt=="green"` 但返回 `"red"` 的帧数。
- 每帧记 g_frac/r_frac(=g_n/total、r_n/total,total=ROI 面积),看投票有多决断。

## 3. 要回答的三个问题(报告结论段)
1. **可救性上界**:Arm O 颜色正确率——若 ~100%,说明 05 gov 灯在正确位置**是可由 prior 直采救回**的(低饱和但 S≥60 够,`_sample_roi` 宽松阈值本就为此设计)→ 重定位方向成立;若明显低,说明该灯即便局部采样也颜色不清,是更深的问题,重定位救不了,要另想(报出来)。
2. **固定够不够 vs 要逐帧**:Arm F vs Arm O 的差距——05 是手持机位,gov cx 跨度 0.252~0.667(见前诊断),固定 `[0.58,0.34]` 在漂移帧(尤其 fi=1652/1947 的 cx~0.25)大概率落空。量化 Arm F 在哪些帧掉链子 → **直接决定第③步是固定 prior 够、还是必须 `derive_priors.py`/逐帧跟踪**。这条是给 cc/Jacob 定第③步实现路线的关键。
3. **假绿是否真被杀**:Arm F / Arm O 的假绿数 vs Arm C 的 15——重定位后假绿降到多少?若 Arm O 仍有残余假绿,说明正确位置的 ROI 里仍混入红相位的绿干扰(经车窗反光那种),报出来。

## 4. 复用与口径(别引新口径)
- 复用 qw 自己 `diag_05_candidate_gen.py` 的骨架(`gd._read_frames_at`/GT 过滤/gov box 选取/prior 直采调用 `det._last_frame=frame; det.signal_prior=(px,py); det.prior_roi_px=160; det._sample_prior_color()`)。
- g_n/r_n 从 `det._last_sample` 读(`_sample_roi` 命中时会写);total 用实际 ROI 像素面积(注意边界裁剪)。
- GT=`datasets/gt/light_canonical_gt.json` 只取 `违章05`。
- 输出:新脚本 `scripts/diag_05_prior_relocation.py` + 报告 `docs/reports/2026-08-03-qw-05-prior-relocation.md` + 逐帧 CSV `data/output/qw/05_prior_relocation_per_frame.csv`(列:fi, gov_cx, gov_cy, gov_color_gt, armC_color/g_frac/r_frac, armF_color/g_frac/r_frac, armO_color/g_frac/r_frac)。署名 `Co-Authored-By: 千问办公 <qw@crosswalk-guard.agents>`,main 上提交。

## 5. 交付后
交 cc。cc 两层复核(从 CSV 重算聚合 + 从零重建若干帧的三臂颜色/占比到 4 位小数),据 §3 三问裁定第③步路线(固定 prior vs 逐帧)与是否放行写 `configs/light_priors.json`。红线:只读、不写 configs、不碰缓存、不动生产代码与 wb 训练。

---
**一句话**:动写之前先只读验修法。三臂(现状[0.7,0.15] / 固定重定位[0.58,0.34] / oracle 逐帧 gov 中心)各跑 `_sample_prior_color`,量颜色正确率 + 假绿数。答三问:①正确位置能否救回 05(Arm O 上界)②固定 prior 够不够还是被手持漂移打穿(Arm F vs O,定第③步是固定还是逐帧)③重定位是否真杀掉那 15 个假绿。GT 仅作诊断 oracle 非生产路径。cc 复核后据此定第③步实现路线与放行写 configs。

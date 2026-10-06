# CC 派活 qw:违章05 候选生成瓶颈根因诊断(只读,第②步单点)

> 出自 cc(arbiter)派给 qw。承 [[selection-precision-ranking-bottleneck]] / `docs/handoff/2026-08-03-cc-verify-qw-recall-ceiling.md`:天花板诊断把候选生成瓶颈从"全局"收窄到 **违章05 单视频**(天花板仅 10.0%、mean_ceil_iou 0.170,其余 10 视频是排序问题)。这个任务只诊断 05:候选池为什么没好框、怎么修。**只读,不训练,不改生产代码,不碰 s1/`_A` 缓存,不动 wb 正在跑的训练。**

## 0. 任务边界
- **唯一目标视频**:违章05(30 个 governing 帧,过滤口径同 `eval_video:117-124`:gov_boxes 非空、gcolors 全 unclear 跳过)。
- 输出:新脚本 `scripts/diag_05_candidate_gen.py` + 报告 `docs/reports/2026-08-03-qw-05-candidate-gen.md` + 逐帧 CSV 到 `data/output/qw/`。署名 `Co-Authored-By: 千问办公 <qw@crosswalk-guard.agents>`,在 main 上提交(新文件,无冲突)。

## 1. 背景:cc 已定位的机制(qw 据此设计,但要亲验)
- **eval/sel_prec 候选口径**(`eval_selection_quality.py:99-104`,也是天花板诊断用的口径):`det._candidates(frame)`(全帧 HSV 亮斑,带 `min_area_px`/`sat_min`/`value_floor`/`max_aspect`/`_classify` 过滤)+ YOLO cls9 conf0.05 imgsz1280,经 `build_candidates` 合并。**这条路径没有 set_video_prior、不走 `_sample_prior_color` 的 prior 直采。**
- **生产 observe()/detect() 路径**会 `set_video_prior("违章05")` → prior=`[0.7, 0.15, 160]`(cx,cy 归一化 + roi_px)→ 触发 `_sample_prior_color`(绕过面积/饱和过滤,专收 <30px 暗淡行人灯)。
- **⚠️ 核心口径疑点(必须先答)**:05 天花板 10% 是用"裸 `_candidates`(无 prior)"口径量的。如果 05 的 gov 灯是小/暗灯,被 `_candidates` 的 `min_area_px`/`sat_min` 过滤掉,但生产 prior 直采能捞回——那 1.3% sel_prec 可能是**口径伪影**(eval 没跑 prior 路径),不是生产真烂。**这决定 05 到底该不该修、怎么修**,是本任务最高价值产出。

## 2. 要回答的问题(逐 30 帧,写进 CSV)
对违章05 每个 gov 帧,取 gov_box(px 坐标 = box_norm×[W,H]),报:
1. **gov 灯物理属性**:px 宽高、面积(px²)、在帧内位置(cx,cy 归一化)。→ 判是否小灯(面积 < `_cfg_tl` 的 `min_area_px`?打印该阈值)。
2. **gov 灯的 HSV**:在 gov_box 内取 patch,算 mean H/S/V。→ 判是否被 `sat_min`/`value_floor`/`_classify`(ms<22 或 H 不在绿/红带)过滤。
3. **YOLO 是否命中**:该帧 YOLO cls9 conf0.05 imgsz1280 的框里,离 gov_box 最近的 IoU 是多少(=0 则 YOLO 完全没检到)。
4. **裸 `_candidates` 是否命中**:HSV 亮斑候选里离 gov_box 最近 IoU。
5. **prior 命中度**:prior 中心 `[0.7,0.15]`(px=[0.7W,0.15H])到 gov_box 中心的归一化距离;prior ROI(边长 160px,以 prior 中心为心)与 gov_box 的 IoU。→ 判 prior 偏框程度([[prior-misframe-rootcause]] 记 05 曾 prior IoU=0,亲验是否仍如此)。
6. **prior 直采能否捞回(口径疑点的答案)**:调 `set_video_prior("违章05")` 后走 `_sample_prior_color`(或 observe 的 prior 直采路径)在该帧产出的框,与 gov_box 的 IoU。→ 若这条能命中而裸 `_candidates` 不能,则 05 是**口径伪影**,生产其实能捞到候选。

## 3. 裁断产出(报告结论段)
把 30 帧的失败归到具体 stage,给一句可执行方向,四选一(或组合):
- **(A) 小灯被面积过滤** → 方向:prior ROI 裁剪+上采样后重跑 YOLO/HSV,或降 `min_area_px`(但要评估对其他视频误报的溢出,只诊断不改)。
- **(B) 暗/低饱和被 HSV 阈值过滤** → 方向:prior 直采路径(已绕过过滤)是否本就该纳入 eval/生产候选池。
- **(C) prior 偏框**(prior 中心离 gov 远/ROI IoU≈0) → 方向:重定位 05 的 prior(第③步 prior 重定位),配 `derive_priors.py`。
- **(D) 口径伪影**:prior 直采能命中、只是 eval 候选口径没跑它 → 方向:sel_prec 天花板诊断低估了生产,05 未必要修候选生成;需在天花板口径里补 prior 直采重测。

## 4. 复用与口径(别引新口径)
- 复用 qw 自己已写的 `scripts/diag_candidate_recall_ceiling.py` 的候选重建骨架(`gd._read_frames_at`/`gd._lazy_yolo`/`gd._cfg_tl`/`det._candidates`/`build_candidates`/`iou`),口径逐行对齐 `eval_video:99-104`。
- prior 相关调用看 `traffic_light.py:114 set_video_prior` / `:597 _sample_prior_color`,priors 文件 `configs/light_priors.json`。
- GT = `datasets/gt/light_canonical_gt.json`,只取 `video=="违章05"` 的帧。
- 逐帧 CSV 落 `data/output/qw/05_candidate_gen_per_frame.csv`,列全上面 6 项 IoU/HSV/px,供 cc bit-for-bit 复核。

## 5. 交付后
交 cc。cc 照旧两层独立复核(从 CSV 重算聚合 + 从零重建若干帧对 IoU/HSV 到 4 位小数),再裁定 05 归 (A)/(B)/(C)/(D) 哪条、进不进第③步 prior 重定位。红线不变:只读、不碰缓存、不动生产代码、权重不进库。

---
**一句话**:qw 空闲 → 派第②步单点(违章05 候选生成根因)。逐 30 gov 帧解剖 gov 灯 px/HSV + YOLO/HSV/prior 各自命中度,重点答一个口径疑点:**05 天花板 10% 是生产真烂,还是 eval 候选口径没跑 prior 直采路径的伪影**。归到 (A)小灯过滤/(B)HSV阈值/(C)prior偏框/(D)口径伪影 之一,给可执行方向。只读,不碰 wb 训练与缓存。cc 复核后定 05 是否进第③步 prior 重定位。

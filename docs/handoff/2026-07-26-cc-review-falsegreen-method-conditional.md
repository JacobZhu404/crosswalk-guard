# CC 审 wb 误绿测量方法:去循环口径过关,但 3 个口径缺陷必须先修(含更正我自己的 brief)

> 出自 cc(arbiter)。审 `docs/plans/2026-07-26-wb-falsegreen-measure-method.md`。
> 我独立核验了 wb 每条"坐实的事实"(不取信),下列 ✅=复现属实,⚠️=挖出的问题。

## 0. 独立核验(bit-for-bit)
- ✅ `select_gtfree`(ped_light_selector.py:99)签名 `(candidates, ped_prior, temporal_scores=None, temporal_weight=0.4, l3_scores=None, l3_weight=0.3)`——**无任何 GT 参数**,L3 默认关;`compute_temporal_scores`(:147)只读候选历史。去循环地基成立。
- ✅ `candidates_temporal.json` 实为 `{n:33, imgsz:1280, window:12, step:4, records:[33]}`,每条含 `gt_box_norm`/`gt_wh`(=GT 播种的选灯 eval 候选)。**做不了帧级误绿,必须重抽**,属实。
- ✅ `models/ped_signal.pt` 存在(36KB);无 L3 全量 `.pt`(仅 LOVO)。前置两步(密集重抽 + 训全量 L3)必要,属实。
- ✅ `events.csv` 26 段覆盖全 11 视频,列 `video,start_s,end_s,light_state,...,is_violation`;`light_state/*.csv` 的 `gt_state` 确为空。
- ✅ `light_priors.json` = `{video:[cx,cy,roi_px]}` 生产手写值(如 违章02=[0.72,0.17,160]),**非** `light_location_gt.json` 逐帧 GT → 用作 L1 ped_prior 合法(反映生产真相,含已知偏框)。

## 1. ⚠️ 缺陷 A(最重要):`unknown` 段 + 段间空隙必须**排除**,不能记作"非绿"
核 `events.csv` 段发现:**05/07/09 含 `unknown` 灯态段**;02/03/04/07/09 段间有 **1–3s 空隙**(如 02 的 68–69s、76–79s)。
- wb 的定义 `gt_walk=False ⇒ 计误绿`,会把 `unknown` 段和空隙帧一律当"非绿",**任何 walk 输出都记误绿**。
- **这会虚高误绿率**:`unknown`=真值都判不了,空隙可能正好是过渡/未标——拿不可裁决的帧充误绿分子,**把标尺灌水、错误地把天平推向 B**。这正是"误绿数字要当 A/B 决策标尺"最不能犯的错。
- **要求(硬)**:三态映射——`green→gt_walk=True`;`red/occluded/none→gt_walk=False(计误绿)`;**`unknown` 段 + 段间空隙 → `gt_walk=UNKNOWN` → 移出"可评帧",既不计误绿也不计分母**。报告单列"因 unknown/空隙排除的帧数",透明化。

## 2. ⚠️ 缺陷 B:主行应是**颜色路径(生产默认)**,分类器是 A 候选——wb 主次拿反了(更正我自己 brief 的措辞)
核 `configs/config.yaml:52` **生产灯态默认 `method: "color"`**(v7-stable 颜色兜底);`ped_classifier`(走 `ped_signal.pt`)是**未接线的选项**,且 `ped_signal.pt` 已诊断过拟合/屠真绿([[light-classifier-retrain]])。
- 我的 brief §3 写"状态用现有分类器"措辞不准——**当前生产真相=颜色路径**。审计要的标尺="现役管线误绿多严重"=**color 路径**。
- 分类器路径=A 的候选改进(且将来配 ImVisible 预训练)。
- **要求**:**color 路径为主行(标注"现役默认"=真正标尺)**,`ped_signal.pt` 分类器路径为**并列对照行**(=A 打补丁能否降误绿)。两行都要,但别把未接线的 broken 分类器当主标尺。

## 3. ⚠️ 缺陷 C:L3 权重去掉 `w=1.0` 樱桃摘,用固定保守 `w=0.5`
`w=1.0`"最优"是 32 实例上的 sweep 噪声(per-video 最优四散,[[light-classifier-retrain]] 多 seed 铁律同理)。
- **要求**:报**两行**——`L1+L2-only`(无 L3 基线)+ `L1+L2+L3 @ 固定 w=0.5`(保守,我此前裁定的 +9.4pp 稳健点)。**不featured `w=1.0`**(过拟合小 eval)。附录可提一句 sweep 敏感性,但主表不用它。

## 4. 五点拍板(逐条回 §9)
1. **状态源** → 见缺陷 B:**color 主行(现役标尺)+ 分类器对照行**。不是"分类器主行"。
2. **L3 权重** → 见缺陷 C:**no-L3 基线 + w=0.5**;弃 w=1.0/0.7。
3. **密集候选 step** → **step=4 同意**(源 fps≈29.70 → ~7.4 采样fps,足够 ~1s 误绿事件;step=1 白烧 YOLO)。重抽须走**全 GT-free**:`collect_candidates` **不得写 `gt_box_norm`**(那是 eval 用),纯 YOLO∪HSV。
4. **漏绿** → **同意纳入附注**(不计主率)。它对审计有额外价值:量"召回洞"多大(候选没检出真绿的帧),佐证"recall 洞对误绿安全但对违章召回有代价"。
5. **fps 映射** → fps=29.70、`t=source_fi/fps` 映射 **同意**;但**空隙帧改判 UNKNOWN 排除**(见缺陷 A),不是 `gt_walk=False`。

## 5. 另外两条执行纪律(写进代码)
- **禁调 `seed_with_gt`(:84)/`derive_ped_priors`(:41)**:这两是 GT 播种/GT 派生 prior,测量路径一律不许碰(护栏1)。ped_prior 只从 `light_priors.json` 取。
- **`light_state` 语义自查**:确认 `events.csv.light_state` 是**行人灯**态(违章语义 [[project-semantics-spec]] 靠它驱动 is_violation)。跑前拿 1–2 个已知帧(如 08 全 green、01 全 red)对一眼灯态方向,避免灯态源搞反导致整份误绿反号。

## 6. 放行边界
- **条件放行**:缺陷 A(unknown/空隙排除)+ B(color 主行)+ C(w=0.5/弃 w=1.0)+ §5 两纪律 落进方法与代码,即可执行(密集重抽 → 训全量 L3 → 测量 → 报告+画廊)。**其余口径(去循环、GT 只比对、双行、成因归类、交付形态)已过,无需再审。**
- 红线不变:只测不接线、生产 prior/权重只读、禁 GT 进推理、scoped git、署名、trunk main、TDD 先。报告出来后 cc 复核数字 + Jacob 抽检画廊,再定 A 收工 / 转 B。

---
**起手**:wb 把缺陷 A/B/C + §5 补进方法文档(尤其 A 的 unknown/空隙排除,这是标尺不灌水的命门),即可按口径跑。方法主体已过,不必重写。

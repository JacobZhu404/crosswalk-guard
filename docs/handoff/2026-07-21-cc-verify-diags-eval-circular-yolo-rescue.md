# CC 复核裁定:wb 两诊断均**独立证实** + 两点影响追加(eval 循环血缘 / YOLO-crop 战略叉)

> 出自 cc(arbiter)。独立复核 wb 的 Diag1(生产影响)+ Diag2(指标有效性),没采信叙述,读码 + 读 wb 的 eval 输出验证。

## 0. 裁定
- **Diag2(自证循环)= 结构上证实**:`identify_pedestrian_signal.py` 按"与 GT 颜色时间线匹配率"选 prior(:136-156),`eval_temporal_fusion.py` **默认加载 light_priors.json 并在该 prior 处采样**(:191/233/242 + :55-57)再比**同一份 GT**(:108-129)。→ 指标优化的正是 prior 被调优的那个目标,背景点只要颜色随 GT 变即可高分。Jacob 硬 GT(IoU=0)是独立证伪。**wb 对。**
- **Diag1(YOLO 兜底)= eval 输出证实**:06 `--no-yolo` 输出单段 `[0–48s green]`(漏真实红段),`yolo` 输出 `[green→33.35s][red 33.35–48s]`(抓回红)。两模式确不同。→ 偏框 prior 单独瞎(采背景读绿),YOLO 在生产兜住;训练挖矿是 prior-ROI 直抽、无 YOLO 兜底 → crop 被直接毒化。**wb 对。**
- wb 自省"只有 YOLO 开关 Δ 可信、绝对准确率来自循环指标"——**正确**,采纳为口径。

## 1. Diag2 血缘比 wb 说的更广(必须钉死)
循环指标不止污染那 4 个历史 commit,而是**凡经 observe()+prior 对 GT 打的"灯态准确率"绝对值都不可信**:
- 历史 prior 调优史(472d9a1/8b863f1/8e6f6d2/c26b3f3)的"提升"叙事 → **不再作定位正确性证据**。
- E2E 灯态准确率、Phase B 四关 gate 里凡依赖 prior-ROI 采样的绝对读数 → **prior 重定位 + 指标换血前一律打折**。
- **仅两类可信**:①YOLO 开关 Δ(GT 固定,反映 YOLO 贡献);②IoU vs 硬 GT(位置级,独立于颜色循环)。
- 呼应 [[eval-methodology-gap-overfit]]:又一处"eval 自证"。**判 prior 是否框住灯,今后只认 IoU vs light_location_gt.json。**

## 2. 追加影响(cc,wb 未展开)
1. **偏框 prior 在生产不只是"不帮忙",还会主动误导**:observe() 有 prior 时"选离 prior 最近的 YOLO 框"(traffic_light.py:174-183)→ 偏框 prior 可能让生产选中**错的** YOLO 框(如错位处的车灯)。故 prior 重定位**对生产也有价值**(改善 YOLO 选框),非仅救训练。
2. **战略叉:既然 YOLO 能定位灯(06 被救到含红段),是否还该用固定 prior-ROI 裁图重挖?**
   - 现架构 = 固定 prior 位置裁图 → 判色;偏框即毒化。
   - 备选 = **YOLO 框(逐帧形状定位)裁图 → 判色**,天然框住灯、免手设 prior、对机位漂移鲁棒、对新视频可迁移。
   - **前置验证**(重挖前必做,cheap):YOLO 的 traffic_light 框(COCO cls9)到底框的是**行人信号灯**还是**车用红绿灯**?用 `models/yolov8n.pt` 在几视频上跑,量化 YOLO 框与 Jacob 硬 GT 框的 IoU/命中率。若 YOLO 稳定框住行人灯 → 重挖应走 YOLO-crop 而非固定 prior;若 YOLO 只框车灯 → 回固定 prior 重定位。**别在没验这个前就大规模按固定 prior 重挖。**

## 3. 下一步
1. **cc 建候选 `configs/light_priors.rebuilt.json`**(GT 中心,roi_px 先不动=单变量;02/04 保留,10 多灯甄别),非破坏、附 diff,不覆盖生产。
2. **YOLO-vs-GT 命中率验证**(§2.2 前置):决定重挖走 YOLO-crop 还是固定 prior。归 wb(read-only,cc verify)。
3. 干净重挖(`classifier_retrain_v2/`)+ 重训 → 才谈 gate,且 gate 口径须加 IoU/位置级验证,旧准确率仅作相对哨兵。
4. 03/若有仍偏的:按 GT 重定位;no_light 帧正常不算偏。

## 4. 红线(不变)
先计划后动、绝不擅自重挖、light_priors.json 改动留旧值可回滚 + 验生产不回退、gate 不过不接线、Phase C 全 11 重训、不碰 enforce_transition_limit、scoped git、署名、trunk main、TDD 先。

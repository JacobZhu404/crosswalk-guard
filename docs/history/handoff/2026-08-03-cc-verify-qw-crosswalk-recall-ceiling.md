# CC 复核 qw 斑马线召回天花板诊断(237b49d)— 两层 bit-for-bit 通过, 结论成立且**正确推翻了 cc spec 的错误预期**, 放行 Phase B(但接现成 v6 计划, 别重造)

> 出自 cc(arbiter)。qw 交 Phase A 只读诊断(237b49d): v11 mask-IoU 均值 **0.011**(35/35 帧<0.5)、recall 0.012/precision 0.093 **都极低**、v2(0.249)大幅优于 v11。cc 两层独立复核 + 与历史对拍 + 定性 spec 预期。**裁定: 数据两层 bit-for-bit 通过, 结论成立, qw 正确证伪了 cc spec 里"全宽带→recall 高 precision 低"的错误预期。Phase A 收官, 放行 Phase B。**

## 0. 独立复核(两层 + 历史对拍)
- **聚合层**(cc 从 `crosswalk_recall_ceiling_per_frame.csv` 自算, 不跑 qw 脚本): v11_iou 均值 35 帧 =0.0111、v11_recall 均值 =0.0123 → **对上** qw 报告的 0.011/0.012。
- **重建层 A(跑现成、非 qw 脚本)**: cc 跑**未改动的** `scripts/eval_crosswalk_mask.py --detector v11`(cc 没写、qw 没碰的 canonical 骨架)→ **总体平均 mask-IoU=0.011, 9/9 视频<0.5**。独立骨架复现 qw 的数。
- **重建层 B(cc 自写脚本, 独立重跑 v11+v2 检测 + poly 栅格化)**: 抽 4 帧(02@1.0/05@3.0/06@14.5/09@17.1)自算 iou/recall/precision/面积 → **每位数与 qw CSV 完全一致**: 02@1.0 iou0.0388/rec0.0417/prec0.3621/gt_area255627/v11_area29440; 05@3.0 0.0002/0.0002/0.0014; v2 05@3.0 iou0.4207/rec0.917。bit-for-bit。
- **历史对拍(消除"矛盾")**: 侦察报告把 v6 计划表末列标成"mask-IoU 0.43"是**误标** —— 那列是**覆盖(coverage)**。v6 计划**正文 line 16 白纸黑字"v11 mask-IoU 仅 0.011"**。**qw 的 0.011 与历史记载精确吻合**, 不存在 40× 矛盾, 反而是第三处独立佐证。

## 1. cc spec 的预期被 qw 正确推翻(记录在案)
- cc spec §2 Q2 预设"全宽横带 → **recall 高 / precision 低**(几何过宽溢出)"。**这个预期错了。**
- 实测 recall 0.012 / precision 0.093 **都近零**。根因(cc 重建帧确认): v11 出的是**很薄的全宽横带**(CSV band 高度仅 ~22–50px, 如 02 的 `(288,310)`), 且**竖直位置常落在 GT poly 之外/边缘**(05: v11 带 y[582,615] 在 GT y[612,1078] **之上**几乎不交; 06@14.5: v11_area=0 根本没出带)。即便带子落进 GT 竖直范围内, **透视斑马线在远端收窄成梯形**, 全宽薄带在那个高度只切到一薄条 → recall/precision 双低。
- **不是"带子太宽溢出"(cc 预期), 是"薄带定位错高度 + 出不了透视梯形"**。qw 判"recall 和 precision 都极低=真召回失败"**方向正确**。**junior 敢用数据顶掉上级 spec 的预设、且顶对了, 记一次好。**
- 唯一措辞微调(非错): qw 说"根本没检到"稍强 —— v11 多数帧仍出了非零带(只是薄+错位), 仅 05@15/06@2.9/06@14.5 area=0。更准: "薄带错位/形状错, 有效覆盖近零"。不影响结论。

## 2. v2 方向确认(接现成 v6 计划)
- v2 IoU 均值 **0.249 ≫ v11 0.011**, cc 重建帧确认(05@3.0 v2 **recall=0.917**, 梯形确实抓到了透视斑马线)。**梯形方向可行。**
- 但 v2 单帧仍多数<0.5(均值 0.249)。这与**已存在的 v6 计划**(`docs/history/plans/2026-07-17-wb-plan-v6-...`)完全对齐: 该计划 §3 早已设计 v2 走**时序聚合(running-max)抗遮挡**、§0/改定1 预期 train-free 触顶 ~0.4(车底 60% 遮挡外推不了)、Phase 2 才上 seg。**qw 的诊断=对 v6 计划 Phase-1 前提的实测确证。**

## 3. 放行 Phase B —— 但**接现成, 别从零造**
Phase A 过关, qw 进 Phase B(自己提方案→cc 审→实现→cc 验)。**关键约束: 修法不是绿地** ——
- **已有 `src/redlight/models/crosswalk_v2.py`**(透视梯形 + running-max 聚合骨架), 且 **v6 计划已把整条修法路径设计好了**(v2 时序聚合 + `cli.run(crosswalk_detector=, occ_denom="box")` 注入 + `box_overlap` 阈值扫描 + 红线单测 `test_no_gt_leakage`)。qw 的 B1 方案**必须建立在扩展 v6 计划 + 现有 v2 之上**, 不要重新发明检测器。
- **B1 方案要点**(qw 写 `docs/plans/`): ① 诊断已指向"薄带错位/无梯形"(引本诊断数据) ② 具体改法 = 让 v2 的 running-max 时序聚合真正跑起来(现在 eval 是每帧独立实例, 没走时序), 目标把 mask-IoU 从 0.249 往 v6 预期的 ~0.4 抬 ③ 验: 近端 mask-IoU(`eval_crosswalk_mask.py`) + 北极星端到端(`eval_violations.py`/`diag_gt_crosswalk_ceiling.py` 口径喂真 detector) + denom=box ④ 风险: 负例 01/10 零新误报、7 好视频不回退。
- **红线(v6 计划 §6, 继承)**: **GT 绝不进生产**(新检测器只吃帧, 不 import `datasets/gt`); 无 per-video 硬编码多边形; 新检测器走注入/flag **不静默替换默认 v11**(cc 验完 Jacob 拍板才接线); 独立 worktree、scoped、qw 署名。
- **效果 gate(cc 第二关口)**: cc 独立重跑真 detector 量 mask-IoU before/after(必须升) AND 端到端 F1 不回退且最好升 AND 负例 01/10 零新误报 AND 不拖垮别模块。**注意 v6 已诚实下调预期: train-free 大概率卡 ~0.4 达不到 0.5 bar** —— 所以 Phase B 的验收不能硬卡 mask-IoU≥0.5, 而看**端到端 F1 是否真升 + 是否根除 flicker**; 若确认触顶 ~0.4, 则记 Phase 2(seg 微调)为独立立项交 Jacob, 不硬凑。

## 4. 对 qw 的反馈
- **正向(强)**: 数据两层 bit-for-bit 干净; **敢用实测顶掉 cc spec 的错误预期且顶对了**; denom=mask 稀释效应也正确点出并与历史 denom=box>mask 对上; band 列(竖直范围)留得好, 让 cc 能定位"薄带错高度"根因。
- **要注意**: "根本没检到"措辞稍强(多数帧有薄带, 是错位非零检出) —— 下次区分"零检出"vs"检出但错位/错形状", 二者修法不同。进 Phase B 先读 v6 计划 + 现有 crosswalk_v2.py, **接着修别重造**。

## 5. 红线
只读诊断 Phase A 已交, cc 两层独立坐实(canonical 骨架 + cc 自写重建 + 历史对拍三重佐证)。cc 未改任何生产代码/GT(重跑仅读, 临时脚本已删)。Phase B 的实现/接线是后续, 走 cc 双关口 + Jacob 拍板。

---
**一句话**: qw 斑马线召回天花板诊断(237b49d)数据**两层 bit-for-bit 通过**(cc 跑 canonical `eval_crosswalk_mask.py` 得 0.011 + cc 自写重建 4 帧每位数吻合 + v6 计划正文 line16 记载 0.011 三重佐证)。结论成立: v11 mask-IoU 0.011、recall/precision 双低, **正确推翻了 cc spec"全宽带→recall 高"的错误预期** —— 根因是薄带定位错高度 + 出不了透视梯形, 非"带太宽溢出"。v2(0.249, 05 recall 0.917)方向可行但需时序聚合(v6 计划早有设计)。**放行 Phase B, 但必须接现成 v6 计划 + crosswalk_v2.py 扩展, 不重造; 验收看端到端 F1 而非硬卡 mask-IoU≥0.5(train-free 预期触顶 ~0.4)。**

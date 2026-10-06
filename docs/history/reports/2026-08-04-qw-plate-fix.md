# 车牌事件回填修复报告(qw, P2/P1/P3 — cc 双 gate 链路)

> 前置: 诊断 `docs/reports/2026-08-04-qw-plate-diagnosis.md`(0c6c09f) → 方案 `docs/plans/2026-08-04-qw-plate-fix-plan.md`(3825f30) → cc 第二关 gate PASS(cf1d246) → P2 效果 gate 送回(f26c91f) → P2 v7.1 效果 gate PASS(a9c48a8) → P1/P3 解冻 → 本报告。
> 分支 `plate-fix`(worktree `crosswalk-guard-plate`), commits: 4c23859(P2) / 3cc0859(P2 v7 修) / 783bd13(记账①) / 068bfea(P1+P3)。

## 0. 验收指标(cc 背书: 事件车牌命中率为主指标)

| 阶段 | 命中 | 真事件误罚 | 说明 |
|---|---|---|---|
| 基线(2026-07-16 旧逻辑) | 6/12 | 2(02 京A14672 + 05 京N541E6) | 开错罚单 |
| P2 v7.1(gate PASS) | 5/12 | 0 | 宁缺毋滥: 02/05 空 |
| **P1+P3(本报告)** | **7/12** | **0** | +08 京ACW6553 +09 京NNM526 |

盲区/负例标注: 01(负例 FP 事件层, cc carve-out)、11(GT 无牌, 不可核验——记账②)。

## 1. P2: 事件车牌安全回填(三重约束, cc gate PASS a9c48a8)

`_episode_plate_all`(cli.py)约束:
1. **违章车组**: 候选归属 tid 事件窗口内 stationary≥0.6(挡过路/移动车, 02 京A14672 静止 0.38 被挡);
2. **全局真实性**: 候选全视频读取帧数≥5(挡孤证幻觉, 03 京ABV200/020 全视频 3 帧被挡);
3. **空间聚集**: 候选 ED≤1 变体系归属 tid 质心 x 范围≤0.25×帧宽(挡跨车关联污染, 05 京N541E6 系 7 tid 质心 607px 被挡 → 05 空串, cc 送回焦点);
4. 无候选宁缺毋滥返空; 时间语义: 牌读取帧可落窗口外(GT 语义"车牌需视频全局读取")。

迭代要点(cc 复核认可): 弃"牌读取时间窗"(03 银灰车 143s/06 黑车 46s 才看清, 窗口外读取是真牌); 弃"代表 track 单锚空间检查"(07 京Q5D2N8 第二违章车被误杀); 定稿 = stationary 车组 + 全局帧数 + 质心聚集度。

## 2. P1: 多车事件多牌(命中 5/12 → 7/12)

`ev["plates"]` 多牌列表(ev["plate"] 主牌兼容), 次牌三约束:
- 全局帧数≥10(挡低帧过路车: 08 京PK9B77 6帧 / 06 京WPM966 6帧);
- 与主牌 ED>1(非变体);
- **归属 tid 不含代表 track**(代表车的牌应为主牌; 挂在代表 track 上的其他牌=代表车误读/污染, 03 京FJQ279 挂代表 tid103 → 挡; 多车事件第二违章车 08 京ACW6553 tid92 / 09 京NNM526 tid99 均非代表 → 保留)。

另加**同 tid 互斥**(一车一牌, 挡同车误读变体: 08 tid92 上京ACW6553 vs 京J00542 互斥取前者)。

结果: 08 两车 2/2(京ACD5358+京ACW6553)、09 两车 2/2(京AC63971+京NNM526); 07 三车仍 1/3(京ACG0878 全局仅 3 帧, 见 §4 天花板)。

## 3. P3: ROI 放大重试(机制就绪, 02/05 零收获 = 物理极限)

主牌为空时, 事件窗口内代表车框 ROI 放大 2x 重跑 HyperLPR3(conf≥0.6+格式校验)。
- 帧级验证(02 44s ROI 抽帧): 违章车前牌**侧角 ~60°**(画面右侧前保险杠), HyperLPR3 对斜角牌弱, ROI 重试零收获;
- 结论: 02 京LNE560/05 京ADH9206 的不可读是**识别层物理极限**(角度/分辨率), ROI 重试不解决 → 记 **seg 检测模型独立立项交 Jacob**(换检测/识别层, 超本 scope)。

## 4. 信号天花板(9/12 未达的诚实归因)

cc 预估 P1 后 9/12, 实测 7/12, 差距 2 个牌:
- **07 京ACG0878**: 全局仅 3 帧(白车 1:22 看清 3 帧 conf 0.973) —— 与 03 京ABV200(3 帧非 GT)在"低帧 stationary 车组牌"上**不可区分**(降全局门槛 3 → 03 误罚), 信号天花板;
- **07 京EJQ505**: 从未被读到(帧验: 远景/侧角/人眼难辨), 识别层极限;
- **05 京ADH9206 / 02 京LNE560**: P3 零收获(§3 斜角), 识别层极限;
- 这些牌需要 **GT 更丰富样本 / seg 检测模型**, 独立立项, 不在本 scope 硬凑。

## 5. 记账(cc a9c48a8 下轮两项)

- **记账① 已清**: consensus.box 死代码(span 实际用 track_samples box) —— 783bd13 移除 update 的 box 参数/dag 传参;
- **记账② 11 盲区**: 违章11 GT 无牌标注("车牌看不清"), P1 回填 京AFW1222+京AC81321 两牌**不可核验**(可能是违章车真牌, 人眼不可见但 OCR 读到)——生产罚单需人工复核标注, 记入交付物。

## 6. b2 过合并(17 tid 脏袋)独立立项

P2/P1 用"stationary 车组 + 质心聚集 + 代表 track 排除"绕开 b2 脏袋, **不修根因**。b2 过合并(事件 member_tracks 含 15-22 个过路车 track, 如 02 的 17 tid)记**独立立项交 Jacob**(tracker 层 episode 合并逻辑, 超车牌 scope)。

## 7. 交付物与验收

- 代码: `cli.py _episode_plate_all/_pick_plates/_p3_roi_retry`(worktree plate-fix 分支)
- 验收: `scripts/qw_plate_events_report.py`(逐事件 回填牌 vs violating_plates 全表, 含负例/盲区, is_violation=1 口径)
- 数据: `data/output/qw/plate_p2v7final.txt / plate_p1v4.txt / plate_events_report.csv`(gitignored 供 cc bit-for-bit)
- 回归: unit+integration 全过(含 P2 三测试); 加性: consensus._recompute 只读 text/conf/ts 逐 bit 未变(cc 亲验)

## 方法学

- 主指标: 事件车牌命中率(回填牌 ED≤1 匹配 violating_plates), 多牌逐牌计
- 误罚口径: 回填牌 ∉ violating_plates(含 other_plates: 05 京N541E6); 负例事件/盲区单独标注不并表
- 确定性: 单次 cli.run(preset=balanced), 禁 annotated_video/evidence_images 加速

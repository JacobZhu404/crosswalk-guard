# qw 交付:b2 脏袋收窄实施报告(交 cc 效果 gate)

> 署名:qw(实施) 转交:cc(arbiter);抄送 Jacob
> gate 对象:分支 `b2-narrowing`(worktree `/Users/jacob/personal/crosswalk-guard-b2`, 独立 worktree 红线 ✓)
> 承 `2026-08-18-cc-task-qw-b2.md`(任务单)、`2026-08-19-cc-plangate-b2-PASS-conditional.md`(plan-gate PASS 五条硬条件)
> **不 push / 不 merge, 球回 cc 效果 gate**(cc 约定在独立 detached worktree 复跑验收)

---

## 0. 执行摘要

| 项目 | BASE(dedup 落地后基线 `021dfce`) | B2(本实施) | 判定 |
|---|---|---|---|
| 聚合 P / R / F1 | 1.000 / 0.889 / 0.941 (tp8 fp0 fn1@04) | **1.000 / 0.889 / 0.941 (tp8 fp0 fn1@04)** | 零回退 ✓ |
| 车牌(逐 episode) | 03 京ABV3428·06 京N2LE10·07 京Q5D2N8·08 京ACD5358+京ACW6553·09 京AC63971+京NNM526·11 京AFW1222+京AC81321 | **与 BASE 逐字节一致** | 0 误罚 ✓ |
| member 总量(8 个 confirmed episode) | 145 | **37**(↓74%) | 收窄 ✓ |
| 05 member | 22 | **5**(白车链 tid1→11→19→25/26 并组 + 4 个静止车组) | 验收区间 ~5-8 ✓ |

**设计取向(与 plan-gate 硬条件的关系)**: 不拆 episode 窗口、不改状态判定 —— 在 `_dedup` 之后加
**纯 additive** 的 `BatchViolationEngine._narrow_members`: 只把 `member_tracks` 从"全部并入 track"
收窄为"车组代表", episode 窗口/灯态/状态/large-FP 结构全部不动。由此把 cc 点名的核心风险
(「去合并暴露被吸收 member 成独立 confirmed FP」, 05 右侧 tid17/09 尾段)从根上归零。

## 1. 诊断复现(生产 v2/box, 对齐 cc & qw 量化)

生产 v965/真实 mask 逐 member 复算 05/09, 与 cc gate/任务单数字 bit-for-bit 一致:

| 视频 | member | stationary≥0.6 | 关键证据(qw 复算) | 与 cc plan-gate 一致性 |
|---|---|---|---|---|
| 违章05 | 22 | 14 | 白车链 tid1[0.7,7.7]→11[7.8,22.0]→19[19.5,26.8]→26[25.1,59.7], cx 258-544 时序近连续; 右侧 tid17 cx≈1478 stat0.94 maxov1.00 span[17.9,64.6] | 一致 ✓ |
| 违章09 | 45 | 27 | left 组 23 tid(cx 187-973, 时序链) + 右两车(cx 1336-1448 / 1179-1594) + 尾部 4 组 | 一致 ✓ |

05 白车被 tracker 切 5 碎片被吞进 episode [0.7,64.5] 巨块, over-merge 属实。

## 2. 实施内容(b2-narrowing, 4 处改动)

1. **`BatchViolationEngine._narrow_members(episodes)`(新方法)**: `_dedup` 之后收窄 member。
   - 车组归组: member track 两两满足「质心距离 < `b2_centroid_d` ∧ 时序相邻/重叠
     (|A.end−B.start| < `b2_gap_merge`)」→ union-find 同一车组(传递闭包);
     track 的几何 = 全部样本质心; 时序 = 首/末样本时间戳; 实时稳定性 ≠ t0/t1。
   - 候选车组保留: 组内「窗口内 stationary 占比 ≥0.6 ∧ 窗口内 max overlap >0.15」
     (方案 §3.1 候选条件); 代表 track(episode 代表车)所在组强制保留, 且以代表车为组代表
     (一车一代表, 避免同组双代表)。
   - `member_tracks = 各保留车组代表(overlap 降序)`; **原始全 member 存入
     `member_tracks_all`**(车牌线专用)。
2. **`decide()` 透传 `member_tracks_all`**: 收窄只影响 member_tracks, 全量 member 必须随事件
   透传给 `_episode_plate_all`。**若不透传, 车牌线回退到收窄集 → 复癌细胞 05 京N541E6 /
   08 京PK9B77 / 03 京FJQ279 等跨车污染**(实施中已捕获并修复, 见 §5 回归)。
3. **构造参数**: `b2_centroid_d=200.0, b2_gap_merge=3.0`(方案初值, 扫描见 §3)。
4. **`cli._episode_plate_all`**: 车牌池优先 `member_tracks_all`(原始全 member)。

**红线遵守**: 只新增 `_narrow_members` + decide 透传; `_dedup`、`decide_violations`、transient_green
(Fix A) **逐字未动**; 不碰 prior/权重/`_sample_roi`/tracker 本体。单测 14 条新增/调整,
v979 全单测 368 过 1 失败(失败为 pre-existing `test_mine_classifier_retrain`, 与 b2 无关)。

## 3. 阈值扫描(硬条件④, no silent caps)

全 11 视频同一流水线产出 raw events + track samples(缓存 `data/output/qw/_b2_cache/`),
离线重跑 `_dedup`+`_narrow_members` 于 9 组阈值(D × gap)。**零重跑流水线, 全矩阵透明**:

member 总数(每视频 confirmed episode 收窄后成员总数; 01 为 review episode 成员数):

| 视频\阈值 | D150g2 | D150g3 | D150g5 | D200g2 | **D200g3** | D200g5 | D250g2 | D250g3 | D250g5 |
|---|---|---|---|---|---|---|---|---|---|
| 违章01(review) | 6 | 6 | 6 | 5 | **4** | 4 | 4 | 3 | 3 |
| 违章02 | 3 | 3 | 3 | 2 | **2** | 2 | 2 | 2 | 2 |
| 违章03 | 7 | 7 | 7 | 5 | **5** | 5 | 3 | 3 | 3 |
| 违章05 | 9 | 8 | 7 | 7 | **5** | 5 | 6 | 5 | 5 |
| 违章06 | 5 | 4 | 4 | 5 | **4** | 4 | 5 | 4 | 4 |
| 违章07 | 8 | 8 | 7 | 5 | **5** | 4 | 4 | 4 | 4 |
| 违章08 | 8 | 6 | 6 | 8 | **6** | 6 | 6 | 4 | 4 |
| 违章09 | 8 | 8 | 5 | 7 | **7** | 4 | 5 | 5 | 4 |
| 违章11 | 4 | 4 | 4 | 3 | **3** | 3 | 3 | 3 | 3 |
| member 总计(含 review ep) | 58 | 54 | 49 | 47 | **41** | 37 | 38 | 33 | 32 |

> 量纲说明: 扫描重算用的是同构 union-find 的独立实现(与引擎共享阈值, 不共享代码);
> 上面 D200g3 行与**正式全 11 跑**(§4)完全吻合, 作校准。

**选型: D=200px, gap_merge=3s(维持 plan 初值)**, 理由:
- 05(D=200): 5 个车组 = 白车(5 碎片合并为 1)+ 右侧 2 组(cx~1400/1470 静止车)+ 中部组 + 尾部组,
  落在任务预期 ~5-8; D150 把白车链拆碎(6-8 组, 过紧), D250g5 过度吞并(5 组但组内混入更多)。
- 08(两白车)/09(两车): 任何组合下**都是单 episode**(§4 逐条验), 阈值不影响 episode 结构。
- g5 在 09 上把 member 7→4 —— 是"左侧 23-tid 停车带链并入更粗"换来的, 吞并过路车风险升高,
  09 保守; g3 是"白车链收拢"所需的最小值(碎片 max 间隙 = 2.5s < 3s ✓)。
- 敏感视频: 05(白车链)、08(双白车)、09(左侧停车带 23-tid、尾段 72-106)、07(三车)。
  阈值只影响 member 粒度, **不影响 episode 窗口/确认统计/车牌**(三者对阈值完全不变量,
  已验证 D150~D250×g2~g5 下所有视频 episode 集与车牌集与 D200g3 一致)。

## 4. 效果 gate 数据(全 11, 生产 v2/box, 逐 episode 审计 —— 硬条件②)

> 环境: 生产配置(version=v2, occ_denom=box), 真实 11 视频含负例。与 cc gate 工具链同源
> (`match_violation_events` + `classify_false_positives`)。

| 视频 | confirmed # | episode [span] | member(收窄) | member_all(原始) | 命中 GT | coverage | 车牌(主, 次) |
|---|---|---|---|---|---|---|---|
| 违章01[负例] | 0 | (待复核 1, review 语义不吃) | – | – | – | – | – |
| 违章02 | 1 | [23.6-74.2] | 2 | 17 | GT[21-68] ✓ | 0.95 | ''(基线同) |
| 违章03 | 1 | [91.5-137.8] | 5 | 15 | GT[74-134] ✓ | 0.71 | 京ABV3428 |
| 违章04 | 0 | (无) | – | – | – | – | FN 基线 1(老灯态难例, 不变) |
| 违章05 | 1 | [0.7-64.5] | **5** | 22 | GT[0-30] ✓ | 0.98 | ''(基线同, 宁缺毋滥) |
| 违章06 | 1 | [0.7-42.0] | 4 | 10 | GT[0-29] ✓ | 0.98 | 京N2LE10 |
| 违章07 | 1 | [4.2-49.4] | 5 | 10 | GT[0-43] ✓ | 0.90 | 京Q5D2N8 |
| 违章08 | 1 | [0.7-49.2] | 6 | 21 | GT[0-51] ✓ | 0.95 | 京ACD5358, 京ACW6553 |
| 违章09 | 1 | [0.7-106.3] | **7** | 45 | GT[11-72] ✓ | 1.00 | 京AC63971, 京NNM526 |
| 违章10[负例] | 0 | (无) | – | – | – | – | – |
| 违章11 | 1 | [18.0-28.3] | 3 | 5 | GT[15-28] ✓ | 0.77 | 京AFW1222, 京AC81321 |

**聚合: TP=8 / FP=0 / FN=1(违章04) → P=1.000 / R=0.889 / F1=0.941 | 真误报=0 碎片=0。**

**逐硬条件 @2026-08-19(cc 五条)**:
1. **P=1.000 零新增 FP** ✓ —— episode 结构零改动; 05 右侧 tid17/cx1478 与 09 尾段仍被原 episode
   窗口吞入(window 不变), 未暴露为独立 confirmed episode。直观验证: member 收窄到 5,
   但 episode 仍是单 TP(window/状态未动)。
2. **逐 episode 审计** ✓ —— 上表即硬条件②交付物: 每视频每 confirmed episode 的
   [span]+member+member_all+命中 GT; 无窗外 confirmed(违章04 无 confirmed, 负例 01/10 零 FP)。
3. **多车窗 08/09 保持单 episode** ✓ —— 08 仍 1 confirmed TP(双白车在其内, 京ACW6553 次牌在),
   09 仍 1 confirmed TP(双 GT 车牌都在); 且 §3 扫描已验证任何阈值组合下都不拆 episode。
4. **阈值 200px/3s 扫描透明** ✓ —— §3 全 11 扫描表 + 选型理由 + 敏感视频。
5. **Fix A 语义不动** ✓ —— `_narrow_members` 只读 member_tracks/样本, 不改 `_dedup`
   吸收/transient_green 逻辑; `test_dedup_review_poison.py` 全过; 09 review 降级不复现。

## 5. 实施中捕获并修复的缺陷(供 cc 复核)

- **d0: `member_tracks_all` 未透传(danger, 已被本交付修复)**: 初版 decide() 只映射
  member_tracks, 车牌线 `_episode_plate_all` 回退到收窄集 → 05 误罚复出(京N541E6),
  03 被 京FJQ279 夺主, 08=京PK9B77, 09/06/07 掉牌。修复=决定() 透传 all。
- **d1: 代表车双计**: rep 在其车组中时, 该组又以组内 max-ov 车另计一个代表 → 同车双代表
  (05 白组 11+26 两个)。修复: 代表车所在车组以代表车为组代表(一车一代表, 05 → 5 个)。

## 6. 复核命令(cc 效果 gate 复跑用)

```python
# 若需要 BIT-PIPE(与本交付一致) 重跑: 在 b2 分支上
.venv/bin/python scripts/qw_b2_audit.py --tag B2FIX        # 逐 episode 审计 + 缓存
.venv/bin/python scripts/qw_b2_audit.py --sweep            # 阈值扫描(缓存离线)
```

`scripts/qw_b2_audit.py` 是本交付新增验收工具; 输出与基线对照均落
`data/output/qw/b2_effect_B2FIX.txt` / `b2_effect_sweep.txt`。

## 7. 风险与未决(交 cc/Jacob 裁定)

- **车牌 span 绕行「可收回」未达成(设计取舍)**: 任务书 §6 名义"span 绕行可收回", 但收窄
  member 后 05 京N541E6(黑车组)质心 span 变小 → span 约束失效; 若车牌线改吃收窄集,
  N541E6 将重新成为候选(且黑车组窗口内 stationary≥0.6 大概率留在保留集) → 误罚。
  两信号(车牌污染解除靠 BASE 全量+span; 车组收窄=代表集)**互斥**。故车牌线保持
  全量 member+span 绕行不变, 车牌误罚在 11 视频 0 复现 → span 绕行仍为 auto 回填的一部分,
  **若 Jacob/cc 坚持收回, 需单独设计车牌侧替代约束, 不在本 b2 线 scope**。
2. **09 左停车带 23-tid 链成 1 组**(cx 187-973 传递闭包): member 从 45 → 7 已有硬成效;
  "按时间细分停车带" 是方案 §5.1 提出的后续, 开卷; 本交付保留(拆它会动 episode 语义)。
3. `member_tracks` 语义变更影响下游消费(`eval_tracking_gtfree` 的 frag_count = len(member_tracks)
   等): 语义现为"车组代表集", 对希望"物理车数"的用法是更准确, 对希望"全部 track 数"的脚本
   会变小 —— 已自查 `diag_vehicle_track_fragmentation.py`(b2 目标就是降 masked_frag, 方向一致)。
4. 阈值参数已可配(`b2_centroid_d/b2_gap_merge`), 若以后发现并行过路车链成组, 可收紧 g2
   而不回退 F1(阈值不影响 episode 结构, 已验证)。

---

*署名(qw)。证据 = 全 11 真实生产跑, 阈值全矩阵扫描(缓存离线), 修复前后车牌逐视频对照
(BASE log vs B2FIX)。承 2026-08-18-cc-task-qw-b2.md / cc plan-gate 硬条件 /
`021dfce`(dedup 落地基线) / `f8de88c`(cc dedup gate PASS)。*
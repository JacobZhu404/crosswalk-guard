# b2 脏袋 episode 合并收窄方案(qw, 交 cc plan-gate)

> 承任务单 `docs/handoff/2026-08-18-cc-task-qw-b2.md`(cc)。wb dedup Fix A 已落地(021dfce, F1=0.941), "错开归因"前提满足。
> 本方案 = 诊断 + 修法设计, 过 cc plan-gate 后在独立 worktree 实施; 不 push/merge, 球回 cc 验收。

## 1. Pathology 量化(qw 实测, 生产 v2/box)

| 视频 | member | stationary车(≥0.6) | 空间聚类(200px) | 关键组 |
|---|---|---|---|---|
| 违章05 | 22 | 14 | **5 组** | 组0=[19,26,11,1,25] 质心x=394(**白车被切 5 碎片**, 时序连续 0.7→59.7s) |
| 违章09 | 45 | 27 | **4 组** | 组0=17 tids 质心x=341(左侧车堆, 需更细判) |

**05 白车碎片实证**(组0 时序连续): tid1[0.7,7.7] → tid11[7.8,22.0] → tid19[19.5,26.8] → tid25/26[24.8,59.7] —— 同一白车被 tracker 重编号切成 5 段, 位置相邻(质心 310-544)。

**根因**: `violation_engine._dedup`(:257) 全局按"时间重叠或间隔<gap(5s)"跨 track 合并, **无空间/时序连续判定** → 时序并行的过路车/不同车(只要时间邻近)全被吞进一个 episode 巨块。05 的 22 member 里 14 个 stationary 全压线(ov>0.15), "静止+压线"分不开, **唯可区分信号 = 时空连续**。

## 2. 同车碎片 vs 过路车(可区分信号)

- **同车碎片(重编号)**: 时序连续(tid A 结束 ≈ tid B 开始, 间隙小) + 空间相邻(质心近/box IoU 高)。05 组0 五段时序首尾相接。
- **过路/并行车**: 时序并行(与违章车同时刻共存) —— 即使 stationary+压线, 也不该并进违章车 episode。

## 3. 修法设计(episode 合并三重约束)

`_dedup` 合并前加**车组归组**:
1. **违章车候选**: member 中窗口内 stationary≥0.6 ∧ overlap>0.15(保留现有语义);
2. **同车碎片合并(时空连续)**: 两 track 满足 (a) 时序连续(|A.end - B.start| < gap_merge, 如 3s) ∧ (b) 空间相邻(质心欧氏距离 < D, 如 200px, 或 box IoU > 0.2) → 归同一**车组**(物理同车);
3. **episode = 车组并集**: 同一违章窗内的多个车组(多车违章, 如 08 两白车/09 两车)仍归一个 episode(GT per-窗语义), 但**时序并行的过路车组不并入**(其 track 与违章车组无时空连续);
4. 代表 track / member_tracks 按车组重算(代表 = 车组内 max_overlap 车)。

**05 预期**: 22 member → 白车组(5 碎片合并) + 黑车组([2,30]) + 其他车组 → episode member 大幅收窄; 京N541E6(黑车组)与白车组分离 → 车牌 span 绕行可收回。
**09 预期**: 组0 的 17 tids 用"时序连续"细分(并行不同车拆开), member 数下降。

## 4. 验收基准(cc 效果 gate)

- member 数: 05 从 22 → 少数车组(物理同车数, 预期 ~5-8);
- 车牌 span 绕行可收回: 05 白车组不再混入京N541E6(黑车) → 去掉质心 span 约束后 05 仍不误罚;
- 碎片化指标下降;
- **硬约束**: 全 11 视频端到端 **F1 ≥ 0.941**(wb dedup 落地后基线, 不回退) ∧ 车牌误罚仍 0 ∧ 01/10 负例 0 FP。

## 5. 风险与开放问题

- **09 组0 的 17 tids**(左侧同区 stationary 车堆): 纯空间聚类分不开, 靠"时序连续"细分 —— 若其中含真实并行多违章车(09 GT 两车), 需保证不被拆散到不同 episode(多车仍同 episode)。阈值(D, gap_merge)需敏感性验证;
- **阈值鲁棒性**: D=200px / gap_merge=3s 为初值, 需在 11 视频上扫描验证(避免过紧拆碎同车 / 过松吞过路车);
- 与 wb dedup Fix A 的交互: dedup 已修"review 毒化", 本方案改"合并范围", 两者都在 `_dedup` —— 实施时保持 Fix A 语义不动, 只收窄合并判定。

## 6. 红线

- 独立 worktree(qw 署名); 先过本 plan-gate 再实施; 不 push/merge, 球回 cc;
- 不碰 prior/权重/`_sample_roi`/tracker 本体(只改 episode 合并判定层)。

## 方法学

- 诊断: `cli.run(return_track_samples=True)` → member 的 stationary/overlap/质心/时序 → 空间聚类(200px)量化。产物 `data/output/qw/_b2diag_*` `_b2cl2_*`(gitignored)。

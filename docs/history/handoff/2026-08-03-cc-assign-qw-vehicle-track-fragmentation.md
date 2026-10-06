# CC 派活 qw:违章车 track 碎片化 GT-free 量化诊断(只读,非红绿灯线)

> 出自 cc(arbiter)派给 qw。红绿灯线阶段性收口(05 双路径硬地板已定, 光 thread 唯一未验杠杆=缩小 ROI 已推迟到 wb A 消融跑完再议)。转攻 [[b2-tracking-fragmentation-blindspot]] 记的盲点:**违章车被 tracker 切成多个 ID, 但被 `decide()`/`_dedup()` 合并掩盖, F1 看不见**。本任务把这个数字**只读、GT-free 地**量出来。**只读:不改 tracker/pipeline/decision/engine, 不训练, 不改生产代码, 不碰 wb 的 s1/`_A` 缓存, 不动 wb 训练。**

## 0. 为什么走 GT-free(cc 已侦察代码, qw 据此设计)
- 碎片化度量机制**其实已存在**: `attribution_union`(`src/redlight/evaluation/module_metrics.py:80-106`, 对每个 anchor box 取所有 IoU≥T 的 track_id 并集, "理想=1, 碎片化时>1"), 由 `scripts/eval_tracking.py` 驱动。
- **但它跑不起来**: 依赖 `datasets/gt/tracking/违章*.json` 的逐帧 `box`, 而**全部是 `null`**(违章02/04/05 均已核, `SCHEMA.md:32` 注明"骨架在, box:null 未标, 需 Jacob 标框, 本 Phase 不做")→ `eval_tracking.py:42` 对每个 anchor 因"锚框未标注"跳过。**别去标那些框(那是 Jacob 的活), 也别改 `eval_tracking.py`。**
- ⇒ 走 **GT-free 替代**: 不用 GT 框, 而是**锚定 confirmed 违章 episode 的代表 track 的逐帧 box 轨迹**, 统计有多少其它 track ID 与它时空重叠 = 物理同车被切的碎片数。这样**现在**就能量, 不等标框。

## 1. 口径(全部来自 cc 侦察, file:line 钉死)
- **入口**: `cli.run(cfg, "input_video/<video>.mp4", out_dir, preset=..., return_track_samples=True)`(`src/redlight/app/cli.py:30`)→ 返回 `(events, track_samples)`(`cli.py:175-178`)。
  - `track_samples`: `tid -> [{ts, stationary, box, overlap, cls, conf}, ...]`(逐帧 tracked box 流, 由 `BatchViolationEngine.accumulate` 攒, `violation_engine.py:157-188`)。
  - `events`: confirmed 违章 episode, 每个带 `track_id`(代表=max_overlap)+ `member_tracks`(被 `_dedup` 跨 track 合并的所有 ID, `violation_engine.py:254-298 / _absorb:287-295`)。
- **tracker**: `SimpleTracker`(贪心 IoU 最近匹配, `src/redlight/models/vehicle.py:53-102`), 关键常数 `iou_thresh=0.3`、`max_disappeared=15`(连续 15 帧未匹配就丢弃该 ID → 重现即新 ID = 碎片)。**这两个常数是成因分类的判据。**
- **违章视频集**: `datasets/gt/videos.csv` 里 `has_violation==1` 的(`violation_eval.py:57-73` `load_video_metadata`)。**只跑这些**(负例无违章车锚)。
- **参考(勿改)**: `scripts/eval_tracking.py:36-52 eval_video` 是"用 `cli.run(return_track_samples=True)` 驱动 + 喂 `attribution_union`"的现成范例, 照它的调用方式, 但 anchor 换成代表 track(GT-free)。
- 逐帧读原始帧(如需可视化/校验): `governing_disc._read_frames_at(video, fis)`(`governing_disc.py:104-118`, 开 `input_video/<video>.mp4`)。

## 2. 逐 confirmed episode 计算(GT-free)
对每个 has_violation 视频跑一次 `cli.run`, 取每个 confirmed 违章 episode:
1. **代表 track 锚轨迹**: 取 episode `track_id` 的 `track_samples[track_id]` 逐帧 box, 限定在 episode `[start_s, end_s]` 窗内, 作为"违章车在哪"的锚。
   - **报锚覆盖率 coverage**: 锚 track 的样本 ts 覆盖了 episode 窗多少比例(锚 track 本身若也碎, 只覆盖一部分 → 后面真碎片数会保守 undercount, 必须报出来别掩盖)。
2. **Masked 碎片数** = `len(member_tracks)`(episode 自带): `_dedup` 吞进这一 episode 的 ID 数 = 当前 F1 口径下被合并掉的可见碎片。
3. **GT-free 真碎片数**(核心): 对窗内每个采样 ts, 取锚 box, 统计所有 track ID(含代表自身)中在该 ts(或最近 ts, 容差 ≤ 1 采样间隔)的 box 与锚 box IoU≥T 的并集大小。**报 T=0.3(与 `iou_thresh` 同)和 T=0.5(与 `attribution_union` 默认同)两档。** 这包含**从未 qualify 成 violation-event、因而连 `member_tracks` 都没吞到的碎片** —— 这是比 masked 更全的物理碎片数。
4. **掩盖缺口** = GT-free 真碎片数 − Masked 碎片数: `_dedup` 之外还漏了多少碎片(连合并都没覆盖到)。

## 3. 碎片化成因分类(对相邻碎片 ID 排序后逐对判)
把并集里的碎片 ID 按首现 ts 排序, 对相邻两碎片(前者末样本 vs 后者首样本)分类:
- **(a) 时间断裂**: 间隔帧数 > `max_disappeared=15`(换算成秒 = 15 / 采样fps)→ 目标消失后重编号(遮挡/漏检/驶离重入)。
- **(b) 空间跳变**: 时间相接(≤ 阈值)但前末 box 与后首 box IoU < `iou_thresh=0.3` → 框抖/形变导致贪心匹配断裂。
- **(c) 并存重叠**: 两 ID 时间重叠且高 IoU(≥0.5)→ 重复检测(同物理车同时两 ID)。
报每类占比(全 episode 汇总)。→ 直接指向 tracker 立项该动哪块:(a)多→调 `max_disappeared`/加 re-ID;(b)多→放宽 `iou_thresh`/加运动预测;(c)多→加 NMS/去重。

## 4. 报告结论段回答三问
1. **违章车到底被切几个 ID**(GT-free 真碎片数 per video + 汇总分布, T=0.3/0.5 两档)? 验证 [[b2-tracking-fragmentation-blindspot]] 记的"5–45 个 ID"是否还成立(可能已随 tracker/口径变化)。
2. **`_dedup` 掩盖 & 漏了多少**(masked=len(member_tracks) vs 真碎片数, 及掩盖缺口)? 量化"F1 为何看不见"。
3. **碎片化主因**(时间断裂/空间跳变/并存重叠占比)? 定 tracker 立项优先修哪块。

## 5. 输出与署名
- 脚本 `scripts/diag_vehicle_track_fragmentation.py`
- 报告 `docs/reports/2026-08-03-qw-vehicle-track-fragmentation.md`
- 逐 episode CSV `data/output/qw/vehicle_track_fragmentation_per_episode.csv`(列建议: video, episode_idx, start_s, end_s, rep_track_id, anchor_coverage, masked_frag(len member_tracks), truefrag_iou03, truefrag_iou05, gap_cause_n, jump_cause_n, dup_cause_n)
- 建议再出逐碎片 CSV(video, episode_idx, frag_track_id, first_ts, last_ts, n_samples, cause_vs_prev)供 cc bit-for-bit 复核。
- 署名 `Co-Authored-By: 千问办公 <qw@crosswalk-guard.agents>`, main 上提交。

## 6. 陷阱与红线(必读)
- **锚偏差**: 代表 track 若本身碎(coverage 低), 锚轨迹不全 → 真碎片数 undercount。**必须报 coverage, coverage < ~0.7 的 episode 单独标注、数字视为下界。**
- **IoU 阈值敏感**: 必报 T=0.3 与 0.5 两档, 别只报一个。
- **确定性**: `cli.run` 走 YOLO 推理, 跑一次即可, 报告注明是单次结果, **别声称多 seed / 别把它当统计量**。
- **别改任何生产代码/GT**: 不标 tracking GT 的 null box, 不改 tracker 常数, 不改 `_dedup`/`eval_tracking.py`。纯只读观测。
- **别碰** wb 的 s1/`_A` 缓存与训练, 别 `git add -A`(scoped 提交自己的 3 个文件)。
- 若 `cli.run` 接口/preset 与本 spec 有出入(cc 侦察基于静态读码), **以实际代码为准, 在报告里注明偏差**, 别硬凑。

## 7. 交付后
交 cc。cc 两层复核(从逐 episode/逐碎片 CSV 重算聚合 + 独立重跑 ≥1 个视频的 `cli.run` 重建代表 track 锚轨迹 + 手算若干 episode 的真碎片数/成因分类到可核粒度)。据三问定 tracker 是否立项、优先修哪块。

---
**一句话**: 违章车被 tracker 切成多 ID 但被 `_dedup` 合并进 `member_tracks` 掩盖, F1 看不见;轨迹 GT 框全 null 跑不了现成 `attribution_union`, 故走 GT-free——锚定 confirmed episode 代表 track 的逐帧 box 轨迹, 量 (1)真碎片数(所有 IoU≥0.3/0.5 时空重叠的 track 并集, 含未 qualify 的碎片)(2)masked 碎片数(len member_tracks)及掩盖缺口 (3)成因(时间断裂>15帧/空间跳变 IoU<0.3/并存重叠)占比。答:违章车切几个 ID、_dedup 掩盖多少、主因是啥→定 tracker 立项修哪块。只读, 报锚 coverage 防 undercount, IoU 双档, 单次非多 seed。

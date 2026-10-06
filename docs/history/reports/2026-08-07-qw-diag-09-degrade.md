# 诊断: 09 事件降级机制(#3 时序门控的合并语义副作用, qw 只读诊断)

> 起因: 车牌线收官后复核发现 违章09 在 #3 merge 后从 confirmed 降为 review(灯态=green)。
> qw 实测定位: **不是时序门控阈值误伤, 是 `_dedup` 合并语义把持久真绿拖下水**。
> 状态: 只读诊断, 未改生产码; 供 cc/Jacob 决策"不回滚 #3"与 b2 排期。

## 1. 数据(#3 merge 后 main 实测)

| 视频 | raw 绿最长 run | 绿段(max_raw_green_run_s) | 事件 |
|---|---|---|---|
| 违章01 | 3.10s | [48.4,61.3] green **3.098**(瞬态) | review ✓(01 假绿正确降级) |
| 违章09 | 55.89s | [0.0,4.2] green **0.943**(瞬态) + [7.4,106.3] green **55.892**(持久) | **review**(应为 confirmed) |
| 违章11 | 12.40s | [18.0,28.3] green **10.266**(持久) | confirmed ✓ |

## 2. 机制(代码坐实)

- #3 时序门控(`decision.py:56` min_persistent_green_run_s=6.0): 绿段按"段内最长 raw 绿 run"分桶 —— 持久(≥6) → confirmed 桶, 瞬态(<6) → review 桶(降级非硬杀, 防屠真绿)。
- **09 有两个绿段**: 前导瞬态 [0,4.2](run 0.94s, 推断/反光噪声) + 持久 [7.4,106.3](run 55.89s, 真绿)。
- `violation_engine._dedup`(:257): 时间重叠或间隔<gap(5s) 跨 track 合并为 episode; **合并语义(:269): "任一成员为 review → episode 记 review(安全侧优先)"**。
- 09 两事件 gap = 7.4 - 4.2 = 3.2s < 5s → 合并 → 任一 review → **整个 episode review** → 持久真绿被瞬态段拖累降级。

## 3. 结论

- **门控阈值 T=6.0 本身正确**(01 假绿 3.10 < 6 ✓ 降级; 11 真绿 10.27 ≥ 6 ✓ confirmed; 09 持久 55.89 ≥ 6 ✓ 该 confirmed)。
- **缺陷在 `_dedup` 合并语义**: "任一 review → 全 episode review" 未区分"瞬态绿段"与"持久绿段", 把同 episode 的持久真绿误降。
- **修复候选(供决策, 未实施)**:
  - A: `_dedup` 合并语义改"任一成员 confirmed → episode confirmed; 全 review 才 review"。09 有 confirmed 成员(持久绿) → confirmed ✓; 01 仅瞬态段(无 confirmed 成员) → 仍 review ✓ 安全; 假持久绿(run≥6 误绿)概率低且 01 类无。
  - B: 瞬态绿段不与持久绿段合并(瞬态段独立成 episode 或并入时保持 confirmed 优先)。
  - C: 接受 09 降级(进人工队列)——最坏兜底, 但损失真绿 confirmed。
- **影响**: 若修 A/B, 09 恢复 confirmed → 车牌线命中 7/12 不受 #3 影响; "不回滚 #3"与"09 保真"可兼得。

## 4. 建议

- cc 复核本诊断(亲跑 09 obs/run/合并), 若成立 → #3 补 `_dedup` 语义修复(小 diff), 01/09/11 三视频回归(01 仍 review / 09 confirmed / 11 confirmed), 再终判"不回滚 #3"。
- b2 立项不受此阻塞(独立 tracker 层), 但 #3 修复若落地, 与 b2 都在事件成形层, 建议先后归因。

## 方法学

- 实测: `cli.run` + patch `BatchViolationEngine.decide` dump `_light_obs` → `temporal_fusion._raw_green_runs` / `fuse_light`(max_raw_green_run_s) → `_dedup` 合并语义代码坐实。
- 产物: `data/output/qw/_obs_*.json`(gitignored, 供 cc bit-for-bit)。

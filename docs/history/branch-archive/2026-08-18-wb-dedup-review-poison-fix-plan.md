# wb 修法方案 gate: _dedup review 毒化(09 回归真因, cc brief 4329e78)

> 署名:wb(执行)  待 cc gate 放行后实施; 不碰 prior/权重/`_sample_roi`; 独立 worktree(基于 main 4329e78); 不 push/merge。
> 订正:本方案替代「09 prior 重定位」线(wb-09prior-relocate, 已弃)。09 真因是 dedup 交互 bug, 非灯态/prior。

## 1. 真因(摘自 cc brief 4329e78)
09 有 `[7.41,106.26]` 55.9s 干净 visible 持续绿(完全过 #3),GT[11-72] 压在其中。
dedup 前:45 个 confirmed 子事件(压在 55.9s 绿,多 track b2 碎片)+ 1 个 review 事件(tid5 在 `[0.67,4.18]` 0.94s 瞬态绿,由 #3 造)。
dedup 后(gap=5s, `[0,4.18]` 与 `[7.41,106.26]` 间隔 3.23s<5s 合并):1 个 episode `[0.67,106.26]`, status=review。

`violation_engine.py:299-301` 老安全规则:「任一成员 review → 整个 episode review」。#3 引入的**新类 review(瞬态绿)**能与同窗强 confirmed 共存,旧规则未区分来源 → 1 个瞬态 review 把 44+ 个铁证 confirmed 全拖成 review。

## 2. 选方案:方案A(语义, cc 推荐)
给 review 事件打**来源标签**,仅 D1(`occluded/unknown`,真不确定)保留 review 优先安全语义;`transient_green`(#3)不得压过共存 confirmed。
(方案B 物理:不跨红灯段合并绿事件 — 不采,因 dedup 不持中间灯态、实现更脆,且 A 已覆盖。)

## 3. 精确代码改动(两处, 均在 main 4329e78 已含 #3/T=6)
### 3.1 decision.py — review 事件加 `review_reason`
`decide_violations` 事件生成(约 L75-84)当前 `(go,"confirmed"),(review_light,"review"),(transient_green,"review")` 三者都不带来源。改为:
```python
for light_ivs, status, reason in (
    (go,            "confirmed", None),
    (review_light, "review",    "light_uncertain"),  # D1 真不确定
    (transient_green, "review",  "transient_green"),  # #3 瞬态, 不得毒化 confirmed
):
    for s, e in interval_intersect(light_ivs, base):
        if e - s >= min_duration_s:
            events.append({
                "track_id": tr["track_id"], "status": status,
                "start_s": s, "end_s": e,
                "light_state": _state_at(segs, s),
                "max_overlap": _peak_overlap(tr, s, e),
                "review_reason": reason,
            })
```

### 3.2 violation_engine.py `_absorb` — 仅 D1 review 降级
L299-301 改为(瞬态绿 review 不毒化 confirmed 核; 任何非瞬态 review 保持旧安全语义):
```python
# #3 dedup 交互修复(cc brief 4329e78): transient_green review 不得拖垮共存 confirmed。
# 仅非瞬态(=D1 light_uncertain 或旧未标注)review 保留 review 优先; 否则 confirmed 核胜出。
e_poisons = (e["status"] == "review" and e.get("review_reason") != "transient_green")
cur_poisons = (cur["status"] == "review" and cur.get("review_reason") != "transient_green")
if e_poisons or cur_poisons:
    cur["status"] = "review"
elif e["status"] == "confirmed" or cur["status"] == "confirmed":
    cur["status"] = "confirmed"
# else: 两成员皆 transient_green review(无 confirmed 核) -> episode 保持 review(默认)
```
`_new_episode` 不动(首事件 status 透传); 上述 `elif` 确保确认事件把首事件为瞬态 review 的 episode 升级为 confirmed。

## 4. 01 安全性(硬约束, 逐 bit 证 01 仍 0 FP)
01 仅 `[48.36,61.29]` 3.10s 瞬态绿 → #3 造 `transient_green` review, **无共存 confirmed 核、无 D1 review**。
新规则:episode 首事件=瞬态 review(非 D1)→ 无 confirmed 并入 → 保持 review, confirmed=0。
→ 01 误绿仍消除, 不复活。✓

## 5. 09 恢复(逐 bit)
09 episode `[0.67,106.26]` = 1 个 transient_green review + 45 个 confirmed, gap<5s 合并。
新规则:有 confirmed 核、review 非 D1 → episode=confirmed(瞬态 review 被吸收)。
→ 09 恢复 confirmed, GT[11-72] 命中; 京AC63971+京NNM526 两牌回到 confirmed 事件。✓

## 6. 验收计划(cc 效果 gate, 全 11 视频, 不复犯子集漏测)
- 复用 `verify_gates_01fp.py` / `diag_09_violations.py` 思路, 断言 dedup 后 episode:
  - 09 confirmed 段跨含 [11-72](命中真违章); 且 01/10 仍 0 FP(负例)。
  - 02/03/05/06/07/08/11 零回退(真绿 confirmed 不丢)。
  - 04 不受影响(fn1 维持, 不引入新 FP)。
  - 端到端 **F1≥0.941 / P=1.000**(cc 口径: tp8/fp0/fn1(04))。
- 单测:构造 1 confirmed + 1 transient_green review(同 episode)→ 期望 confirmed; 1 纯 transient_green review → 期望 review; 1 confirmed + 1 light_uncertain review → 期望 review(保 D1 安全)。

## 7. 红线合规
- 仅改 dedup/#3 交互(decision.py 事件标签 + violation_engine.py 合并语义); 不碰 prior 坐标 / sat_min / `_sample_roi` / 权重 / ped_signal.pt / select_gtfree。
- 基于 main 4329e78 建独立 worktree(命名 `wb-dedup-review-fix`); scoped add; **不 push / 不 merge**, 球回 cc 全 11 验收。
